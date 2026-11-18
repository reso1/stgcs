from __future__ import annotations

from dataclasses import dataclass
import math
import time
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydrake.all import (
    Binding,
    CommonSolverOption,
    Constraint,
    Cost,
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    HPolyhedron,
    L2NormCost,
    LinearEqualityConstraint,
    LorentzConeConstraint,
    MosekSolver,
    MosekSolverDetails,
    Point as DrakePoint,
    SolverOptions,
    VPolytope,
)
from shapely import constrained_delaunay_triangles
from shapely.geometry import GeometryCollection, Point as ShapelyPoint, Polygon, box
from shapely.ops import unary_union

from baselines.gcs_utils import make_Cartesian_power_hpoly, randomForwardPathSearch
from stgcs_native.types import MPQuery
from stgcs_native.types import STTrajectory


@dataclass(frozen=True)
class CBGCSOptions:
    time_step: Optional[float] = None
    time_horizon: Optional[float] = None
    time_horizon_factor: float = 2.0
    low_level_gap: float = 0.05
    conflict_mode: str = "lb"
    max_rounded_paths: int = 100
    max_high_level_expansions: int = 1000
    conflict_score_samples: int = 100
    conflict_tolerance: float = 1e-9
    constraint_padding: float = 1e-9
    spatial_restriction_margin: float = 2e-8
    cspace_validation_tolerance: float = 1e-6

    def resolved_robot_width(self, robot_radius: float) -> float:
        width = 2.0 * float(robot_radius)
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError(f"CB-GCS robot width must be positive and finite, got {width!r}.")
        return width

    def resolved_time_step(self, robot_width: float, vlimit: float) -> float:
        if self.time_step is not None:
            value = float(self.time_step)
        else:
            value = float(robot_width) / max(float(vlimit), 1e-9)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"CB-GCS time_step must be positive and finite, got {value!r}.")
        return value

    def resolved_time_horizon(
        self,
        instance,
        queries: Sequence[MPQuery],
        vlimit: float,
        time_step: float,
    ) -> float:
        if self.time_horizon is not None:
            value = float(self.time_horizon)
        else:
            direct_times = [
                float(np.linalg.norm(query.goal - query.start) / max(float(query.vlimit), 1e-9))
                for query in queries
            ]
            value = max(float(time_step), self.time_horizon_factor * max(direct_times, default=0.0))
        tmax = float(getattr(instance.stgcs, "tmax", value))
        value = min(value, tmax)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"CB-GCS time_horizon must be positive and finite, got {value!r}.")
        if value + 1e-9 < max(float(query.t_start) for query in queries):
            raise ValueError("CB-GCS time_horizon must cover all query start times.")
        return value


@dataclass(frozen=True)
class CBGCSRectangle:
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def from_center_lengths(
        cls,
        center: np.ndarray,
        lengths: np.ndarray,
        padding: float = 0.0,
    ) -> "CBGCSRectangle":
        center = np.asarray(center, dtype=float).reshape(2)
        lengths = np.asarray(lengths, dtype=float).reshape(2)
        half = 0.5 * lengths + float(padding)
        return cls(lower=center - half, upper=center + half)

    def contains(self, point: np.ndarray, tol: float = 0.0) -> bool:
        point = np.asarray(point, dtype=float).reshape(2)
        return bool(np.all(point >= self.lower - tol) and np.all(point <= self.upper + tol))

    def to_polygon(self) -> Polygon:
        return box(float(self.lower[0]), float(self.lower[1]), float(self.upper[0]), float(self.upper[1]))


@dataclass(frozen=True)
class CBGCSConstraint:
    agent_idx: int
    time_step: int
    region: CBGCSRectangle
    kind: str


@dataclass(frozen=True)
class CBGCSConflict:
    robot_idx_i: int
    robot_idx_j: int
    time_step: int
    position_i: np.ndarray
    position_j: np.ndarray


@dataclass
class CBGCSLowLevelPlan:
    trajectory: STTrajectory
    positions: np.ndarray
    cost: float
    runtime: float
    arrival_time: float


@dataclass
class CBGCSResult:
    solutions: Optional[List[STTrajectory]]
    success: bool
    runtime: float
    cost: float = math.inf
    popped_nodes: int = 0
    generated_children: int = 0
    update_calls: int = 0
    low_level_calls: int = 0
    low_level_runtime: float = 0.0
    conflict_checks: int = 0
    conflict_runtime: float = 0.0
    time_step: float = math.inf
    time_horizon: float = math.inf


@dataclass(frozen=True)
class _LayerSet:
    layer_idx: int
    base_idx: Optional[int]
    name: str
    shape: object
    convex_set: object


@dataclass(frozen=True)
class _SpatialPathSet:
    name: str
    convex_set: object


@dataclass
class _HighLevelNode:
    plans: List[CBGCSLowLevelPlan]
    constraints: Tuple[CBGCSConstraint, ...]
    cost: float
    node_id: int
    collision_score: float
    parent_id: int = -1
    mod: str = "lb"


@dataclass
class _HighLevelSearchState:
    nodes: Dict[int, _HighLevelNode]
    open_nodes: Dict[int, _HighLevelNode]
    focal_nodes: Dict[int, _HighLevelNode]
    next_node_id: int = 1
    lb_cost: float = -1.0
    lb_collision: float = math.inf
    ub_cost: float = math.inf
    ub_collision: float = math.inf
    solution_node: Optional[_HighLevelNode] = None
    best_node: Optional[_HighLevelNode] = None
    popped_nodes: int = 0
    generated_children: int = 0
    update_calls: int = 0


@dataclass(frozen=True)
class _CBGCSGeometry:
    shapes: Tuple[Polygon, ...]
    hpolys: Tuple[HPolyhedron, ...]
    neighbors: Dict[int, set[int]]


class CBGCSGeometryAdapter:
    @classmethod
    def from_env(cls, env) -> _CBGCSGeometry:
        if getattr(env, "dim", None) != 2:
            raise ValueError("CB-GCS follows the original 2D formulation and requires a 2D environment.")
        cspace = tuple(getattr(env, "C_Space", ()) or ())
        if cspace:
            return cls._from_cspace(env, cspace)
        return cls._from_static_obstacles(env)

    @classmethod
    def _from_cspace(cls, env, cspace: Sequence[np.ndarray]) -> _CBGCSGeometry:
        shapes = tuple(cls.polygon_from_vertices(vertices) for vertices in cspace)
        if hasattr(env, "_CSpace_hpoly"):
            hpolys = tuple(env._CSpace_hpoly)
        else:
            hpolys = tuple(cls.hpoly_from_polygon(shape) for shape in shapes)
        return _CBGCSGeometry(
            shapes=shapes,
            hpolys=hpolys,
            neighbors=cls._neighbors_from_env(env, shapes),
        )

    @classmethod
    def _from_static_obstacles(cls, env) -> _CBGCSGeometry:
        lb = np.asarray(env.lb, dtype=float).reshape(2)
        ub = np.asarray(env.ub, dtype=float).reshape(2)
        domain = box(float(lb[0]), float(lb[1]), float(ub[0]), float(ub[1]))
        obstacles = [
            cls.polygon_from_vertices(obstacle.vertices)
            for obstacle in getattr(env, "O_Static", ())
            if hasattr(obstacle, "vertices")
        ]
        free_space = domain.difference(unary_union(obstacles)) if obstacles else domain
        shapes = tuple(CBGCSPlanner._convex_pieces(free_space, 1e-9))
        hpolys = tuple(cls.hpoly_from_polygon(shape) for shape in shapes)
        return _CBGCSGeometry(
            shapes=shapes,
            hpolys=hpolys,
            neighbors=cls._shape_neighbors(shapes),
        )

    @classmethod
    def _neighbors_from_env(cls, env, shapes: Sequence[Polygon]) -> Dict[int, set[int]]:
        neighbors = {idx: {idx} for idx in range(len(shapes))}
        if getattr(env, "edges", None):
            for lhs, rhs in env.edges:
                lhs_idx, rhs_idx = int(lhs), int(rhs)
                neighbors.setdefault(lhs_idx, {lhs_idx}).add(rhs_idx)
                neighbors.setdefault(rhs_idx, {rhs_idx}).add(lhs_idx)
            return neighbors
        return cls._shape_neighbors(shapes)

    @staticmethod
    def _shape_neighbors(shapes: Sequence[Polygon]) -> Dict[int, set[int]]:
        neighbors = {idx: {idx} for idx in range(len(shapes))}
        for lhs, rhs in combinations(range(len(shapes)), 2):
            if shapes[lhs].intersects(shapes[rhs]):
                neighbors[lhs].add(rhs)
                neighbors[rhs].add(lhs)
        return neighbors

    @classmethod
    def polygon_from_vertices(cls, vertices: np.ndarray) -> Polygon:
        polygon = Polygon(np.asarray(vertices, dtype=float))
        if not polygon.is_valid or polygon.area <= 0.0:
            polygon = polygon.convex_hull
        if polygon.is_empty or polygon.area <= 0.0:
            raise ValueError("CB-GCS received an empty spatial convex set.")
        return polygon

    @classmethod
    def hpoly_from_polygon(cls, polygon: Polygon) -> HPolyhedron:
        coords = np.asarray(polygon.exterior.coords[:-1], dtype=float)
        if coords.shape[0] < 3:
            raise ValueError("CB-GCS convex polygon must have at least three vertices.")
        return HPolyhedron(VPolytope(coords.T))


class CBGCSPlanner:
    def __init__(self, options: CBGCSOptions | None = None) -> None:
        self.options = CBGCSOptions() if options is None else options
        self._base_shapes: List[Polygon] = []
        self._base_hpolys: List[HPolyhedron] = []
        self._base_neighbors: Dict[int, set[int]] = {}
        self._low_level_calls = 0
        self._low_level_runtime = 0.0
        self._conflict_checks = 0
        self._conflict_runtime = 0.0

    def solve(
        self,
        instance,
        queries: Sequence[MPQuery],
        timeout_secs: float,
    ) -> CBGCSResult:
        start_time = time.perf_counter()
        deadline = start_time + float(timeout_secs)
        self._low_level_calls = 0
        self._low_level_runtime = 0.0
        self._conflict_checks = 0
        self._conflict_runtime = 0.0

        if len(queries) == 0:
            return CBGCSResult([], True, 0.0, cost=0.0)
        if any(abs(float(query.t_start)) > 1e-9 for query in queries):
            raise ValueError("CB-GCS follows the original formulation where all agents start at t=0.")

        self._init_base_geometry(instance.env)
        robot_width = self.options.resolved_robot_width(instance.env.robot_radius)
        shared_vlimit = min(float(query.vlimit) for query in queries)
        nominal_dt = self.options.resolved_time_step(robot_width, shared_vlimit)
        horizon = self.options.resolved_time_horizon(instance, queries, shared_vlimit, nominal_dt)
        num_steps = max(1, int(math.ceil(horizon / nominal_dt - 1e-12)))
        time_points = np.linspace(0.0, horizon, num_steps + 1)

        root_plans: List[CBGCSLowLevelPlan] = []
        for agent_idx, query in enumerate(queries):
            plan = self._low_level_plan(
                instance,
                agent_idx,
                query,
                (),
                time_points,
                deadline,
            )
            if plan is None:
                return self._result(
                    None,
                    False,
                    start_time,
                    math.inf,
                    0,
                    0,
                    0,
                    nominal_dt,
                    horizon,
                )
            root_plans.append(plan)

        root_cost = self._joint_cost(root_plans)
        root = _HighLevelNode(
            plans=root_plans,
            constraints=(),
            cost=root_cost,
            node_id=0,
            collision_score=self._collision_score(root_plans, robot_width),
            parent_id=-1,
            mod="lb",
        )
        state = _HighLevelSearchState(
            nodes={root.node_id: root},
            open_nodes={root.node_id: root},
            focal_nodes={},
        )
        self._update_high_level_bounds(state, root)
        self._anytime_focal_search(
            state,
            instance,
            queries,
            time_points,
            robot_width,
            deadline,
        )

        if state.solution_node is None:
            return self._result(
                None,
                False,
                start_time,
                math.inf,
                state.popped_nodes,
                state.generated_children,
                state.update_calls,
                nominal_dt,
                horizon,
            )
        return self._result(
            [plan.trajectory for plan in state.solution_node.plans],
            True,
            start_time,
            state.solution_node.cost,
            state.popped_nodes,
            state.generated_children,
            state.update_calls,
            nominal_dt,
            horizon,
        )

    def _result(
        self,
        solutions: Optional[List[STTrajectory]],
        success: bool,
        start_time: float,
        cost: float,
        popped_nodes: int,
        generated_children: int,
        update_calls: int,
        time_step: float,
        time_horizon: float,
    ) -> CBGCSResult:
        return CBGCSResult(
            solutions=solutions,
            success=success,
            runtime=time.perf_counter() - start_time,
            cost=float(cost),
            popped_nodes=int(popped_nodes),
            generated_children=int(generated_children),
            update_calls=int(update_calls),
            low_level_calls=self._low_level_calls,
            low_level_runtime=self._low_level_runtime,
            conflict_checks=self._conflict_checks,
            conflict_runtime=self._conflict_runtime,
            time_step=float(time_step),
            time_horizon=float(time_horizon),
        )

    def _anytime_focal_search(
        self,
        state: _HighLevelSearchState,
        instance,
        queries: Sequence[MPQuery],
        time_points: np.ndarray,
        robot_width: float,
        deadline: float,
    ) -> bool:
        weight = math.inf
        found_solution = False
        eps = 1e-2
        while weight >= 1.0 and time.perf_counter() < deadline:
            if not self._focal_search(
                state,
                instance,
                queries,
                time_points,
                robot_width,
                deadline,
                weight,
            ):
                break
            found_solution = True
            if not math.isfinite(state.ub_cost):
                break
            weight = max(1.0, state.ub_cost / (state.lb_cost + eps))
            for node_id, node in list(state.focal_nodes.items()):
                if node.cost >= state.ub_cost:
                    del state.focal_nodes[node_id]
        if found_solution:
            state.best_node = state.solution_node
        else:
            state.best_node = self._least_conflicting_node(state.nodes.values())
        return found_solution

    def _focal_search(
        self,
        state: _HighLevelSearchState,
        instance,
        queries: Sequence[MPQuery],
        time_points: np.ndarray,
        robot_width: float,
        deadline: float,
        weight: float,
    ) -> bool:
        while (
            (state.open_nodes or state.focal_nodes)
            and time.perf_counter() < deadline
            and state.popped_nodes < int(self.options.max_high_level_expansions)
        ):
            f_min = self._focal_f_min(state)
            if state.open_nodes:
                f_min = min(node.cost for node in state.open_nodes.values())
            self._update_focal_list(state, weight * f_min)
            if not state.focal_nodes:
                return False
            node = self._pop_focal_node(state)
            state.popped_nodes += 1

            if self._conflict_mode() == "lb":
                branches = self._first_conflict_branches(node, robot_width, mode="lb")
                if self._focal_finish(state, node, branches):
                    return True
                if not self._focal_expand(
                    state,
                    node,
                    branches,
                    mode="lb",
                    instance=instance,
                    queries=queries,
                    time_points=time_points,
                    robot_width=robot_width,
                    deadline=deadline,
                ):
                    return False
            else:
                branches = self._first_conflict_branches(node, robot_width, mode="ub", ub_level=1)
                if self._focal_finish(state, node, branches):
                    return True

                branches = self._first_conflict_branches(node, robot_width, mode="ub", ub_level=4)
                if self._focal_finish(state, node, branches):
                    return True
                if not self._focal_expand(
                    state,
                    node,
                    branches,
                    mode="ub",
                    instance=instance,
                    queries=queries,
                    time_points=time_points,
                    robot_width=robot_width,
                    deadline=deadline,
                ):
                    return False

                branches = self._first_conflict_branches(node, robot_width, mode="ub", ub_level=0)
                if branches and not self._focal_expand(
                    state,
                    node,
                    branches,
                    mode="ub",
                    instance=instance,
                    queries=queries,
                    time_points=time_points,
                    robot_width=robot_width,
                    deadline=deadline,
                ):
                    return False

                branches = self._first_conflict_branches(node, robot_width, mode="lb")
                if branches and not self._focal_expand(
                    state,
                    node,
                    branches,
                    mode="lb",
                    instance=instance,
                    queries=queries,
                    time_points=time_points,
                    robot_width=robot_width,
                    deadline=deadline,
                ):
                    return False
        return False

    def _focal_expand(
        self,
        state: _HighLevelSearchState,
        node: _HighLevelNode,
        branches: Dict[int, Tuple[CBGCSConstraint, ...]],
        mode: str,
        instance,
        queries: Sequence[MPQuery],
        time_points: np.ndarray,
        robot_width: float,
        deadline: float,
    ) -> bool:
        if not branches:
            return True
        for agent_idx, branch_constraints in branches.items():
            if time.perf_counter() >= deadline:
                return False
            new_constraints = node.constraints + tuple(branch_constraints)
            plan = self._low_level_plan(
                instance,
                agent_idx,
                queries[agent_idx],
                new_constraints,
                time_points,
                deadline,
            )
            state.update_calls += 1
            if plan is None:
                continue
            new_plans = list(node.plans)
            new_plans[agent_idx] = plan
            new_cost = self._joint_cost(new_plans)
            if new_cost >= state.ub_cost:
                continue
            new_node = _HighLevelNode(
                plans=new_plans,
                constraints=new_constraints,
                cost=new_cost,
                node_id=state.next_node_id,
                collision_score=self._collision_score(new_plans, robot_width),
                parent_id=node.node_id,
                mod=self._child_node_mode(node.mod, mode),
            )
            state.next_node_id += 1
            state.generated_children += 1
            state.nodes[new_node.node_id] = new_node
            state.open_nodes[new_node.node_id] = new_node
        return True

    def _focal_finish(
        self,
        state: _HighLevelSearchState,
        node: _HighLevelNode,
        branches: Dict[int, Tuple[CBGCSConstraint, ...]],
    ) -> bool:
        if branches:
            return False
        state.solution_node = node
        self._update_high_level_bounds(state, node)
        return True

    @classmethod
    def _update_focal_list(cls, state: _HighLevelSearchState, focal_bound: float) -> None:
        for node_id, node in sorted(list(state.open_nodes.items()), key=lambda item: (item[1].cost, item[0])):
            if node.cost > focal_bound + 1e-9:
                break
            del state.open_nodes[node_id]
            state.focal_nodes[node_id] = node

    @classmethod
    def _focal_f_min(cls, state: _HighLevelSearchState) -> float:
        if not state.focal_nodes:
            return math.inf
        return min(node.cost for node in state.focal_nodes.values())

    @classmethod
    def _pop_focal_node(cls, state: _HighLevelSearchState) -> _HighLevelNode:
        node = min(state.focal_nodes.values(), key=lambda item: (item.collision_score, item.node_id))
        del state.focal_nodes[node.node_id]
        return node

    @classmethod
    def _child_node_mode(cls, parent_mode: str, branch_mode: str) -> str:
        parent = str(parent_mode).lower()
        branch = str(branch_mode).lower()
        if parent == branch or branch == "ub":
            return branch
        if parent == "ub":
            return "ub"
        return "lb"

    def _update_high_level_bounds(self, state: _HighLevelSearchState, node: _HighLevelNode) -> None:
        if node.mod != "ub" and state.lb_cost <= node.cost:
            state.lb_cost = float(node.cost)
            state.lb_collision = min(state.lb_collision, float(node.collision_score))
        if node.collision_score == 0.0 and state.ub_cost >= node.cost:
            state.ub_cost = float(node.cost)
            state.ub_collision = min(state.ub_collision, float(node.collision_score))

    @classmethod
    def _least_conflicting_node(cls, nodes: Sequence[_HighLevelNode] | object) -> Optional[_HighLevelNode]:
        node_list = list(nodes)
        if not node_list:
            return None
        return min(node_list, key=lambda node: (node.collision_score, node.cost, node.node_id))

    def _conflict_mode(self) -> str:
        mode = str(self.options.conflict_mode).lower()
        if mode not in ("lb", "ub"):
            raise ValueError(f"CB-GCS conflict_mode must be 'lb' or 'ub', got {self.options.conflict_mode!r}.")
        return mode

    def _init_base_geometry(self, env) -> None:
        geometry = CBGCSGeometryAdapter.from_env(env)
        self._base_shapes = list(geometry.shapes)
        self._base_hpolys = list(geometry.hpolys)
        self._base_neighbors = geometry.neighbors

    @classmethod
    def _polygon_from_vertices(cls, vertices: np.ndarray) -> Polygon:
        return CBGCSGeometryAdapter.polygon_from_vertices(vertices)

    @classmethod
    def _hpoly_from_polygon(cls, polygon: Polygon) -> HPolyhedron:
        return CBGCSGeometryAdapter.hpoly_from_polygon(polygon)

    @classmethod
    def _convex_pieces(cls, shape, tol: float) -> List[Polygon]:
        if shape.is_empty:
            return []
        polygons: List[Polygon] = []
        geoms = list(shape.geoms) if hasattr(shape, "geoms") else [shape]
        for geom in geoms:
            if geom.is_empty:
                continue
            if geom.geom_type == "Polygon":
                polygons.extend(cls._triangulate_polygon(geom, tol))
            elif hasattr(geom, "geoms"):
                for subgeom in geom.geoms:
                    if subgeom.geom_type == "Polygon":
                        polygons.extend(cls._triangulate_polygon(subgeom, tol))
        return polygons

    @classmethod
    def _triangulate_polygon(cls, polygon: Polygon, tol: float) -> List[Polygon]:
        if polygon.area <= tol:
            return []
        triangles = constrained_delaunay_triangles(polygon)
        geoms = list(triangles.geoms) if hasattr(triangles, "geoms") else [triangles]
        pieces = [
            tri
            for tri in geoms
            if tri.geom_type == "Polygon" and tri.area > tol and polygon.buffer(tol).covers(tri)
        ]
        if not pieces and polygon.is_valid and polygon.area > tol:
            return [polygon.convex_hull]
        return pieces

    def _low_level_plan(
        self,
        instance,
        agent_idx: int,
        query: MPQuery,
        constraints: Sequence[CBGCSConstraint],
        time_points: np.ndarray,
        deadline: float,
    ) -> Optional[CBGCSLowLevelPlan]:
        call_start = time.perf_counter()
        self._low_level_calls += 1
        try:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                return None
            agent_constraints = tuple(c for c in constraints if c.agent_idx == agent_idx)
            min_arrival_idx = self._minimum_arrival_index(query, time_points)
            for arrival_idx in range(min_arrival_idx, len(time_points)):
                if time.perf_counter() >= deadline:
                    return None
                plan = self._low_level_plan_for_arrival(
                    instance,
                    query,
                    agent_constraints,
                    time_points,
                    arrival_idx,
                    deadline,
                    call_start,
                )
                if plan is not None:
                    return plan
            return None
        finally:
            self._low_level_runtime += time.perf_counter() - call_start

    def _low_level_plan_for_arrival(
        self,
        instance,
        query: MPQuery,
        constraints: Sequence[CBGCSConstraint],
        full_time_points: np.ndarray,
        arrival_idx: int,
        deadline: float,
        call_start: float,
    ) -> Optional[CBGCSLowLevelPlan]:
        arrival_idx = int(arrival_idx)
        arrival_time_points = np.asarray(full_time_points[: arrival_idx + 1], dtype=float)
        layer_sets, forbidden_by_layer = self._build_time_layers(query, constraints, arrival_time_points)
        if any(len(layer) == 0 for layer in layer_sets):
            return None

        gcs = GraphOfConvexSets()
        vertices = {
            layer_set.name: gcs.AddVertex(layer_set.convex_set, layer_set.name)
            for layer in layer_sets
            for layer_set in layer
        }
        edge_count = 0
        for layer_idx in range(len(arrival_time_points) - 1):
            dt = float(arrival_time_points[layer_idx + 1] - arrival_time_points[layer_idx])
            for source_set in layer_sets[layer_idx]:
                reachable = self._reachable_shape(
                    source_set.shape,
                    forbidden_by_layer[layer_idx + 1],
                    float(query.vlimit) * dt,
                )
                if reachable.is_empty:
                    continue
                for target_set in layer_sets[layer_idx + 1]:
                    if not self._base_transition_allowed(source_set, target_set, query):
                        continue
                    if not reachable.intersects(target_set.shape):
                        continue
                    edge = gcs.AddEdge(
                        vertices[source_set.name],
                        vertices[target_set.name],
                        self._edge_name(source_set.name, target_set.name),
                    )
                    self._add_edge_cost_and_constraints(edge, float(query.vlimit), dt)
                    edge_count += 1

        if edge_count == 0:
            return None
        source_vertex = vertices[layer_sets[0][0].name]
        target_vertex = vertices[layer_sets[-1][0].name]
        result = gcs.SolveShortestPath(
            source_vertex,
            target_vertex,
            self._solver_options(max(1e-6, deadline - time.perf_counter())),
        )
        if not self._has_solution(result):
            return None
        spatial_solution = self._rounded_spatial_plan(
            gcs,
            result,
            source_vertex,
            target_vertex,
            layer_sets,
            constraints,
            query,
            arrival_time_points,
            deadline,
            len(arrival_time_points) - 1,
        )
        if spatial_solution is None:
            return None
        trajectory, arrival_positions, path_length = spatial_solution
        full_positions = self._pad_arrival_positions(arrival_positions, full_time_points, arrival_idx)
        if not self._positions_satisfy_constraints(full_positions, constraints):
            return None
        if not self._trajectory_stays_in_cspace(
            instance.env,
            trajectory,
            self.options.cspace_validation_tolerance,
        ):
            return None
        arrival_time = float(full_time_points[arrival_idx] - full_time_points[0])
        if path_length > float(query.vlimit) * arrival_time + 1e-7:
            return None
        return CBGCSLowLevelPlan(
            trajectory=trajectory,
            positions=full_positions,
            cost=arrival_time,
            runtime=time.perf_counter() - call_start,
            arrival_time=arrival_time,
        )

    @classmethod
    def _minimum_arrival_index(cls, query: MPQuery, time_points: np.ndarray) -> int:
        if len(time_points) <= 1:
            return 0
        direct_time = float(np.linalg.norm(query.goal - query.start) / max(float(query.vlimit), 1e-9))
        return max(1, int(np.searchsorted(np.asarray(time_points, dtype=float), direct_time, side="left")))

    @classmethod
    def _pad_arrival_positions(
        cls,
        arrival_positions: np.ndarray,
        full_time_points: np.ndarray,
        arrival_idx: int,
    ) -> np.ndarray:
        arrival_positions = np.asarray(arrival_positions, dtype=float)
        expected = int(arrival_idx) + 1
        if arrival_positions.shape[0] != expected:
            raise ValueError(
                "CB-GCS arrival position count must match the arrival time index: "
                f"{arrival_positions.shape[0]} vs {expected}."
            )
        if arrival_positions.shape[0] == len(full_time_points):
            return arrival_positions
        goal = arrival_positions[-1:]
        padding = np.repeat(goal, len(full_time_points) - arrival_positions.shape[0], axis=0)
        return np.vstack([arrival_positions, padding])

    def _build_time_layers(
        self,
        query: MPQuery,
        constraints: Sequence[CBGCSConstraint],
        time_points: np.ndarray,
    ) -> Tuple[List[List[_LayerSet]], List[object]]:
        num_layers = len(time_points)
        constraints_by_layer: Dict[int, List[CBGCSConstraint]] = {idx: [] for idx in range(num_layers)}
        for constraint in constraints:
            if 0 <= constraint.time_step < num_layers:
                constraints_by_layer[constraint.time_step].append(constraint)

        forbidden_by_layer: List[object] = []
        for layer_idx in range(num_layers):
            polygons = [constraint.region.to_polygon() for constraint in constraints_by_layer[layer_idx]]
            forbidden_by_layer.append(unary_union(polygons) if polygons else GeometryCollection())

        layers: List[List[_LayerSet]] = []
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                start_point = np.asarray(query.start, dtype=float).reshape(2)
                if self._point_violates_constraints(start_point, constraints_by_layer[layer_idx]):
                    layers.append([])
                else:
                    layers.append([
                        _LayerSet(
                            layer_idx=layer_idx,
                            base_idx=None,
                            name=f"l{layer_idx}_start",
                            shape=ShapelyPoint(start_point),
                            convex_set=DrakePoint(start_point),
                        )
                    ])
                continue
            if layer_idx == num_layers - 1:
                goal_point = np.asarray(query.goal, dtype=float).reshape(2)
                if self._point_violates_constraints(goal_point, constraints_by_layer[layer_idx]):
                    layers.append([])
                else:
                    layers.append([
                        _LayerSet(
                            layer_idx=layer_idx,
                            base_idx=None,
                            name=f"l{layer_idx}_goal",
                            shape=ShapelyPoint(goal_point),
                            convex_set=DrakePoint(goal_point),
                        )
                    ])
                continue

            forbidden = forbidden_by_layer[layer_idx]
            layer_sets: List[_LayerSet] = []
            for base_idx, base_shape in enumerate(self._base_shapes):
                if forbidden.is_empty or not base_shape.intersects(forbidden):
                    pieces = [base_shape]
                else:
                    pieces = self._convex_pieces(
                        base_shape.difference(forbidden),
                        self.options.constraint_padding,
                    )
                for piece_idx, piece in enumerate(pieces):
                    layer_sets.append(
                        _LayerSet(
                            layer_idx=layer_idx,
                            base_idx=base_idx,
                            name=f"l{layer_idx}_b{base_idx}_p{piece_idx}",
                            shape=piece,
                            convex_set=self._hpoly_from_polygon(piece),
                        )
                    )
            layers.append(layer_sets)
        return layers, forbidden_by_layer

    @classmethod
    def _point_violates_constraints(
        cls,
        point: np.ndarray,
        constraints: Sequence[CBGCSConstraint],
    ) -> bool:
        return any(constraint.region.contains(point) for constraint in constraints)

    def _base_transition_allowed(
        self,
        source_set: _LayerSet,
        target_set: _LayerSet,
        query: MPQuery,
    ) -> bool:
        source_bases = self._candidate_base_indices(source_set, query.start, query.goal)
        target_bases = self._candidate_base_indices(target_set, query.start, query.goal)
        return any(
            target_base in self._base_neighbors.get(source_base, {source_base})
            for source_base in source_bases
            for target_base in target_bases
        )

    def _candidate_base_indices(
        self,
        layer_set: _LayerSet,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> Tuple[int, ...]:
        if layer_set.base_idx is not None:
            return (layer_set.base_idx,)
        point = start if layer_set.layer_idx == 0 else goal
        shapely_point = ShapelyPoint(np.asarray(point, dtype=float).reshape(2))
        indices = [
            idx
            for idx, shape in enumerate(self._base_shapes)
            if shape.buffer(self.options.constraint_padding).covers(shapely_point)
        ]
        return tuple(indices)

    @classmethod
    def _reachable_shape(cls, source_shape, forbidden_shape, max_distance: float):
        reachable = source_shape.buffer(float(max_distance))
        if not forbidden_shape.is_empty:
            reachable = reachable.difference(forbidden_shape)
        if not hasattr(reachable, "geoms"):
            return reachable
        source_buffer = source_shape.buffer(1e-9)
        pieces = [
            geom
            for geom in reachable.geoms
            if not geom.is_empty and geom.intersects(source_buffer)
        ]
        return unary_union(pieces) if pieces else GeometryCollection()

    @classmethod
    def _add_edge_cost_and_constraints(cls, edge, vlimit: float, dt: float) -> None:
        variables = np.append(edge.xu(), edge.xv())
        l2_matrix = np.array(
            [
                [-1.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 1.0],
            ]
        )
        edge.AddCost(Binding[Cost](L2NormCost(l2_matrix, np.zeros(2)), variables))
        cone_matrix = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 1.0],
            ]
        )
        cone_offset = np.array([float(vlimit) * float(dt), 0.0, 0.0])
        edge.AddConstraint(Binding[Constraint](LorentzConeConstraint(cone_matrix, cone_offset), variables))

    def _solver_options(self, max_runtime: float) -> GraphOfConvexSetsOptions:
        options = GraphOfConvexSetsOptions()
        options.convex_relaxation = True
        options.max_rounded_paths = int(self.options.max_rounded_paths)
        options.max_rounding_trials = int(self.options.max_rounded_paths)
        options.solver_options = SolverOptions()
        options.solver_options.SetOption(CommonSolverOption.kPrintToConsole, 0)
        options.solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_LOG", 0)
        options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_TOL_REL_GAP", float(self.options.low_level_gap))
        options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_MAX_TIME", float(max_runtime))
        options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_OPTIMIZER_MAX_TIME", float(max_runtime))
        return options

    @classmethod
    def _has_solution(cls, result) -> bool:
        details = result.get_solver_details()
        if isinstance(details, MosekSolverDetails):
            return bool(details.rescode == 0 and details.solution_status in (1, 9))
        return bool(result.is_success())

    def _rounded_spatial_plan(
        self,
        time_gcs,
        result,
        source_vertex,
        target_vertex,
        layer_sets: Sequence[Sequence[_LayerSet]],
        constraints: Sequence[CBGCSConstraint],
        query: MPQuery,
        time_points: np.ndarray,
        deadline: float,
        expected_edges: int,
    ) -> Optional[Tuple[STTrajectory, np.ndarray, float]]:
        paths = randomForwardPathSearch(
            time_gcs,
            result,
            source_vertex,
            target_vertex,
            max_paths=int(self.options.max_rounded_paths),
            max_trials=int(self.options.max_rounded_paths),
            seed=0,
        )
        layer_set_by_name = {
            layer_set.name: layer_set
            for layer in layer_sets
            for layer_set in layer
        }
        best_trajectory: Optional[STTrajectory] = None
        best_positions: Optional[np.ndarray] = None
        best_cost = math.inf
        for path in paths:
            edge_path = list(path)
            if len(edge_path) != expected_edges:
                continue
            if time.perf_counter() >= deadline:
                break
            spatial_path = self._spatial_path_from_edges(edge_path, layer_set_by_name, query)
            if len(spatial_path) < 2:
                continue
            restriction = self._solve_spatial_path_restriction(
                spatial_path,
                max(1e-6, deadline - time.perf_counter()),
            )
            if restriction is None:
                continue
            polyline, cost = restriction
            if cost > float(query.vlimit) * float(time_points[-1] - time_points[0]) + 1e-7:
                continue
            positions = self._sample_polyline_at_times(polyline, time_points)
            if not self._positions_satisfy_constraints(positions, constraints):
                continue
            if cost < best_cost:
                best_trajectory = self._trajectory_from_polyline(spatial_path, polyline, time_points)
                best_positions = positions
                best_cost = cost
        if best_trajectory is None or best_positions is None:
            return None
        return best_trajectory, best_positions, best_cost

    def _spatial_path_from_edges(
        self,
        edge_path: Sequence[object],
        layer_set_by_name: Dict[str, _LayerSet],
        query: MPQuery,
    ) -> List[_SpatialPathSet]:
        if not edge_path:
            return []
        path = [_SpatialPathSet("source", DrakePoint(np.asarray(query.start, dtype=float).reshape(2)))]
        last_base_idx: Optional[int] = None
        for edge in edge_path:
            source_set = layer_set_by_name[edge.u().name()]
            target_set = layer_set_by_name[edge.v().name()]
            transition = self._transition_base_pair(source_set, target_set, query)
            if transition is None:
                return []
            for base_idx in transition:
                if last_base_idx == base_idx:
                    continue
                path.append(self._spatial_base_path_set(base_idx))
                last_base_idx = base_idx
        path.append(_SpatialPathSet("target", DrakePoint(np.asarray(query.goal, dtype=float).reshape(2))))
        return path

    def _transition_base_pair(
        self,
        source_set: _LayerSet,
        target_set: _LayerSet,
        query: MPQuery,
    ) -> Optional[Tuple[int, int]]:
        source_bases = self._candidate_base_indices(source_set, query.start, query.goal)
        target_bases = self._candidate_base_indices(target_set, query.start, query.goal)
        for source_base in source_bases:
            neighbors = self._base_neighbors.get(source_base, {source_base})
            for target_base in target_bases:
                if target_base in neighbors:
                    return source_base, target_base
        return None

    def _spatial_base_path_set(self, base_idx: int) -> _SpatialPathSet:
        return _SpatialPathSet(f"b{int(base_idx)}", self._base_hpolys[int(base_idx)])

    def _solve_spatial_path_restriction(
        self,
        spatial_path: Sequence[_SpatialPathSet],
        max_runtime: float,
    ) -> Optional[Tuple[np.ndarray, float]]:
        result = self._solve_spatial_path_restriction_with_margin(
            spatial_path,
            max_runtime,
            float(self.options.spatial_restriction_margin),
        )
        if result is not None or float(self.options.spatial_restriction_margin) <= 0.0:
            return result
        return self._solve_spatial_path_restriction_with_margin(spatial_path, max_runtime, 0.0)

    def _solve_spatial_path_restriction_with_margin(
        self,
        spatial_path: Sequence[_SpatialPathSet],
        max_runtime: float,
        margin: float,
    ) -> Optional[Tuple[np.ndarray, float]]:
        if len(spatial_path) < 2:
            return None
        spatial_gcs = GraphOfConvexSets()
        vertices = [
            spatial_gcs.AddVertex(
                make_Cartesian_power_hpoly(self._spatial_restriction_set(layer_set.convex_set, margin), 2),
                layer_set.name,
            )
            for layer_set in spatial_path
        ]
        for vertex in vertices:
            self._add_spatial_segment_cost(vertex)
        edges = []
        for idx, (source, target) in enumerate(zip(vertices[:-1], vertices[1:])):
            edge = spatial_gcs.AddEdge(source, target, f"s{idx}")
            self._add_spatial_continuity_constraint(edge)
            edges.append(edge)
        restriction_result = spatial_gcs.SolveConvexRestriction(
            edges,
            self._restriction_solver_options(max_runtime),
        )
        if not restriction_result.is_success():
            return None
        segments = [vertex.GetSolution(restriction_result) for vertex in vertices]
        if any(segment is None for segment in segments):
            return None
        return self._polyline_from_spatial_segments(segments)

    @classmethod
    def _spatial_restriction_set(cls, convex_set, margin: float):
        if isinstance(convex_set, HPolyhedron):
            if margin <= 0.0:
                return convex_set
            return HPolyhedron(
                np.asarray(convex_set.A(), dtype=float),
                np.asarray(convex_set.b(), dtype=float).reshape(-1) - margin,
            )
        return convex_set

    def _restriction_solver_options(self, max_runtime: float) -> GraphOfConvexSetsOptions:
        options = self._solver_options(max_runtime)
        options.convex_relaxation = False
        options.max_rounded_paths = 0
        options.max_rounding_trials = 0
        return options

    @classmethod
    def _add_spatial_segment_cost(cls, vertex) -> None:
        matrix = np.array(
            [
                [-1.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 1.0],
            ]
        )
        vertex.AddCost(Binding[Cost](L2NormCost(matrix, np.zeros(2)), vertex.x()))

    @classmethod
    def _add_spatial_continuity_constraint(cls, edge) -> None:
        matrix = np.zeros((2, 8))
        matrix[:, 2:4] = np.eye(2)
        matrix[:, 4:6] = -np.eye(2)
        edge.AddConstraint(
            Binding[Constraint](
                LinearEqualityConstraint(matrix, np.zeros(2)),
                np.append(edge.xu(), edge.xv()),
            )
        )

    @classmethod
    def _polyline_from_spatial_segments(cls, segments: Sequence[np.ndarray]) -> Optional[Tuple[np.ndarray, float]]:
        if not segments:
            return None
        points = [np.asarray(segments[0], dtype=float).reshape(4)[:2]]
        cost = 0.0
        for segment in segments:
            values = np.asarray(segment, dtype=float).reshape(4)
            start = values[:2]
            end = values[2:]
            if not np.allclose(points[-1], start, atol=1e-6):
                points.append(start)
            length = float(np.linalg.norm(end - start))
            cost += length
            points.append(end)
        return np.vstack(points), cost

    @classmethod
    def _sample_polyline_at_times(cls, polyline: np.ndarray, time_points: np.ndarray) -> np.ndarray:
        total_length = cls._polyline_length(polyline)
        if total_length <= 1e-12:
            return np.repeat(polyline[:1], len(time_points), axis=0)
        duration = float(time_points[-1] - time_points[0])
        if duration <= 0.0:
            return np.repeat(polyline[:1], len(time_points), axis=0)
        distances = total_length * (np.asarray(time_points, dtype=float) - float(time_points[0])) / duration
        return cls._sample_polyline_at_distances(polyline, distances)

    @classmethod
    def _sample_polyline_at_distances(cls, polyline: np.ndarray, distances: np.ndarray) -> np.ndarray:
        points = np.asarray(polyline, dtype=float)
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        total_length = float(cumulative[-1])
        samples = []
        for distance in np.asarray(distances, dtype=float):
            target = min(max(float(distance), 0.0), total_length)
            idx = int(np.searchsorted(cumulative, target, side="right") - 1)
            idx = min(max(idx, 0), len(segment_lengths) - 1)
            length = float(segment_lengths[idx])
            if length <= 1e-12:
                samples.append(points[idx].copy())
                continue
            alpha = (target - float(cumulative[idx])) / length
            samples.append((1.0 - alpha) * points[idx] + alpha * points[idx + 1])
        return np.vstack(samples)

    @classmethod
    def _trajectory_from_polyline(
        cls,
        spatial_path: Sequence[_SpatialPathSet],
        polyline: np.ndarray,
        time_points: np.ndarray,
    ) -> STTrajectory:
        total_length = cls._polyline_length(polyline)
        if total_length <= 1e-12:
            point = np.asarray(polyline[0], dtype=float)
            return STTrajectory(
                ["wait"],
                [np.hstack([point, time_points[0], point, time_points[-1]])],
                dim=2,
            )
        points = [
            np.hstack([start, t_start, end, t_end])
            for start, end, t_start, t_end in cls._timed_polyline_segments(polyline, time_points, total_length)
        ]
        vertex_names = [layer_set.name for layer_set in spatial_path]
        vertex_path = vertex_names[: len(points)]
        return STTrajectory(vertex_path, points, dim=2)

    @classmethod
    def _timed_polyline_segments(
        cls,
        polyline: np.ndarray,
        time_points: np.ndarray,
        total_length: float,
    ) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
        t0 = float(time_points[0])
        duration = float(time_points[-1] - time_points[0])
        segments = []
        cumulative = 0.0
        for start, end in zip(polyline[:-1], polyline[1:]):
            length = float(np.linalg.norm(end - start))
            if length <= 1e-12:
                continue
            t_start = t0 + duration * cumulative / total_length
            cumulative += length
            t_end = t0 + duration * cumulative / total_length
            segments.append((start, end, t_start, t_end))
        return segments

    @classmethod
    def _positions_satisfy_constraints(
        cls,
        positions: np.ndarray,
        constraints: Sequence[CBGCSConstraint],
    ) -> bool:
        for constraint in constraints:
            if 0 <= int(constraint.time_step) < len(positions) and constraint.region.contains(positions[constraint.time_step]):
                return False
        return True

    @classmethod
    def _polyline_length(cls, polyline: np.ndarray) -> float:
        if len(polyline) <= 1:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(polyline, axis=0), axis=1)))

    @classmethod
    def _trajectory_stays_in_cspace(cls, env, trajectory: STTrajectory, tol: float) -> bool:
        return all(
            env.is_segment_in_CSpace(
                trajectory.xA(segment_idx)[:-1],
                trajectory.xB(segment_idx)[:-1],
                tol=float(tol),
            )
            for segment_idx in range(trajectory.size)
        )

    @classmethod
    def _edge_name(cls, source_name: str, target_name: str) -> str:
        return f"({source_name},{target_name})"

    @classmethod
    def _path_length(cls, positions: np.ndarray) -> float:
        if len(positions) <= 1:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))

    @classmethod
    def _joint_cost(cls, plans: Sequence[CBGCSLowLevelPlan]) -> float:
        return float(sum(plan.cost for plan in plans))

    def _detect_conflict(
        self,
        plans: Sequence[CBGCSLowLevelPlan],
        robot_width: float,
    ) -> Optional[CBGCSConflict]:
        start = time.perf_counter()
        self._conflict_checks += 1
        try:
            tol = float(self.options.conflict_tolerance)
            for robot_i, robot_j in combinations(range(len(plans)), 2):
                positions_i = plans[robot_i].positions
                positions_j = plans[robot_j].positions
                for time_step, (pos_i, pos_j) in enumerate(zip(positions_i, positions_j)):
                    delta = np.abs(pos_i - pos_j)
                    if np.all(delta < robot_width - tol):
                        return CBGCSConflict(
                            robot_idx_i=robot_i,
                            robot_idx_j=robot_j,
                            time_step=time_step,
                            position_i=pos_i.copy(),
                            position_j=pos_j.copy(),
                        )
            return None
        finally:
            self._conflict_runtime += time.perf_counter() - start

    def _first_conflict_branches(
        self,
        node: _HighLevelNode,
        robot_width: float,
        mode: str,
        ub_level: int = 0,
    ) -> Dict[int, Tuple[CBGCSConstraint, ...]]:
        conflict = self._detect_conflict(node.plans, robot_width)
        if conflict is None:
            return {}
        constraints = self._constraints_from_conflict_mode(
            conflict,
            node.plans,
            robot_width,
            mode=mode,
            ub_level=ub_level,
        )
        branches: Dict[int, List[CBGCSConstraint]] = {}
        for constraint in constraints:
            branches.setdefault(constraint.agent_idx, []).append(constraint)
        return {agent_idx: tuple(values) for agent_idx, values in branches.items()}

    def _constraints_from_conflict_mode(
        self,
        conflict: CBGCSConflict,
        plans: Sequence[CBGCSLowLevelPlan],
        robot_width: float,
        mode: str,
        ub_level: int = 0,
    ) -> Tuple[CBGCSConstraint, ...]:
        constraints = self._constraints_from_conflict(conflict, plans, robot_width)
        mode = str(mode).lower()
        if mode == "lb":
            return tuple(constraint for constraint in constraints if constraint.kind == "lb")
        if mode == "ub":
            upper_constraints = tuple(constraint for constraint in constraints if constraint.kind == "ub")
            if int(ub_level) <= 1:
                return upper_constraints
            expanded_constraints: List[CBGCSConstraint] = []
            for constraint in upper_constraints:
                for offset in range(int(ub_level)):
                    expanded_constraints.append(
                        CBGCSConstraint(
                            agent_idx=constraint.agent_idx,
                            time_step=constraint.time_step + offset,
                            region=constraint.region,
                            kind=constraint.kind,
                        )
                    )
            return tuple(expanded_constraints)
        raise ValueError(f"Unknown CB-GCS conflict branch mode {mode!r}.")

    def _constraints_from_conflict(
        self,
        conflict: CBGCSConflict,
        plans: Sequence[CBGCSLowLevelPlan],
        robot_width: float,
    ) -> Tuple[CBGCSConstraint, ...]:
        pos_i = conflict.position_i
        pos_j = conflict.position_j
        center = 0.5 * (pos_i + pos_j)
        lower_region = CBGCSRectangle.from_center_lengths(
            center,
            np.array([robot_width, robot_width]),
            self.options.constraint_padding,
        )
        constraints = [
            CBGCSConstraint(conflict.robot_idx_i, conflict.time_step, lower_region, "lb"),
            CBGCSConstraint(conflict.robot_idx_j, conflict.time_step, lower_region, "lb"),
        ]

        if conflict.time_step < plans[0].positions.shape[0] - 1:
            step = conflict.time_step
            pi = plans[conflict.robot_idx_i].positions[step + 1] - plans[conflict.robot_idx_i].positions[step]
            pj = plans[conflict.robot_idx_j].positions[step + 1] - plans[conflict.robot_idx_j].positions[step]
            q_j_i = pj - pi
            q_i_j = pi - pj
            constraints.extend(
                [
                    CBGCSConstraint(
                        conflict.robot_idx_i,
                        step,
                        CBGCSRectangle.from_center_lengths(
                            pos_j + 0.5 * q_j_i,
                            np.abs(q_j_i) + robot_width,
                            self.options.constraint_padding,
                        ),
                        "ub",
                    ),
                    CBGCSConstraint(
                        conflict.robot_idx_j,
                        step,
                        CBGCSRectangle.from_center_lengths(
                            pos_i + 0.5 * q_i_j,
                            np.abs(q_i_j) + robot_width,
                            self.options.constraint_padding,
                        ),
                        "ub",
                    ),
                ]
            )
        return tuple(constraints)

    def _collision_score(
        self,
        plans: Sequence[CBGCSLowLevelPlan],
        robot_width: float,
    ) -> float:
        samples = max(1, int(self.options.conflict_score_samples))
        score = 0.0
        for robot_i, robot_j in combinations(range(len(plans)), 2):
            positions_i = plans[robot_i].positions
            positions_j = plans[robot_j].positions
            for step in range(len(positions_i) - 1):
                for sample_idx in range(samples):
                    ratio = sample_idx / samples
                    pos_i = (1.0 - ratio) * positions_i[step] + ratio * positions_i[step + 1]
                    pos_j = (1.0 - ratio) * positions_j[step] + ratio * positions_j[step + 1]
                    overlap = np.maximum(0.0, robot_width - np.abs(pos_i - pos_j))
                    score += float(overlap[0] * overlap[1])
        return score
