# modified from https://github.com/shaoyuancc/large_gcs

import logging
import pickle
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import (
    Binding,
    Constraint,
    Cost,
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    MathematicalProgramResult,
    Parallelism,
    SolverOptions,
    MosekSolver, CommonSolverOption
)
from tqdm import tqdm

from mrmp.geometry.convex_set import ConvexSet
from mrmp.interval import Interval

logger = logging.getLogger(__name__)


@dataclass
class ShortestPathSolution:
    # Whether the optimization was successful
    is_success: bool
    # Optimal cost of the path
    cost: float
    # Time to solve the optimization problem
    time: float
    # List of vertex names and discrete coordinates in the path
    vertex_path: List[str]
    # List of continuous valued points in each vertex along the path
    trajectory: List[np.ndarray]
    # Result of the optimization
    result: Optional[MathematicalProgramResult] = None
    # the naem of ConvexTimedSet that contains source
    # the trajectory time interval
    itvl: Optional[Interval] = None
    # dimension of the ambient space
    dim: Optional[int] = None

    def __str__(self):
        result = []
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, float):
                # Format the float to 3 significant figures
                value = "{:.3g}".format(value)
            elif field.name == "ambient_path":
                value = self.ambient_path_str
            result.append(f"{field.name}: {value}")
        return ", ".join(result)

    def to_serializable_dict(self) -> dict:
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, MathematicalProgramResult):
                continue
            result[field.name] = value
        return result
    
    def lerp(self, t:float) -> np.ndarray:
        if t <= self.itvl.start:
            return self.trajectory[0][:self.dim]

        if t >= self.itvl.end:
            return self.trajectory[-1][-self.dim:]

        for idx in range(len(self.trajectory)):
            xa, xb = self.trajectory[idx][:self.dim], self.trajectory[idx][-self.dim:]
            if round(t - xa[-1], 6) >= 0 and round(t - xb[-1], 6) <= 0:
                break
        
        assert round(t - xa[-1], 6) >= 0 and round(t - xb[-1], 6) <= 0, f"t: {t} not in [{xa[-1]}, {xb[-1]}]"
        
        k = (t - xa[-1]) / (xb[-1] - xa[-1])
        return xa * (1 - k) + xb * k

    @property
    def ambient_path_str(self):
        ambient_path = "["
        for a in self.trajectory:
            ambient_path += (
                f"{np.array2string(a, separator=', ', max_line_width=np.inf)},"
            )
        ambient_path += "]"
        return ambient_path

    def save(self, loc: Path) -> None:
        """Save solution as a .pkl file.

        Note that MathematicalProgramResult cannot be serialized, and
        hence cannot be saved to file.
        """
        # Convert to dictionary and save to JSON
        self_as_dict = self.to_serializable_dict()
        with open(loc, "wb") as f:
            pickle.dump(self_as_dict, f)

    @classmethod
    def load(cls, loc: Path) -> "ShortestPathSolution":
        """Read a solution from a .pkl file."""

        with open(loc, "rb") as f:
            loaded_dict = pickle.load(f)
        
        field_names = {f.name for f in fields(ShortestPathSolution) if f.init}
        filtered_arg_dict = {k: v for k, v in loaded_dict.items() if k in field_names}
        
        return ShortestPathSolution(**filtered_arg_dict)


@dataclass
class DefaultGraphCostsConstraints:
    """Class to hold default costs and constraints for vertices and edges."""

    vertex_costs: List[Cost] = None
    vertex_constraints: List[Constraint] = None
    edge_costs: List[Cost] = None
    edge_constraints: List[Constraint] = None


@dataclass
class Vertex:
    convex_set: ConvexSet
    # Set to empty list to override default costs to be no cost
    costs: List[Cost] = None
    # Set to empty list to override default constraints to be no constraint
    constraints: List[Constraint] = None
    # This will be overwritten when adding vertex to graph
    gcs_vertex: Optional[GraphOfConvexSets.Vertex] = None

    name: str = ""
    itvl: Interval = None
    space_bounds: List[Interval] = None


@dataclass
class Edge:
    # Source/"left" vertex of the edge
    u: str
    # Target/"right" vertex of the edge
    v: str
    # Set to empty list to override default costs to be no cost
    costs: List[Cost] = None
    # Set to empty list to override default constraints to be no constraint
    constraints: List[Constraint] = None
    # This will be overwritten when adding edge to graph
    gcs_edge: Optional[GraphOfConvexSets.Edge] = None
    # Optional key suffix to distinguish between edges with the same source and target
    key_suffix: Optional[str] = None

    def __post_init__(self):
        self._key = self.key_from_uv(self.u, self.v, self.key_suffix)

    @property
    def key(self):
        return self._key

    @staticmethod
    def key_from_uv(u: str, v: str, key_suffix: Optional[str] = None):
        if key_suffix:
            return f'("{u}", "{v}")_{key_suffix}'
        return f'("{u}", "{v}")'


@dataclass
class GraphParams:
    # Tuple of the smallest and largest ambient dimension of the vertices
    dim_bounds: Tuple[int, int]
    # Number of vertices/convex sets
    n_vertices: int
    # Number of edges
    n_edges: int
    # Source and target coordinates, (centers if they are non singleton sets)
    source: Tuple
    target: Tuple
    # Workspace bounds if specified
    workspace: np.ndarray
    # Default costs and constraints if any
    default_costs_constraints: DefaultGraphCostsConstraints


class Graph:
    """Wrapper for Drake GraphOfConvexSets class."""

    def __init__(
        self,
        default_costs_constraints: DefaultGraphCostsConstraints = None,
        workspace: np.ndarray = None,
    ) -> None:
        """
        Args:
            default_costs_constraints: Default costs and constraints for vertices and edges.
            workspace_bounds: Bounds on the workspace that will be plotted. Each row should specify lower and upper bounds for a dimension.
        """

        self._default_costs_constraints = default_costs_constraints
        self.vertices: Dict[str, Vertex] = {}
        self.edges: Dict[str, Edge] = {}
        self._adjacency_list: Dict[str, List[str]] = defaultdict(list)
        self._source_name = None
        self._target_name = None

        if workspace is not None:
            assert (
                workspace.shape[1] == 2
            ), "Each row of workspace_bounds should specify lower and upper bounds for a dimension"
            if workspace.shape[0] > 2:
                raise NotImplementedError(
                    "Workspace bounds with more than 2 dimensions not yet supported"
                )
        self.workspace = workspace

        self._gcs = GraphOfConvexSets()

    def add_vertex(
        self, vertex: Vertex, name: str = "", should_add_to_gcs: bool = True
    ):
        """Add a vertex to the graph."""
        if name == "":
            name = len(self.vertices)
        assert name not in self.vertices

        # Set default costs and constraints if necessary
        v = vertex
        if self._default_costs_constraints:  # Have defaults
            if (
                v.costs
                is None  # Vertex have not been specifically overriden to be no cost (empty list)
                and self._default_costs_constraints.vertex_costs
            ):
                v.costs = self._default_costs_constraints.vertex_costs
            if (
                v.constraints is None
                and self._default_costs_constraints.vertex_constraints
            ):
                v.constraints = self._default_costs_constraints.vertex_constraints
        # Makes a copy so that the original vertex is not modified
        # allows for convenient adding of vertices from one graph to another
        v = copy(vertex)
        if should_add_to_gcs:
            v.gcs_vertex = self._gcs.AddVertex(v.convex_set.set, name)
            # Add costs and constraints to gcs vertex
            if v.costs:
                for cost in v.costs:
                    binding = Binding[Cost](cost, v.gcs_vertex.x().flatten())
                    v.gcs_vertex.AddCost(binding)
            if v.constraints:
                for constraint in v.constraints:
                    binding = Binding[Constraint](
                        constraint, v.gcs_vertex.x().flatten()
                    )
                    v.gcs_vertex.AddConstraint(binding)

        self.vertices[name] = v

    def remove_vertex(self, name: str):
        """Remove a vertex from the graph as well as any edges from or to that
        vertex."""
        self._gcs.RemoveVertex(self.vertices[name].gcs_vertex)
        self.vertices.pop(name)
        for edge in self.edge_keys:
            e = self.edges[edge]
            if name == e.u or name == e.v:
                self.remove_edge(
                    edge, remove_from_gcs=False
                )  # gcs.RemoveVertex already removes edges from gcs

    def add_vertices_from_sets(
        self, sets: List[ConvexSet], costs=None, constraints=None, names=None
    ):
        """Add vertices to the graph.

        Each vertex is a convex set.
        """
        if names is None:
            names = [None] * len(sets)
        else:
            assert len(sets) == len(names)
        if costs is None:
            costs = [None] * len(sets)
        else:
            assert len(costs) == len(sets)
        if constraints is None:
            constraints = [None] * len(sets)
        else:
            assert len(constraints) == len(sets)

        logger.info(f"Adding {len(sets)} vertices to graph...")
        for set, name, cost_list, constraint_list in tqdm(
            list(zip(sets, names, costs, constraints))
        ):
            self.add_vertex(Vertex(set, cost_list, constraint_list), name)

    def add_edge(self, edge: Edge, should_add_to_gcs: bool = True):
        """Add an edge to the graph."""
        e = copy(edge)
        # Set default costs and constraints if necessary
        if self._default_costs_constraints:  # Have defaults
            if (
                e.costs
                is None  # Edge have not been specifically overriden to be no cost (empty list)
                and self._default_costs_constraints.edge_costs
            ):
                e.costs = self._default_costs_constraints.edge_costs
            if (
                e.constraints is None
                and self._default_costs_constraints.edge_constraints
            ):
                e.constraints = self._default_costs_constraints.edge_constraints

        if should_add_to_gcs:
            e.gcs_edge = self._gcs.AddEdge(
                u=self.vertices[e.u].gcs_vertex,
                v=self.vertices[e.v].gcs_vertex,
                name=e.key,
            )

            # Add costs and constraints to gcs edge
            if e.costs:
                for cost in e.costs:
                    x = np.concatenate([e.gcs_edge.xu(), e.gcs_edge.xv()])
                    binding = Binding[Cost](cost, x)
                    e.gcs_edge.AddCost(binding)
            if e.constraints:
                for constraint in e.constraints:
                    x = np.concatenate([e.gcs_edge.xu(), e.gcs_edge.xv()])
                    binding = Binding[Constraint](constraint, x)
                    e.gcs_edge.AddConstraint(binding)

        self.edges[e.key] = e
        self._adjacency_list[e.u].append(e.v)

        return e

    def remove_edge(self, edge_key: str, remove_from_gcs: bool = True):
        """Remove an edge from the graph."""
        e = self.edges[edge_key]
        if remove_from_gcs:
            self._gcs.RemoveEdge(e.gcs_edge)

        self.edges.pop(edge_key)
        self._adjacency_list[e.u].remove(e.v)

    def add_edges_from_vertex_names(
        self,
        us: List[str],
        vs: List[str],
        costs: List[List[Cost]] = None,
        constraints: List[List[Constraint]] = None,
    ):
        """Add edges to the graph."""
        assert len(us) == len(vs)
        if costs is None:
            costs = [None] * len(us)
        else:
            assert len(costs) == len(us)
        if constraints is None:
            constraints = [None] * len(us)
        else:
            assert len(constraints) == len(us)

        logger.info(f"Adding {len(us)} edges to graph...")
        for u, v, cost_list, constraint_list in tqdm(
            list(zip(us, vs, costs, constraints))
        ):
            self.add_edge(Edge(u, v, cost_list, constraint_list))

    def set_source(self, vertex_name: str):
        assert vertex_name in self.vertices, f"{vertex_name} not in graph vertices"
        self._source_name = vertex_name

    def set_target(self, vertex_name: str):
        assert vertex_name in self.vertices
        self._target_name = vertex_name

    def successors(self, vertex_name: str) -> List[str]:
        """Get the successors of a vertex."""
        return self._adjacency_list[vertex_name]

    def outgoing_edges(self, vertex_name: str) -> List[Edge]:
        """Get the outgoing edges of a vertex.

        Note that this is an expensive call and should not be used in
        the loop of a search algorithm.
        """
        assert vertex_name in self.vertices
        return [edge for edge in self.edges.values() if edge.u == vertex_name]

    def incoming_edges(self, vertex_name: str) -> List[Edge]:
        """Get the incoming edges of a vertex.

        Note that this is an expensive call and should not be used in
        the loop of a search algorithm.
        """
        assert vertex_name in self.vertices
        return [edge for edge in self.edges.values() if edge.v == vertex_name]

    def incident_edges(self, vertex_name: str) -> List[Edge]:
        """Get the incident edges of a vertex.

        Note that this is an expensive call and should not be used in
        the loop of a search algorithm.
        """
        assert vertex_name in self.vertices
        return [
            edge
            for edge in self.edges.values()
            if edge.u == vertex_name or edge.v == vertex_name
        ]

    def is_neighbor(self, u: str, v: str) -> bool:
        return v in self.successors(u) or u in self.successors(v)

    def solve_shortest_path(
        self, max_rounded_paths:int, max_rounding_trials:int
    ) -> ShortestPathSolution:
        """Solve the shortest path problem."""
        assert self._source_name is not None
        assert self._target_name is not None
        
        options = GraphOfConvexSetsOptions()
        solver_options = SolverOptions()
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3)
        solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)
        solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_TOL_REL_GAP", 1e-3)
        solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_MAX_TIME", 3600.0)
        options.convex_relaxation = True
        options.preprocessing = False

        options.max_rounded_paths = max_rounded_paths
        options.max_rounding_trials = max_rounding_trials
        rounded_result = self._gcs.SolveShortestPath(
            self.vertices[self._source_name].gcs_vertex,
            self.vertices[self._target_name].gcs_vertex,
            options
        )
        sol = self._parse_result(rounded_result)
        self._post_solve(sol)
        return sol
    
    def solve_shortest_path_optimally(self) -> ShortestPathSolution:
        assert self._source_name is not None
        assert self._target_name is not None
        
        options = GraphOfConvexSetsOptions()
        solver_options = SolverOptions()
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3)
        solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)
        solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_TOL_REL_GAP", 1e-3)
        solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_MAX_TIME", 3600.0)
        options.convex_relaxation = False
        options.preprocessing = False

        rounded_result = self._gcs.SolveShortestPath(
            self.vertices[self._source_name].gcs_vertex,
            self.vertices[self._target_name].gcs_vertex,
            options
        )
        sol = self._parse_result(rounded_result)
        self._post_solve(sol)
        return sol

    def solve_convex_restriction(
        self,
        active_edge_keys: List[str],
        skip_post_solve: bool = False,
        should_return_result: bool = False,
        solver_options: Optional[SolverOptions] = None,
    ) -> ShortestPathSolution:
        edge_path = [self.edges[edge_key] for edge_key in active_edge_keys]
        vertex_name_path = self._convert_conv_res_edges_to_vertex_path(edge_path)
        vertex_path = [self.vertices[vertex_name] for vertex_name in vertex_name_path]
        # gcs_edges = [edge.gcs_edge for edge in edge_path]
        # logger.debug(f"Solving convex restriction for vpath: {vertex_name_path}")
        # logger.debug(f"Edge path: {active_edge_keys}")

        # BUILD DUPLICATE DRAKE GCS GRAPH (Workaround to allow for cycles)
        gcs = GraphOfConvexSets()
        gcs_vertex_path = []
        for vertex_name, v in zip(vertex_name_path, vertex_path):
            gcs_vertex = gcs.AddVertex(
                self.vertices[vertex_name].convex_set.set, vertex_name
            )
            gcs_vertex_path.append(gcs_vertex)
            # Add costs and constraints to gcs
            if v.costs:
                for cost in v.costs:
                    binding = Binding[Cost](cost, gcs_vertex.x().flatten())
                    gcs_vertex.AddCost(binding)
            if v.constraints:
                for constraint in v.constraints:
                    binding = Binding[Constraint](constraint, gcs_vertex.x().flatten())
                    gcs_vertex.AddConstraint(binding)
        gcs_edge_path = []
        for i, e in enumerate(edge_path):
            gcs_edge = gcs.AddEdge(
                u=gcs_vertex_path[i],
                v=gcs_vertex_path[i + 1],
                name=e.key,
            )
            gcs_edge_path.append(gcs_edge)
            # Add costs and constraints to gcs edge
            if e.costs:
                for cost in e.costs:
                    x = np.concatenate([gcs_edge.xu(), gcs_edge.xv()])
                    binding = Binding[Cost](cost, x)
                    gcs_edge.AddCost(binding)
            if e.constraints:
                for constraint in e.constraints:
                    x = np.concatenate([gcs_edge.xu(), gcs_edge.xv()])
                    binding = Binding[Constraint](constraint, x)
                    gcs_edge.AddConstraint(binding)

        if solver_options is not None:
            self._gcs_options_wo_relaxation.solver_options = solver_options

        # Now solve convex restriction on the duplicate graph
        result = gcs.SolveConvexRestriction(
            gcs_edge_path,
            self._gcs_options_wo_relaxation,
        )
        self._gcs_options_wo_relaxation.solver_options = SolverOptions()

        if skip_post_solve:
            sol = self._parse_partial_convex_restriction_result(
                result, should_return_result
            )
        else:
            sol = self._parse_convex_restriction_result(
                result, vertex_name_path, gcs_vertex_path, should_return_result
            )
            # Optional post solve hook for subclasses
            self._post_solve(sol)

        return sol

    def solve_convex_restrictions(
        self,
        active_edge_keys: List[List[str]],
    ) -> List[ShortestPathSolution]:
        """Solve multiple convex restrictions.

        Note that to use this, have to use Drake built from this source: https://github.com/shaoyuancc/drake/tree/parallelize-convex-restrictions.
        This is used in generating the lower bound graph.
        """

        paths = [
            [self.edges[edge_key] for edge_key in path] for path in active_edge_keys
        ]
        gcs_paths = [[edge.gcs_edge for edge in path] for path in paths]

        all_results: List[MathematicalProgramResult] = (
            self._gcs.SolveConvexRestrictions(
                active_edges=gcs_paths, parallelism=Parallelism(True)
            )
        )

        sols = []
        for e_path, result in zip(paths, all_results):
            v_path = self._convert_conv_res_edges_to_vertex_path(e_path)
            a_path = [
                result.GetSolution(self.vertices[v].gcs_vertex.x()) for v in v_path
            ]

            sols.append(
                ShortestPathSolution(
                    result.is_success(),
                    result.get_optimal_cost(),
                    result.get_solver_details().optimizer_time,
                    v_path,
                    a_path,
                    None,
                    None,
                )
            )
        return sols

    def check_feasiblity(self) -> bool:
        if self._source_name is None or self._target_name is None:
            raise ValueError("Source and target not set")

        stack = [self._source_name]
        visited = set()
        while stack:
            v = stack.pop()
            if v == self._target_name:
                return True
            visited.add(v)
            for u in self.successors(v):
                if u not in visited:
                    stack.append(u)

        return False

    def _post_solve(self, sol: ShortestPathSolution):
        """Optional post solve hook for subclasses."""

    @staticmethod
    def _parse_partial_convex_restriction_result(
        result: MathematicalProgramResult, should_return_result: bool = False
    ) -> ShortestPathSolution:
        """Only return is_success, cost and time.

        Skip the work of processing the vertex, edge, ambient and flow
        paths.
        """
        cost = result.get_optimal_cost()
        time = result.get_solver_details().optimizer_time

        return ShortestPathSolution(
            result.is_success(),
            cost,
            time,
            None,
            None,
            result=result if should_return_result else None,
        )

    @staticmethod
    def _parse_convex_restriction_result(
        result: MathematicalProgramResult,
        vertex_name_path: List[str],
        gcs_vertex_path: List,
        should_return_result: bool = False,
    ) -> ShortestPathSolution:
        cost = result.get_optimal_cost()
        time = result.get_solver_details().optimizer_time

        ambient_path = []
        if result.is_success():
            ambient_path = [result.GetSolution(v.x()) for v in gcs_vertex_path]

        return ShortestPathSolution(
            result.is_success(),
            cost,
            time,
            vertex_name_path,
            ambient_path,
            result=result if should_return_result else None,
        )

    def _parse_result(self, result: MathematicalProgramResult) -> ShortestPathSolution:
        cost = result.get_optimal_cost()
        time = result.get_solver_details().optimizer_time
        vertex_path = []
        ambient_path = []
        flows = []
        if result.is_success():
            flow_variables = [e.phi() for e in self._gcs.Edges()]
            flows = [result.GetSolution(p) for p in flow_variables]
            edge_path = []
            for k, flow in enumerate(flows):
                if flow >= 0.99:
                    edge_path.append(self.edges[self.edge_keys[k]])
            assert len(self._gcs.Edges()) == self.n_edges
            # Edges are in order they were added to the graph and not in order of the path
            vertex_path = self._convert_active_edges_to_vertex_path(
                self.source_name, self.target_name, edge_path
            )
            # vertex_path = [self.source_name]
            ambient_path = [
                result.GetSolution(self.vertices[v].gcs_vertex.x()) for v in vertex_path
            ]

        return ShortestPathSolution(
            result.is_success(), cost, time, vertex_path, ambient_path, result
        )

    @staticmethod
    def _convert_conv_res_edges_to_vertex_path(
        edges: List[Edge],
    ):
        v_path = [edges[0].u]
        for e in edges:
            v_path += [e.v]
        return v_path

    @staticmethod
    def _convert_active_edges_to_vertex_path(
        source_name,
        target_name,
        edges: List[Edge],
    ):

        # Create a dictionary where the keys are the vertices and the values are their neighbors
        neighbors = {e.u: e.v for e in edges}
        # Start with the source vertex
        path = [source_name]

        # While the last vertex in the path has a neighbor

        while path[-1] in neighbors:
            if neighbors[path[-1]] in path:
                # We have a cycle
                raise RuntimeError(
                    f"Cycle detected in path {np.array(path)}\n{np.array(edges)}"
                )
            # Add the neighbor to the path
            path.append(neighbors[path[-1]])

        assert path[-1] == target_name, "Path does not end at target"

        return path

    def plot_sets(self, **kwargs):
        # Set gridlines to be drawn below the data
        plt.rc("axes", axisbelow=True)
        plt.gca().set_aspect("equal")
        for v in self.vertices.values():
            v.convex_set.plot(**kwargs)

    def plot_edge(self, edge_key, **kwargs):
        options = {
            "color": "k",
            "zorder": 2,
            "arrowstyle": "->, head_width=3, head_length=8",
        }
        options.update(kwargs)
        edge = self.edges[edge_key]
        tail = self.vertices[edge.u].convex_set.center
        head = self.vertices[edge.v].convex_set.center
        arrow = patches.FancyArrowPatch(tail, head, **options)
        plt.gca().add_patch(arrow)

    def plot_edges(self, **kwargs):
        for edge in self.edge_keys:
            self.plot_edge(edge, **kwargs)

    def plot_set_labels(self, labels=None, **kwargs):
        options = {
            "color": "black",
            "zorder": 5,
            "size": 8,
        }
        options.update(kwargs)
        if labels is None:
            labels = self.vertex_names
        offset = np.array([0.2, 0.1])
        patch_offset = np.array([-0.05, -0.1])  # from text
        for v, label in zip(self.vertices.values(), labels):
            if v.convex_set.dim == 1:
                # Add extra dimension for 1D sets
                offset = np.array([0, -0.2])
                patch_offset = np.array([-0.05, -0.5])  # from text
                pos = np.array([v.convex_set.center[0], 0]) + offset
            else:
                pos = v.convex_set.center + offset
            # Create a colored rectangle as the background
            rect = patches.FancyBboxPatch(
                pos + patch_offset,
                0.8,
                0.4,
                color="white",
                alpha=0.8,
                zorder=4,
                boxstyle="round,pad=0.1",
            )
            plt.gca().add_patch(rect)
            plt.text(*pos, label, **options)

    def plot_edge_labels(self, labels, **kwargs):
        options = {"c": "r", "va": "top"}
        options.update(kwargs)
        for edge, label in zip(self.edges, labels):
            center = (
                self.vertices[edge[0]].convex_set.center
                + self.vertices[edge[1]].convex_set.center
            ) / 2
            plt.text(*center, label, **options)

    def plot_points(self, x, **kwargs):
        options = {"marker": "o", "facecolor": "w", "edgecolor": "k", "zorder": 3}
        options.update(kwargs)
        plt.scatter(*x.T, **options)

    def plot_path(self, path: List[np.ndarray], **kwargs):
        options = {
            "color": "g",
            "marker": "o",
            "markeredgecolor": "k",
            "markerfacecolor": "w",
        }
        options.update(kwargs)
        plt.plot(*np.array([x for x in path]).T, **options)

    def edge_key_index(self, edge_key):
        return self.edge_keys.index(edge_key)

    def edge_indices(self, edge_keys):
        return [self.edge_keys.index(key) for key in edge_keys]

    def vertex_name_index(self, vertex_name):
        return self.vertex_names.index(vertex_name)

    def vertex_name_indices(self, vertex_names):
        return [self.vertex_names.index(name) for name in vertex_names]

    def generate_successors(self, vertex_name: str) -> None:
        pass

    @property
    def vertex_names(self):
        return list(self.vertices.keys())

    @property
    def edge_keys(self):
        return list(self.edges.keys())

    @property
    def source_name(self):
        return self._source_name

    @property
    def target_name(self):
        return self._target_name

    @property
    def source(self):
        return self.vertices[self.source_name]

    @property
    def target(self):
        return self.vertices[self.target_name]

    @property
    def dim_bounds(self):
        # Return a tuple of the smallest and largest dimension of the vertices
        smallest_dim = min([v.convex_set.dim for v in self.vertices.values()])
        largest_dim = max([v.convex_set.dim for v in self.vertices.values()])
        return (smallest_dim, largest_dim)

    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def params(self):
        return GraphParams(
            dim_bounds=self.dim_bounds,
            n_vertices=self.n_vertices,
            n_edges=self.n_edges,
            source=self.source.convex_set.center if self.source_name else None,
            target=self.target.convex_set.center if self.target_name else None,
            default_costs_constraints=self._default_costs_constraints,
            workspace=self.workspace,
        )
