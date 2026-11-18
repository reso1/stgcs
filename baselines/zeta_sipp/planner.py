from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, box
from shapely.ops import unary_union
from shapely.prepared import prep

from baselines.common import ShortestPathSolution
from experiments.environment.env import Env
from experiments.environment.obstacle import (
    ConcatDynamicSphere, DynamicSphere, StaticPolygon, StaticSphere,
)
from experiments.environment.interval import Interval
from stgcs_native.types import MPQuery


TIME_EPS = 1e-9


class _ZetaUtils:
    @staticmethod
    def clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def merge_intervals(intervals: Iterable[Tuple[float, float]], t_max: float) -> List[Tuple[float, float]]:
        ordered: List[Tuple[float, float]] = []
        for start, end in intervals:
            lo = max(0.0, float(start))
            hi = min(float(end), float(t_max))
            if hi < lo:
                continue
            ordered.append((lo, hi))
        ordered.sort(key=lambda item: (item[0], item[1]))

        merged: List[Tuple[float, float]] = []
        for start, end in ordered:
            if not merged or start > merged[-1][1] + TIME_EPS:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    @staticmethod
    def flatten_intervals(intervals: Iterable[Tuple[float, float]]) -> List[float]:
        flat: List[float] = []
        for start, end in intervals:
            flat.extend([float(start), float(end)])
        return flat

    @staticmethod
    def complement_intervals(blocked: Sequence[Tuple[float, float]], t_max: float) -> List[Tuple[float, float]]:
        safe: List[Tuple[float, float]] = []
        cursor = 0.0
        # Collision intervals are closed, including instantaneous contacts.
        margin = 10 * TIME_EPS
        for start, end in blocked:
            if start - margin >= cursor:
                safe.append((cursor, start - margin))
            cursor = max(cursor, end + margin)
            if cursor > t_max:
                return safe
        if cursor <= t_max:
            safe.append((cursor, t_max))
        return safe

    @staticmethod
    def intersect_intervals(a: Sequence[float], b: Sequence[float]) -> List[float]:
        lo = max(float(a[0]), float(b[0]))
        hi = min(float(a[1]), float(b[1]))
        if hi < lo - TIME_EPS:
            return []
        return [lo, hi]

    @staticmethod
    def line_seg_on_grid(x0: float, y0: float, x1: float, y1: float, distance: float) -> List[dict]:
        # Trace positive-length cell crossings. A corner touch does not enter
        # either side cell; the adjoining free cells already cover that point.
        grids: List[dict] = []
        dx, dy = x1 - x0, y1 - y0
        x, y = math.floor(x0), math.floor(y0)
        x_inc = 1 if dx > 0 else -1
        y_inc = 1 if dy > 0 else -1
        dtdx = 1.0 / abs(dx) if dx else math.inf
        dtdy = 1.0 / abs(dy) if dy else math.inf
        next_x = ((x + 1 - x0) if dx > 0 else (x0 - x)) * dtdx if dx else math.inf
        next_y = ((y + 1 - y0) if dy > 0 else (y0 - y)) * dtdy if dy else math.inf
        begin = 0.0
        while True:
            end = min(next_x, next_y, 1.0)
            if end > begin or distance == 0:
                grids.append({"x": x, "y": y, "dist": [distance * begin, distance * end]})
            if end >= 1.0:
                break
            cross_x, cross_y = next_x <= next_y, next_y <= next_x
            if cross_x:
                x += x_inc
                next_x += dtdx
            if cross_y:
                y += y_inc
                next_y += dtdy
            begin = end
        return grids


class _KeyedHeap:
    """Indexed priority queue with exactly one live entry per item."""

    def __init__(self, key_fn: Callable[[object], float]) -> None:
        self._key_fn = key_fn
        self._heap: List[Tuple[float, int, object]] = []
        self._positions: dict[int, int] = {}
        self._counter = 0

    def push(self, item: object) -> None:
        entry = (float(self._key_fn(item)), self._counter, item)
        self._counter += 1
        index = self._positions.get(id(item))
        if index is None:
            self._heap.append(entry)
            self._sift_up(len(self._heap) - 1)
        else:
            self._heap[index] = entry
            self._sift_down(self._sift_up(index))

    def sort_element(self, item: object) -> None:
        self.push(item)

    def _sift_up(self, index: int) -> int:
        entry = self._heap[index]
        while index:
            parent = (index - 1) // 2
            if entry >= self._heap[parent]:
                break
            self._heap[index] = self._heap[parent]
            self._positions[id(self._heap[index][2])] = index
            index = parent
        self._heap[index] = entry
        self._positions[id(entry[2])] = index
        return index

    def _sift_down(self, index: int) -> None:
        entry = self._heap[index]
        size = len(self._heap)
        while 2 * index + 1 < size:
            child = 2 * index + 1
            if child + 1 < size and self._heap[child + 1] < self._heap[child]:
                child += 1
            if entry <= self._heap[child]:
                break
            self._heap[index] = self._heap[child]
            self._positions[id(self._heap[index][2])] = index
            index = child
        self._heap[index] = entry
        self._positions[id(entry[2])] = index

    def peek_key(self) -> float:
        if not self._heap:
            return math.inf
        return float(self._heap[0][0])

    def shift(self) -> object:
        _, _, item = self._heap[0]
        last = self._heap.pop()
        del self._positions[id(item)]
        if self._heap:
            self._heap[0] = last
            self._sift_down(0)
        return item

    def size(self) -> int:
        return len(self._heap)


@dataclass(eq=False)
class _Node:
    grid_point: np.ndarray
    world_point: np.ndarray
    safe_interval: Tuple[float, float]
    cell: "_Cell"
    f: float = math.inf
    g: float = math.inf
    g_low: float = math.inf
    h: float = 0.0
    parent: Optional["_Node"] = None
    closed: bool = False
    best_potential_parent: Optional["_Node"] = None
    potential_parents: List["_Node"] = field(default_factory=list)
    g_low_array: List[float] = field(default_factory=list)
    wait_time: float = 0.0


@dataclass(eq=False)
class _Cell:
    grid_point: np.ndarray
    world_point: np.ndarray
    weight: bool
    risk_interval: List[float]
    base_index: Optional[Tuple[int, int]]
    is_query_cell: bool = False
    h: float = 0.0
    fh: float = math.inf
    visited: bool = False
    closed: bool = False
    nodes: List[_Node] = field(default_factory=list)
    visible_cells: List["_Cell"] = field(default_factory=list)


class _ZetaGeometry:
    """Cached exact 2D geometry; robot_radius is the square's half extent."""

    def __init__(self, env: Env):
        self.env = env
        self.use_cspace = not env.O_Static or str(env.name).lower().startswith("maze")
        self.space = (unary_union([MultiPoint(vertices).convex_hull for vertices in env.C_Space])
                      if self.use_cspace else box(*env.lb, *env.ub))
        self.prepared_space = prep(self.space)
        self.polygons = []
        self.spheres = []
        for obstacle in env.O_Static:
            if isinstance(obstacle, StaticPolygon):
                inflated = self.square_sweep(obstacle.vertices, env.robot_radius)
                self.polygons.append(prep(inflated))
            elif isinstance(obstacle, StaticSphere):
                self.spheres.append((Point(obstacle.pos), obstacle.radius))
            else:
                raise TypeError(f"Unsupported static obstacle {type(obstacle)!r}")

    @staticmethod
    def square_sweep(vertices, half_extent):
        corners = half_extent * np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        return MultiPoint((np.asarray(vertices)[:, None, :] + corners).reshape(-1, 2)).convex_hull

    def is_free(self, shape) -> bool:
        if not self.prepared_space.covers(shape):
            return False
        # Scaled Maze corners can differ by ~1e-16 after Minkowski addition.
        # Treat numerical contact consistently with the independent collision solver.
        if any(obstacle.context.distance(shape) <= TIME_EPS for obstacle in self.polygons):
            return False
        if self.spheres:
            vertices = shape.exterior.coords if shape.geom_type == "Polygon" else shape.coords
            swept = self.square_sweep(vertices, self.env.robot_radius)
            if any(swept.distance(center) <= radius for center, radius in self.spheres):
                return False
        return True

    def point_is_free(self, point) -> bool:
        return self.is_free(Point(point))

    def aligned_maze_grid(self, origin: np.ndarray, cell_size: float) -> bool:
        """Only axis-aligned maze clearance boundaries on grid lines qualify."""
        if not str(self.env.name).lower().startswith("maze") or self.spheres:
            return False
        boundaries = [np.asarray(vertices) for vertices in self.env.C_Space]
        boundaries.extend(np.asarray(p.context.exterior.coords) for p in self.polygons)
        boundaries.append(np.asarray(box(*self.env.lb, *self.env.ub).exterior.coords))
        for vertices in boundaries:
            edges = np.roll(vertices, -1, axis=0) - vertices
            if not np.all(np.any(np.abs(edges) <= TIME_EPS, axis=1)):
                return False
            aligned = origin + np.rint((vertices - origin) / cell_size) * cell_size
            if not np.allclose(vertices, aligned, atol=10 * TIME_EPS, rtol=0):
                return False
        return True

    def cell_is_free(self, center, cell_size, wall_aligned=False) -> bool:
        if wall_aligned:
            # With every clearance boundary on a grid line, the center exactly
            # classifies the cell interior. Search rejects blocked edge/corner
            # contacts explicitly, instead of discarding the adjacent free cell.
            return self.is_free(Point(center))
        # Certify the entire cell once, including the robot's square footprint.
        lo, hi = center - cell_size / 2, center + cell_size / 2
        return self.is_free(box(*lo, *hi))

    def segment_is_free(self, start, goal) -> bool:
        shape = Point(start) if np.array_equal(start, goal) else LineString([start, goal])
        return self.is_free(shape)


class _DynamicIntervals:
    """Analytic square occupancy intervals for piecewise-linear moving obstacles."""

    def __init__(self, env: Env, t_max: float):
        self.t_max = t_max
        phases = []
        for obstacle in env.O_Dynamic:
            if isinstance(obstacle, ConcatDynamicSphere):
                pieces = obstacle.segments
            elif isinstance(obstacle, DynamicSphere):
                pieces = [obstacle]
            else:
                raise TypeError(f"Unsupported dynamic obstacle {type(obstacle)!r}")
            raw = [(p.x0, p.xt, p.itvl.start, p.itvl.end) for p in pieces]
            if not isinstance(obstacle, ConcatDynamicSphere) or obstacle.reserve_start:
                raw.insert(0, (pieces[0].x0, pieces[0].x0, 0., pieces[0].itvl.start))
            if not isinstance(obstacle, ConcatDynamicSphere) or obstacle.reserve_goal:
                raw.append((pieces[-1].xt, pieces[-1].xt, pieces[-1].itvl.end, t_max))
            for x0, x1, begin, end in raw:
                lo, hi = max(0., begin), min(t_max, end)
                if hi < lo:
                    continue
                velocity = (x1 - x0) / (end - begin) if end > begin else np.zeros(env.dim)
                phases.append((x0 + velocity * (lo - begin), velocity, lo, hi, obstacle.radius))
        self.origins = np.array([p[0] for p in phases]).reshape(-1, env.dim)
        self.velocities = np.array([p[1] for p in phases]).reshape(-1, env.dim)
        self.begins = np.array([p[2] for p in phases])
        self.ends = np.array([p[3] for p in phases])
        self.radii = np.array([p[4] for p in phases])

    def blocked(self, point, half_extent) -> List[Tuple[float, float]]:
        if not len(self.begins):
            return []
        # Match the continuous box checker, including boundary contacts.
        radius = self.radii + half_extent + 10 * TIME_EPS
        delta = self.origins - point
        lo, hi = np.zeros(len(self.begins)), self.ends - self.begins
        for axis in range(delta.shape[1]):
            velocity = self.velocities[:, axis]
            moving = velocity != 0
            divisor = np.where(moving, velocity, 1.)
            a = (-radius - delta[:, axis]) / divisor
            b = (radius - delta[:, axis]) / divisor
            lo = np.maximum(lo, np.where(moving, np.minimum(a, b), -np.inf))
            hi = np.minimum(hi, np.where(moving, np.maximum(a, b), np.inf))
            hi = np.where(~moving & (np.abs(delta[:, axis]) > radius), -np.inf, hi)
        keep = lo <= hi
        return _ZetaUtils.merge_intervals(zip(self.begins[keep] + lo[keep],
                                              self.begins[keep] + hi[keep]), self.t_max)


class _ProjectedGrid:
    def __init__(self, env: Env, t_max: float, cell_size: float, deadline: float = math.inf) -> None:
        started = time.perf_counter()
        if env.dim != 2:
            raise ValueError(f"Zeta*-SIPP projection currently supports only 2D environments, got dim={env.dim}")
        if cell_size <= 0.0:
            raise ValueError(f"cell_size must be positive, got {cell_size}")

        self.env = env
        self.t_max = float(t_max)
        self.deadline = deadline
        self.check_deadline()
        self.geometry = _ZetaGeometry(env)
        self.dynamic = _DynamicIntervals(env, t_max)
        self.cell_size = float(cell_size)
        self.origin = np.asarray(env.lb[:2], dtype=float)
        span = np.asarray(env.ub[:2], dtype=float) - self.origin
        self.wall_aligned = self.geometry.aligned_maze_grid(self.origin, self.cell_size)
        self.shape = tuple(int(np.rint(axis / self.cell_size) if self.wall_aligned
                               else np.ceil(axis / self.cell_size)) for axis in span)
        self.grids: List[List[_Cell]] = []
        self.cells: List[_Cell] = []
        self.checks = 0
        self.scanned_grids = 0
        self._build_cells()
        self.build_runtime = time.perf_counter() - started

    def check_deadline(self) -> None:
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("Zeta-SIPP planning deadline")

    def _build_cells(self) -> None:
        for ix in range(self.shape[0]):
            row: List[_Cell] = []
            for iy in range(self.shape[1]):
                center_grid = np.array([ix + 0.5, iy + 0.5], dtype=float)
                center_world = self.grid_to_world(center_grid)
                self.check_deadline()
                weight = self.geometry.cell_is_free(center_world, self.cell_size, self.wall_aligned)
                # Every domain blocks the entire cell for the moving square footprint.
                blocked = self.dynamic.blocked(
                    center_world, self.env.robot_radius + self.cell_size / 2)
                cell = _Cell(
                    grid_point=center_grid,
                    world_point=center_world,
                    weight=weight,
                    risk_interval=_ZetaUtils.flatten_intervals(blocked),
                    base_index=(ix, iy),
                )
                if cell.weight:
                    for safe_interval in _ZetaUtils.complement_intervals(blocked, self.t_max):
                        cell.nodes.append(_Node(center_grid.copy(), center_world.copy(), safe_interval, cell))
                row.append(cell)
                self.cells.append(cell)
            self.grids.append(row)

    def world_to_grid(self, point: np.ndarray) -> np.ndarray:
        grid = (np.asarray(point, dtype=float) - self.origin) / self.cell_size
        upper = np.asarray(self.shape, dtype=float) - 1e-9
        return np.clip(grid, 0.0, upper)

    def grid_to_world(self, point: np.ndarray) -> np.ndarray:
        return self.origin + self.cell_size * np.asarray(point, dtype=float)

    def safe_intervals_for_point(self, world_point: np.ndarray) -> List[Tuple[float, float]]:
        """Use the containing cell's intervals without snapping the query point."""
        cell = self.cell_for_point(world_point)
        if cell is None or not cell.weight:
            return []
        if self.wall_aligned:
            point = self.world_to_grid(world_point)
            if any(not self.get_cell(g["x"], g["y"]).weight
                   for g in self.trace_segment(point, point, 0.0)):
                return []
        return [node.safe_interval for node in cell.nodes]

    def cell_for_point(self, world_point: np.ndarray) -> Optional[_Cell]:
        point = np.asarray(world_point, dtype=float)
        if not np.all(np.isfinite(point)) or np.any(point < self.env.lb) or np.any(point > self.env.ub):
            return None
        grid = self.world_to_grid(point)
        return self.get_cell(math.floor(grid[0]), math.floor(grid[1]))

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.shape[0] and 0 <= iy < self.shape[1]

    def get_cell(self, ix: int, iy: int) -> Optional[_Cell]:
        if not self.in_bounds(ix, iy):
            return None
        return self.grids[ix][iy]

    def trace_segment(self, start: np.ndarray, goal: np.ndarray, distance: float) -> List[dict]:
        grids = _ZetaUtils.line_seg_on_grid(*start, *goal, distance)
        if not self.wall_aligned:
            return grids
        # Aligned maze cells certify their interiors. Include all cells touched
        # along edges or at corners so a path cannot graze an inflated wall.
        tolerance = 10 * TIME_EPS / self.cell_size
        traversed = {(g["x"], g["y"]) for g in grids}
        extra = []
        delta = goal - start
        for axis in (0, 1):
            if delta[axis] == 0 and abs(start[axis] - round(start[axis])) <= tolerance:
                for grid in grids:
                    other = dict(grid)
                    key = "x" if axis == 0 else "y"
                    boundary = round(start[axis])
                    other[key] = boundary - 1 if grid[key] == boundary else boundary
                    extra.append(other)
                    traversed.add((other["x"], other["y"]))
        cuts = {0.0, distance}
        cuts.update(d for grid in grids for d in grid["dist"])
        for d in cuts:
            point = start + (d / distance if distance else 0.0) * delta
            indices = [([round(v) - 1, round(v)] if abs(v - round(v)) <= tolerance
                        else [math.floor(v)]) for v in point]
            for ix in indices[0]:
                for iy in indices[1]:
                    if (ix, iy) not in traversed:
                        extra.append({"x": ix, "y": iy, "dist": [d, d]})
                        traversed.add((ix, iy))
        grids.extend(extra)
        grids = [g for g in grids if self.in_bounds(g["x"], g["y"])]
        grids.sort(key=lambda g: (g["dist"][0], g["dist"][1]))
        return grids

    def line_of_sight_grid(self, a: _Cell | _Node, b: _Cell | _Node) -> bool:
        self.check_deadline()
        self.checks += 1
        trace = self.trace_segment(a.grid_point, b.grid_point, 1.0)
        for grid in trace:
            self.scanned_grids += 1
            cell = self.get_cell(grid["x"], grid["y"])
            if cell is None or not cell.weight:
                return False
        return True


class _Shadowcast:
    def __init__(self, projection: _ProjectedGrid) -> None:
        self.projection = projection
        self.inverted = False
        self.max_cost = math.inf
        self.visible: List[_Cell] = []
        self.transforms = [
            {"xx": 1, "xy": 0, "yx": 0, "yy": 1},
            {"xx": 1, "xy": 0, "yx": 0, "yy": -1},
            {"xx": -1, "xy": 0, "yx": 0, "yy": 1},
            {"xx": -1, "xy": 0, "yx": 0, "yy": -1},
            {"xx": 0, "xy": 1, "yx": 1, "yy": 0},
            {"xx": 0, "xy": -1, "yx": 1, "yy": 0},
            {"xx": 0, "xy": 1, "yx": -1, "yy": 0},
            {"xx": 0, "xy": -1, "yx": -1, "yy": 0},
        ]

    def find(self, x: float, y: float) -> Optional[_Cell]:
        return self.projection.get_cell(math.floor(x), math.floor(y))

    def record(self, node: _Cell) -> None:
        self.visible.append(node)

    def in_range(self, node: _Cell) -> bool:
        if not self.inverted:
            return True
        return node.fh <= self.max_cost + TIME_EPS

    def blocked(self, node: Optional[_Cell]) -> bool:
        return node is None or (not node.weight) or (not self.in_range(node))

    def scan(self, origin: _Cell) -> List[_Cell]:
        self.visible = []
        if origin.is_query_cell:
            return []
        for transform in self.transforms:
            self.compute(origin, 1, 1.0, 0.0, transform)
        dedup: List[_Cell] = []
        seen: set[int] = set()
        for cell in self.visible:
            key = id(cell)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(cell)
        return dedup

    def compute(self, origin: _Cell, delta_x: int, top_slope: float, bottom_slope: float, transform: dict) -> None:
        self.projection.check_deadline()
        if bottom_slope >= top_slope:
            return

        ymin = delta_x * bottom_slope
        ymax = delta_x * top_slope
        was_blocked = False
        y = math.floor(ymin)
        while y <= math.ceil(ymax):
            real_x = origin.grid_point[0] + transform["xx"] * delta_x + transform["xy"] * y
            real_y = origin.grid_point[1] + transform["yx"] * delta_x + transform["yy"] * y
            node = self.find(real_x, real_y)
            if not self.blocked(node):
                if ymin - TIME_EPS <= y <= ymax + TIME_EPS:
                    self.record(node)
                was_blocked = False
            else:
                if not was_blocked:
                    new_top_slope = (y - 0.5) / (delta_x + 0.5)
                    self.compute(origin, delta_x + 1, min(new_top_slope, top_slope), bottom_slope, transform)
                    was_blocked = True
                new_bottom_slope = (y + 0.5) / max(delta_x - 0.5, TIME_EPS)
                bottom_slope = max(bottom_slope, new_bottom_slope)
                if bottom_slope >= top_slope:
                    return
                ymin = delta_x * bottom_slope
            y += 1
        self.compute(origin, delta_x + 1, top_slope, bottom_slope, transform)


class _TOAASIPP:
    def __init__(
        self,
        projection: _ProjectedGrid,
        query: MPQuery,
        t_max: float,
        runtime_limit_secs: float,
        time_buffer: float,
    ) -> None:
        self.projection = projection
        self.query = query
        self.t_max = float(t_max)
        self.runtime_limit_secs = float(runtime_limit_secs)
        projection.deadline = min(projection.deadline, time.perf_counter() + self.runtime_limit_secs)
        self.time_buffer = float(time_buffer)
        self.open_heap = _KeyedHeap(lambda node: node.f)
        self.bound_heap = _KeyedHeap(lambda cell: cell.fh)
        self.open_cells: List[_Cell] = []
        self.tree_cells: List[_Cell] = []
        self.steps = 0

    @staticmethod
    def _travel_time_world(a: np.ndarray, b: np.ndarray, vlimit: float) -> float:
        return float(np.max(np.abs(np.asarray(b, dtype=float) - np.asarray(a, dtype=float))) / vlimit)

    def heuristic(self, a: _Cell | _Node, b: _Cell | _Node) -> float:
        return self._travel_time_world(a.world_point, b.world_point, self.query.vlimit)

    def visit_cell(self, cell: _Cell, end_cell: _Cell) -> None:
        if cell.visited:
            return
        if not cell.nodes:
            return
        if cell.h == 0.0 and cell is not end_cell:
            cell.h = self.heuristic(cell, end_cell)
        else:
            cell.h = self.heuristic(cell, end_cell)
        cell.visited = True
        self.tree_cells.append(cell)
        self.open_cells.append(cell)
        for node in cell.nodes:
            node.h = cell.h
            self.open_heap.push(node)

    def init_start(self, start_cell: _Cell, end_cell: _Cell) -> None:
        self.visit_cell(start_cell, end_cell)
        start_node = start_cell.nodes[0]
        start_node.g = float(self.query.t_start)
        start_node.g_low = float(self.query.t_start)
        start_node.f = start_node.g + start_node.h
        self.open_heap.sort_element(start_node)

    def init(self) -> Tuple[_Cell, _Cell]:
        start_grid = self.projection.world_to_grid(self.query.start)
        goal_grid = self.projection.world_to_grid(self.query.goal)

        start_intervals = [
            interval
            for interval in self.projection.safe_intervals_for_point(self.query.start)
            if interval[0] - TIME_EPS <= self.query.t_start <= interval[1] + TIME_EPS
        ]
        if not start_intervals:
            raise ValueError("Start query is not safe at t_start in projected Zeta*-SIPP.")

        goal_intervals = self.projection.safe_intervals_for_point(self.query.goal)
        if self.query.is_stay:
            goal_intervals = [interval for interval in goal_intervals if interval[1] >= self.t_max - TIME_EPS]
        if not goal_intervals:
            raise ValueError("Goal query is not feasible in projected Zeta*-SIPP under is_stay/t_max constraints.")

        start_base = self.projection.cell_for_point(self.query.start)
        goal_base = self.projection.cell_for_point(self.query.goal)

        start_cell = _Cell(
            grid_point=start_grid,
            world_point=np.asarray(self.query.start, dtype=float),
            weight=True,
            risk_interval=start_base.risk_interval,
            base_index=start_base.base_index,
            is_query_cell=True,
        )
        start_cell.nodes = [_Node(start_grid.copy(), np.asarray(self.query.start, dtype=float), start_intervals[0], start_cell)]

        goal_cell = _Cell(
            grid_point=goal_grid,
            world_point=np.asarray(self.query.goal, dtype=float),
            weight=True,
            risk_interval=goal_base.risk_interval,
            base_index=goal_base.base_index,
            is_query_cell=True,
        )
        goal_cell.nodes = [
            _Node(goal_grid.copy(), np.asarray(self.query.goal, dtype=float), interval, goal_cell)
            for interval in goal_intervals
        ]

        self.init_start(start_cell, goal_cell)
        if not np.allclose(self.query.start, self.query.goal):
            self.visit_cell(goal_cell, goal_cell)
        return start_cell, goal_cell

    def search(self) -> List[_Node]:
        self.projection.check_deadline()
        start_cell, end_cell = self.init()
        while self.open_heap.size() > 0 and self.open_heap.peek_key() < math.inf:
            self.projection.check_deadline()
            self.steps += 1
            current_node = self.find_next_closed_node()
            if current_node is not None:
                current_cell = current_node.cell
                if current_cell is end_cell:
                    return self.path_to(current_node)
                self.inverted_expansion(current_node, current_cell, end_cell)
            self.forward_expansion(start_cell, end_cell)
        return []

    def find_next_closed_node(self) -> Optional[_Node]:
        current_node = self.open_heap.shift()

        if current_node.best_potential_parent is not None:
            for idx, parent in enumerate(list(current_node.potential_parents)):
                if parent is current_node.best_potential_parent:
                    current_node.potential_parents.pop(idx)
                    current_node.g_low_array.pop(idx)
                    break

        if current_node.best_potential_parent is None:
            g_new = current_node.g
        else:
            g_new = self.transition(current_node.best_potential_parent, current_node) + current_node.best_potential_parent.g
        if g_new < current_node.g - TIME_EPS:
            current_node.g = g_new
            current_node.parent = current_node.best_potential_parent
            current_node.wait_time = max(0.0, g_new - current_node.g_low)

        current_node.g_low = current_node.g
        current_node.best_potential_parent = current_node.parent
        current_node.f = current_node.g_low + current_node.h

        if self.new_best_potential_parent_exists(current_node) or current_node.g >= math.inf:
            self.open_heap.push(current_node)
            return None

        best_open = self.open_heap.peek_key()
        best_bound = self.bound_heap.peek_key()
        if current_node.g + current_node.h <= best_open + TIME_EPS and current_node.g + current_node.h <= best_bound + TIME_EPS:
            current_node.closed = True
            self.update_close_cell(current_node.cell)
            return current_node

        self.open_heap.push(current_node)
        return None

    def update_close_cell(self, current_cell: _Cell) -> None:
        return

    def update_open_cell(self, current_cell: _Cell) -> None:
        if current_cell.nodes and all(node.closed for node in current_cell.nodes):
            if current_cell in self.open_cells:
                self.open_cells.remove(current_cell)

    def _visible_open_cells(self, current_cell: _Cell) -> List[_Cell]:
        visible: List[_Cell] = []
        for cell in self.open_cells:
            if cell is current_cell:
                continue
            if self.projection.line_of_sight_grid(current_cell, cell):
                visible.append(cell)
        return visible

    def inverted_expansion(self, current_node: _Node, current_cell: _Cell, end_cell: _Cell) -> None:
        self.update_open_cell(current_cell)
        if not current_cell.visible_cells:
            current_cell.visible_cells = self._visible_open_cells(current_cell)
        for cell in current_cell.visible_cells:
            if not cell.visited:
                self.visit_cell(cell, end_cell)
            for node in cell.nodes:
                if node.closed:
                    continue
                self.add_potential_parent(current_node, node)

    def forward_expansion(self, start_cell: _Cell, end_cell: _Cell) -> None:
        return

    def add_potential_parent(self, parent: _Node, node: _Node) -> None:
        g_low_new = self.potential_parent_lower_bound(parent, node)
        if not math.isfinite(g_low_new):
            return
        if g_low_new < node.g_low - TIME_EPS:
            node.g_low = g_low_new
            node.best_potential_parent = parent
            node.f = node.g_low + node.h
            self.open_heap.sort_element(node)
        node.potential_parents.append(parent)
        node.g_low_array.append(g_low_new)

    def endpoint_wait_bounds(self, parent: _Node, node: _Node, travel_time: float,
                             time_buffer: float = 0.0) -> Tuple[float, float]:
        # Fractions before leaving the source cell / after entering the target
        # cell. Unlike a half-cell formula, this also handles arbitrary query
        # points, boundary endpoints, and transitions within one cell.
        first_fraction = last_fraction = 1.0
        for start, end in zip(parent.grid_point, node.grid_point):
            delta = float(end - start)
            if delta > 0:
                first_fraction = min(first_fraction, (math.floor(start) + 1 - start) / delta)
                last_fraction = min(last_fraction, (end - math.floor(end)) / delta)
            elif delta < 0:
                first_fraction = min(first_fraction, (math.floor(start) - start) / delta)
                last_fraction = min(last_fraction, (math.floor(end) + 1 - end) / -delta)
        first_time, last_time = travel_time * first_fraction, travel_time * last_fraction
        wait_time = max(0.0, node.safe_interval[0] - parent.g - travel_time + last_time)
        if wait_time > TIME_EPS:
            wait_time += time_buffer
        max_wait_time = min(parent.safe_interval[1] - parent.g - first_time,
                            node.safe_interval[1] - parent.g - travel_time,
                            self.t_max - parent.g - travel_time)
        if wait_time > max_wait_time + TIME_EPS:
            return math.inf, max_wait_time
        return wait_time, max_wait_time

    def potential_parent_lower_bound(self, parent: _Node, node: _Node) -> float:
        travel_time = self.heuristic(parent, node)
        wait_time, _ = self.endpoint_wait_bounds(parent, node, travel_time)
        arrival = parent.g + wait_time + travel_time
        # Only a verified feasible arrival (g), not another candidate's lower
        # bound (g_low), can rule out a potential improvement.
        if arrival >= node.g - TIME_EPS:
            return math.inf
        return arrival

    def new_best_potential_parent_exists(self, node: _Node) -> bool:
        node.g_low = node.g
        node.best_potential_parent = node.parent
        node.f = node.g_low + node.h
        exists = False
        for parent, g_low_new in zip(node.potential_parents, node.g_low_array):
            if g_low_new < node.g_low - TIME_EPS:
                node.g_low = g_low_new
                node.best_potential_parent = parent
                node.f = node.g_low + node.h
                exists = True
        return exists

    @staticmethod
    def path_to(node: _Node) -> List[_Node]:
        current = node
        path: List[_Node] = []
        while current is not None:
            path.insert(0, current)
            current = current.parent
        return path

    def transition(self, parent: _Node, node: _Node) -> float:
        if parent is node:
            return 0.0
        # Static occupancy is checked during the temporal grid traversal below.
        return self.get_cost_sipp(parent, node)

    def get_cost_sipp(self, parent: _Node, node: _Node) -> float:
        travel_time = self._travel_time_world(parent.world_point, node.world_point, self.query.vlimit)
        if travel_time <= TIME_EPS:
            arrival = max(parent.g, node.safe_interval[0])
            if arrival <= min(parent.safe_interval[1], node.safe_interval[1]) + TIME_EPS:
                return arrival - parent.g
            return math.inf

        wait_time, max_wait_time = self.endpoint_wait_bounds(parent, node, travel_time, self.time_buffer)
        if not math.isfinite(wait_time):
            return math.inf

        grid_distance = float(np.linalg.norm(node.grid_point - parent.grid_point))
        trace_grids = self.projection.trace_segment(parent.grid_point, node.grid_point, grid_distance)

        intervals: List[List[float]] = []
        for grid in trace_grids:
            if grid_distance <= TIME_EPS:
                lam0 = lam1 = 0.0
            else:
                lam0 = _ZetaUtils.clamp(grid["dist"][0] / grid_distance, 0.0, 1.0)
                lam1 = _ZetaUtils.clamp(grid["dist"][1] / grid_distance, 0.0, 1.0)
            intervals.append([lam0 * travel_time, lam1 * travel_time])

        wait_time = self.update_wait_time(
            trace_grids, intervals, wait_time, parent, max_wait_time)

        arrival = parent.g + wait_time + travel_time
        if arrival < node.safe_interval[0] - TIME_EPS or arrival > node.safe_interval[1] + TIME_EPS:
            return math.inf
        if arrival > self.t_max + TIME_EPS:
            return math.inf
        return wait_time + travel_time

    def update_wait_time(
        self,
        trace_grids: Sequence[dict],
        intervals: Sequence[Sequence[float]],
        wait_time: float,
        parent: _Node,
        max_wait_time: float,
    ) -> float:
        # Keep traversal times relative; delays change only a scalar offset.
        # A changed offset rechecks the affected prefix, including the cell
        # that caused it. Iterators implement upstream's recursive rechecking
        # without growing the Python call stack.
        pending = [iter(range(len(trace_grids)))]
        while pending:
            index = next(pending[-1], None)
            if index is None:
                pending.pop()
                continue
            self.projection.check_deadline()
            grid = trace_grids[index]
            cell = self.projection.get_cell(int(grid["x"]), int(grid["y"]))
            if cell is None or not cell.weight:
                return math.inf
            old_wait_time = wait_time
            relative_in, relative_out = intervals[index]
            for offset in range(0, len(cell.risk_interval), 2):
                time_in = parent.g + wait_time + relative_in
                time_out = parent.g + wait_time + relative_out
                risk_start, risk_end = cell.risk_interval[offset:offset + 2]
                if risk_start > time_out + TIME_EPS:
                    break  # Later risk intervals are also beyond this traversal.
                if time_in > risk_end + TIME_EPS:
                    continue
                wait_time = risk_end - parent.g - relative_in + self.time_buffer
                if wait_time > max_wait_time + TIME_EPS:
                    return math.inf
            if wait_time > old_wait_time:
                pending.append(iter(range(index + 1)))
        return wait_time


class _ZetaSIPP(_TOAASIPP):
    def __init__(
        self,
        projection: _ProjectedGrid,
        query: MPQuery,
        t_max: float,
        runtime_limit_secs: float,
        time_buffer: float,
    ) -> None:
        super().__init__(projection, query, t_max, runtime_limit_secs, time_buffer)
        self.close_cells: List[_Cell] = []

    def init(self) -> Tuple[_Cell, _Cell]:
        start_cell, end_cell = super().init()
        self.init_bound(start_cell, end_cell)
        return start_cell, end_cell

    def init_bound(self, start_cell: _Cell, end_cell: _Cell) -> None:
        start_cell.fh = float(self.query.t_start) + start_cell.h
        for cell in self.projection.cells:
            self.projection.check_deadline()
            if not cell.weight or cell is start_cell or not cell.nodes:
                continue
            self.bound_cell(cell, start_cell, end_cell)

    def bound_cell(self, cell: _Cell, start_cell: _Cell, end_cell: _Cell) -> None:
        cell.h = self.heuristic(cell, end_cell)
        cell.fh = float(self.query.t_start) + self.heuristic(start_cell, cell) + cell.h
        self.bound_heap.push(cell)

    def update_close_cell(self, current_cell: _Cell) -> None:
        if not current_cell.closed:
            current_cell.closed = True
            self.close_cells.append(current_cell)

    def forward_expansion(self, start_cell: _Cell, end_cell: _Cell) -> None:
        while self.bound_heap.size() > 0 and (
            self.open_heap.size() == 0 or self.bound_heap.peek_key() <= self.open_heap.peek_key() + TIME_EPS
        ):
            self.projection.check_deadline()
            new_cell = self.bound_heap.shift()
            self.visit_cell(new_cell, end_cell)
            self.inverted_scan(new_cell)

    def inverted_scan(self, new_cell: _Cell) -> None:
        for cell in self.close_cells:
            if self.projection.line_of_sight_grid(cell, new_cell):
                if new_cell not in cell.visible_cells:
                    cell.visible_cells.append(new_cell)
                for parent in cell.nodes:
                    if not parent.closed:
                        continue
                    for node in new_cell.nodes:
                        self.add_potential_parent(parent, node)


class _ZetaStarSIPP(_ZetaSIPP):
    def __init__(
        self,
        projection: _ProjectedGrid,
        query: MPQuery,
        t_max: float,
        runtime_limit_secs: float,
        time_buffer: float,
    ) -> None:
        super().__init__(projection, query, t_max, runtime_limit_secs, time_buffer)
        self.shadowcast = _Shadowcast(projection)
        self.shadowcast.inverted = True
        self.buffer = math.sqrt(2.0) * projection.cell_size / query.vlimit

    def inverted_scan(self, scan_cell: _Cell) -> None:
        self.shadowcast.max_cost = scan_cell.fh + self.buffer
        visible_cells = self.shadowcast.scan(scan_cell)
        self.projection.scanned_grids += len(visible_cells)
        scan_cell.visible_cells = [cell for cell in visible_cells if cell.visited]
        for query_cells in (self.close_cells, self.open_cells):
            for cell in query_cells:
                if not cell.is_query_cell:
                    continue
                if cell is scan_cell or not self.projection.line_of_sight_grid(cell, scan_cell):
                    continue
                if cell not in scan_cell.visible_cells:
                    scan_cell.visible_cells.append(cell)
        for cell in scan_cell.visible_cells:
            if scan_cell not in cell.visible_cells:
                cell.visible_cells.append(scan_cell)
            if not cell.closed:
                continue
            for parent in cell.nodes:
                if not parent.closed:
                    continue
                for node in scan_cell.nodes:
                    self.add_potential_parent(parent, node)

    def inverted_expansion(self, current_node: _Node, current_cell: _Cell, end_cell: _Cell) -> None:
        self.update_open_cell(current_cell)
        if not current_cell.visible_cells:
            if current_cell.is_query_cell:
                current_cell.visible_cells = self._visible_open_cells(current_cell)
            else:
                current_cell.visible_cells = self.shadowcast.scan(current_cell)
                self.projection.scanned_grids += len(current_cell.visible_cells)
                current_cell.visible_cells = [cell for cell in current_cell.visible_cells if cell.visited]
                for cell in self.open_cells:
                    if not cell.is_query_cell or cell is current_cell:
                        continue
                    if self.projection.line_of_sight_grid(current_cell, cell) and cell not in current_cell.visible_cells:
                        current_cell.visible_cells.append(cell)

        for cell in current_cell.visible_cells:
            for node in cell.nodes:
                if node.closed:
                    continue
                self.add_potential_parent(current_node, node)


class ZetaStarSIPPPlanner:
    def __init__(
        self,
        env: Env,
        seed: int,
        vlimit: float,
        cell_size: Optional[float] = None,
        runtime_limit_secs: float = math.inf,
        time_buffer: float = 1e-3,
        use_fov: bool = True,
    ) -> None:
        self.env = env
        self.seed = seed
        self.vlimit = float(vlimit)
        self.cell_size = float(cell_size) if cell_size is not None else max(env.robot_radius * 2.0, 0.1)
        self.runtime_limit_secs = float(runtime_limit_secs)
        self.time_buffer = float(time_buffer)
        self.use_fov = bool(use_fov)
        self.last_projection: Optional[_ProjectedGrid] = None
        self.last_path: List[_Node] = []
        self.last_status = "not_started"

    @staticmethod
    def componentwise_travel_time(start: np.ndarray, goal: np.ndarray, vlimit: float) -> float:
        return _TOAASIPP._travel_time_world(np.asarray(start, dtype=float), np.asarray(goal, dtype=float), vlimit)

    def solve(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        t_start: float,
        t_max: float,
        is_stay: bool = True,
    ) -> ShortestPathSolution:
        query = MPQuery(
            start=np.asarray(start, dtype=float),
            goal=np.asarray(goal, dtype=float),
            t_start=float(t_start),
            is_stay=bool(is_stay),
            vlimit=float(self.vlimit),
        )
        return self.plan(query, t_max)

    def plan(self, query: MPQuery, t_max: float) -> ShortestPathSolution:
        ts = time.perf_counter()
        self.last_projection, self.last_path = None, []
        self.last_status = "exhausted"
        try:
            projection = _ProjectedGrid(self.env, t_max=t_max, cell_size=self.cell_size,
                                        deadline=ts + self.runtime_limit_secs)
            self.last_projection = projection
            if not (projection.safe_intervals_for_point(query.start)
                    and projection.safe_intervals_for_point(query.goal)):
                self.last_status = "infeasible_query"
                return ShortestPathSolution(False, -1., time.perf_counter() - ts, [], [])
            if np.array_equal(query.start, query.goal):
                feasible = any(lo <= query.t_start <= hi and (not query.is_stay or hi >= t_max)
                               for lo, hi in projection.safe_intervals_for_point(query.start))
                projection.check_deadline()
                if feasible:
                    self.last_status = "success"
                    return ShortestPathSolution(True, 0., time.perf_counter() - ts, [],
                        [np.hstack([query.start, query.t_start, query.goal, query.t_start])],
                        Interval(query.t_start, query.t_start), self.env.dim + 1)
                return ShortestPathSolution(False, -1., time.perf_counter() - ts, [], [])
            planner_cls = _ZetaStarSIPP if self.use_fov else _ZetaSIPP
            planner = planner_cls(projection, query, t_max,
                                  max(0., projection.deadline - time.perf_counter()), self.time_buffer)
            try:
                path = planner.search()
            except ValueError:
                self.last_status = "infeasible_query"
                return ShortestPathSolution(False, -1., time.perf_counter() - ts, [], [])
            projection.check_deadline()
            self.last_path = path
            if path:
                solution = self._path_to_solution(path, query, time.perf_counter() - ts)
                projection.check_deadline()
                solution.time = time.perf_counter() - ts
                self.last_status = "success"
                return solution
        except TimeoutError:
            self.last_status = "timeout"
        return ShortestPathSolution(False, -1., time.perf_counter() - ts, [], [])

    def _path_to_solution(self, path: Sequence[_Node], query: MPQuery, runtime: float) -> ShortestPathSolution:
        if not path:
            return ShortestPathSolution(False, -1.0, runtime, [], [])

        trajectory: List[np.ndarray] = []
        current_time = float(query.t_start)
        for prev, curr in zip(path[:-1], path[1:]):
            travel_time = self.componentwise_travel_time(prev.world_point, curr.world_point, query.vlimit)
            depart_time = curr.g - travel_time
            if depart_time > current_time + TIME_EPS:
                trajectory.append(
                    np.hstack([prev.world_point, current_time, prev.world_point, depart_time])
                )
            if travel_time > TIME_EPS or not np.allclose(prev.world_point, curr.world_point):
                trajectory.append(
                    np.hstack([prev.world_point, depart_time, curr.world_point, curr.g])
                )
            current_time = curr.g

        if not trajectory:
            final_time = float(path[-1].g)
            trajectory.append(
                np.hstack([query.start, query.t_start, query.goal, final_time])
            )

        return ShortestPathSolution(
            is_success=True,
            cost=float(path[-1].g - query.t_start),
            time=runtime,
            vertex_path=[],
            trajectory=trajectory,
            itvl=Interval(float(query.t_start), float(path[-1].g)),
            dim=self.env.dim + 1,
        )
