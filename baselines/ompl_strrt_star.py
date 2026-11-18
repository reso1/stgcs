from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from baselines.common import ShortestPathSolution
from experiments.environment.obstacle import ConcatDynamicSphere, DynamicSphere
from experiments.environment.collision import collision_geometries, collision_geometries_collide
from experiments.environment.interval import Interval
from stgcs_native.types import STTrajectory


@dataclass(frozen=True)
class OfficialOMPLSTRRTStarOptions:
    max_runtime_in_secs: float = math.inf
    time_upper_bound: Optional[float] = None
    range: Optional[float] = None
    time_weight: float = 0.5
    batch_size: Optional[int] = None
    initial_time_bound_factor: Optional[float] = None
    time_bound_factor_increase: Optional[float] = None
    max_time_bound_factor: Optional[float] = 10000.0
    optimum_approx_factor: Optional[float] = None
    goal_tolerance: float = 1e-7
    return_first_valid: bool = False
    stay_at_goal: bool = False


@dataclass(frozen=True)
class OfficialOMPLSTRRTStarRunSnapshots:
    first_solution: ShortestPathSolution
    final_solution: ShortestPathSolution


class OfficialOMPLSTRRTStar:
    """Adapter for official OMPL STRRTstar over this repo's Env geometry.

    The native extension implements a custom OMPL spatial state space whose
    distance is max(abs(dx_i) / vlimit_i). OMPL's SpaceTimeStateSpace then uses
    vMax=1, so its speed feasibility check matches this repo's travel-time
    lower bound.

    Static 2D polygonal environment collision checking is performed in the
    native OMPL motion validator. Dynamic obstacles use the same swept
    hypercube occupancy as ECD. ``collision_checker`` is reserved for extra
    time-dependent constraints, such as fixed-priority reservations.
    """

    _NATIVE_MODULE = "baselines._ompl_strrt_star_native"

    def __init__(
        self,
        env,
        seed: int,
        vlimit: float | np.ndarray,
        collision_checker: Optional[
            Callable[[np.ndarray, np.ndarray, float, float], bool]
        ] = None,
    ) -> None:
        self.env = env
        self.seed = int(seed)
        self.vlimit = np.asarray(vlimit, dtype=float)
        self.collision_checker = collision_checker

    @classmethod
    def _native_module(cls):
        try:
            return importlib.import_module(cls._NATIVE_MODULE)
        except ImportError as exc:
            raise RuntimeError(
                "Official OMPL STRRT* backend is unavailable. Build "
                "baselines/ompl_strrt_star_native against OMPL and pybind11 "
                f"so {cls._NATIVE_MODULE!r} is importable."
            ) from exc

    @classmethod
    def is_available(cls) -> bool:
        try:
            cls._native_module()
        except RuntimeError:
            return False
        return True

    @staticmethod
    def _failure_solution(runtime: float = -1.0) -> ShortestPathSolution:
        return ShortestPathSolution(False, math.inf, float(runtime), [], [])

    @staticmethod
    def _payload_to_solution(payload: Dict, spatial_dim: int) -> ShortestPathSolution:
        if not bool(payload.get("is_success", False)):
            return OfficialOMPLSTRRTStar._failure_solution(float(payload.get("time", -1.0)))
        trajectory = [np.asarray(point, dtype=float) for point in payload.get("trajectory", [])]
        if not trajectory:
            return OfficialOMPLSTRRTStar._failure_solution(float(payload.get("time", -1.0)))
        start_time = float(trajectory[0][spatial_dim])
        end_time = float(trajectory[-1][-1])
        return ShortestPathSolution(
            is_success=True,
            cost=float(payload.get("cost", end_time - start_time)),
            time=float(payload.get("time", -1.0)),
            vertex_path=[],
            trajectory=trajectory,
            itvl=Interval(start_time, end_time),
            dim=spatial_dim + 1,
        )

    @staticmethod
    def solution_to_trajectory(
        solution: ShortestPathSolution,
        spatial_dim: int,
        vertex_prefix: str = "ompl-strrt",
    ) -> STTrajectory:
        if not solution.is_success or len(solution.trajectory) == 0:
            return STTrajectory([], [], spatial_dim)
        return STTrajectory(
            [f"{vertex_prefix}-{idx}" for idx in range(len(solution.trajectory))],
            [np.asarray(point, dtype=float) for point in solution.trajectory],
            spatial_dim,
        )

    @staticmethod
    def _options_payload(options: OfficialOMPLSTRRTStarOptions) -> Dict:
        return {
            "max_runtime_in_secs": float(options.max_runtime_in_secs),
            "time_upper_bound": (
                None if options.time_upper_bound is None else float(options.time_upper_bound)
            ),
            "range": None if options.range is None else float(options.range),
            "time_weight": float(options.time_weight),
            "batch_size": None if options.batch_size is None else int(options.batch_size),
            "initial_time_bound_factor": (
                None
                if options.initial_time_bound_factor is None
                else float(options.initial_time_bound_factor)
            ),
            "time_bound_factor_increase": (
                None
                if options.time_bound_factor_increase is None
                else float(options.time_bound_factor_increase)
            ),
            "max_time_bound_factor": (
                None
                if options.max_time_bound_factor is None
                else float(options.max_time_bound_factor)
            ),
            "optimum_approx_factor": (
                None
                if options.optimum_approx_factor is None
                else float(options.optimum_approx_factor)
            ),
            "goal_tolerance": float(options.goal_tolerance),
            "return_first_valid": bool(options.return_first_valid),
            "stay_at_goal": bool(options.stay_at_goal),
        }

    @staticmethod
    def _polygon_list_payload(polygons: Sequence, name: str) -> List[np.ndarray]:
        payload = []
        for idx, polygon in enumerate(polygons):
            vertices = np.asarray(polygon, dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
                raise ValueError(
                    f"Official OMPL STRRT* native {name}[{idx}] must be a 2D polygon "
                    f"with at least 3 vertices, got shape {vertices.shape}."
                )
            payload.append(vertices)
        return payload

    @classmethod
    def _environment_payload(cls, env) -> Dict:
        dim = int(getattr(env, "dim"))
        static_polygons = []
        cspace_polygons = []

        if dim == 2:
            for idx, obstacle in enumerate(getattr(env, "O_Static", [])):
                if not hasattr(obstacle, "vertices"):
                    raise ValueError(
                        "Official OMPL STRRT* native adapter currently supports polygonal "
                        f"static obstacles only; obstacle {idx} has type {type(obstacle).__name__}."
                    )
                static_polygons.append(np.asarray(obstacle.vertices, dtype=float))
            cspace_polygons = list(getattr(env, "C_Space", []))
        elif list(getattr(env, "O_Static", [])):
            raise ValueError(
                "Official OMPL STRRT* native static obstacle checking currently supports 2D polygonal Env geometry only."
            )

        return {
            "robot_radius": float(getattr(env, "robot_radius", 0.0)),
            "static_polygons": cls._polygon_list_payload(static_polygons, "static_polygons"),
            "cspace_polygons": cls._polygon_list_payload(cspace_polygons, "cspace_polygons"),
        }

    @staticmethod
    def _dynamic_obstacle_collision_checker(env, t0: float = 0.0, tmax: float = 1e9) -> Optional[
        Callable[[np.ndarray, np.ndarray, float, float], bool]
    ]:
        """Check complete segments against moving boxes, including endpoint holds."""
        dynamic_obstacles = tuple(getattr(env, "O_Dynamic", []))
        if not dynamic_obstacles:
            return None
        robot_radius = float(getattr(env, "robot_radius", 0.0))

        obstacle_geometries = []
        for obstacle in dynamic_obstacles:
            if isinstance(obstacle, DynamicSphere):
                pieces = [obstacle]
            elif isinstance(obstacle, ConcatDynamicSphere):
                pieces = obstacle.segments
            else:
                raise TypeError(f"Unsupported piecewise-linear obstacle: {type(obstacle).__name__}")
            trajectory = [np.hstack([piece.x0, piece.itvl.start, piece.xt, piece.itvl.end])
                          for piece in pieces]
            reserve_start = not isinstance(obstacle, ConcatDynamicSphere) or obstacle.reserve_start
            reserve_goal = not isinstance(obstacle, ConcatDynamicSphere) or obstacle.reserve_goal
            obstacle_geometries.append(collision_geometries(
                trajectory, cc_to_t0=reserve_start and t0 < pieces[0].itvl.start,
                cc_to_tf=reserve_goal and tmax > pieces[-1].itvl.end, space_dim=int(env.dim),
                robot_radius=float(obstacle.radius), t0=t0, tmax=tmax,
                tolerance=0.0, collision_geometry_cache={},
            ))

        def collision_checker(p: np.ndarray, q: np.ndarray, tp: float, tq: float) -> bool:
            # Use the same swept hypercube geometry as ECD and MRMP conflict checks.
            robot_geometry = collision_geometries(
                [np.hstack([p, tp, q, tq])], cc_to_t0=False, cc_to_tf=False,
                space_dim=int(env.dim), robot_radius=robot_radius, t0=t0, tmax=tmax,
                tolerance=0.0, collision_geometry_cache={},
            )
            return any(collision_geometries_collide(robot_geometry, obstacle)
                       for obstacle in obstacle_geometries)

        return collision_checker

    def _extra_collision_checker(self, t0: float = 0.0, tmax: float = 1e9) -> Optional[
        Callable[[np.ndarray, np.ndarray, float, float], bool]
    ]:
        dynamic_checker = self._dynamic_obstacle_collision_checker(self.env, t0, tmax)
        if dynamic_checker is None:
            return self.collision_checker
        if self.collision_checker is None:
            return dynamic_checker

        def combined_checker(p: np.ndarray, q: np.ndarray, tp: float, tq: float) -> bool:
            return bool(dynamic_checker(p, q, tp, tq) or self.collision_checker(p, q, tp, tq))

        return combined_checker

    def _vlimit_vector(self) -> np.ndarray:
        if self.vlimit.ndim == 0:
            return np.full(int(self.env.dim), float(self.vlimit), dtype=float)
        if self.vlimit.shape != (int(self.env.dim),):
            raise ValueError(
                f"vlimit must be scalar or shape ({int(self.env.dim)},), "
                f"got shape {self.vlimit.shape}."
            )
        return np.asarray(self.vlimit, dtype=float)

    def solve_with_snapshots(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        t0: float,
        options: OfficialOMPLSTRRTStarOptions,
    ) -> OfficialOMPLSTRRTStarRunSnapshots:
        if not math.isfinite(float(options.max_runtime_in_secs)):
            raise ValueError("Official OMPL STRRT* requires a finite runtime limit.")
        if options.stay_at_goal and options.time_upper_bound is None:
            raise ValueError("stay_at_goal requires a finite time_upper_bound.")
        if options.time_upper_bound is not None:
            if not math.isfinite(float(options.time_upper_bound)):
                raise ValueError("Official OMPL STRRT* time upper bound must be finite when provided.")
            if float(options.time_upper_bound) < float(t0):
                raise ValueError("Official OMPL STRRT* time upper bound must be greater than or equal to t0.")
        native = self._native_module()
        ts = time.perf_counter()
        payload = native.solve_linf_strrt(
            self._environment_payload(self.env),
            self._extra_collision_checker(
                t0, options.time_upper_bound if options.time_upper_bound is not None else 1e9,
            ),
            np.asarray(start_pos, dtype=float),
            np.asarray(goal_pos, dtype=float),
            float(t0),
            self._vlimit_vector(),
            np.asarray(self.env.lb, dtype=float),
            np.asarray(self.env.ub, dtype=float),
            int(self.seed),
            self._options_payload(options),
        )
        measured_runtime = time.perf_counter() - ts
        first = self._payload_to_solution(payload.get("first", {}), self.env.dim)
        final = self._payload_to_solution(payload.get("final", {}), self.env.dim)
        if first.time < 0.0:
            first.time = float(measured_runtime)
        if final.time < 0.0:
            final.time = float(measured_runtime)
        return OfficialOMPLSTRRTStarRunSnapshots(first_solution=first, final_solution=final)

    def solve(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        t0: float,
        options: OfficialOMPLSTRRTStarOptions,
    ) -> ShortestPathSolution:
        snapshots = self.solve_with_snapshots(start_pos, goal_pos, t0, options)
        return snapshots.first_solution if options.return_first_valid else snapshots.final_solution
