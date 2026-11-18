from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import multiprocessing
import queue
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from stgcs_native.types import MPQuery
from stgcs_native.types import STTrajectory


@dataclass(frozen=True)
class OfficialOMPLKCBSOptions:
    max_runtime_in_secs: float = math.inf
    low_level_solve_time: float = 1.0
    propagation_step_size: float = 0.1
    min_control_duration: int = 1
    max_control_duration: int = 10
    goal_tolerance: float = 1e-3
    goal_bias: float = 0.05
    intermediate_states: bool = False
    merge_bound: Optional[int] = None
    num_threads: int = 4


@dataclass
class OfficialOMPLKCBSResult:
    success: bool
    runtime: float
    cost: float = math.inf
    solutions: Optional[List[STTrajectory]] = None
    num_nodes_expanded: int = 0
    num_approximate_solutions: int = 0
    root_solve_time: float = -1.0
    root_exact_solutions: int = 0
    root_solution_paths: int = 0
    root_all_exact: bool = False


class OfficialOMPLKCBS:
    """Adapter for the official Multi-Robot-OMPL K-CBS implementation.

    The native backend is intentionally optional because K-CBS is not part of
    upstream OMPL. It must be built against the K-CBS Multi-Robot-OMPL fork.
    """

    _NATIVE_MODULE = "baselines._ompl_kcbs_native"

    def __init__(self, env, seed: int = 0) -> None:
        self.env = env
        self.seed = int(seed)

    @classmethod
    def _native_module(cls):
        try:
            return importlib.import_module(cls._NATIVE_MODULE)
        except ImportError as exc:
            raise RuntimeError(
                "Official OMPL K-CBS backend is unavailable. Build "
                "baselines/ompl_kcbs_native against the K-CBS Multi-Robot-OMPL "
                f"fork so {cls._NATIVE_MODULE!r} is importable."
            ) from exc

    @classmethod
    def is_available(cls) -> bool:
        try:
            cls._native_module()
        except RuntimeError:
            return False
        return True

    @staticmethod
    def _options_payload(options: OfficialOMPLKCBSOptions) -> Dict:
        return {
            "max_runtime_in_secs": float(options.max_runtime_in_secs),
            "low_level_solve_time": float(options.low_level_solve_time),
            "propagation_step_size": float(options.propagation_step_size),
            "min_control_duration": int(options.min_control_duration),
            "max_control_duration": int(options.max_control_duration),
            "goal_tolerance": float(options.goal_tolerance),
            "goal_bias": float(options.goal_bias),
            "intermediate_states": bool(options.intermediate_states),
            "merge_bound": None if options.merge_bound is None else int(options.merge_bound),
            "num_threads": int(options.num_threads),
        }

    @staticmethod
    def _polygon_list_payload(polygons: Sequence, name: str) -> List[np.ndarray]:
        payload = []
        for idx, polygon in enumerate(polygons):
            vertices = np.asarray(polygon, dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
                raise ValueError(
                    f"K-CBS native {name}[{idx}] must be a 2D polygon with at least 3 vertices, "
                    f"got shape {vertices.shape}."
                )
            payload.append(vertices)
        return payload

    @classmethod
    def _environment_payload(cls, env) -> Dict:
        dynamic_obstacles = list(getattr(env, "O_Dynamic", []))
        if dynamic_obstacles:
            raise ValueError("Official OMPL K-CBS native adapter currently supports static MRMP domains only.")

        static_polygons = []
        for idx, obstacle in enumerate(getattr(env, "O_Static", [])):
            if not hasattr(obstacle, "vertices"):
                raise ValueError(
                    "Official OMPL K-CBS native adapter currently supports polygonal static obstacles only; "
                    f"obstacle {idx} has type {type(obstacle).__name__}."
                )
            static_polygons.append(np.asarray(obstacle.vertices, dtype=float))

        return {
            "static_polygons": cls._polygon_list_payload(static_polygons, "static_polygons"),
            "cspace_polygons": cls._polygon_list_payload(getattr(env, "C_Space", []), "cspace_polygons"),
        }

    @staticmethod
    def _failure_result(
        runtime: float = -1.0,
        *,
        num_nodes_expanded: int = 0,
        num_approximate_solutions: int = 0,
        root_solve_time: float = -1.0,
        root_exact_solutions: int = 0,
        root_solution_paths: int = 0,
        root_all_exact: bool = False,
    ) -> OfficialOMPLKCBSResult:
        return OfficialOMPLKCBSResult(
            success=False,
            runtime=float(runtime),
            num_nodes_expanded=int(num_nodes_expanded),
            num_approximate_solutions=int(num_approximate_solutions),
            root_solve_time=float(root_solve_time),
            root_exact_solutions=int(root_exact_solutions),
            root_solution_paths=int(root_solution_paths),
            root_all_exact=bool(root_all_exact),
        )

    @staticmethod
    def _segments_to_trajectory(
        segments: Sequence[Sequence[float]],
        spatial_dim: int,
        robot_idx: int,
    ) -> STTrajectory:
        points = [np.asarray(segment, dtype=float) for segment in segments]
        if not points:
            return STTrajectory([], [], spatial_dim)
        return STTrajectory(
            [f"ompl-kcbs-r{robot_idx}-{idx}" for idx in range(len(points))],
            points,
            spatial_dim,
        )

    @classmethod
    def _payload_to_result(cls, payload: Dict, spatial_dim: int) -> OfficialOMPLKCBSResult:
        runtime = float(payload.get("time", -1.0))
        if not bool(payload.get("is_success", False)):
            return cls._failure_result(
                runtime,
                num_nodes_expanded=int(payload.get("num_nodes_expanded", 0)),
                num_approximate_solutions=int(payload.get("num_approximate_solutions", 0)),
                root_solve_time=float(payload.get("root_solve_time", -1.0)),
                root_exact_solutions=int(payload.get("root_exact_solutions", 0)),
                root_solution_paths=int(payload.get("root_solution_paths", 0)),
                root_all_exact=bool(payload.get("root_all_exact", False)),
            )
        trajectories = payload.get("trajectories", [])
        solutions = [
            cls._segments_to_trajectory(segments, spatial_dim, robot_idx)
            for robot_idx, segments in enumerate(trajectories)
        ]
        if not solutions or any(solution.size == 0 for solution in solutions):
            return cls._failure_result(runtime)
        return OfficialOMPLKCBSResult(
            success=True,
            runtime=runtime,
            cost=float(payload.get("cost", sum(solution.duration for solution in solutions))),
            solutions=solutions,
            num_nodes_expanded=int(payload.get("num_nodes_expanded", 0)),
            num_approximate_solutions=int(payload.get("num_approximate_solutions", 0)),
            root_solve_time=float(payload.get("root_solve_time", -1.0)),
            root_exact_solutions=int(payload.get("root_exact_solutions", 0)),
            root_solution_paths=int(payload.get("root_solution_paths", 0)),
            root_all_exact=bool(payload.get("root_all_exact", False)),
        )

    def _validate_queries(self, queries: Sequence[MPQuery]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if int(self.env.dim) != 2:
            raise ValueError("Official OMPL K-CBS adapter currently follows the 2D disk-robot setup.")
        starts = []
        goals = []
        vlimits = []
        for query in queries:
            if abs(float(query.t_start)) > 1e-9:
                raise ValueError("Official OMPL K-CBS adapter requires all MRMP queries to start at t=0.")
            starts.append(np.asarray(query.start, dtype=float).reshape(int(self.env.dim)))
            goals.append(np.asarray(query.goal, dtype=float).reshape(int(self.env.dim)))
            vlimit = float(query.vlimit)
            if not math.isfinite(vlimit) or vlimit <= 0.0:
                raise ValueError(f"K-CBS query vlimit must be positive and finite, got {vlimit!r}.")
            vlimits.append(vlimit)
        if not starts:
            raise ValueError("K-CBS requires at least one query.")
        return np.asarray(starts, dtype=float), np.asarray(goals, dtype=float), np.asarray(vlimits, dtype=float)

    def solve(
        self,
        queries: Sequence[MPQuery],
        options: OfficialOMPLKCBSOptions,
    ) -> OfficialOMPLKCBSResult:
        if not math.isfinite(float(options.max_runtime_in_secs)) or float(options.max_runtime_in_secs) <= 0.0:
            raise ValueError("Official OMPL K-CBS requires a finite positive runtime limit.")
        starts, goals, vlimits = self._validate_queries(tuple(queries))
        native = self._native_module()
        ts = time.perf_counter()
        payload = native.solve_kcbs(
            starts,
            goals,
            vlimits,
            np.asarray(self.env.lb, dtype=float),
            np.asarray(self.env.ub, dtype=float),
            float(self.env.robot_radius),
            int(self.seed),
            self._options_payload(options),
            self._environment_payload(self.env),
        )
        measured_runtime = time.perf_counter() - ts
        result = self._payload_to_result(payload, int(self.env.dim))
        if result.runtime < 0.0:
            result.runtime = float(measured_runtime)
        return result


class OfficialOMPLKCBSPlanner:
    _RESULT_POLL_INTERVAL_SECS = 0.05
    _PROCESS_SHUTDOWN_GRACE_SECS = 0.5

    def __init__(
        self,
        options: Optional[OfficialOMPLKCBSOptions] = None,
        seed: int = 0,
    ) -> None:
        self.options = OfficialOMPLKCBSOptions() if options is None else options
        self.seed = int(seed)

    @staticmethod
    def _solve_child(
        result_queue,
        env,
        queries: Sequence[MPQuery],
        options: OfficialOMPLKCBSOptions,
        seed: int,
    ) -> None:
        try:
            result = OfficialOMPLKCBS(env, seed=int(seed)).solve(tuple(queries), options)
            result_queue.put(("ok", result))
        except BaseException as exc:  # pragma: no cover - exercised through parent process handling.
            result_queue.put(("error", repr(exc)))

    def _solve_isolated(
        self,
        env,
        queries: Sequence[MPQuery],
        options: OfficialOMPLKCBSOptions,
    ) -> OfficialOMPLKCBSResult:
        context = multiprocessing.get_context("fork")
        result_queue = context.Queue(maxsize=1)
        process = context.Process(
            target=self._solve_child,
            args=(result_queue, env, tuple(queries), options, int(self.seed)),
        )
        start_time = time.perf_counter()
        process.start()
        deadline = start_time + float(options.max_runtime_in_secs) + 2.0
        result_item = None
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                break
            try:
                result_item = result_queue.get(
                    timeout=min(self._RESULT_POLL_INTERVAL_SECS, remaining)
                )
                break
            except queue.Empty:
                if not process.is_alive():
                    break

        measured_runtime = time.perf_counter() - start_time
        if result_item is None:
            try:
                result_item = result_queue.get_nowait()
            except queue.Empty:
                pass

        if result_item is None:
            if process.is_alive():
                process.terminate()
            process.join()
            result_queue.close()
            result_queue.join_thread()
            return OfficialOMPLKCBS._failure_result(runtime=measured_runtime)

        process.join(self._PROCESS_SHUTDOWN_GRACE_SECS)
        if process.is_alive():
            process.terminate()
            process.join()
        result_queue.close()
        result_queue.join_thread()
        status, payload = result_item
        if status == "error":
            raise RuntimeError(f"OMPL K-CBS child process failed: {payload}")
        return payload

    def solve(
        self,
        instance,
        queries: Sequence[MPQuery],
        timeout_secs: float,
    ) -> OfficialOMPLKCBSResult:
        options = OfficialOMPLKCBSOptions(
            max_runtime_in_secs=float(timeout_secs),
            low_level_solve_time=float(self.options.low_level_solve_time),
            propagation_step_size=float(self.options.propagation_step_size),
            min_control_duration=int(self.options.min_control_duration),
            max_control_duration=int(self.options.max_control_duration),
            goal_tolerance=float(self.options.goal_tolerance),
            goal_bias=float(self.options.goal_bias),
            intermediate_states=bool(self.options.intermediate_states),
            merge_bound=self.options.merge_bound,
            num_threads=int(self.options.num_threads),
        )
        env = instance.env.copy()
        if isinstance(OfficialOMPLKCBS, type):
            return self._solve_isolated(env, tuple(queries), options)
        planner = OfficialOMPLKCBS(env, seed=self.seed)
        return planner.solve(tuple(queries), options)
