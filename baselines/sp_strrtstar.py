from __future__ import annotations
from dataclasses import dataclass, field
import math
import multiprocessing
import queue
from typing import List, Optional, Sequence

import time
import numpy as np

from experiments.environment.env import Env

from baselines.common import ShortestPathSolution
from baselines.ompl_strrt_star import OfficialOMPLSTRRTStar, OfficialOMPLSTRRTStarOptions

from baselines.priority_search import PriorityBasedSearch
from stgcs_native.types import MPQuery
from stgcs_native.types import STTrajectory

_STRRT_STAR_IMPLEMENTATION = OfficialOMPLSTRRTStar


class _PriorityReservationCollisionChecker:
    def __init__(
        self,
        env: Env,
        checker: PriorityBasedSearch,
        reserved: Sequence[tuple[STTrajectory, bool]],
        candidate_goal: np.ndarray,
        candidate_is_stay: bool,
        goal_tolerance: float,
    ) -> None:
        self.env = env
        self.checker = checker
        self.reserved = tuple((trajectory, bool(is_stay)) for trajectory, is_stay in reserved)
        self.candidate_goal = np.asarray(candidate_goal, dtype=float)
        self.candidate_is_stay = bool(candidate_is_stay)
        self.goal_tolerance = float(goal_tolerance)

    def __call__(self, p: np.ndarray, q: np.ndarray, tp: float, tq: float) -> bool:
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        tp = float(tp)
        tq = float(tq)

        space_dim = int(self.env.dim)
        candidate = [np.concatenate([p[:space_dim], [tp], q[:space_dim], [tq]])]
        for trajectory, is_stay in self.reserved:
            if self.checker.collision_checking(
                candidate,
                trajectory.points,
                cc_to_t0_a=False,
                cc_to_tf_a=False,
                cc_to_t0_b=True,
                cc_to_tf_b=is_stay,
            ):
                return True
        if self.candidate_is_stay and np.allclose(
            q[:space_dim],
            self.candidate_goal,
            atol=self.goal_tolerance,
            rtol=0.0,
        ):
            candidate_stay = [
                np.concatenate([q[:space_dim], [tq], q[:space_dim], [self.checker.tmax]])
            ]
            for trajectory, is_stay in self.reserved:
                if self.checker.collision_checking(
                    candidate_stay,
                    trajectory.points,
                    cc_to_t0_a=False,
                    cc_to_tf_a=False,
                    cc_to_t0_b=True,
                    cc_to_tf_b=is_stay,
                ):
                    return True
        return False


@dataclass(frozen=True)
class FixedPrioritySTRRTStarOptions:
    seed: int = 0
    time_upper_bound: Optional[float] = None
    range: Optional[float] = None
    time_weight: float = 0.5
    batch_size: Optional[int] = None
    initial_time_bound_factor: Optional[float] = None
    time_bound_factor_increase: Optional[float] = None
    max_time_bound_factor: Optional[float] = 10000.0
    optimum_approx_factor: Optional[float] = None
    goal_tolerance: float = 1e-7
    improvement_tolerance: float = 1e-6


@dataclass
class FixedPrioritySTRRTStarSnapshot:
    solutions: Optional[List[STTrajectory]]
    success: bool
    runtime: float
    cost: float = math.inf
    low_level_calls: int = 0
    low_level_runtime: float = 0.0
    collision_checks: int = 0
    collision_runtime: float = 0.0


@dataclass
class FixedPrioritySTRRTStarResult:
    first_solution: FixedPrioritySTRRTStarSnapshot
    final_solution: FixedPrioritySTRRTStarSnapshot
    improvement_rounds: int = 0
    histories: List[List[STTrajectory]] = field(default_factory=list)

    @property
    def solutions(self) -> Optional[List[STTrajectory]]:
        return self.final_solution.solutions

    @property
    def success(self) -> bool:
        return self.final_solution.success

    @property
    def runtime(self) -> float:
        return self.final_solution.runtime

    @property
    def cost(self) -> float:
        return self.final_solution.cost

    @property
    def low_level_calls(self) -> int:
        return self.final_solution.low_level_calls

    @property
    def low_level_runtime(self) -> float:
        return self.final_solution.low_level_runtime

    @property
    def collision_checks(self) -> int:
        return self.final_solution.collision_checks

    @property
    def collision_runtime(self) -> float:
        return self.final_solution.collision_runtime


class FixedPrioritySTRRTStarPlanner:
    _RESULT_POLL_INTERVAL_SECS = 0.05
    _PROCESS_SHUTDOWN_GRACE_SECS = 0.5

    def __init__(
        self,
        options: Optional[FixedPrioritySTRRTStarOptions] = None,
        priority_order: Optional[Sequence[int]] = None,
    ) -> None:
        self.options = FixedPrioritySTRRTStarOptions() if options is None else options
        self.priority_order = None if priority_order is None else tuple(int(idx) for idx in priority_order)
        self.low_level_calls = 0
        self.low_level_runtime = 0.0
        self.histories: List[List[STTrajectory]] = []
        self.improvement_rounds = 0

    @staticmethod
    def _resolved_priority_order(
        priority_order: Optional[Sequence[int]],
        num_agents: int,
    ) -> tuple[int, ...]:
        if priority_order is None:
            return tuple(range(num_agents))
        order = tuple(int(idx) for idx in priority_order)
        if len(order) != num_agents or sorted(order) != list(range(num_agents)):
            raise ValueError(
                "priority_order must be a permutation of all agent indices "
                f"0..{num_agents - 1}, got {order}."
            )
        return order

    @staticmethod
    def _empty_trajectory(spatial_dim: int) -> STTrajectory:
        return STTrajectory([], [], spatial_dim)

    @staticmethod
    def _wait_trajectory(query: MPQuery, spatial_dim: int) -> STTrajectory:
        point = np.hstack([query.start, query.t_start, query.goal, query.t_start])
        return STTrajectory(["strrt-wait"], [point], spatial_dim)

    @staticmethod
    def _same_position(a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.allclose(a, b, atol=1e-9))

    @staticmethod
    def _solution_cost(solutions: Sequence[STTrajectory]) -> float:
        return float(sum(solution.duration for solution in solutions))

    @staticmethod
    def _copy_solutions(solutions: Sequence[STTrajectory]) -> List[STTrajectory]:
        return [solution.copy() for solution in solutions]

    def _strrt_options(
        self,
        runtime_limit_secs: float,
    ) -> OfficialOMPLSTRRTStarOptions:
        return OfficialOMPLSTRRTStarOptions(
            max_runtime_in_secs=float(runtime_limit_secs),
            time_upper_bound=(
                None
                if self.options.time_upper_bound is None
                else float(self.options.time_upper_bound)
            ),
            range=self.options.range,
            time_weight=float(self.options.time_weight),
            batch_size=self.options.batch_size,
            initial_time_bound_factor=self.options.initial_time_bound_factor,
            time_bound_factor_increase=self.options.time_bound_factor_increase,
            max_time_bound_factor=self.options.max_time_bound_factor,
            optimum_approx_factor=self.options.optimum_approx_factor,
            goal_tolerance=float(self.options.goal_tolerance),
            return_first_valid=True,
        )

    @staticmethod
    def _reservation_collision_checker(
        env: Env,
        checker: PriorityBasedSearch,
        solutions: Sequence[STTrajectory],
        queries: Sequence[MPQuery],
        excluded_agent_idx: int,
        goal_tolerance: float,
    ) -> _PriorityReservationCollisionChecker:
        reserved = []
        for agent_idx, solution in enumerate(solutions):
            if agent_idx == excluded_agent_idx or solution.size == 0:
                continue
            reserved.append((solution, bool(queries[agent_idx].is_stay)))
        query = queries[excluded_agent_idx]
        return _PriorityReservationCollisionChecker(
            env,
            checker,
            reserved,
            query.goal,
            bool(query.is_stay),
            goal_tolerance,
        )

    def _seed_for_call(self, agent_idx: int) -> int:
        return int((self.options.seed + int(agent_idx) * 1_000_003 + self.low_level_calls) % (2**31 - 1))

    @staticmethod
    def _solve_ompl_child(
        result_queue,
        env: Env,
        query: MPQuery,
        seed: int,
        options: OfficialOMPLSTRRTStarOptions,
        collision_checker,
    ) -> None:
        try:
            planner = _STRRT_STAR_IMPLEMENTATION(
                env,
                int(seed),
                float(query.vlimit),
                collision_checker=collision_checker,
            )
            solution = planner.solve(
                query.start,
                query.goal,
                t0=float(query.t_start),
                options=options,
            )
            result_queue.put(("ok", solution))
        except BaseException as exc:  # pragma: no cover - exercised via parent error handling.
            result_queue.put(("error", repr(exc)))

    @classmethod
    def _solve_ompl_isolated(
        cls,
        env: Env,
        query: MPQuery,
        seed: int,
        options: OfficialOMPLSTRRTStarOptions,
        collision_checker=None,
    ) -> ShortestPathSolution:
        context = multiprocessing.get_context("fork")
        result_queue = context.Queue(maxsize=1)
        process = context.Process(
            target=cls._solve_ompl_child,
            args=(result_queue, env, query, int(seed), options, collision_checker),
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
                    timeout=min(cls._RESULT_POLL_INTERVAL_SECS, remaining)
                )
                break
            except queue.Empty:
                if not process.is_alive():
                    break

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
                return _STRRT_STAR_IMPLEMENTATION._failure_solution(float(options.max_runtime_in_secs))
            process.join()
            result_queue.close()
            result_queue.join_thread()
            if process.exitcode == 0:
                return _STRRT_STAR_IMPLEMENTATION._failure_solution(float(options.max_runtime_in_secs))
            raise RuntimeError(f"OMPL ST-RRT* child process exited with code {process.exitcode}.")

        process.join(cls._PROCESS_SHUTDOWN_GRACE_SECS)
        if process.is_alive():
            process.terminate()
            process.join()
        result_queue.close()
        result_queue.join_thread()
        status, payload = result_item
        if status == "error":
            raise RuntimeError(f"OMPL ST-RRT* child process failed: {payload}")
        return payload

    def _plan_agent_once(
        self,
        env: Env,
        query: MPQuery,
        agent_idx: int,
        runtime_limit_secs: float,
        collision_checker=None,
    ) -> Optional[STTrajectory]:
        if self._same_position(query.start, query.goal):
            return self._wait_trajectory(query, env.dim)
        if runtime_limit_secs <= 0.0:
            return None
        seed = self._seed_for_call(agent_idx)
        options = self._strrt_options(runtime_limit_secs)
        start_time = time.perf_counter()
        if OfficialOMPLSTRRTStar is _STRRT_STAR_IMPLEMENTATION:
            solution = self._solve_ompl_isolated(
                env,
                query,
                seed,
                options,
                collision_checker,
            )
        else:
            planner = OfficialOMPLSTRRTStar(
                env,
                seed,
                float(query.vlimit),
                collision_checker=collision_checker,
            )
            solution = planner.solve(
                query.start,
                query.goal,
                t0=float(query.t_start),
                options=options,
            )
        runtime = time.perf_counter() - start_time
        self.low_level_calls += 1
        self.low_level_runtime += float(runtime)
        if not solution.is_success:
            return None
        trajectory = _STRRT_STAR_IMPLEMENTATION.solution_to_trajectory(solution, env.dim)
        if trajectory.size == 0:
            return None
        return trajectory

    @staticmethod
    def _conflicts_with_incumbents(
        checker: PriorityBasedSearch,
        candidate: STTrajectory,
        agent_idx: int,
        solutions: Sequence[STTrajectory],
        queries: Sequence[MPQuery],
    ) -> bool:
        for other_idx, other in enumerate(solutions):
            if other_idx == agent_idx or other.size == 0:
                continue
            if checker.collision_checking(
                candidate.points,
                other.points,
                cc_to_tf_a=bool(queries[agent_idx].is_stay),
                cc_to_tf_b=bool(queries[other_idx].is_stay),
            ):
                return True
        return False

    def _build_snapshot(
        self,
        solutions: Optional[List[STTrajectory]],
        success: bool,
        start_time: float,
        checker: Optional[PriorityBasedSearch],
    ) -> FixedPrioritySTRRTStarSnapshot:
        collision_profile = np.zeros(2, dtype=float)
        if checker is not None:
            collision_profile = checker._profiler["cc"]
        return FixedPrioritySTRRTStarSnapshot(
            solutions=None if solutions is None else self._copy_solutions(solutions),
            success=bool(success),
            runtime=float(time.perf_counter() - start_time),
            cost=self._solution_cost(solutions) if success and solutions is not None else math.inf,
            low_level_calls=int(self.low_level_calls),
            low_level_runtime=float(self.low_level_runtime),
            collision_checks=int(collision_profile[0]),
            collision_runtime=float(collision_profile[1]),
        )

    def _build_result(
        self,
        first_solution: FixedPrioritySTRRTStarSnapshot,
        final_solution: FixedPrioritySTRRTStarSnapshot,
    ) -> FixedPrioritySTRRTStarResult:
        return FixedPrioritySTRRTStarResult(
            first_solution=first_solution,
            final_solution=final_solution,
            improvement_rounds=int(self.improvement_rounds),
            histories=[
                [trajectory.copy() for trajectory in agent_history]
                for agent_history in self.histories
            ],
        )

    def solve(
        self,
        instance,
        queries: Sequence[MPQuery],
        timeout_secs: float,
    ) -> FixedPrioritySTRRTStarResult:
        start_time = time.perf_counter()
        deadline = start_time + float(timeout_secs)
        self.low_level_calls = 0
        self.low_level_runtime = 0.0
        self.improvement_rounds = 0
        queries = tuple(queries)
        order = self._resolved_priority_order(self.priority_order, len(queries))
        self.histories = [[] for _ in queries]
        checker = PriorityBasedSearch(instance.stgcs, None, instance.env.robot_radius)
        solutions = [self._empty_trajectory(instance.env.dim) for _ in queries]

        for agent_idx in order:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                failure = self._build_snapshot(solutions, False, start_time, checker)
                return self._build_result(failure, failure)
            collision_checker = self._reservation_collision_checker(
                instance.env,
                checker,
                solutions,
                queries,
                excluded_agent_idx=agent_idx,
                goal_tolerance=self.options.goal_tolerance,
            )
            trajectory = self._plan_agent_once(
                instance.env,
                queries[agent_idx],
                agent_idx,
                remaining,
                collision_checker=collision_checker,
            )
            if trajectory is None or self._conflicts_with_incumbents(
                checker,
                trajectory,
                agent_idx,
                solutions,
                queries,
            ):
                failure = self._build_snapshot(solutions, False, start_time, checker)
                return self._build_result(failure, failure)
            solutions[agent_idx] = trajectory
            self.histories[agent_idx].append(trajectory.copy())

        first_solution = self._build_snapshot(solutions, True, start_time, checker)

        while time.perf_counter() < deadline:
            attempted = False
            for agent_idx in order:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                current = solutions[agent_idx]
                if current.size == 0 or current.duration <= self.options.improvement_tolerance:
                    continue
                attempted = True
                collision_checker = self._reservation_collision_checker(
                    instance.env,
                    checker,
                    solutions,
                    queries,
                    excluded_agent_idx=agent_idx,
                    goal_tolerance=self.options.goal_tolerance,
                )
                trajectory = self._plan_agent_once(
                    instance.env,
                    queries[agent_idx],
                    agent_idx,
                    remaining,
                    collision_checker=collision_checker,
                )
                if trajectory is None:
                    continue
                if trajectory.duration >= current.duration - self.options.improvement_tolerance:
                    continue
                if self._conflicts_with_incumbents(checker, trajectory, agent_idx, solutions, queries):
                    continue
                solutions[agent_idx] = trajectory
                self.histories[agent_idx].append(trajectory.copy())
            if not attempted:
                break
            self.improvement_rounds += 1

        final_solution = self._build_snapshot(solutions, True, start_time, checker)
        return self._build_result(first_solution, final_solution)
