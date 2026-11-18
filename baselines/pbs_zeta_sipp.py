from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from baselines.common import ShortestPathSolution
from baselines.zeta_sipp import ZetaStarSIPPPlanner
from experiments.environment.env import Env
from experiments.environment.obstacle import ConcatDynamicSphere
from baselines.priority_search import (
    DEFAULT_CHILD_EXPANSION_MODE,
    ChildExpansionMode,
    LowLevelPlanCacheKey,
    PBSNode,
    PriorityBasedSearch,
)
from stgcs_native.types import MPQuery, STPlanStatus
from experiments.environment.space_time_graph import STGCS
from stgcs_native.types import STTrajectory


_ZETA_STAR_SIPP_PLANNER = ZetaStarSIPPPlanner


class ZetaSIPPPriorityBasedSearch(PriorityBasedSearch):
    """Priority-based search with Zeta*-SIPP as the low-level 2D planner."""

    def __init__(
        self,
        stgcs: STGCS,
        env: Env,
        robot_radius: Optional[float] = None,
        child_expansion_mode: ChildExpansionMode = DEFAULT_CHILD_EXPANSION_MODE,
        seed: int = 0,
        cell_size: Optional[float] = None,
        runtime_limit_secs: float = math.inf,
        time_buffer: float = 1e-3,
        use_fov: bool = True,
    ) -> None:
        if int(env.dim) != 2:
            raise ValueError("PBS + Zeta*-SIPP is defined only for 2D environments.")
        if int(stgcs.dimension) != int(env.dim):
            raise ValueError(
                f"ST-GCS dimension must match environment dimension, got {stgcs.dimension} and {env.dim}."
            )
        super().__init__(
            stgcs,
            st_planner=None,
            robot_radius=float(env.robot_radius if robot_radius is None else robot_radius),
            child_expansion_mode=child_expansion_mode,
        )
        self.env = env
        self.seed = int(seed)
        self.cell_size = None if cell_size is None else float(cell_size)
        self.runtime_limit_secs = float(runtime_limit_secs)
        self.time_buffer = float(time_buffer)
        self.use_fov = bool(use_fov)
        self._run_started_at = 0.0
        self._timeout_secs = math.inf

    @staticmethod
    def solution_to_trajectory(
        solution: ShortestPathSolution,
        spatial_dim: int,
        vertex_prefix: str = "zeta-sipp",
    ) -> STTrajectory:
        if not solution.is_success or len(solution.trajectory) == 0:
            return STTrajectory([], [], spatial_dim)
        return STTrajectory(
            [f"{vertex_prefix}-{idx}" for idx in range(len(solution.trajectory))],
            [np.asarray(point, dtype=float) for point in solution.trajectory],
            spatial_dim,
        )

    def _planner_cache_key(self) -> Tuple[object, ...]:
        return (
            f"{type(self).__module__}.{type(self).__qualname__}",
            int(self.seed),
            None if self.cell_size is None else float(self.cell_size),
            float(self.runtime_limit_secs),
            float(self.time_buffer),
            bool(self.use_fov),
        )

    def _reset_run_state(self, queries: Sequence[MPQuery], timeout_secs: float) -> None:
        self.last_solution_prio_graph = None
        self.last_solution_order = tuple()
        self._queries = tuple(queries)
        self.num_popped_nodes = 0
        self.num_generated_children = 0
        self.num_update_calls = 0
        self._collision_geometry_cache.clear()
        self._reserved_stgcs_cache.clear()
        self._low_level_plan_cache.clear()
        self._structural_failure_cache.clear()
        self._ecd_pair_cache.clear()
        self._collision_pair_cache.clear()
        self._run_started_at = time.perf_counter()
        self._timeout_secs = float(timeout_secs)

    def _remaining_runtime_limit(self) -> float:
        elapsed = time.perf_counter() - self._run_started_at
        remaining = self._timeout_secs - elapsed
        return min(float(self.runtime_limit_secs), max(0.0, float(remaining)))

    def _seed_for_agent(self, agent_idx: int) -> int:
        return int((self.seed + int(agent_idx) * 1_000_003) % (2**31 - 1))

    def _base_num_edges(self) -> int:
        graph = getattr(self._stgcs_base, "G", None)
        if graph is None or not hasattr(graph, "number_of_edges"):
            return 0
        return int(graph.number_of_edges())

    def _set_or_append_solution(self, node: PBSNode, agent_idx: int, sol: STTrajectory) -> None:
        if int(agent_idx) < len(node.sols):
            node.set_solution(int(agent_idx), sol)
            return
        if int(agent_idx) != len(node.sols):
            raise IndexError(f"Cannot append solution for agent {agent_idx}; node has {len(node.sols)} solutions.")
        node.append_solution(sol)

    def _reserved_env_for_node(
        self,
        node: PBSNode,
        high_priority_agents: Sequence[int],
        queries: Sequence[MPQuery],
    ) -> Env:
        reserved_env = self.env.copy()
        for agent_idx in high_priority_agents:
            sol = node.sols[int(agent_idx)]
            if sol.size == 0:
                continue
            x0_staying, xt_staying = self._reservation_stay_modes(int(agent_idx), list(queries))
            reserved_env.O_Dynamic.append(
                ConcatDynamicSphere.from_solution(
                    sol,
                    radius=self.robot_radius,
                    reserve_start=x0_staying,
                    reserve_goal=xt_staying,
                )
            )
        return reserved_env

    def _plan_agent(
        self,
        node: PBSNode,
        agent_idx: int,
        high_priority_agents: Sequence[int],
        queries: Sequence[MPQuery],
    ) -> bool:
        runtime_limit = self._remaining_runtime_limit()
        if runtime_limit <= 0.0:
            return False

        query = queries[int(agent_idx)]
        reserved_env = self._reserved_env_for_node(
            node,
            high_priority_agents,
            queries,
        )
        planner = _ZETA_STAR_SIPP_PLANNER(
            reserved_env,
            self._seed_for_agent(agent_idx),
            float(query.vlimit),
            cell_size=self.cell_size,
            runtime_limit_secs=runtime_limit,
            time_buffer=self.time_buffer,
            use_fov=self.use_fov,
        )
        started_at = time.perf_counter()
        solution = planner.plan(query, self.tmax)
        measured_runtime = time.perf_counter() - started_at
        runtime = float(solution.time)
        if not math.isfinite(runtime) or runtime < 0.0:
            runtime = measured_runtime
        self._profiler["mp"] += np.array([1, runtime])

        if not solution.is_success:
            self.print(f"\tFails to plan agent {agent_idx} with Zeta*-SIPP.")
            return False

        sol = self.solution_to_trajectory(solution, self.env.dim)
        self._set_or_append_solution(node, int(agent_idx), sol)
        node._stgcs_num_edges.append(self._base_num_edges())
        self.print(f"\tPlans agent {agent_idx} with Zeta*-SIPP in {runtime:.3f}s.")
        return True

    def run(
        self,
        queries: List[MPQuery],
        timeout_secs: float,
        verbose: bool = True,
    ) -> Tuple[Optional[List[STTrajectory]], float, bool]:
        self._verbose = verbose
        self._reset_run_state(queries, timeout_secs)
        num_agents = len(queries)

        graph = nx.DiGraph()
        graph.add_nodes_from(range(num_agents))
        root = PBSNode(graph)

        graph_size = getattr(self._stgcs_base, "graph_size_string", "unknown")
        self.print(f"\n-> PBS + Zeta*-SIPP: initial STGCS: {graph_size}")
        for agent_idx in range(num_agents):
            if not self._plan_agent(root, agent_idx, (), queries):
                self.print("\n-> PBS + Zeta*-SIPP: initial solution not found.")
                return None, time.perf_counter() - self._run_started_at, False

        stack = [(root, None)]
        while stack:
            node, updated_agent = stack.pop()
            self.num_popped_nodes += 1
            self.print(f"-> PBS + Zeta*-SIPP: Update plan for child {node}.")
            if updated_agent is not None:
                self.num_update_calls += 1
                if not self.update_node(node, updated_agent, queries):
                    continue

            ci, cj = self.find_first_conflict(num_agents, node)
            if ci is None:
                print(f"\nPBS + Zeta*-SIPP: Successfully found a valid set of plans: {node}")
                self.last_solution_prio_graph = node.prio_graph.copy()
                self.last_solution_order = tuple(
                    nx.lexicographical_topological_sort(node.prio_graph, key=lambda idx: idx)
                )
                return node.sols, time.perf_counter() - self._run_started_at, True
            if (ci, cj) in node.prio_graph.edges or (cj, ci) in node.prio_graph.edges:
                continue

            self.print(f"\n-> PBS + Zeta*-SIPP: Current node = {str(node)}")
            self.print(f"-> PBS + Zeta*-SIPP: Found conflict between {ci} and {cj}")

            if self.child_expansion_mode == ChildExpansionMode.LAZY:
                for i, j in self.partial_orders(ci, cj):
                    child = node.get_child(i, j)
                    if child is not None:
                        self.num_generated_children += 1
                        stack.append((child, j))
            else:
                child_entries = []
                for i, j in self.partial_orders(ci, cj):
                    child = node.get_child(i, j)
                    if child is None:
                        continue
                    self.num_generated_children += 1
                    self.num_update_calls += 1
                    if self.update_node(child, j, queries):
                        child_entries.append((
                            (
                                self._child_expansion_priority(
                                    parent=node,
                                    child=child,
                                    conflict_i=ci,
                                    conflict_j=cj,
                                    ordered_pair=(i, j),
                                ),
                                self._child_tie_break_value(num_agents, child),
                            ),
                            child,
                        ))

                child_entries.sort(key=lambda item: item[0])
                for _, child in reversed(child_entries):
                    stack.append((child, None))

            if time.perf_counter() - self._run_started_at > self._timeout_secs:
                self.print("\n-> PBS + Zeta*-SIPP: Timeout.")
                break

        self.print("\n-> PBS + Zeta*-SIPP: Failed to find a path for the robot.")
        return node.sols, time.perf_counter() - self._run_started_at, False

    def update_node(self, node: PBSNode, idx: int, queries: List[MPQuery]) -> bool:
        self._queries = tuple(queries)
        replan_list = set([idx] + [j for j in range(len(node.sols)) if nx.has_path(node.prio_graph, idx, j)])

        for agent_idx in nx.topological_sort(node.prio_graph.subgraph(replan_list)):
            high_priority_agents = self._high_priority_agents(node, agent_idx)
            replan = agent_idx == idx
            if not replan:
                replan = self._conflicts_with_node_agents(
                    node,
                    agent_idx,
                    high_priority_agents,
                    queries=queries,
                )

            if not replan:
                continue

            safe_radius = 2.0 * self.robot_radius
            reservation_key = self._reservation_cache_key(
                node,
                high_priority_agents,
                queries,
                reserve_tf=self.tmax,
                safe_radius=safe_radius,
            )
            plan_cache_key: LowLevelPlanCacheKey = self._low_level_plan_cache_key(
                agent_idx,
                queries[int(agent_idx)],
                reservation_key,
            )
            cached_plan = self._low_level_plan_cache.get(plan_cache_key)
            if cached_plan is not None:
                cached_sol, _, solution_cache_serial = cached_plan
                node.set_solution(agent_idx, cached_sol, cache_serial=solution_cache_serial)
                self.print(f"\tReuses cached Zeta*-SIPP plan for {agent_idx}.")
                continue

            if not self._plan_agent(node, int(agent_idx), high_priority_agents, queries):
                return False
            self._low_level_plan_cache[plan_cache_key] = (
                node.sols[int(agent_idx)],
                STPlanStatus.SUCCESS,
                node.solution_cache_serial(int(agent_idx)),
            )
        return True


def pbs_zeta_sipp(
    stgcs: STGCS,
    env: Env,
    queries: List[MPQuery],
    timeout_secs: float,
    child_expansion_mode: ChildExpansionMode = DEFAULT_CHILD_EXPANSION_MODE,
    seed: int = 0,
    cell_size: Optional[float] = None,
    runtime_limit_secs: float = math.inf,
    time_buffer: float = 1e-3,
    use_fov: bool = True,
    verbose: bool = True,
) -> Tuple[Optional[List[STTrajectory]], float, bool, ZetaSIPPPriorityBasedSearch]:
    planner = ZetaSIPPPriorityBasedSearch(
        stgcs,
        env,
        child_expansion_mode=child_expansion_mode,
        seed=seed,
        cell_size=cell_size,
        runtime_limit_secs=runtime_limit_secs,
        time_buffer=time_buffer,
        use_fov=use_fov,
    )
    solutions, runtime, success = planner.run(queries, timeout_secs, verbose=verbose)
    return solutions, runtime, success, planner
