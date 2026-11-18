from __future__ import annotations
from typing import Any, Dict, Iterable, List, Tuple, Optional, Sequence
from itertools import combinations
from enum import IntEnum

import time
import numpy as np
import networkx as nx

from experiments.environment.space_time_graph import STGCS
from stgcs_native.types import MPQuery, STPlanStatus, ChildExpansionMode
from stgcs_native.types import STTrajectory
from experiments.environment.reservation import reserve as ecd_reserve
from experiments.environment.collision import (
    CollisionGeometry,
    CollisionGeometryCacheEntry,
    CollisionPairCacheKey,
    cached_node_pair_collision,
    collision_checking as check_trajectory_collision,
)


ReservationAgentSignature = Tuple[int, int, bool, bool]
ReservationCacheKey = Tuple[Tuple[ReservationAgentSignature, ...], float, float, float]
QueryCacheKey = Tuple[Tuple[float, ...], Tuple[float, ...], float, bool, float]
PlannerCacheKey = Tuple[object, ...]
LowLevelPlanCacheKey = Tuple[int, QueryCacheKey, ReservationCacheKey, PlannerCacheKey]
LowLevelPlanCacheEntry = Tuple[STTrajectory, STPlanStatus, int]


class PBSNode:
    _next_cache_serial = 0
    _next_solution_cache_serial = 0
    
    def __init__(self, prio_graph:nx.DiGraph) -> None:
        self.prio_graph = prio_graph
        self.sols: List[STTrajectory] = []
        self._stgcs_num_edges: List[int] = [] # for debugging
        self._solution_versions: List[int] = []
        self._solution_cache_serials: List[int] = []
        self.cache_serial = PBSNode._next_cache_serial
        PBSNode._next_cache_serial += 1
    
    def __str__(self) -> str:
        edges = list(self.prio_graph.edges)
        if len(edges) <= 32:
            return f"({', '.join([f'{u}<{v}' for u, v in edges])})"
        preview = ", ".join(f"{u}<{v}" for u, v in edges[:16])
        return (
            f"({preview}, ...; "
            f"{self.prio_graph.number_of_nodes()} nodes, {len(edges)} edges)"
        )

    @classmethod
    def _new_solution_cache_serial(cls) -> int:
        cache_serial = cls._next_solution_cache_serial
        cls._next_solution_cache_serial += 1
        return cache_serial

    def _sync_solution_versions(self) -> None:
        while len(self._solution_versions) < len(self.sols):
            self._solution_versions.append(0)
        while len(self._solution_cache_serials) < len(self.sols):
            self._solution_cache_serials.append(self._new_solution_cache_serial())

    def append_solution(self, sol:STTrajectory) -> None:
        self.sols.append(sol)
        self._solution_versions.append(0)
        self._solution_cache_serials.append(self._new_solution_cache_serial())

    def set_solution(self, idx:int, sol:STTrajectory, cache_serial:Optional[int]=None) -> None:
        self._sync_solution_versions()
        self.sols[idx] = sol
        self._solution_versions[idx] += 1
        self._solution_cache_serials[idx] = (
            self._new_solution_cache_serial()
            if cache_serial is None
            else int(cache_serial)
        )

    def solution_version(self, idx:int) -> int:
        self._sync_solution_versions()
        return self._solution_versions[idx]

    def solution_cache_serial(self, idx:int) -> int:
        self._sync_solution_versions()
        return self._solution_cache_serials[idx]
  
    def get_child(self, i:int, j:int) -> Optional[PBSNode]:
        G = self.prio_graph.copy()
        G.add_edge(i, j)
        if nx.is_directed_acyclic_graph(G):
            self._sync_solution_versions()
            child = PBSNode(G)
            child.sols = list(self.sols)
            child._solution_versions = list(self._solution_versions)
            child._solution_cache_serials = list(self._solution_cache_serials)
            return child

        return None




DEFAULT_CHILD_EXPANSION_MODE = ChildExpansionMode.NUM_CONFLICTS


class PriorityBasedSearch:
    CHILD_CONFLICT_COUNT_LIMIT = 64
    
    def __init__(
        self,
        stgcs:STGCS,
        st_planner,
        robot_radius:float,
        child_expansion_mode:ChildExpansionMode = DEFAULT_CHILD_EXPANSION_MODE,
    ) -> None:
        self._stgcs_base = stgcs
        self.st_planner = st_planner
        self.robot_radius = robot_radius
        self.child_expansion_mode = child_expansion_mode
        self.t0, self.tmax = stgcs.t0, stgcs.tmax
        self.last_solution_prio_graph: Optional[nx.DiGraph] = None
        self.last_solution_order: Tuple[int, ...] = tuple()
        self._queries: Tuple[MPQuery, ...] = tuple()
        self._profiler: Dict[str, np.ndarray] = {
                "mp": np.zeros(2, dtype=float),
                "gcs": np.zeros(2, dtype=float),
                "gub": np.zeros(2, dtype=float),
                "search": np.zeros(2, dtype=float),
                "cr": np.zeros(2, dtype=float),
                "dc": np.zeros(2, dtype=float),
                "dc_cr": np.zeros(2, dtype=float),
                "cc": np.zeros(2, dtype=float),
                "ecd": np.zeros(2, dtype=float),
            } # (count, total_time)
        self._collision_geometry_cache: Dict[Tuple[Any, ...], CollisionGeometryCacheEntry] = {}
        self._reserved_stgcs_cache: Dict[ReservationCacheKey, STGCS] = {}
        self._low_level_plan_cache: Dict[LowLevelPlanCacheKey, LowLevelPlanCacheEntry] = {}
        self._structural_failure_cache: set[LowLevelPlanCacheKey] = set()
        self._ecd_pair_cache: Dict[Any, object] = {}
        self._collision_pair_cache: Dict[CollisionPairCacheKey, bool] = {}
        self.num_popped_nodes = 0
        self.num_generated_children = 0
        self.num_update_calls = 0

    def run(
        self,
        queries:List[MPQuery],
        timeout_secs:float,
        verbose:bool=True,
    ) -> Tuple[Optional[List[STTrajectory]], float, bool]:
        self._verbose = verbose
        ts = time.perf_counter()
        num_agents = len(queries)
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

        G = nx.DiGraph()
        G.add_nodes_from(range(num_agents))
        root = PBSNode(G)
        
        self.print(f"\n-> PBS: initial STGCS: {self._stgcs_base.graph_size_string}")
        for query in queries:
            sol, runtime, status = self.st_planner.plan(self._stgcs_base, query)
            self._record_low_level_profile(runtime)
            if status == STPlanStatus.FAIL or sol is None:
                self.print("\n-> PBS:intial solution not found")
                return None, time.perf_counter() - ts, False
            root.append_solution(sol)
            root._stgcs_num_edges.append(self._stgcs_base.G.number_of_edges())

        stack = [(root, None)]
        while stack != []:
            node, j = stack.pop()
            self.num_popped_nodes += 1
            self.print(f"-> PBS: Update plan for child {node}.")
            if j is not None:
                self.num_update_calls += 1
                if not self.update_node(node, j, queries):
                    continue

            ci, cj = self.find_first_conflict(num_agents, node)
            if ci is None:
                print(f"\nPBS: Successfully found a valid set of plans: {node}")
                self.last_solution_prio_graph = node.prio_graph.copy()
                self.last_solution_order = tuple(
                    nx.lexicographical_topological_sort(
                        node.prio_graph, key=lambda idx: idx
                    )
                )
                return node.sols, time.perf_counter() - ts, True
            elif ((ci, cj) in node.prio_graph.edges or (cj, ci) in node.prio_graph.edges):
                continue
            
            self.print(f"\n-> PBS: Current node = {str(node)}")
            self.print(f"-> PBS: Found conflict between {ci} and {cj}")

            if self.child_expansion_mode == ChildExpansionMode.LAZY:
                for i, j in self.partial_orders(ci, cj):
                    child = node.get_child(i, j)
                    if child:
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
                        child_rank = (
                            self._child_expansion_priority(
                                parent=node,
                                child=child,
                                conflict_i=ci,
                                conflict_j=cj,
                                ordered_pair=(i, j),
                            ),
                            self._child_tie_break_value(num_agents, child),
                        )
                        child_entries.append((
                            child_rank,
                            child,
                        ))

                # Use reverse push order so the best-ranked child is popped first (LIFO stack).
                child_entries.sort(key=lambda x: x[0])
                for _, child in reversed(child_entries):
                    # use None for the second element to indicate that the child has already been updated, 
                    # so it won't be updated again when popped. 
                    stack.append((child, None))

            if time.perf_counter() - ts > timeout_secs:
                self.print(f"\n-> PBS: Timeout.")
                break
            
        self.print(f"\n-> PBS: Failed to find a path for the robot.")
        return node.sols, time.perf_counter() - ts, False

    def update_node(self, node:PBSNode, idx:int, queries:List[MPQuery]) -> bool:
        self._queries = tuple(queries)
        replan_list = set([idx] + [j for j in range(len(node.sols)) if nx.has_path(node.prio_graph, idx, j)])
        
        for j in nx.topological_sort(node.prio_graph.subgraph(replan_list)):
            high_priority_agents = self._high_priority_agents(node, j)
            replan = j == idx
            
            if not replan:
                replan = self._conflicts_with_node_agents(
                    node,
                    j,
                    high_priority_agents,
                    queries=queries,
                )
            
            if replan:
                reserve_tf = self.tmax
                safe_radius = 2 * self.robot_radius
                reservation_key = self._reservation_cache_key(
                    node,
                    high_priority_agents,
                    queries,
                    reserve_tf,
                    safe_radius,
                )
                plan_cache_key = self._low_level_plan_cache_key(j, queries[j], reservation_key)
                cached_plan = self._low_level_plan_cache.get(plan_cache_key)
                if cached_plan is not None:
                    cached_sol, _, solution_cache_serial = cached_plan
                    node.set_solution(j, cached_sol, cache_serial=solution_cache_serial)
                    self.print(f"\t\u2713 Reuses cached plan for {j}.")
                    continue
                if plan_cache_key in self._structural_failure_cache:
                    self.print(f"\t\u2713 Reuses cached structural failure for {j}.")
                    return False

                stgcs_reserved, ecd_calls, ecd_runtime = self._reserved_stgcs_for_node(
                    node,
                    high_priority_agents,
                    queries,
                    reserve_tf=reserve_tf,
                    reservation_key=reservation_key,
                )
                self._profiler["ecd"] += np.array([ecd_calls, ecd_runtime])

                sol, runtime, status = self.st_planner.plan(stgcs_reserved, queries[j])
                self._record_low_level_profile(runtime)

                if status == STPlanStatus.FAIL or sol is None:
                    self.print(f"\t\u2713 Fails to update plan for {j}: {stgcs_reserved.graph_size_string}")
                    return False
                node.set_solution(j, sol)
                self._low_level_plan_cache[plan_cache_key] = (
                    sol,
                    status,
                    node.solution_cache_serial(j),
                )
                node._stgcs_num_edges.append(stgcs_reserved.G.number_of_edges())
                self.print(f"\t\u2713 Succeeds to update plan for {j}: {stgcs_reserved.graph_size_string}")

        return True

    def _high_priority_agents(self, node: PBSNode, agent_idx: int) -> List[int]:
        return sorted(int(idx) for idx in node.prio_graph.predecessors(agent_idx))

    def _record_low_level_profile(self, runtime:float) -> None:
        self._profiler["mp"] += np.array([1, runtime])
        profile = getattr(self.st_planner, "last_profile", None)
        if not profile:
            return
        self._profiler["gcs"] += np.array([1, float(profile.get("gcs_construction_time", 0.0))])
        self._profiler["gub"] += np.array([1, float(profile.get("gub_time", 0.0))])
        self._profiler["search"] += np.array([1, float(profile.get("main_search_time", 0.0))])
        self._profiler["cr"] += np.array([
            float(profile.get("convex_restriction_calls", 0.0)),
            float(profile.get("convex_restriction_time", 0.0)),
        ])
        self._profiler["dc"] += np.array([1, float(profile.get("dominance_check_time", 0.0))])
        self._profiler["dc_cr"] += np.array([
            float(profile.get("dominance_convex_restriction_calls", 0.0)),
            float(profile.get("dominance_convex_restriction_time", 0.0)),
        ])

    def partial_orders(self, i:int, j:int) -> List[Tuple[int, int]]:
        return [(i, j), (j, i)]

    def _child_expansion_priority(
        self,
        parent: PBSNode,
        child: PBSNode,
        conflict_i: int,
        conflict_j: int,
        ordered_pair: Tuple[int, int],
    ) -> int:
        """Subclass hook to bias child ranking ahead of mode-specific scores.

        Lower values are preferred. The default preserves the original PBS
        behavior by applying no extra bias.
        """
        return 0

    @staticmethod
    def _query_cache_key(query: MPQuery) -> QueryCacheKey:
        return (
            tuple(float(value) for value in np.asarray(query.start, dtype=float).ravel()),
            tuple(float(value) for value in np.asarray(query.goal, dtype=float).ravel()),
            float(query.t_start),
            bool(query.is_stay),
            float(query.vlimit),
        )

    def _planner_cache_key(self) -> PlannerCacheKey:
        dc_types = tuple(
            f"{type(dc).__module__}.{type(dc).__qualname__}"
            for dc in getattr(self.st_planner, "dc_list", [])
        )
        heur = getattr(self.st_planner, "heur", None)
        return (
            id(self.st_planner),
            f"{type(self.st_planner).__module__}.{type(self.st_planner).__qualname__}",
            float(getattr(self.st_planner, "runtime_limit_secs", float("inf"))),
            float(getattr(self.st_planner, "eps", 1.0)),
            bool(getattr(self.st_planner, "enable_gub_fallback", False)),
            f"{type(heur).__module__}.{type(heur).__qualname__}" if heur is not None else None,
            dc_types,
        )

    def _low_level_plan_cache_key(
        self,
        agent_idx:int,
        query: MPQuery,
        reservation_key: ReservationCacheKey,
    ) -> LowLevelPlanCacheKey:
        return (
            int(agent_idx),
            self._query_cache_key(query),
            reservation_key,
            self._planner_cache_key(),
        )

    def _reservation_stay_modes(self, agent_idx:int, queries:List[MPQuery]) -> Tuple[bool, bool]:
        return True, bool(queries[agent_idx].is_stay)

    def _reservation_agent_signature(
        self,
        node:PBSNode,
        agent_idx:int,
        queries:List[MPQuery],
    ) -> ReservationAgentSignature:
        x0_staying, xt_staying = self._reservation_stay_modes(agent_idx, queries)
        return (
            int(agent_idx),
            node.solution_cache_serial(agent_idx),
            bool(x0_staying),
            bool(xt_staying),
        )

    def _reservation_cache_key(
        self,
        node:PBSNode,
        high_priority_agents: Sequence[int],
        queries:List[MPQuery],
        reserve_tf: float,
        safe_radius: float,
    ) -> ReservationCacheKey:
        return (
            tuple(
                self._reservation_agent_signature(node, int(agent_idx), queries)
                for agent_idx in high_priority_agents
            ),
            float(self.t0),
            float(reserve_tf),
            float(safe_radius),
        )

    @staticmethod
    def _reservation_prefix_cache_key(
        reservation_key: ReservationCacheKey,
        prefix_len: int,
    ) -> ReservationCacheKey:
        agent_signatures, t0, reserve_tf, safe_radius = reservation_key
        return (
            agent_signatures[:prefix_len],
            t0,
            reserve_tf,
            safe_radius,
        )

    def _reservation_trajectory_points(
        self,
        node:PBSNode,
        agent_idx:int,
        queries:List[MPQuery],
        reserve_tf: float,
    ) -> List[np.ndarray]:
        del queries, reserve_tf
        return node.sols[agent_idx].points

    def _reserved_stgcs_for_node(
        self,
        node:PBSNode,
        high_priority_agents: Sequence[int],
        queries:List[MPQuery],
        reserve_tf: float,
        reservation_key: Optional[ReservationCacheKey]=None,
    ) -> Tuple[STGCS, int, float]:
        ts = time.perf_counter()
        safe_radius = 2 * self.robot_radius
        agents = tuple(int(agent_idx) for agent_idx in high_priority_agents)
        key = (
            self._reservation_cache_key(node, agents, queries, reserve_tf, safe_radius)
            if reservation_key is None
            else reservation_key
        )
        cached_stgcs = self._reserved_stgcs_cache.get(key)
        if cached_stgcs is not None:
            return cached_stgcs.copy(), 0, time.perf_counter() - ts

        stgcs_reserved: Optional[STGCS] = None
        start_idx = 0
        for prefix_len in range(len(agents) - 1, 0, -1):
            prefix_key = self._reservation_prefix_cache_key(key, prefix_len)
            cached_prefix = self._reserved_stgcs_cache.get(prefix_key)
            if cached_prefix is not None:
                stgcs_reserved = cached_prefix.copy()
                start_idx = prefix_len
                break

        if stgcs_reserved is None:
            stgcs_reserved = self._stgcs_base.copy()

        ecd_calls = 0
        for offset, agent_idx in enumerate(agents[start_idx:], start=start_idx + 1):
            x0_staying, xt_staying = self._reservation_stay_modes(agent_idx, queries)
            stgcs_reserved = ecd_reserve(
                stgcs_reserved,
                self._reservation_trajectory_points(node, agent_idx, queries, reserve_tf),
                safe_radius,
                x0_staying=x0_staying,
                xt_staying=xt_staying,
                ecd_pair_cache=self._ecd_pair_cache,
            )
            ecd_calls += 1
            self._reserved_stgcs_cache[
                self._reservation_prefix_cache_key(key, offset)
            ] = stgcs_reserved.copy()

        if len(agents) == 0:
            self._reserved_stgcs_cache[key] = stgcs_reserved.copy()
        return stgcs_reserved, ecd_calls, time.perf_counter() - ts

    def _goal_stay_enabled(self, agent_idx:int) -> bool:
        if 0 <= agent_idx < len(self._queries):
            return bool(self._queries[agent_idx].is_stay)
        return True

    def _agent_pair_conflict(
        self,
        agent_a:int,
        pi_a:List[np.ndarray],
        agent_b:int,
        pi_b:List[np.ndarray],
        queries: Optional[List[MPQuery]]=None,
        cc_to_t0:bool=True,
        tolerance:float=1e-4,
    ) -> bool:
        if queries is None:
            goal_stay_a = self._goal_stay_enabled(agent_a)
            goal_stay_b = self._goal_stay_enabled(agent_b)
        else:
            goal_stay_a = bool(queries[agent_a].is_stay)
            goal_stay_b = bool(queries[agent_b].is_stay)

        return self.collision_checking(
            pi_a,
            pi_b,
            cc_to_t0=cc_to_t0,
            cc_to_tf=False,
            tolerance=tolerance,
            cc_to_tf_a=goal_stay_a,
            cc_to_tf_b=goal_stay_b,
        )

    def _child_tie_break_value(self, num_agents:int, node:PBSNode) -> float:
        if self.child_expansion_mode == ChildExpansionMode.SOC:
            return sum(sol.duration for sol in node.sols)
        if self.child_expansion_mode == ChildExpansionMode.MAKESPAN:
            return max(sol.duration for sol in node.sols)
        if self.child_expansion_mode == ChildExpansionMode.NUM_CONFLICTS:
            return float(
                self.number_of_conflicts(
                    num_agents,
                    node,
                    max_count=self.CHILD_CONFLICT_COUNT_LIMIT,
                )
            )
        raise RuntimeError(
            f"_child_tie_break_value called for unsupported mode {self.child_expansion_mode!r}"
        )

    def number_of_conflicts(
        self,
        num_agents:int,
        node:PBSNode,
        max_count: Optional[int]=None,
    ) -> int:
        _, num_conflicts = self._scan_node_pair_conflicts(
            node,
            agent_indices=range(num_agents),
            pair_order=combinations(range(num_agents), 2),
            stop_at_first=False,
            max_conflicts=max_count,
        )
        return num_conflicts

    def find_first_conflict(self, num_agents:int, node:PBSNode) -> Tuple[Optional[int], Optional[int]]:
        first_conflict, _ = self._scan_node_pair_conflicts(
            node,
            agent_indices=range(num_agents),
            pair_order=combinations(range(num_agents), 2),
            stop_at_first=True,
        )
        if first_conflict is None:
            return None, None
        return first_conflict

    def collision_checking(
        self, pi_a:List[np.ndarray], pi_b:List[np.ndarray], 
        cc_to_t0:bool=True, cc_to_tf:bool=True, tolerance:float=1e-4,
        cc_to_t0_a:Optional[bool]=None, cc_to_t0_b:Optional[bool]=None,
        cc_to_tf_a:Optional[bool]=None, cc_to_tf_b:Optional[bool]=None,
    ) -> bool:
        return check_trajectory_collision(
            pi_a,
            pi_b,
            space_dim=self._stgcs_base.dimension,
            robot_radius=self.robot_radius,
            t0=self.t0,
            tmax=self.tmax,
            profiler=self._profiler,
            collision_geometry_cache=self._collision_geometry_cache,
            cc_to_t0=cc_to_t0,
            cc_to_tf=cc_to_tf,
            tolerance=tolerance,
            cc_to_t0_a=cc_to_t0_a,
            cc_to_t0_b=cc_to_t0_b,
            cc_to_tf_a=cc_to_tf_a,
            cc_to_tf_b=cc_to_tf_b,
        )

    def _goal_stay_by_agent(
        self,
        agent_indices: Sequence[int],
        queries: Optional[List[MPQuery]]=None,
    ) -> Dict[int, bool]:
        if queries is None:
            return {int(agent_idx): self._goal_stay_enabled(int(agent_idx)) for agent_idx in agent_indices}
        return {int(agent_idx): bool(queries[int(agent_idx)].is_stay) for agent_idx in agent_indices}

    def _conflicts_with_node_agents(
        self,
        node: PBSNode,
        agent_idx: int,
        other_agents: Sequence[int],
        queries: Optional[List[MPQuery]]=None,
        cc_to_t0: bool=True,
        tolerance: float=1e-4,
    ) -> bool:
        if not other_agents:
            return False
        first_conflict, _ = self._scan_node_pair_conflicts(
            node,
            agent_indices=tuple(other_agents) + (agent_idx,),
            pair_order=((other_idx, agent_idx) for other_idx in other_agents),
            queries=queries,
            cc_to_t0=cc_to_t0,
            tolerance=tolerance,
            stop_at_first=True,
        )
        return first_conflict is not None

    def _scan_node_pair_conflicts(
        self,
        node: PBSNode,
        agent_indices: Iterable[int],
        pair_order: Iterable[Tuple[int, int]],
        queries: Optional[List[MPQuery]]=None,
        points_by_agent: Optional[Dict[int, List[np.ndarray]]]=None,
        goal_stay_by_agent: Optional[Dict[int, bool]]=None,
        cc_to_t0: bool=True,
        tolerance: float=1e-4,
        stop_at_first: bool=True,
        max_conflicts: Optional[int]=None,
    ) -> Tuple[Optional[Tuple[int, int]], int]:
        agent_indices = tuple(int(agent_idx) for agent_idx in agent_indices)
        if goal_stay_by_agent is None:
            goal_stay_by_agent = self._goal_stay_by_agent(agent_indices, queries)
        if points_by_agent is None:
            points_by_agent = {
                agent_idx: node.sols[agent_idx].points
                for agent_idx in agent_indices
            }

        geometry_cache: Dict[int, CollisionGeometry] = {}

        num_conflicts = 0
        for i, j in pair_order:
            colliding = cached_node_pair_collision(
                node,
                int(i),
                int(j),
                points_by_agent,
                goal_stay_by_agent,
                geometry_cache,
                space_dim=self._stgcs_base.dimension,
                robot_radius=self.robot_radius,
                t0=self.t0,
                tmax=self.tmax,
                profiler=self._profiler,
                collision_geometry_cache=self._collision_geometry_cache,
                collision_pair_cache=self._collision_pair_cache,
                cc_to_t0=cc_to_t0,
                tolerance=tolerance,
            )
            if colliding:
                if stop_at_first:
                    return (int(i), int(j)), 1
                num_conflicts += 1
                if max_conflicts is not None and num_conflicts >= max_conflicts:
                    return None, num_conflicts

        return None, num_conflicts

    def print(self, msg:str) -> None:
        if self._verbose:
            print(msg)
