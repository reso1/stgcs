"""Build MRMP instance manifests and validate their queries."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from experiments.base import BaseInstanceFactory
from experiments.manifests.base import BaseBenchmarkRecord
from experiments.native import STGCSPlannerClient
from experiments.manifests.mrmp import MRMPBenchmarkRecord
from experiments.manifests.mrmp import QuerySpec
from experiments.planners.mrmp import MRMPPerformanceComparison, PBSExpansionAblation
from stgcs_native.types import MRMPQuery
from baselines.priority_search import PriorityBasedSearch
from stgcs_native.types import MPQuery
from stgcs_native.types import STTrajectory


RecordCallback = Callable[[MRMPBenchmarkRecord, Sequence[MRMPBenchmarkRecord]], None]
StatusCallback = Callable[[str, Dict[str, Any]], None]


@dataclass(frozen=True)
class ConflictStats:
    num_conflicting_pairs: int
    num_conflicting_agents: int
    largest_component: int
    density: float


class MRMPManifestBuilder:
    """Current intersection-query policy, independent-path validation, and conflict statistics."""
    MAX_QUERY_ATTEMPTS = 300
    MAX_RECORD_ATTEMPTS = 24
    MAX_PAIR_ATTEMPTS = 40
    MIN_POINT_SEPARATION_SCALE = 2.0
    MIN_POINT_SEPARATION_DIAG_RATIO = 0.02
    ENDPOINT_SEPARATION_SCALE_BY_DIM = {2: 2.85, 3: 3.5}
    INDEPENDENT_REFERENCE_BUDGET = {"grid2d": 5.0, "grid3d": 15.0, "iris-2d": 5.0, "maze": 5.0}
    MAX_CONFLICT_DENSITY_BY_DOMAIN: Dict[str, float] = {}
    NATIVE_BACKEND = None

    @classmethod
    def native_backend(cls):
        if cls.NATIVE_BACKEND is None:
            cls.NATIVE_BACKEND = STGCSPlannerClient(
                STGCSPlannerClient.ROOT / "stgcs_cplusplus/build/bin/stgcs_planner",
                STGCSPlannerClient.ROOT / "build/mrmp-manifest")
        return cls.NATIVE_BACKEND

    @classmethod
    def independent_reference_budget(cls, domain_key: str) -> float:
        return float(cls.INDEPENDENT_REFERENCE_BUDGET[domain_key])

    @staticmethod
    def _emit_status(
        on_status: StatusCallback | None,
        event: str,
        **payload: Any,
    ) -> None:
        if on_status is not None:
            on_status(event, payload)

    @classmethod
    def _point_separation(cls, env) -> float:
        bbox_diag = float(np.linalg.norm(env.ub - env.lb))
        return max(cls.MIN_POINT_SEPARATION_SCALE * env.robot_radius, cls.MIN_POINT_SEPARATION_DIAG_RATIO * bbox_diag)

    @classmethod
    def _endpoint_separation_scale(cls, env) -> float:
        dim = int(env.dim)
        if dim not in cls.ENDPOINT_SEPARATION_SCALE_BY_DIM:
            raise ValueError(f"Unsupported endpoint-separation dimension {dim!r}.")
        return float(cls.ENDPOINT_SEPARATION_SCALE_BY_DIM[dim])

    @classmethod
    def _endpoint_separation_tol(cls, env) -> float:
        return cls._endpoint_separation_scale(env) * float(env.robot_radius)

    @staticmethod
    def _rect_bounds(rect: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.min(rect, axis=0), np.max(rect, axis=0)

    @classmethod
    def _rect_sample_bounds(
        cls,
        env,
        rect_idx: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        rect_lb, rect_ub = cls._rect_bounds(np.asarray(env.C_Space[rect_idx], dtype=float))
        sample_lb = rect_lb.copy()
        sample_ub = rect_ub.copy()
        if np.any(sample_ub <= sample_lb):
            return None
        return sample_lb, sample_ub

    @staticmethod
    def _box_measure(bounds: Tuple[np.ndarray, np.ndarray]) -> float:
        lb, ub = bounds
        if np.any(ub <= lb):
            return 0.0
        return float(np.prod(ub - lb))

    @classmethod
    def _overlapping_exclusion_boxes(
        cls,
        bounds: Tuple[np.ndarray, np.ndarray],
        existing_points: Sequence[np.ndarray],
        exclusion_tol: float,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        if exclusion_tol <= 0.0 or len(existing_points) == 0:
            return []

        bounds_lb, bounds_ub = bounds
        overlap_boxes: List[Tuple[np.ndarray, np.ndarray]] = []
        for point in existing_points:
            point_arr = np.asarray(point, dtype=float)
            exc_lb = point_arr - exclusion_tol
            exc_ub = point_arr + exclusion_tol
            if np.any(exc_ub <= bounds_lb) or np.any(bounds_ub <= exc_lb):
                continue
            overlap_boxes.append((exc_lb, exc_ub))
        return overlap_boxes

    @classmethod
    def _subtract_box(
        cls,
        bounds: Tuple[np.ndarray, np.ndarray],
        exclusion: Tuple[np.ndarray, np.ndarray],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        bounds_lb, bounds_ub = bounds
        exc_lb, exc_ub = exclusion
        overlap_lb = np.maximum(bounds_lb, exc_lb)
        overlap_ub = np.minimum(bounds_ub, exc_ub)
        if np.any(overlap_ub <= overlap_lb):
            return [(bounds_lb.copy(), bounds_ub.copy())]

        pieces: List[Tuple[np.ndarray, np.ndarray]] = []
        core_lb = bounds_lb.copy()
        core_ub = bounds_ub.copy()
        for axis in range(bounds_lb.shape[0]):
            if core_lb[axis] < overlap_lb[axis]:
                piece_lb = core_lb.copy()
                piece_ub = core_ub.copy()
                piece_ub[axis] = overlap_lb[axis]
                pieces.append((piece_lb, piece_ub))
                core_lb[axis] = overlap_lb[axis]
            if overlap_ub[axis] < core_ub[axis]:
                piece_lb = core_lb.copy()
                piece_ub = core_ub.copy()
                piece_lb[axis] = overlap_ub[axis]
                pieces.append((piece_lb, piece_ub))
                core_ub[axis] = overlap_ub[axis]

        return [piece for piece in pieces if cls._box_measure(piece) > 0.0]

    @classmethod
    def _subtract_exclusion_boxes(
        cls,
        bounds: Tuple[np.ndarray, np.ndarray],
        exclusion_boxes: Sequence[Tuple[np.ndarray, np.ndarray]],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        remaining = [(bounds[0].copy(), bounds[1].copy())]
        for exclusion in exclusion_boxes:
            next_remaining: List[Tuple[np.ndarray, np.ndarray]] = []
            for candidate in remaining:
                next_remaining.extend(cls._subtract_box(candidate, exclusion))
            remaining = next_remaining
            if len(remaining) == 0:
                break
        return remaining

    @classmethod
    def _sample_carved_rect_point(
        cls,
        env,
        rect_indices: Sequence[int],
        rng: np.random.RandomState,
        existing_points: Sequence[np.ndarray],
        exclusion_tol: float,
    ) -> Optional[np.ndarray]:
        remaining_boxes: List[Tuple[np.ndarray, np.ndarray]] = []
        for rect_idx in rect_indices:
            bounds = cls._rect_sample_bounds(env, int(rect_idx))
            if bounds is None:
                continue
            exclusion_boxes = cls._overlapping_exclusion_boxes(bounds, existing_points, exclusion_tol)
            remaining_boxes.extend(cls._subtract_exclusion_boxes(bounds, exclusion_boxes))

        if len(remaining_boxes) == 0:
            return None

        weights = np.asarray([cls._box_measure(box) for box in remaining_boxes], dtype=float)
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            return None
        box_idx = int(rng.choice(len(remaining_boxes), p=weights / total_weight))
        sample_lb, sample_ub = remaining_boxes[box_idx]
        return rng.uniform(sample_lb, sample_ub)

    @staticmethod
    def _query(start: np.ndarray, goal: np.ndarray, vlimit: float) -> MPQuery:
        return MPQuery(start=start, goal=goal, t_start=0.0, is_stay=True, vlimit=vlimit)

    @classmethod
    def _validate_queries_quick(
        cls,
        instance,
        queries: Sequence[MPQuery],
    ) -> bool:
        for query in queries:
            is_valid, _ = instance.stgcs.validate_query(query)
            if not is_valid:
                return False
        return True

    @staticmethod
    def _pairwise_min_distance(points: Sequence[np.ndarray]) -> float:
        if len(points) < 2:
            return float("inf")
        min_dist = float("inf")
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                min_dist = min(min_dist, float(np.linalg.norm(points[i] - points[j])))
        return min_dist

    @classmethod
    def _satisfies_endpoint_separation(
        cls,
        env,
        queries: Sequence[MPQuery],
    ) -> bool:
        min_endpoint_separation = cls._endpoint_separation_tol(env)
        endpoints = [query.start for query in queries] + [query.goal for query in queries]
        return cls._pairwise_min_distance(endpoints) >= min_endpoint_separation

    @classmethod
    def _conflict_stats(
        cls,
        instance,
        sols: Sequence[object],
    ) -> ConflictStats:
        conflict_checker = PriorityBasedSearch(instance.stgcs, None, instance.env.robot_radius)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(sols)))
        for i in range(len(sols)):
            for j in range(i + 1, len(sols)):
                if conflict_checker.collision_checking(sols[i].points, sols[j].points):
                    graph.add_edge(i, j)

        num_pairs = graph.number_of_edges()
        conflicting_nodes = [node for node in graph.nodes if graph.degree[node] > 0]
        conflicting_agents = len(conflicting_nodes)
        if conflicting_nodes:
            conflict_graph = graph.subgraph(conflicting_nodes)
            largest_component = max((len(component) for component in nx.connected_components(conflict_graph)), default=0)
        else:
            largest_component = 0
        density = 0.0
        if len(sols) >= 2:
            density = (2.0 * num_pairs) / (len(sols) * (len(sols) - 1))
        return ConflictStats(
            num_conflicting_pairs=num_pairs,
            num_conflicting_agents=conflicting_agents,
            largest_component=largest_component,
            density=density,
        )

    @classmethod
    def max_conflict_density(cls, domain_key: str) -> Optional[float]:
        return cls.MAX_CONFLICT_DENSITY_BY_DOMAIN.get(str(domain_key))

    @classmethod
    def conflict_density_allowed(cls, domain_key: str, density: float) -> bool:
        max_density = cls.max_conflict_density(domain_key)
        if max_density is None:
            return True
        return float(density) <= float(max_density)

    @classmethod
    def conflict_density_selection_enabled(cls, domain_key: str) -> bool:
        return False

    @classmethod
    def conflict_density_selection_pool_count(
        cls,
        domain_key: str,
        target_count: int,
        available_count: int,
    ) -> int:
        del domain_key
        return min(int(target_count), int(available_count))

    @classmethod
    def select_records_for_domain(
        cls,
        domain_key: str,
        records: Sequence[MRMPBenchmarkRecord],
        target_count: int,
    ) -> List[MRMPBenchmarkRecord]:
        del domain_key
        return list(records)[: int(target_count)]

    @classmethod
    def _sample_random_query(cls, env, rng, existing_points, sep_tol,
                             min_start_goal_dist, vlimit, direct_exclusion_tol):
        rect_indices = tuple(range(len(env.C_Space)))
        for _ in range(cls.MAX_PAIR_ATTEMPTS):
            start = cls._sample_carved_rect_point(env, rect_indices, rng, existing_points, direct_exclusion_tol)
            if start is None:
                continue
            goal = cls._sample_carved_rect_point(env, rect_indices, rng, list(existing_points) + [start], direct_exclusion_tol)
            if goal is None or np.linalg.norm(start - goal) < min_start_goal_dist:
                continue
            return cls._query(start, goal, vlimit)
        return None

    @classmethod
    def _sample_valid_random_queries(
        cls,
        base_instance,
        rng: np.random.RandomState,
        num_agents: int,
        sep_tol: float,
        min_start_goal_dist: float,
        direct_exclusion_tol: float,
    ) -> Optional[List[MPQuery]]:
        env = base_instance.env
        existing_points: List[np.ndarray] = []
        queries: List[MPQuery] = []
        for _agent_idx in range(num_agents):
            query = None
            for _ in range(cls.MAX_QUERY_ATTEMPTS):
                candidate = cls._sample_random_query(
                    env=env,
                    rng=rng,
                    existing_points=existing_points,
                    sep_tol=sep_tol,
                    min_start_goal_dist=min_start_goal_dist,
                    vlimit=base_instance.stgcs.vlimit,
                    direct_exclusion_tol=direct_exclusion_tol,
                )
                if candidate is None:
                    continue
                if cls._validate_queries_quick(base_instance, [candidate]):
                    query = candidate
                    break
            if query is None:
                return None
            existing_points.extend([query.start, query.goal])
            queries.append(query)
        return queries

    @classmethod
    def synthesize_record(cls, base_instance, base_record, num_agents, rng, instance_id,
                          min_conflicting_pairs=0, on_status=None):
        if min_conflicting_pairs < 0:
            raise ValueError("min_conflicting_pairs must be non-negative")
        domain = base_record.domain_key
        reference_budget = cls.independent_reference_budget(domain)
        for attempt in range(cls.MAX_RECORD_ATTEMPTS):
            cls._emit_status(on_status, "synthesize_attempt", num_agents=num_agents,
                             spatial_seed=base_instance.seed, attempt=attempt + 1,
                             max_attempts=cls.MAX_RECORD_ATTEMPTS)
            queries = cls.sample_queries(base_instance, num_agents, rng)
            if queries is None or len(queries) != num_agents:
                cls._emit_status(on_status, "reject", reason="sample")
                continue
            if not cls._satisfies_endpoint_separation(base_instance.env, queries):
                cls._emit_status(on_status, "reject", reason="endpoint_separation")
                continue
            cls._emit_status(on_status, "independent_plans")
            sols, metrics = cls._solve_independent_queries(base_instance, queries, reference_budget)
            if sols is None or metrics is None:
                cls._emit_status(on_status, "reject", reason="solve")
                continue
            cls._emit_status(on_status, "conflict_screen")
            conflicts = cls._conflict_stats(base_instance, sols)
            if conflicts.num_conflicting_pairs < min_conflicting_pairs:
                cls._emit_status(on_status, "reject", reason="conflict_count",
                                 num_conflicting_pairs=conflicts.num_conflicting_pairs,
                                 min_conflicting_pairs=min_conflicting_pairs)
                continue
            maximum = cls.max_conflict_density(domain)
            if maximum is not None and conflicts.density > maximum:
                cls._emit_status(on_status, "reject", reason="conflict_density",
                                 conflict_density=conflicts.density, max_conflict_density=maximum)
                continue
            total_cost, makespan, runtime = metrics
            return MRMPBenchmarkRecord(
                instance_id=instance_id, base_instance_id=base_record.instance_id,
                queries=[QuerySpec.from_query(q) for q in queries], num_agents=num_agents,
                independent_cost_sum=total_cost, independent_makespan=makespan,
                independent_runtime=runtime, num_conflicting_pairs=conflicts.num_conflicting_pairs,
                num_conflicting_agents=conflicts.num_conflicting_agents,
                conflict_largest_component=conflicts.largest_component, conflict_density=conflicts.density,
                stgcs_num_vertices=base_instance.stgcs.G.number_of_nodes(),
                stgcs_num_edges=base_instance.stgcs.G.number_of_edges(), tmax=base_instance.stgcs.tmax)
        raise RuntimeError(f"Failed to synthesize MRMP record for seed={base_instance.seed}, robots={num_agents}.")


class PBSExpansionMRMPManifestBuilder(MRMPManifestBuilder):
    NUM_AGENTS = 10
    MIN_START_GOAL_DIST_RATIO = 0.35
    MIN_TRAJECTORY_INTERSECTING_PAIRS = 5

    @classmethod
    def record_matches_benchmark(cls, record):
        return record.num_agents == len(record.queries) == cls.NUM_AGENTS

    @classmethod
    def sample_queries(
        cls,
        base_instance,
        num_agents: int,
        rng: np.random.RandomState,
    ) -> Optional[MRMPQuery]:
        env = base_instance.env
        bbox_diag = float(np.linalg.norm(env.ub - env.lb))
        endpoint_sep_tol = cls._endpoint_separation_tol(env)
        sep_tol = max(cls._point_separation(env), endpoint_sep_tol)
        queries = cls._sample_valid_random_queries(
            base_instance=base_instance,
            rng=rng,
            num_agents=num_agents,
            sep_tol=sep_tol,
            min_start_goal_dist=cls.MIN_START_GOAL_DIST_RATIO * bbox_diag,
            direct_exclusion_tol=sep_tol,
        )
        return None if queries is None else MRMPQuery(queries)

    @classmethod
    def _solve_independent_queries(
        cls,
        instance,
        queries: Sequence[MPQuery],
        reference_budget: float,
    ) -> Tuple[Optional[List[object]], Optional[Tuple[float, float, float]]]:
        backend = cls.native_backend()
        path = instance.native_reference_input
        sols = []
        total_cost = 0.0
        total_runtime = 0.0
        makespan = 0.0
        for index, query in enumerate(queries):
            backend.write_input(path, instance.env, query, instance.stgcs.tmax, [])
            command = backend.command(path, PBSExpansionAblation.LOW_LEVEL_SPEC,
                                      reference_budget, "search", 0, 100)
            result = backend.run_process(command, reference_budget + 30,
                                         backend.work_dir / "reference-logs" / f"{path.stem}-{index}.log")
            if not result.get("success") or result["runtime"] > reference_budget:
                return None, None
            segments = result["segments"]
            sol = STTrajectory([f"native:{i}" for i in range(len(segments))],
                               [np.asarray(s).ravel() for s in segments], instance.env.dim)
            sols.append(sol)
            total_cost += float(sol.duration)
            total_runtime += float(result["runtime"])
            makespan = max(makespan, float(sol.duration))
        return sols, (total_cost, makespan, total_runtime)

    @classmethod
    def _instance_id(cls, base_instance_id: str) -> str:
        return f"mrmp-{base_instance_id}-{cls.NUM_AGENTS}-robot-intersection"

    @classmethod
    def _prepare_low_level_heuristics(cls, instance, base_record: BaseBenchmarkRecord) -> None:
        backend = cls.native_backend()
        query = SimpleNamespace(start=[0.] * instance.env.dim, goal=[0.] * instance.env.dim,
                                t_start=0., is_stay=True, vlimit=instance.stgcs.vlimit)
        path = backend.work_dir / "inputs" / f"{base_record.instance_id}.txt"
        backend.write_input(path, instance.env, query, instance.stgcs.tmax, [])
        backend.bind_cache(path, base_record.instance_id, base_record.domain_key)
        backend.cache(path)
        instance.native_reference_input = path

    @classmethod
    def build_manifest(
        cls,
        base_records: Sequence[BaseBenchmarkRecord],
        on_record: RecordCallback | None = None,
        on_status: StatusCallback | None = None,
    ) -> List[MRMPBenchmarkRecord]:
        records: List[MRMPBenchmarkRecord] = []
        for ordinal, base_record in enumerate(base_records):
            cls._emit_status(
                on_status,
                "candidate",
                num_agents=cls.NUM_AGENTS,
                spatial_seed=base_record.spatial_seed,
                remaining=len(base_records) - len(records),
                base_instance_id=base_record.instance_id,
            )
            base = BaseInstanceFactory.from_record(base_record, compute_heuristics=False)
            cls._prepare_low_level_heuristics(base, base_record)
            record = cls.synthesize_record(
                base_instance=base,
                base_record=base_record,
                num_agents=cls.NUM_AGENTS,
                rng=np.random.RandomState(base_record.spatial_seed + 7919 * (ordinal + 1)),
                instance_id=cls._instance_id(base_record.instance_id),
                min_conflicting_pairs=cls.MIN_TRAJECTORY_INTERSECTING_PAIRS,
                on_status=on_status,
            )
            records.append(record)
            if on_record is not None:
                on_record(record, records)
        return records


class WindowedCoordinationMRMPManifestBuilder(PBSExpansionMRMPManifestBuilder):
    NUM_AGENTS = 20
    MIN_TRAJECTORY_INTERSECTING_PAIRS = 10
    MAX_CONFLICT_DENSITY_BY_DOMAIN = {
        "grid2d": 0.45,
        "maze": 0.45,
    }
    TARGET_CONFLICT_MEAN_BY_DOMAIN = {
        "grid2d": 0.25,
        "maze": 0.25,
    }
    TARGET_CONFLICT_MEDIAN_BY_DOMAIN = {
        "grid2d": 0.25,
        "maze": 0.25,
    }
    TARGET_CONFLICT_MAX_BY_DOMAIN = {
        "grid2d": 0.45,
        "maze": 0.45,
    }

    @classmethod
    def record_matches_benchmark(cls, record: MRMPBenchmarkRecord) -> bool:
        return (
            record.num_agents == len(record.queries) == cls.NUM_AGENTS
            and cls.conflict_density_allowed(record.domain_key, record.conflict_density)
        )

    @classmethod
    def conflict_density_selection_enabled(cls, domain_key: str) -> bool:
        return str(domain_key) in cls.TARGET_CONFLICT_MEAN_BY_DOMAIN

    @classmethod
    def conflict_density_selection_pool_count(
        cls,
        domain_key: str,
        target_count: int,
        available_count: int,
    ) -> int:
        if not cls.conflict_density_selection_enabled(domain_key):
            return super().conflict_density_selection_pool_count(domain_key, target_count, available_count)
        return int(available_count)

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        if not values:
            return float("nan")
        sorted_values = sorted(float(value) for value in values)
        mid = len(sorted_values) // 2
        if len(sorted_values) % 2:
            return sorted_values[mid]
        return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])

    @classmethod
    def _density_selection_score(
        cls,
        domain_key: str,
        records: Sequence[MRMPBenchmarkRecord],
    ) -> Tuple[float, float, float, float, Tuple[str, ...]]:
        densities = [float(record.conflict_density) for record in records]
        mean_density = float(sum(densities) / len(densities))
        median_density = cls._median(densities)
        max_density = max(densities)
        target_mean = float(cls.TARGET_CONFLICT_MEAN_BY_DOMAIN[str(domain_key)])
        target_median = float(cls.TARGET_CONFLICT_MEDIAN_BY_DOMAIN[str(domain_key)])
        target_max = float(cls.TARGET_CONFLICT_MAX_BY_DOMAIN[str(domain_key)])
        overflow = max(0.0, max_density - target_max)
        score = (
            abs(mean_density - target_mean)
            + abs(median_density - target_median)
            + 0.25 * abs(max_density - target_max)
            + 10.0 * overflow
        )
        return (
            score,
            abs(mean_density - target_mean),
            abs(median_density - target_median),
            abs(max_density - target_max),
            tuple(sorted(record.instance_id for record in records)),
        )

    @classmethod
    def _improve_density_selection(
        cls,
        domain_key: str,
        selected: Sequence[MRMPBenchmarkRecord],
        candidates: Sequence[MRMPBenchmarkRecord],
    ) -> List[MRMPBenchmarkRecord]:
        selected_records = list(selected)
        selected_ids = {record.instance_id for record in selected_records}
        unselected_records = [record for record in candidates if record.instance_id not in selected_ids]
        best_score = cls._density_selection_score(domain_key, selected_records)

        improved = True
        while improved:
            improved = False
            best_swap: Tuple[int, int, Tuple[float, float, float, float, Tuple[str, ...]]] | None = None
            for selected_idx, _selected_record in enumerate(selected_records):
                for unselected_idx, unselected_record in enumerate(unselected_records):
                    trial = list(selected_records)
                    trial[selected_idx] = unselected_record
                    trial_score = cls._density_selection_score(domain_key, trial)
                    if trial_score < best_score:
                        best_score = trial_score
                        best_swap = (selected_idx, unselected_idx, trial_score)
            if best_swap is not None:
                selected_idx, unselected_idx, _score = best_swap
                selected_records[selected_idx], unselected_records[unselected_idx] = (
                    unselected_records[unselected_idx],
                    selected_records[selected_idx],
                )
                improved = True
        return selected_records

    @classmethod
    def select_records_for_domain(
        cls,
        domain_key: str,
        records: Sequence[MRMPBenchmarkRecord],
        target_count: int,
    ) -> List[MRMPBenchmarkRecord]:
        domain_key = str(domain_key)
        if not cls.conflict_density_selection_enabled(domain_key):
            return super().select_records_for_domain(domain_key, records, target_count)
        target_count = int(target_count)
        candidates = sorted(
            records,
            key=lambda record: (float(record.conflict_density), record.instance_id),
        )
        if len(candidates) <= target_count:
            return candidates

        target_mean = float(cls.TARGET_CONFLICT_MEAN_BY_DOMAIN[domain_key])
        target_max = float(cls.TARGET_CONFLICT_MAX_BY_DOMAIN[domain_key])
        closest_to_mean = sorted(
            candidates,
            key=lambda record: (abs(float(record.conflict_density) - target_mean), record.instance_id),
        )[:target_count]
        tail_record = min(
            candidates,
            key=lambda record: (abs(float(record.conflict_density) - target_max), record.instance_id),
        )
        with_tail = [tail_record]
        with_tail.extend(
            record
            for record in closest_to_mean
            if record.instance_id != tail_record.instance_id
        )
        if len(with_tail) < target_count:
            with_tail.extend(
                record
                for record in candidates
                if record.instance_id not in {item.instance_id for item in with_tail}
            )
        initial_sets = (
            closest_to_mean,
            with_tail[:target_count],
        )
        best_records = min(
            (cls._improve_density_selection(domain_key, initial, candidates) for initial in initial_sets),
            key=lambda selected: cls._density_selection_score(domain_key, selected),
        )
        return sorted(best_records, key=lambda record: record.instance_id)


class MRMPPerformanceComparisonManifestBuilder(PBSExpansionMRMPManifestBuilder):
    NUM_AGENTS = MRMPPerformanceComparison.MAX_NUM_AGENTS
    MIN_TRAJECTORY_INTERSECTING_PAIRS = 0

    @classmethod
    def record_matches_benchmark(cls, record):
        return (record.num_agents in MRMPPerformanceComparison.num_agent_values()
                and len(record.queries) == record.num_agents)

    @classmethod
    def target_count(cls, replicates_per_robot_count: int) -> int:
        return len(MRMPPerformanceComparison.num_agent_values()) * int(replicates_per_robot_count)

    @classmethod
    def _instance_id(cls, base_instance_id: str, num_agents: int) -> str:
        return f"mrmp-{base_instance_id}-{num_agents}-robot-intersection"

    @staticmethod
    def _replicate_counts(
        records: Sequence[MRMPBenchmarkRecord],
    ) -> Dict[int, int]:
        counts = {num_agents: 0 for num_agents in MRMPPerformanceComparison.num_agent_values()}
        for record in records:
            num_agents = int(record.num_agents)
            if num_agents in counts:
                counts[num_agents] += 1
        return counts

    @classmethod
    def _current_existing_records(
        cls,
        existing_records: Sequence[MRMPBenchmarkRecord],
        replicates_per_robot_count: int,
    ) -> List[MRMPBenchmarkRecord]:
        counts = {num_agents: 0 for num_agents in MRMPPerformanceComparison.num_agent_values()}
        records: List[MRMPBenchmarkRecord] = []
        for record in existing_records:
            if not cls.record_matches_benchmark(record):
                continue
            num_agents = int(record.num_agents)
            if counts[num_agents] >= int(replicates_per_robot_count):
                continue
            records.append(record)
            counts[num_agents] += 1
        return records

    @classmethod
    def build_manifest(
        cls,
        base_records: Sequence[BaseBenchmarkRecord],
        replicates_per_robot_count: int = 1,
        existing_records: Sequence[MRMPBenchmarkRecord] | None = None,
        on_record: RecordCallback | None = None,
        on_status: StatusCallback | None = None,
    ) -> List[MRMPBenchmarkRecord]:
        replicates_per_robot_count = int(replicates_per_robot_count)
        if replicates_per_robot_count <= 0:
            raise ValueError("replicates_per_robot_count must be positive.")

        records = cls._current_existing_records(existing_records or (), replicates_per_robot_count)
        used_instance_ids = {record.instance_id for record in records}
        counts = cls._replicate_counts(records)

        for num_agents in MRMPPerformanceComparison.num_agent_values():
            for base_record in base_records:
                if counts[num_agents] >= replicates_per_robot_count:
                    break
                instance_id = cls._instance_id(base_record.instance_id, num_agents)
                if instance_id in used_instance_ids:
                    continue
                cls._emit_status(
                    on_status,
                    "candidate",
                    num_agents=num_agents,
                    spatial_seed=base_record.spatial_seed,
                    remaining=replicates_per_robot_count - counts[num_agents],
                    base_instance_id=base_record.instance_id,
                )
                base = BaseInstanceFactory.from_record(base_record, compute_heuristics=False)
                cls._prepare_low_level_heuristics(base, base_record)
                try:
                    record = cls.synthesize_record(
                        base_instance=base,
                        base_record=base_record,
                        num_agents=num_agents,
                        rng=np.random.RandomState(
                            base_record.spatial_seed
                            + 7919 * int(num_agents)
                            + 104729 * (counts[num_agents] + 1)
                        ),
                        instance_id=instance_id,
                        min_conflicting_pairs=cls.MIN_TRAJECTORY_INTERSECTING_PAIRS,
                        on_status=on_status,
                    )
                except RuntimeError as exc:
                    if not str(exc).startswith("Failed to synthesize MRMP record"):
                        raise
                    cls._emit_status(
                        on_status,
                        "candidate_failed",
                        base_instance_id=base_record.instance_id,
                        num_agents=num_agents,
                        reason=str(exc),
                    )
                    continue
                records.append(record)
                used_instance_ids.add(record.instance_id)
                counts[num_agents] += 1
                if on_record is not None:
                    on_record(record, records)

            if counts[num_agents] < replicates_per_robot_count:
                raise RuntimeError(
                    f"Unable to build {replicates_per_robot_count} {num_agents}-robot MRMP performance-comparison records; "
                    f"built {counts[num_agents]}."
                )

        return records
