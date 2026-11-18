from __future__ import annotations

import csv
import gc
import math
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np

from baselines.cb_gcs import CBGCSOptions, CBGCSPlanner
from baselines.ompl_kcbs import OfficialOMPLKCBSOptions, OfficialOMPLKCBSPlanner
from baselines.pbs_zeta_sipp import ZetaSIPPPriorityBasedSearch
from baselines.sp_strrtstar import FixedPrioritySTRRTStarPlanner, FixedPrioritySTRRTStarSnapshot
from experiments.manifests.base import BaseBenchmarkRecord
from experiments.base import BaseInstanceFactory
from experiments.instance import Instance
from experiments.native import STGCSPlannerClient, NativeGeometry
from experiments.planners.mrmp import MRMPPlannerSettings, TrajoptSettings, SearchPlannerSpec
from stgcs_native.trajopt import GlobalTrajectoryOptimizer
from stgcs_native.types import MRMPResultEntry
from experiments.manifests.mrmp import MRMPBenchmarkRecord
from stgcs_native.types import ChildExpansionMode
from stgcs_native.types import MRMPQuery


class MRMPExperiment:
    @classmethod
    def run_stgcs(
        cls,
        record: MRMPBenchmarkRecord,
        base_record,
        low_level_spec: SearchPlannerSpec,
        planner_key: str,
        planner_name: str,
        planner_settings: MRMPPlannerSettings,
        budget: float,
        window_alpha: float,
        window_beta: float,
        child_expansion_rule: str,
    ) -> dict[str, Any]:
        from visualization.solution_io import SolutionIO, ViewerMRMPSolutionRun

        instance = queries = None
        try:
            instance, queries = cls.reconstruct_instance(record, base_record, compute_heuristics=False)
            solutions, entry = STGCSPlannerClient.plan(
                instance, queries, low_level_spec, float(budget),
                planner_settings.child_expansion_rules[child_expansion_rule].name.lower(),
                window_span_factor=float(window_alpha),
                dynamic_window_adjustment=bool(planner_settings.dynamic_window_adjustment),
                execution_horizon_factor=float(window_beta),
            )
            payload = SolutionIO.mrmp_run_to_viewer_json(
                record,
                ViewerMRMPSolutionRun(
                    planner_key=planner_key,
                    planner_name=planner_name,
                    budget=float(budget),
                    status="SUCCESS" if entry.is_success else "FAIL",
                    entry=entry,
                    solutions=solutions,
                ),
            )
            payload.update(
                {
                    "backend": "stgcs_cplusplus",
                    "tmax": float(instance.stgcs.tmax),
                    "window_alpha": float(window_alpha),
                    "window_beta": float(window_beta),
                    "epsilon": float(low_level_spec.epsilon),
                    "child_expansion_rule": str(child_expansion_rule),
                    "dynamic_window_adjustment": bool(planner_settings.dynamic_window_adjustment),
                }
            )
            return payload
        finally:
            del queries
            del instance
            gc.collect()

    @classmethod
    def run_trajopt(
        cls,
        record: MRMPBenchmarkRecord,
        base_record,
        mrmp_payload: dict[str, Any],
        trajopt_planner_key: str,
        trajopt_planner_name: str,
        settings: TrajoptSettings,
    ) -> dict[str, Any]:
        from visualization.solution_io import SolutionIO

        trajectories = SolutionIO.pwl_trajectories_from_solution_payload(
            mrmp_payload, expected_num_agents=int(record.num_agents))
        instance = queries = None
        try:
            instance, queries = cls.reconstruct_instance(record, base_record, compute_heuristics=False)
            robot_radius = float(instance.env.robot_radius)
            vlimit = float(instance.stgcs.vlimit)
            config = settings.config(robot_radius, vlimit)
            window_span = settings.window_span(robot_radius, vlimit)
            stride = settings.stride(robot_radius, vlimit)
            cells = NativeGeometry.cells(instance.stgcs)
            result = GlobalTrajectoryOptimizer.smooth(
                trajectories, robot_radius, vlimit, config,
                window_span=window_span, stride=stride,
                goal_stays=[query.is_stay for query in queries],
                collision_tolerance=settings.collision_tolerance,
                cells=cells, env=None if cells else NativeGeometry.environment(instance.env),
            )
        finally:
            del queries
            del instance
            gc.collect()
        return SolutionIO.refined_solution_payload(
            record, mrmp_payload, result, config,
            trajopt_planner_key=trajopt_planner_key, trajopt_planner_name=trajopt_planner_name,
            window_span=window_span, stride=stride, continuous_pairwise_ok=True,
        )

    @staticmethod
    def reconstruct_instance(
        record: MRMPBenchmarkRecord,
        base_record: BaseBenchmarkRecord,
        compute_heuristics: bool = False,
    ) -> Tuple[object, MRMPQuery]:
        instance = BaseInstanceFactory.from_record(base_record, compute_heuristics=False)
        instance.base_record = base_record
        if instance.stgcs.tmax != record.tmax:
            instance.stgcs = Instance.build_stgcs_from_env(
                instance.env, tmax=record.tmax, vlimit=instance.stgcs.vlimit,
            )
        if compute_heuristics:
            instance.compute_heuristics()
        return instance, MRMPQuery([query.to_query() for query in record.queries])

    @staticmethod
    def _profile_stats(profiler, key):
        return profiler.get(key, np.array([0.0, 0.0]))

    @classmethod
    def _entry_from_fixed_priority_strrt_star_snapshot(
        cls,
        snapshot: FixedPrioritySTRRTStarSnapshot,
        queries: Sequence[object],
    ) -> MRMPResultEntry:
        return MRMPResultEntry.from_statistics(
            solutions=snapshot.solutions,
            queries=queries,
            is_success=snapshot.success,
            runtime=snapshot.runtime,
            pbs_calls=1,
            pbs_runtime=snapshot.runtime,
            pbs_popped_nodes=0,
            pbs_generated_children=0,
            pbs_update_calls=0,
            mp_stats=np.array([snapshot.low_level_calls, snapshot.low_level_runtime], dtype=float),
            ecd_stats=np.zeros(2, dtype=float),
            cc_stats=np.array([snapshot.collision_checks, snapshot.collision_runtime], dtype=float),
        )

    @classmethod
    def run_fixed_priority_strrt_star_snapshots(
        cls,
        instance,
        queries: Sequence[object],
        budget: float,
    ) -> Tuple[MRMPResultEntry, MRMPResultEntry]:
        planner = FixedPrioritySTRRTStarPlanner()
        result = planner.solve(instance, queries, timeout_secs=budget)
        return (
            cls._entry_from_fixed_priority_strrt_star_snapshot(result.first_solution, queries),
            cls._entry_from_fixed_priority_strrt_star_snapshot(result.final_solution, queries),
        )

    @classmethod
    def run_fixed_priority_strrt_star(
        cls,
        instance,
        queries: Sequence[object],
        budget: float,
    ) -> MRMPResultEntry:
        _, final_entry = cls.run_fixed_priority_strrt_star_snapshots(instance, queries, budget)
        return final_entry

    @staticmethod
    def _zeta_sipp_cell_size(instance, multiplier: float) -> float:
        return float(multiplier) * float(instance.env.robot_radius)

    @classmethod
    def run_pbs_zeta_sipp_with_solutions(
        cls,
        instance,
        queries: Sequence[object],
        budget: float,
        child_expansion_mode: ChildExpansionMode,
        cell_size_multiplier: float = 1.0,
        seed: int = 0,
    ) -> Tuple[Sequence[object] | None, MRMPResultEntry]:
        planner = ZetaSIPPPriorityBasedSearch(
            instance.stgcs,
            instance.env,
            robot_radius=instance.env.robot_radius,
            child_expansion_mode=child_expansion_mode,
            seed=int(seed),
            cell_size=cls._zeta_sipp_cell_size(instance, cell_size_multiplier),
            runtime_limit_secs=float(budget),
        )
        solutions, runtime, success = planner.run(list(queries), budget, verbose=False)
        entry = MRMPResultEntry.from_statistics(
            solutions=solutions,
            queries=queries,
            is_success=success,
            runtime=runtime,
            pbs_calls=1,
            pbs_runtime=runtime,
            pbs_popped_nodes=planner.num_popped_nodes,
            pbs_generated_children=planner.num_generated_children,
            pbs_update_calls=planner.num_update_calls,
            mp_stats=planner._profiler["mp"],
            ecd_stats=planner._profiler["ecd"],
            cc_stats=planner._profiler["cc"],
            mp_gcs_stats=cls._profile_stats(planner._profiler, "gcs"),
            mp_gub_stats=cls._profile_stats(planner._profiler, "gub"),
            mp_search_stats=cls._profile_stats(planner._profiler, "search"),
            mp_convex_restriction_stats=cls._profile_stats(planner._profiler, "cr"),
            mp_dominance_check_stats=cls._profile_stats(planner._profiler, "dc"),
            mp_dominance_convex_restriction_stats=cls._profile_stats(planner._profiler, "dc_cr"),
        )
        return solutions, entry

    @classmethod
    def run_pbs_zeta_sipp(
        cls,
        instance,
        queries: Sequence[object],
        budget: float,
        child_expansion_mode: ChildExpansionMode,
        cell_size_multiplier: float = 1.0,
        seed: int = 0,
    ) -> MRMPResultEntry:
        return cls.run_pbs_zeta_sipp_with_solutions(
            instance,
            queries,
            budget,
            child_expansion_mode,
            cell_size_multiplier,
            seed,
        )[1]

    @classmethod
    def _cb_gcs_result_entry(
        cls,
        result,
        queries: Sequence[object],
    ) -> MRMPResultEntry:
        if result.success:
            if result.solutions is None:
                raise ValueError("Successful CB-GCS coordination must return solutions.")
            sum_of_costs, makespan = MRMPResultEntry._solution_metrics(result.solutions)
            completed_agents = len(queries)
        else:
            makespan = math.inf
            completed_agents = MRMPResultEntry._num_completed_agents([] if result.solutions is None else result.solutions, queries)
            sum_of_costs = math.inf
        return MRMPResultEntry(
            is_success=bool(result.success),
            runtime=float(result.runtime),
            sum_of_costs=sum_of_costs,
            makespan=makespan,
            num_completed_agents=completed_agents,
            pbs_calls=1,
            pbs_runtime=float(result.runtime),
            pbs_popped_nodes=int(result.popped_nodes),
            pbs_generated_children=int(result.generated_children),
            pbs_update_calls=int(result.update_calls),
            mp_calls=int(result.low_level_calls),
            mp_runtime=float(result.low_level_runtime),
            ecd_calls=0,
            ecd_runtime=0.0,
            cc_calls=int(result.conflict_checks),
            cc_runtime=float(result.conflict_runtime),
        )

    @classmethod
    def run_cb_gcs_with_solutions(
        cls,
        instance,
        queries: Sequence[object],
        budget: float,
        time_horizon: float | None = None,
        max_rounded_paths: int = 100,
        time_step: float | None = None,
    ) -> Tuple[Sequence[object] | None, MRMPResultEntry]:
        planner = CBGCSPlanner(
            CBGCSOptions(
                time_horizon=time_horizon,
                time_step=time_step,
                max_rounded_paths=int(max_rounded_paths),
            )
        )
        result = planner.solve(instance, queries, timeout_secs=budget)
        return result.solutions, cls._cb_gcs_result_entry(result, queries)

    @classmethod
    def run_cb_gcs(
        cls,
        instance,
        queries: Sequence[object],
        budget: float,
        time_horizon: float | None = None,
        max_rounded_paths: int = 100,
        time_step: float | None = None,
    ) -> MRMPResultEntry:
        return cls.run_cb_gcs_with_solutions(
            instance,
            queries,
            budget,
            time_horizon=time_horizon,
            max_rounded_paths=max_rounded_paths,
            time_step=time_step,
        )[1]

    @classmethod
    def _entry_from_ompl_kcbs_result(
        cls,
        result,
        queries: Sequence[object],
    ) -> MRMPResultEntry:
        if result.success:
            if result.solutions is None:
                raise ValueError("Successful K-CBS coordination must return solutions.")
            makespan = float(max(solution.duration for solution in result.solutions))
            completed_agents = len(queries)
            sum_of_costs = float(result.cost)
        else:
            makespan = math.inf
            completed_agents = MRMPResultEntry._num_completed_agents([] if result.solutions is None else result.solutions, queries)
            sum_of_costs = math.inf
        root_solve_time = float(result.root_solve_time)
        return MRMPResultEntry(
            is_success=bool(result.success),
            runtime=float(result.runtime),
            sum_of_costs=sum_of_costs,
            makespan=makespan,
            num_completed_agents=completed_agents,
            pbs_calls=1,
            pbs_runtime=float(result.runtime),
            pbs_popped_nodes=int(result.num_nodes_expanded),
            pbs_generated_children=0,
            pbs_update_calls=0,
            mp_calls=len(queries) if root_solve_time >= 0.0 else 0,
            mp_runtime=max(root_solve_time, 0.0),
            ecd_calls=0,
            ecd_runtime=0.0,
            cc_calls=0,
            cc_runtime=0.0,
        )

    @classmethod
    def run_ompl_kcbs_with_solutions(
        cls,
        instance,
        queries: Sequence[object],
        budget: float,
        low_level_solve_time: float = 1.0,
        propagation_step_size: float = 0.1,
        min_control_duration: int = 1,
        max_control_duration: int = 10,
        goal_tolerance: float = 1e-3,
        goal_bias: float = 0.05,
        intermediate_states: bool = False,
        merge_bound: int | None = None,
        num_threads: int = 4,
        seed: int = 0,
    ) -> Tuple[Sequence[object] | None, MRMPResultEntry]:
        planner = OfficialOMPLKCBSPlanner(
            OfficialOMPLKCBSOptions(
                max_runtime_in_secs=float(budget),
                low_level_solve_time=float(low_level_solve_time),
                propagation_step_size=float(propagation_step_size),
                min_control_duration=int(min_control_duration),
                max_control_duration=int(max_control_duration),
                goal_tolerance=float(goal_tolerance),
                goal_bias=float(goal_bias),
                intermediate_states=bool(intermediate_states),
                merge_bound=merge_bound,
                num_threads=int(num_threads),
            ),
            seed=int(seed),
        )
        result = planner.solve(instance, queries, timeout_secs=budget)
        return result.solutions, cls._entry_from_ompl_kcbs_result(result, queries)

    @classmethod
    def run_ompl_kcbs(
        cls,
        instance,
        queries: Sequence[object],
        budget: float,
        low_level_solve_time: float = 1.0,
        propagation_step_size: float = 0.1,
        min_control_duration: int = 1,
        max_control_duration: int = 10,
        goal_tolerance: float = 1e-3,
        goal_bias: float = 0.05,
        intermediate_states: bool = False,
        merge_bound: int | None = None,
        num_threads: int = 4,
        seed: int = 0,
    ) -> MRMPResultEntry:
        return cls.run_ompl_kcbs_with_solutions(
            instance,
            queries,
            budget,
            low_level_solve_time=low_level_solve_time,
            propagation_step_size=propagation_step_size,
            min_control_duration=min_control_duration,
            max_control_duration=max_control_duration,
            goal_tolerance=goal_tolerance,
            goal_bias=goal_bias,
            intermediate_states=intermediate_states,
            merge_bound=merge_bound,
            num_threads=num_threads,
            seed=seed,
        )[1]

    @staticmethod
    def planner_output_path(
        output_root: str | Path,
        domain_key: str,
        planner_name: str,
    ) -> Path:
        target_dir = Path(output_root) / domain_key / "results"
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = planner_name.replace("/", "_")
        return target_dir / f"{safe_name}.csv"

    RESULT_COLUMNS = (
        'name',
        'is_success',
        'runtime',
        'sum_of_costs',
        'makespan',
        'num_completed_agents',
        'pbs_calls',
        'pbs_runtime',
        'pbs_popped_nodes',
        'pbs_generated_children',
        'pbs_update_calls',
        'mp_calls',
        'mp_runtime',
        'ecd_calls',
        'ecd_runtime',
        'cc_calls',
        'cc_runtime',
        'instance_id',
        'budget',
        'num_agents',
        'stgcs_num_vertices',
        'stgcs_num_edges',
        'mp_gcs_calls',
        'mp_gcs_runtime',
        'mp_gub_calls',
        'mp_gub_runtime',
        'mp_search_calls',
        'mp_search_runtime',
        'mp_convex_restriction_calls',
        'mp_convex_restriction_runtime',
        'mp_dominance_check_calls',
        'mp_dominance_check_runtime',
        'mp_dominance_convex_restriction_calls',
        'mp_dominance_convex_restriction_runtime',
    )

    @classmethod
    def profile_values_available(cls, row):
        return all(math.isfinite(float(row.get(name, math.nan))) and float(row[name]) >= 0
                   for name in cls.RESULT_COLUMNS[22:])

    @classmethod
    def append_result_row(cls, output_path, instance_name, record, budget, entry):
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        values = dict(vars(entry), name=instance_name, instance_id=record.instance_id,
                      budget=budget, num_agents=record.num_agents,
                      stgcs_num_vertices=record.stgcs_num_vertices, stgcs_num_edges=record.stgcs_num_edges)
        with target.open("a", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerow([values[name] for name in cls.RESULT_COLUMNS])

    @classmethod
    def load_result_rows(cls, path):
        rows = []
        with Path(path).open(newline="") as stream:
            for row in csv.reader(stream):
                if len(row) not in (22, len(cls.RESULT_COLUMNS)):
                    raise ValueError(f"Invalid MRMP result row in {path}: {len(row)} columns")
                values = {}
                for name, value in zip(cls.RESULT_COLUMNS, row):
                    if name in ("name", "instance_id"):
                        values[name] = value
                    elif name == "is_success":
                        values[name] = value.lower() == "true"
                    elif name.endswith(("_calls", "_nodes", "_children")) or name in (
                            "num_completed_agents", "num_agents", "stgcs_num_vertices", "stgcs_num_edges"):
                        values[name] = int(float(value))
                    else:
                        values[name] = float(value)
                for name in cls.RESULT_COLUMNS[22:]:
                    values.setdefault(name, 0 if name.endswith("_calls") else 0.0)
                available = len(row) == len(cls.RESULT_COLUMNS) and cls.profile_values_available(values)
                values["has_detailed_profile"] = available
                values["has_disjoint_breakdown_profile"] = available
                rows.append(values)
        return rows
