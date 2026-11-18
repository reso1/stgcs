from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Sequence

from baselines.common import ShortestPathSolution
from baselines.ompl_strrt_star import OfficialOMPLSTRRTStar, OfficialOMPLSTRRTStarOptions
from baselines.zeta_sipp import ZetaStarSIPPPlanner
from experiments.base import BaseManifestStore
from experiments.manifests.mrmp import MRMPBenchmarkRecord
from experiments.planners.mrmp import MRMPPerformanceComparison, SearchPlannerSpec
from experiments.manifests.st_planning import (
    STHeuristicAblationRecord,
    STHeuristicAblationResultEntry,
    STPlanningManifestStore,
)
from experiments.planners.st_planning import STPerformanceComparisonConfig
from experiments.native import STGCSPlannerClient
from visualization.solution_io import SolutionIO, ViewerSolutionRun, ViewerMRMPSolutionRun


class ViewerPlannerService:
    DEFAULT_BUDGET = 10.0
    ST_PLANNER_OPTIONS = (
        ("delta-pos", STPerformanceComparisonConfig.DELTA_POS_PLANNER, (2, 3)),
        ("delta-set", STPerformanceComparisonConfig.DELTA_SET_PLANNER, (2, 3)),
        ("delta-set-eps1", STPerformanceComparisonConfig.DELTA_SET_EPS1_PLANNER, (2, 3)),
        ("micp", STPerformanceComparisonConfig.MICP_PLANNER, (2, 3)),
        ("micpg", STPerformanceComparisonConfig.MICP_ROUNDING_PLANNER, (2, 3)),
        ("strrt-first", STPerformanceComparisonConfig.ST_RRT_FIRST_PLANNER, (2, 3)),
        ("strrt-final", STPerformanceComparisonConfig.ST_RRT_FINAL_PLANNER, (2, 3)),
        ("zeta", STPerformanceComparisonConfig.ZETA_SIPP_PLANNER, (2,)),
        ("zeta2r", STPerformanceComparisonConfig.ZETA_SIPP_2R_PLANNER, (2,)),
    )
    MRMP_PLANNER_BUTTONS = {
        MRMPPerformanceComparison.KCBS_KEY: "K-CBS",
        MRMPPerformanceComparison.CB_GCS_KEY: "CB-GCS",
        MRMPPerformanceComparison.WINDOWED_PBS_KEY: "wPBS",
        MRMPPerformanceComparison.WINDOWED_PP_KEY: "wPP",
        MRMPPerformanceComparison.PBS_KEY: "PBS",
    }
    MRMP_VIEWER_PLANNER_KEYS = (
        MRMPPerformanceComparison.KCBS_KEY,
        MRMPPerformanceComparison.CB_GCS_KEY,
        MRMPPerformanceComparison.WINDOWED_PBS_KEY,
        MRMPPerformanceComparison.WINDOWED_PP_KEY,
        MRMPPerformanceComparison.PBS_KEY,
    )

    def __init__(self, base_root: str | Path, records: Sequence = ()) -> None:
        self.base_root = Path(base_root).resolve()
        self.records_by_id = {record.instance_id: record for record in records}

    def solve_request(self, request: dict) -> dict:
        instance_id = str(request["instance_id"])
        if instance_id not in self.records_by_id:
            raise KeyError(f"Instance {instance_id!r} is not exposed by this viewer.")
        defaults = self.planner_payload()
        return self.run_solution(
            self.records_by_id[instance_id], planner_key=str(request["planner_key"]),
            budget=float(request.get("budget", self.DEFAULT_BUDGET)),
            seed_offset=int(request.get("seed_offset", 0)),
            window_span_factor=float(request.get("window_alpha", defaults["default_window_alpha"])),
            execution_horizon_factor=float(request.get("window_beta", defaults["default_window_beta"])),
            epsilon=float(request.get("epsilon", defaults["default_epsilon"])),
        )

    @classmethod
    def planner_options_for_kind(cls, problem_kind: str) -> list[Dict[str, Any]]:
        if problem_kind == "st_heuristic_ablation":
            return [
                {"key": key, "name": name, "space_dims": list(space_dims)}
                for key, name, space_dims in cls.ST_PLANNER_OPTIONS
            ]
        if problem_kind == "mrmp":
            options: list[Dict[str, Any]] = []
            for key in cls.MRMP_VIEWER_PLANNER_KEYS:
                option = {
                    "key": key,
                    "name": MRMPPerformanceComparison.planner_name(key),
                    "button_label": cls.MRMP_PLANNER_BUTTONS[key],
                }
                if key in {
                    MRMPPerformanceComparison.CB_GCS_KEY,
                    MRMPPerformanceComparison.KCBS_KEY,
                }:
                    option["space_dims"] = [2]
                options.append(option)
            return options
        return []

    @classmethod
    def planner_payload(cls) -> Dict[str, Any]:
        return {
            "ok": True,
            "default_budget": cls.DEFAULT_BUDGET,
            "default_window_alpha": MRMPPerformanceComparison.WINDOW_SPAN_FACTOR,
            "default_window_beta": MRMPPerformanceComparison.EXECUTION_HORIZON_FACTOR,
            "default_epsilon": MRMPPerformanceComparison.LOW_LEVEL_SPEC.epsilon,
            "planners": {
                "st_heuristic_ablation": cls.planner_options_for_kind("st_heuristic_ablation"),
                "mrmp": cls.planner_options_for_kind("mrmp"),
                "base": [],
            },
        }

    @staticmethod
    def positive_finite_float(value: float | int, name: str) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
        return numeric

    @staticmethod
    def unit_interval_float(value: float | int, name: str) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1].")
        return numeric

    @staticmethod
    def _require_single_planner(planner_key: str) -> str:
        planner_name = STPerformanceComparisonConfig.planner_name_from_cli_value(planner_key)
        expanded = STPerformanceComparisonConfig.expand_planner_name(planner_name)
        if len(expanded) != 1:
            raise ValueError(
                f"Planner key {planner_key!r} expands to multiple result streams; choose one explicit stream."
            )
        return expanded[0]

    @staticmethod
    def _run_search_solution(instance, query, planner_name, budget):
        solution, result = STGCSPlannerClient.solve_single(
            instance.env, query, instance.stgcs.tmax,
            spec=STPerformanceComparisonConfig.search_spec_for_planner(planner_name), budget=budget,
            base_record=getattr(instance, "base_record", None))
        return ViewerPlannerService._native_st_result(solution, result)

    @staticmethod
    def _run_micp_solution(instance, query, budget, max_rounded_paths):
        solution, result = STGCSPlannerClient.solve_single(
            instance.env, query, instance.stgcs.tmax, budget=budget,
            method="micpg" if max_rounded_paths else "micp", rounded=max_rounded_paths or 100)
        return ViewerPlannerService._native_st_result(solution, result)

    @staticmethod
    def _native_st_result(solution, result):
        entry = STHeuristicAblationResultEntry(
            is_success=result["success"], runtime=result["runtime"],
            cost=result["cost"] if result["cost"] is not None else math.inf,
            num_expanded_nodes=result["expanded"], num_generated_nodes=result["generated"])
        return solution, entry, "SUCCESS" if entry.is_success else "FAIL"

    @staticmethod
    def _run_strrt_solution(
        instance,
        query,
        budget: float,
        seed: int,
        planner_name: str,
    ) -> tuple[ShortestPathSolution, STHeuristicAblationResultEntry, str]:
        options = OfficialOMPLSTRRTStarOptions(
            max_runtime_in_secs=float(budget),
            return_first_valid=False,
        )
        planner = OfficialOMPLSTRRTStar(instance.env.copy(), int(seed), float(query.vlimit))
        ts = time.perf_counter()
        snapshots = planner.solve_with_snapshots(
            query.start,
            query.goal,
            t0=float(query.t_start),
            options=options,
        )
        measured_runtime = time.perf_counter() - ts
        solution = (
            snapshots.first_solution
            if planner_name == STPerformanceComparisonConfig.ST_RRT_FIRST_PLANNER
            else snapshots.final_solution
        )
        entry = STPerformanceComparisonConfig.entry_from_shortest_path_solution(
            solution,
            measured_runtime,
            budget,
        )
        return solution, entry, "SUCCESS" if solution.is_success else "FAIL"

    @staticmethod
    def _run_zeta_sipp_solution(
        instance,
        query,
        budget: float,
        seed: int,
        cell_size_multiplier: float,
    ) -> tuple[ShortestPathSolution, STHeuristicAblationResultEntry, str]:
        if int(instance.env.dim) != 2:
            raise ValueError("Zeta*-SIPP performance comparison is defined only for 2D ST-planning domains.")
        planner = ZetaStarSIPPPlanner(
            instance.env.copy(),
            int(seed),
            float(query.vlimit),
            cell_size=STPerformanceComparisonConfig.zeta_sipp_cell_size(instance, cell_size_multiplier),
            runtime_limit_secs=float(budget),
        )
        ts = time.perf_counter()
        solution = planner.solve(
            query.start,
            query.goal,
            t_start=float(query.t_start),
            t_max=STPlanningManifestStore.TMAX,
            is_stay=bool(query.is_stay),
        )
        entry = STPerformanceComparisonConfig.entry_from_shortest_path_solution(
            solution,
            time.perf_counter() - ts,
            budget,
        )
        return solution, entry, "SUCCESS" if solution.is_success else "FAIL"

    def run_st_solution(
        self,
        record: STHeuristicAblationRecord,
        planner_key: str,
        budget: float,
        seed_offset: int = 0,
    ) -> Dict[str, Any]:
        budget_value = self.positive_finite_float(budget, "Solution budget")
        planner_name = self._require_single_planner(planner_key)

        instance, query, base_manifest_path, base_record = STPlanningManifestStore.reconstruct_instance(
            record,
            self.base_root,
        )
        instance.base_record = base_record
        if planner_name in STPerformanceComparisonConfig.SEARCH_SPEC_BY_PLANNER:
            solution, entry, status = self._run_search_solution(instance, query, planner_name, budget_value)
        elif planner_name == STPerformanceComparisonConfig.MICP_PLANNER:
            solution, entry, status = self._run_micp_solution(
                instance,
                query,
                budget_value,
                max_rounded_paths=0,
            )
        elif planner_name == STPerformanceComparisonConfig.MICP_ROUNDING_PLANNER:
            solution, entry, status = self._run_micp_solution(
                instance,
                query,
                budget_value,
                max_rounded_paths=STPerformanceComparisonConfig.micp_rounding_paths(record.stgcs_num_edges),
            )
        elif planner_name in STPerformanceComparisonConfig.ST_RRT_OUTPUT_PLANNERS:
            seed = STPerformanceComparisonConfig.seed_for_record(record, seed_offset)
            solution, entry, status = self._run_strrt_solution(
                instance,
                query,
                budget_value,
                seed=seed,
                planner_name=planner_name,
            )
        elif planner_name == STPerformanceComparisonConfig.ZETA_SIPP_PLANNER:
            seed = STPerformanceComparisonConfig.seed_for_record(record, seed_offset)
            solution, entry, status = self._run_zeta_sipp_solution(
                instance,
                query,
                budget_value,
                seed=seed,
                cell_size_multiplier=1.0,
            )
        elif planner_name == STPerformanceComparisonConfig.ZETA_SIPP_2R_PLANNER:
            seed = STPerformanceComparisonConfig.seed_for_record(record, seed_offset)
            solution, entry, status = self._run_zeta_sipp_solution(
                instance,
                query,
                budget_value,
                seed=seed,
                cell_size_multiplier=2.0,
            )
        else:
            raise ValueError(f"Unknown ST solution planner {planner_name!r}.")

        return SolutionIO.run_to_viewer_json(
            record,
            ViewerSolutionRun(
                planner_key=str(planner_key),
                planner_name=planner_name,
                budget=budget_value,
                status=status,
                entry=entry,
                solution=solution,
            ),
        )

    @staticmethod
    def mrmp_low_level_spec(epsilon: float) -> SearchPlannerSpec:
        base = MRMPPerformanceComparison.LOW_LEVEL_SPEC
        return SearchPlannerSpec(
            heuristic=base.heuristic,
            dominance_checks=base.dominance_checks,
            epsilon=float(epsilon),
            exact_astar=base.exact_astar,
        )

    @classmethod
    def mrmp_planner_name(
        cls,
        planner_key: str,
        low_level_spec: SearchPlannerSpec,
        window_span_factor: float,
        execution_horizon_factor: float,
    ) -> str:
        low_level_name = low_level_spec.name
        alpha = float(window_span_factor)
        beta = float(execution_horizon_factor)
        if planner_key == MRMPPerformanceComparison.PBS_KEY:
            return f"PBS-{MRMPPerformanceComparison.CHILD_EXPANSION_LABEL} + {low_level_name}"
        if planner_key == MRMPPerformanceComparison.WINDOWED_PBS_KEY:
            return f"Windowed PBS(alpha={alpha:g},beta={beta:g},rule=NC) + {low_level_name}"
        if planner_key == MRMPPerformanceComparison.WINDOWED_PP_KEY:
            return f"Windowed PP(alpha={alpha:g},beta={beta:g},order=index) + {low_level_name}"
        if planner_key == MRMPPerformanceComparison.CB_GCS_KEY:
            return MRMPPerformanceComparison.planner_name(planner_key)
        if planner_key == MRMPPerformanceComparison.KCBS_KEY:
            return MRMPPerformanceComparison.planner_name(planner_key)
        raise KeyError(f"Unknown MRMP viewer planner {planner_key!r}.")

    def run_mrmp_solution(
        self,
        record: MRMPBenchmarkRecord,
        planner_key: str,
        budget: float,
        window_span_factor: float,
        execution_horizon_factor: float,
        epsilon: float,
    ) -> Dict[str, Any]:
        budget_value = self.positive_finite_float(budget, "Solution budget")
        alpha = self.positive_finite_float(window_span_factor, "MRMP window alpha")
        beta = self.unit_interval_float(execution_horizon_factor, "MRMP window beta")
        epsilon_value = self.positive_finite_float(epsilon, "MRMP low-level epsilon")
        low_level_spec = self.mrmp_low_level_spec(epsilon_value)
        planner_name = self.mrmp_planner_name(planner_key, low_level_spec, alpha, beta)

        base_manifest_path = BaseManifestStore.manifest_path(self.base_root, domain_key=record.domain_key)
        base_record = BaseManifestStore.record_by_id(base_manifest_path, record.base_instance_id)
        from experiments.mrmp import MRMPExperiment

        instance, queries = MRMPExperiment.reconstruct_instance(record, base_record, compute_heuristics=False)

        if planner_key in MRMPPerformanceComparison.SEARCH_BASED_PLANNER_KEYS:
            windowed = planner_key in {MRMPPerformanceComparison.WINDOWED_PBS_KEY,
                                       MRMPPerformanceComparison.WINDOWED_PP_KEY}
            method = "pp" if planner_key in {MRMPPerformanceComparison.PP_BFS_KEY,
                                             MRMPPerformanceComparison.WINDOWED_PP_KEY} else "pbs"
            solutions, entry = STGCSPlannerClient.plan(
                instance, queries, low_level_spec, budget_value, "num_conflicts",
                window_span_factor=alpha if windowed else 0.0,
                execution_horizon_factor=beta, dynamic_window_adjustment=windowed,
                method=method)
        elif planner_key == MRMPPerformanceComparison.CB_GCS_KEY:
            solutions, entry = MRMPExperiment.run_cb_gcs_with_solutions(
                instance,
                queries,
                budget=budget_value,
                time_horizon=MRMPPerformanceComparison.CB_GCS_TIME_HORIZON,
                time_step=MRMPPerformanceComparison.CB_GCS_TIME_STEP,
                max_rounded_paths=MRMPPerformanceComparison.micp_rounding_paths(record.stgcs_num_edges),
            )
        elif planner_key == MRMPPerformanceComparison.KCBS_KEY:
            solutions, entry = MRMPExperiment.run_ompl_kcbs_with_solutions(
                instance,
                queries,
                budget=budget_value,
                low_level_solve_time=MRMPPerformanceComparison.KCBS_LOW_LEVEL_SOLVE_TIME,
                propagation_step_size=MRMPPerformanceComparison.KCBS_PROPAGATION_STEP_SIZE,
                min_control_duration=MRMPPerformanceComparison.KCBS_MIN_CONTROL_DURATION,
                max_control_duration=MRMPPerformanceComparison.KCBS_MAX_CONTROL_DURATION,
                goal_tolerance=MRMPPerformanceComparison.KCBS_GOAL_TOLERANCE,
                goal_bias=MRMPPerformanceComparison.KCBS_GOAL_BIAS,
                intermediate_states=MRMPPerformanceComparison.KCBS_INTERMEDIATE_STATES,
                num_threads=MRMPPerformanceComparison.KCBS_NUM_THREADS,
                seed=MRMPPerformanceComparison.KCBS_SEED,
            )
        else:
            raise KeyError(f"Unknown MRMP viewer planner {planner_key!r}.")

        return SolutionIO.mrmp_run_to_viewer_json(
            record,
            ViewerMRMPSolutionRun(
                planner_key=str(planner_key),
                planner_name=planner_name,
                budget=budget_value,
                status="SUCCESS" if entry.is_success else "FAIL",
                entry=entry,
                solutions=solutions,
            ),
        )

    def run_solution(
        self,
        record: STHeuristicAblationRecord | MRMPBenchmarkRecord,
        planner_key: str,
        budget: float,
        seed_offset: int = 0,
        window_span_factor: float = MRMPPerformanceComparison.WINDOW_SPAN_FACTOR,
        execution_horizon_factor: float = MRMPPerformanceComparison.EXECUTION_HORIZON_FACTOR,
        epsilon: float = MRMPPerformanceComparison.LOW_LEVEL_SPEC.epsilon,
    ) -> Dict[str, Any]:
        if isinstance(record, STHeuristicAblationRecord):
            return self.run_st_solution(record, planner_key, budget, seed_offset=seed_offset)
        if isinstance(record, MRMPBenchmarkRecord):
            return self.run_mrmp_solution(
                record,
                planner_key,
                budget,
                window_span_factor=window_span_factor,
                execution_horizon_factor=execution_horizon_factor,
                epsilon=epsilon,
            )
        raise TypeError(f"Unsupported solution record type {type(record)!r}.")
