from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
from typing import Callable, Dict, Sequence

from tqdm.auto import tqdm

from experiments.base import BaseManifestStore
from experiments.manifests.base import BaseBenchmarkRecord
from experiments.mrmp import MRMPExperiment, MRMPResultEntry
from experiments.native import STGCSPlannerClient
from experiments.manifests.mrmp import MRMPBenchmarkRecord, load_manifest
from experiments.planners.mrmp import MRMPPerformanceComparison


SkipCallback = Callable[[float, str], None]


class MRMPPerformanceComparisonRunCLI:
    BENCHMARK = MRMPPerformanceComparison

    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Run the MRMP performance comparison over 2, 4, ..., 20-robot MRMP manifests."
        )
        STGCSPlannerClient.add_arguments(parser)
        parser.add_argument(
            "manifest",
            type=Path,
            nargs="?",
            default=None,
            help="Optional manifest path. Omit it to use --manifest-root and optional --domain filtering.",
        )
        parser.add_argument("--budget", type=float, nargs="+", required=True)
        parser.add_argument("--output-root", type=Path, default=Path(cls.BENCHMARK.DEFAULT_OUTPUT_ROOT))
        parser.add_argument("--manifest-root", type=Path, default=Path(cls.BENCHMARK.DEFAULT_MANIFEST_ROOT))
        parser.add_argument("--base-root", type=Path, default=Path("data/stgcs_base"))
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument(
            "--planner",
            nargs="+",
            choices=cls.BENCHMARK.PLANNER_SELECTION_KEYS,
            default=cls.BENCHMARK.DEFAULT_PLANNER_KEYS,
            help="MRMP performance-comparison planners to run.",
        )
        parser.add_argument(
            "--domain",
            "--domains",
            dest="domains",
            nargs="+",
            choices=("all", *cls.BENCHMARK.DOMAINS),
            default=None,
            help="Domain(s) to run. Uses --manifest-root/manifest.json when present and filters it.",
        )
        args = parser.parse_args()
        for budget in args.budget:
            if not math.isfinite(float(budget)) or float(budget) <= 0.0:
                parser.error("--budget values must be finite and positive.")
        if args.limit is not None and int(args.limit) <= 0:
            parser.error("--limit must be positive when provided.")
        args.planners = cls.planner_keys(args.planner)
        return args

    @classmethod
    def planner_keys(cls, requested: Sequence[str]) -> tuple[str, ...]:
        if cls.BENCHMARK.ALL_PLANNER_KEY in requested:
            return cls.BENCHMARK.PLANNER_KEYS
        selected: list[str] = []
        for key in requested:
            if key not in cls.BENCHMARK.PLANNER_SELECTION_KEYS:
                raise ValueError(f"Unknown MRMP performance-comparison planner {key!r}.")
            for expanded_key in cls.BENCHMARK.expand_planner_key(key):
                if expanded_key not in selected:
                    selected.append(expanded_key)
        if not selected:
            raise ValueError("At least one MRMP performance-comparison planner must be selected.")
        return tuple(selected)

    @classmethod
    def domains_for_arg(cls, domains: Sequence[str] | None) -> tuple[str, ...]:
        if domains is None or "all" in domains:
            return cls.BENCHMARK.DOMAINS
        return tuple(str(domain) for domain in domains)

    @staticmethod
    def numbered_manifest_sort_key(path: Path) -> tuple[int, str]:
        stem = path.stem
        prefix = "manifest_n"
        if stem.startswith(prefix):
            try:
                return int(stem[len(prefix):]), path.name
            except ValueError:
                pass
        return 10**9, path.name

    @classmethod
    def manifest_paths(cls, args: argparse.Namespace) -> tuple[Path, ...]:
        if args.manifest is not None:
            return (Path(args.manifest).resolve(),)
        manifest_root = Path(args.manifest_root)
        combined_manifest_path = manifest_root / "manifest.json"
        if combined_manifest_path.exists():
            return (combined_manifest_path.resolve(),)
        numbered_manifest_paths = sorted(
            manifest_root.glob("manifest_n*.json"),
            key=cls.numbered_manifest_sort_key,
        )
        if numbered_manifest_paths:
            return tuple(path.resolve() for path in numbered_manifest_paths)
        return tuple(
            (manifest_root / domain / "manifest.json").resolve()
            for domain in cls.domains_for_arg(getattr(args, "domains", None))
        )

    @classmethod
    def filter_records_by_domains(
        cls,
        records: Sequence[MRMPBenchmarkRecord],
        domains: Sequence[str] | None,
    ) -> list[MRMPBenchmarkRecord]:
        if domains is None or "all" in domains:
            return list(records)
        requested_domains = set(cls.domains_for_arg(domains))
        return [record for record in records if record.domain_key in requested_domains]

    @classmethod
    def result_key(cls, instance_id: str, budget: float) -> tuple[str, float]:
        return instance_id, float(budget)

    @classmethod
    def completed_result_keys(cls, output_path: Path) -> set[tuple[str, float]]:
        if not output_path.exists():
            return set()
        return {
            cls.result_key(str(row["instance_id"]), float(row["budget"]))
            for row in MRMPExperiment.load_result_rows(output_path)
        }

    @classmethod
    def result_paths(
        cls,
        output_root: str | Path,
        num_agents: int,
        domain_key: str,
        planner_keys: Sequence[str],
    ) -> Dict[str, Path]:
        return {
            planner_key: cls.BENCHMARK.result_path(
                output_root,
                num_agents,
                domain_key,
                planner_key,
            )
            for planner_key in planner_keys
        }

    @classmethod
    def completed_runs_by_planner(
        cls,
        output_root: str | Path,
        num_agents: int,
        domain_key: str,
        planner_keys: Sequence[str],
    ) -> tuple[Dict[str, Path], Dict[str, set[tuple[str, float]]]]:
        output_paths = cls.result_paths(output_root, num_agents, domain_key, planner_keys)
        completed = {
            planner_key: cls.completed_result_keys(output_paths[planner_key])
            for planner_key in planner_keys
        }
        return output_paths, completed

    @classmethod
    def record_result_group(cls, record: MRMPBenchmarkRecord) -> tuple[int, str]:
        return record.num_agents, record.domain_key

    @classmethod
    def domain_keys_for_records(cls, records: Sequence[MRMPBenchmarkRecord]) -> tuple[str, ...]:
        domain_keys: list[str] = []
        for record in records:
            if record.domain_key not in domain_keys:
                domain_keys.append(record.domain_key)
        return tuple(domain_keys)

    @classmethod
    def progress_description(cls, manifest_path: Path, records: Sequence[MRMPBenchmarkRecord]) -> str:
        domain_keys = cls.domain_keys_for_records(records)
        if len(domain_keys) == 1:
            return f"{domain_keys[0]} MRMP perf"
        return f"{manifest_path.stem} MRMP perf"

    @classmethod
    def pending_runs(
        cls,
        record: MRMPBenchmarkRecord,
        budgets: Sequence[float],
        planner_keys: Sequence[str],
        completed_runs: Dict[str, set[tuple[str, float]]],
        on_skip: SkipCallback | None = None,
    ) -> list[tuple[float, str]]:
        pending: list[tuple[float, str]] = []
        for budget in budgets:
            budget_value = float(budget)
            for planner_key in planner_keys:
                key = cls.result_key(record.instance_id, budget_value)
                if key in completed_runs[planner_key]:
                    if on_skip is None:
                        print(
                            f"{record.instance_id} | {cls.BENCHMARK.planner_name(planner_key)} "
                            f"| budget={budget_value:g} | skipped (resume)"
                        )
                    else:
                        on_skip(budget_value, planner_key)
                    continue
                pending.append((budget_value, planner_key))
        return pending

    @staticmethod
    def progress_disabled() -> bool:
        return os.environ.get("STGCS_DISABLE_PROGRESS", "") == "1"

    @staticmethod
    def _short_id(instance_id: str, max_len: int = 36) -> str:
        if len(instance_id) <= max_len:
            return instance_id
        return "..." + instance_id[-(max_len - 3):]

    @classmethod
    def update_progress_status(
        cls,
        progress,
        state: dict,
        record: MRMPBenchmarkRecord,
        budget: float | None,
        planner_key: str | None,
        last: str,
    ) -> None:
        state["record"] = cls._short_id(record.instance_id)
        state["budget"] = "-" if budget is None else f"{float(budget):g}"
        state["planner"] = "-" if planner_key is None else planner_key
        state["last"] = last
        progress.set_postfix_str(
            f"record={state['record']} planner={state['planner']} budget={state['budget']} "
            f"ok={state['ok']} fail={state['fail']} skip={state['skip']} last={state['last']}",
            refresh=True,
        )

    @classmethod
    def run_pending_planner(
        cls,
        record: MRMPBenchmarkRecord,
        base_record: BaseBenchmarkRecord,
        output_path: Path,
        budget: float,
        planner_key: str,
        backend: STGCSPlannerClient,
    ) -> MRMPResultEntry:
        instance = None
        queries = None
        try:
            instance, queries = MRMPExperiment.reconstruct_instance(record, base_record, compute_heuristics=False)
            if planner_key in cls.BENCHMARK.SEARCH_BASED_PLANNER_KEYS:
                windowed = planner_key in (cls.BENCHMARK.WINDOWED_PBS_KEY, cls.BENCHMARK.WINDOWED_PP_KEY)
                entry = backend.run(
                    instance, queries, cls.BENCHMARK.low_level_spec(planner_key), budget=budget,
                    child_rule="num_conflicts",
                    method="pp" if planner_key in (cls.BENCHMARK.PP_BFS_KEY, cls.BENCHMARK.WINDOWED_PP_KEY) else "pbs",
                    window_span_factor=cls.BENCHMARK.WINDOW_SPAN_FACTOR if windowed else 0.0,
                    dynamic_window_adjustment=cls.BENCHMARK.DYNAMIC_WINDOW_ADJUSTMENT if windowed else False,
                    execution_horizon_factor=cls.BENCHMARK.EXECUTION_HORIZON_FACTOR,
                )
            elif planner_key == cls.BENCHMARK.PBS_ZETA_SIPP_KEY:
                entry = MRMPExperiment.run_pbs_zeta_sipp(
                    instance,
                    queries,
                    budget=budget,
                    child_expansion_mode=cls.BENCHMARK.CHILD_EXPANSION_MODE,
                    cell_size_multiplier=cls.BENCHMARK.ZETA_SIPP_CELL_SIZE_MULTIPLIER,
                )
            elif planner_key == cls.BENCHMARK.CB_GCS_KEY:
                entry = MRMPExperiment.run_cb_gcs(
                    instance,
                    queries,
                    budget=budget,
                    time_horizon=cls.BENCHMARK.CB_GCS_TIME_HORIZON,
                    time_step=cls.BENCHMARK.CB_GCS_TIME_STEP,
                    max_rounded_paths=cls.BENCHMARK.micp_rounding_paths(record.stgcs_num_edges),
                )
            elif planner_key == cls.BENCHMARK.KCBS_KEY:
                entry = MRMPExperiment.run_ompl_kcbs(
                    instance,
                    queries,
                    budget=budget,
                    low_level_solve_time=cls.BENCHMARK.KCBS_LOW_LEVEL_SOLVE_TIME,
                    propagation_step_size=cls.BENCHMARK.KCBS_PROPAGATION_STEP_SIZE,
                    min_control_duration=cls.BENCHMARK.KCBS_MIN_CONTROL_DURATION,
                    max_control_duration=cls.BENCHMARK.KCBS_MAX_CONTROL_DURATION,
                    goal_tolerance=cls.BENCHMARK.KCBS_GOAL_TOLERANCE,
                    goal_bias=cls.BENCHMARK.KCBS_GOAL_BIAS,
                    intermediate_states=cls.BENCHMARK.KCBS_INTERMEDIATE_STATES,
                    num_threads=cls.BENCHMARK.KCBS_NUM_THREADS,
                    seed=cls.BENCHMARK.KCBS_SEED,
                )
            else:
                raise KeyError(f"Unknown MRMP performance-comparison planner {planner_key!r}")

            MRMPExperiment.append_result_row(
                output_path,
                instance.name,
                record,
                budget,
                entry,
            )
            return entry
        finally:
            del queries
            del instance
            gc.collect()

    @classmethod
    def run_pending_fixed_priority_strrt_star(
        cls,
        record: MRMPBenchmarkRecord,
        base_record: BaseBenchmarkRecord,
        budget: float,
    ) -> tuple[MRMPResultEntry, MRMPResultEntry, str]:
        instance = None
        queries = None
        try:
            instance, queries = MRMPExperiment.reconstruct_instance(record, base_record, compute_heuristics=False)
            first_entry, final_entry = MRMPExperiment.run_fixed_priority_strrt_star_snapshots(
                instance,
                queries,
                budget=budget,
            )
            return first_entry, final_entry, instance.name
        finally:
            del queries
            del instance
            gc.collect()

    @classmethod
    def run_manifest(
        cls,
        args: argparse.Namespace,
        manifest_path: Path,
        planner_keys: Sequence[str],
        budgets: Sequence[float],
        backend: STGCSPlannerClient,
    ) -> None:
        manifest_path = Path(manifest_path).resolve()
        records = load_manifest(manifest_path)
        records = cls.filter_records_by_domains(records, getattr(args, "domains", None))
        if args.limit is not None:
            records = records[: int(args.limit)]

        domain_keys = cls.domain_keys_for_records(records)
        base_manifest_paths = {
            domain_key: BaseManifestStore.manifest_path(args.base_root, domain_key=domain_key)
            for domain_key in domain_keys
        }
        output_paths_by_group: dict[tuple[int, str], Dict[str, Path]] = {}
        completed_runs_by_group: dict[tuple[int, str], Dict[str, set[tuple[str, float]]]] = {}
        for record in records:
            group = cls.record_result_group(record)
            if group in output_paths_by_group:
                continue
            num_agents, domain_key = group
            output_paths, completed_runs = cls.completed_runs_by_planner(
                args.output_root,
                num_agents,
                domain_key,
                planner_keys,
            )
            output_paths_by_group[group] = output_paths
            completed_runs_by_group[group] = completed_runs

        total_runs = len(records) * len(budgets) * len(planner_keys)
        with tqdm(
            total=total_runs,
            desc=cls.progress_description(manifest_path, records),
            unit="run",
            dynamic_ncols=True,
            disable=cls.progress_disabled(),
        ) as progress:
            state = {
                "record": "-",
                "planner": "-",
                "budget": "-",
                "ok": 0,
                "fail": 0,
                "skip": 0,
                "last": "start",
            }

            for record in records:
                domain_key = record.domain_key
                base_manifest_path = base_manifest_paths[domain_key]
                group = cls.record_result_group(record)
                output_paths = output_paths_by_group[group]
                completed_runs = completed_runs_by_group[group]

                def on_skip(budget: float, planner_key: str, *, current_record=record) -> None:
                    state["skip"] += 1
                    cls.update_progress_status(progress, state, current_record, budget, planner_key, "skip:resume")
                    progress.update(1)

                pending_runs = cls.pending_runs(record, budgets, planner_keys, completed_runs, on_skip=on_skip)
                if not pending_runs:
                    continue

                cls.update_progress_status(progress, state, record, None, None, f"load:{len(pending_runs)}")
                base_record = BaseManifestStore.record_by_id(base_manifest_path, record.base_instance_id)
                handled_runs: set[tuple[float, str]] = set()
                for budget, planner_key in pending_runs:
                    budget_value = float(budget)
                    if (budget_value, planner_key) in handled_runs:
                        continue
                    if planner_key in cls.BENCHMARK.FIXED_PP_ST_RRT_STAR_OUTPUT_KEYS:
                        snapshot_keys = [
                            pending_key for pending_budget, pending_key in pending_runs
                            if (
                                math.isclose(float(pending_budget), budget_value)
                                and pending_key in cls.BENCHMARK.FIXED_PP_ST_RRT_STAR_OUTPUT_KEYS
                            )
                        ]
                        cls.update_progress_status(
                            progress,
                            state,
                            record,
                            budget_value,
                            cls.BENCHMARK.FIXED_PP_ST_RRT_STAR_KEY,
                            "running",
                        )
                        first_entry, final_entry, instance_name = cls.run_pending_fixed_priority_strrt_star(
                            record,
                            base_record,
                            budget=budget_value,
                        )
                        entries_by_key = {
                            cls.BENCHMARK.FIXED_PP_ST_RRT_STAR_FIRST_KEY: first_entry,
                            cls.BENCHMARK.FIXED_PP_ST_RRT_STAR_FINAL_KEY: final_entry,
                        }
                        for snapshot_key in snapshot_keys:
                            entry = entries_by_key[snapshot_key]
                            MRMPExperiment.append_result_row(
                                output_paths[snapshot_key],
                                instance_name,
                                record,
                                budget_value,
                                entry,
                            )
                            completed_runs[snapshot_key].add(cls.result_key(record.instance_id, budget_value))
                            handled_runs.add((budget_value, snapshot_key))
                            if entry.is_success:
                                state["ok"] += 1
                                last = f"ok:agents={entry.num_completed_agents}"
                            else:
                                state["fail"] += 1
                                last = f"fail:agents={entry.num_completed_agents}"
                            cls.update_progress_status(progress, state, record, budget_value, snapshot_key, last)
                            progress.update(1)
                        continue

                    cls.update_progress_status(progress, state, record, budget, planner_key, "running")
                    entry = cls.run_pending_planner(
                        record,
                        base_record,
                        output_paths[planner_key],
                        budget=budget,
                        planner_key=planner_key,
                        backend=backend,
                    )
                    completed_runs[planner_key].add(cls.result_key(record.instance_id, budget))
                    if entry.is_success:
                        state["ok"] += 1
                        last = f"ok:agents={entry.num_completed_agents}"
                    else:
                        state["fail"] += 1
                        last = f"fail:agents={entry.num_completed_agents}"
                    cls.update_progress_status(progress, state, record, budget, planner_key, last)
                    progress.update(1)

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        backend = STGCSPlannerClient.from_args(args)
        budgets = tuple(float(budget) for budget in args.budget)
        for manifest_path in cls.manifest_paths(args):
            cls.run_manifest(args, manifest_path, args.planners, budgets, backend)


if __name__ == "__main__":
    MRMPPerformanceComparisonRunCLI.main()
