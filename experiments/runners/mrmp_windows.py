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
from experiments.planners.mrmp import (
    WindowedCoordinationAblation,
    WindowedCoordinationSpec,
)


SkipCallback = Callable[[float, WindowedCoordinationSpec], None]


class MRMPWindowedCoordinationRunCLI:
    ABLATION = WindowedCoordinationAblation

    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=(
                "Run dynamic MRMP windows with alpha=2,4,8,16,inf; "
                "alpha=inf runs full-horizon PBS."
            )
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
        parser.add_argument("--output-root", type=Path, default=Path(cls.ABLATION.DEFAULT_OUTPUT_ROOT))
        parser.add_argument("--manifest-root", type=Path, default=Path(cls.ABLATION.DEFAULT_MANIFEST_ROOT))
        parser.add_argument("--base-root", type=Path, default=Path("data/stgcs_base"))
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument(
            "--span-factors",
            "--factors",
            dest="span_factors",
            type=float,
            nargs="+",
            default=list(cls.ABLATION.DEFAULT_SPAN_FACTORS),
            help="Window span factors alpha for alpha * robot_radius / vlimit; inf uses the full horizon.",
        )
        parser.add_argument(
            "--domain",
            "--domains",
            dest="domains",
            nargs="+",
            choices=("all", *cls.ABLATION.DOMAINS),
            default=None,
            help="Domain(s) to run. Uses --manifest-root/manifest.json when present and filters it.",
        )
        args = parser.parse_args()
        for budget in args.budget:
            if not math.isfinite(float(budget)) or float(budget) <= 0.0:
                parser.error("--budget values must be finite and positive.")
        for factor in args.span_factors:
            if math.isnan(float(factor)) or float(factor) <= 0.0:
                parser.error("--span-factors values must be positive; inf selects the full horizon.")
        return args

    @classmethod
    def domains_for_arg(cls, domains: Sequence[str] | None) -> tuple[str, ...]:
        if domains is None or "all" in domains:
            return cls.ABLATION.DOMAINS
        return tuple(str(domain) for domain in domains)

    @classmethod
    def manifest_paths(cls, args: argparse.Namespace) -> tuple[Path, ...]:
        if args.manifest is not None:
            return (Path(args.manifest).resolve(),)
        manifest_root = Path(args.manifest_root)
        combined_manifest_path = manifest_root / "manifest.json"
        if combined_manifest_path.exists():
            return (combined_manifest_path.resolve(),)
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
        domain_key: str,
        specs: Sequence[WindowedCoordinationSpec],
    ) -> Dict[str, Path]:
        return {
            spec.key: MRMPExperiment.planner_output_path(
                output_root,
                domain_key,
                cls.ABLATION.planner_name(spec),
            )
            for spec in specs
        }

    @classmethod
    def completed_runs_by_spec(
        cls,
        output_root: str | Path,
        domain_key: str,
        specs: Sequence[WindowedCoordinationSpec],
    ) -> tuple[Dict[str, Path], Dict[str, set[tuple[str, float]]]]:
        output_paths = cls.result_paths(output_root, domain_key, specs)
        completed = {
            spec.key: cls.completed_result_keys(output_paths[spec.key])
            for spec in specs
        }
        return output_paths, completed

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
            return f"{domain_keys[0]} WC"
        return f"{manifest_path.stem} WC"

    @classmethod
    def pending_runs(
        cls,
        record: MRMPBenchmarkRecord,
        budgets: Sequence[float],
        specs: Sequence[WindowedCoordinationSpec],
        completed_runs: Dict[str, set[tuple[str, float]]],
        on_skip: SkipCallback | None = None,
    ) -> list[tuple[float, WindowedCoordinationSpec]]:
        pending: list[tuple[float, WindowedCoordinationSpec]] = []
        for budget in budgets:
            budget_value = float(budget)
            for spec in specs:
                key = cls.result_key(record.instance_id, budget_value)
                if key in completed_runs[spec.key]:
                    if on_skip is None:
                        print(
                            f"{record.instance_id} | {cls.ABLATION.planner_name(spec)} "
                            f"| budget={budget_value} | skipped (resume)"
                        )
                    else:
                        on_skip(budget_value, spec)
                    continue
                pending.append((budget_value, spec))
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
        spec: WindowedCoordinationSpec | None,
        last: str,
    ) -> None:
        state["record"] = cls._short_id(record.instance_id)
        state["budget"] = "-" if budget is None else f"{float(budget):g}"
        state["spec"] = "-" if spec is None else spec.label
        state["last"] = last
        progress.set_postfix_str(
            f"record={state['record']} spec={state['spec']} budget={state['budget']} "
            f"ok={state['ok']} fail={state['fail']} skip={state['skip']} last={state['last']}",
            refresh=True,
        )

    @classmethod
    def run_pending_spec(
        cls,
        record: MRMPBenchmarkRecord,
        base_record: BaseBenchmarkRecord,
        backend: STGCSPlannerClient,
        output_path: Path,
        budget: float,
        spec: WindowedCoordinationSpec,
    ) -> MRMPResultEntry:
        instance = None
        queries = None
        try:
            instance, queries = MRMPExperiment.reconstruct_instance(record, base_record, compute_heuristics=False)
            entry = backend.run(
                instance,
                queries,
                cls.ABLATION.LOW_LEVEL_SPEC,
                budget=budget,
                window_span_factor=spec.window_span_factor,
                dynamic_window_adjustment=spec.dynamic_window_adjustment,
                child_rule=spec.child_expansion_mode.name.lower(),
                execution_horizon_factor=spec.execution_horizon_factor,
            )
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
    def run_manifest(
        cls,
        args: argparse.Namespace,
        manifest_path: Path,
        specs: Sequence[WindowedCoordinationSpec],
        budgets: Sequence[float],
    ) -> None:
        manifest_path = Path(manifest_path).resolve()
        backend = STGCSPlannerClient.from_args(args)
        records = load_manifest(manifest_path)
        records = cls.filter_records_by_domains(records, getattr(args, "domains", None))
        if args.limit is not None:
            records = records[: args.limit]

        domain_keys = cls.domain_keys_for_records(records)
        base_manifest_paths = {
            domain_key: BaseManifestStore.manifest_path(args.base_root, domain_key=domain_key)
            for domain_key in domain_keys
        }
        output_paths_by_domain: dict[str, Dict[str, Path]] = {}
        completed_runs_by_domain: dict[str, Dict[str, set[tuple[str, float]]]] = {}
        for domain_key in domain_keys:
            output_paths, completed_runs = cls.completed_runs_by_spec(args.output_root, domain_key, specs)
            output_paths_by_domain[domain_key] = output_paths
            completed_runs_by_domain[domain_key] = completed_runs

        total_runs = len(records) * len(budgets) * len(specs)
        with tqdm(
            total=total_runs,
            desc=cls.progress_description(manifest_path, records),
            unit="run",
            dynamic_ncols=True,
            disable=cls.progress_disabled(),
        ) as progress:
            state = {
                "record": "-",
                "spec": "-",
                "budget": "-",
                "ok": 0,
                "fail": 0,
                "skip": 0,
                "last": "start",
            }

            for record in records:
                domain_key = record.domain_key
                base_manifest_path = base_manifest_paths[domain_key]
                output_paths = output_paths_by_domain[domain_key]
                completed_runs = completed_runs_by_domain[domain_key]

                def on_skip(budget: float, spec: WindowedCoordinationSpec, *, current_record=record) -> None:
                    state["skip"] += 1
                    cls.update_progress_status(progress, state, current_record, budget, spec, "skip:resume")
                    progress.update(1)

                pending_runs = cls.pending_runs(record, budgets, specs, completed_runs, on_skip=on_skip)
                if not pending_runs:
                    continue

                cls.update_progress_status(progress, state, record, None, None, f"load:{len(pending_runs)}")
                base_record = BaseManifestStore.record_by_id(base_manifest_path, record.base_instance_id)
                for budget, spec in pending_runs:
                    cls.update_progress_status(progress, state, record, budget, spec, "running")
                    entry = cls.run_pending_spec(
                        record,
                        base_record,
                        backend,
                        output_paths[spec.key],
                        budget=budget,
                        spec=spec,
                    )
                    completed_runs[spec.key].add(cls.result_key(record.instance_id, budget))
                    if entry.is_success:
                        state["ok"] += 1
                        last = f"ok:calls={entry.pbs_calls}"
                    else:
                        state["fail"] += 1
                        last = f"fail:calls={entry.pbs_calls}"
                    cls.update_progress_status(progress, state, record, budget, spec, last)
                    progress.update(1)

    @classmethod
    def specs_from_args(cls, args: argparse.Namespace) -> tuple[WindowedCoordinationSpec, ...]:
        return cls.ABLATION.specs(args.span_factors)

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        specs = cls.specs_from_args(args)
        budgets = tuple(float(budget) for budget in args.budget)
        for manifest_path in cls.manifest_paths(args):
            cls.run_manifest(args, manifest_path, specs, budgets)


if __name__ == "__main__":
    MRMPWindowedCoordinationRunCLI.main()
