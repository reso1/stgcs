from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Callable, Dict, Sequence

from tqdm.auto import tqdm

from experiments.base import BaseManifestStore
from experiments.mrmp import MRMPExperiment
from experiments.native import STGCSPlannerClient
from experiments.manifests.mrmp import MRMPBenchmarkRecord, load_manifest
from experiments.planners.mrmp import PBSExpansionAblation, PBSExpansionRuleSpec


SkipCallback = Callable[[float, PBSExpansionRuleSpec], None]


class MRMPRunCLI:
    PBS_PLANNER_KEY = "pbs"
    ALL_PLANNER_KEY = "all"
    PLANNER_KEYS = (PBS_PLANNER_KEY,)
    DOMAIN_KEYS = ("grid2d", "grid3d", "maze", "iris-2d", "simple2d", "empty-square2d")

    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Run MRMP full-horizon PBS experiments on an MRMP manifest."
        )
        STGCSPlannerClient.add_arguments(parser)
        parser.add_argument("manifest", type=Path, nargs="?",
                            default=Path(PBSExpansionAblation.DEFAULT_MANIFEST_PATH),
                            help="MRMP manifest (default: current 150-instance, 10-robot benchmark).")
        parser.add_argument("--budget", type=float, nargs="+",
                            default=(PBSExpansionAblation.DEFAULT_BUDGET,),
                            help="Runtime budgets in seconds (default: 60).")
        parser.add_argument("--output-root", type=Path, default=None)
        parser.add_argument("--base-root", type=Path, default=Path("data/stgcs_base"))
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument(
            "--planner",
            nargs="+",
            choices=(MRMPRunCLI.ALL_PLANNER_KEY, *MRMPRunCLI.PLANNER_KEYS),
            default=(MRMPRunCLI.PBS_PLANNER_KEY,),
            help="MRMP planners to run: pbs runs the PBS rule set.",
        )
        parser.add_argument(
            "--rules",
            nargs="+",
            choices=[rule.key for rule in PBSExpansionAblation.RULES],
            default=[rule.key for rule in PBSExpansionAblation.RULES],
        )
        args = parser.parse_args()
        for budget in args.budget:
            if not math.isfinite(float(budget)) or float(budget) <= 0.0:
                parser.error("--budget values must be finite and positive.")
        args.planners = MRMPRunCLI.planner_keys(args.planner)
        if args.output_root is None:
            args.output_root = MRMPRunCLI.default_output_root(args.planners, args.manifest)
        return args

    @classmethod
    def planner_keys(cls, requested: Sequence[str]) -> tuple[str, ...]:
        if cls.ALL_PLANNER_KEY in requested:
            return cls.PLANNER_KEYS
        selected: list[str] = []
        for key in requested:
            if key not in cls.PLANNER_KEYS:
                raise ValueError(f"Unknown MRMP planner key {key!r}.")
            if key not in selected:
                selected.append(key)
        if not selected:
            raise ValueError("At least one MRMP planner must be selected.")
        return tuple(selected)

    @classmethod
    def default_output_root(cls, planner_keys: Sequence[str], manifest_path: str | Path | None = None) -> Path:
        if manifest_path is not None:
            return cls.default_output_root_from_manifest(manifest_path)
        return Path(PBSExpansionAblation.DEFAULT_OUTPUT_ROOT)

    @classmethod
    def default_output_root_from_manifest(cls, manifest_path: str | Path) -> Path:
        path = Path(manifest_path).resolve()
        if path.name != "manifest.json":
            return Path(PBSExpansionAblation.DEFAULT_OUTPUT_ROOT)
        parts = path.parts
        for idx, part in enumerate(parts[:-1]):
            if part == "data" and idx + 1 < len(parts) and parts[idx + 1] == "instances":
                result_path = Path(*parts[: idx + 1]) / "exp_res" / Path(*parts[idx + 2:])
                result_root = result_path.parent
                if result_root.name in cls.DOMAIN_KEYS:
                    return result_root.parent
                return result_root
        if path.parent.name in cls.DOMAIN_KEYS:
            return path.parent.parent
        return path.parent

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
        rules: Sequence[PBSExpansionRuleSpec],
    ) -> Dict[str, Path]:
        return {
            rule.key: MRMPExperiment.planner_output_path(
                output_root,
                domain_key,
                PBSExpansionAblation.planner_name(rule),
            )
            for rule in rules
        }

    @classmethod
    def completed_runs_by_rule(
        cls,
        output_root: str | Path,
        domain_key: str,
        rules: Sequence[PBSExpansionRuleSpec],
    ) -> tuple[Dict[str, Path], Dict[str, set[tuple[str, float]]]]:
        output_paths = cls.result_paths(output_root, domain_key, rules)
        completed = {
            rule.key: cls.completed_result_keys(output_paths[rule.key])
            for rule in rules
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
            return f"{domain_keys[0]} MRMP"
        return f"{manifest_path.stem} MRMP"

    @classmethod
    def pending_runs(
        cls,
        record: MRMPBenchmarkRecord,
        budgets: Sequence[float],
        rules: Sequence[PBSExpansionRuleSpec],
        completed_runs: Dict[str, set[tuple[str, float]]],
        on_skip: SkipCallback | None = None,
    ) -> list[tuple[float, PBSExpansionRuleSpec]]:
        pending: list[tuple[float, PBSExpansionRuleSpec]] = []
        for budget in budgets:
            budget_value = float(budget)
            for rule in rules:
                key = cls.result_key(record.instance_id, budget_value)
                if key in completed_runs[rule.key]:
                    if on_skip is None:
                        print(
                            f"{record.instance_id} | {PBSExpansionAblation.planner_name(rule)} "
                            f"| budget={budget_value} | skipped (resume)"
                        )
                    else:
                        on_skip(budget_value, rule)
                    continue
                pending.append((budget_value, rule))
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
        planner_label: str | None,
        last: str,
    ) -> None:
        state["record"] = cls._short_id(record.instance_id)
        state["budget"] = "-" if budget is None else f"{float(budget):g}"
        state["planner"] = "-" if planner_label is None else planner_label
        state["last"] = last
        progress.set_postfix_str(
            f"record={state['record']} planner={state['planner']} budget={state['budget']} "
            f"ok={state['ok']} fail={state['fail']} skip={state['skip']} last={state['last']}",
            refresh=True,
        )

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        backend = STGCSPlannerClient.from_args(args)
        manifest_path = Path(args.manifest).resolve()
        records = load_manifest(manifest_path)
        if args.limit is not None:
            records = records[: args.limit]

        planner_keys = tuple(args.planners)
        run_pbs = cls.PBS_PLANNER_KEY in planner_keys
        rules = [PBSExpansionAblation.rule_by_key(key) for key in args.rules] if run_pbs else []
        budgets = tuple(float(budget) for budget in args.budget)
        domain_keys = cls.domain_keys_for_records(records)
        base_manifest_paths = {
            domain_key: BaseManifestStore.manifest_path(args.base_root, domain_key=domain_key)
            for domain_key in domain_keys
        }
        output_paths_by_domain: dict[str, Dict[str, Path]] = {}
        completed_runs_by_domain: dict[str, Dict[str, set[tuple[str, float]]]] = {}
        if run_pbs:
            for domain_key in domain_keys:
                output_paths, completed_runs = cls.completed_runs_by_rule(args.output_root, domain_key, rules)
                output_paths_by_domain[domain_key] = output_paths
                completed_runs_by_domain[domain_key] = completed_runs

        total_runs = len(records) * len(budgets) * len(rules)
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
                output_paths = output_paths_by_domain.get(domain_key, {})
                completed_runs = completed_runs_by_domain.get(domain_key, {})

                def on_skip(budget: float, rule: PBSExpansionRuleSpec, *, current_record=record) -> None:
                    state["skip"] += 1
                    cls.update_progress_status(progress, state, current_record, budget, rule.label, "skip:resume")
                    progress.update(1)

                pending_runs = cls.pending_runs(record, budgets, rules, completed_runs, on_skip=on_skip)
                if not pending_runs:
                    continue

                cls.update_progress_status(progress, state, record, None, None, f"load:{len(pending_runs)}")
                base_record = BaseManifestStore.record_by_id(base_manifest_path, record.base_instance_id)
                instance, queries = MRMPExperiment.reconstruct_instance(record, base_record, compute_heuristics=False)
                for budget, rule in pending_runs:
                    cls.update_progress_status(progress, state, record, budget, rule.label, "running")
                    entry = backend.run(
                        instance,
                        queries,
                        PBSExpansionAblation.LOW_LEVEL_SPEC,
                        budget=budget,
                        child_rule=rule.key,
                    )
                    MRMPExperiment.append_result_row(
                        output_paths[rule.key],
                        instance.name,
                        record,
                        budget,
                        entry,
                    )
                    completed_runs[rule.key].add(cls.result_key(record.instance_id, budget))
                    if entry.is_success:
                        state["ok"] += 1
                        last = f"ok:nodes={entry.pbs_popped_nodes}"
                    else:
                        state["fail"] += 1
                        last = f"fail:nodes={entry.pbs_popped_nodes}"
                    cls.update_progress_status(progress, state, record, budget, rule.label, last)
                    progress.update(1)


if __name__ == "__main__":
    MRMPRunCLI.main()
