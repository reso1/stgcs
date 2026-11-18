from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from tqdm.auto import tqdm

from experiments.datasets.base import ManifestLock
from experiments.base import BaseManifestStore
from experiments.manifests.base import BaseBenchmarkRecord, load_manifest as load_base_manifest
from experiments.native import STGCSPlannerClient
from experiments.datasets.mrmp_queries import (
    MRMPPerformanceComparisonManifestBuilder,
    PBSExpansionMRMPManifestBuilder,
    WindowedCoordinationMRMPManifestBuilder,
)
from experiments.manifests.mrmp import MRMPBenchmarkRecord, load_manifest as load_mrmp_manifest, save_manifest
from experiments.planners.mrmp import MRMPPerformanceComparison, PBSExpansionAblation, WindowedCoordinationAblation


class MRMPManifestCLI:
    PBS_NODE_EXPANSION_BENCHMARK = "pbs-node-expansion"
    WINDOWED_COORDINATION_BENCHMARK = "windowed-coordination"
    PERFORMANCE_COMPARISON_BENCHMARK = "performance-comparison"
    DEFAULT_BENCHMARK = PBS_NODE_EXPANSION_BENCHMARK
    DOMAINS = ("grid2d", "maze", "iris-2d")
    DOMAIN_ALIASES = {
        "grid2d": "grid2d",
        "maze": "maze",
        "maze2d": "maze",
        "iris-2d": "iris-2d",
        "iris2d": "iris-2d",
    }
    DEFAULT_RECORDS_PER_DOMAIN = 50
    DEFAULT_MIN_GRID_SIZE = 3
    WINDOWED_COORDINATION_GRID2D_SIZE = 4
    PERFORMANCE_COMPARISON_GRID2D_SIZE = 4

    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Build MRMP manifests for ablation and performance-comparison benchmarks."
        )
        parser.add_argument("domain", choices=("all", "grid2d", "maze", "maze2d", "iris-2d", "iris2d"))
        parser.add_argument("--base-root", type=Path, default=Path("data/stgcs_base"))
        parser.add_argument("--native-executable", type=Path, default=Path("stgcs_cplusplus/build/bin/stgcs_planner"))
        parser.add_argument("--native-work-dir", type=Path, default=Path("build/mrmp-manifest"))
        parser.add_argument(
            "--benchmark",
            choices=(
                cls.PBS_NODE_EXPANSION_BENCHMARK,
                cls.WINDOWED_COORDINATION_BENCHMARK,
                cls.PERFORMANCE_COMPARISON_BENCHMARK,
            ),
            default=cls.DEFAULT_BENCHMARK,
        )
        parser.add_argument(
            "--replicates-per-robot-count",
            "--k",
            dest="replicates_per_robot_count",
            type=int,
            default=1,
            help="For performance-comparison manifests, generate this many records for each even robot count 2..20.",
        )
        parser.add_argument("--output-root", type=Path, default=None)
        args = parser.parse_args()
        if int(args.replicates_per_robot_count) <= 0:
            parser.error("--replicates-per-robot-count/--k must be positive.")
        if args.output_root is None:
            args.output_root = cls.default_output_root(args.benchmark)
        return args

    @classmethod
    def default_output_root(cls, benchmark: str) -> Path:
        benchmark = cls.canonical_benchmark(benchmark)
        if benchmark == cls.PBS_NODE_EXPANSION_BENCHMARK:
            return Path(PBSExpansionAblation.DEFAULT_MANIFEST_ROOT)
        if benchmark == cls.WINDOWED_COORDINATION_BENCHMARK:
            return Path(WindowedCoordinationAblation.DEFAULT_MANIFEST_ROOT)
        if benchmark == cls.PERFORMANCE_COMPARISON_BENCHMARK:
            return Path(MRMPPerformanceComparison.DEFAULT_MANIFEST_ROOT)
        raise AssertionError(f"Unhandled MRMP benchmark {benchmark!r}.")

    @classmethod
    def canonical_benchmark(cls, benchmark: str | None) -> str:
        if benchmark is None:
            return cls.DEFAULT_BENCHMARK
        benchmark = str(benchmark)
        if benchmark in {cls.PBS_NODE_EXPANSION_BENCHMARK, "pbs"}:
            return cls.PBS_NODE_EXPANSION_BENCHMARK
        if benchmark in {cls.WINDOWED_COORDINATION_BENCHMARK, "wc", "windowed"}:
            return cls.WINDOWED_COORDINATION_BENCHMARK
        if benchmark in {cls.PERFORMANCE_COMPARISON_BENCHMARK, "performance", "perf"}:
            return cls.PERFORMANCE_COMPARISON_BENCHMARK
        raise ValueError(f"Unsupported MRMP benchmark {benchmark!r}.")

    @classmethod
    def builder_for_benchmark(cls, benchmark: str | None) -> type[PBSExpansionMRMPManifestBuilder]:
        benchmark = cls.canonical_benchmark(benchmark)
        if benchmark == cls.PBS_NODE_EXPANSION_BENCHMARK:
            return PBSExpansionMRMPManifestBuilder
        if benchmark == cls.WINDOWED_COORDINATION_BENCHMARK:
            return WindowedCoordinationMRMPManifestBuilder
        if benchmark == cls.PERFORMANCE_COMPARISON_BENCHMARK:
            return MRMPPerformanceComparisonManifestBuilder
        raise AssertionError(f"Unhandled MRMP benchmark {benchmark!r}.")

    @classmethod
    def canonical_domain(cls, domain: str) -> str:
        try:
            return cls.DOMAIN_ALIASES[str(domain)]
        except KeyError as exc:
            raise ValueError(f"Unsupported MRMP benchmark domain {domain!r}") from exc

    @classmethod
    def domains_for_arg(cls, domain: str) -> tuple[str, ...]:
        if domain == "all":
            return cls.DOMAINS
        return (cls.canonical_domain(domain),)

    @classmethod
    def manifest_path(cls, output_root: str | Path, domain: str) -> Path:
        if str(domain) == "all":
            return cls.combined_manifest_path(output_root)
        return Path(output_root).resolve() / cls.canonical_domain(domain) / "manifest.json"

    @classmethod
    def combined_manifest_path(cls, output_root: str | Path) -> Path:
        return Path(output_root).resolve() / "manifest.json"

    @classmethod
    def load_existing_records(
        cls,
        manifest_path: Path,
        builder: type[PBSExpansionMRMPManifestBuilder] = PBSExpansionMRMPManifestBuilder,
    ) -> list[MRMPBenchmarkRecord]:
        if not manifest_path.exists():
            return []
        return [
            record for record in load_mrmp_manifest(manifest_path)
            if cls.record_matches_current_benchmark(record, builder)
        ]

    @classmethod
    def record_matches_current_benchmark(
        cls,
        record: MRMPBenchmarkRecord,
        builder: type[PBSExpansionMRMPManifestBuilder] = PBSExpansionMRMPManifestBuilder,
    ) -> bool:
        return builder.record_matches_benchmark(record)

    @classmethod
    def save_progress(cls, manifest_path: Path, records: Sequence[MRMPBenchmarkRecord]) -> None:
        save_manifest(manifest_path, list(records))

    @classmethod
    def records_for_domain(
        cls,
        records: Sequence[MRMPBenchmarkRecord],
        domain: str,
    ) -> list[MRMPBenchmarkRecord]:
        canonical_domain = cls.canonical_domain(domain)
        return [record for record in records if record.domain_key == canonical_domain]

    @classmethod
    def grid_size(cls, record: BaseBenchmarkRecord) -> int:
        return int(record.env_params.get("size", record.env_params.get("N", 0)))

    @classmethod
    def base_record_is_eligible(cls, domain: str, record: BaseBenchmarkRecord) -> bool:
        if not record.supports_sampling:
            return False
        canonical_domain = cls.canonical_domain(domain)
        if record.domain_key != canonical_domain:
            return False
        if canonical_domain == "grid2d":
            return cls.grid_size(record) >= cls.DEFAULT_MIN_GRID_SIZE
        return True

    @classmethod
    def base_records_from_base_manifest(
        cls,
        domain: str,
        base_root: str | Path,
    ) -> list[BaseBenchmarkRecord]:
        canonical_domain = cls.canonical_domain(domain)
        base_manifest_path = BaseManifestStore.manifest_path(base_root, domain_key=canonical_domain)
        records = [
            record for record in load_base_manifest(base_manifest_path)
            if cls.base_record_is_eligible(canonical_domain, record)
        ]
        if len(records) < cls.DEFAULT_RECORDS_PER_DOMAIN:
            raise RuntimeError(
                f"Expected at least {cls.DEFAULT_RECORDS_PER_DOMAIN} eligible base records "
                f"for {canonical_domain}, found {len(records)} in {base_manifest_path.resolve()}."
            )
        return records

    @classmethod
    def filter_base_records_for_benchmark(
        cls,
        domain: str,
        benchmark: str | None,
        records: Sequence[BaseBenchmarkRecord],
    ) -> list[BaseBenchmarkRecord]:
        canonical_domain = cls.canonical_domain(domain)
        canonical_benchmark = cls.canonical_benchmark(benchmark)
        if canonical_benchmark == cls.WINDOWED_COORDINATION_BENCHMARK and canonical_domain == "grid2d":
            grid_size = cls.WINDOWED_COORDINATION_GRID2D_SIZE
            filtered = [record for record in records if cls.grid_size(record) == grid_size]
            if len(filtered) < cls.DEFAULT_RECORDS_PER_DOMAIN:
                raise RuntimeError(
                    f"Expected at least {cls.DEFAULT_RECORDS_PER_DOMAIN} eligible size-{grid_size} "
                    f"base records for {canonical_domain}, found {len(filtered)}."
                )
            return filtered
        if canonical_benchmark == cls.PERFORMANCE_COMPARISON_BENCHMARK and canonical_domain == "grid2d":
            grid_size = cls.PERFORMANCE_COMPARISON_GRID2D_SIZE
            filtered = [record for record in records if cls.grid_size(record) == grid_size]
            if len(filtered) < cls.DEFAULT_RECORDS_PER_DOMAIN:
                raise RuntimeError(
                    f"Expected at least {cls.DEFAULT_RECORDS_PER_DOMAIN} eligible size-{grid_size} "
                    f"base records for {canonical_domain}, found {len(filtered)}."
                )
            return filtered
        return list(records)

    @classmethod
    def target_count_for_benchmark(cls, args: argparse.Namespace, builder: type[PBSExpansionMRMPManifestBuilder]) -> int:
        if builder is MRMPPerformanceComparisonManifestBuilder:
            return builder.target_count(int(args.replicates_per_robot_count))
        return cls.DEFAULT_RECORDS_PER_DOMAIN

    @classmethod
    def base_records_for_benchmark(
        cls,
        domain: str,
        base_root: str | Path,
        benchmark: str | None,
    ) -> list[BaseBenchmarkRecord]:
        records = cls.base_records_from_base_manifest(
            domain=domain,
            base_root=base_root,
        )
        return cls.filter_base_records_for_benchmark(domain, benchmark, records)

    @classmethod
    def build_records(
        cls,
        domain: str,
        base_records: Sequence[BaseBenchmarkRecord],
        target_count: int = DEFAULT_RECORDS_PER_DOMAIN,
        existing_records: Sequence[MRMPBenchmarkRecord] | None = None,
        builder: type[PBSExpansionMRMPManifestBuilder] = PBSExpansionMRMPManifestBuilder,
        replicates_per_robot_count: int = 1,
        on_record=None,
        on_status=None,
    ) -> list[MRMPBenchmarkRecord]:
        if builder is MRMPPerformanceComparisonManifestBuilder:
            records = builder.build_manifest(
                base_records=base_records,
                replicates_per_robot_count=replicates_per_robot_count,
                existing_records=existing_records,
                on_record=on_record,
                on_status=on_status,
            )
            if len(records) < int(target_count):
                raise RuntimeError(
                    f"Unable to build {int(target_count)} {cls.canonical_domain(domain)} MRMP records "
                    f"from {len(base_records)} eligible base candidates; built {len(records)}."
                )
            return records

        canonical_domain = cls.canonical_domain(domain)
        selection_enabled = builder.conflict_density_selection_enabled(canonical_domain)
        pool_target_count = builder.conflict_density_selection_pool_count(
            canonical_domain,
            int(target_count),
            len(base_records),
        )
        records: list[MRMPBenchmarkRecord] = list(existing_records or [])
        used_base_ids = {record.base_instance_id for record in records}
        used_instance_ids = {record.instance_id for record in records}
        for base_record in base_records:
            if len(records) >= int(pool_target_count):
                break
            if base_record.instance_id in used_base_ids:
                continue
            try:
                new_records = builder.build_manifest(
                    base_records=[base_record],
                    on_status=on_status,
                )
            except RuntimeError as exc:
                if not str(exc).startswith("Failed to synthesize MRMP record"):
                    raise
                if on_status is not None:
                    on_status(
                        "candidate_failed",
                        {
                            "base_instance_id": base_record.instance_id,
                            "reason": str(exc),
                        },
                    )
                continue

            for record in new_records:
                if record.base_instance_id in used_base_ids or record.instance_id in used_instance_ids:
                    continue
                records.append(record)
                used_base_ids.add(record.base_instance_id)
                used_instance_ids.add(record.instance_id)
                if on_record is not None and not selection_enabled:
                    on_record(record, records)
                if len(records) >= int(pool_target_count):
                    break

        if selection_enabled:
            records = builder.select_records_for_domain(canonical_domain, records, int(target_count))
            if len(records) >= int(target_count) and on_record is not None:
                on_record(records[-1], records)

        if len(records) < int(target_count):
            raise RuntimeError(
                f"Unable to build {int(target_count)} {canonical_domain} MRMP records "
                f"from {len(base_records)} eligible base candidates; built {len(records)}."
            )
        return records

    @staticmethod
    def progress_disabled() -> bool:
        return os.environ.get("STGCS_DISABLE_PROGRESS", "") == "1"

    @staticmethod
    def _short_id(instance_id: str, max_len: int = 36) -> str:
        if len(instance_id) <= max_len:
            return instance_id
        return "..." + instance_id[-(max_len - 3):]

    @classmethod
    def _status_message(cls, event: str, payload: dict) -> str:
        if event == "reject":
            reason = str(payload["reason"])
            if reason == "conflict_count":
                return f"reject:conflicts={payload['num_conflicting_pairs']}/{payload['min_conflicting_pairs']}"
            if reason == "conflict_density":
                density = 100.0 * float(payload["conflict_density"])
                max_density = 100.0 * float(payload["max_conflict_density"])
                return f"reject:density={density:.1f}>{max_density:.1f}%"
            return f"reject:{reason}"
        if event == "candidate_failed":
            return "skip:failed"
        if event == "independent_plans":
            return "plan:single"
        if event == "conflict_screen":
            return "check:conflicts"
        return str(event)

    @classmethod
    def update_progress_status(cls, progress, state: dict, event: str, payload: dict) -> None:
        if event == "candidate":
            state["candidates"] += 1
            state["base"] = cls._short_id(str(payload["base_instance_id"]))
            state["attempt"] = "-"
            state["last"] = "candidate"
        elif event == "synthesize_attempt":
            state["attempt"] = f"{payload['attempt']}/{payload['max_attempts']}"
            state["last"] = "sampling"
        elif event in {"reject", "candidate_failed", "independent_plans", "conflict_screen"}:
            state["last"] = cls._status_message(event, payload)
        else:
            state["last"] = str(event)

        progress.set_postfix_str(
            f"candidates={state['candidates']} base={state['base']} "
            f"attempt={state['attempt']} last={state['last']}",
            refresh=True,
        )

    @classmethod
    def build_domain_manifest(cls, args: argparse.Namespace, domain: str) -> list[MRMPBenchmarkRecord]:
        benchmark = getattr(args, "benchmark", cls.DEFAULT_BENCHMARK)
        builder = cls.builder_for_benchmark(benchmark)
        base_records = cls.base_records_for_benchmark(
            domain=domain,
            base_root=args.base_root,
            benchmark=benchmark,
        )
        target_count = cls.target_count_for_benchmark(args, builder)
        manifest_path = cls.manifest_path(args.output_root, domain)
        lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")

        with ManifestLock(lock_path):
            existing_records = cls.load_existing_records(manifest_path, builder)
            records = cls.build_domain_records_with_progress(
                domain=domain,
                base_records=base_records,
                target_count=target_count,
                existing_records=existing_records,
                manifest_path=manifest_path,
                completed_records=(),
                builder=builder,
                replicates_per_robot_count=int(getattr(args, "replicates_per_robot_count", 1)),
            )
            save_manifest(manifest_path, records)
        return records

    @classmethod
    def build_domain_records_with_progress(
        cls,
        domain: str,
        base_records: Sequence[BaseBenchmarkRecord],
        target_count: int,
        existing_records: Sequence[MRMPBenchmarkRecord],
        manifest_path: Path,
        completed_records: Sequence[MRMPBenchmarkRecord],
        builder: type[PBSExpansionMRMPManifestBuilder] = PBSExpansionMRMPManifestBuilder,
        replicates_per_robot_count: int = 1,
    ) -> list[MRMPBenchmarkRecord]:
        progress_total = max(target_count, len(existing_records))

        with tqdm(
            total=progress_total,
            initial=len(existing_records),
            desc=f"{domain} MRMP",
            unit="record",
            dynamic_ncols=True,
            disable=cls.progress_disabled(),
        ) as progress:
            state = {
                "candidates": 0,
                "base": "-",
                "attempt": "-",
                "last": f"resume:{len(existing_records)}",
            }
            progress.set_postfix_str(
                f"candidates={state['candidates']} base={state['base']} "
                f"attempt={state['attempt']} last={state['last']}",
                refresh=True,
            )

            def on_record(record, records, *, target_path=manifest_path, prefix=tuple(completed_records)):
                cls.save_progress(target_path, [*prefix, *records])
                progress.update(max(0, len(records) - progress.n))
                state["base"] = cls._short_id(record.base_instance_id)
                state["attempt"] = "-"
                state["last"] = f"accepted:conflicts={record.num_conflicting_pairs}"
                progress.set_postfix_str(
                    f"candidates={state['candidates']} base={state['base']} "
                    f"attempt={state['attempt']} last={state['last']}",
                    refresh=True,
                )

            def on_status(event, payload):
                cls.update_progress_status(progress, state, event, payload)

            return cls.build_records(
                domain=domain,
                base_records=base_records,
                target_count=target_count,
                existing_records=existing_records,
                builder=builder,
                replicates_per_robot_count=replicates_per_robot_count,
                on_record=on_record,
                on_status=on_status,
            )

    @classmethod
    def build_combined_manifest(cls, args: argparse.Namespace) -> list[MRMPBenchmarkRecord]:
        benchmark = getattr(args, "benchmark", cls.DEFAULT_BENCHMARK)
        builder = cls.builder_for_benchmark(benchmark)
        manifest_path = cls.combined_manifest_path(args.output_root)
        lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
        target_count = cls.target_count_for_benchmark(args, builder)

        with ManifestLock(lock_path):
            existing_records = cls.load_existing_records(manifest_path, builder)
            records: list[MRMPBenchmarkRecord] = []
            for domain in cls.DOMAINS:
                base_records = cls.base_records_for_benchmark(
                    domain=domain,
                    base_root=args.base_root,
                    benchmark=benchmark,
                )
                domain_records = cls.build_domain_records_with_progress(
                    domain=domain,
                    base_records=base_records,
                    target_count=target_count,
                    existing_records=cls.records_for_domain(existing_records, domain),
                    manifest_path=manifest_path,
                    completed_records=records,
                    builder=builder,
                    replicates_per_robot_count=int(getattr(args, "replicates_per_robot_count", 1)),
                )
                records.extend(domain_records)
                save_manifest(manifest_path, records)
        return records

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        builder = cls.builder_for_benchmark(getattr(args, "benchmark", cls.DEFAULT_BENCHMARK))
        builder.NATIVE_BACKEND = STGCSPlannerClient(args.native_executable, args.native_work_dir)
        if args.domain == "all":
            records = cls.build_combined_manifest(args)
            manifest_path = cls.combined_manifest_path(args.output_root)
            print(
                f"Wrote {len(records)} MRMP records to {manifest_path}"
            )
            return

        for domain in cls.domains_for_arg(args.domain):
            records = cls.build_domain_manifest(args, domain)
            manifest_path = cls.manifest_path(args.output_root, domain)
            print(
                f"Wrote {len(records)} MRMP records to {manifest_path}"
            )


if __name__ == "__main__":
    MRMPManifestCLI.main()
