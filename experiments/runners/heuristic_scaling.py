from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import shutil
import json
import signal
from types import SimpleNamespace
from typing import Sequence

from experiments.datasets.base import BaseManifestBuilder, ManifestLock
from experiments.base import BaseInstanceFactory
from experiments.manifests.base import BaseBenchmarkRecord, load_manifest, save_manifest
from experiments.native import STGCSPlannerClient


class HeuristicComputationTimeScalingBuilder(BaseManifestBuilder):
    backend: STGCSPlannerClient | None = None
    DEFAULT_OUTPUT_ROOT = Path("data/stgcs_base/heur_computation_time_scaling")
    DEFAULT_N_VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    DEFAULT_M_VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    DEFAULT_SAMPLES_PER_SHAPE = 1
    DEFAULT_CANDIDATE_MULTIPLIER = 3
    DEFAULT_STOP_AFTER_FAILED_SHAPES = 3
    DEFAULT_H_TAB_TIMEOUT_SECS = 10_000.0

    @staticmethod
    def instance_id(n: int, m: int, seed: int) -> str:
        return f"base-heur-scaling-grid2d-{n}x{m}-seed{seed:05d}"

    @staticmethod
    def grid_env_params(n: int, m: int) -> dict[str, int]:
        return {"N": int(n), "M": int(m)}

    @staticmethod
    def grid_shape(record: BaseBenchmarkRecord) -> tuple[int, int]:
        return int(record.env_params["N"]), int(record.env_params["M"])

    @staticmethod
    def grid_shapes(n_values: Sequence[int], m_values: Sequence[int]) -> list[tuple[int, int]]:
        shapes = {(int(n), int(m)) for n in n_values for m in m_values}
        return sorted(
            shapes,
            key=lambda item: (item[0] * item[1], item[0] + item[1], item[0], item[1]),
        )

    @classmethod
    def build_grid2d_record(cls, n: int, m: int, seed: int) -> BaseBenchmarkRecord:
        env_params = cls.grid_env_params(n, m)
        instance = BaseInstanceFactory.build(
            domain="grid",
            env_params=env_params,
            spatial_seed=int(seed),
            space_dim=2,
            compute_heuristics=False,
        )
        return cls._record(
            instance_id=cls.instance_id(n, m, seed),
            domain="grid",
            space_dim=2,
            spatial_seed=int(seed),
            env_params=env_params,
            instance=instance,
        )

    @staticmethod
    def records_by_shape(records: Sequence[BaseBenchmarkRecord]) -> dict[tuple[int, int], list[BaseBenchmarkRecord]]:
        grouped: dict[tuple[int, int], list[BaseBenchmarkRecord]] = {}
        for record in records:
            shape = HeuristicComputationTimeScalingBuilder.grid_shape(record)
            grouped.setdefault(shape, []).append(record)
        return grouped

    @classmethod
    def cleanup_failed_candidate(cls, record: BaseBenchmarkRecord) -> None:
        cache_dir = cls.backend.cache_root / record.domain_key / "heuristics" / record.instance_id
        shutil.rmtree(cache_dir, ignore_errors=True)

    @classmethod
    def compute_native_cache(
        cls,
        manifest_path: Path,
        record: BaseBenchmarkRecord,
        h_tab_timeout_secs: float,
    ) -> dict[str, float]:
        if cls.backend is None:
            raise RuntimeError("A Release native executable is required for the scaling study")
        instance = BaseInstanceFactory.from_record(record, compute_heuristics=False)
        query = SimpleNamespace(start=[0.] * instance.env.dim, goal=[0.] * instance.env.dim,
                                t_start=0., is_stay=True, vlimit=instance.stgcs.vlimit)
        path = cls.backend.work_dir / "inputs" / f"{record.instance_id}.txt"
        cls.backend.write_input(path, instance.env, query, instance.stgcs.tmax, [])
        cls.backend.bind_cache(path, record.instance_id, record.domain_key)
        return cls.backend.cache(path, h_tab_timeout_secs)

    @classmethod
    def build_or_extend(
        cls,
        output_root: Path,
        n_values: Sequence[int],
        m_values: Sequence[int],
        samples_per_shape: int,
        seed_start: int,
        candidate_multiplier: int,
        stop_after_failed_shapes: int,
        h_tab_timeout_secs: float,
    ) -> list[BaseBenchmarkRecord]:
        output_dir = Path(output_root)
        manifest_path = output_dir / "manifest.json"
        lock_path = output_dir / ".heur_scaling_manifest.lock"
        output_dir.mkdir(parents=True, exist_ok=True)

        if samples_per_shape <= 0:
            raise ValueError(f"samples_per_shape must be positive, got {samples_per_shape}.")
        if candidate_multiplier <= 0:
            raise ValueError(f"candidate_multiplier must be positive, got {candidate_multiplier}.")
        if stop_after_failed_shapes < 0:
            raise ValueError(f"stop_after_failed_shapes must be non-negative, got {stop_after_failed_shapes}.")
        if h_tab_timeout_secs <= 0.0:
            raise ValueError(f"h_tab_timeout_secs must be positive, got {h_tab_timeout_secs}.")

        with ManifestLock(lock_path):
            records = load_manifest(manifest_path) if manifest_path.exists() else []
            seen_ids = {record.instance_id for record in records}
            grouped = cls.records_by_shape(records)
            max_attempts = int(samples_per_shape) * int(candidate_multiplier)
            consecutive_failed_shapes = 0

            for n, m in cls.grid_shapes(n_values, m_values):
                kept = list(grouped.get((n, m), []))
                shape_label = f"{n}x{m}"
                if len(kept) >= int(samples_per_shape):
                    print(f"shape={shape_label}: already has {len(kept)}/{samples_per_shape} records")
                    consecutive_failed_shapes = 0
                    continue

                for attempt in range(max_attempts):
                    if len(kept) >= int(samples_per_shape):
                        break
                    seed = int(seed_start) + attempt
                    instance_id = cls.instance_id(n, m, seed)
                    if instance_id in seen_ids:
                        continue
                    record: BaseBenchmarkRecord | None = None
                    try:
                        record = cls.build_grid2d_record(n=n, m=m, seed=seed)
                        print(
                            f"{record.instance_id}: graph=({record.stgcs_num_vertices}V,"
                            f" {record.stgcs_num_edges}E), running h_tab"
                        )
                        info = cls.compute_native_cache(
                            manifest_path=manifest_path,
                            record=record,
                            h_tab_timeout_secs=float(h_tab_timeout_secs),
                        )
                        times = {name: info[name] for name in ("h_tri", "h_tab")}
                        record = replace(record, stgcs_num_vertices=info["vertices"],
                                         stgcs_num_edges=info["edges"]).with_heuristic_computation_times(times)
                    except Exception as exc:
                        if record is None:
                            print(f"{instance_id}: skip, failed to build base ({exc})")
                        else:
                            cls.cleanup_failed_candidate(record)
                            print(f"{record.instance_id}: skip, h_tab failed ({exc})")
                        continue

                    records.append(record)
                    kept.append(record)
                    grouped.setdefault((n, m), []).append(record)
                    seen_ids.add(record.instance_id)
                    save_manifest(manifest_path, records)
                    print(
                        f"{record.instance_id}: kept {len(kept)}/{samples_per_shape} "
                        f"for shape={shape_label}, "
                        f"h_tab={times['h_tab']:.3f}s"
                    )

                if len(kept) < int(samples_per_shape):
                    print(
                        f"shape={shape_label}: kept {len(kept)}/{samples_per_shape} "
                        f"records after {max_attempts} attempts"
                    )
                    consecutive_failed_shapes += 1
                    if (
                        int(stop_after_failed_shapes) > 0
                        and consecutive_failed_shapes >= int(stop_after_failed_shapes)
                    ):
                        print(
                            f"Stopping after {consecutive_failed_shapes} consecutive "
                            f"h_tab-infeasible shapes."
                        )
                        break
                else:
                    consecutive_failed_shapes = 0

            save_manifest(manifest_path, records)

        return records


class HeuristicComputationTimeScalingCLI:
    @staticmethod
    def stop(*_):
        STGCSPlannerClient.stop()
        raise KeyboardInterrupt

    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=(
                "Incrementally build a rectangular grid2d base manifest for offline heuristic scaling. "
                "A candidate is kept only if h_tab finishes within the requested timeout."
            )
        )
        parser.add_argument("--native-executable", type=Path, required=True)
        parser.add_argument("--native-work-dir", type=Path, required=True)
        parser.add_argument(
            "--output-root",
            type=Path,
            default=HeuristicComputationTimeScalingBuilder.DEFAULT_OUTPUT_ROOT,
        )
        parser.add_argument(
            "--n-values",
            type=int,
            nargs="+",
            default=HeuristicComputationTimeScalingBuilder.DEFAULT_N_VALUES,
        )
        parser.add_argument(
            "--m-values",
            type=int,
            nargs="+",
            default=HeuristicComputationTimeScalingBuilder.DEFAULT_M_VALUES,
        )
        parser.add_argument(
            "--samples-per-shape",
            type=int,
            default=HeuristicComputationTimeScalingBuilder.DEFAULT_SAMPLES_PER_SHAPE,
        )
        parser.add_argument("--seed-start", type=int, default=0)
        parser.add_argument(
            "--candidate-multiplier",
            type=int,
            default=HeuristicComputationTimeScalingBuilder.DEFAULT_CANDIDATE_MULTIPLIER,
        )
        parser.add_argument(
            "--stop-after-failed-shapes",
            type=int,
            default=HeuristicComputationTimeScalingBuilder.DEFAULT_STOP_AFTER_FAILED_SHAPES,
            help="Stop after this many consecutive shapes fail to produce an h_tab-feasible sample; use 0 to disable.",
        )
        parser.add_argument(
            "--h-tab-timeout-secs",
            type=float,
            default=HeuristicComputationTimeScalingBuilder.DEFAULT_H_TAB_TIMEOUT_SECS,
        )
        return parser.parse_args()

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        signal.signal(signal.SIGTERM, cls.stop)
        signal.signal(signal.SIGINT, cls.stop)
        HeuristicComputationTimeScalingBuilder.backend = STGCSPlannerClient(args.native_executable, args.native_work_dir)
        records = HeuristicComputationTimeScalingBuilder.build_or_extend(
            output_root=args.output_root,
            n_values=tuple(int(value) for value in args.n_values),
            m_values=tuple(int(value) for value in args.m_values),
            samples_per_shape=int(args.samples_per_shape),
            seed_start=int(args.seed_start),
            candidate_multiplier=int(args.candidate_multiplier),
            stop_after_failed_shapes=int(args.stop_after_failed_shapes),
            h_tab_timeout_secs=float(args.h_tab_timeout_secs),
        )
        print(f"Wrote {len(records)} records to {Path(args.output_root) / 'manifest.json'}")
        print(json.dumps({"complete": True, "records": len(records), "backend": "C++ Release"}), flush=True)


if __name__ == "__main__":
    HeuristicComputationTimeScalingCLI.main()
