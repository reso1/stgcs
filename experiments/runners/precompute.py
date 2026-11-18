from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Dict, Sequence

from experiments.base import BaseInstanceFactory, BaseManifestStore
from experiments.offline_heuristics import BaseOfflineHeuristicStore
from experiments.manifests.base import BaseBenchmarkRecord, load_manifest


class BaseOfflineHeuristicCLI:
    _BASE_RECORD_KEYS = frozenset(
        {
            "instance_id",
            "domain",
            "space_dim",
            "spatial_seed",
            "env_params",
            "supports_sampling",
        }
    )
    _ST_HEURISTIC_ABLATION_KEYS = frozenset({"source_domain", "base_instance_id"})

    @staticmethod
    def _requested_heuristics(h_tab_timeout_secs: float | None) -> set[str]:
        requested = {
            BaseOfflineHeuristicStore.MOTION_ONLY_NAME,
            BaseOfflineHeuristicStore.TRIPLET_RELAXATION_NAME,
        }
        if h_tab_timeout_secs is not None:
            requested.add(BaseOfflineHeuristicStore.INTERFACE_TO_SET_COST_TABLE_NAME)
        return requested

    @classmethod
    def _is_base_record_payload(cls, payload: object) -> bool:
        return isinstance(payload, dict) and cls._BASE_RECORD_KEYS.issubset(payload)

    @classmethod
    def _is_st_heuristic_ablation_payload(cls, payload: object) -> bool:
        return isinstance(payload, dict) and cls._ST_HEURISTIC_ABLATION_KEYS.issubset(payload)

    @classmethod
    def _load_work_items(
        cls,
        manifest_path: Path,
        base_root: Path | None,
    ) -> list[tuple[Path, BaseBenchmarkRecord]]:
        manifest_path = Path(manifest_path).resolve()
        payload = json.loads(manifest_path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"Expected {manifest_path} to contain a list of records.")
        if not payload:
            return []

        if all(cls._is_base_record_payload(item) for item in payload):
            return [(manifest_path, record) for record in load_manifest(manifest_path)]

        if all(cls._is_st_heuristic_ablation_payload(item) for item in payload):
            resolved_base_root = (BaseManifestStore.default_base_root(manifest_path)
                                  if base_root is None else Path(base_root).resolve())
            seen: set[tuple[Path, str]] = set()
            work_items: list[tuple[Path, BaseBenchmarkRecord]] = []
            for item in payload:
                domain_key = str(item["source_domain"])
                base_instance_id = str(item["base_instance_id"])
                base_manifest_path = BaseManifestStore.manifest_path(resolved_base_root, domain_key=domain_key)
                key = (base_manifest_path.resolve(), base_instance_id)
                if key in seen:
                    continue
                seen.add(key)
                work_items.append(
                    (
                        base_manifest_path,
                        BaseManifestStore.record_by_id(base_manifest_path, base_instance_id),
                    )
                )
            return work_items

        raise ValueError(
            f"Unsupported manifest schema in {manifest_path}. Expected a base manifest or an ST "
            "heuristic-ablation manifest with source_domain/base_instance_id fields."
        )

    @classmethod
    def _work_items_to_update(
        cls,
        work_items: Sequence[tuple[Path, BaseBenchmarkRecord]],
        h_tab_timeout_secs: float | None,
        force: bool,
    ) -> list[tuple[Path, BaseBenchmarkRecord]]:
        if force:
            return list(work_items)

        required_heuristics = cls._requested_heuristics(h_tab_timeout_secs)
        pending: list[tuple[Path, BaseBenchmarkRecord]] = []
        for manifest_path, record in work_items:
            manifest_path = Path(manifest_path).resolve()
            if not BaseOfflineHeuristicStore.has_complete_caches(
                manifest_path,
                record,
                required_heuristics,
            ):
                pending.append((manifest_path, record))
        return pending


    @staticmethod
    def _build_record_caches(manifest_path, record_payload, h_tab_timeout_secs, force):
        manifest_file = Path(manifest_path).resolve()
        record = BaseBenchmarkRecord.from_dict(record_payload).with_manifest_path(manifest_file)
        instance = BaseInstanceFactory.from_record(record, compute_heuristics=False)
        result = BaseOfflineHeuristicStore.compute(
            instance, record=record, timeout=h_tab_timeout_secs or 10000,
            cost_table=h_tab_timeout_secs is not None, force=force)
        times = {"h_mot": 0.0, "h_tri": result["h_tri"]}
        if h_tab_timeout_secs is not None:
            times["h_tab"] = result["h_tab"]
        return record.instance_id, times


    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Build shared base h_tri/h_tab heuristic caches."
        )
        parser.add_argument("manifest", type=Path)
        parser.add_argument(
            "--base-root",
            type=Path,
            default=None,
            help="Base manifest root used when the input is an ST heuristic-ablation manifest.",
        )
        parser.add_argument("--workers", type=int, default=1)
        parser.add_argument("--h-tab-timeout-secs", type=float, default=None)
        parser.add_argument("--force", action="store_true")
        return parser.parse_args()

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        manifest_path = Path(args.manifest).resolve()
        if args.h_tab_timeout_secs is not None and float(args.h_tab_timeout_secs) <= 0.0:
            raise ValueError(f"--h-tab-timeout-secs must be > 0, got {args.h_tab_timeout_secs}")
        work_items = cls._load_work_items(manifest_path, args.base_root)
        if not work_items:
            print(f"No base records found in {manifest_path}")
            return

        h_tab_timeout_secs = None if args.h_tab_timeout_secs is None else float(args.h_tab_timeout_secs)
        work_items_to_update = cls._work_items_to_update(
            work_items,
            h_tab_timeout_secs,
            bool(args.force),
        )
        skipped_count = len(work_items) - len(work_items_to_update)
        if skipped_count:
            print(f"skipped {skipped_count} already complete base instances")
        if not work_items_to_update:
            requested = ", ".join(sorted(cls._requested_heuristics(h_tab_timeout_secs)))
            print(f"{manifest_path}: all {len(work_items)} base instances already have {requested} caches")
            return

        completed: Dict[str, Dict[str, float]] = {}
        failures: Dict[str, str] = {}
        worker_count = max(1, int(args.workers))

        if worker_count == 1:
            for item_manifest_path, record in work_items_to_update:
                try:
                    instance_id, times = cls._build_record_caches(
                        str(item_manifest_path),
                        record.to_dict(),
                        h_tab_timeout_secs,
                        bool(args.force),
                    )
                    completed[instance_id] = times
                    print(f"updated {instance_id}")
                except Exception as exc:
                    failures[record.instance_id] = str(exc)
        else:
            try:
                executor_context = ProcessPoolExecutor(max_workers=worker_count)
            except PermissionError:
                executor_context = ThreadPoolExecutor(max_workers=worker_count)
            with executor_context as executor:
                futures = {
                    executor.submit(
                        cls._build_record_caches,
                        str(item_manifest_path),
                        record.to_dict(),
                        h_tab_timeout_secs,
                        bool(args.force),
                    ): record.instance_id
                    for item_manifest_path, record in work_items_to_update
                }
                for future in as_completed(futures):
                    instance_id = futures[future]
                    try:
                        built_id, times = future.result()
                        completed[built_id] = times
                        print(f"updated {built_id}")
                    except Exception as exc:
                        failures[instance_id] = str(exc)

        print(
            f"{manifest_path}: updated caches for {len(completed)} of {len(work_items_to_update)} pending base "
            f"instances ({skipped_count} already complete, {len(work_items)} total)"
        )
        if failures:
            for instance_id in sorted(failures):
                print(f"{instance_id}: {failures[instance_id]}")
            raise RuntimeError(f"Failed to build caches for {len(failures)} base instances.")


if __name__ == "__main__":
    BaseOfflineHeuristicCLI.main()
