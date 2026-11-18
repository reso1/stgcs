"""Shared C++ heuristic-cache preparation for benchmark base instances."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory


class BaseOfflineHeuristicStore:
    ZERO_NAME = "h_zero"
    MOTION_ONLY_NAME = "h_mot"
    TRIPLET_RELAXATION_NAME = "h_tri"
    INTERFACE_TO_SET_COST_TABLE_NAME = "h_tab"
    MAX_NAME = "h_max"

    @classmethod
    def has_complete_caches(cls, manifest_path, record, required_heuristics):
        cost_table = bool({cls.INTERFACE_TO_SET_COST_TABLE_NAME, cls.MAX_NAME} & required_heuristics)
        filename = "heuristics.bin" if cost_table else "triplets.bin"
        path = Path(manifest_path).resolve().parent / "heuristics" / record.instance_id / filename
        if not path.is_file():
            return False
        from stgcs_native.planning import NativeBackend
        with path.open("rb") as stream:
            if stream.read(len(NativeBackend.CACHE_MAGIC)) != NativeBackend.CACHE_MAGIC:
                return False
        NativeBackend.read_cache_tables(path)
        return True

    @staticmethod
    def compute(instance, *, timeout=10000, cost_table=True, record=None, force=False):
        from experiments.native import STGCSPlannerClient
        from stgcs_native.client import NativeClient
        from stgcs_native.types import MPQuery
        import numpy as np

        record = record or getattr(instance, "base_record", None)
        if record is not None:
            instance.base_record = record
        root = Path(record.manifest_path).resolve().parent.parent if record and record.manifest_path else None
        query = MPQuery(np.zeros(instance.env.dim), np.zeros(instance.env.dim), 0, True, instance.stgcs.vlimit)
        with TemporaryDirectory(prefix="stgcs-heuristics-") as work:
            client = STGCSPlannerClient(NativeClient.executable("stgcs_planner"), Path(work), root)
            path, _ = client.prepare_instance(instance, [query], compute_cache=False)
            if force:
                cache = client.cache_path(path, cost_table=cost_table)
                cache.unlink(missing_ok=True)
                cache.with_suffix(".json").unlink(missing_ok=True)
            return client.cache(path, timeout=timeout, cost_table=cost_table)
