from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from experiments.base import BaseInstanceFactory, BaseManifestStore
from experiments.instance import Instance
from stgcs_native.types import MPQuery


@dataclass(frozen=True)
class STQuerySpec:
    start: List[float]
    goal: List[float]
    t_start: float
    is_stay: bool
    vlimit: float

    @staticmethod
    def from_query(query: MPQuery) -> "STQuerySpec":
        return STQuerySpec(
            start=query.start.tolist(),
            goal=query.goal.tolist(),
            t_start=float(query.t_start),
            is_stay=bool(query.is_stay),
            vlimit=float(query.vlimit),
        )

    def to_query(self) -> MPQuery:
        return MPQuery(
            start=np.asarray(self.start, dtype=float),
            goal=np.asarray(self.goal, dtype=float),
            t_start=float(self.t_start),
            is_stay=bool(self.is_stay),
            vlimit=float(self.vlimit),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "goal": self.goal,
            "t_start": self.t_start,
            "is_stay": self.is_stay,
            "vlimit": self.vlimit,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "STQuerySpec":
        return STQuerySpec(
            start=[float(value) for value in data["start"]],
            goal=[float(value) for value in data["goal"]],
            t_start=float(data["t_start"]),
            is_stay=bool(data["is_stay"]),
            vlimit=float(data["vlimit"]),
        )


@dataclass(frozen=True)
class STHeuristicAblationRecord:
    instance_id: str
    group: str
    source_domain: str
    base_instance_id: str
    query: STQuerySpec
    dynamic_obstacles: List[Dict[str, Any]]
    stgcs_num_vertices: int
    stgcs_num_edges: int
    tmax: float = 1000.0
    generation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "group": self.group,
            "source_domain": self.source_domain,
            "base_instance_id": self.base_instance_id,
            "query": self.query.to_dict(),
            "dynamic_obstacles": self.dynamic_obstacles,
            "stgcs_num_vertices": self.stgcs_num_vertices,
            "stgcs_num_edges": self.stgcs_num_edges,
            "tmax": self.tmax,
            "generation": self.generation,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "STHeuristicAblationRecord":
        return STHeuristicAblationRecord(
            instance_id=str(data["instance_id"]),
            group=str(data["group"]),
            source_domain=str(data["source_domain"]),
            base_instance_id=str(data["base_instance_id"]),
            query=STQuerySpec.from_dict(data["query"]),
            dynamic_obstacles=[dict(item) for item in data["dynamic_obstacles"]],
            stgcs_num_vertices=int(data["stgcs_num_vertices"]),
            stgcs_num_edges=int(data["stgcs_num_edges"]),
            tmax=float(data.get("tmax", 1000.0)),
            generation=dict(data.get("generation", {})),
        )


@dataclass(frozen=True)
class STHeuristicAblationResultEntry:
    is_success: bool
    runtime: float
    cost: float
    num_expanded_nodes: int
    num_generated_nodes: int


class STPlanningManifestStore:
    @staticmethod
    def save_manifest(path: str | Path, records: Sequence[STHeuristicAblationRecord]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps([record.to_dict() for record in records], indent=2)
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        tmp_path.write_text(payload)
        tmp_path.replace(target)

    @staticmethod
    def load_manifest(path: str | Path) -> list[STHeuristicAblationRecord]:
        return [
            STHeuristicAblationRecord.from_dict(item)
            for item in json.loads(Path(path).read_text())
        ]

    @staticmethod
    def merge_records_by_instance_id(
        existing: Sequence[STHeuristicAblationRecord],
        incoming: Sequence[STHeuristicAblationRecord],
    ) -> list[STHeuristicAblationRecord]:
        merged_by_id = {record.instance_id: record for record in existing}
        for record in incoming:
            merged_by_id[record.instance_id] = record
        return list(merged_by_id.values())

    @classmethod
    def reconstruct_instance(
        cls,
        record: STHeuristicAblationRecord,
        base_root: str | Path,
    ):
        base_manifest_path = BaseManifestStore.manifest_path(base_root, domain_key=record.source_domain)
        base_record = BaseManifestStore.record_by_id(base_manifest_path, record.base_instance_id)
        instance = BaseInstanceFactory.from_record(base_record, compute_heuristics=False)
        env = instance.env.copy()
        env.O_Dynamic = Instance.dynamic_obstacles_from_specs(record.dynamic_obstacles)
        instance.env = env
        instance.stgcs = Instance.build_stgcs_from_env(
            env,
            tmax=record.tmax,
            vlimit=record.query.vlimit,
        )
        return instance, record.query.to_query(), base_manifest_path, base_record
