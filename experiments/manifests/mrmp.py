from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List

from stgcs_native.types import MPQuery


@dataclass(frozen=True)
class QuerySpec:
    start: List[float]
    goal: List[float]
    t_start: float
    is_stay: bool
    vlimit: float

    @staticmethod
    def from_query(query: MPQuery) -> "QuerySpec":
        return QuerySpec(
            start=query.start.tolist(),
            goal=query.goal.tolist(),
            t_start=float(query.t_start),
            is_stay=bool(query.is_stay),
            vlimit=float(query.vlimit),
        )

    def to_query(self) -> MPQuery:
        import numpy as np

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
    def from_dict(data: Dict[str, Any]) -> "QuerySpec":
        return QuerySpec(
            start=[float(v) for v in data["start"]],
            goal=[float(v) for v in data["goal"]],
            t_start=float(data["t_start"]),
            is_stay=bool(data["is_stay"]),
            vlimit=float(data["vlimit"]),
        )


@dataclass(frozen=True)
class MRMPBenchmarkRecord:
    instance_id: str
    base_instance_id: str
    queries: List[QuerySpec]
    num_agents: int
    independent_cost_sum: float
    independent_makespan: float
    independent_runtime: float
    num_conflicting_pairs: int
    num_conflicting_agents: int
    conflict_largest_component: int
    conflict_density: float
    stgcs_num_vertices: int = 0
    stgcs_num_edges: int = 0
    tmax: float = 100.0

    @staticmethod
    def domain_key_from_base_instance_id(base_instance_id: str) -> str:
        prefixes = (
            ("base-fastpathplanning-village3d-", "fastpathplanning-village3d"),
            ("base-iris-2d-", "iris-2d"),
            ("base-empty-square2d-", "empty-square2d"),
            ("base-sphere-ball-", "sphere-ball"),
            ("base-simple2d-", "simple2d"),
            ("base-grid2d-", "grid2d"),
            ("base-grid3d-", "grid3d"),
            ("base-maze-", "maze"),
        )
        for prefix, domain_key in prefixes:
            if str(base_instance_id).startswith(prefix):
                return domain_key
        raise ValueError(f"Unable to infer MRMP domain from base_instance_id {base_instance_id!r}.")

    @property
    def domain_key(self) -> str:
        return self.domain_key_from_base_instance_id(self.base_instance_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "base_instance_id": self.base_instance_id,
            "queries": [query.to_dict() for query in self.queries],
            "num_agents": self.num_agents,
            "independent_cost_sum": self.independent_cost_sum,
            "independent_makespan": self.independent_makespan,
            "independent_runtime": self.independent_runtime,
            "num_conflicting_pairs": self.num_conflicting_pairs,
            "num_conflicting_agents": self.num_conflicting_agents,
            "conflict_largest_component": self.conflict_largest_component,
            "conflict_density": self.conflict_density,
            "stgcs_num_vertices": self.stgcs_num_vertices,
            "stgcs_num_edges": self.stgcs_num_edges,
            "tmax": self.tmax,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MRMPBenchmarkRecord":
        return MRMPBenchmarkRecord(
            instance_id=str(data["instance_id"]),
            base_instance_id=str(data["base_instance_id"]),
            queries=[QuerySpec.from_dict(item) for item in data["queries"]],
            num_agents=int(data["num_agents"]),
            independent_cost_sum=float(data["independent_cost_sum"]),
            independent_makespan=float(data["independent_makespan"]),
            independent_runtime=float(data["independent_runtime"]),
            num_conflicting_pairs=int(data["num_conflicting_pairs"]),
            num_conflicting_agents=int(data["num_conflicting_agents"]),
            conflict_largest_component=int(data["conflict_largest_component"]),
            conflict_density=float(data["conflict_density"]),
            stgcs_num_vertices=int(data.get("stgcs_num_vertices", 0)),
            stgcs_num_edges=int(data.get("stgcs_num_edges", 0)),
            tmax=float(data.get("tmax", 100.0)),
        )


def save_manifest(path: str | Path, records: List[MRMPBenchmarkRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps([record.to_dict() for record in records], indent=2)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    tmp_path.write_text(payload)
    tmp_path.replace(target)


def load_manifest(path: str | Path) -> List[MRMPBenchmarkRecord]:
    payload = json.loads(Path(path).read_text())
    return [MRMPBenchmarkRecord.from_dict(item) for item in payload]
