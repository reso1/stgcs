from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class BaseBenchmarkRecord:
    instance_id: str
    domain: str
    space_dim: int
    spatial_seed: int
    env_params: Dict[str, Any]
    supports_sampling: bool
    stgcs_num_vertices: int = 0
    stgcs_num_edges: int = 0
    heuristic_computation_times: Dict[str, float] = field(default_factory=dict)
    manifest_path: Path | None = field(default=None, compare=False, repr=False)

    @property
    def domain_key(self) -> str:
        if self.domain == "maze":
            return "maze"
        if self.domain == "simple2d":
            return "simple2d"
        if self.domain == "empty-square2d":
            return "empty-square2d"
        if self.domain == "sphere-ball":
            return "sphere-ball"
        if self.domain == "fastpathplanning-village3d":
            return "fastpathplanning-village3d"
        if self.domain == "iris-2d":
            return "iris-2d"
        return f"grid{self.space_dim}d"

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "instance_id": self.instance_id,
            "domain": self.domain,
            "space_dim": self.space_dim,
            "spatial_seed": self.spatial_seed,
            "env_params": self.env_params,
            "supports_sampling": self.supports_sampling,
            "stgcs_num_vertices": self.stgcs_num_vertices,
            "stgcs_num_edges": self.stgcs_num_edges,
        }
        if self.heuristic_computation_times:
            payload["heuristic_computation_times"] = {
                str(key): float(value)
                for key, value in self.heuristic_computation_times.items()
            }
        return payload

    def with_manifest_path(self, path: str | Path) -> "BaseBenchmarkRecord":
        return replace(self, manifest_path=Path(path).resolve())

    def with_heuristic_computation_times(self, times: Dict[str, float]) -> "BaseBenchmarkRecord":
        return replace(
            self,
            heuristic_computation_times={str(key): float(value) for key, value in times.items()},
        )

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "BaseBenchmarkRecord":
        return BaseBenchmarkRecord(
            instance_id=str(data["instance_id"]),
            domain=str(data["domain"]),
            space_dim=int(data["space_dim"]),
            spatial_seed=int(data["spatial_seed"]),
            env_params={str(key): value for key, value in data["env_params"].items()},
            supports_sampling=bool(data["supports_sampling"]),
            stgcs_num_vertices=int(data.get("stgcs_num_vertices", 0)),
            stgcs_num_edges=int(data.get("stgcs_num_edges", 0)),
            heuristic_computation_times={
                str(key): float(value)
                for key, value in dict(data.get("heuristic_computation_times", {})).items()
            },
        )


def save_manifest(path: str | Path, records: List[BaseBenchmarkRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps([record.to_dict() for record in records], indent=2)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    tmp_path.write_text(payload)
    tmp_path.replace(target)


def load_manifest(path: str | Path) -> List[BaseBenchmarkRecord]:
    target = Path(path).resolve()
    payload = json.loads(target.read_text())
    return [BaseBenchmarkRecord.from_dict(item).with_manifest_path(target) for item in payload]
