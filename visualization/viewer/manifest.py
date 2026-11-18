from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, TypeAlias

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from experiments.base import BaseInstanceFactory, BaseManifestStore
from experiments.manifests.base import BaseBenchmarkRecord, load_manifest as load_base_manifest
from experiments.manifests.mrmp import MRMPBenchmarkRecord, load_manifest as load_mrmp_manifest
from experiments.environment.obstacle import StaticPolygon, StaticSphere
from experiments.manifests.st_planning import STHeuristicAblationRecord, STPlanningManifestStore


ViewerRecord: TypeAlias = MRMPBenchmarkRecord | BaseBenchmarkRecord | STHeuristicAblationRecord


class ViewerManifestBuilder:
    @staticmethod
    def _detect_manifest_kind(sample: Dict[str, Any]) -> str:
        if "queries" in sample and "num_agents" in sample:
            return "mrmp"
        if "query" in sample and "dynamic_obstacles" in sample and "source_domain" in sample:
            return "st_heuristic_ablation"
        if "domain" in sample and "space_dim" in sample and "env_params" in sample:
            return "base"
        raise ValueError(
            "Unsupported manifest schema. Expected an mrmp manifest with 'queries'/'num_agents', "
            "an ST planning manifest with 'query'/'dynamic_obstacles'/'source_domain', "
            "or a base manifest with 'domain'/'space_dim'/'env_params'."
        )

    @classmethod
    def load_raw_manifest(cls, path: str | Path) -> List[ViewerRecord]:
        manifest_path = Path(path)
        payload = json.loads(manifest_path.read_text())
        if not isinstance(payload, list):
            raise ValueError("Instance manifest must be a JSON array of records.")
        if not payload:
            return []
        manifest_kind = cls._detect_manifest_kind(payload[0])
        if manifest_kind == "base":
            return load_base_manifest(manifest_path)
        if manifest_kind == "st_heuristic_ablation":
            return STPlanningManifestStore.load_manifest(manifest_path)
        return load_mrmp_manifest(manifest_path)

    @staticmethod
    def _base_geometry_cache_key(record: BaseBenchmarkRecord) -> Tuple[Any, ...]:
        return (
            record.domain,
            record.space_dim,
            record.spatial_seed,
            json.dumps(record.env_params, sort_keys=True),
        )

    @staticmethod
    def _build_base_geometry(record: BaseBenchmarkRecord) -> Dict[str, Any]:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            instance = BaseInstanceFactory.from_record(record, compute_heuristics=False)
        static_obstacles = getattr(instance.env, "O_Static", []) or []

        return {
            "dimension": int(instance.env.dim),
            "robot_radius": float(instance.env.robot_radius),
            "bounds": {
                "min": np.asarray(instance.env.lb, dtype=float).tolist(),
                "max": np.asarray(instance.env.ub, dtype=float).tolist(),
            },
            "spatial_sets": [
                ViewerManifestBuilder._spatial_set_to_viewer_json(
                    np.asarray(vertices, dtype=float), instance.env.dim, idx
                )
                for idx, vertices in enumerate(instance.env.C_Space)
            ],
            "static_obstacles": [
                ViewerManifestBuilder._static_obstacle_to_viewer_json(
                    obstacle, instance.env.dim, idx
                )
                for idx, obstacle in enumerate(static_obstacles)
            ],
        }

    @staticmethod
    def _spatial_set_to_viewer_json(vertices: np.ndarray, dim: int, idx: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": f"v{idx}",
            "vertices": vertices.tolist(),
        }
        if dim != 3 or len(vertices) < 4:
            return payload

        try:
            hull = ConvexHull(vertices)
        except QhullError:
            return payload

        faces = [[int(v) for v in simplex] for simplex in hull.simplices]
        edges = sorted(
            {
                tuple(sorted((int(simplex[a]), int(simplex[b]))))
                for simplex in hull.simplices
                for a, b in ((0, 1), (1, 2), (2, 0))
            }
        )
        payload["faces"] = faces
        payload["edges"] = [[u, v] for u, v in edges]
        return payload

    @classmethod
    def _static_obstacle_to_viewer_json(
        cls,
        obstacle: Any,
        dim: int,
        idx: int,
    ) -> Dict[str, Any]:
        if isinstance(obstacle, StaticPolygon):
            vertices = np.asarray(obstacle.vertices, dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] != dim:
                raise ValueError(
                    f"Static polygon obstacle {idx} has shape {vertices.shape}, expected (*, {dim})."
                )
            payload = cls._spatial_set_to_viewer_json(vertices, dim, idx)
            payload["id"] = f"static-{idx}"
            payload["type"] = "polygon"
            return payload

        if isinstance(obstacle, StaticSphere):
            center = np.asarray(obstacle.pos, dtype=float)
            if center.ndim != 1 or center.shape[0] != dim:
                raise ValueError(
                    f"Static sphere obstacle {idx} has shape {center.shape}, expected ({dim},)."
                )
            return {
                "id": f"static-{idx}",
                "type": "sphere",
                "center": center.tolist(),
                "radius": float(obstacle.radius),
            }

        raise TypeError(f"Unsupported static obstacle type {type(obstacle)!r}.")

    @staticmethod
    def _record_kind(record: ViewerRecord) -> str:
        if isinstance(record, BaseBenchmarkRecord):
            return "base"
        if isinstance(record, STHeuristicAblationRecord):
            return "st_heuristic_ablation"
        return "mrmp"

    @staticmethod
    def _common_instance_fields(
        record: ViewerRecord,
        base_record: BaseBenchmarkRecord,
        base_geometry: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "instance_id": record.instance_id,
            "base_instance_id": getattr(record, "base_instance_id", base_record.instance_id),
            "problem_kind": ViewerManifestBuilder._record_kind(record),
            "domain": base_record.domain,
            "space_dim": base_record.space_dim,
            "spatial_seed": base_record.spatial_seed,
            "env_params": dict(base_record.env_params),
            "stgcs_num_vertices": int(record.stgcs_num_vertices),
            "stgcs_num_edges": int(record.stgcs_num_edges),
            "dimension": int(base_geometry["dimension"]),
            "robot_radius": float(base_geometry["robot_radius"]),
            "bounds": base_geometry["bounds"],
            "spatial_sets": base_geometry["spatial_sets"],
            "static_obstacles": base_geometry.get("static_obstacles", []),
        }

    @classmethod
    def _base_record_to_viewer_instance(
        cls,
        record: BaseBenchmarkRecord,
        base_geometry: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            **cls._common_instance_fields(record, record, base_geometry),
            "time_horizon": 0.0,
            "time_range": [0.0, 0.0],
            "queries": [],
            "dynamic_obstacles": [],
            "supports_sampling": bool(record.supports_sampling),
        }

    @classmethod
    def _mrmp_record_to_viewer_instance(
        cls,
        record: MRMPBenchmarkRecord,
        base_record: BaseBenchmarkRecord,
        base_geometry: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            **cls._common_instance_fields(record, base_record, base_geometry),
            "time_horizon": 0.0,
            "time_range": [0.0, 0.0],
            "queries": [query.to_dict() for query in record.queries],
            "dynamic_obstacles": [],
            "num_agents": int(record.num_agents),
            "independent_cost_sum": float(record.independent_cost_sum),
            "independent_makespan": float(record.independent_makespan),
            "independent_runtime": float(record.independent_runtime),
            "num_conflicting_pairs": int(record.num_conflicting_pairs),
            "num_conflicting_agents": int(record.num_conflicting_agents),
            "conflict_largest_component": int(record.conflict_largest_component),
            "conflict_density": float(record.conflict_density),
        }

    @staticmethod
    def _dynamic_obstacle_to_viewer_json(spec: Dict[str, Any]) -> Dict[str, Any]:
        segments = [
            {
                "start": [float(value) for value in segment["start"]],
                "goal": [float(value) for value in segment["goal"]],
                "t_start": float(segment["t_start"]),
                "t_end": float(segment["t_end"]),
            }
            for segment in spec["segments"]
        ]
        path: List[List[float]] = []
        for segment in segments:
            if not path or path[-1] != segment["start"]:
                path.append(segment["start"])
            path.append(segment["goal"])
        return {
            "type": str(spec["type"]),
            "radius": float(spec["radius"]),
            "segments": segments,
            "path": path,
        }

    @staticmethod
    def _st_time_range(record: STHeuristicAblationRecord) -> List[float]:
        min_time = min(0.0, float(record.query.t_start))
        max_time = max(0.0, float(record.query.t_start))
        for obstacle in record.dynamic_obstacles:
            for segment in obstacle["segments"]:
                t_start = float(segment["t_start"])
                t_end = float(segment["t_end"])
                if math.isfinite(t_start):
                    min_time = min(min_time, t_start)
                    max_time = max(max_time, t_start)
                if math.isfinite(t_end):
                    min_time = min(min_time, t_end)
                    max_time = max(max_time, t_end)
        return [float(min_time), float(max_time)]

    @classmethod
    def _st_heuristic_record_to_viewer_instance(
        cls,
        record: STHeuristicAblationRecord,
        base_record: BaseBenchmarkRecord,
        base_geometry: Dict[str, Any],
    ) -> Dict[str, Any]:
        time_range = cls._st_time_range(record)
        return {
            **cls._common_instance_fields(record, base_record, base_geometry),
            "time_horizon": float(time_range[1]),
            "time_range": time_range,
            "queries": [record.query.to_dict()],
            "dynamic_obstacles": [
                cls._dynamic_obstacle_to_viewer_json(spec)
                for spec in record.dynamic_obstacles
            ],
            "group": record.group,
            "source_domain": record.source_domain,
        }

    @classmethod
    def _record_to_viewer_instance(
        cls,
        record: ViewerRecord,
        base_record: BaseBenchmarkRecord,
        base_geometry: Dict[str, Any],
    ) -> Dict[str, Any]:
        if isinstance(record, BaseBenchmarkRecord):
            return cls._base_record_to_viewer_instance(record, base_geometry)
        if isinstance(record, STHeuristicAblationRecord):
            return cls._st_heuristic_record_to_viewer_instance(record, base_record, base_geometry)
        return cls._mrmp_record_to_viewer_instance(record, base_record, base_geometry)

    @staticmethod
    def _select_records(
        records: Sequence[ViewerRecord],
        instance_ids: Sequence[str],
        limit: int | None,
    ) -> List[ViewerRecord]:
        if instance_ids:
            wanted = set(instance_ids)
            selected = [record for record in records if record.instance_id in wanted]
            found = {record.instance_id for record in selected}
            missing = sorted(wanted - found)
            if missing:
                raise FileNotFoundError(f"Instance ids not found in manifest: {missing}")
        else:
            selected = list(records)
        if limit is not None:
            selected = selected[:limit]
        return selected

    @staticmethod
    def base_root_for_manifest(source_manifest_path: str | Path, base_root: str | Path | None) -> Path:
        if base_root is not None:
            return Path(base_root).resolve()
        return BaseManifestStore.default_base_root(source_manifest_path)

    @classmethod
    def _base_manifest_path_for_record(
        cls,
        record: ViewerRecord,
        source_manifest_path: str | Path,
        base_root: str | Path | None,
    ) -> Path | None:
        if isinstance(record, BaseBenchmarkRecord):
            return None
        root = cls.base_root_for_manifest(source_manifest_path, base_root)
        if isinstance(record, STHeuristicAblationRecord):
            return BaseManifestStore.manifest_path(root, domain_key=record.source_domain)
        return BaseManifestStore.manifest_path(root, domain_key=record.domain_key)

    @classmethod
    def build_manifest(
        cls,
        records: Sequence[ViewerRecord],
        source_manifest_path: str | Path,
        instance_ids: Sequence[str],
        limit: int | None,
        base_root: str | Path | None = None,
    ) -> Dict[str, Any]:
        selected = cls._select_records(records, instance_ids=instance_ids, limit=limit)
        geometry_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        instances: List[Dict[str, Any]] = []

        for record in selected:
            if isinstance(record, BaseBenchmarkRecord):
                base_record = record
            else:
                base_manifest_path = cls._base_manifest_path_for_record(
                    record,
                    source_manifest_path=source_manifest_path,
                    base_root=base_root,
                )
                assert base_manifest_path is not None
                base_record = BaseManifestStore.record_by_id(base_manifest_path, record.base_instance_id)
            key = cls._base_geometry_cache_key(base_record)
            if key not in geometry_cache:
                geometry_cache[key] = cls._build_base_geometry(base_record)
            instances.append(cls._record_to_viewer_instance(record, base_record, geometry_cache[key]))

        source_kind = "unknown"
        if selected:
            source_kind = cls._record_kind(selected[0])

        return {

            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_kind": source_kind,
            "instance_count": len(instances),
            "instances": instances,
        }


class ViewerManifestCLI:
    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=(
                "Export an enriched viewer manifest from a base, mrmp, or ST planning "
                "instance manifest."
            )
        )
        parser.add_argument(
            "--input",
            type=Path,
            required=True,
            help="Path to the raw instance manifest.json file.",
        )
        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            help="Path to the viewer-ready JSON manifest.",
        )
        parser.add_argument(
            "--instance-id",
            action="append",
            default=[],
            help="Specific instance id to export. Can be passed multiple times.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Optional cap on the number of exported records.",
        )
        parser.add_argument(
            "--base-root",
            type=Path,
            default=None,
            help="Optional override for the shared base manifest root.",
        )
        return parser.parse_args()

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        records = ViewerManifestBuilder.load_raw_manifest(args.input)
        payload = ViewerManifestBuilder.build_manifest(
            records,
            source_manifest_path=args.input,
            instance_ids=args.instance_id,
            limit=args.limit,
            base_root=args.base_root,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(
            json.dumps(
                {
                    "input": str(args.input),
                    "output": str(args.output),
                    "source_kind": payload["source_kind"],
                    "instance_count": payload["instance_count"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    ViewerManifestCLI.main()
