from __future__ import annotations

import argparse
from copy import deepcopy
import json
from dataclasses import dataclass
from importlib import resources
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Sequence
from urllib.parse import parse_qs, urlencode, urlparse

import numpy as np

from experiments.base import BaseInstanceFactory
from experiments.manifests.base import BaseBenchmarkRecord, load_manifest as load_base_manifest
from experiments.instance import Instance
from experiments.manifests.st_planning import (
    STHeuristicAblationRecord,
    STPlanningManifestStore,
    STQuerySpec,
)
from visualization.viewer.manifest import ViewerManifestBuilder
from visualization.viewer.planning import ViewerPlannerService
from visualization.viewer.assets import ViewerStaticAssets


@dataclass(frozen=True)
class DraftBuildResult:
    record: STHeuristicAblationRecord
    viewer_manifest: Dict[str, Any]


class STPlanningInstanceCreator:
    DEFAULT_TMAX = 1000.0
    DEFAULT_VLIMIT = 1.0
    DEFAULT_OUTPUT = Path("data/instances/st_planning/custom/manifest.json")
    DEFAULT_QUERY_SAMPLE_ATTEMPTS = 64


    def __init__(self, base_root: str | Path, default_output: str | Path) -> None:
        self.base_root = Path(base_root).resolve()
        self.default_output = Path(default_output)
        self.records_by_domain = self.load_base_records(self.base_root)
        self.current_record: STHeuristicAblationRecord | None = None
        self.current_manifest: Dict[str, Any] = self.empty_viewer_manifest()
        self.solution_service = ViewerPlannerService(self.base_root)

    @staticmethod
    def empty_viewer_manifest() -> Dict[str, Any]:
        return {

            "generated_at": None,
            "source_kind": "st_heuristic_ablation",
            "instance_count": 0,
            "instances": [],
        }

    @staticmethod
    def load_base_records(base_root: str | Path) -> Dict[str, List[BaseBenchmarkRecord]]:
        root = Path(base_root)
        records_by_domain: Dict[str, List[BaseBenchmarkRecord]] = {}
        for manifest_path in sorted(root.glob("*/manifest.json")):
            domain_key = manifest_path.parent.name
            records_by_domain[domain_key] = load_base_manifest(manifest_path)
        if not records_by_domain:
            raise FileNotFoundError(f"No base manifests found under {root.resolve()}")
        return records_by_domain

    @staticmethod
    def record_to_selector_json(record: BaseBenchmarkRecord) -> Dict[str, Any]:
        return {
            "instance_id": record.instance_id,
            "domain": record.domain,
            "domain_key": record.domain_key,
            "space_dim": int(record.space_dim),
            "spatial_seed": int(record.spatial_seed),
            "env_params": dict(record.env_params),
            "supports_sampling": bool(record.supports_sampling),
            "stgcs_num_vertices": int(record.stgcs_num_vertices),
            "stgcs_num_edges": int(record.stgcs_num_edges),
        }

    def domains_payload(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "default_output": str(self.default_output),
            "domains": [
                {
                    "domain_key": domain_key,
                    "records": [self.record_to_selector_json(record) for record in records],
                }
                for domain_key, records in sorted(self.records_by_domain.items())
            ],
        }

    def base_record(self, domain_key: str, base_instance_id: str) -> BaseBenchmarkRecord:
        for record in self.records_by_domain.get(domain_key, []):
            if record.instance_id == base_instance_id:
                return record
        raise KeyError(f"Unknown base instance {base_instance_id!r} in domain {domain_key!r}.")

    @staticmethod
    def parse_point(value: Any, dim: int, field_name: str) -> List[float]:
        if isinstance(value, str):
            parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        elif isinstance(value, Sequence):
            parts = list(value)
        else:
            raise ValueError(f"{field_name} must be a comma-separated point.")
        if len(parts) != dim:
            raise ValueError(f"{field_name} must have exactly {dim} coordinates.")
        point = [float(part) for part in parts]
        if not all(np.isfinite(point)):
            raise ValueError(f"{field_name} contains a non-finite coordinate.")
        return point

    @staticmethod
    def parse_float(value: Any, field_name: str, *, positive: bool = False) -> float:
        parsed = float(value)
        if not np.isfinite(parsed):
            raise ValueError(f"{field_name} must be finite.")
        if positive and parsed <= 0.0:
            raise ValueError(f"{field_name} must be positive.")
        return parsed

    @classmethod
    def normalize_obstacle_specs(
        cls,
        specs: Sequence[Dict[str, Any]],
        dim: int,
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for obstacle_index, spec in enumerate(specs):
            segments = spec.get("segments", [])
            if len(segments) != 1:
                raise ValueError("Only single-segment sphere obstacles are supported by this creator.")
            segment = segments[0]
            t_start = cls.parse_float(segment["t_start"], f"obstacle {obstacle_index} t_start")
            t_end = cls.parse_float(segment["t_end"], f"obstacle {obstacle_index} t_end")
            if t_end <= t_start:
                raise ValueError(f"obstacle {obstacle_index} t_end must be greater than t_start.")
            normalized.append(
                {
                    "type": "sphere",
                    "radius": cls.parse_float(spec["radius"], f"obstacle {obstacle_index} radius", positive=True),
                    "segments": [
                        {
                            "start": cls.parse_point(segment["start"], dim, f"obstacle {obstacle_index} start"),
                            "goal": cls.parse_point(segment["goal"], dim, f"obstacle {obstacle_index} goal"),
                            "t_start": t_start,
                            "t_end": t_end,
                        }
                    ],
                }
            )
        return normalized

    @staticmethod
    def default_point(bounds: Dict[str, Sequence[float]], ratio: float) -> List[float]:
        lb = np.asarray(bounds["min"], dtype=float)
        ub = np.asarray(bounds["max"], dtype=float)
        return (lb + ratio * (ub - lb)).tolist()

    @staticmethod
    def spatial_set_centers(geometry: Dict[str, Any]) -> List[np.ndarray]:
        return [
            np.mean(np.asarray(spatial_set["vertices"], dtype=float), axis=0)
            for spatial_set in geometry["spatial_sets"]
        ]

    @classmethod
    def geometry_query_points(cls, geometry: Dict[str, Any]) -> tuple[List[float], List[float]]:
        centers = cls.spatial_set_centers(geometry)
        if len(centers) < 2:
            return (
                cls.default_point(geometry["bounds"], 0.25),
                cls.default_point(geometry["bounds"], 0.75),
            )
        best_pair = (centers[0], centers[1])
        best_dist = -1.0
        for index, lhs in enumerate(centers):
            for rhs in centers[index + 1:]:
                dist = float(np.linalg.norm(lhs - rhs))
                if dist > best_dist:
                    best_pair = (lhs, rhs)
                    best_dist = dist
        return best_pair[0].tolist(), best_pair[1].tolist()

    @classmethod
    def default_query_is_valid(
        cls,
        stgcs,
        start: Sequence[float],
        goal: Sequence[float],
    ) -> bool:
        query = STQuerySpec(
            start=[float(value) for value in start],
            goal=[float(value) for value in goal],
            t_start=0.0,
            is_stay=True,
            vlimit=cls.DEFAULT_VLIMIT,
        )
        valid, _ = stgcs.validate_query(query.to_query())
        return bool(valid)

    @classmethod
    def default_query_points(
        cls,
        base_record: BaseBenchmarkRecord,
        geometry: Dict[str, Any],
    ) -> tuple[List[float], List[float]]:
        instance = BaseInstanceFactory.from_record(base_record, compute_heuristics=False)
        start, goal = cls.geometry_query_points(geometry)
        if cls.default_query_is_valid(instance.stgcs, start, goal):
            return start, goal

        for _ in range(cls.DEFAULT_QUERY_SAMPLE_ATTEMPTS):
            query = instance.sample_MP_query(t0=0.0, is_stay=True, timeout_secs=1.0)
            if query is None:
                continue
            if cls.default_query_is_valid(instance.stgcs, query.start, query.goal):
                return query.start.tolist(), query.goal.tolist()

        raise ValueError(
            f"Unable to sample a feasible default query for {base_record.instance_id!r} "
            f"after {cls.DEFAULT_QUERY_SAMPLE_ATTEMPTS} attempts."
        )

    def default_draft(self, domain_key: str, base_instance_id: str) -> Dict[str, Any]:
        base_record = self.base_record(domain_key, base_instance_id)
        geometry = ViewerManifestBuilder._build_base_geometry(base_record)
        start, goal = self.default_query_points(base_record, geometry)
        return {
            "domain_key": domain_key,
            "base_instance_id": base_instance_id,
            "instance_id": f"st-custom-{base_instance_id}",
            "group": "custom",
            "tmax": self.DEFAULT_TMAX,
            "dimension": int(geometry["dimension"]),
            "robot_radius": float(geometry["robot_radius"]),
            "bounds": geometry["bounds"],
            "query": {
                "start": start,
                "goal": goal,
                "t_start": 0.0,
                "is_stay": True,
                "vlimit": self.DEFAULT_VLIMIT,
            },
            "dynamic_obstacles": [],
        }

    def draft_from_record(self, record: STHeuristicAblationRecord) -> Dict[str, Any]:
        base_record = self.base_record(record.source_domain, record.base_instance_id)
        geometry = ViewerManifestBuilder._build_base_geometry(base_record)
        return {
            "domain_key": record.source_domain,
            "base_instance_id": record.base_instance_id,
            "instance_id": record.instance_id,
            "group": record.group,
            "tmax": record.tmax,
            "dimension": int(geometry["dimension"]),
            "robot_radius": float(geometry["robot_radius"]),
            "bounds": geometry["bounds"],
            "query": record.query.to_dict(),
            "dynamic_obstacles": deepcopy(record.dynamic_obstacles),
        }

    def record_from_draft(self, draft: Dict[str, Any]) -> STHeuristicAblationRecord:
        domain_key = str(draft["domain_key"])
        base_instance_id = str(draft["base_instance_id"])
        base_record = self.base_record(domain_key, base_instance_id)
        dim = int(base_record.space_dim)
        query_data = dict(draft["query"])
        query = STQuerySpec(
            start=self.parse_point(query_data["start"], dim, "query start"),
            goal=self.parse_point(query_data["goal"], dim, "query goal"),
            t_start=self.parse_float(query_data["t_start"], "query t_start"),
            is_stay=bool(query_data.get("is_stay", True)),
            vlimit=self.parse_float(query_data["vlimit"], "query vlimit", positive=True),
        )
        obstacle_specs = self.normalize_obstacle_specs(draft.get("dynamic_obstacles", []), dim)
        tmax = self.parse_float(draft.get("tmax", self.DEFAULT_TMAX), "tmax", positive=True)
        group = str(draft.get("group") or "custom")

        instance = BaseInstanceFactory.from_record(base_record, compute_heuristics=False)
        env = instance.env.copy()
        env.O_Dynamic = Instance.dynamic_obstacles_from_specs(obstacle_specs)
        stgcs = Instance.build_stgcs_from_env(env, tmax=tmax, vlimit=query.vlimit)
        valid, reason = stgcs.validate_query(query.to_query())
        if not valid:
            detail = "" if reason is None else f": {reason}"
            raise ValueError(f"Query is invalid for the generated ST-GCS{detail}")

        instance_id = str(draft.get("instance_id") or f"st-custom-{base_instance_id}")
        return STHeuristicAblationRecord(
            instance_id=instance_id,
            group=group,
            source_domain=domain_key,
            base_instance_id=base_instance_id,
            query=query,
            dynamic_obstacles=obstacle_specs,
            tmax=tmax,
            stgcs_num_vertices=int(stgcs.G.number_of_nodes()),
            stgcs_num_edges=int(stgcs.G.number_of_edges()),
        )

    def build_from_draft(self, draft: Dict[str, Any]) -> DraftBuildResult:
        record = self.record_from_draft(draft)
        viewer_manifest = ViewerManifestBuilder.build_manifest(
            [record],
            source_manifest_path=Path("data/instances/st_planning/custom/manifest.json"),
            instance_ids=[],
            limit=None,
            base_root=self.base_root,
        )
        return DraftBuildResult(record=record, viewer_manifest=viewer_manifest)

    def update_current(self, draft: Dict[str, Any]) -> DraftBuildResult:
        result = self.build_from_draft(draft)
        self.current_record = result.record
        self.current_manifest = result.viewer_manifest
        return result

    @staticmethod
    def resolve_manifest_path(output_path: str | Path) -> Path:
        target = Path(output_path).expanduser()
        if not target.is_absolute():
            target = Path.cwd() / target
        return target

    @staticmethod
    def load_st_records(path: str | Path) -> List[STHeuristicAblationRecord]:
        target = STPlanningInstanceCreator.resolve_manifest_path(path)
        if not target.exists():
            return []
        return STPlanningManifestStore.load_manifest(target)

    @staticmethod
    def existing_record_to_selector_json(record: STHeuristicAblationRecord) -> Dict[str, Any]:
        return {
            "instance_id": record.instance_id,
            "group": record.group,
            "source_domain": record.source_domain,
            "base_instance_id": record.base_instance_id,
            "obstacle_count": len(record.dynamic_obstacles),
            "stgcs_num_vertices": int(record.stgcs_num_vertices),
            "stgcs_num_edges": int(record.stgcs_num_edges),
        }

    def existing_instances_payload(self, output_path: str | Path) -> Dict[str, Any]:
        target = self.resolve_manifest_path(output_path)
        records = self.load_st_records(target)
        return {
            "ok": True,
            "manifest_path": str(target),
            "records": [self.existing_record_to_selector_json(record) for record in records],
        }

    def existing_draft(self, output_path: str | Path, instance_id: str) -> Dict[str, Any]:
        records = self.load_st_records(output_path)
        for record in records:
            if record.instance_id == instance_id:
                return self.draft_from_record(record)
        raise KeyError(f"Unknown ST planning instance {instance_id!r}.")

    def export_current(self, output_path: str | Path) -> Path:
        if self.current_record is None:
            raise ValueError("No current instance has been previewed.")
        target = self.resolve_manifest_path(output_path)
        records = STPlanningManifestStore.load_manifest(target) if target.exists() else []
        records = STPlanningManifestStore.merge_records_by_instance_id(records, [self.current_record])
        STPlanningManifestStore.save_manifest(target, records)
        return target

    @staticmethod
    def json_response(handler: SimpleHTTPRequestHandler, status: HTTPStatus, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    @staticmethod
    def html_response(handler: SimpleHTTPRequestHandler, html: str) -> None:
        data = html.encode("utf-8")
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    @staticmethod
    def error_response(handler: SimpleHTTPRequestHandler, status: HTTPStatus, exc: Exception) -> None:
        STPlanningInstanceCreator.json_response(handler, status, {"ok": False, "error": str(exc)})

    @staticmethod
    def read_json(handler: SimpleHTTPRequestHandler) -> Dict[str, Any]:
        length = int(handler.headers.get("Content-Length", "0"))
        payload = handler.rfile.read(length)
        return json.loads(payload.decode("utf-8")) if payload else {}

    def make_handler(self):
        app = self

        class Handler(SimpleHTTPRequestHandler):
            def end_headers(self) -> None:
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                super().end_headers()

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                try:
                    if ViewerStaticAssets.is_viewer_path(parsed.path):
                        ViewerStaticAssets.write_html_response(self)
                        return
                    if parsed.path == "/":
                        app.html_response(self, resources.files("visualization.viewer").joinpath("editor.html").read_text())
                        return
                    if parsed.path == "/api/domains":
                        app.json_response(self, HTTPStatus.OK, app.domains_payload())
                        return
                    if parsed.path == "/api/default-draft":
                        params = parse_qs(parsed.query)
                        draft = app.default_draft(
                            str(params.get("domain_key", [""])[0]),
                            str(params.get("base_instance_id", [""])[0]),
                        )
                        app.json_response(self, HTTPStatus.OK, {"ok": True, "draft": draft})
                        return
                    if parsed.path == "/api/manifest":
                        app.json_response(self, HTTPStatus.OK, app.current_manifest)
                        return
                    if parsed.path == "/api/solution-planners":
                        app.json_response(self, HTTPStatus.OK, app.solution_service.planner_payload())
                        return
                    app.error_response(self, HTTPStatus.NOT_FOUND, FileNotFoundError(parsed.path))
                except Exception as exc:
                    app.error_response(self, HTTPStatus.BAD_REQUEST, exc)

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                try:
                    data = app.read_json(self)
                    if parsed.path == "/api/preview":
                        draft = dict(data["draft"])
                        result = app.update_current(draft)
                        app.json_response(
                            self,
                            HTTPStatus.OK,
                            {
                                "ok": True,
                                "draft": draft,
                                "record": result.record.to_dict(),
                            },
                        )
                        return
                    if parsed.path == "/api/existing-instances":
                        app.json_response(
                            self,
                            HTTPStatus.OK,
                            app.existing_instances_payload(str(data["output_path"])),
                        )
                        return
                    if parsed.path == "/api/load-existing":
                        draft = app.existing_draft(
                            str(data["output_path"]),
                            str(data["instance_id"]),
                        )
                        app.json_response(self, HTTPStatus.OK, {"ok": True, "draft": draft})
                        return
                    if parsed.path == "/api/export":
                        target = app.export_current(str(data["output_path"]))
                        app.json_response(self, HTTPStatus.OK, {"ok": True, "output_path": str(target)})
                        return
                    if parsed.path == "/api/solution":
                        if app.current_record is None:
                            raise ValueError("No current instance has been previewed.")
                        instance_id = str(data["instance_id"])
                        if instance_id != app.current_record.instance_id:
                            raise KeyError(f"Instance {instance_id!r} is not the current previewed instance.")
                        result = app.solution_service.run_st_solution(
                            app.current_record,
                            planner_key=str(data["planner_key"]),
                            budget=float(data.get("budget", app.solution_service.DEFAULT_BUDGET)),
                            seed_offset=int(data.get("seed_offset", 0)),
                        )
                        app.json_response(self, HTTPStatus.OK, result)
                        return
                    app.error_response(self, HTTPStatus.NOT_FOUND, FileNotFoundError(parsed.path))
                except Exception as exc:
                    app.error_response(self, HTTPStatus.BAD_REQUEST, exc)

        return Handler


class STPlanningInstanceCreatorCLI:
    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Serve a browser UI for creating one-query dynamic-obstacle ST planning manifests."
        )
        parser.add_argument("--base-root", type=Path, default=Path("data/instances/st_planning/base"))
        parser.add_argument("--output", type=Path, default=STPlanningInstanceCreator.DEFAULT_OUTPUT)
        parser.add_argument("--host", type=str, default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8770)
        return parser.parse_args()

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        app = STPlanningInstanceCreator(base_root=args.base_root, default_output=args.output)
        server = ThreadingHTTPServer((args.host, args.port), app.make_handler())
        viewer_url = (
            f"http://{args.host}:{args.port}/?"
            f"{urlencode({'base_root': str(Path(args.base_root).resolve())})}"
        )
        print(
            json.dumps(
                {
                    "base_root": str(Path(args.base_root).resolve()),
                    "default_output": str(args.output),
                    "url": viewer_url,
                },
                indent=2,
            ),
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    STPlanningInstanceCreatorCLI.main()
