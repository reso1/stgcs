"""Serve saved instances and solutions in the shared browser viewer."""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from visualization.viewer.assets import ViewerStaticAssets
from visualization.viewer.manifest import ViewerManifestBuilder


class ViewerServer:
    def __init__(self, manifest: dict, planners: dict, solve: Callable[[dict], dict]):
        self.manifest = manifest
        self.planners = planners
        self.solve = solve

    def make_handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            def end_headers(self):
                self.send_header("Cache-Control", "no-store")
                super().end_headers()

            def json_response(self, status: HTTPStatus, payload: dict):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)

            def do_HEAD(self):
                self.do_GET()

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/":
                    self.send_response(HTTPStatus.FOUND)
                    self.send_header("Location", ViewerStaticAssets.viewer_url())
                    self.end_headers()
                elif ViewerStaticAssets.is_viewer_path(path):
                    ViewerStaticAssets.write_html_response(self)
                elif path == "/api/manifest":
                    self.json_response(HTTPStatus.OK, app.manifest)
                elif path == "/api/solution-planners":
                    self.json_response(HTTPStatus.OK, app.planners)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self):
                if urlparse(self.path).path != "/api/solution":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    request = json.loads(self.rfile.read(length)) if length else {}
                    payload = app.solve(request)
                except Exception as error:
                    self.json_response(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})
                    return
                self.json_response(HTTPStatus.OK, payload)

        return Handler

    def serve(self, host: str = "127.0.0.1", port: int = 8765):
        with ThreadingHTTPServer((host, port), self.make_handler()) as server:
            print(f"Viewer: http://{host}:{server.server_port}{ViewerStaticAssets.viewer_url()}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        base_root: Path | None = None,
        instance_ids: Sequence[str] = (),
        solution_paths: Sequence[Path] = (),
        limit: int | None = None,
    ) -> ViewerServer:
        from visualization.viewer.planning import ViewerPlannerService

        records = ViewerManifestBuilder.load_raw_manifest(manifest_path)
        payload = ViewerManifestBuilder.build_manifest(
            records, source_manifest_path=manifest_path, base_root=base_root,
            instance_ids=instance_ids, limit=limit,
        )
        solutions = []
        for path in solution_paths:
            solution = json.loads(Path(path).read_text())
            solutions.extend(solution if isinstance(solution, list) else [solution])
        if solutions:
            payload["solutions"] = solutions
        exposed = {instance["instance_id"] for instance in payload["instances"]}
        service = ViewerPlannerService(
            ViewerManifestBuilder.base_root_for_manifest(manifest_path, base_root),
            records=[record for record in records if record.instance_id in exposed],
        )
        return cls(payload, service.planner_payload(), service.solve_request)

    @classmethod
    def start(
        cls, manifest_path: Path, base_root: Path, instance_id: str,
        solution_json: Path | Sequence[Path], host: str = "127.0.0.1", port: int = 8765,
    ) -> None:
        solutions = [solution_json] if isinstance(solution_json, Path) else list(solution_json)
        for path in solutions:
            if not path.is_file():
                raise FileNotFoundError(f"Missing saved solution: {path}")
        cls.from_manifest(manifest_path, base_root, [instance_id], solutions).serve(host, port)

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--input", type=Path, required=True, help="Base, ST, or MRMP manifest")
        parser.add_argument("--base-root", type=Path)
        parser.add_argument("--instance-id", action="append", default=[])
        parser.add_argument("--limit", type=int)
        parser.add_argument("--solution-json", action="append", type=Path, default=[])
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8765)
        args = parser.parse_args()
        cls.from_manifest(args.input, args.base_root, args.instance_id, args.solution_json, args.limit).serve(args.host, args.port)


if __name__ == "__main__":
    ViewerServer.main()
