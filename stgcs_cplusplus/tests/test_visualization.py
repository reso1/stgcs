"""Exercise saved-data visualization without running planners or precomputation."""
from __future__ import annotations

import contextlib
import http.client
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from unittest.mock import patch

import numpy as np

from experiments.manifests.st_planning import STPlanningManifestStore
from stgcs_native import NativeClient
from visualization.animation.common import GifAnimation
from visualization.solution_io import SolutionIO
from visualization.viewer.assets import ViewerStaticAssets
from visualization.viewer.editor import STPlanningInstanceCreator
from visualization.viewer.planning import ViewerPlannerService
from visualization.viewer.server import ViewerServer


class VisualizationTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]

    def setUp(self):
        self.native = patch.object(NativeClient, "run_process", side_effect=AssertionError("Unexpected planning/precomputation"))
        self.native.start()
        self.addCleanup(self.native.stop)

    @staticmethod
    @contextlib.contextmanager
    def serving(app):
        with ThreadingHTTPServer(("127.0.0.1", 0), app.make_handler()) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            try:
                yield connection
            finally:
                connection.close()
                server.shutdown()
                thread.join(timeout=5)

    @staticmethod
    def request(connection, method, path, payload=None):
        body = None if payload is None else json.dumps(payload)
        connection.request(method, path, body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()

    def test_saved_demo_http_routes_and_planner_dispatch(self):
        directory = self.ROOT / "data/solutions/demo_r50_four_room"
        paths = [directory / f"{stage}_solution.json" for stage in ("mrmp", "trajopt")]
        saved = [json.loads(p.read_text()) for p in paths]
        app = ViewerServer.from_manifest(
            self.ROOT / "data/instances/mrmp/demo_r50_four_room/manifest.json",
            solution_paths=paths,
        )
        with self.serving(app) as connection:
            status, headers, _ = self.request(connection, "GET", "/")
            self.assertEqual(status, 302)
            self.assertEqual(headers["Location"], ViewerStaticAssets.viewer_url())
            status, _, html = self.request(connection, "GET", headers["Location"])
            self.assertEqual(status, 200)
            self.assertEqual(html, ViewerStaticAssets.html_bytes())
            status, headers, body = self.request(connection, "HEAD", "/viewer.html")
            self.assertEqual((status, body), (200, b""))
            self.assertEqual(int(headers["Content-Length"]), len(html))
            status, _, body = self.request(connection, "GET", "/api/manifest")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["solutions"], saved)
            self.assertEqual(len(json.loads(body)["instances"][0]["queries"]), 50)
            status, _, body = self.request(connection, "GET", "/api/solution-planners")
            self.assertEqual(json.loads(body), ViewerPlannerService.planner_payload())
            request = dict(instance_id=saved[0]["instance_id"], planner_key="test", budget=7,
                           window_alpha=2, window_beta=.5, epsilon=8)
            with patch.object(ViewerPlannerService, "run_solution", return_value=saved[0]) as solve:
                status, _, body = self.request(connection, "POST", "/api/solution", request)
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body), saved[0])
                self.assertEqual(solve.call_args.kwargs, dict(planner_key="test", budget=7.,
                    seed_offset=0, window_span_factor=2., execution_horizon_factor=.5, epsilon=8.))
                request["instance_id"] = "unexposed"
                self.assertEqual(self.request(connection, "POST", "/api/solution", request)[0], 400)
                solve.assert_called_once()
            self.assertEqual(self.request(connection, "HEAD", "/README.md")[0], 404)
            self.assertEqual(self.request(connection, "POST", "/unknown", {})[0], 404)

    def test_base_and_st_manifests(self):
        for relative, kind in (("data/stgcs_base/grid2d/manifest.json", "base"),
                               ("data/instances/st_planning/manifest.json", "st_heuristic_ablation")):
            with self.subTest(kind=kind):
                app = ViewerServer.from_manifest(self.ROOT / relative, limit=1)
                self.assertEqual(app.manifest["source_kind"], kind)
                self.assertEqual(app.manifest["instance_count"], 1)
                self.assertTrue(app.manifest["instances"][0]["spatial_sets"])
                if kind == "st_heuristic_ablation":
                    self.assertTrue(app.manifest["instances"][0]["dynamic_obstacles"])

    def test_editor_exports_and_reloads_the_selected_time_horizon(self):
        record = STPlanningManifestStore.load_manifest(self.ROOT / "data/instances/st_planning/manifest.json")[0]
        with TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            editor = STPlanningInstanceCreator(self.ROOT / "data/instances/st_planning/base", output)
            draft = editor.draft_from_record(record)
            self.assertEqual(draft["tmax"], record.tmax)
            draft.update(instance_id="editor-test", tmax=123., dynamic_obstacles=[])
            built = editor.update_current(draft)
            self.assertEqual(built.record.tmax, 123.)
            editor.export_current(output)
            reloaded, = STPlanningManifestStore.load_manifest(output)
            self.assertEqual(reloaded.to_dict(), built.record.to_dict())
            self.assertEqual(editor.existing_draft(output, "editor-test")["tmax"], 123.)
            with self.serving(editor) as connection:
                status, _, html = self.request(connection, "GET", "/")
                self.assertEqual(status, 200)
                self.assertIn(b"ST Planning Instance Creator", html)
                self.assertEqual(self.request(connection, "GET", "/viewer.html")[0], 200)

    def test_saved_trajectories_keep_coordinates_and_interpolate_waits(self):
        payload = dict(dimension=2, segments=[
            dict(start=[0., 0.], goal=[0., 0.], t_start=0., t_end=1.),
            dict(start=[0., 0.], goal=[1., 0.], t_start=1., t_end=2.),
        ])
        trajectory = SolutionIO.trajectory_from_viewer_json(payload)
        serialized = SolutionIO.trajectory_to_viewer_json(trajectory)
        self.assertEqual(serialized["segments"], payload["segments"])
        segments = np.asarray(trajectory.points).reshape(-1, 2, 3)
        for time, expected in ((0, [0, 0]), (.5, [0, 0]), (1.5, [.5, 0]), (3, [1, 0])):
            np.testing.assert_allclose(GifAnimation.position(segments, time), expected)
        self.assertEqual(GifAnimation.times(0., 1., 2, 1.), [0., .5, 1.])
        for fps, speed in ((0, 1), (20, 0), (float("nan"), 1)):
            with self.assertRaises(ValueError):
                GifAnimation.times(0., 1., fps, speed)


if __name__ == "__main__":
    unittest.main()
