from concurrent.futures import ThreadPoolExecutor
import struct
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

from pydrake.geometry.optimization import HPolyhedron

from experiments.native import STGCSPlannerClient


class SharedNativeHeuristicTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        executable = self.root / "exe"
        executable.write_text("native build")
        self.st = STGCSPlannerClient(executable, self.root / "st", self.root / "shared")
        self.mrmp = STGCSPlannerClient(executable, self.root / "mrmp", self.root / "shared")
        self.env = SimpleNamespace(_CSpace_hpoly=[HPolyhedron.MakeBox([-2, -2], [2, 2])], robot_radius=.1)
        self.query = SimpleNamespace(start=[-1, 0], goal=[1, 0], t_start=0, is_stay=True, vlimit=1)
        self.path = self.root / "st.txt"
        self.st.write_input(self.path, self.env, self.query, 100, [])
        self.st.bind_cache(self.path, "base-shared-test", "simple2d")
        self.builds = []

    def fake_process(self, command, *_):
        if command[1] == "inspect":
            return {"st_vertices": 1, "st_edges": 0}
        self.assertEqual(command[1], "cache")
        self.builds.append(command)
        time.sleep(.02)
        Path(command[3]).write_bytes(struct.pack("<8sBBdd5Q", STGCSPlannerClient.CACHE_MAGIC, 1, 1, .1, .2, 0, 0, 0, 0, 0))
        return dict(complete=True, h_tri=.1, h_tab=.2, vertices=1, edges=0, st_vertices=1, st_edges=0)

    def test_query_and_reservation_independence_and_domain_invalidation(self):
        original = self.st.cache_path(self.path)
        other = self.root / "mrmp.txt"
        query = SimpleNamespace(start=[0, -1], goal=[0, 1], t_start=3, is_stay=False, vlimit=1)
        obstacles = [dict(radius=.1, segments=[dict(start=[1, 1], goal=[1, 0], t_start=0, t_end=20)])]
        self.st.write_input(other, self.env, query, 100, obstacles)
        self.st.bind_cache(other, "base-shared-test", "simple2d")
        self.assertEqual(original, self.mrmp.cache_path(other))
        with patch.object(STGCSPlannerClient, "run_process", side_effect=self.fake_process):
            self.st.cache(self.path)
            for horizon, speed, extent in ((20, 1, 2), (100, 2, 2), (100, 1, 3)):
                query.vlimit = speed
                env = SimpleNamespace(_CSpace_hpoly=[HPolyhedron.MakeBox([-extent, -2], [2, 2])], robot_radius=.1)
                self.st.write_input(other, env, query, horizon, [])
                with self.assertRaisesRegex(ValueError, "Cached geometry differs"):
                    self.mrmp.cache(other)
            table = original.read_bytes()
            receipt = original.with_suffix(".json").read_bytes()
            self.st.executable.write_text("planner-only rebuild")
            self.st.cache(self.path)
            self.mrmp.cache(self.path)
            self.assertEqual(len(self.builds), 1)
            self.assertEqual(original.read_bytes(), table)
            self.assertEqual(original.with_suffix(".json").read_bytes(), receipt)
        self.assertEqual(STGCSPlannerClient.CACHE_ROOT, STGCSPlannerClient.CACHE_ROOT)
        self.assertEqual(STGCSPlannerClient.CACHE_ROOT, STGCSPlannerClient.ROOT / "data/stgcs_base")

    def test_incomplete_table_is_rejected(self):
        with patch.object(STGCSPlannerClient, "run_process", side_effect=self.fake_process):
            self.st.cache(self.path)
            path = self.st.cache_path(self.path)
            table = bytearray(path.read_bytes())
            table[8] = 0
            path.write_bytes(table)
            with self.assertRaisesRegex(ValueError, "Incomplete or invalid"):
                self.mrmp.cache(self.path)
            self.assertEqual(len(self.builds), 1)

    def test_concurrent_st_mrmp_build_once_and_reject_corruption(self):
        with patch.object(STGCSPlannerClient, "run_process", side_effect=self.fake_process):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda backend: backend.cache(self.path), (self.st, self.mrmp)))
            self.assertEqual(len(self.builds), 1)
            self.assertTrue(all(result["complete"] for result in results))
            path = self.st.cache_path(self.path)
            self.assertEqual(path, self.root / "shared/simple2d/heuristics/base-shared-test/heuristics.bin")
            self.assertEqual(self.st.cache(self.path), results[0])
            self.assertEqual(len(self.builds), 1)
            path.write_text("damaged table")
            with self.assertRaisesRegex(ValueError, "Truncated native cache"):
                self.mrmp.cache(self.path)


if __name__ == "__main__":
    unittest.main()
