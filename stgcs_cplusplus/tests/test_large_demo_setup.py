"""Check that demo entry points use their saved comparison setup."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib
import io
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from experiments.base import BaseManifestStore
from experiments.base_serialization import SerializedBaseInstanceStore
from experiments.manifests.mrmp import load_manifest
from experiments.mrmp import MRMPExperiment
from stgcs_native.client import NativeClient
from stgcs_native.types import SearchPlannerSpec
from visualization.solution_io import SolutionIO


class LargeDemoSetupTests(unittest.TestCase):
    CASES = (
        ("demos.demo_r50_four_room", 50, 120., "h_max"),
        ("demos.demo_r100_empty", 100, 120., "h_max"),
        ("demos.demo_r32_sphere_exchange", 32, 120., "h_max"),
        ("demos.demo_r48_village", 48, 300., "h_mot_tri"),
    )

    @staticmethod
    def setup_case(module_name):
        module = importlib.import_module(module_name)
        demo = module.DemoR48Village if module_name.endswith("village") else module
        record, = [r for r in load_manifest(demo.DEFAULT_MANIFEST_PATH) if r.instance_id == demo.INSTANCE_ID]
        base = BaseManifestStore.record_by_id(
            demo.DEFAULT_BASE_ROOT / record.domain_key / "manifest.json", demo.BASE_INSTANCE_ID)
        mrmp = json.loads(demo.DEFAULT_SOLUTION_OUTPUT.read_text())
        smooth = json.loads(demo.DEFAULT_TRAJOPT_OUTPUT.read_text())
        return demo, record, base, mrmp, smooth

    @staticmethod
    def run_cli(module_name, arguments):
        with patch.object(sys, "argv", [module_name, *arguments]), redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            runpy.run_module(module_name, run_name="__main__")

    def test_saved_inputs_and_solver_settings_match(self):
        for module_name, count, budget, heuristic in self.CASES:
            with self.subTest(demo=module_name):
                demo, record, base, mrmp, smooth = self.setup_case(module_name)
                instance = SerializedBaseInstanceStore.load_instance(base)
                self.assertIsNotNone(instance)
                self.assertEqual((record.num_agents, record.tmax), (count, 100.))
                self.assertEqual(record.base_instance_id, base.instance_id)
                self.assertEqual(demo.DEFAULT_SOLUTION_OUTPUT.parent.name, demo.DEFAULT_MANIFEST_PATH.parent.name)
                self.assertEqual(demo.DEFAULT_TRAJOPT_OUTPUT.parent.name, demo.DEFAULT_MANIFEST_PATH.parent.name)
                self.assertEqual(base.instance_id, f"base-{base.domain_key}-seed{base.spatial_seed:05d}")
                query_kind = "swap" if count == 32 else "random"
                self.assertEqual(record.instance_id, f"mrmp-{base.instance_id}-{count}-robot-{query_kind}")
                self.assertNotIn("sampling", base.env_params)
                self.assertNotIn("sampling", getattr(instance, "env_params", {}))
                self.assertEqual(demo.LOW_LEVEL_HEURISTIC, heuristic)
                self.assertEqual(demo.LOW_LEVEL_DOMINANCE_CHECKS, ("GUB", "delta_pos"))
                self.assertTrue(demo.MRMP_PLANNER_SETTINGS.dynamic_window_adjustment)
                self.assertEqual((demo.WINDOW_ALPHA, demo.WINDOW_BETA, demo.EPSILON), (2., 1., 8.))
                spec = SearchPlannerSpec(heuristic, ("GUB", "delta_pos"), epsilon=8.)
                radius, speed = instance.env.robot_radius, instance.stgcs.vlimit
                settings = demo.TRAJOPT_SETTINGS
                config = settings.config(radius, speed)
                np.testing.assert_allclose(
                    [config.sample_dt, settings.window_span(radius, speed), settings.stride(radius, speed)],
                    [.1, .5, .25], rtol=0, atol=1e-14)
                self.assertEqual(config.solver_max_iter, 1_000_000)
                for payload in (mrmp, smooth):
                    SolutionIO.validate_solution_payload(payload, record, payload["planner_key"], budget)
                    if heuristic == "h_mot_tri":
                        self.assertIn("h-mot-tri", payload["planner_key"])
                    else:
                        self.assertIn(spec.name, payload["planner_name"])
                    for key, value in dict(window_alpha=2., window_beta=1., epsilon=8.,
                                           dynamic_window_adjustment=True, child_expansion_rule="num_conflicts",
                                           tmax=100.).items():
                        self.assertEqual(payload[key], value)
                    paths = SolutionIO.pwl_trajectories_from_solution_payload(payload, count)
                    for path, query in zip(paths, record.queries, strict=True):
                        np.testing.assert_allclose(path.x0[:-1], query.start, rtol=0, atol=1e-6)
                        np.testing.assert_allclose(path.xT[:-1], query.goal, rtol=0, atol=1e-6)
                for key in ("sample_dt", "min_sample_dt", "clearance_margin", "velocity_gradient_weight",
                            "displacement_weight", "displacement_jitter_weight", "solver_eps_abs", "solver_eps_rel",
                            "include_interval_collision_constraints", "include_trajectory_knot_times", "fix_terminal_states"):
                    self.assertAlmostEqual(getattr(config, key), smooth["global_trajopt"][key], places=14)
                if count == 100:
                    np.testing.assert_allclose(np.asarray(instance.env.ub) - instance.env.lb, [10.5, 10.5])
                    self.assertAlmostEqual(radius, .15)

    def test_default_entrypoints_only_load_saved_solutions(self):
        with patch.object(MRMPExperiment, "run_stgcs", side_effect=AssertionError("Unexpected planning")), \
             patch.object(MRMPExperiment, "run_trajopt", side_effect=AssertionError("Unexpected smoothing")), \
             patch.object(NativeClient, "run_process", side_effect=AssertionError("Unexpected native solve")):
            for module_name, *_ in self.CASES:
                with self.subTest(demo=module_name):
                    self.run_cli(module_name, ["--no-serve"])

    def test_force_solve_uses_shared_api_and_refreshes_smoothing(self):
        for module_name, count, budget, heuristic in self.CASES:
            with self.subTest(demo=module_name), TemporaryDirectory() as directory:
                _, record, base, mrmp, smooth = self.setup_case(module_name)
                planned, refined = Path(directory) / "mrmp.json", Path(directory) / "trajopt.json"
                planned.write_text(json.dumps(mrmp))
                refined.write_text(json.dumps(smooth))
                with patch.object(MRMPExperiment, "run_stgcs", return_value=deepcopy(mrmp)) as plan, \
                     patch.object(MRMPExperiment, "run_trajopt", return_value=deepcopy(smooth)) as optimize, \
                     patch.object(NativeClient, "run_process", side_effect=AssertionError("Unexpected native solve")):
                    self.run_cli(module_name, ["--force-solve", "--no-serve",
                        "--solution-output", str(planned), "--trajopt-output", str(refined)])
                plan.assert_called_once()
                optimize.assert_called_once()
                args, kwargs = plan.call_args
                self.assertEqual(args[0].to_dict(), record.to_dict())
                self.assertEqual(args[1].to_dict(), base.to_dict())
                self.assertEqual(args[2], SearchPlannerSpec(heuristic, ("GUB", "delta_pos"), epsilon=8.))
                self.assertEqual(kwargs, dict(budget=budget, window_alpha=2., window_beta=1., child_expansion_rule="num_conflicts"))
                self.assertEqual(optimize.call_args.args[2], mrmp)
                self.assertEqual(json.loads(planned.read_text()), mrmp)
                self.assertEqual(json.loads(refined.read_text()), smooth)

    def test_village_viewer_uses_shared_api(self):
        demo, record, base, mrmp, smooth = self.setup_case("demos.demo_r48_village")
        with patch.object(MRMPExperiment, "run_stgcs", return_value=mrmp) as plan, \
             patch.object(MRMPExperiment, "run_trajopt", return_value=smooth) as optimize:
            result = demo.solve_record(record.to_dict(), base, 300., 2., 8., demo.PLANNER_KEY)
        self.assertEqual(result, smooth)
        self.assertEqual(plan.call_args.args[1].instance_id, base.instance_id)
        self.assertEqual(plan.call_args.args[2].heuristic, "h_mot_tri")
        optimize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
