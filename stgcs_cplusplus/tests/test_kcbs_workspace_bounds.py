"""Native integration regressions for bounded K-CBS control propagation."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from baselines.ompl_kcbs import OfficialOMPLKCBS, OfficialOMPLKCBSOptions, OfficialOMPLKCBSPlanner
from experiments.environment.env import Env
from experiments.environment.obstacle import StaticPolygon
from stgcs_native.types import MPQuery


class KCBSWorkspaceBoundsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("STGCS_TEST_KCBS_NATIVE")
        if path:
            name = OfficialOMPLKCBS._NATIVE_MODULE
            spec = importlib.util.spec_from_file_location(name, Path(path).resolve())
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[name] = module
        if not OfficialOMPLKCBS.is_available():
            raise unittest.SkipTest("Build the native K-CBS adapter to run its integration regressions")

    @staticmethod
    def solve(blocked):
        square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        # The wall extends past both workspace edges: an in-bounds solution
        # cannot pass it. The old checker found paths around its exterior ends.
        wall = StaticPolygon(np.array([[.45, -.01], [.55, -.01], [.55, 1.01], [.45, 1.01]]))
        env = Env("bounds-regression", [square], .02, [wall] if blocked else [], [],
                  domain_lb=np.zeros(2), domain_ub=np.ones(2))
        query = MPQuery(np.array([.2, .5]), np.array([.8, .5]), 0., True, 1.)
        planner = OfficialOMPLKCBSPlanner(OfficialOMPLKCBSOptions(
            low_level_solve_time=.5, propagation_step_size=.02, min_control_duration=1,
            max_control_duration=10, goal_tolerance=.03, goal_bias=.1,
            intermediate_states=True, num_threads=1), seed=1)
        return planner.solve(SimpleNamespace(env=env), [query], timeout_secs=3.)

    def test_cannot_escape_workspace_to_bypass_a_spanning_wall(self):
        result = self.solve(blocked=True)
        self.assertFalse(result.success, "A reported success crossed a wall or escaped the workspace")

    def test_feasible_query_retains_an_in_bounds_trajectory(self):
        result = self.solve(blocked=False)
        self.assertTrue(result.success)
        points = np.asarray(result.solutions[0].points).reshape(-1, 3)
        self.assertTrue(np.all(points[:, :2] >= -1e-9))
        self.assertTrue(np.all(points[:, :2] <= 1. + 1e-9))


if __name__ == "__main__":
    unittest.main()
