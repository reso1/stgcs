import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from experiments.reports.mrmp_trajectories import Config, MRMPQualitativeComparison as Report


class MRMPQualitativeTests(unittest.TestCase):
    @staticmethod
    def rows():
        rows = {key: {} for key in Report.PLANNERS}
        # Prefer an advantage over both baselines to a larger one over only one.
        examples = {"balanced": ((1, 4, 2), (4, 8, 6), (4, 8, 5)),
                    "one_baseline": ((1, 100, .5), (4, 40, 8), (4, 40, 8)),
                    "failed": ((1, 5, 5), (1, 10, 10), (1, 10, 10)),
                    "wrong_budget": ((1, 5, 5), (1, 10, 10), (1, 10, 10))}
        for name, measurements in examples.items():
            for i, key in enumerate(rows):
                rows[key][name] = dict(instance_id=name, domain_key="maze", num_agents=Report.NUM_AGENTS,
                                      budget=120., is_success=True,
                                      **{field: measurements[j][i] for j, field in enumerate(Report.METRICS)})
        rows[Report.PLANNERS[-1]]["failed"]["is_success"] = False
        rows[Report.PLANNERS[-1]]["wrong_budget"]["budget"] = 30.
        return rows

    def test_selection_compares_both_baselines_on_matching_successes(self):
        ranked = Report.rank_candidates(self.rows(), "maze")
        self.assertEqual([row["instance_id"] for row in ranked], ["balanced", "one_baseline"])
        self.assertEqual(ranked[0]["ratios"], dict(runtime=2., sum_of_costs=1.5, makespan=1.25))
        self.assertEqual(Report.rank_candidates(self.rows(), "iris-2d"), [])

    def test_selection_requires_figure16_common_success(self):
        rows = {key: [row for name, row in entries.items() if name in ("balanced", "one_baseline")]
                for key, entries in self.rows().items()}
        rows[Config.PBS_KEY] = [dict(row, is_success=row["instance_id"] != "balanced")
                                for row in rows[Report.PLANNERS[0]]]
        records = {row["instance_id"]: SimpleNamespace(instance_id=row["instance_id"], domain_key="maze")
                   for row in rows[Report.PLANNERS[0]]}
        with tempfile.TemporaryDirectory() as directory, patch.object(Report, "DOMAINS", {"maze": "maze"}):
            root = Path(directory)
            measurements = {}
            for key in rows:
                path = Config.result_path(root, Report.NUM_AGENTS, "maze", key)
                path.parent.mkdir(parents=True, exist_ok=True)
                measurements[path] = [dict(row) for row in rows[key]]
            snapshot = root / "figure16.json"
            snapshot.write_text(json.dumps(dict(budget=120., partial=False, rows=rows)))
            with patch("experiments.reports.mrmp_trajectories.MRMPPerformanceComparisonReport._load_domain_rows",
                       side_effect=lambda path, domain: measurements[path]):
                selected, _ = Report.select(root, records, snapshot)
                # The strongest three-planner example failed another eligible Fig. 16 planner.
                self.assertEqual(selected["maze"]["instance_id"], "one_baseline")
                self.assertEqual(selected["maze"]["figure16_common_success_ids"], ["one_baseline"])
                with self.assertRaisesRegex(ValueError, "not a common success"):
                    Report.select(root, records, snapshot, {"maze": "balanced"})
                measurements[path][0]["runtime"] += 1
                with self.assertRaisesRegex(ValueError, "Measurements differ"):
                    Report.select(root, records, snapshot)

    def test_qualitative_window_span_is_two(self):
        backend = Mock()
        backend.run_with_solutions.return_value = ([], SimpleNamespace(
            is_success=False, runtime=120., sum_of_costs=float("inf"), makespan=float("inf")))
        Report.solve(Report.PLANNERS[0], Mock(), [], backend)
        self.assertEqual(backend.run_with_solutions.call_args.args[5], 2.)

    def test_strrt_uses_benchmark_defaults(self):
        module = "experiments.reports.mrmp_trajectories"
        entry = SimpleNamespace(is_success=False, runtime=120., sum_of_costs=float("inf"), makespan=float("inf"))
        with patch(module + ".FixedPrioritySTRRTStarPlanner") as planner, patch(
                module + ".MRMPExperiment._entry_from_fixed_priority_strrt_star_snapshot", return_value=entry):
            planner.return_value.solve.return_value.final_solution.solutions = []
            Report.solve(Report.PLANNERS[1], SimpleNamespace(stgcs=SimpleNamespace(tmax=100.)), [], None)
            planner.assert_called_once_with()

    @staticmethod
    def contains_segment(start, goal, tol):
        return True

    @classmethod
    def validation_inputs(cls, crossing=False):
        endpoints = (([0, 0], [1, 1]), ([0, 1], [1, 0])) if crossing else (
                     ([0, 0], [1, 0]), ([0, 1], [1, 1]))
        queries = [SimpleNamespace(start=np.array(a), goal=np.array(b), t_start=0., vlimit=1., is_stay=True)
                   for a, b in endpoints]
        result = dict(success=True, runtime=.1, sum_of_costs=2., makespan=1.,
                      trajectories=[[[*a, 0., *b, 1.]] for a, b in endpoints])
        env = SimpleNamespace(lb=[0, 0], ub=[1, 1], robot_radius=.05, O_Static=[],
                              is_segment_in_CSpace=cls.contains_segment)
        return result, env, queries

    def test_validation_checks_actual_plotted_costs(self):
        result, env, queries = self.validation_inputs()
        checked = Report.validate(result, Report.PLANNERS[0], env, queries, 2.)
        self.assertEqual(checked["continuous_pair_checks"], 1)
        result["sum_of_costs"] = 3.
        with self.assertRaisesRegex(ValueError, "costs differ"):
            Report.validate(result, Report.PLANNERS[0], env, queries, 2.)

    def test_validation_detects_collision_between_segment_endpoints(self):
        result, env, queries = self.validation_inputs(crossing=True)
        with self.assertRaisesRegex(ValueError, "Inter-robot collision"):
            Report.validate(result, Report.PLANNERS[0], env, queries, 2.)

    def test_kcbs_goal_tolerance_matches_native_infinity_norm(self):
        result, env, queries = self.validation_inputs()
        for query in queries:
            query.goal = query.goal + .049
        checked = Report.validate(result, Report.PLANNERS[-1], env, queries, 2.)
        self.assertEqual(checked["collision_check_step"], .1)

    def test_kcbs_separation_matches_native_disk_checker(self):
        result, env, queries = self.validation_inputs()
        env.robot_radius = .1
        env.ub = [2., 2.]
        result["trajectories"][1] = [[.15, .15, 0., 1.15, .15, 1.]]
        queries[1].start = np.array([.15, .15])
        queries[1].goal = np.array([1.15, .15])
        Report.validate(result, Report.PLANNERS[-1], env, queries, 2.)
        with self.assertRaisesRegex(ValueError, "Inter-robot collision"):
            Report.validate(result, Report.PLANNERS[0], env, queries, 2.)

    def test_static_wall_contact_and_penetration_are_distinguished(self):
        result, env, queries = self.validation_inputs()
        wall = np.array([[-1., -1.], [2., -1.], [2., -.05], [-1., -.05]])
        env.O_Static = [SimpleNamespace(vertices=wall)]
        Report.validate(result, Report.PLANNERS[0], env, queries, 2.)
        wall[2:, 1] += .001
        with self.assertRaisesRegex(ValueError, "intersects a static obstacle"):
            Report.validate(result, Report.PLANNERS[0], env, queries, 2.)

    def test_workspace_caveat_is_explicit_and_limited_to_kcbs(self):
        result, env, queries = self.validation_inputs()
        result["trajectories"][0][0][3] = 1.1
        queries[0].goal = np.array([1.1, 0.])
        queries[0].vlimit = 2.
        with self.assertRaisesRegex(ValueError, "workspace bounds"):
            Report.validate(result, Report.PLANNERS[-1], env, queries, 2.)
        checked = Report.validate(result, Report.PLANNERS[-1], env, queries, 2., True)
        self.assertFalse(checked["workspace_bounds_valid"])
        self.assertEqual(checked["workspace_bounds_violations"][0]["robot"], 1)
        with self.assertRaisesRegex(ValueError, "workspace bounds"):
            Report.validate(result, Report.PLANNERS[0], env, queries, 2., True)


if __name__ == "__main__":
    unittest.main()
