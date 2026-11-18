import csv
from dataclasses import replace
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from experiments.planners.st_planning import STPerformanceComparisonConfig as Config
from experiments.planners.mrmp import SearchPlannerSpec
from experiments.runners.st import STSuite
from experiments.native import STGCSPlannerClient
from experiments.reports.heuristic_scaling import HeuristicInflationAndScalingPlot as InflationPlot


class STScheduleTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("STGCS_NATIVE_EXECUTABLE"), "Requires installed native baselines")
    def test_strrt_final_keeps_in_budget_solution_after_search_deadline(self):
        import numpy as np
        from baselines.ompl_strrt_star import OfficialOMPLSTRRTStar, OfficialOMPLSTRRTStarOptions
        from experiments.environment.env import Env

        env = Env('strrt_budget', [np.array([[0., 0.], [4., 0.], [4., 4.], [0., 4.]])],
                  robot_radius=0.1)
        budget = 0.2
        snapshots = OfficialOMPLSTRRTStar(env, seed=0, vlimit=1).solve_with_snapshots(
            np.array([0.5, 0.5]), np.array([3.5, 3.5]), 0,
            OfficialOMPLSTRRTStarOptions(max_runtime_in_secs=budget,
                                        time_upper_bound=10, stay_at_goal=True))
        for solution in (snapshots.first_solution, snapshots.final_solution):
            self.assertTrue(solution.is_success)
            self.assertLessEqual(solution.time, budget)
            self.assertTrue(Config.entry_from_shortest_path_solution(solution, budget + 1, budget).is_success)
        self.assertLessEqual(snapshots.first_solution.time, snapshots.final_solution.time)
        self.assertLessEqual(snapshots.final_solution.cost, snapshots.first_solution.cost)

    def test_external_results_respect_budget_and_snapshot_time(self):
        from baselines.common import ShortestPathSolution

        for timestamp, elapsed, expected in ((599.9, 601, True), (600, 601, True),
                                             (600.1, 601, False), (-1, 601, False)):
            with self.subTest(timestamp=timestamp):
                solution = ShortestPathSolution(True, 5.0, timestamp, [], [])
                entry = Config.entry_from_shortest_path_solution(solution, elapsed, 600)
                self.assertEqual(entry.is_success, expected)
                self.assertEqual(entry.cost, 5.0 if expected else math.inf)
                self.assertLessEqual(entry.runtime, 600)

    def test_b_series_performance_schedule(self):
        jobs = STSuite.jobs(suite="performance_comparison",
                            performance_planners=("delta-set", "delta-state", "delta-pos"))
        self.assertEqual([job[1] for job in jobs], [spec.name for spec in Config.SEARCH_SPECS])
        self.assertTrue(all(job[0] == "performance_comparison" and job[-1] == "search" for job in jobs))
        self.assertEqual([job[3] for job in jobs], [30, 30, 30])
        self.assertEqual([job[2].epsilon for job in jobs], [1, 8, 8])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "stgcs_planner"
            executable.write_text("test executable")
            input_path = root / "input.txt"
            input_path.write_text("1 1 100 1 0 1 0 1 2 1 -1 2 2 0\n")
            backend = STGCSPlannerClient(executable, root)
            backend.bind_cache(input_path, "base-native-test", "simple2d")
            commands = [backend.command(input_path, spec, budget, method, 0, 10)
                        for _, _, spec, budget, method in jobs]
            self.assertEqual([command[5] for command in commands],
                             ["delta_set", "delta_state", "delta_pos"])
            self.assertEqual([float(command[6]) for command in commands], [1, 8, 8])

    def test_shared_schedule_has_expected_outcomes(self):
        jobs = STSuite.jobs()
        self.assertEqual(60 * sum(2 if job[-1] == "strrt" else 1 for job in jobs), 3300)
        self.assertEqual({job[0] for job in jobs}, set(STSuite.SUITES))
        self.assertNotIn("micp", {job[-1] for job in jobs})
        self.assertIn("micpg", {job[-1] for job in jobs})
        for name in ("micp", Config.MICP_PLANNER):
            self.assertNotIn(name, Config.planner_cli_choices())
            with self.assertRaises(ValueError):
                STSuite.jobs(suite="performance_comparison", performance_planners=(name,))
        self.assertEqual({job[3] for job in jobs if job[0] == "performance_comparison"}, {30})
        self.assertEqual({job[3] for job in jobs if job[0] == "heuristic_inflation"}, {30})
        self.assertEqual({job[3] for job in jobs if job[0] == "heuristic_ablation"}, {30})
        self.assertEqual({job[3] for job in jobs if job[0] == "dominance_ablation"}, {100})

    def test_joint_motion_performance_uses_independent_inflation(self):
        jobs = STSuite.jobs(suite="performance_comparison", performance_planners=("delta-pos",),
                            joint_motion=True)
        self.assertEqual(len(jobs), 2)
        reference, joint = jobs
        self.assertEqual(reference[2], Config.DELTA_POS_SPEC)
        self.assertEqual(joint[0], "performance_comparison")
        self.assertEqual(joint[1], "Search(h_mot+GUB+delta_pos,eps=2,joint,gub=h_mot)")
        self.assertEqual(joint[2].heuristic, "h_mot")
        self.assertEqual(joint[2].dominance_checks, reference[2].dominance_checks)
        self.assertEqual(reference[2].epsilon, 8)
        self.assertEqual(joint[2].epsilon, 2)
        self.assertEqual(joint[2].upper_bound_heuristic, reference[2].upper_bound_heuristic)
        self.assertEqual(joint[3:], (30, "search"))
        default_jobs = STSuite.jobs(suite="performance_comparison")
        self.assertEqual(STSuite.jobs(suite="performance_comparison", joint_motion=True),
                         [*default_jobs, joint])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "stgcs_planner"
            executable.write_text("test executable")
            path = root / "input.txt"
            path.write_text("1 1 100 1 0 1 0 1 2 1 -1 2 2 0\n")
            backend = STGCSPlannerClient(executable, root)
            command = backend.command(path, joint[2], joint[3], joint[4], 0, 10)
            self.assertEqual(command[4:9], ["h_mot", "delta_pos", "2.0", "30.0", "search"])
            self.assertEqual(command[11:], ["joint", "h_mot"])

    @unittest.skipUnless(os.environ.get("STGCS_NATIVE_EXECUTABLE"), "Set STGCS_NATIVE_EXECUTABLE to a Release runner")
    def test_native_joint_performance_with_motion_prepass(self):
        from pydrake.geometry.optimization import HPolyhedron
        from experiments.manifests.st_planning import STQuerySpec

        with TemporaryDirectory() as directory:
            root = Path(directory)
            backend = STGCSPlannerClient(Path(os.environ["STGCS_NATIVE_EXECUTABLE"]), root, root / "shared-caches")
            path = root / "input.txt"
            env = SimpleNamespace(_CSpace_hpoly=[
                HPolyhedron.MakeBox([-2, -1], [-0.5, 1]),
                HPolyhedron.MakeBox([-1, -1], [1, 1]),
                HPolyhedron.MakeBox([0.5, -1], [2, 1]),
            ])
            query = STQuerySpec([-1.5, 0], [1.5, 0], 0, False, 1)
            backend.write_input(path, env, query, 10, [])
            backend.bind_cache(path, "base-native-test", "simple2d")
            self.assertTrue(backend.cache(path)["complete"])
            joint = STSuite.jobs(suite="performance_comparison", joint_motion=True)[-1][2]
            for epsilon in (1, joint.epsilon):
                with self.subTest(epsilon=epsilon):
                    command = backend.command(path, replace(joint, epsilon=epsilon), 10, "search", 0, 10)
                    result = backend.run_process(command, 40, root / f"joint-{epsilon}.log")
                    self.assertTrue(result["success"])
                    self.assertAlmostEqual(result["cost"], 3, places=5)

    @unittest.skipUnless(os.environ.get("STGCS_NATIVE_EXECUTABLE"), "Set STGCS_NATIVE_EXECUTABLE to a Release runner")
    def test_native_micp_precision(self):
        # Default MICP tolerances produced a 4.94e-5 region violation and a
        # 6.18e-5 continuity gap on this captured maze instance.
        path = Path(__file__).parent / "fixtures/st_micp_maze_precision.txt"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            backend = STGCSPlannerClient(Path(os.environ["STGCS_NATIVE_EXECUTABLE"]), root, root / "shared-caches")
            spec = SearchPlannerSpec("h_zero", ("GUB",))
            result = backend.run_process(backend.command(path, spec, 10, "micp", 0, 10),
                                         40, root / "micp.log")
            self.assertTrue(result["success"], result)
            self.assertAlmostEqual(result["cost"], 29.35336046494714, places=6)
            for first, second in zip(result["segments"], result["segments"][1:]):
                self.assertLess(max(abs(a - b) for a, b in zip(first[1], second[0])), 1e-8)

    @unittest.skipUnless(os.environ.get("STGCS_NATIVE_EXECUTABLE"), "Set STGCS_NATIVE_EXECUTABLE to a Release runner")
    def test_native_micpg_iris_rounding(self):
        # Captured current IRIS instance: forced-primal relaxation and basis
        # recovery exhausted the budget without a valid rounded trajectory.
        path = Path(__file__).parent / "fixtures/st_micpg_iris_rounding.txt"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            backend = STGCSPlannerClient(Path(os.environ["STGCS_NATIVE_EXECUTABLE"]), root, root / "shared-caches")
            job = next(job for job in STSuite.jobs(suite="performance_comparison") if job[-1] == "micpg")
            result = backend.run_process(backend.command(path, job[2], 30, "micpg", 0, 1000),
                                         60, root / "micpg.log")
            # The native runner validates cell containment, speed, continuity,
            # endpoints, continuous obstacle clearance, and goal occupancy.
            self.assertTrue(result["success"], result)
            self.assertLessEqual(result["runtime"], 30)
            self.assertGreater(len(result["segments"]), 0)

    def test_inflation_uses_position_dominance_with_powers_of_two(self):
        jobs = STSuite.jobs(suite="heuristic_inflation", joint_motion=True)
        self.assertEqual(len(jobs), 30)
        self.assertEqual({job[1] for job in jobs}, set(InflationPlot.planner_names()))
        reference = set(STSuite.heuristic_specs(("GUB", "delta_pos"), include_zero=False))
        self.assertEqual({job[2] for job in jobs if job[2].epsilon == 1}, reference)
        self.assertEqual({(job[2].heuristic, job[2].key_evaluation, job[2].epsilon) for job in jobs},
                         {(spec.heuristic, spec.key_evaluation, e)
                          for spec in reference for e in (1, 2, 4, 8, 16, 32)})
        self.assertTrue(all(job[2].dominance_checks == ("GUB", "delta_pos")
                            and job[2].upper_bound_heuristic == "h_mot" for job in jobs))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "stgcs_planner"
            executable.write_text("test executable")
            input_path = root / "input.txt"
            input_path.write_text("1 1 100 1 0 1 0 1 2 1 -1 2 2 0\n")
            backend = STGCSPlannerClient(executable, root)
            backend.bind_cache(input_path, "test-inflation", "simple2d")
            for _, _, spec, budget, method in jobs:
                command = backend.command(input_path, spec, budget, method, 0, 10)
                self.assertEqual(command[5], "delta_pos")
                self.assertEqual(float(command[6]), spec.epsilon)
                self.assertEqual(float(command[7]), 30)
                self.assertEqual(command[11:], [spec.key_evaluation, "h_mot"])
        changed = STSuite.jobs(inflation_budget=45)
        budgets = {"performance_comparison": 30, "heuristic_inflation": 45,
                   "heuristic_ablation": 30, "dominance_ablation": 100}
        self.assertTrue(all(j[3] == budgets[j[0]] for j in changed))

    def test_factorial_ablation_uses_common_prepass(self):
        jobs = STSuite.jobs(suite="heuristic_ablation")
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({job[1] for job in jobs}), 12)
        self.assertTrue(all(job[2].epsilon == 1 and job[2].upper_bound_heuristic == "h_mot" for job in jobs))
        arms = [{replace(j[2], dominance_checks=()) for j in jobs if j[2].dominance_checks == checks}
                for checks in STSuite.HEURISTIC_DOMINANCE]
        self.assertEqual(arms[0], arms[1])
        self.assertEqual(len(arms[0]), 6)
        self.assertEqual(sum(s.key_evaluation == "joint" for s in arms[0]), 1)
        dc = STSuite.jobs(suite="dominance_ablation")
        self.assertTrue(all(j[2].upper_bound_heuristic == "h_mot" and j[2].epsilon == 1 for j in dc))
        shared = STSuite.jobs(suite="all", joint_motion=True)
        self.assertEqual(60 * sum(2 if j[-1] == "strrt" else 1 for j in shared), 3360)
        record = SimpleNamespace(instance_id="one")
        unique = {STSuite.measurement_key(record, job): job for job in shared}
        self.assertEqual(len(unique), 52)
        self.assertEqual(60 * sum(2 if j[-1] == "strrt" else 1 for j in unique.values()), 3180)

    def test_inflation_plot_pairs_matching_prepass_and_joint_baselines(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for _, name, spec, budget, _ in STSuite.jobs(suite="heuristic_inflation", joint_motion=True):
                baseline = 20 if spec.key_evaluation == "joint" else 10
                with (root / f"{name}.csv").open("w", newline="") as handle:
                    csv.writer(handle).writerow([name, True, baseline / spec.epsilon, 2, 3, 4,
                                                 "test-instance", budget, "maze", "maze", "base", 10, 20, 40])
            # Neither older dominance nor per-heuristic prepass results may enter the pairs.
            for stale in ("Search(h_mot+GUB+delta_set,gub=h_mot)",
                          "Search(h_mot+GUB+delta_state,gub=h_mot)", "Search(h_mot+GUB+delta_pos)"):
                with (root / f"{stale}.csv").open("w", newline="") as handle:
                    csv.writer(handle).writerow([stale, True, 999, 2, 3, 4,
                                                 "test-instance", 30, "maze", "maze", "base", 10, 20, 40])
            rows = InflationPlot.load_inflation_rows_by_planner(root, root)
            self.assertEqual(InflationPlot.select_budget(rows, None), 30)
            for heuristic in InflationPlot.HEURISTICS:
                values = InflationPlot.inflation_values_by_epsilon(rows, heuristic, 30, "runtime")
                for epsilon, reduction in values.items():
                    self.assertEqual(len(reduction), 1)
                    self.assertAlmostEqual(reduction[0], 100 * (1 - 1 / epsilon))


if __name__ == "__main__":
    unittest.main()
