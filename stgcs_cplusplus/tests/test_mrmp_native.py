import contextlib
from collections import Counter
from dataclasses import replace
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from pydrake.geometry.optimization import HPolyhedron

from experiments.base import BaseInstanceFactory, BaseManifestStore
from experiments.manifests.mrmp import MRMPBenchmarkRecord, load_manifest
from experiments.planners.mrmp import MRMPPerformanceComparison, PBSExpansionAblation, WindowedCoordinationAblation, SearchPlannerSpec
from experiments.mrmp import MRMPExperiment, MRMPResultEntry
from experiments.native import STGCSPlannerClient
from experiments.runners.mrmp import MRMPRunCLI
from experiments.runners.mrmp_comparison import MRMPPerformanceComparisonRunCLI
from experiments.runners.mrmp_windows import MRMPWindowedCoordinationRunCLI
from stgcs_native.types import MPQuery
from experiments.environment.space_time_graph import STGCS
from experiments.environment.interval import Interval
from baselines.priority_search import PriorityBasedSearch


class MRMPHorizonTests(unittest.TestCase):
    def test_manifest_horizon_overrides_serialized_base_in_each_domain(self):
        root = STGCSPlannerClient.ROOT
        records = load_manifest(root / "data/instances/mrmp/performance_comparison/manifest_n10.json")
        with TemporaryDirectory() as directory:
            executable = Path(directory) / "executable"
            executable.write_bytes(b"cache identity test")
            backend = STGCSPlannerClient(executable, Path(directory) / "native")
            for domain in ("grid2d", "maze", "iris-2d"):
                with self.subTest(domain=domain):
                    record = next(r for r in records if r.domain_key == domain)
                    self.assertEqual(record.tmax, 100.)
                    custom = replace(record, tmax=37.)
                    self.assertEqual(MRMPBenchmarkRecord.from_dict(custom.to_dict()), custom)
                    base = BaseManifestStore.record_by_id(
                        root / "data/stgcs_base" / domain / "manifest.json", record.base_instance_id)
                    stored = BaseInstanceFactory.load_cached_instance(base)
                    self.assertIsNotNone(stored)
                    self.assertEqual(base.env_params["tmax"], 100.)
                    self.assertEqual(stored.stgcs.tmax, 100.)
                    old_graph = stored.stgcs
                    with patch.object(BaseInstanceFactory, "from_record", return_value=stored):
                        instance, queries = MRMPExperiment.reconstruct_instance(custom, base)
                    self.assertEqual(instance.stgcs.tmax, 37.)
                    self.assertEqual(instance.stgcs.vlimit, old_graph.vlimit)
                    self.assertEqual(set(instance.stgcs.G.nodes), set(old_graph.G.nodes))
                    self.assertEqual(set(instance.stgcs.G.edges), set(old_graph.G.edges))
                    self.assertTrue(all(v.time_itvl.end == 37. for v in instance.stgcs._st_vertices.values()))
                    self.assertEqual([q.start.tolist() for q in queries], [q.start for q in record.queries])
                    path, _ = backend.prepare_instance(instance, queries, compute_cache=False)
                    self.assertEqual(float(path.read_text().split()[2]), 37.)
                    previous = Path(directory) / "previous.txt"
                    backend.write_input(previous, instance.env, queries[0], old_graph.tmax, [])
                    self.assertNotEqual(backend.base_input(path), backend.base_input(previous))
                    matching = Path(directory) / "st.txt"
                    backend.write_input(matching, instance.env, queries[0], 37., [])
                    backend.bind_cache(matching, record.base_instance_id, record.domain_key)
                    self.assertEqual(backend.cache_path(path), backend.cache_path(matching))


class PBSExpansionScheduleTests(unittest.TestCase):
    def test_current_benchmark_schedule_and_resume(self):
        with patch.object(sys, "argv", ["run_search", "--executable", "stgcs_planner"]):
            args = MRMPRunCLI.parse_args()
        self.assertEqual(args.budget, (60.,))
        self.assertEqual(args.output_root, Path("data/exp_res/mrmp/pbs_node_expansion_eps8_60s"))
        self.assertIsNone(args.limit)
        records = load_manifest(args.manifest)
        self.assertEqual(Counter(record.domain_key for record in records),
                         {"grid2d": 50, "maze": 50, "iris-2d": 50})
        self.assertEqual(len({record.instance_id for record in records}), 150)
        self.assertTrue(all(record.num_agents == len(record.queries) == 10 for record in records))
        self.assertEqual(WindowedCoordinationAblation.LOW_LEVEL_SPEC, PBSExpansionAblation.LOW_LEVEL_SPEC)
        backend = Mock()
        backend.run.return_value = MRMPResultEntry(
            is_success=False, runtime=60., sum_of_costs=math.inf, makespan=math.inf,
            num_completed_agents=0, pbs_calls=1, pbs_runtime=60., pbs_popped_nodes=0,
            pbs_generated_children=0, pbs_update_calls=0, mp_calls=0, mp_runtime=0.,
            ecd_calls=0, ecd_runtime=0., cc_calls=0, cc_runtime=0.)
        with TemporaryDirectory() as directory:
            args.output_root = Path(directory)
            with (patch.object(MRMPRunCLI, "parse_args", return_value=args),
                  patch.object(MRMPRunCLI, "progress_disabled", return_value=True),
                  patch.object(STGCSPlannerClient, "from_args", return_value=backend),
                  patch.object(MRMPExperiment, "reconstruct_instance",
                               return_value=(SimpleNamespace(name="schedule-check"), []))):
                MRMPRunCLI.main()
                self.assertEqual(backend.run.call_count, 600)
                self.assertEqual(Counter(call.kwargs["child_rule"] for call in backend.run.call_args_list),
                                 {"lazy": 150, "soc": 150, "makespan": 150, "num_conflicts": 150})
                for call in backend.run.call_args_list:
                    self.assertEqual(call.args[2], SearchPlannerSpec("h_max", ("GUB", "delta_pos"), epsilon=8.))
                    self.assertEqual(call.kwargs["budget"], 60.)
                paths = list(args.output_root.glob("*/results/*.csv"))
                self.assertEqual(len(paths), 12)
                for path in paths:
                    saved = MRMPExperiment.load_result_rows(path)
                    self.assertEqual(len(saved), 50)
                    self.assertTrue(all(row["budget"] == 60. and row["num_agents"] == 10 for row in saved))
                    self.assertIn("eps=8", path.name)
                backend.run.reset_mock()
                MRMPRunCLI.main()
                backend.run.assert_not_called()


class MRMPPerformanceScheduleTests(unittest.TestCase):
    def test_kcbs_is_explicit_only_in_performance_schedule(self):
        runner = MRMPPerformanceComparisonRunCLI
        expected = ('pbs', 'pp-bfs', 'fixed-pp-st-rrt-star-first',
                    'fixed-pp-st-rrt-star-final', 'pbs-zeta-sipp', 'cb-gcs')
        with patch.object(sys, 'argv', ['run_performance_comparison', '--executable', 'unused', '--budget', '120']):
            self.assertEqual(runner.parse_args().planners, expected)
        self.assertIn('kcbs', runner.planner_keys(('all',)))
        self.assertEqual(runner.planner_keys(('kcbs',)), ('kcbs',))
        with patch.object(sys, 'argv', ['run_performance_comparison', '--executable', 'unused',
                                       '--budget', '120', '--planner', 'kcbs']):
            self.assertEqual(runner.parse_args().planners, ('kcbs',))
        with (patch.object(MRMPExperiment, 'reconstruct_instance',
                           return_value=(SimpleNamespace(name='kcbs-dispatch-check'), [])),
              patch.object(MRMPExperiment, 'run_ompl_kcbs') as solve,
              patch.object(MRMPExperiment, 'append_result_row') as save):
            entry = runner.run_pending_planner(Mock(), Mock(), Path('unused.csv'), 120., 'kcbs', Mock())
            solve.assert_called_once()
            self.assertEqual(solve.call_args.kwargs['budget'], 120.)
            self.assertIs(entry, solve.return_value)
            self.assertIs(save.call_args.args[4], entry)

    def test_native_dispatch_and_saved_settings(self):
        benchmark = MRMPPerformanceComparison
        backend = Mock()
        backend.run.return_value = Mock()
        instance = SimpleNamespace(name="dispatch-check")
        with (patch.object(MRMPExperiment, "reconstruct_instance", return_value=(instance, [])),
              patch.object(MRMPExperiment, "append_result_row") as save):
            for key, method, alpha in (("pbs", "pbs", 0.), ("pp-bfs", "pp", 0.),
                                       ("wpbs", "pbs", 2.), ("wpp", "pp", 2.)):
                with self.subTest(key=key):
                    MRMPPerformanceComparisonRunCLI.run_pending_planner(
                        Mock(), Mock(), Path("unused.csv"), 60., key, backend)
                    call = backend.run.call_args
                    self.assertEqual(call.args[2], SearchPlannerSpec("h_max", ("GUB", "delta_pos"), epsilon=8.))
                    self.assertEqual(call.kwargs["budget"], 60.)
                    self.assertEqual(call.kwargs["method"], method)
                    self.assertEqual(call.kwargs["window_span_factor"], alpha)
                    self.assertEqual(call.kwargs["dynamic_window_adjustment"], alpha > 0.)
                    self.assertEqual(call.kwargs["execution_horizon_factor"], 1.)
                    self.assertEqual(save.call_args.args[3], 60.)
                    self.assertIn("eps=8", benchmark.planner_name(key))
                    if alpha:
                        self.assertIn("alpha=2,beta=1", benchmark.planner_name(key))


@unittest.skipUnless(os.environ.get("STGCS_NATIVE_EXECUTABLE"), "Set STGCS_NATIVE_EXECUTABLE to a Release runner")
class NativeMRMPTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.backend = STGCSPlannerClient(Path(os.environ["STGCS_NATIVE_EXECUTABLE"]), Path(self.directory.name),
                                        Path(self.directory.name) / "shared-caches")
        self.instance = SimpleNamespace(
            base_record=SimpleNamespace(instance_id="base-native-test", domain_key="simple2d"),
            env=SimpleNamespace(dim=2, robot_radius=0.15,
                                _CSpace_hpoly=[HPolyhedron.MakeBox([-2, -2], [2, 2])]),
            stgcs=SimpleNamespace(tmax=8, vlimit=1.0))
        self.queries = [MPQuery(np.array([-1., 0.]), np.array([1., 0.]), 0., True, 1.),
                        MPQuery(np.array([0., -1.]), np.array([0., 1.]), 0., True, 1.)]

    def test_motion_triplet_cache_and_native_windowed_plan(self):
        self.instance.env._CSpace_hpoly = [
            HPolyhedron.MakeBox([left, -2], [right, 2])
            for left, right in ((-2, -.5), (-1, 1), (.5, 2))]
        queries = [MPQuery(np.array([-1.5, y]), np.array([1.5, y]), 0., True, 1.)
                   for y in (-.8, .8)]
        path, _ = self.backend.prepare_instance(self.instance, queries)
        full = self.backend.cache_path(path)
        original = full.read_bytes()
        spec = SearchPlannerSpec("h_mot_tri", ("GUB", "delta_pos"), epsilon=8.)
        paths, entry = self.backend.run_with_solutions(
            self.instance, queries, spec, 10., "num_conflicts", 2., 1., True)
        self.assertTrue(entry.is_success)
        self.assertEqual(len(paths), 2)
        for trajectory, query in zip(paths, queries):
            np.testing.assert_allclose(trajectory.x0[:-1], query.start, atol=1e-6)
            np.testing.assert_allclose(trajectory.xT[:-1], query.goal, atol=1e-6)
        partial = self.backend.cache_path(path, cost_table=False)
        tables = self.backend.read_cache_tables(partial)
        self.assertFalse(tables["has_table"])
        self.assertTrue(tables["triplets"])
        self.assertFalse(tables["table"])
        self.assertFalse(tables["incoming"])
        self.assertEqual(full.read_bytes(), original)
        reused = partial.read_bytes()
        self.backend.cache(path, cost_table=False)
        self.assertEqual(partial.read_bytes(), reused)
        receipt, = (self.backend.work_dir / "logs").glob("*.json")
        command = json.loads(receipt.read_text())["command"]
        self.assertEqual(Path(command[3]), partial)
        command[4] = "h_max"
        rejected = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("CostTable requires", rejected.stderr)

    def test_four_child_rules_and_native_status(self):
        for rule in PBSExpansionAblation.RULES:
            with self.subTest(rule=rule.key):
                result = self.backend.run(self.instance, self.queries, PBSExpansionAblation.LOW_LEVEL_SPEC,
                                          10., rule.key)
                self.assertTrue(result.is_success)
                self.assertEqual(result.num_completed_agents, 2)
                self.assertGreaterEqual(result.pbs_popped_nodes, 2)
                self.assertGreaterEqual(result.pbs_generated_children, 2)
                self.assertGreaterEqual(result.pbs_generated_children, result.pbs_update_calls)
                self.assertGreater(result.cc_calls, 0)
                self.assertGreater(result.mp_gub_runtime, 0)
                self.assertGreater(result.mp_convex_restriction_runtime, 0)
                self.assertGreaterEqual(result.mp_convex_restriction_runtime,
                                        result.mp_dominance_convex_restriction_runtime)

    def test_windowed_profile_roundtrip_and_disjoint_components(self):
        from experiments.reports.mrmp_runtime import MRMPWPBSRuntimeBreakdownReport as Report
        for method in ('pbs', 'pp'):
            with self.subTest(method=method):
                entry = self.backend.run(self.instance, self.queries, MRMPPerformanceComparison.LOW_LEVEL_SPEC,
                    10., 'num_conflicts', window_span_factor=0.6, execution_horizon_factor=1.,
                    dynamic_window_adjustment=True, method=method)
                self.assertTrue(entry.is_success)
                self.assertGreater(entry.mp_convex_restriction_calls, 0)
                self.assertGreater(entry.mp_convex_restriction_runtime, 0)
                self.assertGreater(entry.mp_dominance_convex_restriction_calls, 0)
                self.assertEqual(entry.mp_gcs_calls, entry.mp_calls)
                self.assertEqual(entry.mp_gub_calls, entry.mp_calls)
                self.assertEqual(entry.mp_search_calls, entry.mp_calls)
                inclusive = entry.mp_gcs_runtime + entry.mp_gub_runtime + entry.mp_search_runtime
                self.assertLessEqual(inclusive, entry.mp_runtime + 1e-6)
                nested = (entry.mp_convex_restriction_runtime + entry.mp_dominance_check_runtime
                          - entry.mp_dominance_convex_restriction_runtime)
                self.assertLessEqual(nested, entry.mp_gub_runtime + entry.mp_search_runtime + 1e-6)
                path = Path(self.directory.name) / f'{method}.csv'
                record = SimpleNamespace(instance_id='profile-test', num_agents=2,
                                         stgcs_num_vertices=1, stgcs_num_edges=0)
                MRMPExperiment.append_result_row(path, 'profile-test', record, 10., entry)
                row = MRMPExperiment.load_result_rows(path)[0]
                self.assertTrue(row['has_disjoint_breakdown_profile'])
                components = Report.row_components(row)
                self.assertEqual(components['convex_restriction'], entry.mp_convex_restriction_runtime)
                self.assertAlmostEqual(sum(components.values()), entry.runtime)

    def test_full_and_windowed_prioritized_planning(self):
        for alpha in (0., 5.):
            with self.subTest(alpha=alpha):
                result = self.backend.run(
                    self.instance, self.queries, MRMPPerformanceComparison.LOW_LEVEL_SPEC,
                    10., "num_conflicts", window_span_factor=alpha,
                    dynamic_window_adjustment=alpha > 0., method="pp")
                self.assertTrue(result.is_success)
                self.assertEqual(result.num_completed_agents, 2)
                self.assertEqual(result.pbs_generated_children, 0)
                self.assertGreater(result.ecd_calls, 0)
                self.assertGreaterEqual(result.pbs_calls, 1)

    def test_native_crossing_paths_are_continuously_collision_free(self):
        from stgcs_native.trajopt import GlobalTrajectoryOptimizer
        spec = SearchPlannerSpec("h_mot", ("GUB", "delta_pos"))
        for rule in PBSExpansionAblation.RULES:
            with self.subTest(rule=rule.key):
                paths, native = self.backend.run_with_solutions(self.instance, self.queries, spec, 10., rule.key)
                self.assertTrue(native.is_success)
                self.assertTrue(GlobalTrajectoryOptimizer.pairwise_collision_free(paths, .15, [True]*2))
                self.assertAlmostEqual(native.sum_of_costs, sum(p.duration for p in paths), places=4)
                self.assertAlmostEqual(native.makespan, max(p.duration for p in paths), places=4)
                for i, path in enumerate(paths):
                    np.testing.assert_allclose(path.xT[:-1], self.queries[i].goal, atol=1e-6)


    def test_dynamic_window_cli_and_full_horizon_native_dispatch(self):
        with patch.object(sys, "argv", ["run_windowed_coordination", "--executable",
                                        str(self.backend.executable), "--budget", "10"]):
            args = MRMPWindowedCoordinationRunCLI.parse_args()
        specs = MRMPWindowedCoordinationRunCLI.specs_from_args(args)
        self.assertEqual([spec.window_span_factor for spec in specs], [2., 4., 8., 16., math.inf])
        self.assertTrue(all(spec.dynamic_window_adjustment and spec.execution_horizon_factor == 1. for spec in specs))
        self.assertEqual(WindowedCoordinationAblation.planner_name(specs[-1]),
                         PBSExpansionAblation.planner_name(PBSExpansionAblation.default_rule()))
        for spec in specs:
            with self.subTest(spec=spec.key):
                result = self.backend.run(
                    self.instance, self.queries, WindowedCoordinationAblation.LOW_LEVEL_SPEC,
                    10., "num_conflicts", spec.window_span_factor, spec.execution_horizon_factor,
                    spec.dynamic_window_adjustment)
                self.assertTrue(result.is_success)
                self.assertEqual(result.num_completed_agents, 2)
                self.assertGreaterEqual(result.pbs_calls, 1)
        commands = [json.loads(path.read_text())["command"]
                    for path in (self.backend.work_dir/"logs").glob("*.json")]
        self.assertEqual({tuple(command[10:13]) for command in commands},
                         {(str(alpha), "1.0", "1") for alpha in (2., 4., 8., 16.)} | {("0.0", "1.0", "0")})

    def test_timeout_and_late_success_are_failures(self):
        result = self.backend.run(self.instance, self.queries, PBSExpansionAblation.LOW_LEVEL_SPEC,
                                  0., "num_conflicts")
        self.assertFalse(result.is_success)
        self.assertEqual(result.mp_calls, 0)
        self.assertTrue(math.isinf(result.sum_of_costs))
        with patch.object(self.backend, "run_process", return_value={
                "success": True, "runtime": 2., "trajectories": [], "st_vertices": 1, "st_edges": 0}):
            late = self.backend.run(self.instance, self.queries, PBSExpansionAblation.LOW_LEVEL_SPEC,
                                    1., "num_conflicts")
        self.assertFalse(late.is_success)
        self.assertEqual(late.runtime, 2.)

    def test_cache_reuse_and_velocity_validation(self):
        first, first_queries = self.backend.prepare_instance(self.instance, self.queries)
        second, second_queries = self.backend.prepare_instance(self.instance, self.queries[:1])
        self.assertEqual(self.backend.cache_path(first), self.backend.cache_path(second))
        self.assertNotEqual(first_queries, second_queries)
        self.assertEqual(len(list(self.backend.cache_root.rglob("*.bin"))), 1)
        self.queries[0].vlimit = 2.
        with self.assertRaisesRegex(ValueError, "velocity"):
            self.backend.prepare_instance(self.instance, self.queries)

    def test_st_and_mrmp_load_same_cache_with_different_reservations(self):
        st = STGCSPlannerClient(self.backend.executable, self.backend.work_dir / "st", self.backend.cache_root)
        path = st.work_dir / "input.txt"
        obstacles = [dict(radius=.15, segments=[dict(start=[1, 1], goal=[1, 1], t_start=0, t_end=8)])]
        st.write_input(path, self.instance.env, self.queries[0], 8, obstacles)
        st.bind_cache(path, self.instance.base_record.instance_id, self.instance.base_record.domain_key)
        st_info = st.cache(path)
        self.assertEqual(st.cache_path(path).read_bytes()[:8], STGCSPlannerClient.CACHE_MAGIC)
        self.assertTrue(st.read_cache_tables(st.cache_path(path))["table"])
        with patch.object(self.backend, "run_process", wraps=self.backend.run_process) as run:
            base_path, _ = self.backend.prepare_instance(self.instance, self.queries)
            self.assertFalse(any(call.args[0][1] == "cache" for call in run.call_args_list))
        self.assertEqual(st.cache_path(path), self.backend.cache_path(base_path))
        self.assertGreater(st_info["st_vertices"], self.backend.inspect(base_path)["st_vertices"])
        result = self.backend.run(self.instance, self.queries, PBSExpansionAblation.LOW_LEVEL_SPEC,
                                  10, "num_conflicts")
        self.assertTrue(result.is_success)

    def test_shared_single_robot_command(self):
        path = self.backend.work_dir / "single.txt"
        self.backend.write_input(path, self.instance.env, self.queries[0], 8., [])
        self.backend.bind_cache(path, self.instance.base_record.instance_id, self.instance.base_record.domain_key)
        command = self.backend.command(path, SearchPlannerSpec("h_mot", ("GUB", "delta_pos")),
                                       2., "search", 0, 100)
        result = self.backend.run_process(command, 10., path.with_suffix(".log"))
        self.assertTrue(result['success'])
        self.assertEqual(result['status'], 0)
        self.assertAlmostEqual(result['cost'], 2., places=4)
        command = self.backend.command(path, SearchPlannerSpec("h_mot", ("GUB", "delta_pos")),
                                       0., "search", 0, 100)
        result = self.backend.run_process(command, 10., path.with_suffix(".log"))
        self.assertFalse(result['success'])
        self.assertEqual(result['status'], 3)
        self.assertIsNone(result['cost'])

    def test_existing_results_need_no_backend_version_marker(self):
        output = self.backend.work_dir / "results"
        output.mkdir()
        saved = output / "saved.csv"
        saved.write_text("saved results\n")
        args = SimpleNamespace(output_root=output, work_dir=None, executable=self.backend.executable)
        STGCSPlannerClient.from_args(args)
        self.assertEqual(saved.read_text(), "saved results\n")
        self.assertFalse((output / "native_backend.json").exists())


if __name__ == "__main__":
    unittest.main()
