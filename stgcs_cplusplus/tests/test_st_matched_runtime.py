from dataclasses import replace
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from baselines.common import ShortestPathSolution
from experiments.manifests.st_planning import STPlanningManifestStore
from experiments.planners.st_planning import STPerformanceComparisonConfig as Config
from experiments.reports.st_comparison import STPerformanceComparisonReport as Report
from experiments.runners.st_baselines import BaselineWorker
from experiments.runners.st import STSuite
from experiments.results import STResultStore as Store


class MatchedRuntimeTests(unittest.TestCase):
    class FixtureSuite(STSuite):
        def prepare(self, record):
            path = self.work / 'inputs' / f'{record.instance_id}.txt'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixed input')
            self.backend.bind_cache(path, record.base_instance_id, record.source_domain)
            return replace(record, stgcs_num_vertices=20, stgcs_num_edges=40)

        @staticmethod
        def fake_process(command, timeout, log):
            if command[1] == 'plan':
                instance = Path(command[2]).stem
                runtime = 31 if 'maze' in instance else 0 if 'iris' in instance else 0.25
                return dict(success=runtime < 30, status=0, runtime=runtime, cost=10,
                            expanded=2, generated=3)
            instance = Path(command[3]).stem
            root = next(p for p in log.parents if p.name == 'work').parent / 'results' / 'performance_comparison'
            rows = Store.load_result_rows(root / f'{Config.DELTA_POS_PLANNER}.csv')
            source = next(row for row in rows if row['instance_id'] == instance)
            budget = float(command[-1])
            assert budget == min(30, source['runtime'])
            assert timeout == budget + 30
            return {Config.ST_RRT_MATCHED_PLANNER: dict(
                is_success=budget > 0, runtime=budget, cost=9 if budget else math.inf,
                num_expanded_nodes=0, num_generated_nodes=0)}

    @staticmethod
    def args(root):
        executable = root / 'exe'
        executable.write_text('fixed executable')
        return SimpleNamespace(work_dir=root / 'work', executable=executable,
            manifest=Path('data/instances/st_planning/manifest.json'), base_root=root,
            output_root=root / 'results', workers=3, limit=3, suite='performance_comparison',
            planner=['strrt-matched'], joint_motion=False, performance_budget=30,
            ablation_budget=100, cache_timeout=10000, skip_scaling=True,
            first_planner=Config.ST_RRT_MATCHED_PLANNER)

    def test_parallel_matching_and_receipt_resume(self):
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            (args.work_dir / 'logs').mkdir(parents=True)
            with patch('experiments.runners.st.STGCSPlannerClient.run_process',
                       side_effect=self.FixtureSuite.fake_process) as run, \
                    patch('sys.stdout', new=io.StringIO()):
                suite = self.FixtureSuite(args)
                suite.execute()
                self.assertEqual((run.call_count, suite.completed), (6, 6))
                self.assertEqual(suite.execution_jobs[0][1], Config.DELTA_POS_PLANNER)
                path = args.output_root / 'performance_comparison' / f'{Config.ST_RRT_MATCHED_PLANNER}.csv'
                rows = Store.load_result_rows(path)
                self.assertEqual({row['runtime'] for row in rows}, {0, 0.25, 30})
                self.assertEqual({row['budget'] for row in rows}, {30})
                for receipt_path in (args.work_dir / 'logs').rglob('*.json'):
                    receipt = json.loads(receipt_path.read_text())
                    if receipt['planner'] == Config.ST_RRT_MATCHED_PLANNER:
                        self.assertEqual(receipt['runtime_budget'], min(30, receipt['budget_source']['runtime']))
                        self.assertEqual(receipt['instance'], receipt['budget_source']['instance_id'])
                        self.assertEqual(receipt['budget_source']['name'], Config.DELTA_POS_PLANNER)
                path.unlink()
                resumed = self.FixtureSuite(args)
                resumed.execute()
                self.assertEqual((run.call_count, resumed.completed), (6, 6))
                self.assertEqual({row['instance_id']: row for row in Store.load_result_rows(path)},
                                 {row['instance_id']: row for row in rows})

    def test_missing_or_invalid_b3_runtime_is_rejected(self):
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            suite = self.FixtureSuite(args)
            record = suite.records[0]
            job = next(job for job in suite.jobs if job[-1] == 'strrt_matched')
            entry = SimpleNamespace(is_success=True, runtime=1, cost=10,
                                    num_expanded_nodes=0, num_generated_nodes=0)
            b2_path = args.output_root / 'performance_comparison' / f'{Config.DELTA_STATE_PLANNER}.csv'
            Store.write_result_rows(b2_path, Config.DELTA_STATE_PLANNER, [(record, entry)], 30)
            with self.assertRaisesRegex(ValueError, 'B3 runtime'):
                suite.run_job(record, job)
            path = args.output_root / 'performance_comparison' / f'{Config.DELTA_POS_PLANNER}.csv'
            for runtime in (-1, math.nan, math.inf):
                with self.subTest(runtime=runtime):
                    entry = SimpleNamespace(is_success=False, runtime=runtime, cost=math.inf,
                                            num_expanded_nodes=0, num_generated_nodes=0)
                    Store.write_result_rows(path, Config.DELTA_POS_PLANNER, [(record, entry)], 30)
                    with self.assertRaisesRegex(ValueError, 'B3 runtime'):
                        suite.run_job(record, job)

    def test_worker_uses_final_snapshot_and_preserves_query_settings(self):
        record = STPlanningManifestStore.load_manifest('data/instances/st_planning/manifest.json')[0]
        first = ShortestPathSolution(True, 12, 0.1, [], [])
        final = ShortestPathSolution(True, 9, 0.8, [], [])
        with patch('experiments.runners.st_baselines.STGCSPlannerClient.base_instance', return_value=MagicMock()), \
                patch('experiments.runners.st_baselines.OfficialOMPLSTRRTStar') as planner:
            planner.return_value.solve_with_snapshots.return_value = SimpleNamespace(
                first_solution=first, final_solution=final)
            entries = BaselineWorker.run(record, Path('.'), 'strrt_matched', 1.25)
            self.assertEqual(set(entries), {Config.ST_RRT_MATCHED_PLANNER})
            self.assertEqual(entries[Config.ST_RRT_MATCHED_PLANNER]['cost'], 9)
            options = planner.return_value.solve_with_snapshots.call_args.kwargs['options']
            self.assertEqual(options.max_runtime_in_secs, 1.25)
            self.assertEqual(options.time_upper_bound, record.tmax)
            self.assertEqual(options.stay_at_goal, record.query.is_stay)
            self.assertFalse(options.return_first_valid)
            self.assertEqual(planner.call_args.args[1:], (Config.seed_for_record(record, 0), record.query.vlimit))
            planner.reset_mock()
            entries = BaselineWorker.run(record, Path('.'), 'strrt_matched', 0)
            self.assertFalse(entries[Config.ST_RRT_MATCHED_PLANNER]['is_success'])
            self.assertEqual(entries[Config.ST_RRT_MATCHED_PLANNER]['runtime'], 0)
            planner.assert_not_called()

    def test_plot_keeps_b1_and_matched_as_separate_methods(self):
        self.assertEqual(Report.METHOD_STYLES[Config.DELTA_SET_PLANNER]['abbrev'], 'B1')
        self.assertEqual(Config.DELTA_SET_SPEC.epsilon, 1)
        self.assertEqual(Report.METHOD_STYLES[Config.DELTA_STATE_PLANNER]['abbrev'], 'B2')
        self.assertEqual(Report.JOINT_SPEC.dominance_checks, ('GUB', 'delta_pos'))
        self.assertEqual(Config.DELTA_STATE_SPEC.epsilon, 8)
        self.assertEqual(Config.DELTA_POS_SPEC.epsilon, 8)
        self.assertEqual(Report.JOINT_SPEC, Config.JOINT_SPEC)
        self.assertEqual(Report.JOINT_SPEC.epsilon, 2)
        self.assertEqual(Report.JOINT_SPEC.heuristic, 'h_mot')
        self.assertEqual(Report.JOINT_SPEC.key_evaluation, 'joint')
        self.assertNotIn('Search(h_max+GUB+delta_set,eps=5,gub=h_mot)', Report.PLANNERS)
        self.assertEqual(Report.DEFAULT_BUDGET, 30)
        self.assertIn(Config.ST_RRT_MATCHED_PLANNER, Report.PLANNERS)
        self.assertNotIn(Config.ST_RRT_MATCHED_PLANNER, Report.ST_RRT_PLANNERS)


if __name__ == '__main__':
    unittest.main()
