import csv
from dataclasses import replace
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.runners.st import STSuite
from experiments.native import STGCSPlannerClient
from experiments.reports.st_reference_costs import STReferenceCosts
from experiments.results import STResultStore as Store
from experiments.reports.st_heuristic_factorial import STHeuristicFactorialReport as Fig10
from experiments.reports.heuristic_scaling import HeuristicInflationAndScalingPlot as Inflation
from experiments.reports.st_inflation import HeuristicInflationTable as Table


class SharedSTProtocolTests(unittest.TestCase):
    @staticmethod
    def reference_row(instance='a', cost=10, budget=100, runtime=2, success=True):
        return dict(instance_id=instance, cost=cost, budget=budget, runtime=runtime, is_success=success)

    def test_references_use_successful_native_rows_at_the_selected_budget(self):
        rows = {
            'Search(h_max)': [self.reference_row(), self.reference_row(cost=1, budget=30),
                             self.reference_row(cost=1, runtime=101),
                             self.reference_row(cost=1, success=False),
                             self.reference_row('failed', success=False)],
            'MICP(g)': [self.reference_row(cost=9)],
            'OMPL ST-RRT*': [self.reference_row(cost=1), self.reference_row('external')],
        }
        for cost in (0, -1, float('inf'), float('nan')):
            rows['Search(h_max)'].append(self.reference_row('invalid', cost=cost))
        report = STReferenceCosts.from_rows(rows, 100)
        self.assertEqual(report['references'], {'a': dict(cost=9, kind='best_known')})
        self.assertEqual(report['instance_ids'], ['a', 'failed', 'invalid'])
        self.assertEqual(STReferenceCosts.quality_reference(report)[0], 'best_known')
        self.assertAlmostEqual(STReferenceCosts.gaps([self.reference_row()], report)[0], 1/9)
        with self.assertRaisesRegex(ValueError, 'beats'):
            STReferenceCosts.gaps([self.reference_row(cost=8)], report)

    def test_success_curve_keeps_failures_in_denominator(self):
        rows = [dict(runtime=2, cost=1, is_success=True), dict(runtime=100, cost=float('inf'), is_success=False),
                dict(runtime=101, cost=1, is_success=True)]
        x, y = Fig10.success_curve(rows, 100)
        self.assertEqual(x, [0, 2, 100])
        self.assertEqual(y, [0, 1/3, 1/3])
        self.assertEqual(Fig10.success_curve(rows[1:2], 100)[1], [0, 0])

    def test_partial_performance_uses_completed_trials_and_preserves_default_pairing(self):
        from experiments.reports.st_comparison import STPerformanceComparisonReport as Fig14
        rows = {'a': [dict(instance_id='old', budget=30, runtime=1, is_success=True),
                      dict(instance_id='new', budget=30, runtime=30, is_success=False)],
                'b': [dict(instance_id='old', budget=30, runtime=1, is_success=True)]}
        with patch.object(Fig14, 'PARTIAL', False):
            self.assertEqual(Fig14.success_rate_points(rows, 30, 'a')[1][-1], 1)
        with patch.object(Fig14, 'PARTIAL', True):
            self.assertEqual(len(Fig14.planner_budget_rows(rows, 30, 'a')), 2)
            self.assertEqual(len(Fig14.planner_budget_rows(rows, 30, 'b')), 1)
            self.assertEqual(Fig14.success_rate_points(rows, 30, 'a')[1][-1], .5)
            self.assertEqual(Fig14.success_rate_points(rows, 30, 'b')[1][-1], 1)

    @classmethod
    def factorial_rows(cls):
        return {name: [dict(instance_id=f'{g}-{i}', source_domain=g, group=g, budget=100,
                           is_success=spec.heuristic != 'h_zero' and (i == 0 or 'delta_set' in spec.dominance_checks),
                           runtime=5, cost=10, num_expanded_nodes=20)
                       for g in ('maze', 'grid2d', 'iris-2d') for i in range(2)]
                for name, spec in zip(Fig10.FIGURE_PLANNERS, Fig10.SPECS)}

    def test_factorial_summary_handles_zero_common_successes_and_pairs(self):
        rows = self.factorial_rows()
        summary = Fig10.summary(rows, 100)
        group = summary['groups']['maze']
        self.assertEqual(group['total'], 2)
        self.assertEqual(group['solved'][Fig10.SPECS[0].name], 0)
        self.assertEqual(group['solved'][Fig10.SPECS[7].name], 2)
        self.assertEqual(group['expansion_pairs'][0]['paired_n'], 0)
        self.assertIsNone(group['expansion_pairs'][0]['median_expansion_ratio'])
        rows[Fig10.FIGURE_PLANNERS[0]].pop()
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            Fig10.summary(rows, 100)

    def test_delta_set_only_schedule_and_report(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / 'exe'
            executable.write_text('fixed executable')
            args = SimpleNamespace(work_dir=root / 'work', executable=executable,
                manifest=Path('data/instances/st_planning/manifest.json'), workers=1,
                limit=3, suite='heuristic_ablation', planner=None, joint_motion=False,
                performance_budget=100, ablation_budget=600, heuristic_dominance='delta_set',
                first_planner='Search(h_max+GUB+delta_set,gub=h_mot)')
            suite = STSuite(args)
            self.assertEqual((len(suite.jobs), suite.expected), (6, 18))
            self.assertEqual(suite.execution_jobs[0][1], args.first_planner)
            self.assertTrue(all(job[3] == 30 and 'delta_set' in job[2].dominance_checks
                                for job in suite.jobs))
            self.assertEqual(sum(job[2].key_evaluation == 'joint' for job in suite.jobs), 1)
            args.heuristic_budget = 600
            self.assertTrue(all(job[3] == 600 for job in STSuite(args).jobs))
        rows = {name: values for name, values in self.factorial_rows().items() if 'delta_set' in name}
        report = Fig10.summary(rows, 100, dominance='delta_set')
        for group in report['groups'].values():
            self.assertEqual(len(group['solved']), 6)
            self.assertEqual(len(group['expansion_pairs']), 15)
            self.assertEqual(group['total'], 2)
        rows[next(iter(rows))].pop()
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            Fig10.summary(rows, 100, dominance='delta_set')

    def test_table_uses_its_planner_family_and_checks_manifest_coverage(self):
        records = [SimpleNamespace(instance_id=i) for i in ('a', 'b', 'c')]
        with TemporaryDirectory() as directory, patch(
                'experiments.reports.st_inflation.STPlanningManifestStore.load_manifest',
                return_value=records) as manifest:
            results = Path(directory)
            for heuristic in Inflation.HEURISTICS:
                for epsilon in (1, *Inflation.EPSILONS):
                    name = Inflation.planner_name(heuristic, epsilon)
                    with (results / f'{name}.csv').open('w', newline='') as stream:
                        writer = csv.writer(stream)
                        for i in ('a', 'b', 'c'):
                            success = i == 'a' or epsilon != 1
                            cost = 10 if i == 'a' or (i == 'b' and epsilon == 2) else 20
                            writer.writerow([name, success, 10 if epsilon == 1 else {'a': 5, 'b': 15, 'c': 25}[i],
                                             cost if success else 'inf',
                                             1, 2, i, 100, 'maze', 'maze', 'base', 10, 20, 40])
            # A different dominance family must not lower this table's references.
            (results / 'Search(h_max+GUB+delta_state).csv').write_text(
                'Search(h_max+GUB+delta_state),True,1,1,1,1,a,100,maze,maze,base,10,20,40\n')
            table = Table.build(results, 100)
            self.assertEqual(len(table['rows']), 30)
            self.assertEqual(table['reference_kind'], 'best_known')
            self.assertEqual(table['reference_coverage'], 3)
            for row in table['rows']:
                if row['epsilon'] == 1:
                    self.assertEqual((row['solved'], row['runtime_n'], row['quality_n']), (1, 1, 1))
                    self.assertEqual(row['runtime_seconds'], dict(mean=10, p95=10, maximum=10))
                    self.assertEqual(row['cost_gap_percent'], dict(mean=0, p95=0, maximum=0))
                    continue
                self.assertEqual((row['solved'], row['runtime_n'], row['quality_n']), (3, 3, 3))
                self.assertEqual(row['runtime_seconds'], dict(mean=15, p95=24, maximum=25))
                self.assertAlmostEqual(row['cost_gap_percent']['mean'], 0 if row['epsilon'] == 2 else 100/3)
                self.assertAlmostEqual(row['cost_gap_percent']['p95'], 0 if row['epsilon'] == 2 else 90)
            self.assertIn('Best-known cost gap', Table.latex(table))
            self.assertIn(r'\delta_\mathrm{pos}', Table.latex(table))
            self.assertNotIn(r'\delta_\mathrm{state}', Table.latex(table))
            self.assertIn('$100$s per run', Table.latex(table))
            self.assertIn('Runtime (s)', Table.latex(table))
            manifest.return_value = [*records, SimpleNamespace(instance_id='missing')]
            with self.assertRaisesRegex(ValueError, 'mismatched'):
                Table.build(results, 100)
            manifest.return_value = records
            path = results / f'{Inflation.planner_name("h_max", 2)}.csv'
            path.write_text('\n'.join(path.read_text().splitlines()[1:]) + '\n')
            with self.assertRaisesRegex(ValueError, 'mismatched'):
                Table.build(results, 100)

    def test_runner_reuses_one_measurement_and_resumes_missing_copy(self):
        class FixtureSuite(STSuite):
            @classmethod
            def jobs(cls, *args):
                spec = cls.heuristic_specs(('GUB', 'delta_set'))[1]
                return [(suite, spec.name, spec, 100, 'search') for suite in
                        ('heuristic_ablation', 'dominance_ablation', 'heuristic_inflation')]

            def prepare(self, record):
                path = self.work / 'inputs' / f'{record.instance_id}.txt'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('1 1 100 1 0 1 0 1 2 1 -1 2 2 0\n')
                self.backend.bind_cache(path, record.base_instance_id, record.source_domain)
                return replace(record, stgcs_num_vertices=20, stgcs_num_edges=40)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / 'exe'
            executable.write_text('fixed executable')
            work = root / 'work'
            (work / 'logs').mkdir(parents=True)
            args = SimpleNamespace(work_dir=work, executable=executable, manifest=Path('data/instances/st_planning/manifest.json'),
                base_root=root, output_root=root / 'results', workers=1, limit=1, suite='all', planner=None,
                joint_motion=True, performance_budget=100, ablation_budget=100, cache_timeout=10000, skip_scaling=True)
            result = dict(success=True, status=0, runtime=1, cost=10, expanded=2, generated=3, segments=[])
            with patch('experiments.runners.st.STGCSPlannerClient.run_process', return_value=result) as run, patch('sys.stdout', new=io.StringIO()):
                suite = FixtureSuite(args)
                suite.execute()
                self.assertEqual((run.call_count, suite.completed), (1, 3))
                csvs = sorted(args.output_root.glob('*/*.csv'))
                self.assertEqual(len(csvs), 3)
                self.assertEqual(len({p.read_bytes() for p in csvs}), 1)
                csvs[-1].unlink()
                resumed = FixtureSuite(args)
                resumed.execute()
                self.assertEqual((run.call_count, resumed.completed), (1, 3))
                executable.write_text('changed executable')
                FixtureSuite(args).execute()
                self.assertEqual(run.call_count, 1)

    def test_prepare_writes_current_instance_values(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            backend = STGCSPlannerClient(root / 'exe', root)
            query = SimpleNamespace(start=[0, 0], goal=[1, 1], vlimit=1, t_start=0, is_stay=True)
            record = SimpleNamespace(instance_id='example', base_instance_id='base-example', source_domain='simple2d',
                                     query=query, tmax=20, dynamic_obstacles=[])
            instance = SimpleNamespace(env=SimpleNamespace(_CSpace_hpoly=[]))
            with patch.object(STGCSPlannerClient, 'base_instance', return_value=instance):
                path = backend.prepare(record, root)
                self.assertEqual(float(path.read_text().split()[2]), 20)
                record.tmax = 21
                self.assertEqual(backend.prepare(record, root), path)
                self.assertEqual(float(path.read_text().split()[2]), 21)

    def test_prepare_checks_saved_graph_instead_of_original_selection_caps(self):
        from experiments.manifests.st_planning import STPlanningManifestStore

        record = STPlanningManifestStore.load_manifest('data/instances/st_planning/manifest.json')[0]
        record = replace(record, tmax=100, stgcs_num_vertices=435, stgcs_num_edges=2694,
                         generation={'parameters': {'tmax': 20, 'max_vertices': 400, 'max_edges': 2500}})
        with TemporaryDirectory() as directory:
            root = Path(directory)
            suite = STSuite.__new__(STSuite)
            suite.work = root
            (root / 'inputs').mkdir()
            suite.args = SimpleNamespace(base_root=root, cache_timeout=10000)
            suite.backend = STGCSPlannerClient(root / 'exe', root)
            with patch.object(suite.backend, 'prepare', return_value=root / 'input.txt'), \
                    patch.object(suite.backend, 'cache', return_value={'st_vertices': 435, 'st_edges': 2694}) as cache:
                self.assertEqual(suite.prepare(record), record)
                cache.return_value = {'st_vertices': 435, 'st_edges': 2695}
                with self.assertRaisesRegex(ValueError, 'differs from manifest'):
                    suite.prepare(record)

    def test_append_suite_reuses_saved_measurements_and_preserves_protocol(self):
        class FixtureSuite(STSuite):
            @staticmethod
            def fake_process(command, *_args):
                if command[1] == 'plan':
                    return dict(success=True, status=0, runtime=1, cost=10, expanded=2, generated=3, segments=[])
                from experiments.planners.st_planning import STPerformanceComparisonConfig as Config
                names = {'strrt': Config.ST_RRT_OUTPUT_PLANNERS,
                         'strrt_matched': (Config.ST_RRT_MATCHED_PLANNER,), 'zeta': (Config.ZETA_SIPP_PLANNER,),
                         'zeta2r': (Config.ZETA_SIPP_2R_PLANNER,)}[command[-2]]
                return {name: dict(is_success=True, runtime=1, cost=10,
                                   num_expanded_nodes=2, num_generated_nodes=3) for name in names}

            def prepare(self, record):
                path = self.work / 'inputs' / f'{record.instance_id}.txt'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('1 1 100 1 0 1 0 1 2 1 -1 2 2 0\n')
                self.backend.bind_cache(path, record.base_instance_id, record.source_domain)
                return replace(record, stgcs_num_vertices=20, stgcs_num_edges=40)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / 'exe'
            executable.write_text('fixed executable')
            args = SimpleNamespace(work_dir=root / 'work', executable=executable,
                manifest=Path('data/instances/st_planning/manifest.json'), base_root=root,
                output_root=root / 'results', workers=1, limit=1, suite='heuristic_ablation',
                planner=None, joint_motion=True, performance_budget=100, ablation_budget=100, heuristic_budget=100,
                cache_timeout=10000, skip_scaling=True, append_suite=[])
            (args.work_dir / 'logs').mkdir(parents=True)
            with patch('experiments.runners.st.STGCSPlannerClient.run_process', side_effect=FixtureSuite.fake_process) as run, \
                    patch('sys.stdout', new=io.StringIO()):
                FixtureSuite(args).execute()
                self.assertEqual(run.call_count, 12)
                original = (args.work_dir / 'metadata.json').read_bytes()
                self.assertEqual(json.loads(original)['heuristic_budget'], 100)
                retained = {p: p.read_bytes() for p in (args.work_dir / 'logs').rglob('*.json')}
                retained.update({p: p.read_bytes() for p in args.output_root.glob('*/*.csv')})
                args.append_suite = ['dominance_ablation']
                extended = FixtureSuite(args)
                extended.execute()
                self.assertEqual((run.call_count, extended.completed), (14, 16))
                self.assertEqual({p: p.read_bytes() for p in retained}, retained)
                for job in extended.jobs[-4:-2]:
                    name = job[1]
                    self.assertEqual((args.output_root / 'dominance_ablation' / f'{name}.csv').read_bytes(),
                                     (args.output_root / 'heuristic_ablation' / f'{name}.csv').read_bytes())
                self.assertFalse((args.work_dir / 'metadata-history').exists())
                FixtureSuite(args).execute()
                self.assertEqual(run.call_count, 14)
                for appended, calls, rows in (('performance_comparison', 22, 26), ('heuristic_inflation', 52, 56)):
                    args.append_suite.append(appended)
                    continued = FixtureSuite(args)
                    continued.execute()
                    self.assertEqual((run.call_count, continued.completed), (calls, rows))
                    self.assertEqual({p: p.read_bytes() for p in retained}, retained)
                args.inflation_budget = 31
                with self.assertRaisesRegex(ValueError, 'changed jobs'):
                    FixtureSuite(args).execute()
                args.inflation_budget = 30
                args.heuristic_budget = 31
                with self.assertRaisesRegex(ValueError, 'changed jobs'):
                    FixtureSuite(args).execute()
                args.heuristic_budget = 100
                args.ablation_budget = 99
                with self.assertRaisesRegex(ValueError, 'changed jobs'):
                    FixtureSuite(args).execute()
                args.ablation_budget = 100
                args.append_suite = []
                with self.assertRaisesRegex(ValueError, 'changed jobs'):
                    FixtureSuite(args).execute()
                args.append_suite = ['heuristic_ablation']
                with self.assertRaisesRegex(ValueError, 'already scheduled'):
                    FixtureSuite(args)

    def test_fixed_instances_and_retire_planner_preserve_completed_measurements(self):
        from experiments.manifests.st_planning import STPlanningManifestStore
        from experiments.planners.st_planning import STPerformanceComparisonConfig as Config

        class FixtureSuite(STSuite):
            @classmethod
            def jobs(cls, *args):
                spec = Config.DELTA_POS_SPEC
                return [('performance_comparison', spec.name, spec, 30, 'search'),
                        ('performance_comparison', Config.MICP_PLANNER, spec, 30, 'micp')]

            def prepare(self, record):
                path = self.work / 'inputs' / f'{record.instance_id}.txt'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f'1 1 {record.tmax} 1 0 1 0 1 2 1 -1 2 2 0\n')
                self.backend.bind_cache(path, record.base_instance_id, record.source_domain)
                return replace(record, stgcs_num_vertices=20, stgcs_num_edges=40)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / 'exe'
            executable.write_text('fixed executable')
            records = STPlanningManifestStore.load_manifest('data/instances/st_planning/manifest.json')[:2]
            manifest = root / 'manifest.json'
            STPlanningManifestStore.save_manifest(manifest, records)
            args = SimpleNamespace(work_dir=root / 'work', executable=executable, manifest=manifest,
                base_root=root, output_root=root / 'results', workers=1, limit=None,
                suite='performance_comparison', planner=None, joint_motion=False,
                performance_budget=30, ablation_budget=100, cache_timeout=10000, skip_scaling=True)
            (args.work_dir / 'logs').mkdir(parents=True)
            result = dict(success=True, status=0, runtime=1, cost=10, expanded=2, generated=3, segments=[])
            with patch('experiments.runners.st.STGCSPlannerClient.run_process', return_value=result) as run, \
                    patch('sys.stdout', new=io.StringIO()):
                FixtureSuite(args).execute()
                self.assertEqual(run.call_count, 4)
                retained = {p: p.read_bytes() for p in (args.work_dir / 'logs').rglob('*.json')}
                old_csv = (args.output_root / 'performance_comparison' / f'{Config.DELTA_POS_PLANNER}.csv').read_bytes()
                resumed = FixtureSuite(args)
                resumed.jobs = resumed.jobs[:1]
                resumed.execution_jobs = resumed.jobs
                resumed.expected = 2
                with self.assertRaisesRegex(ValueError, 'changed jobs'):
                    resumed.execute()
                args.retire_planner = [Config.MICP_PLANNER]
                resumed.execute()
                self.assertEqual((run.call_count, resumed.completed), (4, 2))
                self.assertEqual({p: p.read_bytes() for p in retained}, retained)
                self.assertFalse((args.output_root / 'run.json').exists())
                self.assertFalse(list(args.output_root.rglob('*.json')))
                self.assertEqual((args.output_root / 'performance_comparison' / f'{Config.DELTA_POS_PLANNER}.csv').read_bytes(), old_csv)
                self.assertFalse((args.work_dir / 'metadata-history').exists())
                resumed.completed = 0
                resumed.execute()
                self.assertEqual((run.call_count, resumed.completed), (4, 2))
                added = replace(records[0], instance_id='additional-instance')
                for changed in ([replace(records[0], tmax=21), records[1]], records[1:], [*records, added]):
                    STPlanningManifestStore.save_manifest(manifest, changed)
                    with self.assertRaisesRegex(ValueError, 'changed instances'):
                        FixtureSuite(args).execute()
                STPlanningManifestStore.save_manifest(manifest, [*records, records[0]])
                with self.assertRaisesRegex(ValueError, 'Duplicate instance'):
                    FixtureSuite(args)

    def test_resume_prioritizes_reference_without_repeating_completed_runs(self):
        class FixtureSuite(STSuite):
            @classmethod
            def jobs(cls, *args):
                specs = [*cls.heuristic_specs(('GUB',))[:2],
                         cls.heuristic_specs(('GUB', 'delta_set'))[4]]
                return [('heuristic_ablation', spec.name, spec, 100, 'search') for spec in specs]

            def prepare(self, record):
                path = self.work / 'inputs' / f'{record.instance_id}.txt'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('1 1 100 1 0 1 0 1 2 1 -1 2 2 0\n')
                self.backend.bind_cache(path, record.base_instance_id, record.source_domain)
                return replace(record, stgcs_num_vertices=20, stgcs_num_edges=40)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / 'exe'
            executable.write_text('fixed executable')
            args = SimpleNamespace(work_dir=root / 'work', executable=executable,
                manifest=Path('data/instances/st_planning/manifest.json'), base_root=root,
                output_root=root / 'results', workers=1, limit=2, suite='heuristic_ablation',
                planner=None, joint_motion=False, performance_budget=100, ablation_budget=100,
                cache_timeout=10000, skip_scaling=True)
            (args.work_dir / 'logs').mkdir(parents=True)
            result = dict(success=True, status=0, runtime=1, cost=10, expanded=2, generated=3, segments=[])
            with patch('experiments.runners.st.STGCSPlannerClient.run_process', return_value=result) as run, \
                    patch('sys.stdout', new=io.StringIO()):
                suite = FixtureSuite(args)
                suite.execute()
                original_jobs = json.loads((args.work_dir / 'metadata.json').read_text())['jobs']
                retained = Store.planner_output_path(args.output_root / 'heuristic_ablation', suite.jobs[0][1])
                retained_bytes = retained.read_bytes()
                for job in suite.jobs[1:]:
                    Store.planner_output_path(args.output_root / 'heuristic_ablation', job[1]).unlink()
                for path in (args.work_dir / 'logs').rglob('*.json'):
                    if json.loads(path.read_text())['planner'] != suite.jobs[0][1]:
                        path.unlink()
                run.reset_mock()
                args.first_planner = suite.jobs[-1][1]
                resumed = FixtureSuite(args)
                resumed.execute()
                self.assertEqual(resumed.completed, 6)
                self.assertEqual([call.args[0][4:6] for call in run.call_args_list],
                                 [['h_max', 'delta_set']] * 2 + [['h_mot', 'none']] * 2)
                self.assertEqual(retained.read_bytes(), retained_bytes)
                metadata = json.loads((args.work_dir / 'metadata.json').read_text())
                self.assertEqual(metadata['jobs'], original_jobs)
                self.assertEqual(metadata['execution_order'][0], args.first_planner)
                args.first_planner = 'not scheduled'
                with self.assertRaisesRegex(ValueError, 'not in this suite'):
                    FixtureSuite(args)


if __name__ == '__main__':
    unittest.main()
