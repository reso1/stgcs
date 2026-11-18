import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from matplotlib.collections import LineCollection, PathCollection, PolyCollection

from experiments.planners.mrmp import WindowedCoordinationAblation, WindowedCoordinationSpec
from experiments.reports.mrmp_windows import MRMPWindowedCoordinationReport as Report
from experiments.reports.common import ViolinStyle, plt


class WindowedPartialPlotTest(unittest.TestCase):
    @staticmethod
    def row(instance_id, success, runtime, budget=60.0):
        return dict(instance_id=instance_id, domain_key="grid2d", budget=budget,
                    is_success=success, runtime=runtime, sum_of_costs=100.0, makespan=5.0)

    def setUp(self):
        self.specs = WindowedCoordinationAblation.specs()
        self.rows = {
            spec.key: [self.row("a", True, 10), self.row("b", False, 60.1)]
            for spec in self.specs
        }
        self.rows[self.specs[0].key].append(self.row("unfinished", True, 1))
        self.rows[self.specs[0].key].append(self.row("a", True, 1, budget=600))
        self.rows[self.specs[-1].key][0] = self.row("a", False, 60.1)

    def test_partial_cohort_keeps_failures_and_excludes_missing_and_other_budgets(self):
        shared = Report.common_evaluated_rows(self.rows, 60)
        for spec in self.specs:
            self.assertEqual([row["instance_id"] for row in shared[spec.key]], ["a", "b"])
            self.assertEqual(Report.success_rate_curve_points(shared, 60, spec.key)[0][-1], 60)
        self.assertEqual(Report.success_rate(shared, 60, self.specs[0].key), 0.5)
        self.assertEqual(Report.success_rate(shared, 60, self.specs[-1].key), 0.0)
        self.assertEqual(len({Report.success_rate_curve_points(shared, 60, spec.key)[0][0]
                              for spec in self.specs}), 1)

    def test_two_by_two_figure_uses_common_successes_for_all_distributions(self):
        for index, spec in enumerate(self.specs):
            row = self.row("common", True, 20 + index)
            row.update(sum_of_costs=200 + 10 * index, makespan=10 + index)
            self.rows[spec.key].append(row)
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "figure"
            with patch("experiments.reports.mrmp_windows._save_figure") as save:
                Report.plot_figure(self.rows, self.specs, 60, prefix, ("png",), domains=("grid2d",))
                figure = save.call_args.args[0]
                try:
                    success, runtime, soc, makespan = figure.axes
                    figure.canvas.draw()
                    for axis in figure.axes:
                        self.assertEqual(axis.get_subplotspec().get_gridspec().get_geometry(), (2, 2))
                        self.assertEqual(axis.get_box_aspect(), 1)
                        extent = axis.get_window_extent()
                        self.assertAlmostEqual(extent.width, extent.height)
                    self.assertEqual(len(success.lines), 5)
                    self.assertEqual([line.get_ydata()[-1] for line in success.lines],
                                     [2 / 3, 2 / 3, 2 / 3, 2 / 3, 1 / 3])
                    self.assertEqual(success.get_xlim(), (10, 60))
                    self.assertEqual(success.get_xticks()[0], 10)
                    self.assertEqual([tick.get_text() for tick in success.get_xticklabels()],
                                     ["", "30", "60"])
                    for axis, xlabel, expected in (
                        (runtime, "success runtime (s)", [20, 21, 22, 23, 24]),
                        (soc, "sum-of-costs", [200, 210, 220, 230, 240]),
                        (makespan, "makespan", [10, 11, 12, 13, 14]),
                    ):
                        self.assertEqual(axis.get_xlabel(), xlabel)
                        self.assertEqual(axis.get_ylabel(), "")
                        self.assertEqual(axis.get_xscale(), "log")
                        self.assertEqual(len(axis.patches), 0)
                        self.assertEqual(len(axis.get_yticks()), 5)
                        self.assertTrue(axis.yaxis_inverted())
                        points = [item for item in axis.collections if isinstance(item, PathCollection)]
                        medians = [item for item in axis.collections if isinstance(item, LineCollection)]
                        self.assertEqual([len(item.get_offsets()) for item in points],
                                         [1, 1, 1, 1, 1])
                        self.assertEqual([float(item.get_offsets()[0, 0]) for item in points],
                                         expected)
                        self.assertEqual([float(item.get_segments()[0][0, 0]) for item in medians],
                                         expected)
                finally:
                    plt.close(figure)
            summary = json.loads(prefix.with_suffix(".json").read_text())
            self.assertEqual(summary["instance_ids"], ["a", "b", "common"])
            self.assertEqual(summary["domain_counts"], {"grid2d": 3})
            self.assertEqual(summary["common_success_instance_ids"], ["common"])
            self.assertEqual(summary["planners"]["dynamic_inf"]["solved"], 1)
            self.assertEqual(summary["planners"]["dynamic_inf"]["common_success_medians"],
                             dict(runtime=24, sum_of_costs=240, makespan=14))

    def test_shared_violin_preserves_values_and_medians_in_both_orientations(self):
        figure, axes = plt.subplots(1, 2)
        try:
            for axis, orientation, value_axis in zip(axes, ("horizontal", "vertical"), (0, 1)):
                with self.subTest(orientation=orientation):
                    ViolinStyle.draw(axis, 3, [2, 4, 8], "blue", orientation=orientation)
                    body, median = axis.collections
                    self.assertIsInstance(body, PolyCollection)
                    self.assertIsInstance(median, LineCollection)
                    vertices = body.get_paths()[0].vertices
                    self.assertEqual(vertices[:, value_axis].min(), 2)
                    self.assertEqual(vertices[:, value_axis].max(), 8)
                    segment = median.get_segments()[0]
                    self.assertEqual(segment[:, value_axis].tolist(), [4, 4])
                    self.assertAlmostEqual(segment[:, 1 - value_axis].mean(), 3)
        finally:
            plt.close(figure)

    def test_no_common_successes_keeps_curves_and_marks_empty_distributions(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "empty"
            with patch("experiments.reports.mrmp_windows._save_figure") as save:
                Report.plot_figure(self.rows, self.specs, 60, prefix, ("png",), domains=("grid2d",))
                figure = save.call_args.args[0]
                try:
                    self.assertEqual(len(figure.axes[0].lines), 5)
                    for axis in figure.axes[1:]:
                        self.assertEqual(len(axis.patches), 0)
                        self.assertEqual([text.get_text() for text in axis.texts], ["No common successes"])
                finally:
                    plt.close(figure)
            summary = json.loads(prefix.with_suffix(".json").read_text())
            self.assertEqual(summary["common_success_instance_ids"], [])
            self.assertIsNone(summary["planners"]["dynamic_2"]["common_success_medians"]["runtime"])

    def test_no_shared_results_is_an_error(self):
        self.rows[self.specs[-1].key] = []
        with self.assertRaisesRegex(ValueError, "every selected planner"):
            Report.common_evaluated_rows(self.rows, 60)

    def test_completed_comparison_requires_exact_manifest_coverage(self):
        specs = (
            WindowedCoordinationSpec(4, True),
            WindowedCoordinationSpec(4, False),
            WindowedCoordinationSpec(4, True, execution_horizon_factor=0.5),
        )
        records = [SimpleNamespace(instance_id=i, num_agents=20, domain_key="grid2d") for i in ("a", "b")]
        rows = {
            specs[0].key: [dict(self.row(i, True, 10), num_agents=20) for i in ("a", "b")],
            specs[1].key: [dict(self.row("a", True, 10), num_agents=20)],
            specs[2].key: [dict(self.row(i, True, 10), num_agents=20) for i in ("a", "wrong")],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / 'grid2d/results'
            results.mkdir(parents=True)
            for spec in specs:
                Report.windowed_result_path(root, 'grid2d', spec).touch()
            with patch("experiments.reports.mrmp_windows.load_manifest", return_value=records), \
                 patch.object(Report, "load_rows_by_planner", return_value=rows):
                selected, loaded, budget, incomplete = Report.load_completed_results(root, 60, [s.key for s in specs])
            self.assertEqual([s.key for s in selected], ["dynamic_4"])
            self.assertEqual(set(loaded), {"dynamic_4"})
            self.assertEqual(budget, 60)
            self.assertEqual(set(incomplete), {"fixed_4", "dynamic_4_beta0p5"})

    def test_consolidated_results_load_all_variants_with_local_manifest(self):
        specs = tuple(WindowedCoordinationSpec(alpha, True) for alpha in (1, 2, 4, 8, 16, float('inf'))) + (
            WindowedCoordinationSpec(2, False), WindowedCoordinationSpec(4, False),
            WindowedCoordinationSpec(2, True, execution_horizon_factor=0.5),
            WindowedCoordinationSpec(4, True, execution_horizon_factor=0.5),
        )
        records = [SimpleNamespace(instance_id=i, num_agents=20, domain_key="grid2d") for i in ("a", "b")]
        rows = {s.key: [dict(self.row(i, True, 10, budget=120), num_agents=20) for i in ("a", "b")]
                for s in specs}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / 'grid2d/results'
            results.mkdir(parents=True)
            for spec in specs:
                Report.windowed_result_path(root, 'grid2d', spec).touch()
            with patch("experiments.reports.mrmp_windows.load_manifest", return_value=records) as manifest, \
                 patch.object(Report, "load_rows_by_planner", return_value=rows):
                selected, loaded, budget, incomplete = Report.load_completed_results(root)
            manifest.assert_called_once_with(root / "manifest.json")
            self.assertEqual([s.key for s in selected], [s.key for s in specs])
            self.assertEqual(set(loaded), {s.key for s in specs})
            self.assertEqual(budget, 120)
            self.assertEqual(incomplete, [])

    def test_unknown_planner_files_and_missing_requested_variants_are_rejected(self):
        records = [SimpleNamespace(instance_id='a', num_agents=20, domain_key='grid2d')]
        with tempfile.TemporaryDirectory() as directory, patch(
                'experiments.reports.mrmp_windows.load_manifest', return_value=records):
            root = Path(directory)
            results = root / 'grid2d/results'
            results.mkdir(parents=True)
            unknown = results / 'unknown.csv'
            unknown.touch()
            with self.assertRaisesRegex(ValueError, 'Unknown windowed planner'):
                Report.load_completed_results(root)
            unknown.unlink()
            Report.windowed_result_path(root, 'grid2d', WindowedCoordinationSpec(2, True)).touch()
            with self.assertRaisesRegex(ValueError, 'missing'):
                Report.load_completed_results(root, planner_keys=['fixed_4'])

    def test_mixed_alpha_four_variants_have_distinct_colors_and_line_styles(self):
        specs = (WindowedCoordinationSpec(4, True), WindowedCoordinationSpec(4, False),
                 WindowedCoordinationSpec(4, True, execution_horizon_factor=0.5))
        rows = {spec.key: [self.row("common", True, 10)] for spec in specs}
        with tempfile.TemporaryDirectory() as directory:
            with patch("experiments.reports.mrmp_windows._save_figure") as save:
                Report.plot_figure(rows, specs, 60, Path(directory) / "mixed", ("png",))
                figure = save.call_args.args[0]
                try:
                    lines = figure.axes[0].lines
                    self.assertEqual(len({line.get_label() for line in lines}), 3)
                    self.assertEqual(len({line.get_color() for line in lines}), 3)
                    self.assertEqual([line.get_linestyle() for line in lines], ["-", "--", "-."])
                    self.assertEqual([text.get_text() for text in figure.legends[0].get_texts()], [
                        r"D4: dynamic, $\alpha=4$",
                        r"F4: fixed, $\alpha=4$",
                        r"H4: dynamic, $\alpha=4$, $\beta=0.5$",
                    ])
                    for axis in figure.axes[1:]:
                        self.assertEqual([tick.get_text() for tick in axis.get_yticklabels()],
                                         ["D4", "F4", "H4"])
                finally:
                    plt.close(figure)


if __name__ == "__main__":
    unittest.main()
