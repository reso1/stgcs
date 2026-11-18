from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
from statistics import median
from typing import Mapping, Sequence

from experiments.reports.style import configure_matplotlib_for_latex

configure_matplotlib_for_latex(backend="Agg")

from matplotlib.ticker import NullFormatter, PercentFormatter

from experiments.datasets.st_stress import STStressDataset
from experiments.results import STResultStore
from experiments.planners.st_planning import STExperimentConfig
from experiments.manifests.base import BaseBenchmarkRecord, load_manifest
from experiments.offline_heuristics import BaseOfflineHeuristicStore
from experiments.planners.mrmp import SearchPlannerSpec
from experiments.reports.common import PlotPalette, _save_figure, dedupe_result_rows, plt
from experiments.reports.st_comparison import STPerformanceComparisonReport


class HeuristicInflationAndScalingPlot:
    DEFAULT_INFLATION_RESULTS_ROOT = Path("data/exp_res/st_planning/heuristic_inflation")
    DEFAULT_SCALING_MANIFEST = Path("data/stgcs_base/heur_computation_time_scaling/manifest.json")
    DEFAULT_STRESS_RESULTS_ROOT = Path("data/exp_res/st_planning/st_planning_stress/performance_comparison")
    DEFAULT_STRESS_BUDGET = 30.0
    DEFAULT_OUTPUT_PREFIX = Path("latex/figs/heur_precomputation_scaling")
    DEFAULT_BUDGET = STExperimentConfig.DEFAULT_INFLATION_BUDGET
    DEFAULT_FONT_SIZE = 18
    CURVE_LINEWIDTH = 2.0
    AXIS_LINEWIDTH = 1.2
    GRID_LINEWIDTH = 0.9
    HEURISTICS = (*STExperimentConfig.HEURISTICS, "joint")
    BASELINE_EPSILON = 1.0
    EPSILONS = STExperimentConfig.EPSILONS
    DOMINANCE_CHECK_STACK = STExperimentConfig.INFLATION_DOMINANCE
    SCALING_HEURISTICS = (
        BaseOfflineHeuristicStore.INTERFACE_TO_SET_COST_TABLE_NAME,
        BaseOfflineHeuristicStore.TRIPLET_RELAXATION_NAME,
    )
    SCALING_INLINE_LABEL_TARGET_X = {
        BaseOfflineHeuristicStore.INTERFACE_TO_SET_COST_TABLE_NAME: 125.0,
        BaseOfflineHeuristicStore.TRIPLET_RELAXATION_NAME: 130.0,
    }
    SCALING_INLINE_LABEL_Y_FACTOR = {
        BaseOfflineHeuristicStore.INTERFACE_TO_SET_COST_TABLE_NAME: 3.0,
        BaseOfflineHeuristicStore.TRIPLET_RELAXATION_NAME: 0.45,
    }
    SCALING_X_LOWER_LIMIT = -10
    SCALING_SIZE_UPPER_LIMIT = 210
    SCALING_X_TICKS = (0, 100, 200)
    HEURISTIC_STYLES = {
        "h_mot": {
            "label": r"$h_\text{mot}$",
            "color": PlotPalette.heuristic_color("h_mot"),
            "marker": "o",
        },
        "h_tri": {
            "label": r"$h_\text{tri}$",
            "color": PlotPalette.heuristic_color("h_tri"),
            "marker": "^",
        },
        "h_tab": {
            "label": r"$h_\text{tab}$",
            "color": PlotPalette.heuristic_color("h_tab"),
            "marker": "s",
        },
        "h_max": {
            "label": r"$h_\text{max}$",
            "color": PlotPalette.heuristic_color("h_max"),
            "marker": "D",
        },
        "joint": {"label": "joint", "color": "#8c6bb1", "marker": "P"},
    }
    SCALING_HEURISTIC_STYLES = {
        BaseOfflineHeuristicStore.INTERFACE_TO_SET_COST_TABLE_NAME: HEURISTIC_STYLES["h_tab"],
        BaseOfflineHeuristicStore.TRIPLET_RELAXATION_NAME: HEURISTIC_STYLES["h_tri"],
    }

    @classmethod
    def planner_spec(cls, heuristic: str, epsilon: float) -> SearchPlannerSpec:
        return SearchPlannerSpec("h_mot" if heuristic == "joint" else str(heuristic),
                                 cls.DOMINANCE_CHECK_STACK, epsilon=float(epsilon),
                                 key_evaluation="joint" if heuristic == "joint" else "separate",
                                 upper_bound_heuristic="h_mot")

    @classmethod
    def planner_name(cls, heuristic: str, epsilon: float) -> str:
        return cls.planner_spec(heuristic, epsilon).name

    @classmethod
    def method_label(cls, heuristic: str) -> str:
        return str(cls.HEURISTIC_STYLES[str(heuristic)]["label"])

    @classmethod
    def planner_names(cls) -> tuple[str, ...]:
        plotted_names = tuple(
            cls.planner_name(heuristic, epsilon)
            for heuristic in cls.HEURISTICS
            for epsilon in cls.EPSILONS
        )
        baseline_names = tuple(cls.planner_name(heuristic, cls.BASELINE_EPSILON) for heuristic in cls.HEURISTICS)
        return (*baseline_names, *plotted_names)

    @classmethod
    def load_inflation_rows_by_planner(
        cls,
        baseline_results_root: Path,
        inflation_results_root: Path,
    ) -> dict[str, list[dict[str, object]]]:
        rows_by_planner: dict[str, list[dict[str, object]]] = {}
        for heuristic in cls.HEURISTICS:
            baseline_name = cls.planner_name(heuristic, cls.BASELINE_EPSILON)
            baseline_path = STResultStore.planner_output_path(baseline_results_root, baseline_name)
            rows_by_planner[baseline_name] = dedupe_result_rows(
                STResultStore.load_result_rows(baseline_path)
            )
            for epsilon in cls.EPSILONS:
                planner_name = cls.planner_name(heuristic, epsilon)
                path = STResultStore.planner_output_path(inflation_results_root, planner_name)
                rows_by_planner[planner_name] = dedupe_result_rows(STResultStore.load_result_rows(path))
        return rows_by_planner

    @classmethod
    def select_budget(
        cls,
        rows_by_planner: Mapping[str, Sequence[dict[str, object]]],
        requested_budget: float | None,
    ) -> float:
        if requested_budget is not None:
            return float(requested_budget)
        budgets_by_planner: list[set[float]] = []
        observed_budgets: set[float] = set()
        for planner_name in cls.planner_names():
            planner_budgets: set[float] = set()
            for row in rows_by_planner.get(planner_name, []):
                budget = float(row["budget"])
                if budget > 0.0:
                    planner_budgets.add(budget)
                    observed_budgets.add(budget)
            if planner_budgets:
                budgets_by_planner.append(planner_budgets)
        if budgets_by_planner:
            common_budgets = set.intersection(*budgets_by_planner)
            if common_budgets:
                return max(common_budgets)
        if observed_budgets:
            return max(observed_budgets)
        return cls.DEFAULT_BUDGET

    @staticmethod
    def same_budget(row: dict[str, object], budget: float) -> bool:
        return math.isclose(float(row["budget"]), float(budget))

    @staticmethod
    def finite_positive(value: object) -> float | None:
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            return None
        return number

    @classmethod
    def success_rows_by_instance(
        cls,
        rows: Sequence[dict[str, object]],
        budget: float,
    ) -> dict[str, dict[str, object]]:
        rows_by_instance: dict[str, dict[str, object]] = {}
        for row in rows:
            if not cls.same_budget(row, budget) or not bool(row["is_success"]):
                continue
            rows_by_instance[str(row["instance_id"])] = row
        return rows_by_instance


    @classmethod
    def inflation_values_by_epsilon(
        cls,
        rows_by_planner: Mapping[str, Sequence[dict[str, object]]],
        heuristic: str,
        budget: float,
        metric: str,
    ) -> dict[float, list[float]]:
        baseline_name = cls.planner_name(heuristic, cls.BASELINE_EPSILON)
        baseline_rows = cls.success_rows_by_instance(rows_by_planner.get(baseline_name, []), budget)
        values_by_epsilon: dict[float, list[float]] = {}
        for epsilon in cls.EPSILONS:
            planner_name = cls.planner_name(heuristic, epsilon)
            rows = cls.success_rows_by_instance(rows_by_planner.get(planner_name, []), budget)
            values: list[float] = []
            for instance_id in sorted(set(baseline_rows) & set(rows)):
                base_value = cls.finite_positive(baseline_rows[instance_id][metric])
                value = cls.finite_positive(rows[instance_id][metric])
                if base_value is None or value is None:
                    continue
                if metric == "cost":
                    values.append(100.0 * (value / base_value - 1.0))
                elif metric == "runtime":
                    values.append(100.0 * (1.0 - value / base_value))
                else:
                    raise ValueError(f"Unsupported inflation metric {metric!r}.")
            if values:
                values_by_epsilon[float(epsilon)] = values
        return values_by_epsilon


    @staticmethod
    def graph_size(record: BaseBenchmarkRecord) -> int:
        return int(record.stgcs_num_vertices) + int(record.stgcs_num_edges)

    @classmethod
    def max_graph_size(cls, manifest_path: Path) -> int | None:
        records = load_manifest(manifest_path)
        if not records:
            return None
        return max(cls.graph_size(record) for record in records)

    @classmethod
    def scaling_size_upper_limit(
        cls,
        manifest_path: Path,
        requested_upper_limit: float | None,
    ) -> float:
        if requested_upper_limit is not None:
            upper_limit = float(requested_upper_limit)
        else:
            max_graph_size = cls.max_graph_size(manifest_path)
            upper_limit = float(max_graph_size) if max_graph_size is not None else float(max(cls.SCALING_X_TICKS))
        if not math.isfinite(upper_limit) or upper_limit <= cls.SCALING_X_LOWER_LIMIT:
            raise ValueError(
                f"Scaling size upper limit must be finite and greater than "
                f"{cls.SCALING_X_LOWER_LIMIT}, got {upper_limit}."
            )
        return upper_limit

    @classmethod
    def timing_points(
        cls,
        manifest_path: Path,
        heuristic: str,
    ) -> list[tuple[int, float, str]]:
        points: list[tuple[int, float, str]] = []
        for record in load_manifest(manifest_path):
            runtime = cls.finite_positive(record.heuristic_computation_times.get(str(heuristic), 0.0))
            if runtime is None:
                continue
            points.append((cls.graph_size(record), runtime, record.instance_id))
        return points

    @staticmethod
    def median_curve(points: Sequence[tuple[int, float, str]]) -> tuple[list[int], list[float]]:
        grouped: dict[int, list[float]] = defaultdict(list)
        for graph_size, runtime, _ in points:
            grouped[int(graph_size)].append(float(runtime))
        xs = sorted(grouped)
        ys = [median(grouped[x]) for x in xs]
        return xs, ys

    @staticmethod
    def rotate_y_tick_labels(ax: plt.Axes) -> None:
        ax.tick_params(axis="y", labelrotation=90)
        for label in [*ax.get_yticklabels(), *ax.get_yticklabels(minor=True)]:
            label.set_rotation(90)
            label.set_verticalalignment("center")


    @classmethod
    def plot_scaling_panel(
        cls,
        ax: plt.Axes,
        manifest_path: Path,
        font_size: float,
        size_upper_limit: float | None = SCALING_SIZE_UPPER_LIMIT,
    ) -> bool:
        plotted = False
        for heuristic in cls.SCALING_HEURISTICS:
            points = cls.timing_points(manifest_path, heuristic)
            if not points:
                print(f"No {heuristic} timing records found for {manifest_path}")
                continue
            plotted = True
            style = cls.SCALING_HEURISTIC_STYLES[heuristic]
            curve_xs, curve_ys = cls.median_curve(points)
            ax.plot(
                curve_xs,
                curve_ys,
                color=str(style["color"]),
                linewidth=cls.CURVE_LINEWIDTH,
                label=cls.method_label(str(heuristic)),
            )
            cls.draw_scaling_inline_label(ax, heuristic, curve_xs, curve_ys, font_size)
        ax.set_yscale("log")
        ax.set_yticks((1e-5, 1e-1, 1e3))
        ax.set_yticklabels((r"$10^{-5}$", r"$10^{-1}$", r"$10^{3}$"), fontsize=font_size)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"$G_0$ size ($|V_0| + |E_0|$)", fontsize=font_size)
        x_upper_limit = cls.scaling_size_upper_limit(manifest_path, size_upper_limit)
        ax.set_xticks(tuple(tick for tick in cls.SCALING_X_TICKS if tick <= x_upper_limit))
        ax.set_xlim(cls.SCALING_X_LOWER_LIMIT, x_upper_limit)
        if ax.get_xticklabels():
            ax.get_xticklabels()[-1].set_horizontalalignment("center")
        ax.set_ylabel("precomputation time (s)", fontsize=font_size, rotation=90)
        ax.tick_params(axis="both", labelsize=font_size)
        ax.tick_params(axis="y", which="both", left=True, labelleft=True)
        cls.rotate_y_tick_labels(ax)
        ax.grid(alpha=0.25, linewidth=cls.GRID_LINEWIDTH)
        return plotted

    @classmethod
    def draw_scaling_inline_label(
        cls,
        ax: plt.Axes,
        heuristic: str,
        curve_xs: Sequence[int],
        curve_ys: Sequence[float],
        font_size: float,
    ) -> None:
        if not curve_xs or not curve_ys:
            return
        target_x = float(cls.SCALING_INLINE_LABEL_TARGET_X[str(heuristic)])
        label_idx = min(range(len(curve_xs)), key=lambda idx: abs(float(curve_xs[idx]) - target_x))
        style = cls.SCALING_HEURISTIC_STYLES[str(heuristic)]
        y_factor = float(cls.SCALING_INLINE_LABEL_Y_FACTOR[str(heuristic)]) / 25
        ax.text(
            float(curve_xs[label_idx]) - 100,
            float(curve_ys[label_idx]) * y_factor,
            cls.method_label(str(heuristic)),
            color=str(style["color"]),
            fontsize=font_size,
            ha="left",
            va="center",
            zorder=4,
        )

    @classmethod
    def plot(
        cls,
        scaling_manifest_path: Path,
        output_prefix: Path,
        formats: Sequence[str],
        font_size: float = DEFAULT_FONT_SIZE,
        scaling_size_upper_limit: float | None = SCALING_SIZE_UPPER_LIMIT,
        stress_results_root: Path = DEFAULT_STRESS_RESULTS_ROOT,
        stress_budget: float = DEFAULT_STRESS_BUDGET,
    ) -> None:
        """Plot offline precomputation and online stress-test results (paper Figure B1)."""
        font_size = float(font_size)
        if not math.isfinite(font_size) or font_size <= 0.0:
            raise ValueError(f"font_size must be positive and finite, got {font_size}.")

        fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.2), layout="constrained")
        scaling_plotted = cls.plot_scaling_panel(
            axes[0],
            scaling_manifest_path,
            font_size,
            scaling_size_upper_limit,
        )
        if not scaling_plotted:
            plt.close(fig)
            raise ValueError(f"No h_tab/h_tri timing records found for {scaling_manifest_path}.")

        cls.plot_stress_panel(axes[1], stress_results_root, stress_budget, font_size)
        for ax in axes:
            ax.set_box_aspect(1)
            for spine in ax.spines.values():
                spine.set_linewidth(cls.AXIS_LINEWIDTH)
            ax.tick_params(which="major", width=cls.AXIS_LINEWIDTH, length=4.5)
            ax.tick_params(which="minor", width=cls.AXIS_LINEWIDTH, length=2.5)
        fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.08, wspace=0.08)
        _save_figure(fig, output_prefix, formats, bbox_inches=None)

    @classmethod
    def plot_stress_panel(
        cls,
        ax: plt.Axes,
        results_root: Path,
        budget: float,
        font_size: float,
    ) -> None:
        if not math.isfinite(budget) or budget <= 0.0:
            raise ValueError(f"stress_budget must be positive and finite, got {budget}.")
        report = STPerformanceComparisonReport
        planners = report.planner_names(("delta-pos", "delta-state"))
        rows_by_planner = {
            planner: [row for row in rows if report.same_budget(row, budget)]
            for planner, rows in report.load_rows_by_planner(results_root, planners).items()
        }
        sizes_by_planner = {
            planner: {str(row["instance_id"]): int(row["stgcs_num_vertices"]) + int(row["stgcs_num_edges"])
                      for row in rows}
            for planner, rows in rows_by_planner.items()
        }
        sizes = sizes_by_planner[planners[0]]
        if not sizes or any(values != sizes for values in sizes_by_planner.values()):
            raise ValueError(f"Stress-test planners require matching instances and graph sizes at budget {budget}.")
        bounds = STStressDataset.BOUNDS
        if any(not bounds[0] <= size < bounds[-1] for size in sizes.values()):
            raise ValueError("Stress-test graph sizes fall outside the dataset's size intervals.")
        upper_limit = bounds[-1] - 1
        midpoints = [(lower + min(upper, upper_limit)) / 2 for lower, upper in zip(bounds, bounds[1:])]

        for planner, dominance, linestyle, marker in zip(
            planners, ("delta_pos", "delta_state"), ("-", "--"), ("o", "s")
        ):
            success_rates = []
            for lower, upper in zip(bounds, bounds[1:]):
                rows = [row for row in rows_by_planner[planner]
                        if lower <= sizes[str(row["instance_id"])] < upper]
                solved = sum(bool(row["is_success"]) and math.isfinite(float(row["runtime"]))
                             and 0 < float(row["runtime"]) <= budget for row in rows)
                success_rates.append(solved / len(rows) if rows else math.nan)
            suffix = dominance.removeprefix("delta_")
            ax.plot(
                midpoints,
                success_rates,
                color=PlotPalette.dominance_color(dominance),
                linestyle=linestyle,
                linewidth=cls.CURVE_LINEWIDTH,
                marker=marker,
                markersize=7,
                markerfacecolor="white",
                markeredgewidth=1.4,
                label=rf"$\delta_\text{{{suffix}}}$",
            )
        padding = 0.04 * (upper_limit - bounds[0])
        ax.set_xlim(bounds[0] - padding, upper_limit + padding)
        ax.set_xticks((bounds[0], 10000, 20000, upper_limit))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3), useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(font_size)
        ax.set_ylim(-0.03, 1.04)
        ax.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_xlabel(r"$G$ size ($|V| + |E|$)", fontsize=font_size)
        ax.set_ylabel("success rate", fontsize=font_size)
        ax.tick_params(axis="both", labelsize=font_size)
        cls.rotate_y_tick_labels(ax)
        ax.grid(alpha=0.25, linewidth=cls.GRID_LINEWIDTH)
        ax.legend(loc="lower right", frameon=False, fontsize=font_size, handlelength=1.5, labelspacing=0.3)


class HeuristicInflationAndScalingPlotCLI:
    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Plot Figure B1 with square precomputation and spatiotemporal stress-test panels."
        )
        parser.add_argument(
            "--scaling-manifest",
            type=Path,
            default=HeuristicInflationAndScalingPlot.DEFAULT_SCALING_MANIFEST,
        )
        parser.add_argument("--output-prefix", type=Path, default=HeuristicInflationAndScalingPlot.DEFAULT_OUTPUT_PREFIX)
        parser.add_argument("--formats", nargs="+", default=("pdf", "png"))
        parser.add_argument("--font-size", type=float, default=HeuristicInflationAndScalingPlot.DEFAULT_FONT_SIZE)
        parser.add_argument(
            "--stress-results-root",
            type=Path,
            default=HeuristicInflationAndScalingPlot.DEFAULT_STRESS_RESULTS_ROOT,
        )
        parser.add_argument(
            "--stress-budget", type=float, default=HeuristicInflationAndScalingPlot.DEFAULT_STRESS_BUDGET
        )
        parser.add_argument(
            "--scaling-size-upper-limit",
            "--scaling-x-upper-limit",
            type=float,
            default=HeuristicInflationAndScalingPlot.SCALING_SIZE_UPPER_LIMIT,
            help="Upper limit for the ST-GCS size axis (default: %(default)s).",
        )
        return parser.parse_args()

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        HeuristicInflationAndScalingPlot.plot(
            scaling_manifest_path=args.scaling_manifest,
            output_prefix=args.output_prefix,
            formats=tuple(str(fmt) for fmt in args.formats),
            font_size=float(args.font_size),
            scaling_size_upper_limit=(
                None if args.scaling_size_upper_limit is None else float(args.scaling_size_upper_limit)
            ),
            stress_results_root=args.stress_results_root,
            stress_budget=args.stress_budget,
        )


if __name__ == "__main__":
    HeuristicInflationAndScalingPlotCLI.main()
