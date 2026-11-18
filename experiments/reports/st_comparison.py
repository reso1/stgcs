from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Dict, Mapping, Sequence

from experiments.reports.style import configure_matplotlib_for_latex

configure_matplotlib_for_latex(backend="Agg")

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter, NullLocator

from experiments.results import STResultStore
from experiments.planners.st_planning import STPerformanceComparisonConfig
from experiments.reports.common import PlotPalette, ViolinStyle, _save_figure, dedupe_result_rows, plt


class STPerformanceComparisonReport:
    DEFAULT_RESULTS_ROOT = Path("data/exp_res/st_planning/performance_comparison")
    DEFAULT_OUTPUT_PREFIX = Path("latex/figs/st_performance_comparison")
    DEFAULT_BUDGET = STPerformanceComparisonConfig.DEFAULT_BUDGET
    DELTA_POS_PLANNER = STPerformanceComparisonConfig.DELTA_POS_PLANNER
    DELTA_SET_PLANNER = STPerformanceComparisonConfig.DELTA_SET_PLANNER
    DELTA_SET_EPS1_PLANNER = STPerformanceComparisonConfig.DELTA_SET_EPS1_PLANNER
    DELTA_STATE_PLANNER = STPerformanceComparisonConfig.DELTA_STATE_PLANNER
    JOINT_SPEC = STPerformanceComparisonConfig.JOINT_SPEC
    JOINT_PLANNER = JOINT_SPEC.name
    SEARCH_PLANNERS = (*STPerformanceComparisonConfig.SEARCH_SPEC_BY_PLANNER, JOINT_PLANNER)
    BASELINE_PLANNERS = STPerformanceComparisonConfig.BASELINE_PLANNERS
    PLANNERS = (*SEARCH_PLANNERS, *STPerformanceComparisonConfig.BASELINE_PLANNERS)
    LEGEND_PLANNERS = (*SEARCH_PLANNERS, *STPerformanceComparisonConfig.BASELINE_PLANNERS)
    ST_RRT_PLANNERS = (
        STPerformanceComparisonConfig.ST_RRT_FIRST_PLANNER,
        STPerformanceComparisonConfig.ST_RRT_FINAL_PLANNER,
    )
    ST_RRT_LEGEND_ABBREV = "S1"
    ST_RRT_LEGEND_LABEL = "ST-RRT*"
    PARTIAL = False
    COST_MIN_SUCCESS_RATE = 0.33
    FIGURE_SIZE = (8.0, 7.7)
    AXIS_FONT_SIZE = 18
    LEGEND_FONT_SIZE = 14.5
    LEGEND_COLUMNS = 3
    CURVE_LINEWIDTH = 2.0
    AXIS_LINEWIDTH = 1.2
    GRID_LINEWIDTH = 0.9
    METRIC_LINEWIDTH = 1.1
    GROUP_ORDER = ("grid2d", "maze", "iris-2d")
    GROUP_LABELS = {
        "grid2d": "rand",
        "maze": "maze",
        "iris-2d": "iris",
    }
    METRIC_ROW_LABELS = ("", "")
    METRIC_X_LABELS = {
        "runtime": "runtime (s)",
        "cost": "solution cost",
    }
    SUCCESS_RATE_Y_LIMITS = (-0.03, 1.04)
    SUCCESS_RATE_TICKS = (0.0, 0.5, 1.0)
    SUCCESS_RATE_X_MIN_BY_GROUP = {"grid2d": 0.1, "maze": 0.3, "iris-2d": 0.01}
    SUCCESS_RATE_X_TICK_CANDIDATES = (1e-2, 1e0, 1e1, 1e2, 1e3)
    METRIC_VIOLIN_WIDTH = 0.85
    METRIC_SCATTER_SIZE = 20.0
    METRIC_SCATTER_ALPHA = 0.65
    METRIC_SCATTER_WIDTH_FRACTION = 0.65
    METHOD_REFERENCE_COLOR = PlotPalette.NEUTRAL_MID
    METHOD_REFERENCE_ALPHA = 0.28
    METHOD_REFERENCE_LINEWIDTH = 0.9
    METHOD_REFERENCE_LINESTYLE = "--"
    METRIC_X_TICK_LIMIT_PADDING = 1.12
    METRIC_X_TICKS_BY_GROUP = {
        "runtime": {
            "grid2d": (1e-2, 1, 1e2),
            "maze": (1e-1, 3, 100),
            "iris-2d": (1e-2, 1, 1e2),
        },
        "cost": {
            "grid2d": None,
            "maze": None,
            "iris-2d": None,
        },
    }
    METRIC_X_LIMITS_BY_GROUP = {
        "runtime": {
            "grid2d": (8e-3, 2e2),
            "maze": (2e-2, 2e2),
            "iris-2d": (1e-3, 2e2),
        },
        "cost": {
            "grid2d": None,
            "maze": None,
            "iris-2d": None,
        },
    }
    Y_TICK_LABEL_PAD = 2
    METHOD_STYLES = {
        JOINT_PLANNER: {
            "label": rf"$\delta_\text{{pos}} + \mathrm{{joint}}$ ($\varepsilon={JOINT_SPEC.epsilon:g}$)",
            "abbrev": "B4",
            "color": "#7B3294",
            "linestyle": "-",
            "marker": "P",
        },
        DELTA_POS_PLANNER: {
            "label": rf"$\delta_\text{{pos}} + h_\text{{max}}$ ($\varepsilon={STPerformanceComparisonConfig.DELTA_POS_SPEC.epsilon:g}$)",
            "abbrev": "B3",
            "color": "#009E73",
            "linestyle": "-",
            "marker": "o",
        },
        DELTA_SET_EPS1_PLANNER: {
            "label": r"$\delta_\text{set} + h_\text{max}$ ($\varepsilon=1$)",
            "abbrev": "B1",
            "color": "#D62728",
            "linestyle": "-",
            "marker": "*",
        },
        DELTA_STATE_PLANNER: {
            "label": rf"$\delta_\text{{state}} + h_\text{{max}}$ ($\varepsilon={STPerformanceComparisonConfig.DELTA_STATE_SPEC.epsilon:g}$)",
            "abbrev": "B2",
            "color": "#000000",
            "linestyle": "-",
            "marker": "s",
        },
        STPerformanceComparisonConfig.MICP_PLANNER: {
            "label": "MICP",
            "abbrev": "M1",
            "color": PlotPalette.ST_METHOD_COLORS["micp"],
            "linestyle": "--",
            "marker": "s",
        },
        STPerformanceComparisonConfig.MICP_ROUNDING_PLANNER: {
            "label": "MICP (g)",
            "abbrev": "M",
            "color": "#0072B2",
            "linestyle": "--",
            "marker": "D",
        },
        STPerformanceComparisonConfig.ST_RRT_FIRST_PLANNER: {
            "label": "ST-RRT* (first)",
            "abbrev": "R1",
            "color": "#E69F00",
            "linestyle": ":",
            "marker": "^",
        },
        STPerformanceComparisonConfig.ST_RRT_FINAL_PLANNER: {
            "label": "ST-RRT* (final)",
            "abbrev": "R2",
            "color": "#E69F00",
            "linestyle": ":",
            "marker": "P",
        },
        STPerformanceComparisonConfig.ST_RRT_MATCHED_PLANNER: {
            "label": "ST-RRT*(m)",
            "abbrev": "S2",
            "color": "#56B4E9",
            "linestyle": ":",
            "marker": "h",
        },
        STPerformanceComparisonConfig.ZETA_SIPP_PLANNER: {
            "label": "Zeta*-SIPP (r)",
            "abbrev": "Z1",
            "color": "#CC79A7",
            "linestyle": "-.",
            "marker": "v",
        },
        STPerformanceComparisonConfig.ZETA_SIPP_2R_PLANNER: {
            "label": "Zeta*-SIPP (2r)",
            "abbrev": "Z2",
            "color": "#8A8F00",
            "linestyle": "-.",
            "marker": "d",
        },
    }

    @classmethod
    def load_rows_by_planner(
        cls,
        results_root: Path,
        planners: Sequence[str] | None = None,
    ) -> Dict[str, list[Dict[str, object]]]:
        rows_by_planner: Dict[str, list[Dict[str, object]]] = {}
        for planner in cls.planner_names(planners):
            path = STResultStore.planner_output_path(results_root, planner)
            rows_by_planner[planner] = dedupe_result_rows(STResultStore.load_result_rows(path))
        return rows_by_planner

    @classmethod
    def planner_names(cls, requested: Sequence[str] | None = None) -> tuple[str, ...]:
        if requested is None:
            return cls.PLANNERS
        selected: list[str] = []
        for value in requested:
            token = str(value)
            if token.lower() == "all":
                return cls.PLANNERS
            names = ((cls.JOINT_PLANNER,) if token == cls.JOINT_PLANNER or token.lower() == "joint"
                     else STPerformanceComparisonConfig.planner_names((token,)))
            for name in names:
                if name not in selected:
                    selected.append(name)
        if not selected:
            raise ValueError("At least one ST performance-comparison planner must be selected.")
        return tuple(selected)

    @classmethod
    def planner_cli_help(cls) -> str:
        return STPerformanceComparisonConfig.planner_cli_help() + f", joint={cls.JOINT_PLANNER}"

    @classmethod
    def validate_plot_planners(cls, requested: Sequence[str] | None = None) -> tuple[str, ...]:
        return cls.planner_names(requested)

    @classmethod
    def select_budget(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        requested_budget: float | None,
    ) -> float:
        if requested_budget is not None:
            return float(requested_budget)
        common_budgets = cls.common_budgets(rows_by_planner)
        if common_budgets:
            return max(common_budgets)
        observed_budgets: set[float] = set()
        for rows in rows_by_planner.values():
            observed_budgets.update(cls.positive_budgets(rows))
        return max(observed_budgets) if observed_budgets else cls.DEFAULT_BUDGET

    @staticmethod
    def positive_budgets(rows: Sequence[Dict[str, object]]) -> set[float]:
        return {
            float(row["budget"])
            for row in rows
            if math.isfinite(float(row["budget"])) and float(row["budget"]) > 0.0
        }

    @classmethod
    def common_budgets(cls, rows_by_planner: Mapping[str, Sequence[Dict[str, object]]]) -> list[float]:
        budget_sets = [cls.positive_budgets(rows_by_planner.get(planner, [])) for planner in rows_by_planner]
        if not budget_sets or any(not budgets for budgets in budget_sets):
            return []
        return sorted(set.intersection(*budget_sets))

    @staticmethod
    def same_budget(row: Dict[str, object], budget: float) -> bool:
        return math.isclose(float(row["budget"]), float(budget))

    @classmethod
    def rows_by_instance(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
    ) -> Dict[str, Dict[str, Dict[str, object]]]:
        by_instance: Dict[str, Dict[str, Dict[str, object]]] = {}
        for planner, rows in rows_by_planner.items():
            for row in rows:
                if cls.same_budget(row, budget):
                    by_instance.setdefault(str(row["instance_id"]), {})[planner] = row
        return by_instance

    @classmethod
    def rows_by_group(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        group: str,
    ) -> Dict[str, list[Dict[str, object]]]:
        grouped_rows_by_planner: Dict[str, list[Dict[str, object]]] = {}
        for planner, rows in rows_by_planner.items():
            group_rows = [row for row in rows if str(row["source_domain"]) == group]
            if group_rows:
                grouped_rows_by_planner[planner] = group_rows
        return grouped_rows_by_planner

    @classmethod
    def common_instance_ids(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
    ) -> set[str]:
        common: set[str] = set()
        for instance_id, planner_rows in cls.rows_by_instance(rows_by_planner, budget).items():
            if all(planner in planner_rows for planner in rows_by_planner):
                common.add(instance_id)
        return common

    @classmethod
    def common_success_instance_ids(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planners: Sequence[str],
        metric: str | None = None,
    ) -> set[str]:
        _eligible_planners, common_instance_ids = cls.common_success_metric_context(
            rows_by_planner,
            budget,
            planners,
            metric=metric,
        )
        return common_instance_ids

    @classmethod
    def planner_success_rate(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner: str,
    ) -> float | None:
        rows = [
            row for row in rows_by_planner.get(planner, [])
            if cls.same_budget(row, budget)
        ]
        if not rows:
            return None
        successes = sum(
            1 for row in rows
            if bool(row["is_success"])
            and cls.finite_positive(row["runtime"]) is not None
            and float(row["runtime"]) <= budget
        )
        return successes / len(rows)

    @classmethod
    def common_success_metric_context(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planners: Sequence[str],
        metric: str | None = None,
    ) -> tuple[tuple[str, ...], set[str]]:
        eligible_planners = tuple(
            planner for planner in planners
            if metric != "cost" or (cls.planner_success_rate(rows_by_planner, budget, planner) or 0.0)
            >= cls.COST_MIN_SUCCESS_RATE
        )
        selected_rows_by_planner = {
            planner: rows_by_planner.get(planner, [])
            for planner in eligible_planners
        }
        if not selected_rows_by_planner or any(not rows for rows in selected_rows_by_planner.values()):
            return (), set()
        successful_ids_by_planner = {
            planner: {
                str(row["instance_id"])
                for row in rows
                if cls.same_budget(row, budget)
                and bool(row["is_success"])
                and (metric != "cost" or (cls.finite_positive(row["runtime"]) is not None
                                         and float(row["runtime"]) <= budget))
                and (metric is None or cls.finite_positive(row[metric]) is not None)
            }
            for planner, rows in selected_rows_by_planner.items()
        }
        return eligible_planners, set.intersection(
            *(successful_ids_by_planner[planner] for planner in eligible_planners)
        )

    @classmethod
    def planner_budget_rows(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner: str,
    ) -> list[Dict[str, object]]:
        instance_ids = None if cls.PARTIAL else cls.common_instance_ids(rows_by_planner, budget)
        return [
            row for row in rows_by_planner.get(planner, [])
            if cls.same_budget(row, budget) and (instance_ids is None or str(row["instance_id"]) in instance_ids)
        ]

    @staticmethod
    def finite_positive(value: object) -> float | None:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            return None
        return numeric

    @classmethod
    def effective_runtime(cls, row: Dict[str, object], budget: float) -> float:
        runtime = cls.finite_positive(row["runtime"])
        if runtime is None:
            return float(budget)
        return min(float(runtime), float(budget))

    @classmethod
    def success_rate_points(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner: str,
    ) -> tuple[list[float], list[float]]:
        rows = cls.planner_budget_rows(rows_by_planner, budget, planner)
        if not rows:
            return [], []
        events: dict[float, int] = {}
        for row in rows:
            measured_runtime = cls.finite_positive(row["runtime"])
            runtime = max(
                measured_runtime
                if bool(row["is_success"]) and measured_runtime is not None
                else cls.effective_runtime(row, budget),
                1e-3,
            )
            events.setdefault(runtime, 0)
            if bool(row["is_success"]) and runtime <= budget:
                events[runtime] += 1
        success_count = 0
        xs: list[float] = [1e-3]
        ys: list[float] = [0.0]
        for runtime in sorted(events):
            success_count += events[runtime]
            xs.append(runtime)
            ys.append(success_count / len(rows))
        if xs and xs[-1] < float(budget):
            xs.append(float(budget))
            ys.append(ys[-1])
        return xs, ys

    @classmethod
    def metric_values(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner: str,
        metric: str,
        instance_ids: set[str] | None = None,
    ) -> list[float]:
        if metric not in ("runtime", "cost"):
            raise ValueError(f"Unsupported metric {metric!r}.")
        values: list[float] = []
        if instance_ids is None:
            rows = cls.planner_budget_rows(rows_by_planner, budget, planner)
        else:
            rows = [
                row for row in rows_by_planner.get(planner, [])
                if cls.same_budget(row, budget) and str(row["instance_id"]) in instance_ids
            ]
        for row in rows:
            if not bool(row["is_success"]):
                continue
            value = cls.finite_positive(row[metric])
            if value is not None:
                values.append(float(value))
        return values

    @classmethod
    def success_rows_by_instance(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner: str,
        metric: str,
    ) -> dict[str, Dict[str, object]]:
        return {
            str(row["instance_id"]): row
            for row in rows_by_planner.get(planner, [])
            if cls.same_budget(row, budget)
            and bool(row["is_success"])
            and cls.finite_positive(row[metric]) is not None
        }


    @classmethod
    def summary_rows(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
    ) -> list[tuple[str, int, int, float, float]]:
        rows: list[tuple[str, int, int, float, float]] = []
        for planner in rows_by_planner:
            planner_rows = cls.planner_budget_rows(rows_by_planner, budget, planner)
            success_rows = [row for row in planner_rows if bool(row["is_success"])]
            runtimes = [float(row["runtime"]) for row in success_rows if cls.finite_positive(row["runtime"]) is not None]
            costs = [float(row["cost"]) for row in success_rows if cls.finite_positive(row["cost"]) is not None]
            rows.append(
                (
                    planner,
                    len(success_rows),
                    len(planner_rows),
                    statistics.median(runtimes) if runtimes else math.nan,
                    statistics.median(costs) if costs else math.nan,
                )
            )
        return rows

    @classmethod
    def plot_figure(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        output_prefix: Path,
        formats: Sequence[str],
        legend_font_size: float | None = None,
    ) -> None:
        fig, axes = plt.subplots(
            2,
            len(cls.GROUP_ORDER),
            figsize=(8.5, 7.5) if cls.PARTIAL else cls.FIGURE_SIZE,
            squeeze=False,
            gridspec_kw={"height_ratios": (1.0, 1.1)},
        )
        metric_planners = cls.cost_panel_planners(rows_by_planner)
        for col_idx, group in enumerate(cls.GROUP_ORDER):
            group_rows_by_planner = cls.rows_by_group(rows_by_planner, group)
            success_ax = axes[0][col_idx]
            cost_ax = axes[1][col_idx]
            cls.plot_success_panel(
                success_ax,
                group_rows_by_planner,
                budget,
                group,
                show_y_tick_labels=col_idx == 0,
            )
            cls.plot_metric_absolute_panel(
                cost_ax,
                group_rows_by_planner,
                budget,
                "cost",
                metric_planners,
                group,
                show_method_tick_labels=col_idx == 0,
            )
            title = cls.GROUP_LABELS[group]
            if cls.PARTIAL:
                counts = [f"{cls.performance_method_abbrev(p)}={len(cls.planner_budget_rows(group_rows_by_planner, budget, p))}"
                          for p in cls.success_panel_planners(group_rows_by_planner)]
                title += "\ncompleted n: " + ", ".join(counts[:3])
                title += "\n" + ", ".join(counts[3:6]) + "\n" + ", ".join(counts[6:])
            success_ax.set_title(title, fontsize=8.5 if cls.PARTIAL else cls.AXIS_FONT_SIZE)
        cls.add_method_legend(fig, rows_by_planner, legend_font_size=legend_font_size)
        cls.add_metric_row_labels(axes)
        for ax in axes.flat:
            for spine in ax.spines.values():
                spine.set_linewidth(cls.AXIS_LINEWIDTH)
            ax.tick_params(which="major", width=cls.AXIS_LINEWIDTH, length=4.5)
            ax.tick_params(which="minor", width=cls.AXIS_LINEWIDTH, length=2.5)
        if cls.PARTIAL:
            fig.text(.52, .84, "PARTIAL: completed trials only; planner samples differ", ha="center", fontsize=10)
            fig.text(.52, .005, "Partial results; sample counts vary by planner.", ha="center", fontsize=8)
        fig.set_layout_engine(
            "constrained",
            rect=(0.0, 0.025 if cls.PARTIAL else 0.0, 1.0, 0.755 if cls.PARTIAL else 0.85),
            w_pad=0.025, h_pad=0.035, wspace=0.02, hspace=0.02,
        )
        _save_figure(fig, output_prefix, formats)

    @classmethod
    def add_metric_row_labels(cls, axes: np.ndarray) -> None:
        for row_idx, label in enumerate(cls.METRIC_ROW_LABELS):
            axes[row_idx][0].set_ylabel(label, fontsize=cls.AXIS_FONT_SIZE)

    @classmethod
    def plot_success_panel(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        group: str,
        show_y_tick_labels: bool = True,
    ) -> None:
        plotted_xs: list[float] = []
        for planner in cls.success_panel_planners(rows_by_planner):
            xs, ys = cls.success_rate_points(rows_by_planner, budget, planner)
            if not xs:
                continue
            plotted_xs.extend(xs)
            style = cls.performance_plot_style(planner)
            ax.step(
                xs,
                ys,
                label=cls.method_legend_label(planner),
                color=str(style["color"]),
                linestyle=style["linestyle"],
                linewidth=cls.CURVE_LINEWIDTH,
                where="post",
            )
        ax.set_xscale("log")
        ax.set_box_aspect(1)
        x_upper_limit = 1.08 * max(float(budget), max(plotted_xs, default=float(budget)))
        x_lower_limit = cls.SUCCESS_RATE_X_MIN_BY_GROUP[group]
        ax.set_xlim(left=x_lower_limit, right=x_upper_limit)
        ax.set_ylim(*cls.SUCCESS_RATE_Y_LIMITS)
        ax.set_xlabel("runtime (s)", fontsize=cls.AXIS_FONT_SIZE)
        ax.set_yticks(cls.SUCCESS_RATE_TICKS)
        ax.yaxis.set_major_formatter(FuncFormatter(cls.success_rate_tick_label))
        ax.xaxis.set_major_locator(FixedLocator(cls.success_rate_x_ticks(budget, x_lower_limit)))
        ax.xaxis.set_major_formatter(FuncFormatter(cls.success_runtime_tick_label))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="both", labelsize=cls.AXIS_FONT_SIZE)
        ax.grid(alpha=0.24, linewidth=cls.GRID_LINEWIDTH)
        cls.rotate_y_tick_labels(ax, show_tick_labels=show_y_tick_labels)

    @classmethod
    def add_method_legend(
        cls,
        fig: plt.Figure,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        legend_font_size: float | None = None,
    ) -> None:
        fontsize = cls.LEGEND_FONT_SIZE if legend_font_size is None else float(legend_font_size)
        planners = cls.legend_planners(rows_by_planner)
        fig.legend(
            handles=[cls.legend_handle(planner) for planner in planners],
            labels=[cls.method_legend_label(planner) for planner in planners],
            frameon=False,
            fontsize=fontsize,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=cls.LEGEND_COLUMNS,
            handlelength=1.5,
            columnspacing=0.5,
            labelspacing=0.3,
            borderaxespad=0.0,
        )

    @classmethod
    def legend_handle(cls, planner: str) -> Line2D:
        style = cls.performance_plot_style(planner)
        return Line2D(
            [0],
            [0],
            color=str(style["color"]),
            linestyle=style["linestyle"],
            linewidth=cls.CURVE_LINEWIDTH,
        )

    @classmethod
    def method_legend_label(cls, planner: str) -> str:
        if planner in cls.ST_RRT_PLANNERS:
            return f"{cls.ST_RRT_LEGEND_ABBREV}: {cls.ST_RRT_LEGEND_LABEL}"
        style = cls.METHOD_STYLES[planner]
        return f'{style["abbrev"]}: {style["label"]}'

    @classmethod
    def method_abbrev(cls, planner: str) -> str:
        return str(cls.METHOD_STYLES[planner]["abbrev"])

    @classmethod
    def performance_method_abbrev(cls, planner: str) -> str:
        if planner in cls.ST_RRT_PLANNERS:
            return cls.ST_RRT_LEGEND_ABBREV
        return cls.method_abbrev(planner)

    @classmethod
    def performance_plot_style(cls, planner: str) -> Mapping[str, object]:
        if planner in cls.ST_RRT_PLANNERS:
            return cls.METHOD_STYLES[STPerformanceComparisonConfig.ST_RRT_FIRST_PLANNER]
        return cls.METHOD_STYLES[planner]

    @classmethod
    def legend_planners(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
    ) -> tuple[str, ...]:
        planners: list[str] = []
        has_st_rrt = any(planner in rows_by_planner for planner in cls.ST_RRT_PLANNERS)
        added_st_rrt = False
        for planner in cls.LEGEND_PLANNERS:
            if planner in cls.ST_RRT_PLANNERS:
                if has_st_rrt and not added_st_rrt:
                    planners.append(
                        STPerformanceComparisonConfig.ST_RRT_FIRST_PLANNER
                        if STPerformanceComparisonConfig.ST_RRT_FIRST_PLANNER in rows_by_planner
                        else STPerformanceComparisonConfig.ST_RRT_FINAL_PLANNER
                    )
                    added_st_rrt = True
                continue
            if planner in rows_by_planner:
                planners.append(planner)
        return tuple(planners)

    @classmethod
    def success_panel_planners(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
    ) -> tuple[str, ...]:
        planners: list[str] = []
        has_strrt_first = STPerformanceComparisonConfig.ST_RRT_FIRST_PLANNER in rows_by_planner
        for planner in rows_by_planner:
            if planner == STPerformanceComparisonConfig.ST_RRT_FINAL_PLANNER and has_strrt_first:
                continue
            planners.append(planner)
        return tuple(planners)

    @classmethod
    def cost_panel_planners(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
    ) -> tuple[str, ...]:
        return tuple(
            planner for planner in cls.LEGEND_PLANNERS
            if planner in rows_by_planner
            and planner != STPerformanceComparisonConfig.ST_RRT_FIRST_PLANNER
        )

    @classmethod
    def rotate_y_tick_labels(cls, ax: plt.Axes, show_tick_labels: bool = True) -> None:
        ax.tick_params(
            axis="y",
            labelrotation=90,
            labelsize=cls.AXIS_FONT_SIZE,
            pad=cls.Y_TICK_LABEL_PAD,
            labelleft=show_tick_labels,
        )
        for label in [*ax.get_yticklabels(), *ax.get_yticklabels(minor=True)]:
            label.set_rotation(90)
            label.set_verticalalignment("center")
            label.set_fontsize(cls.AXIS_FONT_SIZE)


    @staticmethod
    def success_rate_tick_label(value: float, _position: int) -> str:
        if math.isclose(float(value), 0.0):
            return "0%"
        return f"{float(value):.0%}"

    @classmethod
    def success_rate_x_ticks(cls, budget: float, x_lower_limit: float) -> tuple[float, ...]:
        budget_value = float(budget)
        ticks = [x_lower_limit] + [
            tick for tick in cls.SUCCESS_RATE_X_TICK_CANDIDATES
            if x_lower_limit < tick <= budget_value
        ]
        if budget_value > 0.0 and not any(math.isclose(tick, budget_value) for tick in ticks):
            ticks.append(budget_value)
        if len(ticks) > 1 and ticks[-1] / ticks[-2] < 4.0:
            ticks.pop(-2)
        return tuple(ticks)

    @staticmethod
    def success_runtime_tick_label(value: float, _position: int) -> str:
        numeric = float(value)
        if numeric <= 0.0 or not math.isfinite(numeric):
            return ""
        if numeric >= 1.0 and math.isclose(numeric, round(numeric)):
            return str(int(round(numeric)))
        return f"{numeric:g}"

    @staticmethod
    def metric_tick_label(value: float, _position: int) -> str:
        if float(value) < 0.0 or not math.isfinite(float(value)):
            return ""
        numeric = float(value)
        return f"{numeric:g}"

    @classmethod
    def plot_metric_absolute_panel(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        metric: str,
        planners: Sequence[str],
        group: str,
        show_method_tick_labels: bool = True,
    ) -> None:
        eligible_planners, common_ids = cls.common_success_metric_context(
            rows_by_planner, budget, planners, metric=metric)
        positions = np.arange(1, len(planners) + 1)
        plotted_values: list[float] = []
        for position, planner in zip(positions, planners):
            if planner not in eligible_planners:
                ax.text(.03, float(position), rf"$<{100 * cls.COST_MIN_SUCCESS_RATE:g}\%$ success rate",
                        transform=ax.get_yaxis_transform(), va="center", fontsize=14,
                        color="black")
                continue
            values = cls.metric_values(rows_by_planner, budget, planner, metric, instance_ids=common_ids)
            if not values:
                continue
            plotted_values.extend(values)
            style = cls.performance_plot_style(planner)
            cls.draw_metric_violin(ax, float(position), values, str(style["color"]))
            cls.plot_metric_scatter_points(
                ax,
                float(position),
                values,
                str(style["color"]),
                orientation="horizontal",
            )
        nonpositive_values = [value for value in plotted_values if value <= 0.0]
        if nonpositive_values:
            raise ValueError(f"{metric.capitalize()} log scale requires all plotted values to be positive.")
        if metric == "runtime":
            ax.set_xscale("log")
        ax.autoscale(enable=True, axis="x", tight=False)
        if metric == "cost":
            ax.margins(x=0.05)
            ax.set_xlim(left=max(0.0, ax.get_xlim()[0]))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=(1, 2, 2.5, 5, 10)))
        cls.set_metric_x_ticks(ax, metric, group)
        ax.xaxis.set_major_formatter(FuncFormatter(cls.metric_tick_label))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(cls.METRIC_X_LABELS[metric], fontsize=cls.AXIS_FONT_SIZE)
        ax.set_ylim(len(planners) + 0.55, 0.45)
        ax.set_yticks(list(positions))
        ax.set_yticklabels(
            [cls.performance_method_abbrev(planner) for planner in planners],
            fontsize=8.5,
        )
        cls.draw_method_reference_lines(ax, positions)
        ax.tick_params(axis="x", labelsize=cls.AXIS_FONT_SIZE)
        ax.tick_params(
            axis="y",
            labelsize=cls.AXIS_FONT_SIZE,
            pad=cls.Y_TICK_LABEL_PAD,
            labelleft=show_method_tick_labels,
        )
        ax.grid(axis="x", alpha=0.25, linewidth=cls.GRID_LINEWIDTH)


    @classmethod
    def draw_method_reference_lines(cls, ax: plt.Axes, positions: Sequence[float]) -> None:
        for position in positions:
            ax.axhline(
                position,
                color=cls.METHOD_REFERENCE_COLOR,
                linestyle=cls.METHOD_REFERENCE_LINESTYLE,
                linewidth=cls.METHOD_REFERENCE_LINEWIDTH,
                alpha=cls.METHOD_REFERENCE_ALPHA,
                zorder=1,
            )

    @classmethod
    def set_metric_x_ticks(cls, ax: plt.Axes, metric: str, group: str) -> None:
        ticks = cls.METRIC_X_TICKS_BY_GROUP.get(metric, {}).get(group)
        if ticks is not None:
            ax.xaxis.set_major_locator(FixedLocator(ticks))
        limits = cls.METRIC_X_LIMITS_BY_GROUP.get(metric, {}).get(group)
        if limits is not None:
            cls.set_metric_x_limits(ax, limits)
        elif ticks is not None:
            cls.expose_metric_x_ticks(ax, ticks)

    @classmethod
    def set_metric_x_limits(cls, ax: plt.Axes, limits: Sequence[float]) -> None:
        if len(limits) != 2:
            raise ValueError(f"Metric x limits must contain exactly two values, got {limits!r}.")
        left, right = (float(limits[0]), float(limits[1]))
        if not math.isfinite(left) or not math.isfinite(right) or left >= right:
            raise ValueError(f"Invalid metric x limits {limits!r}.")
        if ax.get_xscale() == "log" and (left <= 0.0 or right <= 0.0):
            raise ValueError(f"Log-scale metric x limits must be positive, got {limits!r}.")
        ax.set_xlim(left=left, right=right)

    @classmethod
    def expose_metric_x_ticks(cls, ax: plt.Axes, ticks: Sequence[float]) -> None:
        finite_ticks = [
            float(tick)
            for tick in ticks
            if math.isfinite(float(tick))
        ]
        if not finite_ticks:
            return
        x_low, x_high = ax.get_xlim()
        if ax.get_xscale() == "log":
            positive_ticks = [tick for tick in finite_ticks if tick > 0.0]
            if not positive_ticks:
                return
            padding = cls.METRIC_X_TICK_LIMIT_PADDING
            ax.set_xlim(
                left=min(float(x_low), min(positive_ticks) / padding),
                right=max(float(x_high), max(positive_ticks) * padding),
            )
            return
        ax.set_xlim(
            left=min(float(x_low), min(finite_ticks)),
            right=max(float(x_high), max(finite_ticks)),
        )

    @classmethod
    def draw_metric_violin(
        cls,
        ax: plt.Axes,
        position: float,
        values: Sequence[float],
        color: str,
    ) -> None:
        ViolinStyle.draw(ax, position, values, color, width=cls.METRIC_VIOLIN_WIDTH,
                         linewidth=cls.METRIC_LINEWIDTH)

    @classmethod
    def plot_metric_scatter_points(
        cls,
        ax: plt.Axes,
        position: float,
        values: Sequence[float],
        color: str,
        orientation: str = "vertical",
    ) -> None:
        jittered_positions = cls.metric_scatter_x_positions(position, len(values))
        if orientation == "vertical":
            ax.scatter(
                jittered_positions,
                values,
                s=cls.METRIC_SCATTER_SIZE,
                marker="o",
                facecolors=color,
                edgecolors="white",
                linewidths=0.35,
                alpha=cls.METRIC_SCATTER_ALPHA,
                zorder=3,
            )
            return
        if orientation == "horizontal":
            ax.scatter(
                values,
                jittered_positions,
                s=cls.METRIC_SCATTER_SIZE,
                marker="o",
                facecolors=color,
                edgecolors="white",
                linewidths=0.35,
                alpha=cls.METRIC_SCATTER_ALPHA,
                zorder=3,
            )
            return
        raise ValueError(f"Unsupported scatter orientation {orientation!r}.")

    @classmethod
    def metric_scatter_x_positions(cls, position: float, count: int) -> list[float]:
        if count <= 1:
            return [float(position)] * count
        span = cls.METRIC_VIOLIN_WIDTH * cls.METRIC_SCATTER_WIDTH_FRACTION
        return [
            float(position) + span * (idx / (count - 1) - 0.5)
            for idx in range(count)
        ]

    @classmethod
    def print_summary(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
    ) -> None:
        for planner, success_count, total_count, median_runtime, median_cost in cls.summary_rows(rows_by_planner, budget):
            label = cls.method_abbrev(planner)
            runtime_text = "nan" if math.isnan(median_runtime) else f"{median_runtime:.3f}"
            cost_text = "nan" if math.isnan(median_cost) else f"{median_cost:.3f}"
            print(
                f"{label}: success={success_count}/{total_count}, "
                f"median_runtime={runtime_text}s, median_cost={cost_text}"
            )


class STPerformanceComparisonCLI:
    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Plot the ST-planning performance comparison against external baselines."
        )
        parser.add_argument("--results-root", type=Path, default=STPerformanceComparisonReport.DEFAULT_RESULTS_ROOT)
        parser.add_argument("--output-prefix", type=Path, default=STPerformanceComparisonReport.DEFAULT_OUTPUT_PREFIX)
        parser.add_argument("--budget", type=float, default=None)
        parser.add_argument("--partial", action="store_true", help="Include every completed trial, with per-planner sample counts")
        parser.add_argument("--format", dest="formats", nargs="+", default=("pdf", "png"))
        parser.add_argument("--legend-font-size", type=float, default=STPerformanceComparisonReport.LEGEND_FONT_SIZE)
        parser.add_argument(
            "--planner",
            nargs="+",
            default=None,
            metavar="PLANNER",
            help=STPerformanceComparisonReport.planner_cli_help(),
        )
        args = parser.parse_args()
        try:
            requested_planners = ("all",) if args.planner is None else args.planner
            args.planner_names = STPerformanceComparisonReport.validate_plot_planners(requested_planners)
        except ValueError as exc:
            parser.error(str(exc))
        return args

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        STPerformanceComparisonReport.PARTIAL = args.partial
        planner_names = tuple(args.planner_names)
        print(f"Selected planners: {', '.join(planner_names)}")
        rows_by_planner = STPerformanceComparisonReport.load_rows_by_planner(
            Path(args.results_root),
            planners=planner_names,
        )
        budget = STPerformanceComparisonReport.select_budget(rows_by_planner, args.budget)
        group_common_counts = {
            group: len((STPerformanceComparisonReport.rows_by_instance if args.partial else STPerformanceComparisonReport.common_instance_ids)(
                STPerformanceComparisonReport.rows_by_group(rows_by_planner, group),
                budget,
            ))
            for group in STPerformanceComparisonReport.GROUP_ORDER
        }
        if not any(group_common_counts.values()):
            raise ValueError(f"No common ST performance-comparison rows found at budget={budget:g}.")
        count_text = ", ".join(
            f"{STPerformanceComparisonReport.GROUP_LABELS[group]}={count}"
            for group, count in group_common_counts.items()
        )
        print(f"Plotting budget={budget:g} over grouped {'available' if args.partial else 'common'} ST-planning instances: {count_text}.")
        STPerformanceComparisonReport.print_summary(rows_by_planner, budget)
        STPerformanceComparisonReport.plot_figure(
            rows_by_planner,
            budget,
            Path(args.output_prefix),
            tuple(args.formats),
            legend_font_size=float(args.legend_font_size),
        )


if __name__ == "__main__":
    STPerformanceComparisonCLI.main()
