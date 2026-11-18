from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

from experiments.reports.style import configure_matplotlib_for_latex

configure_matplotlib_for_latex(backend="Agg")

from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter, NullFormatter

from experiments.results import STResultStore
from experiments.reports.common import BoxplotStyle, PlotPalette, dedupe_result_rows, plt


class STHeuristicAblationGroupReport:
    DEFAULT_RESULTS_ROOT = Path("data/exp_res/st_planning/heuristic_ablation")
    DEFAULT_OUTPUT_PREFIX = Path("latex/figs/heur_ablation")
    DEFAULT_TABLE_OUTPUT = Path("latex/tables/st_heuristic_ablation_group_table.tex")
    DEFAULT_BUDGET = 600.0

    FIGURE_PLANNERS = (
        "Search(GUB)",
        "Search(h_mot+GUB)",
        "Search(h_tri+GUB)",
        "Search(h_tab+GUB)",
        "Search(h_max+GUB)",
    )
    PLANNER_SHORT_LABELS = {
        "Search(GUB)": r"$h_\text{zero}$",
        "Search(h_mot+GUB)": r"$h_\text{mot}$",
        "Search(h_tri+GUB)": r"$h_\text{tri}$",
        "Search(h_tab+GUB)": r"$h_\text{tab}$",
        "Search(h_max+GUB)": r"$h_\text{max}$",
    }
    GROUP_ORDER = ("maze", "grid2d", "iris2d")
    GROUP_LABELS = {
        "maze": "maze",
        "grid2d": "grid2d",
        "iris2d": "Iris/iris2d",
    }
    GROUP_BY_SOURCE_DOMAIN = {
        "maze": "maze",
        "grid2d": "grid2d",
        "iris-2d": "iris2d",
        "iris2d": "iris2d",
    }
    FIGURE_GROUP_ORDER = ("grid2d", "maze", "iris2d")
    FIGURE_GROUP_LABELS = {
        "grid2d": r"rand",
        "maze": r"maze",
        "iris2d": r"iris",
    }
    METRIC_FIELDS = {
        "expanded": "num_expanded_nodes",
        "runtime": "runtime",
    }
    METRIC_LABELS = {
        "expanded": "expanded nodes",
        "runtime": "runtime (s)",
    }
    FIGURE_X_LABELS = METRIC_LABELS
    FIGURE_METRIC_ORDER = ("runtime", "expanded")
    FIGURE_HEURISTIC_COLORS = {
        "Search(GUB)": PlotPalette.heuristic_color("h_zero"),
        "Search(h_mot+GUB)": PlotPalette.heuristic_color("h_mot"),
        "Search(h_tri+GUB)": PlotPalette.heuristic_color("h_tri"),
        "Search(h_tab+GUB)": PlotPalette.heuristic_color("h_tab"),
        "Search(h_max+GUB)": PlotPalette.heuristic_color("h_max"),
    }
    FIGURE_Y_TICKS = {
        "runtime": {
            "grid2d": (1e-1, 1e0, 1e1, 1e2),
            "maze": (1e-1, 1e0, 1e1, 1e2),
            "iris2d": (1e-1, 1e0, 1e1, 1e2),
        },
        "expanded": {
            "grid2d": (1e1, 1e2, 1e3, 1e4, 1e5),
            "maze": (1e1, 1e2, 1e3, 1e4, 1e5),
            "iris2d": (1e1, 1e2, 1e3, 1e4, 1e5),
        },
    }
    # Set a tuple to pin one subplot; None keeps that subplot data-derived.
    FIGURE_Y_LIMITS = {
        "runtime": {
            "grid2d": None,
            "maze": None,
            "iris2d": None,
        },
        "expanded": {
            "grid2d": None,
            "maze": None,
            "iris2d": None,
        },
    }
    FIGURE_Y_SCALES = {
        "runtime": "log",
        "expanded": "log",
    }
    FIGURE_BOXPLOT_WIDTH = BoxplotStyle.DEFAULT_WIDTH
    FIGURE_BOXPLOT_WHIS = BoxplotStyle.DEFAULT_WHIS
    FIGURE_ROW_HEIGHT_RATIOS: tuple[float, ...] | None = None
    FIGURE_SHOW_SCATTER_POINTS = True
    FIGURE_SCATTER_WIDTH = 0.24
    FIGURE_SCATTER_SIZE = 12
    FIGURE_SCATTER_ALPHA = 0.58
    FIGURE_BOXPLOT_ORIENTATION = "horizontal"
    FIGURE_HEURISTIC_REFERENCE_COLOR = PlotPalette.NEUTRAL_MID
    FIGURE_HEURISTIC_REFERENCE_ALPHA = 0.28
    FIGURE_HEURISTIC_REFERENCE_LINEWIDTH = 0.7
    FIGURE_HEURISTIC_REFERENCE_LINESTYLE = "--"
    FIGURE_SIZE = (6, 4.25)
    FIGURE_COLUMN_TEXT_SIZE = 14

    @classmethod
    def load_rows_by_planner(cls, results_root: Path) -> Dict[str, list[Dict[str, object]]]:
        rows_by_planner: Dict[str, list[Dict[str, object]]] = {}
        for planner in cls.FIGURE_PLANNERS:
            path = STResultStore.planner_output_path(results_root, planner)
            rows_by_planner[planner] = dedupe_result_rows(STResultStore.load_result_rows(path))
        return rows_by_planner

    @classmethod
    def select_budget(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        requested_budget: float | None,
    ) -> float:
        if requested_budget is not None:
            return float(requested_budget)
        budgets_by_planner: list[set[float]] = []
        observed_budgets: set[float] = set()
        for planner in cls.FIGURE_PLANNERS:
            planner_budgets: set[float] = set()
            for row in rows_by_planner.get(planner, []):
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
    def _same_budget(row: Dict[str, object], budget: float) -> bool:
        return math.isclose(float(row["budget"]), budget)

    @classmethod
    def _row_group(cls, row: Dict[str, object]) -> str:
        source_domain = str(row.get("source_domain", ""))
        if source_domain in cls.GROUP_BY_SOURCE_DOMAIN:
            return cls.GROUP_BY_SOURCE_DOMAIN[source_domain]
        return str(row["group"])

    @classmethod
    def _rows_for_group(
        cls,
        rows: Sequence[Dict[str, object]],
        group: str,
        budget: float,
    ) -> list[Dict[str, object]]:
        return [
            row
            for row in rows
            if cls._row_group(row) == group and cls._same_budget(row, budget)
        ]

    @staticmethod
    def _finite_metric_value(row: Dict[str, object], metric_field: str) -> float | None:
        value = float(row[metric_field])
        if not math.isfinite(value) or value <= 0.0:
            return None
        return value

    @classmethod
    def _success_rows_by_instance(
        cls,
        rows: Sequence[Dict[str, object]],
        metric_field: str,
    ) -> Dict[str, Dict[str, object]]:
        rows_by_instance: Dict[str, Dict[str, object]] = {}
        for row in rows:
            if not bool(row["is_success"]):
                continue
            if cls._finite_metric_value(row, metric_field) is None:
                continue
            rows_by_instance[str(row["instance_id"])] = row
        return rows_by_instance

    @staticmethod
    def _axis_limits(values: Sequence[float]) -> tuple[float, float]:
        finite_values = [value for value in values if math.isfinite(value) and value > 0.0]
        if not finite_values:
            return 1.0, 10.0
        low = min(finite_values)
        high = max(finite_values)
        if math.isclose(low, high):
            return max(low / 2.0, 1e-3), high * 2.0
        return max(low / 1.6, 1e-3), high * 1.6

    @classmethod
    def _figure_y_limits(
        cls,
        values: Sequence[float],
        metric_name: str,
        group: str,
    ) -> tuple[float, float]:
        limits = cls.FIGURE_Y_LIMITS[metric_name][group]
        if limits is None:
            return cls._axis_limits(values)
        y_low, y_high = limits
        if y_low <= 0.0 or y_high <= y_low:
            raise ValueError(
                f"Invalid y-limits for {metric_name}/{group}: "
                f"expected 0 < low < high, got {limits}"
            )
        return float(y_low), float(y_high)

    @classmethod
    def _metric_values_for_instances(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        planner: str,
        group: str,
        budget: float,
        metric_field: str,
        instance_ids: set[str],
    ) -> list[float]:
        rows = cls._success_rows_by_instance(
            cls._rows_for_group(rows_by_planner.get(planner, []), group, budget),
            metric_field,
        )
        values: list[float] = []
        for instance_id in sorted(instance_ids):
            row = rows.get(instance_id)
            if row is None:
                continue
            value = cls._finite_metric_value(row, metric_field)
            if value is not None:
                values.append(value)
        return values

    @classmethod
    def _boxplot_values(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        group: str,
        budget: float,
        metric_field: str,
    ) -> list[list[float]]:
        common_instance_ids = cls._common_success_instance_ids(
            rows_by_planner,
            group,
            budget,
            metric_field,
            cls.FIGURE_PLANNERS,
        )
        return [
            cls._metric_values_for_instances(
                rows_by_planner,
                planner,
                group,
                budget,
                metric_field,
                common_instance_ids,
            )
            for planner in cls.FIGURE_PLANNERS
        ]


    @classmethod
    def _boxplot_gridspec_kwargs(cls) -> dict[str, tuple[float, ...]] | None:
        if cls.FIGURE_ROW_HEIGHT_RATIOS is None:
            return None
        if len(cls.FIGURE_ROW_HEIGHT_RATIOS) != len(cls.FIGURE_METRIC_ORDER):
            raise ValueError(
                "FIGURE_ROW_HEIGHT_RATIOS must have one entry per figure row: "
                f"got {cls.FIGURE_ROW_HEIGHT_RATIOS!r} for {cls.FIGURE_METRIC_ORDER!r}."
            )
        return {"height_ratios": cls.FIGURE_ROW_HEIGHT_RATIOS}

    @classmethod
    def _plot_boxplot_axis(
        cls,
        ax: plt.Axes,
        values_by_heuristic: Sequence[Sequence[float]],
        metric_name: str,
        group: str,
        y_low: float,
        y_high: float,
        row_idx: int,
        col_idx: int,
    ) -> None:
        if cls.FIGURE_BOXPLOT_ORIENTATION == "horizontal":
            cls._plot_horizontal_boxplot_axis(
                ax,
                values_by_heuristic,
                metric_name,
                group,
                y_low,
                y_high,
                row_idx,
                col_idx,
            )
            return
        if cls.FIGURE_BOXPLOT_ORIENTATION != "vertical":
            raise ValueError(f"Unsupported boxplot orientation {cls.FIGURE_BOXPLOT_ORIENTATION!r}.")

        positions = tuple(range(1, len(cls.FIGURE_PLANNERS) + 1))
        nonempty = [
            (position, planner, list(values))
            for position, planner, values in zip(positions, cls.FIGURE_PLANNERS, values_by_heuristic)
            if values
        ]
        if nonempty:
            for position, planner, values in nonempty:
                BoxplotStyle.draw(
                    ax,
                    [values],
                    positions=[position],
                    widths=cls.FIGURE_BOXPLOT_WIDTH,
                    color=cls.FIGURE_HEURISTIC_COLORS[planner],
                    whis=cls.FIGURE_BOXPLOT_WHIS,
                )
                if cls.FIGURE_SHOW_SCATTER_POINTS:
                    cls._plot_boxplot_scatter_points(
                        ax,
                        position,
                        values,
                        cls.FIGURE_HEURISTIC_COLORS[planner],
                        orientation="vertical",
                    )
        else:
            ax.text(
                0.5,
                0.5,
                "no common\nsuccesses",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9.5,
                color=PlotPalette.NEUTRAL_DARK,
            )

        ax.set_xlim(0.35, len(cls.FIGURE_PLANNERS) + 0.65)
        ax.set_ylim(y_low, y_high)
        y_scale = cls.FIGURE_Y_SCALES.get(metric_name, "log")
        if y_scale == "log":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(cls._format_log_tick))
            ax.yaxis.set_minor_formatter(NullFormatter())
        elif y_scale == "linear":
            ax.set_yscale("linear")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        else:
            raise ValueError(f"Unsupported y-scale for {metric_name}: {y_scale!r}")
        ax.yaxis.set_major_locator(FixedLocator(cls.FIGURE_Y_TICKS[metric_name][group]))
        ax.grid(alpha=0.22, linewidth=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [cls.PLANNER_SHORT_LABELS[planner] for planner in cls.FIGURE_PLANNERS],
            fontsize=cls.FIGURE_COLUMN_TEXT_SIZE,
        )
        if row_idx == 0:
            ax.set_title(cls.FIGURE_GROUP_LABELS[group], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=5)
        if col_idx == 0:
            ax.set_ylabel(cls.METRIC_LABELS[metric_name], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=cls.FIGURE_COLUMN_TEXT_SIZE, labelrotation=90, pad=1.5)
        ax.tick_params(axis="x", labelsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=1.5)

    @classmethod
    def _plot_horizontal_boxplot_axis(
        cls,
        ax: plt.Axes,
        values_by_heuristic: Sequence[Sequence[float]],
        metric_name: str,
        group: str,
        x_low: float,
        x_high: float,
        row_idx: int,
        col_idx: int,
    ) -> None:
        positions = tuple(range(1, len(cls.FIGURE_PLANNERS) + 1))
        nonempty = [
            (position, planner, list(values))
            for position, planner, values in zip(positions, cls.FIGURE_PLANNERS, values_by_heuristic)
            if values
        ]
        if nonempty:
            for position, planner, values in nonempty:
                BoxplotStyle.draw(
                    ax,
                    [values],
                    positions=[position],
                    widths=cls.FIGURE_BOXPLOT_WIDTH,
                    color=cls.FIGURE_HEURISTIC_COLORS[planner],
                    whis=cls.FIGURE_BOXPLOT_WHIS,
                    orientation="horizontal",
                )
                if cls.FIGURE_SHOW_SCATTER_POINTS:
                    cls._plot_boxplot_scatter_points(
                        ax,
                        position,
                        values,
                        cls.FIGURE_HEURISTIC_COLORS[planner],
                        orientation="horizontal",
                    )
        else:
            ax.text(
                0.5,
                0.5,
                "no common\nsuccesses",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9.5,
                color=PlotPalette.NEUTRAL_DARK,
            )

        ax.set_xlim(x_low, x_high)
        ax.set_ylim(len(cls.FIGURE_PLANNERS) + 0.65, 0.35)
        x_scale = cls.FIGURE_Y_SCALES.get(metric_name, "log")
        if x_scale == "log":
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(cls._format_log_tick))
            ax.xaxis.set_minor_formatter(NullFormatter())
        elif x_scale == "linear":
            ax.set_xscale("linear")
            ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        else:
            raise ValueError(f"Unsupported x-scale for {metric_name}: {x_scale!r}")
        ax.xaxis.set_major_locator(FixedLocator(cls.FIGURE_Y_TICKS[metric_name][group]))
        cls._draw_heuristic_reference_lines(ax, positions)
        ax.grid(axis="x", alpha=0.22, linewidth=0.7)
        ax.set_yticks(positions)
        if row_idx == 0:
            ax.set_title(cls.FIGURE_GROUP_LABELS[group], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=5)
        ax.set_xlabel(cls.FIGURE_X_LABELS[metric_name], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE)
        ax.set_yticklabels(
            [cls.PLANNER_SHORT_LABELS[planner] for planner in cls.FIGURE_PLANNERS],
            fontproperties=ax.xaxis.label.get_fontproperties(),
        )
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=1.5)
        ax.tick_params(
            axis="y",
            labelsize=cls.FIGURE_COLUMN_TEXT_SIZE,
            pad=1.5,
            labelleft=col_idx == 0,
        )

    @classmethod
    def _draw_heuristic_reference_lines(cls, ax: plt.Axes, positions: Sequence[int]) -> None:
        for position in positions:
            ax.axhline(
                position,
                color=cls.FIGURE_HEURISTIC_REFERENCE_COLOR,
                linestyle=cls.FIGURE_HEURISTIC_REFERENCE_LINESTYLE,
                linewidth=cls.FIGURE_HEURISTIC_REFERENCE_LINEWIDTH,
                alpha=cls.FIGURE_HEURISTIC_REFERENCE_ALPHA,
                zorder=1,
            )

    @classmethod
    def _plot_boxplot_scatter_points(
        cls,
        ax: plt.Axes,
        position: int,
        values: Sequence[float],
        color: str,
        orientation: str,
    ) -> None:
        jittered_positions = cls._boxplot_scatter_x_positions(position, len(values))
        if orientation == "vertical":
            ax.scatter(
                jittered_positions,
                values,
                s=cls.FIGURE_SCATTER_SIZE,
                marker="o",
                facecolors=color,
                edgecolors="white",
                linewidths=0.25,
                alpha=cls.FIGURE_SCATTER_ALPHA,
                zorder=3,
            )
            return
        if orientation == "horizontal":
            ax.scatter(
                values,
                jittered_positions,
                s=cls.FIGURE_SCATTER_SIZE,
                marker="o",
                facecolors=color,
                edgecolors="white",
                linewidths=0.25,
                alpha=cls.FIGURE_SCATTER_ALPHA,
                zorder=3,
            )
            return
        raise ValueError(f"Unsupported scatter orientation {orientation!r}.")

    @classmethod
    def _boxplot_scatter_x_positions(cls, position: int, count: int) -> list[float]:
        if count <= 1:
            return [float(position)] * count
        span = cls.FIGURE_SCATTER_WIDTH
        return [
            float(position) + span * (idx / (count - 1) - 0.5)
            for idx in range(count)
        ]

    @staticmethod
    def _format_log_tick(value: float, _pos: int) -> str:
        if not math.isfinite(value) or value <= 0.0:
            return ""
        exponent = math.log10(value)
        rounded_exponent = round(exponent)
        if math.isclose(exponent, rounded_exponent, rel_tol=0.0, abs_tol=1e-10):
            return rf"$10^{{{int(rounded_exponent)}}}$"
        if value < 0.01 or value >= 1000.0:
            exponent_floor = math.floor(exponent)
            coefficient = value / (10.0 ** exponent_floor)
            return rf"${coefficient:g}\times10^{{{exponent_floor}}}$"
        return f"{value:g}"

    @classmethod
    def _common_success_instance_ids(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        group: str,
        budget: float,
        metric_field: str,
        planners: Sequence[str],
    ) -> set[str]:
        planner_instance_sets: list[set[str]] = []
        for planner in planners:
            rows = cls._success_rows_by_instance(
                cls._rows_for_group(rows_by_planner.get(planner, []), group, budget),
                metric_field,
            )
            planner_instance_sets.append(set(rows))
        if not planner_instance_sets:
            return set()
        return set.intersection(*planner_instance_sets)


