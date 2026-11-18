from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

from experiments.reports.style import configure_matplotlib_for_latex

configure_matplotlib_for_latex(backend="Agg")

from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter

from experiments.mrmp import MRMPExperiment
from experiments.planners.mrmp import PBSExpansionAblation, PBSExpansionRuleSpec
from experiments.reports.common import (
    BoxplotStyle,
    PlotPalette,
    ViolinStyle,
    _save_figure,
    dedupe_result_rows,
    plt,
)


class MRMPPBSNodeExpansionReport:
    DEFAULT_RESULTS_ROOT = Path(PBSExpansionAblation.DEFAULT_OUTPUT_ROOT)
    DEFAULT_OUTPUT_PREFIX = Path("latex/figs/mrmp_pbs_node_expansion")
    DOMAINS = ("grid2d", "maze", "iris-2d")
    DEFAULT_BUDGET = PBSExpansionAblation.DEFAULT_BUDGET
    FIGURE_SIZE = (5, 3.8)
    SUCCESS_RATE_TICKS = (0.0, 0.5, 1.0)
    SUCCESS_RATE_Y_LIMITS = (-0.05, 1.1)
    RUNTIME_X_TICKS = (5.0, 60.0, 600.0)
    PBS_POPPED_NODES_X_TICKS = (6.0, 24.0, 100.0)
    PBS_GENERATED_CHILDREN_X_TICKS = (10.0, 40.0, 150.0)
    SUCCESS_LOG_X_FIELDS = ("runtime", "pbs_popped_nodes", "pbs_generated_children")
    Y_TICK_LABEL_PAD = 1.5
    METRIC_SCATTER_SIZE = 7.5
    METRIC_SCATTER_ALPHA = 0.72
    RULE_REFERENCE_COLOR = PlotPalette.NEUTRAL_MID
    RULE_REFERENCE_ALPHA = 0.28
    RULE_REFERENCE_LINEWIDTH = 0.7
    RULE_REFERENCE_LINESTYLE = "--"
    METRIC_X_TICKS = {
        "branching_ratio": (1.3, 1.6, 1.9),
        "sum_of_costs": (30.0, 60.0, 90.0),
        "makespan": (4.0, 8.0, 12.0),
    }
    RULE_STYLES = {
        "lazy": {
            "label": "Lazy",
            "tick_label": "Lazy",
            "color": PlotPalette.MRMP_RULE_COLORS["lazy"],
            "linestyle": "-",
        },
        "soc": {
            "label": "SoC",
            "tick_label": "SoC",
            "color": PlotPalette.MRMP_RULE_COLORS["soc"],
            "linestyle": "-",
        },
        "makespan": {
            "label": "MS",
            "tick_label": "MS",
            "color": PlotPalette.MRMP_RULE_COLORS["makespan"],
            "linestyle": "--",
        },
        "num_conflicts": {
            "label": "NC",
            "tick_label": "NC",
            "color": PlotPalette.MRMP_RULE_COLORS["num_conflicts"],
            "linestyle": ":",
        },
    }

    @classmethod
    def domains_for_arg(cls, domain: str) -> tuple[str, ...]:
        if domain == "all":
            return cls.DOMAINS
        return (domain,)

    @staticmethod
    def result_path(
        results_root: str | Path,
        domain_key: str,
        rule: PBSExpansionRuleSpec,
    ) -> Path:
        safe_name = PBSExpansionAblation.planner_name(rule).replace("/", "_")
        return Path(results_root) / domain_key / "results" / f"{safe_name}.csv"

    @classmethod
    def load_rows_by_rule(
        cls,
        results_root: str | Path,
        domains: Sequence[str],
    ) -> Dict[str, list[Dict[str, object]]]:
        rows_by_rule: Dict[str, list[Dict[str, object]]] = {}
        for rule in PBSExpansionAblation.RULES:
            rows: list[Dict[str, object]] = []
            for domain in domains:
                path = cls.result_path(results_root, domain, rule)
                if path.exists():
                    rows.extend(MRMPExperiment.load_result_rows(path))
            rows_by_rule[rule.key] = dedupe_result_rows(rows)
        return rows_by_rule

    @classmethod
    def select_budget(
        cls,
        rows_by_rule: Mapping[str, Sequence[Dict[str, object]]],
        requested_budget: float | None,
    ) -> float:
        if requested_budget is not None:
            return float(requested_budget)
        common_budgets = cls._common_budgets(rows_by_rule)
        if common_budgets:
            return max(common_budgets)
        observed_budgets: set[float] = set()
        for rows in rows_by_rule.values():
            observed_budgets.update(cls._positive_budgets(rows))
        if observed_budgets:
            return max(observed_budgets)
        return cls.DEFAULT_BUDGET

    @staticmethod
    def _positive_budgets(rows: Sequence[Dict[str, object]]) -> set[float]:
        return {
            float(row["budget"])
            for row in rows
            if math.isfinite(float(row["budget"])) and float(row["budget"]) > 0.0
        }

    @classmethod
    def _common_budgets(
        cls,
        rows_by_rule: Mapping[str, Sequence[Dict[str, object]]],
    ) -> list[float]:
        budgets_by_rule = [
            cls._positive_budgets(rows_by_rule.get(rule.key, []))
            for rule in PBSExpansionAblation.RULES
        ]
        if not budgets_by_rule or any(not budgets for budgets in budgets_by_rule):
            return []
        return sorted(set.intersection(*budgets_by_rule))

    @staticmethod
    def _same_budget(row: Dict[str, object], budget: float) -> bool:
        return math.isclose(float(row["budget"]), float(budget))

    @classmethod
    def rows_by_instance(
        cls,
        rows_by_rule: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
    ) -> Dict[str, Dict[str, Dict[str, object]]]:
        rows_by_instance: Dict[str, Dict[str, Dict[str, object]]] = {}
        for rule in PBSExpansionAblation.RULES:
            for row in rows_by_rule.get(rule.key, []):
                if cls._same_budget(row, budget):
                    rows_by_instance.setdefault(str(row["instance_id"]), {})[rule.key] = row
        return rows_by_instance

    @classmethod
    def common_instance_ids(
        cls,
        rows_by_rule: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
    ) -> set[str]:
        instance_ids: set[str] = set()
        for instance_id, rule_rows in cls.rows_by_instance(rows_by_rule, budget).items():
            if all(rule.key in rule_rows for rule in PBSExpansionAblation.RULES):
                instance_ids.add(instance_id)
        return instance_ids

    @classmethod
    def common_success_instance_ids(
        cls,
        rows_by_rule: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
    ) -> set[str]:
        instance_ids: set[str] = set()
        for instance_id, rule_rows in cls.rows_by_instance(rows_by_rule, budget).items():
            if all(
                rule.key in rule_rows and bool(rule_rows[rule.key]["is_success"])
                for rule in PBSExpansionAblation.RULES
            ):
                instance_ids.add(instance_id)
        return instance_ids

    @classmethod
    def common_metric_instance_ids(
        cls,
        rows_by_rule: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        field: str,
    ) -> set[str]:
        instance_ids: set[str] = set()
        for instance_id, rule_rows in cls.rows_by_instance(rows_by_rule, budget).items():
            if all(
                rule.key in rule_rows
                and bool(rule_rows[rule.key]["is_success"])
                and cls._finite_positive(rule_rows[rule.key][field]) is not None
                for rule in PBSExpansionAblation.RULES
            ):
                instance_ids.add(instance_id)
        return instance_ids

    @classmethod
    def rule_budget_rows(
        cls,
        rows_by_rule: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        rule_key: str,
    ) -> list[Dict[str, object]]:
        instance_ids = cls.common_instance_ids(rows_by_rule, budget)
        return [
            row for row in rows_by_rule.get(rule_key, [])
            if cls._same_budget(row, budget)
            and str(row["instance_id"]) in instance_ids
        ]

    @staticmethod
    def _finite_positive(value: object) -> float | None:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            return None
        return numeric

    @classmethod
    def _curve_runtime(cls, row: Dict[str, object], budget: float) -> float | None:
        runtime = cls._finite_positive(row["runtime"])
        if runtime is not None:
            return min(runtime, float(budget)) if math.isfinite(float(budget)) else runtime
        if not bool(row["is_success"]):
            return cls._finite_positive(budget)
        return None

    @classmethod
    def _curve_value(cls, row: Dict[str, object], budget: float, field: str) -> float | None:
        if field == "runtime":
            return cls._curve_runtime(row, budget)
        return cls._finite_positive(row[field])

    @classmethod
    def success_rate_points(
        cls,
        rows_by_rule: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        rule_key: str,
        field: str = "runtime",
    ) -> tuple[list[float], list[float]]:
        rows = cls.rule_budget_rows(rows_by_rule, budget, rule_key)
        if not rows:
            return [], []

        events: dict[float, int] = {}
        for row in rows:
            x_value = cls._curve_value(row, budget, field)
            if x_value is None:
                continue
            events.setdefault(x_value, 0)
            if bool(row["is_success"]):
                events[x_value] += 1
        if not events:
            return [], []

        xs: list[float] = []
        ys: list[float] = []
        success_count = 0
        total_count = len(rows)
        for x_value in sorted(events):
            success_count += events[x_value]
            xs.append(x_value)
            ys.append(success_count / total_count)
        return xs, ys

    @classmethod
    def success_curve_bounds(
        cls,
        rows_by_rule: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        field: str,
    ) -> tuple[float, float] | None:
        values: list[float] = []
        for rule in PBSExpansionAblation.RULES:
            xs, _ = cls.success_rate_points(rows_by_rule, budget, rule.key, field=field)
            values.extend(xs)
        if not values:
            return None
        return min(values), max(values)

    @classmethod
    def metric_values(
        cls,
        rows_by_rule: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        rule_key: str,
        field: str,
    ) -> list[float]:
        rows_by_instance = cls.rows_by_instance(rows_by_rule, budget)
        values: list[float] = []
        for instance_id in sorted(
            cls.common_metric_instance_ids(rows_by_rule, budget, field)
        ):
            row = rows_by_instance[instance_id][rule_key]
            value = cls._finite_positive(row[field])
            if value is not None:
                values.append(value)
        return values

    @classmethod
    def child_branching_ratios(
        cls,
        rows_by_rule: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        rule_key: str,
    ) -> list[float]:
        rows_by_instance = cls.rows_by_instance(rows_by_rule, budget)
        ratios: list[float] = []
        for instance_id in sorted(cls.common_success_instance_ids(rows_by_rule, budget)):
            row = rows_by_instance[instance_id][rule_key]
            popped_nodes = float(row["pbs_popped_nodes"])
            generated_nodes = float(row["pbs_generated_children"])
            if (
                math.isfinite(popped_nodes)
                and popped_nodes > 0.0
                and math.isfinite(generated_nodes)
                and generated_nodes >= 0.0
            ):
                ratios.append(generated_nodes / popped_nodes)
        return ratios

    @classmethod
    def plot_figure(
        cls,
        rows_by_rule: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        output_prefix: Path,
        formats: Sequence[str],
    ) -> None:
        fig, axes = plt.subplots(2, 3, figsize=cls.FIGURE_SIZE)
        cls._plot_success_rate_panel(
            axes[0, 0],
            rows_by_rule,
            budget,
            field="runtime",
            xlabel="runtime (s)",
            show_success_label=True,
            solid_lines=True,
        )
        cls._plot_success_rate_panel(
            axes[0, 1],
            rows_by_rule,
            budget,
            field="pbs_popped_nodes",
            xlabel="expanded nodes",
            show_success_label=False,
            solid_lines=True,
        )
        cls._plot_success_rate_panel(
            axes[0, 2],
            rows_by_rule,
            budget,
            field="pbs_generated_children",
            xlabel="generated nodes",
            show_success_label=False,
            solid_lines=True,
        )
        cls._plot_branching_panel(
            axes[1, 0],
            rows_by_rule,
            budget,
            show_rule_tick_labels=True,
        )
        cls._plot_metric_panel(
            axes[1, 1],
            rows_by_rule,
            budget,
            field="sum_of_costs",
            metric_label="sum-of-costs",
            show_rule_tick_labels=False,
        )
        cls._plot_metric_panel(
            axes[1, 2],
            rows_by_rule,
            budget,
            field="makespan",
            metric_label="makespan",
            show_rule_tick_labels=False,
        )

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.51, 1),
            ncol=4,
            frameon=False,
            fontsize=11.5,
            borderaxespad=0.2,
            handlelength=1.5,
            labelspacing=0.25,
            columnspacing=4,
        )
        # _set_square_cells(axes)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), pad=0.15, w_pad=0.15, h_pad=0.5)
        _save_figure(fig, output_prefix, formats)

    @classmethod
    def _plot_success_rate_panel(
        cls,
        ax: plt.Axes,
        rows_by_rule: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        field: str,
        xlabel: str,
        show_success_label: bool = True,
        solid_lines: bool = False,
    ) -> None:
        for rule in PBSExpansionAblation.RULES:
            style = cls.RULE_STYLES[rule.key]
            xs, ys = cls.success_rate_points(rows_by_rule, budget, rule.key, field=field)
            if not xs:
                continue
            ax.step(
                xs,
                ys,
                where="post",
                label=str(style["label"]),
                color=str(style["color"]),
                linestyle="-" if solid_lines else str(style["linestyle"]),
                linewidth=1.35,
                alpha=0.95,
            )
        cls._style_success_axis(
            ax,
            rows_by_rule,
            budget,
            field,
            xlabel,
            show_success_label,
        )

    @classmethod
    def _plot_metric_panel(
        cls,
        ax: plt.Axes,
        rows_by_rule: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        field: str,
        metric_label: str,
        show_rule_tick_labels: bool,
    ) -> None:
        y_positions = list(range(1, len(PBSExpansionAblation.RULES) + 1))
        all_values: list[float] = []
        for y_position, rule in zip(y_positions, PBSExpansionAblation.RULES):
            values = cls.metric_values(rows_by_rule, budget, rule.key, field)
            if not values:
                continue
            all_values.extend(values)
            style = cls.RULE_STYLES[rule.key]
            ViolinStyle.draw(
                ax,
                float(y_position),
                values,
                color=str(style["color"]),
            )
            BoxplotStyle.scatter_points(
                ax,
                float(y_position),
                values,
                width=BoxplotStyle.DEFAULT_WIDTH,
                color=str(style["color"]),
                size=cls.METRIC_SCATTER_SIZE,
                alpha=cls.METRIC_SCATTER_ALPHA,
                orientation="horizontal",
            )
        cls._set_padded_data_xlim(ax, all_values)
        cls._style_horizontal_metric_axis(
            ax,
            y_positions,
            field,
            metric_label,
            show_rule_tick_labels,
        )

    @classmethod
    def _plot_branching_panel(
        cls,
        ax: plt.Axes,
        rows_by_rule: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        show_rule_tick_labels: bool = True,
    ) -> None:
        y_positions = list(range(1, len(PBSExpansionAblation.RULES) + 1))
        all_ratios: list[float] = []
        for y_position, rule in zip(y_positions, PBSExpansionAblation.RULES):
            ratios = cls.child_branching_ratios(rows_by_rule, budget, rule.key)
            if not ratios:
                continue
            all_ratios.extend(ratios)
            style = cls.RULE_STYLES[rule.key]
            ViolinStyle.draw(
                ax,
                float(y_position),
                ratios,
                color=str(style["color"]),
            )
            BoxplotStyle.scatter_points(
                ax,
                float(y_position),
                ratios,
                width=BoxplotStyle.DEFAULT_WIDTH,
                color=str(style["color"]),
                size=cls.METRIC_SCATTER_SIZE,
                alpha=cls.METRIC_SCATTER_ALPHA,
                orientation="horizontal",
            )
        cls._set_padded_data_xlim(ax, all_ratios)
        cls._style_horizontal_metric_axis(
            ax,
            y_positions,
            "branching_ratio",
            "generated/expanded",
            show_rule_tick_labels,
        )

    @classmethod
    def _style_horizontal_metric_axis(
        cls,
        ax: plt.Axes,
        y_positions: Sequence[int],
        metric_key: str,
        metric_label: str,
        show_rule_tick_labels: bool,
    ) -> None:
        ax.set_ylim(len(PBSExpansionAblation.RULES) + 0.55, 0.45)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [str(cls.RULE_STYLES[rule.key]["tick_label"]) for rule in PBSExpansionAblation.RULES],
        )
        ax.set_xlabel(metric_label, fontsize=12)
        ax.set_ylabel("")
        cls._draw_rule_reference_lines(ax, y_positions)
        cls._set_metric_x_ticks(ax, metric_key)
        ax.grid(axis="x", alpha=0.24, linewidth=0.7)
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(
            axis="y",
            labelsize=9.0,
            pad=cls.Y_TICK_LABEL_PAD,
            labelleft=show_rule_tick_labels,
        )

    @classmethod
    def _draw_rule_reference_lines(cls, ax: plt.Axes, y_positions: Sequence[int]) -> None:
        for y_position in y_positions:
            ax.axhline(
                y_position,
                color=cls.RULE_REFERENCE_COLOR,
                linestyle=cls.RULE_REFERENCE_LINESTYLE,
                linewidth=cls.RULE_REFERENCE_LINEWIDTH,
                alpha=cls.RULE_REFERENCE_ALPHA,
                zorder=1,
            )

    @classmethod
    def _set_metric_x_ticks(cls, ax: plt.Axes, metric_key: str) -> None:
        ticks = cls.METRIC_X_TICKS.get(metric_key)
        if ticks is not None:
            ax.xaxis.set_major_locator(FixedLocator(ticks))

    @staticmethod
    def _set_padded_data_xlim(ax: plt.Axes, values: Sequence[float]) -> None:
        finite_values = [
            float(value)
            for value in values
            if math.isfinite(float(value))
        ]
        if not finite_values:
            return
        low = min(finite_values)
        high = max(finite_values)
        if math.isclose(low, high):
            padding = max(0.05 * abs(low), 0.05)
        else:
            padding = 0.12 * (high - low)
        ax.set_xlim(low - padding, high + padding)

    @classmethod
    def _style_success_axis(
        cls,
        ax: plt.Axes,
        rows_by_rule: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        field: str,
        xlabel: str,
        show_success_label: bool,
    ) -> None:
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("")
        curve_bounds = cls.success_curve_bounds(rows_by_rule, budget, field)
        success_x_ticks: tuple[float, ...] | None = None
        if field in cls.SUCCESS_LOG_X_FIELDS:
            success_x_ticks = cls._set_success_log_x_axis(ax, curve_bounds, budget, field)
        else:
            if curve_bounds is None:
                high = max(float(budget), 10.0)
            else:
                _, high = curve_bounds
            padding = max(1.0, 0.04 * float(high))
            ax.set_xlim(0.0, float(high) + padding)
        ax.set_ylim(*cls.SUCCESS_RATE_Y_LIMITS)
        if field in cls.SUCCESS_LOG_X_FIELDS:
            ax.xaxis.set_major_locator(FixedLocator(success_x_ticks or ()))
            ax.xaxis.set_major_formatter(FuncFormatter(cls._log_x_tick_label))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.yaxis.set_major_locator(FixedLocator(cls.SUCCESS_RATE_TICKS))
        ax.yaxis.set_major_formatter(FuncFormatter(cls._success_rate_tick_label))
        ax.grid(alpha=0.24, linewidth=0.7)
        ax.tick_params(axis="both", labelsize=11)
        cls._style_y_tick_labels(ax, show_tick_labels=show_success_label)

    @classmethod
    def _set_success_log_x_axis(
        cls,
        ax: plt.Axes,
        curve_bounds: tuple[float, float] | None,
        budget: float,
        field: str,
    ) -> tuple[float, ...]:
        positive_ticks = [
            float(tick)
            for tick in cls._success_log_x_ticks(field)
            if math.isfinite(float(tick)) and float(tick) > 0.0
        ]
        if not positive_ticks:
            raise ValueError(f"{field} log-scale x ticks must contain at least one positive finite value.")
        if curve_bounds is None:
            low = min(positive_ticks)
            high = max(positive_ticks)
            if field == "runtime" and math.isfinite(float(budget)) and float(budget) > 0.0:
                high = max(float(budget), high)
        else:
            low, high = curve_bounds
        values = [
            float(value)
            for value in (low, high)
            if math.isfinite(float(value)) and float(value) > 0.0
        ]
        if not values:
            values = [min(positive_ticks), max(positive_ticks)]
        ax.set_xscale("log")
        ax.set_xlim(cls._positive_log_axis_limits(values))
        return tuple(positive_ticks)

    @classmethod
    def _success_log_x_ticks(cls, field: str) -> Sequence[float]:
        if field == "runtime":
            return cls.RUNTIME_X_TICKS
        if field == "pbs_popped_nodes":
            return cls.PBS_POPPED_NODES_X_TICKS
        if field == "pbs_generated_children":
            return cls.PBS_GENERATED_CHILDREN_X_TICKS
        return ()

    @staticmethod
    def _positive_log_axis_limits(values: Sequence[float]) -> tuple[float, float]:
        positive_values = [
            float(value)
            for value in values
            if math.isfinite(float(value)) and float(value) > 0.0
        ]
        if not positive_values:
            return (1.0, 2.0)
        low = min(positive_values)
        high = max(positive_values)
        if math.isclose(low, high):
            return (low / math.sqrt(2.0), high * math.sqrt(2.0))
        log_low = math.log2(low)
        log_high = math.log2(high)
        padding = 0.08 * (log_high - log_low)
        return (2 ** (log_low - padding), 2 ** (log_high + padding))

    @staticmethod
    def _log_x_tick_label(value: float, _position: int) -> str:
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            return ""
        return f"{float(value):g}"

    @classmethod
    def _style_y_tick_labels(cls, ax: plt.Axes, show_tick_labels: bool = True) -> None:
        ax.tick_params(
            axis="y",
            labelrotation=90,
            pad=cls.Y_TICK_LABEL_PAD,
            labelleft=show_tick_labels,
        )

    @staticmethod
    def _success_rate_tick_label(value: float, _position: int) -> str:
        if math.isclose(float(value), 0.0):
            return "0%"
        return f"{float(value):.0%}"


class MRMPPBSNodeExpansionReportCLI:
    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Plot MRMP PBS child-node expansion success-rate and metric diagnostics."
        )
        parser.add_argument(
            "--results-root",
            type=Path,
            default=MRMPPBSNodeExpansionReport.DEFAULT_RESULTS_ROOT,
        )
        parser.add_argument(
            "--output-prefix",
            type=Path,
            default=MRMPPBSNodeExpansionReport.DEFAULT_OUTPUT_PREFIX,
        )
        parser.add_argument("--formats", nargs="+", default=("png", "pdf"))
        parser.add_argument("--budget", type=float, default=None)
        parser.add_argument(
            "--domain",
            choices=("all", *MRMPPBSNodeExpansionReport.DOMAINS),
            default="all",
        )
        return parser.parse_args()

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        domains = MRMPPBSNodeExpansionReport.domains_for_arg(args.domain)
        rows_by_rule = MRMPPBSNodeExpansionReport.load_rows_by_rule(args.results_root, domains)
        budget = MRMPPBSNodeExpansionReport.select_budget(rows_by_rule, args.budget)
        num_instances = len(MRMPPBSNodeExpansionReport.common_instance_ids(rows_by_rule, budget))
        if num_instances == 0:
            raise RuntimeError(
                f"No common MRMP result rows found for all PBS expansion rules at budget={budget:g}."
            )
        print(f"Plotting budget={budget:g} over {num_instances} common MRMP instances.")
        MRMPPBSNodeExpansionReport.plot_figure(
            rows_by_rule,
            budget,
            args.output_prefix,
            args.formats,
        )


if __name__ == "__main__":
    MRMPPBSNodeExpansionReportCLI.main()
