from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Dict, Mapping, Sequence

from experiments.reports.style import configure_matplotlib_for_latex

configure_matplotlib_for_latex(backend="Agg")

from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

from experiments.mrmp import MRMPExperiment
from experiments.planners.mrmp import MRMPPerformanceComparison
from experiments.reports.common import PlotPalette, _save_figure, dedupe_result_rows, plt


class MRMPPerformanceComparisonReport:
    DEFAULT_RESULTS_ROOT = Path(MRMPPerformanceComparison.DEFAULT_OUTPUT_ROOT)
    DEFAULT_OUTPUT_PREFIX = Path("latex/figs/mrmp_performance_comparison")
    DEFAULT_BUDGET = 600.0
    DOMAINS = MRMPPerformanceComparison.DOMAINS
    PLANNER_KEYS = MRMPPerformanceComparison.PLANNER_KEYS
    DOMAIN_LABELS = {
        "grid2d": "rand",
        "maze": "maze",
        "iris-2d": "iris",
    }
    FIGURE_SIZE = (7, 4)
    COLUMN_WIDTH_RATIOS = (1.0, 1.0, 1.0, 1.0)
    ROBOT_COUNT_VALUES = MRMPPerformanceComparison.num_agent_values()
    ROBOT_COUNT_TICK_LABELS = tuple(str(num_agents) for num_agents in ROBOT_COUNT_VALUES)
    SUCCESS_RATE_TICKS = (0.0, 0.5, 1.0)
    SUCCESS_RATE_TICK_LABELS = ("0%", "50%", "100%")
    SUCCESS_RATE_Y_LIMITS = (-0.05, 1.1)
    # Fit the plotted medians; fixed limits can clip newly completed results.
    METRIC_Y_LIMITS_BY_DOMAIN: Dict[str, Dict[str, tuple[float, float]]] = {}
    METRIC_Y_TICK_LABELS_BY_DOMAIN: Dict[str, Dict[str, tuple[str, ...]]] = {
        "runtime": {
            "grid2d": ("0.01", "1", "100"),
            "maze": ("0.1", "3", "100"),
            "iris-2d": ("0.1", "3", "100"),
        },
        "sum_of_costs": {
            "grid2d": ("10", "100", "1000"),
            "maze": ("10", "100", "1000"),
            "iris-2d": ("10", "40", "160"),
        },
        "makespan": {
            "grid2d": ("4", "16", "64"),
            "maze": ("6", "30", "150"),
            "iris-2d": ("4", "8", "16"),
        },
    }
    X_TICK_LABEL_SIZE = 8.0
    X_AXIS_LABEL_SIZE = 9.5
    Y_TICK_LABEL_SIZE = 9
    Y_TICK_LABEL_PAD = 1.5
    CURVE_MARKER = "o"
    CURVE_MARKER_SIZE = 3.0
    CURVE_LINEWIDTH = 1.2
    COMMON_SUCCESS_MIN_SUCCESS_RATE = 0.33
    LEGEND_LABEL_ORDER = (
        "Windowed-PP + BFS",
        "Windowed-PBS + BFS",
        "PP + BFS",
        "PBS + BFS",
        "PP + ST-RRT*",
        "PBS + Zeta*-SIPP",
        "K-CBS",
        "CB-GCS",
    )
    METHOD_STYLES = {
        MRMPPerformanceComparison.WINDOWED_PBS_KEY: {
            "label": "Windowed-PBS + BFS",
            "tick_label": "wPBS",
            "color": PlotPalette.MRMP_WINDOW_COLORS["pbs_nc"],
            "linestyle": "-",
            "marker": "s",
            "markerfacecolor": "none",
        },
        MRMPPerformanceComparison.WINDOWED_PP_KEY: {
            "label": "Windowed-PP + BFS",
            "tick_label": "wPP",
            "color": PlotPalette.MRMP_WINDOW_COLORS["fixed"],
            "linestyle": "-",
            "marker": "s",
            "markerfacecolor": "none",
        },
        MRMPPerformanceComparison.PBS_KEY: {
            "label": "PBS + BFS",
            "tick_label": "PBS",
            "color": PlotPalette.DOMAIN_IRIS,
            "linestyle": "-",
            "marker": "o",
            "markerfacecolor": "none",
        },
        MRMPPerformanceComparison.PP_BFS_KEY: {
            "label": "PP + BFS",
            "tick_label": "PP+BFS",
            "color": PlotPalette.NEUTRAL_DARK,
            "linestyle": "-",
            "marker": "o",
            "markerfacecolor": "none",
        },
        MRMPPerformanceComparison.FIXED_PP_ST_RRT_STAR_FIRST_KEY: {
            "label": "PP + ST-RRT*",
            "tick_label": "PP+ST-RRT*",
            "color": PlotPalette.ST_METHOD_COLORS["micp_rounding"],
            "linestyle": "--",
            "marker": ">",
            "markerfacecolor": "none",
        },
        MRMPPerformanceComparison.FIXED_PP_ST_RRT_STAR_FINAL_KEY: {
            "label": "PP + ST-RRT*",
            "tick_label": "PP+ST-RRT*",
            "color": PlotPalette.ST_METHOD_COLORS["micp_rounding"],
            "linestyle": "--",
            "marker": ">",
            "markerfacecolor": "none",
        },
        MRMPPerformanceComparison.PBS_ZETA_SIPP_KEY: {
            "label": "PBS + Zeta*-SIPP",
            "tick_label": "PBS+Zeta",
            "color": PlotPalette.ST_METHOD_COLORS["zeta_sipp"],
            "linestyle": "--",
            "marker": ">",
            "markerfacecolor": "none",
        },
        MRMPPerformanceComparison.CB_GCS_KEY: {
            "label": "CB-GCS",
            "tick_label": "CB-GCS",
            "color": PlotPalette.ST_METHOD_COLORS["zeta_sipp_2r"],
            "linestyle": ":",
            "marker": "d",
            "markerfacecolor": "none",
        },
        MRMPPerformanceComparison.KCBS_KEY: {
            "label": "K-CBS",
            "tick_label": "K-CBS",
            "color": PlotPalette.BLUE,
            "linestyle": "-.",
            "marker": "d",
            "markerfacecolor": "none",
        },
    }

    @classmethod
    def domains_for_arg(cls, domain: str) -> tuple[str, ...]:
        if domain == "all":
            return cls.DOMAINS
        return (domain,)

    @classmethod
    def planner_keys_for_arg(cls, planners: Sequence[str] | None) -> tuple[str, ...]:
        if planners is None:
            return cls.PLANNER_KEYS
        selected: list[str] = []
        for planner in planners:
            key = str(planner)
            if key == MRMPPerformanceComparison.ALL_PLANNER_KEY:
                return cls.PLANNER_KEYS
            if key not in MRMPPerformanceComparison.PLANNER_SELECTION_KEYS:
                raise ValueError(f"Unknown MRMP performance-comparison planner {key!r}.")
            for expanded_key in MRMPPerformanceComparison.expand_planner_key(key):
                if expanded_key not in selected:
                    selected.append(expanded_key)
        if not selected:
            raise ValueError("At least one MRMP performance-comparison planner must be selected.")
        return tuple(selected)

    @classmethod
    def result_paths(cls, results_root: str | Path, domain_key: str, planner_key: str) -> tuple[Path, ...]:
        return tuple(
            MRMPPerformanceComparison.result_path(results_root, num_agents, domain_key, planner_key)
            for num_agents in MRMPPerformanceComparison.num_agent_values()
        )

    @classmethod
    def load_rows_by_planner(
        cls,
        results_root: str | Path,
        domains: Sequence[str],
        planner_keys: Sequence[str],
    ) -> Dict[str, list[Dict[str, object]]]:
        rows_by_planner: Dict[str, list[Dict[str, object]]] = {}
        for planner_key in planner_keys:
            rows: list[Dict[str, object]] = []
            for domain in domains:
                for path in cls.result_paths(results_root, domain, planner_key):
                    if path.exists():
                        rows.extend(cls._load_domain_rows(path, domain))
            rows_by_planner[planner_key] = dedupe_result_rows(rows)
        return rows_by_planner

    @staticmethod
    def _load_domain_rows(path: str | Path, domain_key: str) -> list[Dict[str, object]]:
        rows = MRMPExperiment.load_result_rows(path)
        for row in rows:
            row["domain_key"] = domain_key
        return rows

    @classmethod
    def row_domain_key(cls, row: Mapping[str, object]) -> str | None:
        domain_key = str(row.get("domain_key", ""))
        if domain_key:
            return domain_key
        instance_id = str(row.get("instance_id", ""))
        for candidate in cls.DOMAINS:
            if f"-{candidate}-" in instance_id:
                return candidate
        return None

    @classmethod
    def rows_for_domain(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        domain_key: str,
    ) -> Dict[str, Sequence[Dict[str, object]]]:
        return {
            planner_key: [
                row for row in rows
                if cls.row_domain_key(row) == domain_key
            ]
            for planner_key, rows in rows_by_planner.items()
        }

    @classmethod
    def domains_from_rows(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
    ) -> tuple[str, ...]:
        observed_domains = {
            domain_key
            for rows in rows_by_planner.values()
            for row in rows
            if (domain_key := cls.row_domain_key(row)) is not None
        }
        return tuple(domain for domain in cls.DOMAINS if domain in observed_domains)

    @staticmethod
    def _positive_budgets(rows: Sequence[Dict[str, object]]) -> set[float]:
        return {
            float(row["budget"])
            for row in rows
            if math.isfinite(float(row["budget"])) and float(row["budget"]) > 0.0
        }

    @classmethod
    def select_budget(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        requested_budget: float | None,
    ) -> float:
        if requested_budget is not None:
            return float(requested_budget)
        nonempty_budget_sets = [
            cls._positive_budgets(rows)
            for rows in rows_by_planner.values()
            if rows
        ]
        if nonempty_budget_sets and all(nonempty_budget_sets):
            common_budgets = set.intersection(*nonempty_budget_sets)
            if common_budgets:
                return max(common_budgets)
        observed_budgets: set[float] = set()
        for rows in rows_by_planner.values():
            observed_budgets.update(cls._positive_budgets(rows))
        return max(observed_budgets) if observed_budgets else cls.DEFAULT_BUDGET

    @staticmethod
    def _same_budget(row: Dict[str, object], budget: float) -> bool:
        return math.isclose(float(row["budget"]), float(budget))

    @classmethod
    def rows_by_instance(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
    ) -> Dict[str, Dict[str, Dict[str, object]]]:
        rows_by_instance: Dict[str, Dict[str, Dict[str, object]]] = {}
        for planner_key, rows in rows_by_planner.items():
            for row in rows:
                if cls._same_budget(row, budget):
                    rows_by_instance.setdefault(str(row["instance_id"]), {})[planner_key] = row
        return rows_by_instance

    @classmethod
    def common_instance_ids(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
    ) -> set[str]:
        active_planners = {
            planner_key
            for planner_key in rows_by_planner
            if cls.planner_budget_rows(rows_by_planner, budget, planner_key)
        }
        return {
            instance_id
            for instance_id, planner_rows in cls.rows_by_instance(rows_by_planner, budget).items()
            if active_planners.issubset(planner_rows)
        }

    @classmethod
    def active_planner_keys(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_keys: Sequence[str],
    ) -> tuple[str, ...]:
        return tuple(
            planner_key
            for planner_key in planner_keys
            if cls.planner_budget_rows(rows_by_planner, budget, planner_key)
        )

    @classmethod
    def planner_budget_rows(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_key: str,
    ) -> list[Dict[str, object]]:
        return [
            row for row in rows_by_planner.get(planner_key, [])
            if cls._same_budget(row, budget)
        ]

    @classmethod
    def budget_row_counts(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
    ) -> Dict[str, int]:
        return {
            planner_key: len(cls.planner_budget_rows(rows_by_planner, budget, planner_key))
            for planner_key in rows_by_planner
        }

    @staticmethod
    def format_budget_row_counts(row_counts: Mapping[str, int]) -> str:
        nonempty_counts = [
            f"{planner_key}={count}"
            for planner_key, count in row_counts.items()
            if count > 0
        ]
        if not nonempty_counts:
            return "none"
        return ", ".join(nonempty_counts)

    @staticmethod
    def _finite_positive(value: object) -> float | None:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            return None
        return numeric

    @classmethod
    def planner_budget_num_agent_rows(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_key: str,
        num_agents: int,
    ) -> list[Dict[str, object]]:
        return [
            row for row in cls.planner_budget_rows(rows_by_planner, budget, planner_key)
            if int(row["num_agents"]) == int(num_agents)
        ]

    @staticmethod
    def _median(values: Sequence[float]) -> float | None:
        if not values:
            return None
        return statistics.median(float(value) for value in values)


    @classmethod
    def success_rate_by_num_agents(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_key: str,
    ) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for num_agents in cls.ROBOT_COUNT_VALUES:
            rows = cls.planner_budget_num_agent_rows(rows_by_planner, budget, planner_key, num_agents)
            if not rows:
                continue
            successes = sum(1 for row in rows if bool(row["is_success"]))
            points.append((float(num_agents), successes / len(rows)))
        return points

    @classmethod
    def success_rate_for_num_agents(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_key: str,
        num_agents: int,
    ) -> float | None:
        rows = cls.planner_budget_num_agent_rows(rows_by_planner, budget, planner_key, num_agents)
        if not rows:
            return None
        successes = sum(1 for row in rows if bool(row["is_success"]))
        return successes / len(rows)


    @classmethod
    def successful_metric_instance_ids(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_key: str,
        num_agents: int,
        field: str,
    ) -> set[str]:
        return {
            str(row["instance_id"])
            for row in cls.planner_budget_num_agent_rows(rows_by_planner, budget, planner_key, num_agents)
            if bool(row["is_success"]) and cls._finite_positive(row[field]) is not None
        }

    @classmethod
    def common_success_metric_context(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_keys: Sequence[str],
        num_agents: int,
        field: str,
    ) -> tuple[tuple[str, ...], set[str]]:
        successful_ids_by_planner = {
            planner_key: cls.successful_metric_instance_ids(
                rows_by_planner,
                budget,
                planner_key,
                num_agents,
                field,
            )
            for planner_key in planner_keys
        }
        eligible_planners = tuple(
            planner_key
            for planner_key in planner_keys
            if successful_ids_by_planner[planner_key]
            and (
                success_rate := cls.success_rate_for_num_agents(
                    rows_by_planner,
                    budget,
                    planner_key,
                    num_agents,
                )
            ) is not None
            and success_rate >= cls.COMMON_SUCCESS_MIN_SUCCESS_RATE
        )
        if not eligible_planners:
            return (), set()
        common_instance_ids = set.intersection(
            *(successful_ids_by_planner[planner_key] for planner_key in eligible_planners)
        )
        return eligible_planners, common_instance_ids

    @classmethod
    def common_success_median_field_by_num_agents(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_keys: Sequence[str],
        planner_key: str,
        field: str,
    ) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for num_agents in cls.ROBOT_COUNT_VALUES:
            eligible_planners, common_instance_ids = cls.common_success_metric_context(
                rows_by_planner,
                budget,
                planner_keys,
                num_agents,
                field,
            )
            if planner_key not in eligible_planners or not common_instance_ids:
                continue
            rows = [
                row
                for row in cls.planner_budget_num_agent_rows(rows_by_planner, budget, planner_key, num_agents)
                if str(row["instance_id"]) in common_instance_ids
            ]
            values = [
                value
                for row in rows
                if bool(row["is_success"]) and (value := cls._finite_positive(row[field])) is not None
            ]
            median = cls._median(values)
            if median is not None:
                points.append((float(num_agents), median))
        return points


    @classmethod
    def plot_figure(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        planner_keys: Sequence[str],
        budget: float,
        output_prefix: Path,
        formats: Sequence[str],
        domains: Sequence[str] | None = None,
        figure_title: str | None = None,
    ) -> None:
        planner_keys = cls.active_planner_keys(rows_by_planner, budget, planner_keys)
        if not planner_keys:
            raise ValueError(f"No MRMP performance-comparison rows found at budget={budget:g}.")
        domain_keys = tuple(domains) if domains is not None else cls.domains_from_rows(rows_by_planner)
        if not domain_keys:
            domain_keys = cls.DOMAINS
        metric_columns = (
            ("success", "success rate", None),
            ("runtime", "runtime (s)", "log"),
            ("sum_of_costs", "sum-of-costs", "log"),
            ("makespan", "makespan", "log"),
        )
        fig, axes = plt.subplots(
            len(domain_keys),
            len(metric_columns),
            figsize=cls.FIGURE_SIZE,
            squeeze=False,
            sharex="col",
            gridspec_kw={"width_ratios": cls.COLUMN_WIDTH_RATIOS},
        )
        for row_idx, domain_key in enumerate(domain_keys):
            domain_rows_by_planner = cls.rows_for_domain(rows_by_planner, domain_key)
            show_xlabel = row_idx == len(domain_keys) - 1
            for col_idx, (field, title, yscale) in enumerate(metric_columns):
                ax = axes[row_idx, col_idx]
                if row_idx == 0:
                    ax.set_title(title, fontsize=9, pad=2.0)
                if field == "success":
                    cls._plot_success_by_num_agents_panel(
                        ax,
                        domain_rows_by_planner,
                        cls.planner_keys_for_panel(planner_keys, field),
                        budget,
                        show_xlabel=show_xlabel,
                        show_ylabel=False,
                        show_y_tick_labels=True,
                    )
                    continue
                cls._plot_median_metric_by_num_agents_panel(
                    ax,
                    domain_rows_by_planner,
                    cls.planner_keys_for_panel(planner_keys, field),
                    budget,
                    field=field,
                    domain_key=domain_key,
                    ylabel=title,
                    yscale=yscale,
                    show_xlabel=show_xlabel,
                    show_ylabel=False,
                    show_y_tick_labels=True,
                )
            axes[row_idx, 0].set_ylabel(cls.domain_label(domain_key), fontsize=9, labelpad=8.0)
        handles, labels = cls.legend_handles_labels(axes)
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.99),
                ncol=min(4, len(handles)),
                frameon=False,
                fontsize=8,
                borderaxespad=0.2,
                handlelength=2,
                handletextpad=0.5,
                labelspacing=0.5,
                columnspacing=6,
            )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), pad=0.2, w_pad=0.55, h_pad=0.45)
        if figure_title:
            fig.suptitle(figure_title, fontsize=9, y=1.11)
        _save_figure(fig, output_prefix, formats)

    @classmethod
    def planner_keys_for_panel(cls, planner_keys: Sequence[str], field: str) -> tuple[str, ...]:
        first_key = MRMPPerformanceComparison.FIXED_PP_ST_RRT_STAR_FIRST_KEY
        final_key = MRMPPerformanceComparison.FIXED_PP_ST_RRT_STAR_FINAL_KEY
        if field in {"success", "runtime"}:
            return tuple(planner_key for planner_key in planner_keys if planner_key != final_key)
        if field in {"sum_of_costs", "makespan"}:
            return tuple(planner_key for planner_key in planner_keys if planner_key != first_key)
        return tuple(planner_keys)

    @classmethod
    def domain_label(cls, domain_key: str) -> str:
        return cls.DOMAIN_LABELS.get(domain_key, domain_key)

    @classmethod
    def legend_handles_labels(cls, axes) -> tuple[list[object], list[str]]:
        handles_by_label: dict[str, object] = {}
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label and label not in handles_by_label:
                    handles_by_label[label] = handle
        ordered_labels = [
            label
            for label in cls.LEGEND_LABEL_ORDER
            if label in handles_by_label
        ]
        ordered_labels.extend(
            label for label in handles_by_label if label not in ordered_labels
        )
        return [handles_by_label[label] for label in ordered_labels], ordered_labels

    @classmethod
    def curve_plot_kwargs(cls, style: Mapping[str, object]) -> Dict[str, object]:
        return {
            "marker": str(style.get("marker", cls.CURVE_MARKER)),
            "markersize": cls.CURVE_MARKER_SIZE,
            "label": str(style["label"]),
            "color": str(style["color"]),
            "linestyle": str(style["linestyle"]),
            "linewidth": cls.CURVE_LINEWIDTH,
            "markerfacecolor": str(style.get("markerfacecolor", style["color"])),
            "markeredgecolor": str(style["color"]),
            "markeredgewidth": 0.9,
        }

    @classmethod
    def _plot_success_by_num_agents_panel(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        planner_keys: Sequence[str],
        budget: float,
        show_xlabel: bool = True,
        show_ylabel: bool = True,
        show_y_tick_labels: bool | None = None,
    ) -> None:
        if show_y_tick_labels is None:
            show_y_tick_labels = show_ylabel
        for planner_key in planner_keys:
            points = cls.success_rate_by_num_agents(rows_by_planner, budget, planner_key)
            if not points:
                continue
            xs, ys = zip(*points)
            style = cls.METHOD_STYLES[planner_key]
            ax.plot(
                xs,
                ys,
                **cls.curve_plot_kwargs(style),
            )
        ax.set_ylabel("success rate" if show_ylabel else "", fontsize=9)
        cls._style_robot_count_axis(ax, show_xlabel=show_xlabel)
        cls._style_success_axis(ax, show_tick_labels=show_y_tick_labels)

    @classmethod
    def _plot_median_metric_by_num_agents_panel(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        planner_keys: Sequence[str],
        budget: float,
        field: str,
        domain_key: str,
        ylabel: str,
        yscale: str | None = None,
        show_xlabel: bool = True,
        show_ylabel: bool = True,
        show_y_tick_labels: bool | None = None,
    ) -> None:
        if show_y_tick_labels is None:
            show_y_tick_labels = show_ylabel
        all_values: list[float] = []
        for planner_key in planner_keys:
            points = cls.common_success_median_field_by_num_agents(
                rows_by_planner,
                budget,
                planner_keys,
                planner_key,
                field,
            )
            if not points:
                continue
            xs, ys = zip(*points)
            all_values.extend(float(value) for value in ys)
            style = cls.METHOD_STYLES[planner_key]
            ax.plot(
                xs,
                ys,
                **cls.curve_plot_kwargs(style),
            )
        ax.set_ylabel(ylabel if show_ylabel else "", fontsize=9)
        cls._style_robot_count_axis(ax, show_xlabel=show_xlabel)
        cls._style_metric_axis(
            ax,
            all_values,
            yscale=yscale,
            show_tick_labels=show_y_tick_labels,
            field=field,
            domain_key=domain_key,
        )

    @staticmethod
    def _positive_log_axis_limits(values: Sequence[float]) -> tuple[float, float]:
        positive_values = [float(value) for value in values if float(value) > 0.0 and math.isfinite(float(value))]
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

    @classmethod
    def _style_success_axis(cls, ax: plt.Axes, show_tick_labels: bool = True) -> None:
        ax.set_ylim(*cls.SUCCESS_RATE_Y_LIMITS)
        ax.set_yticks(cls.SUCCESS_RATE_TICKS, labels=cls.SUCCESS_RATE_TICK_LABELS)
        ax.grid(alpha=0.24, linewidth=0.7)
        ax.tick_params(axis="y", labelsize=cls.Y_TICK_LABEL_SIZE, labelleft=show_tick_labels)
        cls._style_y_tick_labels(ax, show_tick_labels=show_tick_labels)

    @classmethod
    def _style_robot_count_axis(cls, ax: plt.Axes, show_xlabel: bool = True) -> None:
        ticks = tuple(float(num_agents) for num_agents in cls.ROBOT_COUNT_VALUES)
        if not ticks:
            return
        ax.set_xlim(min(ticks) - 0.5, max(ticks) + 0.5)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels(cls.ROBOT_COUNT_TICK_LABELS)
        ax.set_xlabel("robots" if show_xlabel else "", fontsize=cls.X_AXIS_LABEL_SIZE)
        ax.tick_params(axis="x", labelsize=cls.X_TICK_LABEL_SIZE, labelbottom=show_xlabel)

    @classmethod
    def _style_metric_axis(
        cls,
        ax: plt.Axes,
        values: Sequence[float],
        yscale: str | None = None,
        show_tick_labels: bool = True,
        field: str | None = None,
        domain_key: str | None = None,
    ) -> None:
        if yscale is not None:
            ax.set_yscale(yscale)
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_locator(NullLocator())
        if yscale is not None and values:
            ax.set_ylim(cls._positive_log_axis_limits(values))
        else:
            cls._set_padded_data_ylim(ax, values)
        cls._apply_metric_y_axis_overrides(ax, field=field, domain_key=domain_key, yscale=yscale)
        ax.grid(alpha=0.24, linewidth=0.7)
        ax.tick_params(axis="y", labelsize=cls.Y_TICK_LABEL_SIZE, labelleft=show_tick_labels)
        cls._style_y_tick_labels(ax, show_tick_labels=show_tick_labels)

    @classmethod
    def _apply_metric_y_axis_overrides(
        cls,
        ax: plt.Axes,
        field: str | None,
        domain_key: str | None,
        yscale: str | None,
    ) -> None:
        if field is None or domain_key is None:
            return
        limits = cls.METRIC_Y_LIMITS_BY_DOMAIN.get(field, {}).get(domain_key)
        if limits is not None:
            low, high = limits
            if yscale == "log" and (low <= 0.0 or high <= 0.0):
                raise ValueError(f"Log-scale y limits must be positive for {field}/{domain_key}: {limits!r}.")
            ax.set_ylim(*limits)
        labels = cls.METRIC_Y_TICK_LABELS_BY_DOMAIN.get(field, {}).get(domain_key)
        if labels is None:
            return
        ticks = tuple(float(label) for label in labels)
        ax.set_yticks(ticks, labels=labels)

    @staticmethod
    def _set_padded_data_ylim(ax: plt.Axes, values: Sequence[float]) -> None:
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
        ax.set_ylim(low - padding, high + padding)

    @classmethod
    def _style_y_tick_labels(cls, ax: plt.Axes, show_tick_labels: bool = True) -> None:
        ax.tick_params(axis="y", labelrotation=90, pad=cls.Y_TICK_LABEL_PAD, labelleft=show_tick_labels)

class MRMPPerformanceComparisonReportCLI:
    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Plot MRMP performance-comparison curves over robot counts."
        )
        parser.add_argument(
            "--results-root",
            type=Path,
            default=MRMPPerformanceComparisonReport.DEFAULT_RESULTS_ROOT,
        )
        parser.add_argument(
            "--output-prefix",
            type=Path,
            default=MRMPPerformanceComparisonReport.DEFAULT_OUTPUT_PREFIX,
        )
        parser.add_argument("--formats", nargs="+", default=("png", "pdf"))
        parser.add_argument("--budget", type=float, default=None)
        parser.add_argument(
            "--domain",
            choices=("all", *MRMPPerformanceComparisonReport.DOMAINS),
            default="all",
        )
        parser.add_argument(
            "--planner",
            nargs="+",
            choices=MRMPPerformanceComparison.PLANNER_SELECTION_KEYS,
            default=list(MRMPPerformanceComparisonReport.PLANNER_KEYS),
        )
        return parser.parse_args()

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        domains = MRMPPerformanceComparisonReport.domains_for_arg(args.domain)
        planner_keys = MRMPPerformanceComparisonReport.planner_keys_for_arg(args.planner)
        rows_by_planner = MRMPPerformanceComparisonReport.load_rows_by_planner(
            args.results_root,
            domains,
            planner_keys,
        )
        budget = MRMPPerformanceComparisonReport.select_budget(rows_by_planner, args.budget)
        row_counts = MRMPPerformanceComparisonReport.budget_row_counts(rows_by_planner, budget)
        num_rows = sum(row_counts.values())
        if num_rows == 0:
            raise RuntimeError(
                f"No MRMP performance-comparison rows found for {planner_keys} at budget={budget:g}."
            )
        num_common_instances = len(MRMPPerformanceComparisonReport.common_instance_ids(rows_by_planner, budget))
        count_text = MRMPPerformanceComparisonReport.format_budget_row_counts(row_counts)
        print(
            f"Plotting budget={budget:g} with partial MRMP performance rows "
            f"(common={num_common_instances}, rows: {count_text})."
        )
        MRMPPerformanceComparisonReport.plot_figure(
            rows_by_planner,
            planner_keys,
            budget,
            args.output_prefix,
            args.formats,
            domains=domains,
        )


if __name__ == "__main__":
    MRMPPerformanceComparisonReportCLI.main()
