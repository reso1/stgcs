from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Sequence

from experiments.reports.style import configure_matplotlib_for_latex

configure_matplotlib_for_latex(backend="Agg")

from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

from experiments.mrmp import MRMPExperiment
from experiments.manifests.mrmp import load_manifest
from experiments.planners.mrmp import WindowedCoordinationAblation, WindowedCoordinationSpec
from experiments.reports.common import (
    BoxplotStyle, PlotPalette, ViolinStyle, _save_figure, _set_square_cells, dedupe_result_rows, plt,
)


class MRMPWindowedCoordinationReport:
    DEFAULT_WINDOWED_RESULTS_ROOT = Path(WindowedCoordinationAblation.DEFAULT_OUTPUT_ROOT)
    DEFAULT_OUTPUT_PREFIX = Path("latex/figs/mrmp_windowed_coordination")
    DOMAINS = ("grid2d", "maze", "iris-2d")
    DOMAIN_LABELS = {
        "grid2d": "rand",
        "maze": "maze",
        "iris-2d": "iris",
    }
    REFERENCE_KEY = "pbs_nc"
    DEFAULT_SPAN_FACTORS = WindowedCoordinationAblation.DEFAULT_SPAN_FACTORS
    # One SAGE manuscript column: (495.07794 pt - 12 pt) / 2 / 72.27 pt/in.
    FIGURE_SIZE = (3.34217, 3.65)
    AXIS_FONT_SIZE = 8.0
    LEGEND_FONT_SIZE = 6.0
    # Matplotlib fills legend entries down columns.
    LEGEND_ORDER = ("D1", "D2", "D4", "D8", "F2", "F4", "D16", "H2", "H4")
    CURVE_LINEWIDTH = 0.9
    AXIS_LINEWIDTH = 0.55
    GRID_LINEWIDTH = 0.45
    METRIC_LINEWIDTH = 0.5
    MEDIAN_LINEWIDTH = 1.2
    SPAN_COLORS = (
        PlotPalette.HEURISTIC_TRIPLET_RELAXATION,
        PlotPalette.HEURISTIC_INTERFACE_TO_SET_TABLE,
        PlotPalette.HEURISTIC_MOTION_ONLY,
        "#1d8e9b",
        "#e69f00",
        PlotPalette.BLACK,
    )
    SPAN_COLOR_BANDS = {
        2.0: {
            "dynamic": PlotPalette.BLACK,
            "fixed": "#505050",
            "dynamic_beta": "#919191",
        },
        4.0: {
            "dynamic": PlotPalette.HEURISTIC_MOTION_ONLY,
            "fixed": "#245b35",
            "dynamic_beta": "#81b791",
        },
    }
    SUCCESS_RATE_TICKS = (0.0, 0.5, 1.0)
    SUCCESS_RATE_Y_LIMITS = (-0.05, 1.0)
    X_TICK_LABEL_SIZE = 7.5
    PLANNER_TICK_LABEL_SIZE = 7.5
    Y_TICK_LABEL_SIZE = 7.5
    Y_TICK_LABEL_PAD = 1.5
    METRIC_VIOLIN_WIDTH = 0.7
    METRIC_SCATTER_SIZE = 3.5
    METRIC_SCATTER_ALPHA = 0.65
    SUCCESS_RUNTIME_X_TICKS = (1, 5, 10, 30, 60, 120, 300, 600)
    SUCCESS_RUNTIME_LINESTYLES = {
        "dynamic": "-",
        "fixed": "--",
        "dynamic_beta": "-.",
    }
    STYLE_ABBREV_PREFIXES = {
        "fixed": "F",
        "dynamic": "D",
        "dynamic_beta": "H",
    }

    @classmethod
    def domains_for_arg(cls, domain: str) -> tuple[str, ...]:
        if domain == "all":
            return cls.DOMAINS
        return (domain,)

    @staticmethod
    def _safe_result_name(planner_name: str) -> str:
        return planner_name.replace("/", "_")


    @classmethod
    def windowed_result_path(
        cls,
        windowed_results_root: str | Path,
        domain_key: str,
        spec: WindowedCoordinationSpec,
    ) -> Path:
        safe_name = cls._safe_result_name(WindowedCoordinationAblation.planner_name(spec))
        return Path(windowed_results_root) / domain_key / "results" / f"{safe_name}.csv"

    @classmethod
    def load_rows_by_planner(
        cls,
        results_root: str | Path,
        domains: Sequence[str],
        specs: Sequence[WindowedCoordinationSpec],
    ) -> Dict[str, list[Dict[str, object]]]:
        rows_by_planner: Dict[str, list[Dict[str, object]]] = {}
        for spec in specs:
            rows: list[Dict[str, object]] = []
            for domain in domains:
                path = cls.windowed_result_path(results_root, domain, spec)
                if path.exists():
                    rows.extend(cls._load_domain_rows(path, domain))
            rows_by_planner[spec.key] = dedupe_result_rows(rows)
        return rows_by_planner

    @classmethod
    def load_completed_results(
        cls, root: Path, budget: float | None = None, planner_keys: Sequence[str] | None = None,
    ) -> tuple[tuple[WindowedCoordinationSpec, ...], Dict[str, list[Dict[str, object]]], float, list[str]]:
        manifest = load_manifest(root / "manifest.json")
        expected = {record.instance_id: record for record in manifest}
        if not expected or len(expected) != len(manifest):
            raise ValueError("Empty or duplicated windowed-coordination manifest")
        names = sorted({path.stem for domain in cls.DOMAINS
                        for path in (root / domain / "results").glob('*.csv')})
        run_specs = tuple(WindowedCoordinationAblation.spec_from_planner_name(name) for name in names)
        selected = {spec.key: spec for spec in run_specs if planner_keys is None or spec.key in planner_keys}
        if planner_keys is not None and selected.keys() != set(planner_keys):
            raise ValueError(f"Planner result CSVs are missing from {root}")
        if not selected:
            raise ValueError(f"No windowed planner result CSVs in {root}")
        loaded = cls.load_rows_by_planner(root, cls.DOMAINS, tuple(selected.values()))
        budget = cls.select_budget(loaded, budget)
        specs_by_key: dict[str, WindowedCoordinationSpec] = {}
        rows_by_planner: Dict[str, list[Dict[str, object]]] = {}
        incomplete: list[str] = []
        for key, spec in selected.items():
            rows = cls.planner_budget_rows(loaded, budget, key)
            if len(rows) != len(expected) or {str(row["instance_id"]) for row in rows} != expected.keys():
                incomplete.append(key)
                continue
            if any(row["num_agents"] != expected[str(row["instance_id"])].num_agents
                   or row["domain_key"] != expected[str(row["instance_id"])].domain_key
                   for row in rows):
                raise ValueError(f"Results for {key} do not match the benchmark manifest")
            specs_by_key[key] = spec
            rows_by_planner[key] = rows
        specs = tuple(sorted(specs_by_key.values(), key=cls._success_runtime_sort_key))
        return specs, rows_by_planner, budget, incomplete


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
    def rows_for_domains(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        domain_keys: Sequence[str],
    ) -> Dict[str, Sequence[Dict[str, object]]]:
        domain_key_set = set(domain_keys)
        return {
            planner_key: [
                row for row in rows
                if cls.row_domain_key(row) in domain_key_set
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

    @classmethod
    def planner_keys_for_specs(cls, specs: Sequence[WindowedCoordinationSpec]) -> tuple[str, ...]:
        keys = []
        seen_keys = set()
        for spec in specs:
            if spec.key in seen_keys:
                continue
            keys.append(spec.key)
            seen_keys.add(spec.key)
        return tuple(keys)

    @staticmethod
    def rows_for_planner_keys(
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        planner_keys: Sequence[str],
    ) -> Dict[str, Sequence[Dict[str, object]]]:
        return {key: rows_by_planner.get(key, []) for key in planner_keys}

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
        budget_sets = [cls._positive_budgets(rows) for rows in rows_by_planner.values()]
        if budget_sets and all(budget_sets):
            common_budgets = set.intersection(*budget_sets)
            if common_budgets:
                return max(common_budgets)
        observed_budgets: set[float] = set()
        for rows in rows_by_planner.values():
            observed_budgets.update(cls._positive_budgets(rows))
        if observed_budgets:
            return max(observed_budgets)
        return 300.0

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
        planner_keys = set(rows_by_planner)
        return {
            instance_id
            for instance_id, planner_rows in cls.rows_by_instance(rows_by_planner, budget).items()
            if planner_keys.issubset(planner_rows)
        }

    @classmethod
    def common_evaluated_rows(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
    ) -> Dict[str, list[Dict[str, object]]]:
        instance_ids = cls.common_instance_ids(rows_by_planner, budget)
        if not instance_ids:
            raise ValueError("No instances have completed for every selected planner at this budget.")
        return {
            key: [row for row in rows if cls._same_budget(row, budget)
                  and str(row["instance_id"]) in instance_ids]
            for key, rows in rows_by_planner.items()
        }

    @classmethod
    def common_success_instance_ids(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
    ) -> set[str]:
        planner_keys = set(rows_by_planner)
        return {
            instance_id
            for instance_id, planner_rows in cls.rows_by_instance(rows_by_planner, budget).items()
            if planner_keys.issubset(planner_rows)
            and all(bool(planner_rows[key]["is_success"])
                    and 0.0 < float(planner_rows[key]["runtime"]) <= budget
                    for key in planner_keys)
        }

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
    def _curve_runtime(cls, row: Dict[str, object], budget: float) -> float | None:
        runtime = cls._finite_positive(row["runtime"])
        if runtime is not None:
            return min(runtime, float(budget)) if math.isfinite(float(budget)) else runtime
        if not bool(row["is_success"]):
            return cls._finite_positive(budget)
        return None

    @classmethod
    def success_rate_points(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_key: str,
    ) -> tuple[list[float], list[float]]:
        rows = cls.planner_budget_rows(rows_by_planner, budget, planner_key)
        if not rows:
            return [], []

        events: dict[float, int] = {}
        for row in rows:
            x_value = cls._curve_runtime(row, budget)
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
    def success_rate(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_key: str,
    ) -> float | None:
        rows = cls.planner_budget_rows(rows_by_planner, budget, planner_key)
        if not rows:
            return None
        return sum(1 for row in rows if bool(row["is_success"])) / len(rows)

    @classmethod
    def median_field(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_key: str,
        field: str,
        success_only: bool = False,
    ) -> float | None:
        values = cls.field_values(rows_by_planner, budget, planner_key, field, success_only=success_only)
        if not values:
            return None
        return statistics.median(values)

    @classmethod
    def field_values(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_key: str,
        field: str,
        success_only: bool = False,
    ) -> list[float]:
        rows = cls.planner_budget_rows(rows_by_planner, budget, planner_key)
        values: list[float] = []
        for row in rows:
            if success_only and not bool(row["is_success"]):
                continue
            value = cls._finite_positive(row[field])
            if value is not None:
                values.append(value)
        return values

    @classmethod
    def plot_figure(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        specs: Sequence[WindowedCoordinationSpec],
        budget: float,
        output_prefix: Path,
        formats: Sequence[str],
        domains: Sequence[str] | None = None,
    ) -> None:
        domain_keys = tuple(domains) if domains is not None else cls.domains_from_rows(rows_by_planner)
        if not domain_keys:
            domain_keys = cls.DOMAINS
        span_factors = cls._span_factors(specs)
        comparison_specs = tuple(specs)
        planner_keys = cls.planner_keys_for_specs(comparison_specs)
        comparison_rows_by_planner = cls.rows_for_planner_keys(rows_by_planner, planner_keys)
        merged_rows_by_planner = cls.rows_for_domains(comparison_rows_by_planner, domain_keys)
        merged_rows_by_planner = cls.common_evaluated_rows(merged_rows_by_planner, budget)
        num_instances = len(cls.common_instance_ids(merged_rows_by_planner, budget))
        common_success_ids = cls.common_success_instance_ids(merged_rows_by_planner, budget)
        successful_rows_by_planner = {
            key: [row for row in rows if str(row["instance_id"]) in common_success_ids]
            for key, rows in merged_rows_by_planner.items()
        }
        fig, axes = plt.subplots(2, 2, figsize=cls.FIGURE_SIZE, squeeze=False)
        _set_square_cells(axes)
        cls._plot_success_by_runtime_panel(
            axes[0, 0],
            merged_rows_by_planner,
            comparison_specs,
            budget,
            span_factors,
            show_labels=True,
            show_xlabel=True,
            show_ylabel=True,
        )
        metric_panels = (
            (axes[0, 1], "runtime", "success runtime (s)"),
            (axes[1, 0], "sum_of_costs", "sum-of-costs"),
            (axes[1, 1], "makespan", "makespan"),
        )
        for ax, field, xlabel in metric_panels:
            cls._plot_metric_by_planner_panel(
                ax,
                successful_rows_by_planner,
                comparison_specs,
                budget,
                field=field,
                xlabel=xlabel,
            )

        for ax in axes.flat:
            for spine in ax.spines.values():
                spine.set_linewidth(cls.AXIS_LINEWIDTH)
            ax.tick_params(which="major", width=cls.AXIS_LINEWIDTH, length=2.5, pad=1.5)
            ax.tick_params(which="minor", width=cls.AXIS_LINEWIDTH, length=1.5)
            ax.xaxis.labelpad = 2.0
            ax.yaxis.labelpad = 2.0

        handles, labels = cls.legend_handles_labels(axes)
        layout_top = 1.0
        if handles:
            legend_columns = min(3, len(handles))
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=legend_columns,
                frameon=False,
                fontsize=cls.LEGEND_FONT_SIZE,
                borderaxespad=0.2,
                handlelength=1.3,
                handletextpad=0.25,
                labelspacing=0.18,
                columnspacing=0.8,
            )
            legend_rows = math.ceil(len(handles) / legend_columns)
            legend_height = (1.2 * legend_rows + 0.8) * cls.LEGEND_FONT_SIZE / 72.0
            layout_top = 1.0 - legend_height / cls.FIGURE_SIZE[1]
        fig.tight_layout(rect=(0.0, 0.0, 1.0, layout_top), pad=0.35, w_pad=0.75, h_pad=0.9)
        _save_figure(fig, output_prefix, formats, bbox_inches=None)
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "budget_seconds": budget,
            "cohort": "instances evaluated by every selected planner at the same budget",
            "quality_cohort": "instances solved by every selected planner within the budget",
            "instance_ids": sorted(cls.common_instance_ids(merged_rows_by_planner, budget)),
            "common_success_instance_ids": sorted(common_success_ids),
            "domain_counts": {domain: len(cls.rows_for_domain(merged_rows_by_planner, domain)[planner_keys[0]])
                              for domain in domain_keys},
            "available_rows": cls.budget_row_counts(comparison_rows_by_planner, budget),
            "planners": {spec.key: {
                "alpha": "inf" if math.isinf(spec.window_span_factor) else spec.window_span_factor,
                "mode": spec.mode_key,
                "beta": spec.execution_horizon_factor,
                "evaluated": num_instances,
                "solved": sum(bool(row["is_success"]) for row in merged_rows_by_planner[spec.key]),
                "success_rate": cls.success_rate(merged_rows_by_planner, budget, spec.key),
                "common_success_medians": {
                    field: cls.median_field(successful_rows_by_planner, budget, spec.key, field)
                    for field in ("runtime", "sum_of_costs", "makespan")
                },
            } for spec in specs},
        }
        output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")

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
        order = {name: index for index, name in enumerate(cls.LEGEND_ORDER)}
        labels = sorted(handles_by_label, key=lambda label: order.get(label.split(":")[0], len(order)))
        return [handles_by_label[label] for label in labels], labels


    @staticmethod
    def _set_x_ticks(axes: Sequence[plt.Axes], x_ticks: Sequence[float] | None) -> None:
        if x_ticks is None:
            return
        ticks = [float(tick) for tick in x_ticks]
        for ax in axes:
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{float(value):g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())


    @classmethod
    def _plot_metric_by_planner_panel(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        specs: Sequence[WindowedCoordinationSpec],
        budget: float,
        field: str,
        xlabel: str,
    ) -> None:
        tick_labels_by_position = cls._plot_windowed_violins(
            ax,
            rows_by_planner,
            specs,
            budget,
            field=field,
            orientation="horizontal",
        )
        positions = sorted(tick_labels_by_position)
        ax.set_yticks(positions, [tick_labels_by_position[position] for position in positions])
        ax.set_ylim(max(positions) + 0.65, min(positions) - 0.65)
        ax.set_xlabel(xlabel, fontsize=cls.AXIS_FONT_SIZE)
        ax.set_ylabel("")
        if any(rows_by_planner.values()):
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(
                FixedLocator((10, 20, 50, 120)) if field == "runtime"
                else LogLocator(base=10, subs=(1, 2, 5))
            )
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.text(0.5, 0.5, "No common successes", ha="center", va="center",
                    transform=ax.transAxes, fontsize=cls.AXIS_FONT_SIZE)
            ax.set_xticks([])
        ax.grid(axis="x", alpha=0.24, linewidth=cls.GRID_LINEWIDTH)
        ax.tick_params(axis="x", labelsize=cls.X_TICK_LABEL_SIZE)
        ax.tick_params(axis="y", labelsize=cls.PLANNER_TICK_LABEL_SIZE)

    @classmethod
    def _plot_success_by_runtime_panel(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        specs: Sequence[WindowedCoordinationSpec],
        budget: float,
        span_factors: Sequence[float],
        show_labels: bool,
        show_xlabel: bool = True,
        show_ylabel: bool = True,
    ) -> None:
        runtime_values: list[float] = []
        for spec in sorted(specs, key=cls._success_runtime_sort_key):
            xs, ys = cls.success_rate_curve_points(rows_by_planner, budget, spec.key)
            if not xs:
                continue
            runtime_values.extend(xs)
            ax.step(
                xs,
                ys,
                where="post",
                label=cls.success_runtime_label(spec) if show_labels else None,
                color=cls.span_color(spec, span_factors),
                linestyle=cls._success_runtime_linestyle(spec),
                linewidth=cls.CURVE_LINEWIDTH,
            )
        ax.set_xscale("log")
        if runtime_values:
            success_runtimes = [
                runtime for spec in specs
                for runtime in cls.field_values(
                    rows_by_planner, budget, spec.key, "runtime", success_only=True,
                )
                if runtime <= budget
            ]
            lower = min(success_runtimes, default=min(runtime_values))
            ax.set_xlim(lower, budget)
            ticks = sorted({float(tick) for tick in cls.SUCCESS_RUNTIME_X_TICKS
                            if lower <= tick <= budget} | {lower, float(budget)})
            cls._set_x_ticks((ax,), ticks)
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda value, _position: "" if math.isclose(value, lower) else f"{value:g}"
            ))
        ax.set_xlabel("runtime (s)" if show_xlabel else "", fontsize=cls.AXIS_FONT_SIZE)
        ax.set_ylabel("success rate" if show_ylabel else "", fontsize=cls.AXIS_FONT_SIZE)
        ax.tick_params(axis="x", labelsize=cls.X_TICK_LABEL_SIZE)
        cls._style_success_axis(ax, show_tick_labels=show_ylabel)

    @classmethod
    def success_rate_curve_points(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        budget: float,
        planner_key: str,
    ) -> tuple[list[float], list[float]]:
        xs, ys = cls.success_rate_points(rows_by_planner, budget, planner_key)
        if not xs:
            return [], []
        if math.isfinite(float(budget)) and float(budget) > 0.0 and not math.isclose(xs[-1], float(budget)):
            xs.append(float(budget))
            ys.append(ys[-1])
        left_anchor = cls._success_runtime_left_anchor([
            runtime for rows in rows_by_planner.values() for row in rows
            if cls._same_budget(row, budget)
            and (runtime := cls._curve_runtime(row, budget)) is not None
        ])
        if left_anchor < xs[0]:
            xs = [left_anchor, *xs]
            ys = [0.0, *ys]
        return xs, ys

    @staticmethod
    def _success_runtime_left_anchor(xs: Sequence[float]) -> float:
        positive_xs = [float(value) for value in xs if float(value) > 0.0 and math.isfinite(float(value))]
        if not positive_xs:
            return 1e-3
        return max(min(positive_xs) / 2.0, 1e-3)

    @classmethod
    def success_runtime_label(cls, spec: WindowedCoordinationSpec) -> str:
        abbreviation = cls._style_abbrev(cls._style_key_for_spec(spec), spec.window_span_factor)
        if math.isinf(spec.window_span_factor):
            return rf"{abbreviation}: $\alpha=\infty$ (full)"
        label = rf"{spec.mode_key}, $\alpha={spec.window_span_factor:g}$"
        if not math.isclose(spec.execution_horizon_factor, 1.0):
            label += rf", $\beta={spec.execution_horizon_factor:g}$"
        return f"{abbreviation}: {label}"

    @classmethod
    def span_color(cls, spec: WindowedCoordinationSpec, span_factors: Sequence[float]) -> str:
        if math.isinf(spec.window_span_factor):
            return PlotPalette.BLACK
        if spec.window_span_factor in cls.SPAN_COLOR_BANDS:
            return cls.SPAN_COLOR_BANDS[spec.window_span_factor][cls._style_key_for_spec(spec)]
        index = sorted(set(span_factors)).index(spec.window_span_factor)
        return cls.SPAN_COLORS[index % (len(cls.SPAN_COLORS) - 1)]

    @classmethod
    def _style_abbrev(cls, style_key: str, window_span_factor: float) -> str:
        prefix = cls.STYLE_ABBREV_PREFIXES[style_key]
        alpha_label = WindowedCoordinationSpec.factor_label(window_span_factor)
        return f"{prefix}{alpha_label}"

    @classmethod
    def _success_runtime_linestyle(
        cls,
        spec: WindowedCoordinationSpec,
    ) -> str:
        return cls.SUCCESS_RUNTIME_LINESTYLES[cls._style_key_for_spec(spec)]

    @classmethod
    def _success_runtime_sort_key(cls, spec: WindowedCoordinationSpec) -> tuple[int, float]:
        style_order = {"dynamic": 0, "fixed": 1, "dynamic_beta": 2}
        return (style_order[cls._style_key_for_spec(spec)], float(spec.window_span_factor))

    @classmethod
    def _plot_windowed_violins(
        cls,
        ax: plt.Axes,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        specs: Sequence[WindowedCoordinationSpec],
        budget: float,
        field: str,
        orientation: str = "horizontal",
        variant_positions: Mapping[tuple[float, str], float] | None = None,
    ) -> dict[float, str]:
        tick_labels_by_position: dict[float, str] = {}
        span_factors = cls._span_factors(specs)
        for index, spec in enumerate(sorted(specs, key=cls._success_runtime_sort_key)):
            slot = (float(spec.window_span_factor), cls._style_key_for_spec(spec))
            position = variant_positions[slot] if variant_positions is not None else float(index)
            tick_labels_by_position[position] = cls._style_abbrev(
                cls._style_key_for_spec(spec), spec.window_span_factor,
            )
            values = cls.field_values(rows_by_planner, budget, spec.key, field, success_only=True)
            if not values:
                continue
            color = cls.span_color(spec, span_factors)
            ViolinStyle.draw(
                ax, position, values, color, width=cls.METRIC_VIOLIN_WIDTH,
                linewidth=cls.METRIC_LINEWIDTH, median_linewidth=cls.MEDIAN_LINEWIDTH,
                orientation=orientation,
            )
            BoxplotStyle.scatter_points(
                ax, position, values, width=cls.METRIC_VIOLIN_WIDTH, color=color,
                size=cls.METRIC_SCATTER_SIZE, alpha=cls.METRIC_SCATTER_ALPHA,
                orientation=orientation,
            )
        return tick_labels_by_position


    @classmethod
    def _style_key_for_spec(cls, spec: WindowedCoordinationSpec) -> str:
        if spec.mode_key == "dynamic" and not math.isclose(float(spec.execution_horizon_factor), 1.0):
            return "dynamic_beta"
        return spec.mode_key


    @staticmethod
    def _span_factors(specs: Sequence[WindowedCoordinationSpec]) -> list[float]:
        return sorted({float(spec.window_span_factor) for spec in specs})


    @staticmethod
    def _positive_log_axis_limits(values: Sequence[float]) -> tuple[float, float]:
        positive_factors = [float(value) for value in values if float(value) > 0.0 and math.isfinite(float(value))]
        if not positive_factors:
            return (1.0, 2.0)
        low = min(positive_factors)
        high = max(positive_factors)
        if math.isclose(low, high):
            return (low / math.sqrt(2.0), high * math.sqrt(2.0))
        log_low = math.log2(low)
        log_high = math.log2(high)
        padding = 0.08 * (log_high - log_low)
        return (2 ** (log_low - padding), 2 ** (log_high + padding))


    @classmethod
    def _style_success_axis(cls, ax: plt.Axes, show_tick_labels: bool = True) -> None:
        ax.set_ylim(*cls.SUCCESS_RATE_Y_LIMITS)
        ax.yaxis.set_major_locator(FixedLocator(cls.SUCCESS_RATE_TICKS))
        ax.yaxis.set_major_formatter(FuncFormatter(cls._success_rate_tick_label))
        ax.grid(alpha=0.24, linewidth=cls.GRID_LINEWIDTH)
        ax.tick_params(axis="y", labelsize=cls.Y_TICK_LABEL_SIZE, labelleft=show_tick_labels)
        cls._style_y_tick_labels(ax, show_tick_labels=show_tick_labels)

    @classmethod
    def _style_y_tick_labels(
        cls,
        ax: plt.Axes,
        show_tick_labels: bool = True,
        rotation: float = 90.0,
    ) -> None:
        ax.tick_params(
            axis="y",
            labelrotation=rotation,
            pad=cls.Y_TICK_LABEL_PAD,
            labelleft=show_tick_labels,
        )

    @staticmethod
    def _success_rate_tick_label(value: float, _position: int) -> str:
        if math.isclose(float(value), 0.0):
            return "0%"
        return f"{float(value):.0%}"


class MRMPWindowedCoordinationReportCLI:
    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Plot MRMP windowed-coordination ablation results."
        )
        parser.add_argument(
            "--results-root",
            type=Path,
            default=MRMPWindowedCoordinationReport.DEFAULT_WINDOWED_RESULTS_ROOT,
            help="Results folder containing manifest.json and per-domain CSVs.",
        )
        parser.add_argument(
            "--output-prefix",
            type=Path,
            default=MRMPWindowedCoordinationReport.DEFAULT_OUTPUT_PREFIX,
        )
        parser.add_argument("--formats", nargs="+", default=("png", "pdf"))
        parser.add_argument("--budget", type=float, default=None)
        parser.add_argument(
            "--domain",
            choices=("all", *MRMPWindowedCoordinationReport.DOMAINS),
            default="all",
        )
        parser.add_argument(
            "--span-factors",
            type=float,
            nargs="+",
            default=None,
            help="Select dynamic span factors explicitly; otherwise plot all variants present in the CSVs.",
        )
        return parser.parse_args()

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        domains = MRMPWindowedCoordinationReport.domains_for_arg(args.domain)
        planner_keys = None if args.span_factors is None else [
            spec.key for spec in WindowedCoordinationAblation.specs(args.span_factors)]
        specs, rows_by_planner, budget, incomplete = MRMPWindowedCoordinationReport.load_completed_results(
            args.results_root, args.budget, planner_keys,
        )
        if incomplete:
            print(f"Incomplete planners excluded: {', '.join(incomplete)}")
        comparison_specs = tuple(specs)
        comparison_rows_by_planner = MRMPWindowedCoordinationReport.rows_for_planner_keys(
            rows_by_planner, MRMPWindowedCoordinationReport.planner_keys_for_specs(comparison_specs)
        )
        row_counts = MRMPWindowedCoordinationReport.budget_row_counts(comparison_rows_by_planner, budget)
        num_rows = sum(row_counts.values())
        if num_rows == 0:
            raise RuntimeError(
                f"No MRMP rows found for the selected variants at budget={budget:g}."
            )
        num_common_instances = len(
            MRMPWindowedCoordinationReport.common_instance_ids(comparison_rows_by_planner, budget)
        )
        count_text = MRMPWindowedCoordinationReport.format_budget_row_counts(row_counts)
        print(
            f"Plotting budget={budget:g} on shared evaluated instances "
            f"(common={num_common_instances}, rows: {count_text})."
        )
        MRMPWindowedCoordinationReport.plot_figure(
            rows_by_planner,
            specs,
            budget,
            args.output_prefix,
            args.formats,
            domains=domains,
        )
        summary_path = args.output_prefix.with_suffix(".json")
        summary = json.loads(summary_path.read_text())
        summary.update(results_root=str(args.results_root.resolve()), excluded_incomplete=incomplete)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    MRMPWindowedCoordinationReportCLI.main()
