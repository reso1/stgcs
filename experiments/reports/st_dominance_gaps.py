from __future__ import annotations

import argparse
import json
from statistics import median
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

from experiments.reports.style import configure_matplotlib_for_latex

configure_matplotlib_for_latex(backend="Agg")

from matplotlib.ticker import FixedLocator, MaxNLocator, PercentFormatter

from experiments.manifests.st_planning import STPlanningManifestStore
from experiments.reports.common import BoxplotStyle, PlotPalette, _save_figure, plt
from experiments.reports.st_common import STHeuristicAblationGroupReport
from experiments.reports.st_reference_costs import STReferenceCosts


class STDominanceAblationGroupReport(STHeuristicAblationGroupReport):
    DEFAULT_MANIFEST = Path("data/instances/st_planning/manifest.json")
    DEFAULT_RESULTS_ROOT = Path("data/exp_res/st_planning/dominance_ablation")
    DEFAULT_OUTPUT_PREFIX = Path("latex/figs/dc_ablation")

    PLANNERS = (
        "Search(h_max+GUB,gub=h_mot)",
        "Search(h_max+GUB+delta_set,gub=h_mot)",
        "Search(h_max+GUB+delta_state,gub=h_mot)",
        "Search(h_max+GUB+delta_pos,gub=h_mot)",
    )
    FIGURE_PLANNERS = PLANNERS
    QUALITY_FIGURE_PLANNERS = (
        "Search(h_max+GUB+delta_state,gub=h_mot)",
        "Search(h_max+GUB+delta_pos,gub=h_mot)",
    )
    PLANNER_SHORT_LABELS = {
        "Search(h_max+GUB,gub=h_mot)": "UB",
        "Search(h_max+GUB+delta_set,gub=h_mot)": r"$+\delta_\text{set}$",
        "Search(h_max+GUB+delta_pos,gub=h_mot)": r"$+\delta_\text{pos}$",
        "Search(h_max+GUB+delta_state,gub=h_mot)": r"$+\delta_\text{state}$",
    }
    METRIC_FIELDS = {
        "expanded": "num_expanded_nodes",
        "runtime": "runtime",
        "quality": "cost",
    }
    METRIC_LABELS = {
        "expanded": "expanded nodes",
        "runtime": "runtime (s)",
        "quality": "relative cost increase",
    }
    FIGURE_X_LABELS = {
        "expanded": "expanded nodes",
        "runtime": "runtime (s)",
        "quality": "optimality gap",
    }
    FIGURE_METRIC_ORDER = ("runtime", "expanded", "quality")
    FIGURE_GROUP_ORDER = ("grid2d", "maze", "iris2d")
    FIGURE_GROUP_LABELS = {
        "grid2d": "rand",
        "maze": "maze",
        "iris2d": "iris",
    }
    FIGURE_HEURISTIC_COLORS = {
        "Search(h_max+GUB,gub=h_mot)": STHeuristicAblationGroupReport.FIGURE_HEURISTIC_COLORS[
            "Search(h_mot+GUB)"
        ],
        "Search(h_max+GUB+delta_set,gub=h_mot)": STHeuristicAblationGroupReport.FIGURE_HEURISTIC_COLORS[
            "Search(h_tri+GUB)"
        ],
        "Search(h_max+GUB+delta_pos,gub=h_mot)": STHeuristicAblationGroupReport.FIGURE_HEURISTIC_COLORS[
            "Search(h_max+GUB)"
        ],
        "Search(h_max+GUB+delta_state,gub=h_mot)": STHeuristicAblationGroupReport.FIGURE_HEURISTIC_COLORS[
            "Search(h_tab+GUB)"
        ],
    }
    FIGURE_Y_TICKS = {
        "runtime": {
            "grid2d": (1e-1, 1e0, 1e1, 1e2),
            "maze": (1e-1, 1e0, 1e1, 1e2),
            "iris2d": (1e-1, 1e0, 1e1, 1e2),
        },
        "expanded": {
            "grid2d": (1e1, 1e2, 1e3),
            "maze": (1e1, 1e2, 1e3),
            "iris2d": (1e1, 1e2, 1e3),
        },
        "quality": {
            "grid2d": None,
            "maze": None,
            "iris2d": None,
        },
    }
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
        "quality": {
            "grid2d": None,
            "maze": None,
            "iris2d": None,
        },
    }
    FIGURE_Y_SCALES = {
        "runtime": "log",
        "expanded": "log",
        "quality": "linear",
    }
    FIGURE_BOXPLOT_WIDTH = 0.42
    FIGURE_ROW_HEIGHT_RATIOS = (1.0, 1.0, 0.62)
    FIGURE_SHOW_SCATTER_POINTS = True
    FIGURE_SCATTER_SIZE = 10
    FIGURE_SCATTER_ALPHA = 0.62
    FIGURE_BOXPLOT_ORIENTATION = "horizontal"
    QUALITY_ZERO_TOLERANCE = 1e-6
    QUALITY_Y_AXIS_MIN_HIGH = 0.02
    QUALITY_Y_AXIS_PADDING_FRACTION = 0.12
    FIGURE_SIZE = (6, 5)
    REFERENCE_PLANNER = "Search(h_max+GUB+delta_set,gub=h_mot)"
    GROUP_ORDER = ("maze", "grid2d", "iris2d")
    GROUP_LABELS = {
        "maze": "maze",
        "grid2d": "grid2d",
        "iris2d": "Iris/iris2d",
    }

    @classmethod
    def _boxplot_values(
        cls,
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        group: str,
        budget: float,
        metric_field: str,
    ) -> list[list[float]]:
        if metric_field != "cost":
            return super()._boxplot_values(rows_by_planner, group, budget, metric_field)

        common_instance_ids = cls._common_success_instance_ids(
            rows_by_planner,
            group,
            budget,
            metric_field,
            (cls.REFERENCE_PLANNER, *cls.QUALITY_FIGURE_PLANNERS),
        )
        reference_rows = cls._success_rows_by_instance(
            cls._rows_for_group(rows_by_planner.get(cls.REFERENCE_PLANNER, []), group, budget),
            metric_field,
        )
        values_by_planner: list[list[float]] = []
        for planner in cls.QUALITY_FIGURE_PLANNERS:
            planner_rows = cls._success_rows_by_instance(
                cls._rows_for_group(rows_by_planner.get(planner, []), group, budget),
                metric_field,
            )
            values: list[float] = []
            for instance_id in sorted(common_instance_ids):
                reference_value = cls._finite_metric_value(reference_rows[instance_id], metric_field)
                planner_value = cls._finite_metric_value(planner_rows[instance_id], metric_field)
                if reference_value is None or planner_value is None:
                    continue
                relative_increase = planner_value / reference_value - 1.0
                if abs(relative_increase) < cls.QUALITY_ZERO_TOLERANCE:
                    relative_increase = 0.0
                values.append(relative_increase)
            values_by_planner.append(values)
        return values_by_planner

    @classmethod
    def _figure_y_limits(
        cls,
        values: Sequence[float],
        metric_name: str,
        group: str,
    ) -> tuple[float, float]:
        if metric_name != "quality":
            return super()._figure_y_limits(values, metric_name, group)
        limits = cls.FIGURE_Y_LIMITS[metric_name][group]
        if limits is None:
            return cls._quality_axis_limits(values)
        y_low, y_high = limits
        if y_high <= y_low:
            raise ValueError(
                f"Invalid y-limits for {metric_name}/{group}: "
                f"expected low < high, got {limits}"
            )
        return float(y_low), float(y_high)

    @classmethod
    def _quality_axis_limits(cls, values: Sequence[float]) -> tuple[float, float]:
        finite_values = [value for value in values if math.isfinite(value)]
        if not finite_values:
            return -0.01, cls.QUALITY_Y_AXIS_MIN_HIGH
        low = min(min(finite_values), 0.0)
        high = max(max(finite_values), 0.0)
        if math.isclose(low, high):
            high = max(abs(high), cls.QUALITY_Y_AXIS_MIN_HIGH)
            return -0.08 * high, high
        pad = (high - low) * cls.QUALITY_Y_AXIS_PADDING_FRACTION
        y_low = min(low - pad, -0.08 * max(high, cls.QUALITY_Y_AXIS_MIN_HIGH))
        y_high = max(high + pad, cls.QUALITY_Y_AXIS_MIN_HIGH)
        return y_low, y_high

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
        if metric_name != "quality":
            super()._plot_boxplot_axis(
                ax,
                values_by_heuristic,
                metric_name,
                group,
                y_low,
                y_high,
                row_idx,
                col_idx,
            )
            cls._set_boxplot_axis_xlabel(ax, metric_name)
            return
        if cls.FIGURE_BOXPLOT_ORIENTATION == "horizontal":
            cls._plot_horizontal_quality_axis(
                ax,
                values_by_heuristic,
                group,
                y_low,
                y_high,
                row_idx,
                col_idx,
            )
            return
        if cls.FIGURE_BOXPLOT_ORIENTATION != "vertical":
            raise ValueError(f"Unsupported boxplot orientation {cls.FIGURE_BOXPLOT_ORIENTATION!r}.")

        positions = tuple(range(1, len(cls.QUALITY_FIGURE_PLANNERS) + 1))
        nonempty = [
            (position, planner, list(values))
            for position, planner, values in zip(
                positions,
                cls.QUALITY_FIGURE_PLANNERS,
                values_by_heuristic,
            )
            if values
        ]
        if nonempty:
            for position, planner, values in nonempty:
                cls._draw_quality_boxplot(
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

        ax.set_xlim(0.5, len(cls.QUALITY_FIGURE_PLANNERS) + 0.5)
        ax.set_ylim(y_low, y_high)
        ax.set_yscale("linear")
        ticks = cls.FIGURE_Y_TICKS[metric_name][group]
        if ticks is None:
            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=3, steps=(1, 2, 4, 5, 10), min_n_ticks=3)
            )
        else:
            ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.grid(alpha=0.22, linewidth=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [
                cls._quality_tick_label(planner, values)
                for planner, values in zip(cls.QUALITY_FIGURE_PLANNERS, values_by_heuristic)
            ],
            fontsize=cls.FIGURE_COLUMN_TEXT_SIZE,
        )
        if row_idx == 0:
            ax.set_title(cls.FIGURE_GROUP_LABELS[group], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=6)
        if col_idx == 0:
            ax.set_ylabel(cls.METRIC_LABELS[metric_name], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=cls.FIGURE_COLUMN_TEXT_SIZE, labelrotation=90)
        ax.tick_params(axis="x", labelsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=1.5)

    @classmethod
    def _plot_horizontal_quality_axis(
        cls,
        ax: plt.Axes,
        values_by_heuristic: Sequence[Sequence[float]],
        group: str,
        x_low: float,
        x_high: float,
        row_idx: int,
        col_idx: int,
    ) -> None:
        positions = tuple(range(1, len(cls.QUALITY_FIGURE_PLANNERS) + 1))
        nonempty = [
            (position, planner, list(values))
            for position, planner, values in zip(
                positions,
                cls.QUALITY_FIGURE_PLANNERS,
                values_by_heuristic,
            )
            if values
        ]
        if nonempty:
            for position, planner, values in nonempty:
                cls._draw_quality_boxplot(
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
        ax.set_ylim(len(cls.QUALITY_FIGURE_PLANNERS) + 0.5, 0.5)
        ax.set_xscale("linear")
        ticks = cls.FIGURE_Y_TICKS["quality"][group]
        if ticks is None:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=(1, 2, 4, 5, 10), min_n_ticks=3))
        else:
            ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        cls._draw_heuristic_reference_lines(ax, positions)
        ax.grid(axis="x", alpha=0.22, linewidth=0.7)
        ax.set_yticks(positions)
        ax.set_yticklabels(
            [
                cls._quality_tick_label(planner, values)
                for planner, values in zip(cls.QUALITY_FIGURE_PLANNERS, values_by_heuristic)
            ],
            fontsize=cls.FIGURE_COLUMN_TEXT_SIZE,
        )
        if row_idx == 0:
            ax.set_title(cls.FIGURE_GROUP_LABELS[group], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=6)
        cls._set_boxplot_axis_xlabel(ax, "quality")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=1.5)
        ax.tick_params(
            axis="y",
            labelsize=cls.FIGURE_COLUMN_TEXT_SIZE,
            pad=1.5,
            labelleft=col_idx == 0,
        )

    @classmethod
    def _draw_quality_boxplot(
        cls,
        ax: plt.Axes,
        position: int,
        values: Sequence[float],
        color: str,
        orientation: str,
    ) -> None:
        finite_values = [value for value in values if math.isfinite(value)]
        if not finite_values:
            return
        BoxplotStyle.draw(
            ax,
            [finite_values],
            positions=[position],
            widths=cls.FIGURE_BOXPLOT_WIDTH,
            color=color,
            orientation=orientation,
        )
        if cls.FIGURE_SHOW_SCATTER_POINTS:
            BoxplotStyle.scatter_points(
                ax,
                float(position),
                finite_values,
                width=cls.FIGURE_BOXPLOT_WIDTH,
                color=color,
                size=cls.FIGURE_SCATTER_SIZE,
                alpha=cls.FIGURE_SCATTER_ALPHA,
                orientation=orientation,
            )

    @classmethod
    def _set_boxplot_axis_xlabel(cls, ax: plt.Axes, metric_name: str) -> None:
        if cls.FIGURE_BOXPLOT_ORIENTATION == "horizontal":
            ax.set_xlabel(cls.FIGURE_X_LABELS[metric_name], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE)

    @classmethod
    def _quality_tick_label(cls, planner: str, values: Sequence[float]) -> str:
        return cls.PLANNER_SHORT_LABELS[planner]

    @classmethod
    def load_rows_by_planner(
        cls,
        results_root: Path,
        manifest: Path | None = None,
    ) -> Dict[str, list[Dict[str, object]]]:
        rows_by_planner = super().load_rows_by_planner(results_root)
        if manifest is None:
            return rows_by_planner
        instance_ids = {
            record.instance_id
            for record in STPlanningManifestStore.load_manifest(Path(manifest))
        }
        return {
            planner: [
                row
                for row in rows
                if str(row["instance_id"]) in instance_ids
            ]
            for planner, rows in rows_by_planner.items()
        }


    @classmethod
    def report(cls, rows, budget, reference, output_prefix, formats):
        kind, costs = STReferenceCosts.quality_reference(reference)
        summary = dict(budget_seconds=budget, reference_kind=kind, groups={},
                       runtime_pairing="delta_set runtime / variant runtime on paired successes",
                       expansion_pairing="delta_set expansions / variant expansions on paired successes")
        summary['figure'] = dict(layout='horizontal boxplots', groups={})
        fig, axes = plt.subplots(
            3, 3, figsize=cls.FIGURE_SIZE, squeeze=False,
            gridspec_kw=cls._boxplot_gridspec_kwargs(),
        )
        for col, group in enumerate(cls.FIGURE_GROUP_ORDER):
            selected = {p: cls._rows_for_group(rows[p], group, budget) for p in cls.PLANNERS}
            ids = {r['instance_id'] for r in selected[cls.REFERENCE_PLANNER]}
            if not ids or any(len(v) != len(ids) or {r['instance_id'] for r in v} != ids for v in selected.values()):
                raise ValueError(f'Incomplete or mismatched dominance schedule in {group}')
            base = cls._success_rows_by_instance(selected[cls.REFERENCE_PLANNER], 'runtime')
            comparisons = {}
            for planner in cls.PLANNERS:
                successful = cls._success_rows_by_instance(selected[planner], 'runtime')
                paired = sorted(base.keys() & successful.keys())
                runtime = [base[i]['runtime'] / successful[i]['runtime'] for i in paired]
                expanded = [base[i]['num_expanded_nodes'] / successful[i]['num_expanded_nodes']
                            for i in paired if successful[i]['num_expanded_nodes'] > 0]
                quality_ids = sorted(successful.keys() & costs.keys())
                quality = STReferenceCosts.gaps(selected[planner], reference)
                comparisons[planner] = dict(solved=len(successful), total=len(ids), paired_n=len(paired),
                    paired_instance_ids=paired, median_runtime_speedup=median(runtime) if runtime else None,
                    expansion_n=len(expanded), median_expansion_ratio=median(expanded) if expanded else None,
                    quality_n=len(quality_ids), quality_instance_ids=quality_ids, quality_gaps=quality)
            summary['groups'][group] = comparisons
            panels = {}
            for row, metric in enumerate(cls.FIGURE_METRIC_ORDER):
                field = cls.METRIC_FIELDS[metric]
                planners = cls.QUALITY_FIGURE_PLANNERS if metric == 'quality' else cls.FIGURE_PLANNERS
                matched_planners = (cls.REFERENCE_PLANNER, *planners) if metric == 'quality' else planners
                common_ids = cls._common_success_instance_ids(rows, group, budget, field, matched_planners)
                if metric == 'quality':
                    common_ids &= costs.keys()
                    values = [STReferenceCosts.gaps(
                        [r for r in selected[p] if r['instance_id'] in common_ids], reference,
                    ) for p in planners]
                else:
                    values = cls._boxplot_values(rows, group, budget, field)
                panels[metric] = dict(
                    instance_ids=sorted(common_ids),
                    values_by_planner=dict(zip(planners, values)),
                )
                ax = axes[row, col]
                limits = cls._figure_y_limits([v for series in values for v in series], metric, group)
                cls._plot_boxplot_axis(ax, values, metric, group, *limits, row, col)
            summary['figure']['groups'][group] = panels
        fig.tight_layout(pad=.35, w_pad=.35, h_pad=.35)
        _save_figure(fig, output_prefix, formats)
        output_prefix.with_suffix('.json').write_text(json.dumps(summary, indent=2))
        return summary


class STDominanceAblationGroupReportCLI:
    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=(
                "Generate domain-wise runtime, expanded-node, and relative-cost panels "
                "for the ST dominance-check ablation."
            )
        )
        parser.add_argument(
            "--results-root",
            type=Path,
            default=STDominanceAblationGroupReport.DEFAULT_RESULTS_ROOT,
        )
        parser.add_argument(
            "--manifest",
            type=Path,
            default=STDominanceAblationGroupReport.DEFAULT_MANIFEST,
            help="Filter plotted rows to instance IDs in this manifest.",
        )
        parser.add_argument(
            "--output-prefix",
            type=Path,
            default=STDominanceAblationGroupReport.DEFAULT_OUTPUT_PREFIX,
        )
        parser.add_argument("--formats", nargs="+", default=("png", "pdf"))
        parser.add_argument("--budget", type=float, default=100)
        return parser.parse_args()

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        rows_by_planner = STDominanceAblationGroupReport.load_rows_by_planner(
            args.results_root,
            args.manifest,
        )
        budget = STDominanceAblationGroupReport.select_budget(rows_by_planner, args.budget)
        STDominanceAblationGroupReport.report(
            rows_by_planner,
            budget,
            STReferenceCosts.from_rows(rows_by_planner, budget),
            args.output_prefix,
            args.formats,
        )


if __name__ == "__main__":
    STDominanceAblationGroupReportCLI.main()
