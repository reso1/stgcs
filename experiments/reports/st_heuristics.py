"""Heuristic distributions with UB + delta_state and a reporting cutoff."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.reports.st_common import STHeuristicAblationGroupReport
from experiments.reports.common import PlotPalette, ViolinStyle, _save_figure, plt
from experiments.planners.st_planning import STExperimentConfig
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter


class STHeuristicViolinReport(STHeuristicAblationGroupReport):
    FIGURE_SIZE = (6, 5)
    FIGURE_VIOLIN_WIDTH = 0.85
    SPECS = tuple(STExperimentConfig.heuristic_specs(("GUB", "delta_state")))
    FIGURE_PLANNERS = tuple(spec.name for spec in SPECS)
    PLANNER_SHORT_LABELS = {
        spec.name: "joint" if spec.key_evaluation == "joint"
        else rf"$h_\mathrm{{{spec.heuristic[2:]}}}$"
        for spec in SPECS
    }
    FIGURE_HEURISTIC_COLORS = {
        spec.name: "#8c6bb1" if spec.key_evaluation == "joint"
        else PlotPalette.heuristic_color(spec.heuristic)
        for spec in SPECS
    }
    METRIC_X_TICKS = {
        "runtime": {"grid2d": (.2, 1, 5), "maze": (.5, 1, 2, 5), "iris2d": (.2, 1, 5, 20)},
        "expanded": {"grid2d": (100, 300, 1000), "maze": (100, 300, 1000),
                     "iris2d": (100, 1000, 10000)},
    }

    @classmethod
    def reporting_rows(cls, rows_by_planner, source_budget, cutoff):
        if not math.isfinite(cutoff) or not 0 < cutoff <= source_budget:
            raise ValueError("The reporting cutoff must be positive and no greater than the source budget.")
        # Preserve recorded budgets; censor successes after the cutoff only in memory.
        return {
            planner: [dict(row, is_success=bool(row["is_success"])
                           and 0 <= float(row["runtime"]) <= cutoff)
                      for row in rows if cls._same_budget(row, source_budget)]
            for planner, rows in rows_by_planner.items()
        }

    @classmethod
    def plot(cls, rows, source_budget, cutoff, output_prefix, formats):
        reported = cls.reporting_rows(rows, source_budget, cutoff)
        report = dict(source_budget_seconds=source_budget, reporting_cutoff_seconds=cutoff,
                      groups={})
        for group in cls.FIGURE_GROUP_ORDER:
            grouped = {planner: cls._rows_for_group(reported[planner], group, source_budget)
                       for planner in cls.FIGURE_PLANNERS}
            expected = {str(row["instance_id"]) for row in grouped[cls.FIGURE_PLANNERS[0]]}
            if not expected or any(len(values) != len(expected)
                                   or {str(row["instance_id"]) for row in values} != expected
                                   for values in grouped.values()):
                raise ValueError(f"Incomplete or mismatched heuristic schedule in {group}")
            report["groups"][group] = dict(
                total=len(expected),
                solved={planner: sum(row["is_success"] for row in values)
                        for planner, values in grouped.items()},
                metrics={},
            )

        fig, axes = plt.subplots(2, 3, figsize=cls.FIGURE_SIZE, squeeze=False)
        positions = tuple(range(1, len(cls.FIGURE_PLANNERS) + 1))
        for row_idx, metric in enumerate(cls.FIGURE_METRIC_ORDER):
            field = cls.METRIC_FIELDS[metric]
            for col_idx, group in enumerate(cls.FIGURE_GROUP_ORDER):
                ax = axes[row_idx, col_idx]
                ids = cls._common_success_instance_ids(
                    reported, group, source_budget, field, cls.FIGURE_PLANNERS)
                values_by_planner = cls._boxplot_values(reported, group, source_budget, field)
                report["groups"][group]["metrics"][metric] = dict(
                    common_instance_ids=sorted(ids),
                    values=dict(zip(cls.FIGURE_PLANNERS, values_by_planner)),
                )
                for position, planner, values in zip(positions, cls.FIGURE_PLANNERS, values_by_planner):
                    if values:
                        color = cls.FIGURE_HEURISTIC_COLORS[planner]
                        ViolinStyle.draw(ax, position, values, color, width=cls.FIGURE_VIOLIN_WIDTH)
                        cls._plot_boxplot_scatter_points(ax, position, values, color, "horizontal")
                if not ids:
                    ax.text(.5, .5, "no common\nsuccesses", ha="center", va="center",
                            transform=ax.transAxes)
                values = [value for series in values_by_planner for value in series]
                ax.set_xscale("log")
                ax.set_xlim(*cls._axis_limits(values))
                ax.set_ylim(len(positions) + .65, .35)
                ax.xaxis.set_major_locator(FixedLocator(cls.METRIC_X_TICKS[metric][group]))
                ax.xaxis.set_major_formatter(FuncFormatter(
                    lambda value, pos: cls._format_log_tick(value, pos) if value > 1000 else f"{value:g}"))
                ax.xaxis.set_minor_formatter(NullFormatter())
                cls._draw_heuristic_reference_lines(ax, positions)
                ax.grid(axis="x", alpha=.22, linewidth=.7)
                ax.set_yticks(positions, [cls.PLANNER_SHORT_LABELS[p] for p in cls.FIGURE_PLANNERS])
                ax.tick_params(axis="both", labelsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=1.5)
                ax.tick_params(axis="y", labelleft=col_idx == 0)
                ax.set_xlabel(cls.METRIC_LABELS[metric], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE)
                if row_idx == 0:
                    ax.set_title(cls.FIGURE_GROUP_LABELS[group], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=5)
        fig.tight_layout(pad=.35, w_pad=.35, h_pad=.35)
        _save_figure(fig, output_prefix, formats)
        return report

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--results-root", type=Path, default=cls.DEFAULT_RESULTS_ROOT)
        parser.add_argument("--output-prefix", type=Path, default=cls.DEFAULT_OUTPUT_PREFIX)
        parser.add_argument("--source-budget", type=float, default=None)
        parser.add_argument("--reporting-cutoff", type=float, default=30.0)
        parser.add_argument("--formats", nargs="+", default=("pdf", "png"))
        args = parser.parse_args()
        rows = cls.load_rows_by_planner(args.results_root)
        source_budget = cls.select_budget(rows, args.source_budget)
        report = cls.plot(rows, source_budget, args.reporting_cutoff, args.output_prefix, args.formats)
        args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    STHeuristicViolinReport.main()
