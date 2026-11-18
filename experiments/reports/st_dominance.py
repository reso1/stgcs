"""Dominance-check violins with solution costs relative to delta_set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.reports.st_dominance_gaps import STDominanceAblationGroupReport
from experiments.manifests.st_planning import STPlanningManifestStore
from experiments.reports.common import ViolinStyle, _save_figure, plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator, NullFormatter, PercentFormatter


class STDominanceViolinReport(STDominanceAblationGroupReport):
    @classmethod
    def plot(cls, rows, budget, output_prefix, formats):
        report = dict(budget_seconds=budget, reference_planner=cls.REFERENCE_PLANNER, groups={})
        fig, axes = plt.subplots(3, 3, figsize=cls.FIGURE_SIZE, squeeze=False,
                                 gridspec_kw=cls._boxplot_gridspec_kwargs())
        for col, group in enumerate(cls.FIGURE_GROUP_ORDER):
            selected = {p: cls._rows_for_group(rows[p], group, budget) for p in cls.PLANNERS}
            expected = {r["instance_id"] for r in selected[cls.REFERENCE_PLANNER]}
            if not expected or any(len(v) != len(expected)
                                   or {r["instance_id"] for r in v} != expected
                                   for v in selected.values()):
                raise ValueError(f"Incomplete or mismatched dominance schedule in {group}")
            report["groups"][group] = dict(
                total=len(expected),
                solved={p: sum(r["is_success"] for r in v) for p, v in selected.items()},
                metrics={},
            )
            for row, metric in enumerate(cls.FIGURE_METRIC_ORDER):
                field = cls.METRIC_FIELDS[metric]
                planners = cls.QUALITY_FIGURE_PLANNERS if metric == "quality" else cls.PLANNERS
                matched = (cls.REFERENCE_PLANNER, *planners) if metric == "quality" else planners
                ids = cls._common_success_instance_ids(rows, group, budget, field, matched)
                values = cls._boxplot_values(rows, group, budget, field)
                report["groups"][group]["metrics"][metric] = dict(
                    common_instance_ids=sorted(ids), values=dict(zip(planners, values)))
                ax = axes[row, col]
                positions = tuple(range(1, len(planners) + 1))
                for position, planner, series in zip(positions, planners, values):
                    if series:
                        color = cls.FIGURE_HEURISTIC_COLORS[planner]
                        ViolinStyle.draw(ax, position, series, color)
                        cls._plot_boxplot_scatter_points(ax, position, series, color, "horizontal")
                if not ids:
                    ax.text(.5, .5, "no common\nsuccesses", ha="center", va="center",
                            transform=ax.transAxes)
                ax.set_xlim(*cls._figure_y_limits([v for series in values for v in series], metric, group))
                ax.set_ylim(len(planners) + .5, .5)
                if metric == "quality":
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=(1, 2, 4, 5, 10), min_n_ticks=3))
                    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
                else:
                    ax.set_xscale("log")
                    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
                    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
                    ax.xaxis.set_minor_formatter(NullFormatter())
                cls._draw_heuristic_reference_lines(ax, positions)
                ax.grid(axis="x", alpha=.22, linewidth=.7)
                ax.set_yticks(positions, [cls.PLANNER_SHORT_LABELS[p] for p in planners])
                ax.tick_params(axis="both", labelsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=1.5)
                ax.tick_params(axis="y", labelleft=col == 0)
                ax.set_xlabel(cls.FIGURE_X_LABELS[metric], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE)
                if row == 0:
                    ax.set_title(cls.FIGURE_GROUP_LABELS[group], fontsize=cls.FIGURE_COLUMN_TEXT_SIZE, pad=5)
        fig.tight_layout(pad=.35, w_pad=.35, h_pad=.35)
        _save_figure(fig, output_prefix, formats)
        return report

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--results-root", type=Path, default=cls.DEFAULT_RESULTS_ROOT)
        parser.add_argument("--manifest", type=Path, default=cls.DEFAULT_MANIFEST)
        parser.add_argument("--output-prefix", type=Path, default=cls.DEFAULT_OUTPUT_PREFIX)
        parser.add_argument("--budget", type=float, default=600.0)
        parser.add_argument("--formats", nargs="+", default=("pdf", "png"))
        args = parser.parse_args()
        expected = {r.instance_id for r in STPlanningManifestStore.load_manifest(args.manifest)}
        rows = cls.load_rows_by_planner(args.results_root)
        if not expected or any(len(v) != len(expected) or {r["instance_id"] for r in v} != expected
               or any(not cls._same_budget(r, args.budget) for r in v)
               for v in rows.values()):
            raise ValueError("Incomplete or mismatched dominance schedule")
        report = cls.plot(rows, args.budget, args.output_prefix, args.formats)
        args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    STDominanceViolinReport.main()
