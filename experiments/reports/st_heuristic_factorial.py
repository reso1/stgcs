from __future__ import annotations

import argparse
import json
from itertools import combinations
import math
from pathlib import Path
from statistics import median

from experiments.reports.style import configure_matplotlib_for_latex

configure_matplotlib_for_latex(backend="Agg")


from experiments.planners.st_planning import STExperimentConfig
from experiments.reports.common import PlotPalette, _save_figure, plt


from experiments.reports.st_common import STHeuristicAblationGroupReport


class STHeuristicFactorialReport(STHeuristicAblationGroupReport):
    """All-instance success curves; expansion ratios use explicitly paired successes."""
    SPECS = tuple(spec for checks in STExperimentConfig.HEURISTIC_DOMINANCE
                  for spec in STExperimentConfig.heuristic_specs(checks))
    FIGURE_PLANNERS = tuple(spec.name for spec in SPECS)
    DEFAULT_BUDGET = STExperimentConfig.DEFAULT_BUDGET

    @classmethod
    def spec_groups(cls, dominance):
        if dominance not in ("both", "delta_set"):
            raise ValueError(f"Unknown dominance selection: {dominance}")
        return [tuple(spec for spec in cls.SPECS if spec.dominance_checks == checks)
                for checks in STExperimentConfig.HEURISTIC_DOMINANCE
                if dominance == "both" or "delta_set" in checks]

    @staticmethod
    def success_curve(rows, budget):
        times = sorted(float(r["runtime"]) for r in rows
                       if r["is_success"] and math.isfinite(float(r["cost"]))
                       and 0 <= float(r["runtime"]) <= budget)
        total = len(rows)
        return ([0.0, *times, budget],
                [0.0, *[(i + 1) / total for i in range(len(times))], len(times) / total if total else 0.0])

    @classmethod
    def summary(cls, rows_by_planner, budget, dominance="both"):
        arms = cls.spec_groups(dominance)
        planners = tuple(spec.name for arm in arms for spec in arm)
        report = dict(budget_seconds=budget, groups={},
                      primary="All scheduled instances, including failures; no common-success filtering.",
                      secondary="Per-instance expansion ratios on paired successes only; numerator/denominator.")
        for group in cls.FIGURE_GROUP_ORDER:
            rows = {name: cls._rows_for_group(rows_by_planner[name], group, budget)
                    for name in planners}
            expected = {str(r["instance_id"]) for r in rows[planners[0]]}
            if not expected or any(len(v) != len(expected) or {str(r["instance_id"]) for r in v} != expected
                                   for v in rows.values()):
                raise ValueError(f"Incomplete or mismatched factorial schedule in {group}")
            successes = {name: cls._success_rows_by_instance(v, "num_expanded_nodes") for name, v in rows.items()}
            pairs = [pair for arm in arms
                     for pair in combinations([s.name for s in arm], 2)]
            if len(arms) == 2:
                pairs += [(a.name, b.name) for a, b in zip(*arms)]
            comparisons = []
            for numerator, denominator in pairs:
                ids = sorted(successes[numerator].keys() & successes[denominator].keys())
                ratios = [successes[numerator][i]["num_expanded_nodes"] /
                          successes[denominator][i]["num_expanded_nodes"] for i in ids]
                comparisons.append(dict(numerator=numerator, denominator=denominator, paired_n=len(ids),
                                        instance_ids=ids, median_expansion_ratio=median(ratios) if ratios else None))
            report["groups"][group] = dict(total=len(expected), instance_ids=sorted(expected),
                solved={name: round(cls.success_curve(v, budget)[1][-1] * len(expected)) for name, v in rows.items()},
                expansion_pairs=comparisons)
        return report

    @classmethod
    def plot(cls, rows, budget, output_prefix, formats, dominance="both"):
        report = cls.summary(rows, budget, dominance)
        arms = cls.spec_groups(dominance)
        fig, axes = plt.subplots(len(arms), 3, figsize=(6.5, 2.15 * len(arms)),
                                 sharex=True, sharey=True, squeeze=False)
        styles = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)))
        for row, specs in enumerate(arms):
            for col, group in enumerate(cls.FIGURE_GROUP_ORDER):
                ax = axes[row, col]
                for index, spec in enumerate(specs):
                    joint = spec.key_evaluation == "joint"
                    label = "joint" if joint else rf"$h_\mathrm{{{spec.heuristic[2:]}}}$"
                    color = "#8c6bb1" if joint else PlotPalette.heuristic_color(spec.heuristic)
                    x, y = cls.success_curve(cls._rows_for_group(rows[spec.name], group, budget), budget)
                    ax.step(x, y, where="post", color=color, linestyle=styles[index], linewidth=1.5, label=label)
                ax.set(xlim=(0, budget), ylim=(-0.025, 1.025), yticks=(0, .5, 1))
                ax.grid(alpha=.18)
                if row == 0:
                    ax.set_title(f"{cls.FIGURE_GROUP_LABELS[group]} ($n={report['groups'][group]['total']}$)")
                if row == len(arms) - 1:
                    ax.set_xlabel("runtime (s)")
                if col == 0:
                    label = r"GUB + $\delta_\mathrm{set}$" if "delta_set" in specs[0].dominance_checks else "GUB only"
                    ax.set_ylabel(label + "\nsolved fraction")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .9), pad=.5)
        _save_figure(fig, output_prefix, formats)
        output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2))
        return report

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description="Fig. 10: six heuristics crossed with GUB-only / GUB+delta_set.")
        parser.add_argument("--results-root", type=Path, default=cls.DEFAULT_RESULTS_ROOT)
        parser.add_argument("--output-prefix", type=Path, default=cls.DEFAULT_OUTPUT_PREFIX)
        parser.add_argument("--budget", type=float, default=cls.DEFAULT_BUDGET)
        parser.add_argument("--dominance", choices=("both", "delta_set"), default="both")
        parser.add_argument("--formats", nargs="+", default=("png", "pdf"))
        args = parser.parse_args()
        cls.plot(cls.load_rows_by_planner(args.results_root), args.budget, args.output_prefix,
                 args.formats, args.dominance)


if __name__ == "__main__":
    STHeuristicFactorialReport.main()
