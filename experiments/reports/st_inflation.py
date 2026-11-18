"""Write Appendix Table B1 from the complete heuristic-inflation CSV results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.reports.heuristic_scaling import HeuristicInflationAndScalingPlot as Plot
from experiments.reports.st_reference_costs import STReferenceCosts
from experiments.manifests.st_planning import STPlanningManifestStore


class HeuristicInflationTable:
    EPSILONS = (Plot.BASELINE_EPSILON, *Plot.EPSILONS)
    DEFAULT_OUTPUT_PREFIX = Path('latex/tables/table_b1')
    DEFAULT_MANIFEST = Path('data/instances/st_planning/manifest.json')

    @staticmethod
    def statistics(values):
        if not values:
            return dict(mean=None, p95=None, maximum=None)
        return dict(mean=float(np.mean(values)), p95=float(np.percentile(values, 95)), maximum=float(max(values)))

    @classmethod
    def build(cls, results_root: Path, budget: float, manifest_path: Path = DEFAULT_MANIFEST):
        rows = Plot.load_inflation_rows_by_planner(results_root, results_root)
        expected = {r.instance_id for r in STPlanningManifestStore.load_manifest(manifest_path)}
        if not expected:
            raise ValueError("Empty reference schedule")
        for name, outcomes in rows.items():
            selected = [r for r in outcomes if Plot.same_budget(r, budget)]
            if len(selected) != len(expected) or {str(r['instance_id']) for r in selected} != expected:
                raise ValueError(f"Incomplete or mismatched instance schedule for {name}")
        reference = STReferenceCosts.from_rows(rows, budget)
        quality_kind, costs = STReferenceCosts.quality_reference(reference)
        counts, records = {}, []
        for heuristic in Plot.HEURISTICS:
            baseline = Plot.success_rows_by_instance(rows[Plot.planner_name(heuristic, 1.0)], budget)
            counts[heuristic] = len(baseline)
            for epsilon in cls.EPSILONS:
                all_rows = [r for r in rows[Plot.planner_name(heuristic, epsilon)] if Plot.same_budget(r, budget)]
                solved = Plot.success_rows_by_instance(all_rows, budget)
                runtime_ids = sorted(i for i in solved
                                     if Plot.finite_positive(solved[i]['runtime']) is not None)
                quality_ids = sorted(solved.keys() & costs.keys())
                runtimes = [solved[i]['runtime'] for i in runtime_ids]
                gaps = [100 * gap for gap in STReferenceCosts.gaps(all_rows, reference)]
                records.append(dict(heuristic=heuristic, epsilon=epsilon, solved=len(solved),
                    total=len(all_rows), runtime_n=len(runtime_ids), runtime_instance_ids=runtime_ids,
                    quality_n=len(quality_ids), quality_instance_ids=quality_ids,
                    runtime_seconds=cls.statistics(runtimes), cost_gap_percent=cls.statistics(gaps)))
        return dict(budget_seconds=budget, baseline_solved=counts, rows=records,
                    reference_kind=quality_kind, reference_coverage=len(costs),
                    samples="Runtime: successful runs with finite positive runtimes. Quality: successful runs covered by the shared ST-GCS reference.")

    @staticmethod
    def number(value, metric):
        if value is None:
            return r"\text{--}"
        if metric == "runtime_seconds":
            return f"{value:.2f}"
        if abs(value) < 0.005:
            value = 0.0
        if abs(value) >= 100:
            return f"{value:.0f}"
        if metric == "cost_gap_percent" and abs(value) < 10:
            return f"{value:.2f}"
        return f"{value:.1f}"

    @classmethod
    def latex(cls, report):
        quality = 'Optimality gap' if report['reference_kind'] == 'optimal' else 'Best-known cost gap'
        lines = [r"\begin{table*}[t]", r"\centering",
            r"\caption{BFS heuristic-inflation sensitivity with fixed $\delta_\mathrm{pos}$ and a common $h_\mathrm{mot}$ GUB prepass. "
            + f"The planning budget is ${report['budget_seconds']:g}$s per run. "
            + r"Absolute runtime in seconds uses $n_T$ successful runs. "
            + ("Quality uses a shared verified ST-GCS optimum" if report['reference_kind'] == 'optimal'
               else "Quality uses a shared best-known ST-GCS cost, without an optimality claim")
            + r" on $n_c$ covered successes; the runtime and quality samples can differ.}",
            r"\label{tab:heuristic_inflation}", r"\small", r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{@{}cccccccccc@{}}", r"\toprule",
            r" & \multirow{2}{*}{$\epsilon$} & \multirow{2}{*}{$n_T$} & \multicolumn{3}{c}{Runtime (s)} & \multirow{2}{*}{$n_c$} & \multicolumn{3}{c}{" + quality + r" (\%)} \\",
            r"\cmidrule(lr){4-6}\cmidrule(lr){8-10}",
            r" & & & Mean & P95 & Max & & Mean & P95 & Max \\"]
        for i, row in enumerate(report['rows']):
            label = ""
            if i % len(cls.EPSILONS) == 0:
                lines.append(r"\midrule")
                heuristic = row['heuristic']
                name = r"\textit{joint}" if heuristic == 'joint' else "$h_\\mathrm{" + heuristic[2:] + "}$"
                label = r"\multirow{" + str(len(cls.EPSILONS)) + "}{*}{" + name + "}"
            runtime = [f"${cls.number(row['runtime_seconds'][stat], 'runtime_seconds')}$"
                       for stat in ('mean', 'p95', 'maximum')]
            quality_values = [f"${cls.number(row['cost_gap_percent'][stat], 'cost_gap_percent')}$"
                              for stat in ('mean', 'p95', 'maximum')]
            lines.append(' & '.join([label, f"{row['epsilon']:g}",
                                    str(row['runtime_n']), *runtime, str(row['quality_n']), *quality_values]) + r" \\")
        return '\n'.join([*lines, r"\bottomrule", r"\end{tabular}", r"\end{table*}", ''])

    @classmethod
    def write(cls, results_root: Path, budget: float, output_prefix: Path, manifest_path: Path = DEFAULT_MANIFEST):
        report = cls.build(results_root, budget, manifest_path)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        output_prefix.with_suffix('.json').write_text(json.dumps(report, indent=2))
        output_prefix.with_suffix('.tex').write_text(cls.latex(report))
        rows = [{k:v for k,v in row.items() if k not in ('runtime_instance_ids','quality_instance_ids','runtime_seconds','cost_gap_percent')}
                | {f'{metric}_{stat}':row[metric][stat] for metric in ('runtime_seconds','cost_gap_percent')
                   for stat in ('mean','p95','maximum')} for row in report['rows']]
        with output_prefix.with_suffix('.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return report

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--results-root',type=Path,default=Plot.DEFAULT_INFLATION_RESULTS_ROOT)
        parser.add_argument('--budget',type=float,default=Plot.DEFAULT_BUDGET)
        parser.add_argument('--output-prefix',type=Path,default=cls.DEFAULT_OUTPUT_PREFIX)
        parser.add_argument('--manifest',type=Path,default=cls.DEFAULT_MANIFEST)
        args = parser.parse_args()
        cls.write(args.results_root,args.budget,args.output_prefix,args.manifest)


if __name__ == '__main__':
    HeuristicInflationTable.main()
