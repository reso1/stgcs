"""Best-known ST-GCS costs computed from the raw results being compared."""
from __future__ import annotations

import math
from pathlib import Path

from experiments.results import STResultStore


class STReferenceCosts:
    TOLERANCE = 1e-5

    @staticmethod
    def quality_reference(report):
        # CSV measurements do not certify optimality.
        return 'best_known', report['references']

    @classmethod
    def gaps(cls, rows, reference):
        _, costs = cls.quality_reference(reference)
        values = []
        for row in sorted(rows, key=lambda r: r['instance_id']):
            if not row['is_success'] or row['instance_id'] not in costs:
                continue
            gap = row['cost'] / costs[row['instance_id']]['cost'] - 1
            if not math.isfinite(gap) or gap < -cls.TOLERANCE:
                raise ValueError('A solution beats the reference; check the selected comparison rows')
            values.append(0.0 if abs(gap) <= cls.TOLERANCE else gap)
        return values

    @staticmethod
    def from_rows(rows_by_planner, budget):
        references = {}
        instance_ids = set()
        for planner, rows in rows_by_planner.items():
            if not planner.startswith(('Search(', 'MICP(')):
                continue  # External planners use different free-space representations.
            for row in rows:
                if not math.isclose(row['budget'], budget):
                    continue
                instance = row['instance_id']
                instance_ids.add(instance)
                cost = row['cost']
                if (not row['is_success'] or not math.isfinite(cost) or cost <= 0
                        or not 0 <= row['runtime'] <= budget):
                    continue
                if instance not in references or cost < references[instance]['cost']:
                    references[instance] = dict(cost=cost, kind='best_known')
        return dict(instance_ids=sorted(instance_ids), references=references)

    @classmethod
    def load(cls, results_root: Path, budget: float):
        rows = {path.stem: STResultStore.load_result_rows(path)
                for path in sorted(results_root.glob('*.csv'))
                if path.stem.startswith(('Search(', 'MICP('))}
        return cls.from_rows(rows, budget)
