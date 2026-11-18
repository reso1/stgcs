import unittest
from types import SimpleNamespace

from experiments.reports.st_trajectories import STQualitativeComparison as Report


class STQualitativeSelectionTests(unittest.TestCase):
    @staticmethod
    def rows(examples):
        rows = {key: {} for key in Report.PLANNERS}
        for instance_id, (runtime, cost) in examples.items():
            for index, key in enumerate(rows):
                rows[key][instance_id] = dict(instance_id=instance_id, source_domain="maze",
                    is_success=True, runtime=runtime[index], cost=cost[index], budget=30.0)
        return rows

    @staticmethod
    def record(regions=(0, 1, 2, 3), spacing=1):
        centers = [(0, 0), (spacing, 0), (0, spacing), (spacing, spacing)]
        return SimpleNamespace(generation={"placements": [
            dict(region=region, patrol=[[x-.1, y], [x+.1, y]])
            for region, (x, y) in zip(regions, centers)]})

    def test_spread_measures_host_regions_and_patrol_centers(self):
        spread = Report.obstacle_spread(self.record(regions=(0, 0, 1, 2), spacing=2))
        self.assertEqual(spread["occupied_regions"], 3)
        self.assertAlmostEqual(spread["patrol_center_spacing"], 2)

    def test_spread_precedes_runtime_even_when_b2_is_slower(self):
        rows = self.rows({
            "most_regions": ((1, 10, .5), (8, 9, 10)),
            "wider_spacing": ((1, 100, 2), (4, 10, 12)),
            "fastest": ((1, 10, 8), (2, 10, 12)),
        })
        records = {"most_regions": self.record(spacing=.5),
                   "wider_spacing": self.record(regions=(0, 0, 1, 2), spacing=2),
                   "fastest": self.record(regions=(0, 0, 1, 2), spacing=1)}
        ranked = Report.rank_candidates(rows, records, "maze")
        self.assertEqual([r["instance_id"] for r in ranked],
                         ["most_regions", "wider_spacing", "fastest"])

    def test_runtime_breaks_spread_ties_using_both_baselines(self):
        rows = self.rows({
            "balanced": ((1, 10, 6), (8, 9, 10)),
            "one_slow_baseline": ((1, 100, 2), (4, 10, 12)),
            "better_cost": ((1, 5, 5), (2, 10, 12)),
        })
        records = {key: self.record() for key in rows["B2"]}
        ranked = Report.rank_candidates(rows, records, "maze")
        self.assertEqual([r["instance_id"] for r in ranked],
                         ["balanced", "better_cost", "one_slow_baseline"])
        self.assertEqual(ranked[0]["minimum_speedup"], 6)

    def test_quality_breaks_ties_and_invalid_cohorts_are_excluded(self):
        rows = self.rows({
            "quality": ((1, 6, 8), (4, 8, 12)),
            "tie": ((1, 8, 6), (8, 10, 12)),
            "failed": ((1, 30, 20), (2, 8, 10)),
            "other_budget": ((1, 30, 20), (2, 8, 10)),
            "other_domain": ((1, 30, 20), (2, 8, 10)),
        })
        rows["Z1"]["failed"]["is_success"] = False
        rows["S1"]["other_budget"]["budget"] = 60
        rows["B2"]["other_domain"]["source_domain"] = "grid2d"
        records = {key: self.record() for key in rows["B2"]}
        self.assertEqual([r["instance_id"] for r in Report.rank_candidates(rows, records, "maze")],
                         ["quality", "tie"])


if __name__ == "__main__":
    unittest.main()
