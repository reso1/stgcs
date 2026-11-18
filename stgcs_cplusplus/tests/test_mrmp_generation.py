import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.base import BaseInstanceFactory
from experiments.manifests.base import load_manifest as load_bases
from experiments.manifests.mrmp import QuerySpec, load_manifest
from experiments.datasets.mrmp_queries import MRMPPerformanceComparisonManifestBuilder as Builder
from experiments.datasets.mrmp import MRMPManifestCLI
from experiments.runners.mrmp_comparison import MRMPPerformanceComparisonRunCLI
from experiments.native import STGCSPlannerClient


class CurrentMRMPGenerationTests(unittest.TestCase):
    SAMPLER_HASHES = {
        ("grid2d", 2): "11badfdfe8cff1cae29209e7024c4a811ad12b89202ad5419d8661074dce6dca",
        ("grid2d", 10): "4402e3e0ad7bb68ae7d182d7b8aa018d34f6444a5abe1b3aca52f244854cac20",
        ("maze", 2): "5f8fe75ec3f93fef0bf7d7988713ed4e92a2dff9606aed06eeb770a7a130e84a",
        ("maze", 10): "17a1a1aaf0ffb5654c37b7210912d3af66839bd96888bb337c27d3a23ef3c76f",
        ("iris-2d", 2): "45aab2b326f70b134162796fec0ca6b949a29999ff27ebb7f186bf940c9da100",
        ("iris-2d", 10): "85d308fa503234f5c14c54509fa1da63f555442f2c88d3642dc7a2ad6dbde959",
    }

    def test_current_sampler_preserves_fixed_seed_queries(self):
        for (domain, count), expected in self.SAMPLER_HASHES.items():
            with self.subTest(domain=domain, robots=count):
                base = load_bases(Path("data/stgcs_base") / domain / "manifest.json")[0]
                instance = BaseInstanceFactory.from_record(base, compute_heuristics=False)
                queries = Builder.sample_queries(instance, count, np.random.RandomState(1234 + count))
                payload = [QuerySpec.from_query(q).to_dict() for q in queries]
                actual = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                self.assertEqual(actual, expected)

    def test_current_records_resume_without_regeneration(self):
        records = [r for path in Path("data/instances/mrmp/performance_comparison").glob("manifest_n*.json")
                   for r in load_manifest(path)]
        self.assertEqual(len(records), 1500)
        with patch.object(Builder, "synthesize_record", side_effect=AssertionError("Unexpected regeneration")):
            for domain in ("grid2d", "maze", "iris-2d"):
                bases = MRMPManifestCLI.base_records_for_benchmark(domain, "data/stgcs_base", "performance-comparison")
                self.assertEqual(len(bases), 50)
                current = [r for r in records if r.domain_key == domain]
                self.assertEqual(Builder.build_manifest(bases, 50, current), current)
                for record in current:
                    self.assertEqual(MRMPPerformanceComparisonRunCLI.record_result_group(record),
                                     (record.num_agents, domain))
                    self.assertNotIn("traffic_family", record.to_dict())
                    self.assertNotIn("traffic_tier", record.to_dict())

    def test_reference_solver_uses_named_binary_cache_and_native_spec(self):
        base = load_bases("data/stgcs_base/grid2d/manifest.json")[0]
        instance = BaseInstanceFactory.from_record(base, compute_heuristics=False)
        queries = Builder.sample_queries(instance, 2, np.random.RandomState(1236))
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            backend = STGCSPlannerClient(root / "exe", root / "work", root / "bases")
            with patch.object(Builder, "NATIVE_BACKEND", backend), patch.object(backend, "cache") as cache:
                Builder._prepare_low_level_heuristics(instance, base)
                path = cache.call_args.args[0]
                self.assertEqual(backend.cache_path(path), root / "bases/grid2d/heuristics" / base.instance_id / "heuristics.bin")
                query = queries[0]
                segment = [np.r_[query.start, 0.].tolist(), np.r_[query.goal, 5.].tolist()]
                with patch.object(backend, "run_process", return_value=dict(success=True, runtime=.1, segments=[segment])) as run:
                    sols, metrics = Builder._solve_independent_queries(instance, [query], 5.)
                command = run.call_args.args[0]
                self.assertEqual(command[4:8], ["h_max", "delta_pos", "8.0", "5.0"])
                self.assertEqual(metrics, (5., 5., .1))
                np.testing.assert_array_equal(sols[0].xT[:-1], query.goal)


if __name__ == "__main__":
    unittest.main()
