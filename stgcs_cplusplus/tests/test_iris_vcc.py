import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
from pydrake.geometry.optimization import HPolyhedron
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from experiments.base import BaseInstanceFactory
from experiments.environment.iris import Iris2DEnvBuilder, Iris2DProblem
from experiments.manifests.base import BaseBenchmarkRecord, load_manifest
from experiments.datasets.base import Iris2DBaseManifestBuilder
from stgcs_native.types import MPQuery


class IrisVccTests(unittest.TestCase):
    @staticmethod
    def make_problem(obstacles=(), clearance=0.0):
        return Iris2DProblem.from_polygon_obstacles(
            list(obstacles), np.zeros(2), np.full(2, 10.0),
            clearance=clearance, iteration_limit=1,
        )

    def test_visibility_detects_thin_obstacle_and_boundary_contact(self):
        obstacle = np.array([[4.999, 3], [5.001, 3], [5.001, 7], [4.999, 7]])
        problem = self.make_problem([obstacle])
        points = np.array([[1, 5], [9, 5], [1, 2], [9, 2], [1, 3], [9, 3]])
        adjacency = problem.visibility_graph(points).toarray()
        for i in range(len(points)):
            for j in range(i):
                self.assertEqual(
                    adjacency[i, j],
                    not LineString([points[i], points[j]]).intersects(Polygon(obstacle)),
                )
        self.assertFalse(adjacency[0, 1])
        self.assertFalse(adjacency[4, 5])
        self.assertTrue(adjacency[2, 3])

    def test_clique_surrounding_hole_recenters_ellipse(self):
        obstacle = np.array([[4.8, 3.4], [5.2, 3.4], [5.2, 3.9], [4.8, 3.9]])
        problem = self.make_problem([obstacle], clearance=0.1)
        clique = np.array([[1, 1], [9, 1], [5, 9]])
        self.assertTrue(problem.visibility_graph(clique).toarray().all())
        region = problem.grow_region(clique)
        self.assertTrue(any(np.allclose(region.seed, point) for point in clique))
        np.testing.assert_allclose(problem.options.starting_ellipse.center(), region.seed)
        self.assertGreaterEqual(Polygon(region.vertices).distance(Polygon(obstacle)), 0.1 - 1e-5)

    def test_empty_domain_is_one_complete_region(self):
        env = Iris2DEnvBuilder.build_env(0, {"m": 0, "visibility_samples": 20})
        self.assertEqual(len(env.C_Space), 1)
        self.assertAlmostEqual(Polygon(env.C_Space[0]).area, 25.0)
        self.assertEqual(env.robot_radius, 0.125)
        self.assertEqual(env.coverage_estimate, 1.0)
        self.assertEqual(env.region_generation, "vcc-iris")

    def test_invalid_regions_are_candidate_rejections_but_solver_failures_propagate(self):
        obstacle = np.array([[4, 4], [6, 4], [6, 6], [4, 6]])
        problem = self.make_problem([obstacle], clearance=.1)
        clique = np.array([[1., 1.], [2., 1.], [1., 2.]])
        regions = [
            (HPolyhedron(np.array([[1., 0.], [-1., 0.]]), np.array([-1., -1.])), "empty region"),
            (HPolyhedron.MakeBox(np.full(2, 7.), np.full(2, 9.)), "does not contain its seed"),
            (HPolyhedron.MakeBox(np.full(2, -1.), np.full(2, 10.)), "outside the domain"),
            (problem.domain, "intersects an original obstacle"),
        ]
        for region, message in regions:
            with self.subTest(message=message), patch('experiments.environment.iris.Iris', return_value=region):
                with self.assertRaisesRegex(ValueError, message):
                    problem.grow_region(clique)
        with patch('experiments.environment.iris.Iris', side_effect=RuntimeError("Solver unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Solver unavailable"):
                problem.grow_region(clique)

    def test_disconnected_cover_is_rejected(self):
        wall = np.array([[4.9, 0], [5.1, 0], [5.1, 10], [4.9, 10]])
        problem = self.make_problem([wall], clearance=0.1)
        with self.assertRaisesRegex(RuntimeError, "connected region graph"):
            problem.grow_regions_from_clique_cover(
                max_attempts=1000, max_iterations=1, visibility_samples=40,
                minimum_clique_size=3, threshold=0.4, coverage_samples=100,
                random_seed=0,
            )

    def test_sampling_budget_is_bounded(self):
        problem = self.make_problem([np.array([[0, 0], [10, 0], [10, 10], [0, 10]])])
        with self.assertRaisesRegex(RuntimeError, "within max_attempts"):
            problem.grow_regions_from_clique_cover(
                max_attempts=20, max_iterations=1, visibility_samples=20,
                minimum_clique_size=3, threshold=0.7, coverage_samples=100,
                random_seed=0,
            )

    def test_independent_final_coverage_check_rejects(self):
        with patch.object(Iris2DEnvBuilder, "estimate_coverage", side_effect=[1.0, 0.0]):
            with self.assertRaisesRegex(RuntimeError, "coverage estimate.*below"):
                Iris2DEnvBuilder.build_env(0, {"m": 0, "visibility_samples": 20})

    def test_parameters_reject_invalid_vcc_settings(self):
        for params in [
            {"region_generation": "random-seed"}, {"minimum_clique_size": 2},
            {"visibility_samples": 2}, {"max_attempts": 10}, {"max_iterations": 0},
        ]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                Iris2DEnvBuilder.normalize_env_params(params)

    def test_legacy_manifest_cannot_be_resumed_or_silently_rebuilt(self):
        record = BaseBenchmarkRecord(
            instance_id="legacy-iris", domain="iris-2d", space_dim=2,
            spatial_seed=0, env_params={"m": 10}, supports_sampling=True,
        )
        with self.assertRaisesRegex(ValueError, "not generated with VCC/IRIS"):
            Iris2DBaseManifestBuilder._validate_existing_records(
                [record], Iris2DEnvBuilder.normalize_env_params(),
            )
        with self.assertRaisesRegex(ValueError, "legacy IRIS cache is missing"):
            BaseInstanceFactory.from_record(record, compute_heuristics=False)

    def test_generation_manifest_cache_gcs_and_planning(self):
        params = {"m": 3, "visibility_samples": 60}
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "iris-2d" / "manifest.json"
            records = Iris2DBaseManifestBuilder.build_records(1, 0, params, manifest_path=manifest)
            record = load_manifest(manifest)[0]
            self.assertEqual(record.env_params["region_generation"], "vcc-iris")
            with patch.object(BaseInstanceFactory, "build", side_effect=AssertionError("Cache missed")):
                instance = BaseInstanceFactory.from_record(record, compute_heuristics=False)
            env = instance.env
            regenerated = Iris2DEnvBuilder.build_env(record.spatial_seed, params)
            self.assertEqual(env.edges, regenerated.edges)
            for actual, repeated in zip(env.C_Space, regenerated.C_Space, strict=True):
                np.testing.assert_allclose(actual, repeated, atol=1e-5)
            self.assertEqual(records[0].stgcs_num_vertices, len(env.C_Space))
            self.assertTrue(nx.is_strongly_connected(instance.stgcs.G))

            polygons = [Polygon(vertices) for vertices in env.C_Space]
            obstacles = unary_union([Polygon(o.vertices) for o in env.O_Static])
            union = unary_union(polygons)
            self.assertGreaterEqual(union.distance(obstacles), params.get("clearance", 0.125) - 1e-5)
            self.assertLess(union.difference(box(0, 0, 5, 5)).area, 1e-8)
            self.assertGreaterEqual(env.coverage_estimate, 0.8)
            # Independently check every spatial edge and its two ST-GCS directions.
            expected_edges = {
                (i, j) for i in range(len(polygons)) for j in range(i + 1, len(polygons))
                if polygons[i].intersects(polygons[j])
            }
            self.assertEqual(set(env.edges), expected_edges)
            self.assertEqual(set(instance.stgcs.G.edges), {
                pair for i, j in expected_edges
                for pair in [(f"v{i}", f"v{j}"), (f"v{j}", f"v{i}")]
            })

            centers = env.seed_points
            pair = np.unravel_index(
                np.argmax(np.linalg.norm(centers[:, None] - centers[None, :], axis=2)),
                (len(centers), len(centers)),
            )
            start, goal = centers[pair[0]], centers[pair[1]]
            query = MPQuery(start, goal, 0.0, False, 1.0)
            from experiments.native import STGCSPlannerClient
            solution, _ = STGCSPlannerClient.solve_single(instance.env, query, instance.stgcs.tmax, budget=20.)
            self.assertIsNotNone(solution)
            np.testing.assert_allclose(solution.x0[:2], start, atol=1e-5)
            np.testing.assert_allclose(solution.xT[:2], goal, atol=1e-5)
            for segment in solution.points:
                a, b = segment[:2], segment[3:5]
                self.assertGreaterEqual(LineString([a, b]).distance(obstacles), 0.125 - 1e-5)
                self.assertLessEqual(np.max(np.abs(b - a)), segment[-1] - segment[2] + 1e-5)


if __name__ == "__main__":
    unittest.main()
