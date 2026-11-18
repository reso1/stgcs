import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPoint, Polygon

from experiments.base import BaseInstanceFactory
from experiments.base_serialization import SerializedBaseInstanceStore
from experiments.environment.env import Env
from experiments.datasets.st import STInstanceGenerator
from experiments.instance import GridInstance, Instance, MazeInstance
from experiments.manifests.base import BaseBenchmarkRecord, save_manifest
from experiments.manifests.st_planning import STPlanningManifestStore
from experiments.datasets.base import GridBaseManifestBuilder, MazeBaseManifestBuilder
from experiments.runners.precompute import BaseOfflineHeuristicCLI


class STInstanceGenerationTests(unittest.TestCase):
    @staticmethod
    def base():
        boxes = [np.array([[x,0], [x+4.5,0], [x+4.5,4], [x,4]]) for x in (0,4,8)]
        env = Env("test", boxes, 0.2)
        return Instance("test", 7, env, Instance.build_stgcs_from_env(env,20,1))

    def test_defaults_and_invalid_parameters(self):
        g = STInstanceGenerator()
        self.assertEqual((g.tmax,g.motion_duration,g.obstacle_speed,g.obstacle_radius,g.num_obstacles,
                          g.motion,g.min_patrol_length), (100,20,0.25,0.1,10,"region-chord",0.5))
        for options in ({"obstacle_speed":0}, {"tmax":float("inf")}, {"vlimit":-1},
                        {"motion_duration":101}, {"obstacle_radius":0}, {"min_patrol_length":0},
                        {"num_obstacles":0}, {"seed":-1}, {"motion":"unknown"},
                        {"max_vertices":0}, {"max_edges":0}, {"max_attempts":0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                STInstanceGenerator(**options)

    def test_speed_periodicity_continuity_clipping_and_tail(self):
        g = STInstanceGenerator(tmax=30,motion_duration=23)
        obstacle = g.periodic_obstacle(np.array([[0.,0.],[1.,0.]]),0.1)
        self.assertEqual(len(obstacle.segments),6)
        self.assertEqual(obstacle.itvl.end,23)
        for leg in obstacle.segments:
            self.assertAlmostEqual(np.linalg.norm(leg.velocity),0.25)
        for before,after in zip(obstacle.segments,obstacle.segments[1:]):
            self.assertEqual(before.itvl.end,after.itvl.start)
            np.testing.assert_allclose(before.xt,after.x0)
        for t in np.linspace(0,15,53):
            np.testing.assert_allclose(obstacle.x(t),obstacle.x(t+8),atol=1e-12)
        for t in (23,25,30):
            np.testing.assert_allclose(obstacle.x(t),[0.25,0])
        bent = g.periodic_obstacle(np.array([[0.,0.],[1.,0.],[1.,2.]]),0.1)
        for leg in bent.segments:
            self.assertAlmostEqual(np.linalg.norm(leg.velocity),0.25)
        np.testing.assert_allclose(bent.x(4),[1,0])
        np.testing.assert_allclose(bent.x(12),[1,2])

    def test_region_chords_use_actual_radius_and_bound_reversals(self):
        base,g = self.base(),STInstanceGenerator()
        _,reference = next(g.reference_routes(base.env))
        env,placements,_ = g.populate_obstacles(base,reference)
        self.assertEqual([Instance.dynamic_obstacle_to_spec(o) for o in env.O_Dynamic],
                         [Instance.dynamic_obstacle_to_spec(o) for o in g.populate_obstacles(base,reference)[0].O_Dynamic])
        self.assertEqual(len(env.O_Dynamic),10)
        for o,placement in zip(env.O_Dynamic,placements,strict=True):
            self.assertEqual(o.radius,0.1)
            path = np.asarray(placement['patrol'])
            self.assertGreaterEqual(np.linalg.norm(path[1]-path[0]),0.5)
            self.assertLessEqual(len(o.segments),11)
            np.testing.assert_allclose(o.x(placement['encounter_time']),placement['anchor'],atol=1e-9)
            hosts = [i for i,p in enumerate(env._CSpace_hpoly) if all(p.PointInSet(x,1e-7) for x in path)]
            self.assertEqual(len(hosts),1)
            offsets = (env.robot_radius+o.radius)*np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
            tube = MultiPoint((path[:,None]+offsets).reshape(-1,2)).convex_hull
            for j,vertices in enumerate(env.C_Space):
                if j!=hosts[0]: self.assertTrue(tube.disjoint(Polygon(vertices)))
        self.assertEqual(base.env.O_Dynamic,[])
        with self.assertRaisesRegex(ValueError,"long region chords"):
            STInstanceGenerator(min_patrol_length=100).populate_obstacles(base,reference)

    def test_phase_preserves_speed_periodicity_and_horizon(self):
        g = STInstanceGenerator()
        path = np.array([[0.,0.],[1.,0.],[1.,2.]])
        for phase in (-2.5,2.5,26.5):
            shifted = g.periodic_obstacle(path,0.1,phase)
            prior = STInstanceGenerator(tmax=20).periodic_obstacle(path,0.1,phase)
            self.assertEqual(Instance.dynamic_obstacle_to_spec(shifted),
                             Instance.dynamic_obstacle_to_spec(prior))
            self.assertEqual(shifted.itvl.start,0)
            self.assertEqual(shifted.itvl.end,20)
            for leg in shifted.segments:
                self.assertAlmostEqual(np.linalg.norm(leg.velocity),0.25)
            for before,after in zip(shifted.segments,shifted.segments[1:]):
                self.assertEqual(before.itvl.end,after.itvl.start)
                np.testing.assert_allclose(before.xt,after.x0,atol=1e-12)
            # The bent patrol's cycle is 24 s; compare against a longer reference.
            original = STInstanceGenerator(tmax=60,motion_duration=60).periodic_obstacle(path,0.1)
            for t in np.linspace(0,20,51):
                np.testing.assert_allclose(shifted.x(t),original.x(t+phase%24),atol=1e-12)
        with self.assertRaises(ValueError):
            g.periodic_obstacle(path,0.1,float('nan'))

    def assert_witness(self,g,env,metrics):
        obstacles = [g.collision_geometry([np.r_[s.x0,s.itvl.start,s.xt,s.itvl.end] for s in o.segments],
                                          o.radius,g.tmax) for o in env.O_Dynamic]
        reference = [np.array(s) for s in metrics["reference_path"]]
        self.assertTrue(g.collides(reference,env.robot_radius,obstacles))
        wait = metrics["reference_wait"]
        delayed = [s+[0,0,wait,0,0,wait] for s in reference]
        delayed.insert(0,np.r_[reference[0][:3],reference[0][:2],wait])
        self.assertFalse(g.collides(delayed,env.robot_radius,obstacles))
        self.assertLessEqual(delayed[-1][-1],g.tmax)
        for s in delayed:
            self.assertTrue(env.is_segment_in_CSpace(s[:2],s[3:5]))
            self.assertLessEqual(np.max(np.abs(s[:2]-s[3:5])),g.vlimit*(s[-1]-s[2])+1e-9)

    def test_route_first_patrols_all_block_reference_and_preserve_witness(self):
        g = STInstanceGenerator(num_obstacles=3)
        env = Env("square",[np.array([[0,0],[4,0],[4,4],[0,4]])],0.1)
        base = Instance("square",0,env,Instance.build_stgcs_from_env(env,20,1))
        instance,query,metrics = g.generate(base)
        self.assertTrue(instance.stgcs.has_path(query))
        self.assertEqual(len(metrics['placements']),3)
        self.assertEqual(metrics['policy'],'route-first')
        self.assertGreater(metrics["reference_wait"],0)
        self.assert_witness(g,instance.env,metrics)
        reference = [np.asarray(s) for s in metrics['reference_path']]
        for obstacle in instance.env.O_Dynamic:
            geometry = g.collision_geometry([np.r_[s.x0,s.itvl.start,s.xt,s.itvl.end]
                                             for s in obstacle.segments],obstacle.radius,g.tmax)
            self.assertTrue(g.collides(reference,env.robot_radius,[geometry]))
        self.assertLessEqual(metrics['reference_wait'],(g.tmax-metrics['reference_duration'])/2)

    def test_graph_limit_rejects_oversized_instance(self):
        for limit in ({'max_vertices':1},{'max_edges':1}):
            with self.subTest(limit=limit),self.assertRaisesRegex(ValueError,"Graph exceeds limits"):
                STInstanceGenerator(num_obstacles=1,max_attempts=1,**limit).generate(self.base())

    def test_route_first_respects_distinct_regions_and_safe_endpoints(self):
        base,g = self.base(),STInstanceGenerator(num_obstacles=3,distinct_start_sets=True)
        instance,query,metrics = g.generate(base)
        self.assertEqual(len({p['region'] for p in metrics['placements']}),3)
        for p in (query.start,query.goal):
            stay = [np.r_[p,0,p,g.tmax]]
            obstacles = [g.collision_geometry([np.r_[s.x0,s.itvl.start,s.xt,s.itvl.end]
                                               for s in o.segments],o.radius,g.tmax)
                         for o in instance.env.O_Dynamic]
            self.assertFalse(g.collides(stay,instance.env.robot_radius,obstacles))
        self.assert_witness(g,instance.env,metrics)

    def test_manifest_roundtrip_preserves_motion_query_and_witness(self):
        g,base = STInstanceGenerator(num_obstacles=1),self.base()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"data/instances/st_planning"
            base_root = root/"base"
            base_path = base_root/"grid2d/manifest.json"
            record = BaseBenchmarkRecord("test","grid",2,7,{},True).with_manifest_path(base_path)
            save_manifest(base_path,[record])
            BaseInstanceFactory.write_cached_instance(record,base)
            generated = g.build_record(record)
            STPlanningManifestStore.save_manifest(root/"manifest.json",[generated])
            loaded = STPlanningManifestStore.load_manifest(root/"manifest.json")[0]
            self.assertEqual(generated,loaded)
            items = BaseOfflineHeuristicCLI._load_work_items(root/"manifest.json",None)
            self.assertEqual(items,[(base_path.resolve(),record)])
            self.assertEqual(items,BaseOfflineHeuristicCLI._load_work_items(root/"manifest.json",base_root))
            instance,query,_,_ = STPlanningManifestStore.reconstruct_instance(loaded,base_root)
            self.assertEqual(instance.stgcs.tmax,100)
            self.assertEqual(instance.stgcs.G.number_of_nodes(),generated.stgcs_num_vertices)
            self.assertEqual(instance.stgcs.G.number_of_edges(),generated.stgcs_num_edges)
            self.assertTrue(instance.stgcs.has_path(query))
            self.assert_witness(g,instance.env,loaded.generation)

    def test_scaled_grid_generation_and_cache_miss_reconstruction(self):
        scale = 5 / 4.5
        original = GridInstance.generate_base(0, 4, 4, compute_heuristics=False)
        with tempfile.TemporaryDirectory() as directory:
            record = GridBaseManifestBuilder.build_records(
                2, [4], 1, None, 0, Path(directory) / "grid2d/manifest.json", scale)[0]
            instance = BaseInstanceFactory.from_record(record, compute_heuristics=False)
            self.assertEqual(instance.stgcs.G.number_of_nodes(), original.stgcs.G.number_of_nodes())
            self.assertEqual(set(instance.stgcs.G.edges), set(original.stgcs.G.edges))
            self.assertLessEqual(instance.stgcs.G.number_of_nodes(), 16)
            self.assertTrue(np.all(instance.env.ub <= 5))
            self.assertAlmostEqual(instance.env.robot_radius, 0.1 * scale)
            for a, b in zip(original.env.C_Space, instance.env.C_Space, strict=True):
                np.testing.assert_allclose(b, a * scale)
            SerializedBaseInstanceStore.object_path(record).unlink()
            rebuilt = BaseInstanceFactory.from_record(record, compute_heuristics=False)
            for a, b in zip(instance.env.C_Space, rebuilt.env.C_Space, strict=True):
                np.testing.assert_array_equal(a, b)

    def test_scaled_maze_generation_and_cache_miss_reconstruction(self):
        original = MazeInstance.build_base_env(0,8,8)
        with tempfile.TemporaryDirectory() as directory:
            record = MazeBaseManifestBuilder.build_records(1,8,8,0,Path(directory)/"maze/manifest.json")[0]
            instance = BaseInstanceFactory.from_record(record,compute_heuristics=False)
            np.testing.assert_allclose(instance.env.ub,[2,2])
            self.assertEqual(instance.env.robot_radius,0.0625)
            for a,b in zip(original.C_Space,instance.env.C_Space,strict=True):
                np.testing.assert_allclose(b,a*0.25)
            for a,b in zip(original.O_Static,instance.env.O_Static,strict=True):
                np.testing.assert_allclose(b.vertices,a.vertices*0.25)
            SerializedBaseInstanceStore.object_path(record).unlink()
            rebuilt = BaseInstanceFactory.from_record(record,compute_heuristics=False)
            for a,b in zip(instance.env.C_Space,rebuilt.env.C_Space,strict=True):
                np.testing.assert_array_equal(a,b)

    def test_static_routes_respect_walls(self):
        boxes = [np.array([[0,0],[1,0],[1,6],[0,6]]),np.array([[0,5],[4,5],[4,6],[0,6]]),
                 np.array([[3,0],[4,0],[4,6],[3,6]])]
        env = Env("bent-corridor",boxes,0.1)
        points = np.array([[0.5,0.5],[3.5,0.5],[0.5,0.5]])
        np.testing.assert_allclose(STInstanceGenerator.static_routes(env,points)[0],
                                   [[0,11,0],[11,0,11],[0,11,0]])
        env = Env("disconnected",[boxes[0],boxes[2]],0.1)
        self.assertTrue(np.isinf(STInstanceGenerator.static_routes(env,points)[0][0,1]))

    def test_ecd_ignores_bounding_box_false_positive(self):
        env = Env("square", [np.array([[0,0], [1,0], [1,1], [0,1]])], 0.05)
        g = STInstanceGenerator(tmax=10, motion_duration=10)
        env.O_Dynamic = [g.periodic_obstacle(np.array([[2.2,0.5], [0.5,2.2]]), 0.05)]
        graph = Instance.build_stgcs_from_env(env, 10, 1)
        self.assertEqual(list(graph.G), ["v0"])
        self.assertEqual(graph.G.number_of_edges(), 0)

    def test_ecd_preserves_free_time_between_visits(self):
        base = self.base()
        g = STInstanceGenerator(tmax=12, motion_duration=12, obstacle_speed=10/6)
        env = base.env.copy()
        env.O_Dynamic = [g.periodic_obstacle(np.array([[1.,1.], [11.,1.]]), 0.1)]
        graph = Instance.build_stgcs_from_env(env, 12, 1)
        for t in (5.5,6.,6.5):
            self.assertTrue(any(graph.get_vertex(v).st_hpoly.PointInSet(np.array([1.,1.,t]))
                                for v in graph.G))

    def test_ecd_reserves_final_position_after_motion_stops(self):
        g = STInstanceGenerator()
        env = Env("square", [np.array([[-1,-1], [2,-1], [2,2], [-1,2]])], 0.1)
        env.O_Dynamic = [g.periodic_obstacle(np.array([[0.,0.], [1.,0.]]), 0.1)]
        graph = Instance.build_stgcs_from_env(env, g.tmax, 1)
        for t in (20.,60.,100.):
            np.testing.assert_allclose(env.O_Dynamic[0].x(t), [1,0])
            self.assertFalse(any(graph.get_vertex(v).st_hpoly.PointInSet(np.array([1.,0.,t]))
                                 for v in graph.G))
