"""Native demo planning, smoothing, geometry, and saved-result integration."""
from visualization.solution_io import SolutionIO
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from pydrake.geometry.optimization import HPolyhedron

from experiments.environment.env import Env
from experiments.environment.obstacle import StaticPolygon, StaticSphere
from experiments.planners.mrmp import SearchPlannerSpec
from experiments.native import STGCSPlannerClient, NativeGeometry
from stgcs_native.trajopt import GlobalTrajOptConfig, GlobalTrajectoryOptimizer as Optimizer
from stgcs_native.types import MPQuery
from stgcs_native.types import STTrajectory

@unittest.skipUnless((STGCSPlannerClient.ROOT / "stgcs_cplusplus/build/bin/stgcs_trajopt").is_file() or os.environ.get("STGCS_TRAJOPT_EXECUTABLE"),
                     "Build the native demos first")
class NativeDemoTests(unittest.TestCase):
    @staticmethod
    def path(states, name="unknown"):
        values = np.asarray(states, dtype=float)
        return STTrajectory([name] * (len(values)-1), [np.r_[a,b] for a,b in zip(values[:-1],values[1:])], values.shape[1]-1)

    @staticmethod
    def graph(dimension=2):
        if dimension == 2:
            vertices = np.array([[-2,-2], [2,-2], [2,2], [-2,2]], float)
        else:
            from itertools import product
            vertices = np.array(list(product((-2.,2.), repeat=dimension)))
        env = Env("native-test", [vertices], .1)
        graph = env.build_STGCS(t0=0, tmax=10, vlimit=2)
        return SimpleNamespace(env=env, stgcs=graph, base_record=SimpleNamespace(
            instance_id=f"native-test-{dimension}d", domain_key="native-test"))

    def test_python_reference_solutions(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/demo_trajopt_reference.json").read_text())
        for case in fixture["cases"]:
            with self.subTest(case=case["name"]):
                paths = [self.path(states, "cell" if case.get("soft") else "unknown") for states in case["states"]]
                config = GlobalTrajOptConfig(**case["config"])
                instance = self.graph(paths[0].dim-1)
                kwargs = {"cells": NativeGeometry.cells(instance.stgcs)} if case["cells"] else {}
                if case.get("soft"):
                    instance.stgcs = SimpleNamespace(
                        dimension=2, G=SimpleNamespace(nodes=["cell"]),
                        get_vertex=lambda name: SimpleNamespace(st_hpoly=HPolyhedron.MakeBox([0,-2,0],[2,2,10])) if name == "cell" else None)
                    kwargs = {"cells": NativeGeometry.cells(instance.stgcs)}
                if case["window"]:
                    result = Optimizer.optimize_sliding_window(paths, .1, 2, config,
                                window_span=.7, stride=.3, **kwargs)
                else:
                    result = Optimizer.optimize(paths, .1, 2, config, **kwargs)
                np.testing.assert_allclose(result.times, case["times"], atol=1e-12)
                if not case["window"]:
                    # Global objectives are unchanged by the velocity parameterization.
                    np.testing.assert_allclose(result.optimized_positions, case["optimized"], atol=2e-6, rtol=0)
                else:
                    # Retaining the solved overlap intentionally changes later windows.
                    self.assertTrue(result.continuous_pairwise_collision_free)
                    self.assertTrue(result.continuous_environment_collision_free)
                    np.testing.assert_allclose(result.optimized_positions[:,0],result.original_positions[:,0],atol=1e-12)
                    np.testing.assert_allclose(result.optimized_positions[:,-1],result.original_positions[:,-1],atol=1e-12)
                self.assertLessEqual(result.max_velocity_component, 2+1e-5)
                self.assertTrue(result.sampled_pairwise_collision_free)
                if case.get("soft"):
                    self.assertIn("constraint_soft_retry_from", result.solver_message)

    def test_native_planning_to_smoothing_and_validation(self):
        for dim in (2,3):
            instance = self.graph(dim)
            starts = [np.r_[-1.,-.8,np.zeros(dim-2)], np.r_[-1.,.8,np.zeros(dim-2)]]
            queries = [MPQuery(start, start + np.r_[2.,np.zeros(dim-1)], 0, True, 2) for start in starts]
            # A real solve must remain independent of the former Python planner.
            with patch.dict(sys.modules, {"stgcs": None}):
                paths, entry = STGCSPlannerClient.plan(instance, queries,
                    SearchPlannerSpec("h_mot", ("GUB", "delta_pos")), 10, "num_conflicts", 5, 1, True)
            self.assertTrue(entry.is_success)
            self.assertEqual(len(paths), 2)
            self.assertGreater(entry.mp_calls, 0)
            converted = STGCSPlannerClient.windowed_result(paths, entry)
            self.assertTrue(converted.success)
            self.assertEqual(converted.pbs_update_calls, entry.pbs_update_calls)
            self.assertTrue(Optimizer.pairwise_collision_free(paths,.1,[True]*2))
            config = GlobalTrajOptConfig(sample_dt=.1, clearance_margin=0)
            result = Optimizer.smooth(paths,.1,2,config,
                window_span=.7,stride=.3,goal_stays=[True]*2,cells=NativeGeometry.cells(instance.stgcs))
            self.assertTrue(Optimizer.pairwise_collision_free(result.trajectories,.1,[True]*2))
            self.assertTrue(Optimizer.cspace_collision_free(NativeGeometry.cells(instance.stgcs),result.trajectories,1e-5))
            for path,query in zip(result.trajectories,queries):
                np.testing.assert_allclose(path.x0[:-1],query.start,atol=1e-5)
                np.testing.assert_allclose(path.xT[:-1],query.goal,atol=1e-5)

    def test_shared_smoothing_uses_disk_collision_geometry(self):
        paths = [self.path([[0,0,0],[0,0,1]]),
                 self.path([[.15,.15,0],[.15,.15,1]])]
        result = Optimizer.smooth(paths, .1, 1, GlobalTrajOptConfig(sample_dt=.2, clearance_margin=0),
                                  window_span=.7, stride=.3, goal_stays=[True,True])
        for before, after in zip(paths, result.trajectories):
            np.testing.assert_allclose(after.x0, before.x0, atol=1e-6)
            np.testing.assert_allclose(after.xT, before.xT, atol=1e-6)
        self.assertTrue(Optimizer.disk_pairwise_collision_free(result.trajectories,.1,[True,True]))
        self.assertFalse(Optimizer.pairwise_collision_free(result.trajectories,.1,[True,True]))

    def test_shared_large_demo_input_to_viewer_payload(self):
        from dataclasses import replace
        from experiments.base import BaseManifestStore
        from experiments.manifests.mrmp import load_manifest
        from demos import demo_r100_empty as demo
        from experiments.mrmp import MRMPExperiment
        record = load_manifest(demo.DEFAULT_MANIFEST_PATH)[0]
        record = replace(record, queries=record.queries[:2], num_agents=2,
                         instance_id=record.instance_id + "-native-test")
        manifest = BaseManifestStore.manifest_path(demo.DEFAULT_BASE_ROOT, domain_key=record.domain_key)
        base = BaseManifestStore.records_by_id(manifest)[record.base_instance_id]
        with patch.dict(sys.modules, {"stgcs": None}):
            payload = MRMPExperiment.run_stgcs(record, base, SearchPlannerSpec("h_mot", ("GUB", "delta_pos")),
                               "native-test", "native test", demo.MRMP_PLANNER_SETTINGS,
                               10., 5., 1., "num_conflicts")
        SolutionIO.validate_solution_payload(payload, record, "native-test", 10.)
        smoothed = MRMPExperiment.run_trajopt(record, base, payload, "native-test-trajopt", "native smoothing", demo.TRAJOPT_SETTINGS)
        SolutionIO.validate_solution_payload(smoothed, record, "native-test-trajopt", 10.)
        self.assertEqual(smoothed["global_trajopt"]["backend"], "stgcs_cplusplus")
        self.assertEqual(len(smoothed["trajectories"]), 2)

    def test_refinement_preserves_bent_corridor(self):
        env = Env("corner", [np.array([[0,0],[1,0],[1,.5],[0,.5]],float),
                             np.array([[.5,0],[1,0],[1,1],[.5,1]],float)], .05)
        graph = env.build_STGCS(t0=0,tmax=10,vlimit=1)
        path = self.path([[.1,.2,0],[.8,.2,1],[.8,.9,2]])
        config = GlobalTrajOptConfig(sample_dt=2, min_sample_dt=1e-4)
        instance = SimpleNamespace(env=env, stgcs=graph)
        refined = Optimizer.reference_trajectories(
            [path], instance.env.robot_radius, instance.stgcs.vlimit, config,
            cells=NativeGeometry.cells(instance.stgcs))
        self.assertEqual(refined[0].size,2)
        self.assertTrue(Optimizer.cspace_collision_free(NativeGeometry.cells(graph),refined))
        shortcut = self.path([[.1,.2,0],[.8,.9,2]])
        self.assertFalse(Optimizer.cspace_collision_free(NativeGeometry.cells(graph),[shortcut]))
        self.assertFalse(Optimizer.environment_collision_free(NativeGeometry.environment(env),[shortcut]))

    def test_close_cell_transition_knot_is_mandatory(self):
        cells=[HPolyhedron.MakeBox([0,-1,0],[2,1,2]),HPolyhedron.MakeBox([-2,-1,0],[0,1,2])]
        graph=SimpleNamespace(dimension=2,G=SimpleNamespace(nodes=[0,1]),
            get_vertex=lambda i:SimpleNamespace(st_hpoly=cells[i]))
        path=STTrajectory([0,1],[np.array([1.00005,0,0,0,0,1.00005]),
                                np.array([0,0,1.00005,-.99995,0,2])],2)
        config=GlobalTrajOptConfig(sample_dt=.5)
        result=Optimizer.optimize([path],.1,1,config,cells=NativeGeometry.cells(graph))
        self.assertIn(1.00005,result.times)
        self.assertNotIn(1.,result.times)
        self.assertLessEqual(result.max_velocity_component,1+config.solver_eps_abs)
        self.assertTrue(result.continuous_environment_collision_free)

    def test_refinement_preserves_disk_avoiding_bends(self):
        instance=self.graph()
        pair=[self.path([[-1,0,0],[0,.3,1],[1,0,2]]),
              self.path([[1,0,0],[0,-.3,1],[-1,0,2]])]
        reference=Optimizer.reference_trajectories(
            pair, instance.env.robot_radius, instance.stgcs.vlimit, GlobalTrajOptConfig(sample_dt=2),
            cells=NativeGeometry.cells(instance.stgcs))
        self.assertEqual([p.size for p in reference],[2,2])
        self.assertTrue(Optimizer.disk_pairwise_collision_free(reference,.1,[True,True]))
        for original,refined in zip(pair,reference):
            np.testing.assert_allclose(refined.points,original.points,atol=1e-12)

    def test_refinement_fills_stationary_time_gaps(self):
        instance=self.graph()
        path=STTrajectory(['unknown','unknown'],[np.array([0,0,0,.5,0,.5-5e-9]),
                                                np.array([.5,0,.5,1,0,1])],2)
        refined=Optimizer.reference_trajectories(
            [path], instance.env.robot_radius, instance.stgcs.vlimit, GlobalTrajOptConfig(sample_dt=2),
            cells=NativeGeometry.cells(instance.stgcs))
        result=Optimizer.inspect(refined,times=[.5-2.5e-9,.5])
        np.testing.assert_allclose(result.optimized_positions[0],[[.5,0],[.5,0]],atol=1e-12)

    def test_saved_wait_with_rounded_endpoints(self):
        

        segments = [
            dict(start=[0, 0], goal=[0.5, 0], t_start=0, t_end=0.5),
            dict(start=[np.nextafter(0.5, 1), 0], goal=[1, 0], t_start=0.5 + 1e-9, t_end=1),
        ]
        instance = self.graph()
        config = GlobalTrajOptConfig(sample_dt=2)
        path = SolutionIO.trajectory_from_viewer_json(dict(dimension=2, segments=segments))
        refined, = Optimizer.reference_trajectories(
            [path], instance.env.robot_radius, instance.stgcs.vlimit, config,
            cells=NativeGeometry.cells(instance.stgcs))
        np.testing.assert_array_equal(refined.points[1], [0.5, 0, 0.5, 0.5, 0, 0.5 + 1e-9])

        segments[1]["start"] = [0.51, 0]
        path = SolutionIO.trajectory_from_viewer_json(dict(dimension=2, segments=segments))
        with self.assertRaisesRegex(RuntimeError, "spatially discontinuous"):
            Optimizer.reference_trajectories(
            [path], instance.env.robot_radius, instance.stgcs.vlimit, config,
            cells=NativeGeometry.cells(instance.stgcs))

    def test_continuous_disk_checks_and_interval_constraints(self):
        pair=[self.path([[-1,0,0],[1,0,2]]),self.path([[1,0,0],[-1,0,2]])]
        self.assertFalse(Optimizer.disk_pairwise_collision_free(pair,.1,[True,True]))
        with self.assertRaisesRegex(RuntimeError,'QP failed|disk collision'):
            Optimizer.optimize(pair,.1,1,GlobalTrajOptConfig(sample_dt=2,include_interval_collision_constraints=False))
        diagonal=[self.path([[0,0,0],[0,0,1]]),self.path([[.15,.15,0],[.15,.15,1]])]
        self.assertTrue(Optimizer.disk_pairwise_collision_free(diagonal,.1,[True,True]))
        self.assertFalse(Optimizer.pairwise_collision_free(diagonal,.1,[True,True]))
        short=[self.path([[-.3,0,0],[.3,0,1e-8]]),self.path([[0,0,0],[0,0,1e-8]])]
        self.assertFalse(Optimizer.disk_pairwise_collision_free(short,.1,[True,True]))
        visitor=self.path([[1,0,0],[.3,0,.5],[1,0,1]])
        self.assertFalse(Optimizer.disk_pairwise_collision_free([short[0],visitor],.1,[True,True]))
        self.assertTrue(Optimizer.disk_pairwise_collision_free([short[0],visitor],.1,[False,True]))
        stationary=self.path([[0,0,0],[0,0,1]])
        late=self.path([[-1,0,1],[1,0,3]])
        self.assertFalse(Optimizer.disk_pairwise_collision_free([stationary,late],.1,[True,True]))
        self.assertTrue(Optimizer.disk_pairwise_collision_free([stationary,late],.1,[False,True]))

    def test_interval_normals_do_not_add_sampled_halfspaces(self):
        # All relative segments have closest point (.22, 0). Their common
        # horizontal normal permits smoothing past the old normal at sample 1.
        relative=np.array([[.22,-1],[.22,2],[.22,0],[.22,-1]])
        pair=[self.path(np.c_[sign*relative/2,np.arange(4)]) for sign in (1,-1)]
        config=GlobalTrajOptConfig(sample_dt=4,clearance_margin=0,
            velocity_gradient_weight=1,displacement_weight=1e-4,
            displacement_jitter_weight=0,solver_eps_abs=1e-9,solver_eps_rel=1e-9)
        result=Optimizer.optimize(pair,.1,10,config)
        optimized=result.optimized_positions[0]-result.optimized_positions[1]
        old_normal=relative[1]/np.linalg.norm(relative[1])
        self.assertLess(old_normal@optimized[1],.1)
        self.assertTrue(np.all(optimized[:,0]>=.2-1e-8))
        self.assertTrue(Optimizer.disk_pairwise_collision_free(result.trajectories,.1,[True,True]))
        np.testing.assert_allclose(result.optimized_positions[:,[0,-1]],
                                   result.original_positions[:,[0,-1]],atol=1e-12)

    def test_interval_normals_separate_both_endpoints(self):
        for dim in (2,3):
            with self.subTest(dimension=dim):
                relative=np.c_[[-2.,0.,2.],[0.,.6,0.],np.zeros((3,dim-2))]
                pair=[self.path(np.c_[sign*relative/2,np.arange(3)]) for sign in (1,-1)]
                result=Optimizer.optimize(pair,.1,2,GlobalTrajOptConfig(sample_dt=.25,
                    clearance_margin=0,velocity_gradient_weight=10,
                    solver_eps_abs=1e-9,solver_eps_rel=1e-9))
                reference=result.original_positions[0]-result.original_positions[1]
                optimized=result.optimized_positions[0]-result.optimized_positions[1]
                for k,(start,end) in enumerate(zip(reference[:-1],reference[1:])):
                    delta=end-start
                    s=np.clip(-start@delta/(delta@delta),0,1)
                    closest=start+s*delta
                    normal=closest/np.linalg.norm(closest)
                    self.assertTrue(np.all(optimized[k:k+2]@normal>=.2-1e-8))
                self.assertTrue(Optimizer.disk_pairwise_collision_free(result.trajectories,.1,[True,True]))

    def test_short_intervals_and_fixed_window_boundaries(self):
        path=self.path([[0,0,0],[.25,.1,.5],[.2500005,.1000002,.500001],[.5,0,1]])
        config=GlobalTrajOptConfig(sample_dt=.1,solver_eps_abs=1e-8,solver_eps_rel=1e-8)
        result=Optimizer.optimize_sliding_window([path],.1,1,config,window_span=.7,stride=.3)
        self.assertIn(.500001,result.times)
        self.assertLessEqual(result.max_velocity_component,1+1e-7)
        np.testing.assert_allclose(result.optimized_positions[0,0],[0,0],atol=1e-12)
        np.testing.assert_allclose(result.optimized_positions[0,-1],[.5,0],atol=1e-12)
        free=Optimizer.optimize([path],.1,1,GlobalTrajOptConfig(sample_dt=.1,fix_terminal_states=False))
        self.assertTrue(np.isfinite(free.optimized_positions).all())

    def test_short_interval_objective_matches_independent_least_squares(self):
        times=np.array([0,.5,.500001,1.,1.5])
        reference=np.array([[0,0],[.2,.4],[.2000001,.4000002],[.8,.2],[1,0]])
        path=self.path(np.c_[reference,times])
        config=GlobalTrajOptConfig(sample_dt=2,clearance_margin=0,solver_eps_abs=1e-9,solver_eps_rel=1e-9)
        result=Optimizer.optimize([path],.1,100,config)
        rows=[np.eye(len(times))[k]*np.sqrt(config.displacement_weight+5e-11) for k in range(len(times))]
        offsets=[np.zeros(2) for _ in rows]
        for k in range(2,len(times)-1):
            curr,prev=times[k+1]-times[k-1],times[k]-times[k-2]
            row=np.zeros(len(times)); row[[k+1,k-1,k,k-2]]=[1/curr,-1/curr,-1/prev,1/prev]
            rows.append(np.sqrt(config.velocity_gradient_weight)*row)
            offsets.append(np.sqrt(config.velocity_gradient_weight)*(row@reference))
        for k in range(1,len(times)-1):
            prev,next=times[k]-times[k-1],times[k+1]-times[k]
            row=np.zeros(len(times));row[k-1:k+2]=[1/prev,-1/prev-1/next,1/next]
            rows.append(np.sqrt(config.displacement_jitter_weight)*row);offsets.append(np.zeros(2))
        displacement=np.zeros_like(reference)
        displacement[1:-1]=np.linalg.lstsq(np.array(rows)[:,1:-1],-np.array(offsets),rcond=None)[0]
        np.testing.assert_allclose(result.optimized_positions[0],reference+displacement,atol=2e-6,rtol=0)

    def test_static_obstacle_constraints_and_continuous_checks(self):
        for obstacle in (StaticSphere(np.array([0.,0.]),.25),
                         StaticPolygon(np.array([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]]))):
            instance = self.graph()
            instance.env.O_Static = [obstacle]
            path = self.path([[-1,.8,0],[0,.6,1],[1,.8,2]])
            result = Optimizer.optimize([path],.1,2,GlobalTrajOptConfig(sample_dt=.2),env=NativeGeometry.environment(instance.env))
            self.assertTrue(result.continuous_environment_collision_free)
            self.assertTrue(instance.env.collision_checking_seg(np.array([-1.,0]),np.array([1.,0]),0,2))
            self.assertFalse(Optimizer.environment_collision_free(NativeGeometry.environment(instance.env),[self.path([[-1,0,0],[1,0,2]])]))
            if isinstance(obstacle, StaticSphere):
                with self.assertRaisesRegex(RuntimeError,"reference"):
                    Optimizer.optimize([self.path([[-1,0,0],[1,0,2]])],.1,2,env=NativeGeometry.environment(instance.env))

    def test_collisions_between_samples_and_goal_occupancy(self):
        a=self.path([[-1,0,0],[1,0,2]])
        b=self.path([[0,-1,0],[0,1,2]])
        self.assertFalse(Optimizer.pairwise_collision_free([a,b],.1,[True,True]))
        stationary=self.path([[0,0,0],[0,0,1]])
        late=self.path([[-1,0,1],[1,0,3]])
        self.assertFalse(Optimizer.pairwise_collision_free([stationary,late],.1,[True,True]))
        self.assertTrue(Optimizer.pairwise_collision_free([stationary,late],.1,[False,True]))

    def test_bad_settings_and_missing_executable_fail(self):
        path=self.path([[0,0,0],[1,0,2]])
        with self.assertRaises(ValueError):
            Optimizer.optimize_sliding_window([path],.1,1,window_span=1,stride=1)
        with self.assertRaisesRegex(RuntimeError,"positive"):
            Optimizer.optimize([path],.1,1,GlobalTrajOptConfig(sample_dt=0))
        with patch.dict(os.environ,{"STGCS_TRAJOPT_EXECUTABLE":"/nonexistent/stgcs_trajopt"}):
            with self.assertRaises(FileNotFoundError):
                Optimizer.optimize([path],.1,1)


if __name__ == "__main__":
    unittest.main()
