"""Adapt dataset instances and saved records to the native ST-GCS planners."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.base import BaseInstanceFactory, BaseManifestStore
from experiments.environment.obstacle import StaticPolygon, StaticSphere
from stgcs_native.client import NativeClient
from stgcs_native.planning import MRMPNativeBackend
from stgcs_native.types import Environment, PlanningProblem, Polyhedron, Sphere, WindowedCoordinationReturn


class NativeGeometry:
    @staticmethod
    def polyhedron(poly) -> Polyhedron:
        return Polyhedron(np.asarray(poly.A()), np.asarray(poly.b()))

    @classmethod
    def cells(cls, graph):
        if graph is None:
            return []
        return [(name, cls.polyhedron(graph.get_vertex(name).st_hpoly))
                for name in graph.G.nodes
                if name not in {"source", "target"} and not str(name).startswith("sub-target")
                and graph.get_vertex(name) is not None]

    @classmethod
    def environment(cls, env) -> Environment:
        if getattr(env, "O_Dynamic", []):
            raise ValueError("Trajectory optimization expects static geometry; encode dynamic obstacles in cells.")
        obstacles = []
        for obstacle in getattr(env, "O_Static", []):
            if isinstance(obstacle, StaticSphere):
                obstacles.append(Sphere(np.asarray(obstacle.pos), obstacle.radius))
            elif isinstance(obstacle, StaticPolygon):
                obstacles.append(cls.polyhedron(obstacle.hpoly))
            else:
                raise TypeError(f"Unsupported static obstacle: {type(obstacle).__name__}")
        return Environment(np.asarray(env.lb), np.asarray(env.ub), env.robot_radius, obstacles,
                           [cls.polyhedron(region) for region in getattr(env, "_CSpace_hpoly", [])])

    @classmethod
    def planning_problem(cls, env, horizon, velocity_limit) -> PlanningProblem:
        return PlanningProblem([cls.polyhedron(region) for region in env._CSpace_hpoly],
                               float(getattr(env, "robot_radius", 0)), float(horizon), float(velocity_limit))


class STGCSPlannerClient(MRMPNativeBackend):
    """Run the ST-GCS C++ planners on repository dataset instances."""

    CACHE_ROOT = NativeClient.ROOT / "data/stgcs_base"

    def __init__(self, executable, work_dir, cache_root=None):
        super().__init__(executable, work_dir, self.CACHE_ROOT if cache_root is None else cache_root)

    @staticmethod
    def base_instance(record, base_root: Path):
        manifest = BaseManifestStore.manifest_path(base_root, domain_key=record.source_domain)
        base = BaseManifestStore.record_by_id(manifest, record.base_instance_id)
        return BaseInstanceFactory.from_record(base, compute_heuristics=False)

    @staticmethod
    def write_input(path, env, query, horizon, obstacles):
        problem = NativeGeometry.planning_problem(env, horizon, query.vlimit)
        MRMPNativeBackend.write_problem(path, problem, query, obstacles)

    def prepare(self, record, base_root: Path) -> Path:
        path = self.work_dir / "inputs" / f"{record.instance_id}.txt"
        instance = self.base_instance(record, base_root)
        self.write_input(path, instance.env, record.query, record.tmax, record.dynamic_obstacles)
        self.bind_cache(path, record.base_instance_id, record.source_domain)
        return path


    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--executable", type=Path, required=True,
                            help="Release C++ stgcs_planner executable with MRMP support.")
        parser.add_argument("--work-dir", type=Path, default=None,
                            help="Native inputs and per-solve logs (defaults to OUTPUT/native); "
                                 "heuristic caches are shared under data/stgcs_base/<domain>/heuristics/<base-id>.")

    @classmethod
    def from_args(cls, args):
        return cls(args.executable, args.work_dir or Path(args.output_root) / "native")


    def prepare_instance(self, instance, queries, *, compute_cache=True, cost_table=True):
        problem = NativeGeometry.planning_problem(instance.env, instance.stgcs.tmax, instance.stgcs.vlimit)
        base = getattr(instance, "base_record", None)
        return self.prepare_multi(problem, queries, compute_cache=compute_cache, cost_table=cost_table,
                                  base_id=base.instance_id if base else None,
                                  domain=base.domain_key if base else "problems")

    def run(self, instance, queries, spec, budget, child_rule, window_span_factor=0.0,
            execution_horizon_factor=1.0, dynamic_window_adjustment=False, method="pbs"):
        _, entry = self.run_with_solutions(
            instance, queries, spec, budget, child_rule, window_span_factor,
            execution_horizon_factor, dynamic_window_adjustment, method)
        return entry

    def run_with_solutions(self, instance, queries, spec, budget, child_rule,
                           window_span_factor=0.0, execution_horizon_factor=1.0,
                           dynamic_window_adjustment=False, method="pbs"):
        problem = NativeGeometry.planning_problem(instance.env, instance.stgcs.tmax, instance.stgcs.vlimit)
        base = getattr(instance, "base_record", None)
        return self.plan_multi(problem, queries, spec, budget, child_rule,
                               window_span_factor, execution_horizon_factor,
                               dynamic_window_adjustment, method,
                               base_id=base.instance_id if base else None,
                               domain=base.domain_key if base else "problems")

    @classmethod
    def plan(cls, instance, queries, spec, budget, child_rule,
             window_span_factor=0.0, execution_horizon_factor=1.0, dynamic_window_adjustment=False, method="pbs"):
        executable = NativeClient.executable("stgcs_planner")
        with TemporaryDirectory(prefix="stgcs-demo-plan-") as work:
            return cls(executable, Path(work)).run_with_solutions(
                instance, queries, spec, budget, child_rule, window_span_factor,
                execution_horizon_factor, dynamic_window_adjustment, method)

    @staticmethod
    def windowed_result(solutions, entry):
        unavailable = np.array([-1., np.nan])
        return WindowedCoordinationReturn(
            success=entry.is_success, solutions=solutions,
            mp=np.array([entry.mp_calls, entry.mp_runtime]),
            ecd=np.array([entry.ecd_calls, entry.ecd_runtime]),
            cc=np.array([entry.cc_calls, entry.cc_runtime]),
            wpbs=np.array([entry.pbs_calls, entry.pbs_runtime]),
            gcs=unavailable.copy(), gub=unavailable.copy(), search=unavailable.copy(),
            cr=unavailable.copy(), dc=unavailable.copy(), dc_cr=unavailable.copy(),
            runtime=np.array([entry.runtime]), pbs_popped_nodes=entry.pbs_popped_nodes,
            pbs_generated_children=entry.pbs_generated_children, pbs_update_calls=entry.pbs_update_calls)

    @classmethod
    def solve_single(cls, env, query, horizon, *, spec=None, budget=60.0, method="search",
                     seed=0, rounded=100, base_record=None):
        from experiments.instance import Instance

        with TemporaryDirectory(prefix="stgcs-plan-") as work:
            client = cls(NativeClient.executable("stgcs_planner"), Path(work))
            return client.plan_single(
                NativeGeometry.planning_problem(env, horizon, query.vlimit), query,
                [Instance.dynamic_obstacle_to_spec(obs) for obs in env.O_Dynamic],
                spec=spec, budget=budget, method=method, seed=seed, rounded=rounded,
                base_id=base_record.instance_id if base_record else None,
                domain=base_record.domain_key if base_record else "problems")
