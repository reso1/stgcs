"""Run the saved 48-UAV village setup through the shared native planning API."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from experiments.base import BaseManifestStore
from experiments.environment.uav_village import VillageEnv
from experiments.manifests.mrmp import MRMPBenchmarkRecord, load_manifest
from experiments.mrmp import MRMPExperiment
from experiments.planners.mrmp import MRMPPlannerSettings, SearchPlannerSpec, TrajoptSettings
from stgcs_native.types import ChildExpansionMode
from visualization.solution_io import SolutionIO
from visualization.viewer.village import Village3DVisualizationMixin


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/stgcs-mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/stgcs-xdg-cache")


class DemoR48Village(VillageEnv, Village3DVisualizationMixin):
    DEFAULT_BASE_ROOT = ROOT / "data/stgcs_base"
    DEFAULT_MANIFEST_PATH = ROOT / "data/instances/mrmp/demo_r48_village/manifest.json"
    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 8780
    DEFAULT_SOLUTION_BUDGET = 300.0
    WINDOW_ALPHA = 2.0
    WINDOW_BETA = 1.0
    EPSILON = 8.0
    BASE_DOMAIN_KEY = "fastpathplanning-village3d"
    BASE_INSTANCE_ID = "base-fastpathplanning-village3d-seed01000"
    INSTANCE_ID = "mrmp-base-fastpathplanning-village3d-seed01000-48-robot-random"
    DEFAULT_SOLUTION_DIR = ROOT / "data/solutions" / DEFAULT_MANIFEST_PATH.parent.name
    DEFAULT_SOLUTION_OUTPUT = DEFAULT_SOLUTION_DIR / "mrmp_solution.json"
    DEFAULT_TRAJOPT_OUTPUT = DEFAULT_SOLUTION_DIR / "trajopt_solution.json"
    LOW_LEVEL_HEURISTIC = "h_mot_tri"
    LOW_LEVEL_DOMINANCE_CHECKS = ("GUB", "delta_pos")
    MRMP_PLANNER_KEY = "wc-wpbs-h-mot-tri-gub-delta-pos-num_conflicts"
    MRMP_PLANNER_NAME = "WC(alpha=2,beta=1) + wPBS-NumConflicts + Search(max(h_mot,h_tri)+GUB+delta_pos,eps=8)"
    PLANNER_KEY = f"{MRMP_PLANNER_KEY}-global-trajopt"
    PLANNER_NAME = f"{MRMP_PLANNER_NAME} + global trajopt"
    MRMP_PLANNER_SETTINGS = MRMPPlannerSettings(
        child_expansion_rules={"num_conflicts": ChildExpansionMode.NUM_CONFLICTS},
        dynamic_window_adjustment=True,
    )
    TRAJOPT_SETTINGS = TrajoptSettings(
        sample_dt_factor=0.5, window_span_factor=2.5, stride_factor=1.25,
        velocity_gradient_weight=0.05, displacement_weight=0.2,
        displacement_jitter_weight=2e-4, solver_max_iter=1_000_000,
        solver_eps=1e-6, clearance_margin=0.0,
        include_trajectory_knot_times=True, collision_tolerance=1e-6,
    )

    CSPACE_ALPHA = 0.14
    STATIC_OBSTACLE_ALPHA = 0.9
    UAV_MATERIALS = {
        "body_color": [0.92, 0.05, 0.04, 0.92],
        "arm_color": [0.92, 0.05, 0.04, 0.95],
        "propeller_color": [0.50, 0.50, 0.50, 0.88],
        "outline_color": [0.18, 0.04, 0.04, 1.0],
    }
    UAV_VISUAL_SCALE = 1.0
    VIEWER_LAYER_DEFAULTS = {"spatial_sets_fill": False, "spatial_sets_outline": False}

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--base-root", type=Path, default=cls.DEFAULT_BASE_ROOT)
        parser.add_argument("--manifest", type=Path, default=cls.DEFAULT_MANIFEST_PATH)
        parser.add_argument("--solution-output", type=Path, default=cls.DEFAULT_SOLUTION_OUTPUT)
        parser.add_argument("--trajopt-output", type=Path, default=cls.DEFAULT_TRAJOPT_OUTPUT)
        parser.add_argument("--solution-budget", type=float, default=cls.DEFAULT_SOLUTION_BUDGET)
        parser.add_argument("--window-alpha", type=float, default=cls.WINDOW_ALPHA)
        parser.add_argument("--epsilon", type=float, default=cls.EPSILON)
        parser.add_argument("--trajopt", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--force-solve", action="store_true")
        parser.add_argument("--force-trajopt", action="store_true")
        parser.add_argument("--no-serve", action="store_true")
        parser.add_argument("--host", default=cls.DEFAULT_HOST)
        parser.add_argument("--port", type=int, default=cls.DEFAULT_PORT)
        args = parser.parse_args()
        for name in ("solution_budget", "window_alpha", "epsilon"):
            value = getattr(args, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}.")
        return args

    @staticmethod
    def write_json_atomic(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(path)

    @staticmethod
    def load_solution(path, record, planner_key, budget):
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        SolutionIO.validate_solution_payload(payload, record, planner_key, budget)
        print(f"Reusing saved solution: {os.path.relpath(path, ROOT)}")
        return payload

    @classmethod
    def run_planning(cls, record, base_record, budget, window_alpha, epsilon):
        spec = SearchPlannerSpec(cls.LOW_LEVEL_HEURISTIC, cls.LOW_LEVEL_DOMINANCE_CHECKS, epsilon=epsilon)
        name = f"WC(alpha={window_alpha:g},beta=1) + wPBS-NumConflicts + {spec.name}"
        return MRMPExperiment.run_stgcs(
            record, base_record, spec, cls.MRMP_PLANNER_KEY, name, cls.MRMP_PLANNER_SETTINGS,
            budget=budget, window_alpha=window_alpha, window_beta=cls.WINDOW_BETA,
            child_expansion_rule="num_conflicts",
        )

    @classmethod
    def run_smoothing(cls, record, base_record, payload):
        return MRMPExperiment.run_trajopt(
            record, base_record, payload, cls.PLANNER_KEY,
            f"{payload['planner_name']} + global trajopt", cls.TRAJOPT_SETTINGS,
        )

    @classmethod
    def solve_record(cls, record, base_record, budget, window_alpha, epsilon, planner_key=None):
        key = cls.PLANNER_KEY if planner_key is None else planner_key
        cls.planner_name(key)
        query_record = MRMPBenchmarkRecord.from_dict(record)
        payload = cls.run_planning(query_record, base_record, budget, window_alpha, epsilon)
        if key == cls.PLANNER_KEY and payload["is_success"]:
            payload = cls.run_smoothing(query_record, base_record, payload)
        return payload

    @classmethod
    def run(cls):
        args = cls.parse_args()
        record = next(r for r in load_manifest(args.manifest) if r.instance_id == cls.INSTANCE_ID)
        base_manifest = args.base_root / cls.BASE_DOMAIN_KEY / "manifest.json"
        base = BaseManifestStore.record_by_id(base_manifest, cls.BASE_INSTANCE_ID)
        if record.base_instance_id != base.instance_id or record.num_agents != 48:
            raise ValueError("The village demo requires its saved 48-UAV instance.")
        if any(query.vlimit != base.env_params["vlimit"] for query in record.queries):
            raise ValueError("Query velocity limits differ from the saved base.")
        instance, _ = MRMPExperiment.reconstruct_instance(record, base, compute_heuristics=False)

        mrmp = None if args.force_solve else cls.load_solution(
            args.solution_output, record, cls.MRMP_PLANNER_KEY, args.solution_budget)
        if mrmp is None:
            print(f"Running {record.instance_id} with {cls.MRMP_PLANNER_NAME}")
            mrmp = cls.run_planning(record, base, args.solution_budget, args.window_alpha, args.epsilon)
            SolutionIO.validate_solution_payload(mrmp, record, cls.MRMP_PLANNER_KEY, args.solution_budget)
            cls.write_json_atomic(args.solution_output, mrmp)

        refined = None
        if args.trajopt:
            if not (args.force_solve or args.force_trajopt):
                refined = cls.load_solution(args.trajopt_output, record, cls.PLANNER_KEY, args.solution_budget)
            if refined is None:
                print(f"Running trajopt for {record.instance_id}")
                refined = cls.run_smoothing(record, base, mrmp)
                SolutionIO.validate_solution_payload(refined, record, cls.PLANNER_KEY, args.solution_budget)
                cls.write_json_atomic(args.trajopt_output, refined)

        viewer_record = {**base.env_params, **record.to_dict(), "spatial_seed": base.spatial_seed}
        payload = cls.build_viewer_manifest(
            viewer_record, instance.safe_boxes, instance.static_obstacles,
            instance.env.robot_radius, instance.geometry_metadata,
        )
        payload["instances"][0]["viewer_layer_defaults"] = dict(cls.VIEWER_LAYER_DEFAULTS)
        payload["solutions"] = [mrmp] + ([refined] if refined is not None else [])
        summary = dict(
            instance_id=record.instance_id, base_manifest=os.path.relpath(base_manifest, ROOT),
            manifest=os.path.relpath(args.manifest, ROOT), solution=os.path.relpath(args.solution_output, ROOT),
            success=mrmp["is_success"], runtime=mrmp["runtime"], sum_of_costs=mrmp["cost"],
            makespan=mrmp["makespan"], completed_agents=mrmp["num_completed_agents"],
            num_safe_boxes=len(instance.safe_boxes), num_static_obstacles=len(instance.static_obstacles),
        )
        if refined is not None:
            summary.update(trajopt_solution=os.path.relpath(args.trajopt_output, ROOT),
                trajopt_runtime=refined["runtime"], trajopt_sum_of_costs=refined["cost"],
                trajopt_makespan=refined["makespan"])
        print(json.dumps(summary, indent=2))
        if not args.no_serve:
            cls.serve_manifest(payload, viewer_record, base, args.host, args.port,
                               args.solution_budget, args.window_alpha, args.epsilon)


if __name__ == "__main__":
    DemoR48Village.run()
