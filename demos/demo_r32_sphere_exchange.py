from __future__ import annotations
from visualization.solution_io import SolutionIO

import argparse
import json
import math
import os
from pathlib import Path

from experiments.base import BaseManifestStore
from experiments.manifests.mrmp import load_manifest
from experiments.planners.mrmp import SearchPlannerSpec
from experiments.planners.mrmp import MRMPPlannerSettings, TrajoptSettings
from experiments.mrmp import MRMPExperiment
from stgcs_native.types import ChildExpansionMode
from visualization.viewer.server import ViewerServer


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/stgcs-mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/stgcs-xdg-cache")

DEFAULT_BASE_ROOT = ROOT / "data/stgcs_base"
DEFAULT_MANIFEST_PATH = ROOT / "data/instances/mrmp/demo_r32_sphere_exchange/manifest.json"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

BASE_DOMAIN_KEY = "sphere-ball"
BASE_INSTANCE_ID = "base-sphere-ball-seed00000"
INSTANCE_ID = "mrmp-base-sphere-ball-seed00000-32-robot-swap"
DEFAULT_SOLUTION_OUTPUT = ROOT / "data/solutions" / DEFAULT_MANIFEST_PATH.parent.name / "mrmp_solution.json"
DEFAULT_TRAJOPT_OUTPUT = ROOT / "data/solutions" / DEFAULT_MANIFEST_PATH.parent.name / "trajopt_solution.json"
EXPECTED_NUM_AGENTS = 32

BUDGET = 120.0
WINDOW_ALPHA = 2.0
WINDOW_BETA = 1.0
EPSILON = 8.0
DYNAMIC_WINDOW_ADJUSTMENT = True
CHILD_EXPANSION_RULE = "num_conflicts"
CHILD_EXPANSION_RULES = {
    "lazy": ChildExpansionMode.LAZY,
    "soc": ChildExpansionMode.SOC,
    "makespan": ChildExpansionMode.MAKESPAN,
    "num_conflicts": ChildExpansionMode.NUM_CONFLICTS,
}
CHILD_EXPANSION_LABELS = {
    "lazy": "Lazy",
    "soc": "SOC",
    "makespan": "Makespan",
    "num_conflicts": "NumConflicts",
}

LOW_LEVEL_HEURISTIC = "h_max"
LOW_LEVEL_DOMINANCE_CHECKS = ("GUB", "delta_pos")
PLANNER_KEY_PREFIX = "wc-wpbs-h-max-gub-delta-pos"
TRAJOPT_PLANNER_SUFFIX = "global-trajopt"

TRAJOPT_SAMPLE_DT_FACTOR = 2.0
TRAJOPT_WINDOW_SPAN_FACTOR = 10.0
TRAJOPT_STRIDE_FACTOR = 5.0
TRAJOPT_VELOCITY_GRADIENT_WEIGHT = 0.05
TRAJOPT_DISPLACEMENT_WEIGHT = 0.2
TRAJOPT_DISPLACEMENT_JITTER_WEIGHT = 2e-4
TRAJOPT_SOLVER_MAX_ITER = 1_000_000
TRAJOPT_SOLVER_EPS = 1e-6
TRAJOPT_CLEARANCE_MARGIN = 0.0
TRAJOPT_INCLUDE_TRAJECTORY_KNOT_TIMES = True
TRAJOPT_COLLISION_TOLERANCE = 1e-6

MRMP_PLANNER_SETTINGS = MRMPPlannerSettings(
    child_expansion_rules=CHILD_EXPANSION_RULES,
    dynamic_window_adjustment=DYNAMIC_WINDOW_ADJUSTMENT,
)
TRAJOPT_SETTINGS = TrajoptSettings(
    sample_dt_factor=TRAJOPT_SAMPLE_DT_FACTOR,
    window_span_factor=TRAJOPT_WINDOW_SPAN_FACTOR,
    stride_factor=TRAJOPT_STRIDE_FACTOR,
    velocity_gradient_weight=TRAJOPT_VELOCITY_GRADIENT_WEIGHT,
    displacement_weight=TRAJOPT_DISPLACEMENT_WEIGHT,
    displacement_jitter_weight=TRAJOPT_DISPLACEMENT_JITTER_WEIGHT,
    solver_max_iter=TRAJOPT_SOLVER_MAX_ITER,
    solver_eps=TRAJOPT_SOLVER_EPS,
    clearance_margin=TRAJOPT_CLEARANCE_MARGIN,
    include_trajectory_knot_times=TRAJOPT_INCLUDE_TRAJECTORY_KNOT_TIMES,
    collision_tolerance=TRAJOPT_COLLISION_TOLERANCE,
)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve and view the 32-robot 3D sphere antipodal-exchange MRMP demo."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--solution-output", type=Path, default=DEFAULT_SOLUTION_OUTPUT)
    parser.add_argument("--trajopt-output", type=Path, default=DEFAULT_TRAJOPT_OUTPUT)
    parser.add_argument("--solution-budget", type=float, default=BUDGET)
    parser.add_argument("--window-alpha", type=float, default=WINDOW_ALPHA)
    parser.add_argument("--window-beta", type=float, default=WINDOW_BETA)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument(
        "--child-expansion-rule",
        choices=tuple(CHILD_EXPANSION_RULES.keys()),
        default=CHILD_EXPANSION_RULE,
    )
    parser.add_argument("--trajopt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-solve", action="store_true")
    parser.add_argument("--force-trajopt", action="store_true")
    parser.add_argument("--no-serve", action="store_true")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    budget = float(args.solution_budget)
    window_alpha = float(args.window_alpha)
    window_beta = float(args.window_beta)
    epsilon = float(args.epsilon)
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError(f"Solution budget must be finite and positive, got {args.solution_budget!r}.")
    if not math.isfinite(window_alpha) or window_alpha <= 0.0:
        raise ValueError(f"Window alpha must be finite and positive, got {args.window_alpha!r}.")
    if not math.isfinite(window_beta) or window_beta < 0.0 or window_beta > 1.0:
        raise ValueError(f"Window beta must be finite and in [0, 1], got {args.window_beta!r}.")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"Epsilon must be finite and positive, got {args.epsilon!r}.")

    manifest_path = Path(args.manifest)
    base_root = Path(args.base_root)
    records = {record.instance_id: record for record in load_manifest(manifest_path)}
    try:
        record = records[INSTANCE_ID]
    except KeyError as exc:
        raise KeyError(f"Missing demo instance {INSTANCE_ID!r} in {manifest_path}.") from exc

    base_manifest_path = BaseManifestStore.manifest_path(base_root, domain_key=BASE_DOMAIN_KEY)
    base_record = BaseManifestStore.record_by_id(base_manifest_path, BASE_INSTANCE_ID)
    if record.base_instance_id != base_record.instance_id:
        raise ValueError(
            f"MRMP record references {record.base_instance_id!r}, "
            f"but the shared base record is {base_record.instance_id!r}."
        )
    if record.domain_key != BASE_DOMAIN_KEY:
        raise ValueError(f"Expected {BASE_DOMAIN_KEY!r} MRMP record, got {record.domain_key!r}.")
    if int(record.num_agents) != EXPECTED_NUM_AGENTS:
        raise ValueError(f"Expected the {EXPECTED_NUM_AGENTS}-agent record, got {record.num_agents}.")

    base_vlimit = float(base_record.env_params["vlimit"])
    bad_vlimits = sorted(
        {
            float(query.vlimit)
            for query in record.queries
            if not math.isclose(float(query.vlimit), base_vlimit, rel_tol=0.0, abs_tol=1e-12)
        }
    )
    if bad_vlimits:
        raise ValueError(
            f"MRMP query vlimits {bad_vlimits} do not match base ST-GCS vlimit={base_vlimit:g}."
        )

    child_expansion_rule = str(args.child_expansion_rule)
    low_level_spec = SearchPlannerSpec(
        LOW_LEVEL_HEURISTIC,
        LOW_LEVEL_DOMINANCE_CHECKS,
        epsilon=epsilon,
    )
    selected_planner_key = f"{PLANNER_KEY_PREFIX}-{child_expansion_rule}"
    selected_planner_name = (
        f"WC(alpha={window_alpha:g},beta={window_beta:g}) + "
        f"wPBS-{CHILD_EXPANSION_LABELS[child_expansion_rule]} + {low_level_spec.name}"
    )
    trajopt_planner_key = f"{selected_planner_key}-{TRAJOPT_PLANNER_SUFFIX}"
    trajopt_planner_name = f"{selected_planner_name} + global trajopt"
    solution_output = Path(args.solution_output)
    trajopt_output = Path(args.trajopt_output)

    payload = None
    if not bool(args.force_solve) and solution_output.exists():
        payload = json.loads(solution_output.read_text())
        SolutionIO.validate_solution_payload(payload, record, selected_planner_key, budget)
        print(f"Reusing saved MRMP solution: {os.path.relpath(solution_output, ROOT)}")

    if payload is None:
        print(f"Running {record.instance_id} with {selected_planner_name}")
        payload = MRMPExperiment.run_stgcs(
            record,
            base_record,
            low_level_spec,
            selected_planner_key,
            selected_planner_name,
            MRMP_PLANNER_SETTINGS,
            budget=budget,
            window_alpha=window_alpha,
            window_beta=window_beta,
            child_expansion_rule=child_expansion_rule,
        )
        SolutionIO.validate_solution_payload(payload, record, selected_planner_key, budget)
        solution_output.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = solution_output.with_suffix(solution_output.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path.replace(solution_output)

    trajopt_payload = None
    if bool(args.trajopt):
        if not (args.force_solve or args.force_trajopt) and trajopt_output.exists():
            trajopt_payload = json.loads(trajopt_output.read_text())
            SolutionIO.validate_solution_payload(trajopt_payload, record, trajopt_planner_key, budget)
            print(f"Reusing saved trajopt solution: {os.path.relpath(trajopt_output, ROOT)}")
        if trajopt_payload is None:
            print(f"Running trajopt for {record.instance_id}")
            trajopt_payload = MRMPExperiment.run_trajopt(
                record,
                base_record,
                payload,
                trajopt_planner_key,
                trajopt_planner_name,
                TRAJOPT_SETTINGS,
            )
            SolutionIO.validate_solution_payload(trajopt_payload, record, trajopt_planner_key, budget)
            trajopt_output.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = trajopt_output.with_suffix(trajopt_output.suffix + ".tmp")
            tmp_path.write_text(json.dumps(trajopt_payload, indent=2, sort_keys=True))
            tmp_path.replace(trajopt_output)

    summary = {
        "instance_id": record.instance_id,
        "base_manifest": os.path.relpath(base_manifest_path, ROOT),
        "manifest": os.path.relpath(manifest_path, ROOT),
        "solution": os.path.relpath(solution_output, ROOT),
        "success": bool(payload["is_success"]),
        "runtime": payload["runtime"],
        "sum_of_costs": payload["cost"],
        "makespan": payload["makespan"],
        "completed_agents": int(payload["num_completed_agents"]),
    }
    if trajopt_payload is not None:
        summary.update(
            {
                "trajopt_solution": os.path.relpath(trajopt_output, ROOT),
                "trajopt_runtime": trajopt_payload["runtime"],
                "trajopt_sum_of_costs": trajopt_payload["cost"],
                "trajopt_makespan": trajopt_payload["makespan"],
            }
        )
    print(json.dumps(summary, indent=2))

    if not bool(args.no_serve):
        solution_paths = [solution_output]
        if trajopt_payload is not None:
            solution_paths.append(trajopt_output)
        ViewerServer.start(
            manifest_path,
            base_root,
            record.instance_id,
            solution_paths,
            host=str(args.host),
            port=int(args.port),
        )
