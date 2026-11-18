import gc
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/stgcs-mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/stgcs-xdg-cache")

from baselines.sp_strrtstar import FixedPrioritySTRRTStarPlanner
from experiments.native import STGCSPlannerClient
from experiments.base import BaseManifestStore
from experiments.manifests.mrmp import load_manifest
from experiments.planners.mrmp import MRMPPerformanceComparison
from visualization.solution_io import SolutionIO, ViewerMRMPSolutionRun
from visualization.viewer.server import ViewerServer


MANIFEST_PATH = ROOT / "data/instances/mrmp/performance_comparison/manifest_n10.json"
BASE_ROOT = ROOT / "data/stgcs_base"
OUTPUT_ROOT = ROOT / "data/solutions/benchmark_comparison"
BUDGET = 30.0
BUDGET_LABEL = f"{float(BUDGET):.9g}".replace("-", "m").replace(".", "p")
INSTANCE = ("grid2d", "mrmp-base-grid2d-s4-seed00001-scale1p11111-10-robot-intersection")
PLANNER_KEYS = MRMPPerformanceComparison.PLANNER_KEYS
PLANNER_LABELS = {
    MRMPPerformanceComparison.WINDOWED_PBS_KEY: "Windowed-PBS+BFS",
    MRMPPerformanceComparison.WINDOWED_PP_KEY: "Windowed-PP+BFS",
    MRMPPerformanceComparison.PBS_KEY: "PBS+BFS",
    MRMPPerformanceComparison.PP_BFS_KEY: "PP+BFS",
    MRMPPerformanceComparison.FIXED_PP_ST_RRT_STAR_FIRST_KEY: "PP+ST-RRT* (first)",
    MRMPPerformanceComparison.FIXED_PP_ST_RRT_STAR_FINAL_KEY: "PP+ST-RRT* (final)",
    MRMPPerformanceComparison.PBS_ZETA_SIPP_KEY: "PBS+Zeta*-SIPP",
    MRMPPerformanceComparison.CB_GCS_KEY: "CB-GCS",
    MRMPPerformanceComparison.KCBS_KEY: "K-CBS",
}
NATIVE_BASELINE_MODULES = {
    "ST-RRT*": "baselines._ompl_strrt_star_native",
    "K-CBS": "baselines._ompl_kcbs_native",
}
solution_path = lambda record, planner_key: (
    OUTPUT_ROOT / record.instance_id / f"{planner_key}-budget-{BUDGET_LABEL}.json"
)
combined_solution_path = lambda record: (
    OUTPUT_ROOT / record.instance_id / f"all-planners-budget-{BUDGET_LABEL}.json"
)


def validate_payload(payload, record, planner_key):
    if planner_key not in PLANNER_LABELS:
        raise KeyError(f"Unknown demo planner {planner_key!r}.")
    label = PLANNER_LABELS[planner_key]
    if str(payload.get("instance_id")) != record.instance_id:
        raise ValueError(
            f"{label} payload belongs to {payload.get('instance_id')!r}, expected {record.instance_id!r}."
        )
    if str(payload.get("planner_key")) != planner_key:
        raise ValueError(
            f"{label} payload belongs to planner {payload.get('planner_key')!r}, expected {planner_key!r}."
        )
    if abs(float(payload.get("budget")) - float(BUDGET)) > 1e-9:
        raise ValueError(f"{label} payload budget does not match {float(BUDGET):g}.")
    if not bool(payload.get("is_success")):
        raise RuntimeError(f"{label} failed; skipping this planner because it did not produce a viewer solution.")
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, list) or len(trajectories) != int(record.num_agents):
        count = 0 if not isinstance(trajectories, list) else len(trajectories)
        raise ValueError(f"{label} returned {count} trajectories, expected {record.num_agents}.")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp_path.replace(path)


def viewer_payload(record, planner_key, entry, solutions):
    payload = SolutionIO.mrmp_run_to_viewer_json(
        record,
        ViewerMRMPSolutionRun(
            planner_key=planner_key,
            planner_name=PLANNER_LABELS[planner_key],
            budget=float(BUDGET),
            status="SUCCESS" if entry.is_success else "FAIL",
            entry=entry,
            solutions=solutions,
        ),
    )
    validate_payload(payload, record, planner_key)
    return payload


def run_fixed_priority_strrt_star(record):
    from experiments.mrmp import MRMPExperiment

    base_manifest_path = BaseManifestStore.manifest_path(BASE_ROOT, domain_key=record.domain_key)
    base_record = BaseManifestStore.records_by_id(base_manifest_path)[record.base_instance_id]
    instance = queries = None
    try:
        instance, queries = MRMPExperiment.reconstruct_instance(record, base_record, compute_heuristics=False)
        result = FixedPrioritySTRRTStarPlanner().solve(instance, queries, timeout_secs=float(BUDGET))
        payloads = {}
        snapshots = {
            MRMPPerformanceComparison.FIXED_PP_ST_RRT_STAR_FIRST_KEY: result.first_solution,
            MRMPPerformanceComparison.FIXED_PP_ST_RRT_STAR_FINAL_KEY: result.final_solution,
        }
        for planner_key, snapshot in snapshots.items():
            entry = MRMPExperiment._entry_from_fixed_priority_strrt_star_snapshot(snapshot, queries)
            payloads[planner_key] = viewer_payload(record, planner_key, entry, snapshot.solutions)
        return payloads
    finally:
        del queries
        del instance
        gc.collect()


def load_or_run_fixed_priority_strrt_star(record):
    paths = {
        planner_key: solution_path(record, planner_key)
        for planner_key in MRMPPerformanceComparison.FIXED_PP_ST_RRT_STAR_OUTPUT_KEYS
    }
    if all(path.exists() for path in paths.values()):
        payloads = {}
        for planner_key, path in paths.items():
            payload = json.loads(path.read_text())
            validate_payload(payload, record, planner_key)
            print(f"Reusing {PLANNER_LABELS[planner_key]}: {path.relative_to(ROOT)}")
            payloads[planner_key] = payload
        return payloads

    print(f"Running PP+ST-RRT* first/final for {record.instance_id}")
    payloads = run_fixed_priority_strrt_star(record)
    for planner_key, payload in payloads.items():
        write_json(paths[planner_key], payload)
    return payloads


def run_planner(record, planner_key):
    from experiments.mrmp import MRMPExperiment

    if planner_key in MRMPPerformanceComparison.FIXED_PP_ST_RRT_STAR_OUTPUT_KEYS:
        return run_fixed_priority_strrt_star(record)[planner_key]

    base_manifest_path = BaseManifestStore.manifest_path(BASE_ROOT, domain_key=record.domain_key)
    base_record = BaseManifestStore.records_by_id(base_manifest_path)[record.base_instance_id]
    instance = queries = None
    try:
        instance, queries = MRMPExperiment.reconstruct_instance(record, base_record, compute_heuristics=False)
        if planner_key in MRMPPerformanceComparison.SEARCH_BASED_PLANNER_KEYS:
            windowed = planner_key in (
                MRMPPerformanceComparison.WINDOWED_PBS_KEY, MRMPPerformanceComparison.WINDOWED_PP_KEY)
            solutions, entry = STGCSPlannerClient.plan(
                instance, queries, MRMPPerformanceComparison.low_level_spec(planner_key),
                budget=float(BUDGET), child_rule=MRMPPerformanceComparison.CHILD_EXPANSION_MODE.name.lower(),
                method="pp" if planner_key in (
                    MRMPPerformanceComparison.PP_BFS_KEY, MRMPPerformanceComparison.WINDOWED_PP_KEY) else "pbs",
                window_span_factor=MRMPPerformanceComparison.WINDOW_SPAN_FACTOR if windowed else 0.0,
                execution_horizon_factor=MRMPPerformanceComparison.EXECUTION_HORIZON_FACTOR,
                dynamic_window_adjustment=MRMPPerformanceComparison.DYNAMIC_WINDOW_ADJUSTMENT if windowed else False,
            )
        elif planner_key == MRMPPerformanceComparison.PBS_ZETA_SIPP_KEY:
            solutions, entry = MRMPExperiment.run_pbs_zeta_sipp_with_solutions(
                instance,
                queries,
                budget=float(BUDGET),
                child_expansion_mode=MRMPPerformanceComparison.CHILD_EXPANSION_MODE,
                cell_size_multiplier=MRMPPerformanceComparison.ZETA_SIPP_CELL_SIZE_MULTIPLIER,
            )
        elif planner_key == MRMPPerformanceComparison.CB_GCS_KEY:
            solutions, entry = MRMPExperiment.run_cb_gcs_with_solutions(
                instance,
                queries,
                budget=float(BUDGET),
                time_horizon=MRMPPerformanceComparison.CB_GCS_TIME_HORIZON,
                time_step=MRMPPerformanceComparison.CB_GCS_TIME_STEP,
                max_rounded_paths=MRMPPerformanceComparison.micp_rounding_paths(record.stgcs_num_edges),
            )
        elif planner_key == MRMPPerformanceComparison.KCBS_KEY:
            solutions, entry = MRMPExperiment.run_ompl_kcbs_with_solutions(
                instance,
                queries,
                budget=float(BUDGET),
                low_level_solve_time=MRMPPerformanceComparison.KCBS_LOW_LEVEL_SOLVE_TIME,
                propagation_step_size=MRMPPerformanceComparison.KCBS_PROPAGATION_STEP_SIZE,
                min_control_duration=MRMPPerformanceComparison.KCBS_MIN_CONTROL_DURATION,
                max_control_duration=MRMPPerformanceComparison.KCBS_MAX_CONTROL_DURATION,
                goal_tolerance=MRMPPerformanceComparison.KCBS_GOAL_TOLERANCE,
                goal_bias=MRMPPerformanceComparison.KCBS_GOAL_BIAS,
                intermediate_states=MRMPPerformanceComparison.KCBS_INTERMEDIATE_STATES,
                num_threads=MRMPPerformanceComparison.KCBS_NUM_THREADS,
                seed=MRMPPerformanceComparison.KCBS_SEED,
            )
        else:
            raise KeyError(f"Unknown MRMP comparison planner {planner_key!r}.")

        return viewer_payload(record, planner_key, entry, solutions)
    finally:
        del queries
        del instance
        gc.collect()


def load_or_run_planner(record, planner_key):
    path = solution_path(record, planner_key)
    if path.exists():
        payload = json.loads(path.read_text())
        validate_payload(payload, record, planner_key)
        print(f"Reusing {PLANNER_LABELS[planner_key]}: {path.relative_to(ROOT)}")
        return payload

    print(f"Running {PLANNER_LABELS[planner_key]} for {record.instance_id}")
    payload = run_planner(record, planner_key)
    write_json(path, payload)
    return payload


def load_or_run_all_planners(record):
    payloads = []
    fixed_priority_payloads = None
    skipped = []
    missing = {
        name for name, module_name in NATIVE_BASELINE_MODULES.items()
        if importlib.util.find_spec(module_name) is None
    }
    if missing:
        missing_text = ", ".join(sorted(missing))
        print(
            "WARNING: Missing native baseline build(s): "
            f"{missing_text}. The K-CBS and ST-RRT* baselines should be built for the complete comparison; "
            "continuing with available planners."
        )
    for planner_key in PLANNER_KEYS:
        native_baseline = None
        if planner_key in MRMPPerformanceComparison.FIXED_PP_ST_RRT_STAR_OUTPUT_KEYS:
            native_baseline = "ST-RRT*"
        elif planner_key == MRMPPerformanceComparison.KCBS_KEY:
            native_baseline = "K-CBS"
        if planner_key in MRMPPerformanceComparison.FIXED_PP_ST_RRT_STAR_OUTPUT_KEYS:
            try:
                if fixed_priority_payloads is None:
                    paths = [
                        solution_path(record, key)
                        for key in MRMPPerformanceComparison.FIXED_PP_ST_RRT_STAR_OUTPUT_KEYS
                    ]
                    if native_baseline in missing and not all(path.exists() for path in paths):
                        skipped.append((planner_key, f"missing native {native_baseline} build"))
                        continue
                    fixed_priority_payloads = load_or_run_fixed_priority_strrt_star(record)
                payloads.append(fixed_priority_payloads[planner_key])
            except Exception as exc:
                skipped.append((planner_key, str(exc)))
                fixed_priority_payloads = {}
        else:
            if native_baseline in missing and not solution_path(record, planner_key).exists():
                skipped.append((planner_key, f"missing native {native_baseline} build"))
                continue
            try:
                payloads.append(load_or_run_planner(record, planner_key))
            except Exception as exc:
                skipped.append((planner_key, str(exc)))
    for planner_key, reason in skipped:
        print(f"WARNING: Skipping {PLANNER_LABELS[planner_key]}: {reason}")
    return payloads, skipped


if __name__ == "__main__":
    domain_key, instance_id = INSTANCE
    records = {record.instance_id: record for record in load_manifest(MANIFEST_PATH)}
    record = records.get(instance_id)
    if record is None:
        raise ValueError(f"{instance_id!r} is not in {MANIFEST_PATH}.")
    if record.domain_key != domain_key:
        raise ValueError(f"{instance_id!r} belongs to {record.domain_key!r}, not {domain_key!r}.")

    payloads, skipped = load_or_run_all_planners(record)
    for payload in payloads:
        planner_key = str(payload.get("planner_key"))
        if planner_key not in PLANNER_KEYS:
            raise ValueError(f"Unexpected solution planner {planner_key!r}.")
        validate_payload(payload, record, planner_key)
    combined_path = combined_solution_path(record)
    write_json(combined_path, payloads)

    print(json.dumps({
        "domain": record.domain_key,
        "instance_id": record.instance_id,
        "budget": float(BUDGET),
        "combined_solution": str(combined_path.relative_to(ROOT)),
        "requested_planners": [PLANNER_LABELS[planner_key] for planner_key in PLANNER_KEYS],
        "loaded_planners": [payload["planner_name"] for payload in payloads],
        "skipped_planners": [
            {
                "planner": PLANNER_LABELS[planner_key],
                "reason": reason,
            }
            for planner_key, reason in skipped
        ],
        "solutions": [
            {
                "planner": payload["planner_name"],
                "runtime": float(payload["runtime"]),
                "sum_of_costs": float(payload["cost"]),
                "makespan": float(payload["makespan"]),
                "completed_agents": int(payload["num_completed_agents"]),
                "solution": str(solution_path(record, str(payload["planner_key"])).relative_to(ROOT)),
            }
            for payload in payloads
        ],
    }, indent=2))

    ViewerServer.start(
        MANIFEST_PATH,
        BASE_ROOT,
        record.instance_id,
        combined_path,
    )
