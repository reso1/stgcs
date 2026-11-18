from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/stgcs-mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/stgcs-xdg-cache")

from experiments.environment.env import Env
from stgcs_native.types import ChildExpansionMode
from experiments.planners.mrmp import SearchPlannerSpec
from experiments.native import STGCSPlannerClient, NativeGeometry
from types import SimpleNamespace
from stgcs_native.types import MPQuery
from stgcs_native.types import STTrajectory
from stgcs_native.trajopt import GlobalTrajOptConfig, GlobalTrajOptResult, GlobalTrajectoryOptimizer
from visualization.solution_io import SolutionIO
from functools import partial
from visualization.viewer.server import ViewerServer


@dataclass(frozen=True)
class SequentialMRMPStage:
    name: str
    queries: tuple[MPQuery, ...]


@dataclass(frozen=True)
class SequentialMRMPAssignment:
    queries: tuple[MPQuery, ...]
    target_indices: tuple[int, ...]
    cost: float


@dataclass(frozen=True)
class SequentialMRMPTask:
    name: str
    instance_id: str
    task_type: str
    robot_radius: float
    vlimit: float
    tmax: float
    stage_timeout_secs: float
    search_eps: float
    planner_coordination: str
    window_alpha: float
    window_beta: float
    dynamic_window_adjustment: bool
    child_expansion_rule: str
    child_expansion_mode: ChildExpansionMode
    cspace: tuple[np.ndarray, ...]
    stages: tuple[SequentialMRMPStage, ...]

    @property
    def num_agents(self) -> int:
        return len(self.stages[0].queries)

    @property
    def space_dim(self) -> int:
        return int(self.cspace[0].shape[1])

    @property
    def stage_names(self) -> list[str]:
        return [stage.name for stage in self.stages]


class SequentialMRMPDemo:
    DEMO_KEY = "sequential-mrmp"
    DEFAULT_OUTPUT_ROOT = ROOT / "data/solutions/demo_sequential_tasks"
    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 8766

    PBS_COORDINATION = "pbs"
    WINDOWED_PBS_COORDINATION = "windowed-pbs"
    SUPPORTED_COORDINATIONS = frozenset({PBS_COORDINATION, WINDOWED_PBS_COORDINATION})

    DEFAULT_CHILD_EXPANSION_RULE = "num_conflicts"
    CHILD_EXPANSION_RULES = {
        "lazy": ChildExpansionMode.LAZY,
        "soc": ChildExpansionMode.SOC,
        "makespan": ChildExpansionMode.MAKESPAN,
        "num_conflicts": ChildExpansionMode.NUM_CONFLICTS,
    }
    CHILD_EXPANSION_RULE_LABELS = {
        "lazy": "Lazy",
        "soc": "SOC",
        "makespan": "Makespan",
        "num_conflicts": "NumConflicts",
    }

    MRMP_PLANNER_KEY = "sequential-mrmp-wpbs"
    MRMP_PLANNER_NAME = "wPBS-NumConflicts + Search(h_mot+GUB+delta_pos)"
    TRAJOPT_PLANNER_KEY = f"{MRMP_PLANNER_KEY}-global-trajopt"
    TRAJOPT_PLANNER_NAME = f"{MRMP_PLANNER_NAME} + global trajopt"
    ASSIGNMENT_STRATEGY = "fixed-query-order"

    GLOBAL_TRAJOPT_SAMPLE_DT_FACTOR = 0.25
    GLOBAL_TRAJOPT_WINDOW_SPAN_FACTOR = 2.0
    GLOBAL_TRAJOPT_STRIDE_FACTOR = 1.0
    GLOBAL_TRAJOPT_SOLVER_MAX_ITER = 1_000_000
    GLOBAL_TRAJOPT_SOLVER_EPS = 1e-6
    GLOBAL_TRAJOPT_CLEARANCE_MARGIN = 0.0
    GLOBAL_TRAJOPT_COLLISION_TOLERANCE = 1e-4

    @classmethod
    def parse_args(cls, description: str | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=description or "Solve a self-contained sequential MRMP demo.")
        parser.add_argument(
            "phase",
            nargs="?",
            choices=("serve", "targets"),
            default="serve",
            help="`serve` runs MRMP, trajopt, then starts the viewer; `targets` plots stage targets.",
        )
        parser.add_argument("--output-dir", type=Path, default=None)
        parser.add_argument("--stage-timeout-secs", type=float, default=None)
        parser.add_argument("--force-solve", action="store_true",
                            help="Recompute MRMP and refresh trajectory optimization.")
        parser.add_argument("--force-trajopt", action="store_true",
                            help="Recompute trajectory optimization using the MRMP solution.")
        parser.add_argument("--targets-output", type=Path, default=None)
        parser.add_argument("--no-show-targets", action="store_true")
        parser.add_argument("--no-serve", action="store_true")
        parser.add_argument("--host", type=str, default=cls.DEFAULT_HOST)
        parser.add_argument("--port", type=int, default=cls.DEFAULT_PORT)
        return parser.parse_args()

    @classmethod
    def run_cli(cls, config: dict[str, Any], description: str | None = None) -> None:
        args = cls.parse_args(description)
        task = cls.load_task(config, stage_timeout_override=args.stage_timeout_secs)
        paths = cls.resolved_paths(args, task)

        if args.phase == "targets":
            cls.plot_query_targets(task, output_path=args.targets_output, show=not bool(args.no_show_targets))
            return

        payload = cls.run_pipeline(task, paths, force_solve=args.force_solve, force_trajopt=args.force_trajopt)
        print(f"MRMP solution: {paths['mrmp_json']}")
        print(f"Trajopt solution: {paths['trajopt_json']}")
        if args.no_serve:
            print(
                json.dumps(
                    {
                        "instance_id": task.instance_id,
                        "num_saved_solutions": len(payload.get("solutions", [])),
                    },
                    indent=2,
                ),
                flush=True,
            )
            return
        cls.serve_manifest(task, payload, host=str(args.host), port=int(args.port))

    @classmethod
    def resolved_paths(cls, args: argparse.Namespace, task: SequentialMRMPTask) -> dict[str, Path]:
        output_dir = Path(args.output_dir) if args.output_dir is not None else cls.DEFAULT_OUTPUT_ROOT / task.instance_id
        return {
            "mrmp_json": output_dir / "mrmp_solution.json",
            "trajopt_json": output_dir / "trajopt_solution.json",
        }

    @classmethod
    def load_task(
        cls,
        payload: dict[str, Any],
        stage_timeout_override: float | None = None,
    ) -> SequentialMRMPTask:
        name = str(payload["name"])
        instance_id = str(payload["instance_id"])
        task_type = str(payload["task_type"])
        robot_radius = cls.positive_float(payload["robot_radius"], "robot_radius")
        vlimit = cls.positive_float(payload["vlimit"], "vlimit")
        tmax = cls.positive_float(payload["tmax"], "tmax")
        stage_timeout_secs = cls.positive_float(
            payload["stage_timeout_secs"] if stage_timeout_override is None else stage_timeout_override,
            "stage_timeout_secs",
        )
        search_eps = cls.positive_float(payload["search_eps"], "search_eps")
        planner_config = cls.parse_planner_config(payload)
        cspace = cls.parse_cspace(payload)
        stages = cls.parse_stages(payload, vlimit=vlimit, space_dim=int(cspace[0].shape[1]))
        task = SequentialMRMPTask(
            name=name,
            instance_id=instance_id,
            task_type=task_type,
            robot_radius=robot_radius,
            vlimit=vlimit,
            tmax=tmax,
            stage_timeout_secs=stage_timeout_secs,
            search_eps=search_eps,
            planner_coordination=planner_config["coordination"],
            window_alpha=planner_config["window_alpha"],
            window_beta=planner_config["window_beta"],
            dynamic_window_adjustment=planner_config["dynamic_window_adjustment"],
            child_expansion_rule=planner_config["child_expansion_rule"],
            child_expansion_mode=planner_config["child_expansion_mode"],
            cspace=tuple(cspace),
            stages=tuple(stages),
        )
        cls.validate_task(task)
        return task

    @classmethod
    def parse_planner_config(cls, payload: dict[str, Any]) -> dict[str, Any]:
        raw_planner = payload.get("planner", {})
        if not isinstance(raw_planner, dict):
            raise ValueError("planner must be a mapping.")
        coordination = str(raw_planner.get("coordination", cls.WINDOWED_PBS_COORDINATION))
        if coordination not in cls.SUPPORTED_COORDINATIONS:
            raise ValueError(f"Unsupported coordination {coordination!r}.")
        child_expansion_rule = str(raw_planner.get("child_expansion_rule", cls.DEFAULT_CHILD_EXPANSION_RULE))
        return {
            "coordination": coordination,
            "window_alpha": cls.positive_float(raw_planner.get("window_alpha", 5.0), "planner.window_alpha"),
            "window_beta": cls.unit_interval_float(raw_planner.get("window_beta", 1.0), "planner.window_beta"),
            "dynamic_window_adjustment": cls.bool_value(
                raw_planner.get("dynamic_window_adjustment", True),
                "planner.dynamic_window_adjustment",
            ),
            "child_expansion_rule": child_expansion_rule,
            "child_expansion_mode": cls.child_expansion_mode(child_expansion_rule),
        }

    @classmethod
    def parse_cspace(cls, payload: dict[str, Any]) -> list[np.ndarray]:
        if "cspace" in payload:
            raw_cspace = payload["cspace"]
            if not isinstance(raw_cspace, list) or not raw_cspace:
                raise ValueError("cspace must be a non-empty list.")
            cspace = [cls.point_matrix(poly, f"cspace[{idx}]") for idx, poly in enumerate(raw_cspace)]
        else:
            bounds = cls.point_matrix(payload["bounds"], "bounds")
            if bounds.shape != (2, 2):
                raise ValueError("bounds must be [[xmin, ymin], [xmax, ymax]].")
            lb, ub = bounds
            if np.any(ub <= lb):
                raise ValueError("bounds upper corner must be strictly greater than lower corner.")
            cspace = [
                np.asarray(
                    [[lb[0], lb[1]], [ub[0], lb[1]], [ub[0], ub[1]], [lb[0], ub[1]]],
                    dtype=float,
                )
            ]
        if any(poly.shape[1] != 2 for poly in cspace):
            raise ValueError("Sequential MRMP demos currently support 2D C-spaces.")
        return cspace

    @classmethod
    def parse_stages(cls, payload: dict[str, Any], vlimit: float, space_dim: int) -> list[SequentialMRMPStage]:
        raw_stages = payload.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("Task config must contain a non-empty stages list.")
        stages = []
        used_names = set()
        for stage_idx, raw_stage in enumerate(raw_stages):
            stage_name = cls.slug(str(raw_stage.get("name") or f"stage_{stage_idx:03d}"))
            if stage_name in used_names:
                raise ValueError(f"Duplicate stage name {stage_name!r}.")
            used_names.add(stage_name)
            raw_queries = raw_stage.get("queries")
            if not isinstance(raw_queries, list) or not raw_queries:
                raise ValueError(f"{stage_name}.queries must be a non-empty list.")
            queries = tuple(
                cls.parse_query(raw_query, vlimit=vlimit, space_dim=space_dim, name=f"{stage_name}.queries[{idx}]")
                for idx, raw_query in enumerate(raw_queries)
            )
            stages.append(SequentialMRMPStage(stage_name, queries))
        return stages

    @classmethod
    def parse_query(cls, raw_query: Any, vlimit: float, space_dim: int, name: str) -> MPQuery:
        if not isinstance(raw_query, dict):
            raise ValueError(f"{name} must be a mapping.")
        query_vlimit = cls.positive_float(raw_query.get("vlimit", vlimit), f"{name}.vlimit")
        if not math.isclose(query_vlimit, vlimit, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{name}.vlimit={query_vlimit:g} does not match task vlimit={vlimit:g}.")
        return MPQuery(
            start=cls.point_vector(raw_query.get("start"), f"{name}.start", space_dim),
            goal=cls.point_vector(raw_query.get("goal"), f"{name}.goal", space_dim),
            t_start=cls.nonnegative_float(raw_query.get("t_start", 0.0), f"{name}.t_start"),
            is_stay=bool(raw_query.get("is_stay", True)),
            vlimit=float(vlimit),
        )

    @staticmethod
    def point_matrix(raw_points: Any, name: str) -> np.ndarray:
        points = np.asarray(raw_points, dtype=float)
        if points.ndim != 2 or points.shape[1] == 0:
            raise ValueError(f"{name} must be a 2D numeric point array.")
        if not np.all(np.isfinite(points)):
            raise ValueError(f"{name} contains non-finite coordinates.")
        return points

    @staticmethod
    def point_vector(raw_point: Any, name: str, space_dim: int) -> np.ndarray:
        point = np.asarray(raw_point, dtype=float).reshape(-1)
        if point.shape != (space_dim,):
            raise ValueError(f"{name} must contain {space_dim} coordinates.")
        if not np.all(np.isfinite(point)):
            raise ValueError(f"{name} contains non-finite coordinates.")
        return point

    @staticmethod
    def positive_float(value: Any, name: str) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
        return numeric

    @staticmethod
    def nonnegative_float(value: Any, name: str) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative.")
        return numeric

    @staticmethod
    def unit_interval_float(value: Any, name: str) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1].")
        return numeric

    @staticmethod
    def bool_value(value: Any, name: str) -> bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"{name} must be a boolean.")

    @staticmethod
    def slug(value: str) -> str:
        chars = []
        for char in value.strip().lower():
            if char.isalnum():
                chars.append(char)
            elif char in {"-", "_", "."}:
                chars.append(char)
            elif char.isspace():
                chars.append("-")
        return "".join(chars).strip("-_.") or "task"

    @classmethod
    def child_expansion_mode(cls, rule: str) -> ChildExpansionMode:
        try:
            return cls.CHILD_EXPANSION_RULES[str(rule)]
        except KeyError as exc:
            raise ValueError(f"Unknown child expansion rule {rule!r}.") from exc

    @classmethod
    def child_expansion_rule_label(cls, rule: str) -> str:
        return cls.CHILD_EXPANSION_RULE_LABELS[str(rule)]

    @classmethod
    def validate_task(cls, task: SequentialMRMPTask) -> None:
        if task.space_dim != 2:
            raise ValueError("Sequential MRMP demos support 2D tasks.")
        if task.num_agents <= 0:
            raise ValueError("Task must contain at least one robot.")
        expected_starts = [np.asarray(query.start, dtype=float) for query in task.stages[0].queries]
        for stage in task.stages:
            if len(stage.queries) != task.num_agents:
                raise ValueError(f"Stage {stage.name!r} must contain {task.num_agents} queries.")
            for agent_idx, query in enumerate(stage.queries):
                if not np.allclose(query.start, expected_starts[agent_idx], rtol=0.0, atol=1e-9):
                    raise ValueError(
                        f"Stage chain is not contiguous for agent {agent_idx} at stage {stage.name!r}."
                    )
            expected_starts = [np.asarray(query.goal, dtype=float) for query in stage.queries]

    @classmethod
    def initial_positions(cls, task: SequentialMRMPTask) -> list[np.ndarray]:
        return [np.asarray(query.start, dtype=float) for query in task.stages[0].queries]

    @classmethod
    def assign_stage_queries(
        cls,
        task: SequentialMRMPTask,
        current_positions: Sequence[np.ndarray],
        stage: SequentialMRMPStage,
    ) -> SequentialMRMPAssignment:
        assigned_queries = []
        assignment_cost = 0.0
        for agent_idx, (current_position, stage_query) in enumerate(zip(current_positions, stage.queries)):
            current = np.asarray(current_position, dtype=float)
            if not np.allclose(current, stage_query.start, rtol=0.0, atol=1e-8):
                raise ValueError(
                    f"Stage {stage.name!r} query {agent_idx} starts at {stage_query.start.tolist()}, "
                    f"expected {current.tolist()}."
                )
            assigned_queries.append(
                MPQuery(
                    start=current,
                    goal=np.asarray(stage_query.goal, dtype=float),
                    t_start=float(stage_query.t_start),
                    is_stay=bool(stage_query.is_stay),
                    vlimit=float(task.vlimit),
                )
            )
            assignment_cost += float(np.linalg.norm(stage_query.goal - current))
        return SequentialMRMPAssignment(tuple(assigned_queries), tuple(range(task.num_agents)), assignment_cost)

    @classmethod
    def build_env(cls, task: SequentialMRMPTask) -> Env:
        return Env(
            name=task.name,
            CSpace=[np.asarray(poly, dtype=float) for poly in task.cspace],
            robot_radius=float(task.robot_radius),
        )

    @classmethod
    def build_stgcs(cls, task: SequentialMRMPTask, env: Env):
        stgcs = env.build_STGCS(t0=0.0, tmax=float(task.tmax), vlimit=float(task.vlimit))
        stgcs.make_leaves_roots()
        return stgcs

    @classmethod
    def window_span(cls, task: SequentialMRMPTask) -> float:
        return float(task.window_alpha) * float(task.robot_radius) / float(task.vlimit)

    @classmethod
    def mrmp_planner_key(cls, task: SequentialMRMPTask) -> str:
        del task
        return cls.MRMP_PLANNER_KEY

    @classmethod
    def mrmp_planner_name(cls, task: SequentialMRMPTask) -> str:
        if task.planner_coordination == cls.PBS_COORDINATION:
            return "PBS + Search(h_mot+GUB+delta_pos)"
        rule_label = cls.child_expansion_rule_label(task.child_expansion_rule)
        return f"wPBS-{rule_label} + Search(h_mot+GUB+delta_pos)"

    @classmethod
    def trajopt_planner_key(cls, task: SequentialMRMPTask) -> str:
        return f"{cls.mrmp_planner_key(task)}-global-trajopt"

    @classmethod
    def trajopt_planner_name(cls, task: SequentialMRMPTask) -> str:
        return f"{cls.mrmp_planner_name(task)} + global trajopt"

    @classmethod
    def planner_key_for_phase(cls, task: SequentialMRMPTask, phase: str) -> str:
        if phase == "mrmp":
            return cls.mrmp_planner_key(task)
        if phase == "trajopt":
            return cls.trajopt_planner_key(task)
        raise ValueError(f"Unsupported solution phase {phase!r}.")

    @classmethod
    def planner_name_for_phase(cls, task: SequentialMRMPTask, phase: str) -> str:
        if phase == "mrmp":
            return cls.mrmp_planner_name(task)
        if phase == "trajopt":
            return cls.trajopt_planner_name(task)
        raise ValueError(f"Unsupported solution phase {phase!r}.")

    @classmethod
    def planner_metadata(cls, task: SequentialMRMPTask) -> dict[str, Any]:
        metadata = {
            "coordination": task.planner_coordination,
            "child_expansion_rule": task.child_expansion_rule,
            "child_expansion_mode": task.child_expansion_mode.name,
            "low_level_heuristic": "h_mot",
            "dominance_checks": ["GUB", "delta_pos"],
            "epsilon": float(task.search_eps),
            "vlimit": float(task.vlimit),
            "tmax": float(task.tmax),
            "stage_timeout_secs": float(task.stage_timeout_secs),
        }
        if task.planner_coordination == cls.WINDOWED_PBS_COORDINATION:
            metadata.update(
                {
                    "window_alpha": float(task.window_alpha),
                    "window_span": cls.window_span(task),
                    "window_beta": float(task.window_beta),
                    "execution_horizon_factor": float(task.window_beta),
                    "dynamic_window_adjustment": bool(task.dynamic_window_adjustment),
                }
            )
        return metadata

    @classmethod
    def plan_stage(cls, task, env, stgcs, stage_name, queries):
        spec = SearchPlannerSpec("h_mot", ("GUB", "delta_pos"), epsilon=task.search_eps)
        solutions, entry = STGCSPlannerClient.plan(
            SimpleNamespace(env=env, stgcs=stgcs), queries, spec,
            task.stage_timeout_secs, task.child_expansion_rule,
            task.window_alpha if task.planner_coordination == cls.WINDOWED_PBS_COORDINATION else 0.0,
            task.window_beta, task.dynamic_window_adjustment)
        if not entry.is_success:
            raise RuntimeError(f"Native planner failed for stage {stage_name!r} after {entry.runtime:.3f}s "
                               f"(budget {task.stage_timeout_secs:g}s).")
        end = max(float(path.xT[-1]) for path in solutions)
        return [cls.pad_trajectory_to_time(path, end) for path in solutions], end, entry.runtime

    @classmethod
    def solve_mrmp(cls, task: SequentialMRMPTask) -> dict[str, Any]:
        env = cls.build_env(task)
        stgcs = cls.build_stgcs(task, env)
        chunks_by_agent: list[list[STTrajectory]] = [[] for _ in range(task.num_agents)]
        stage_trajectories: dict[str, list[STTrajectory]] = {}
        stage_queries: dict[str, tuple[MPQuery, ...]] = {}
        stage_assignment_target_indices: dict[str, tuple[int, ...]] = {}
        stage_results: list[dict[str, Any]] = []
        current_positions = cls.initial_positions(task)
        stage_offset = 0.0
        started = time.perf_counter()

        for stage in task.stages:
            print(f"Planning {stage.name} with {task.num_agents} robots...")
            stage_started = time.perf_counter()
            assignment = cls.assign_stage_queries(task, current_positions, stage)
            trajectories, stage_end_time, runtime = cls.plan_stage(
                task,
                env,
                stgcs,
                stage.name,
                assignment.queries,
            )
            stage_trajectories[stage.name] = trajectories
            stage_queries[stage.name] = assignment.queries
            stage_assignment_target_indices[stage.name] = assignment.target_indices
            for agent_idx, trajectory in enumerate(trajectories):
                chunks_by_agent[agent_idx].append(cls.shift_trajectory_time(trajectory, stage_offset))
            current_positions = [np.asarray(query.goal, dtype=float) for query in assignment.queries]
            stage_offset += stage_end_time
            stage_results.append(
                {
                    "stage": stage.name,
                    "status": "SUCCESS",
                    "is_success": True,
                    "runtime": cls.finite_float_or_none(runtime),
                    "wall_time": cls.finite_float_or_none(time.perf_counter() - stage_started),
                    "makespan": cls.finite_float_or_none(stage_end_time),
                    "assignment_goal_indices": list(assignment.target_indices),
                    "assignment_cost": cls.finite_float_or_none(assignment.cost),
                }
            )
            print(f"{stage.name} solved in {runtime:.3f}s with stage makespan {stage_end_time:.3f}s.")

        full_trajectories = [STTrajectory.from_chunks(chunks) for chunks in chunks_by_agent]
        return cls.solution_payload(
            task=task,
            phase="mrmp",
            stage_trajectories=stage_trajectories,
            stage_queries=stage_queries,
            stage_assignment_target_indices=stage_assignment_target_indices,
            full_trajectories=full_trajectories,
            stage_results=stage_results,
            runtime=time.perf_counter() - started,
        )

    @classmethod
    def solve_trajopt_from_payload(
        cls,
        task: SequentialMRMPTask,
        mrmp_payload: dict[str, Any],
    ) -> dict[str, Any]:
        env = cls.build_env(task)
        stgcs = cls.build_stgcs(task, env)
        stage_trajectories = cls.stage_trajectories_from_payload(task, mrmp_payload)
        stage_queries, stage_assignment_target_indices = cls.stage_queries_from_payload(task, mrmp_payload)
        optimized_stages: dict[str, list[STTrajectory]] = {}
        stage_results: list[dict[str, Any]] = []
        chunks_by_agent: list[list[STTrajectory]] = [[] for _ in range(task.num_agents)]
        stage_offset = 0.0
        started = time.perf_counter()

        for stage in task.stages:
            print(f"Optimizing {stage.name} from MRMP trajectories...")
            stage_started = time.perf_counter()
            queries = stage_queries[stage.name]
            cls.validate_stage_trajectories(stage.name, queries, stage_trajectories[stage.name])
            config = cls.global_trajopt_config(task, progress=False)
            stored_pairwise_ok = cls.pairwise_collision_free(
                stgcs,
                stage_trajectories[stage.name],
                goal_stays=[query.is_stay for query in queries],
                robot_radius=float(task.robot_radius),
            )
            if not stored_pairwise_ok:
                raise RuntimeError(f"MRMP trajectories for {stage.name!r} are not pairwise collision-free.")
            if cls.is_return_to_start_stage(task, stage):
                optimized = [trajectory.copy() for trajectory in stage_trajectories[stage.name]]
                optimized_stages[stage.name] = optimized
                stage_end_time = max(float(trajectory.xT[-1]) for trajectory in optimized)
                for agent_idx, trajectory in enumerate(optimized):
                    chunks_by_agent[agent_idx].append(cls.shift_trajectory_time(trajectory, stage_offset))
                stage_offset += stage_end_time
                stage_results.append(
                    cls.mrmp_passthrough_stage_result_payload(
                        stage.name,
                        optimized,
                        config,
                        continuous_pairwise_collision_free=stored_pairwise_ok,
                        wall_time=time.perf_counter() - stage_started,
                    )
                )
                print(f"{stage.name} kept as MRMP trajectory in trajopt output with stage makespan {stage_end_time:.3f}s.")
                continue

            optimized, result, config, continuous_pairwise_ok = cls.optimize_stage(
                task,
                stgcs,
                stage_trajectories[stage.name],
                queries,
            )
            optimized_stages[stage.name] = optimized
            stage_end_time = max(float(trajectory.xT[-1]) for trajectory in optimized)
            for agent_idx, trajectory in enumerate(optimized):
                chunks_by_agent[agent_idx].append(cls.shift_trajectory_time(trajectory, stage_offset))
            stage_offset += stage_end_time
            stage_results.append(
                cls.trajopt_stage_result_payload(
                    stage.name,
                    result,
                    config,
                    stage_end_time,
                    continuous_pairwise_ok,
                    wall_time=time.perf_counter() - stage_started,
                )
            )
            print(f"{stage.name} optimized in {result.runtime:.3f}s with stage makespan {stage_end_time:.3f}s.")

        full_trajectories = [STTrajectory.from_chunks(chunks) for chunks in chunks_by_agent]
        return cls.solution_payload(
            task=task,
            phase="trajopt",
            stage_trajectories=optimized_stages,
            stage_queries=stage_queries,
            stage_assignment_target_indices=stage_assignment_target_indices,
            full_trajectories=full_trajectories,
            stage_results=stage_results,
            runtime=time.perf_counter() - started,
        )

    @classmethod
    def stage_queries_from_payload(
        cls,
        task: SequentialMRMPTask,
        payload: dict[str, Any],
    ) -> tuple[dict[str, tuple[MPQuery, ...]], dict[str, tuple[int, ...]]]:
        raw_stage_queries = payload.get("stage_queries")
        if not isinstance(raw_stage_queries, dict):
            raise ValueError("MRMP payload does not contain stage_queries.")
        queries_by_stage: dict[str, tuple[MPQuery, ...]] = {}
        target_indices_by_stage: dict[str, tuple[int, ...]] = {}
        for stage in task.stages:
            raw_queries = raw_stage_queries.get(stage.name)
            if not isinstance(raw_queries, list):
                raise ValueError(f"MRMP payload does not contain assigned queries for stage {stage.name!r}.")
            if len(raw_queries) != task.num_agents:
                raise ValueError(f"Stage {stage.name!r} contains {len(raw_queries)} assigned queries.")
            queries: list[MPQuery] = []
            target_indices: list[int] = []
            for agent_idx, raw_query in enumerate(raw_queries):
                if not isinstance(raw_query, dict):
                    raise ValueError(f"stage_queries[{stage.name!r}][{agent_idx}] must be a mapping.")
                payload_agent_idx = int(raw_query.get("agent_index", agent_idx))
                if payload_agent_idx != agent_idx:
                    raise ValueError(
                        f"stage_queries[{stage.name!r}][{agent_idx}] has agent_index={payload_agent_idx}."
                    )
                target_idx = int(raw_query.get("target_index", -1))
                if target_idx < 0 or target_idx >= task.num_agents:
                    raise ValueError(
                        f"stage_queries[{stage.name!r}][{agent_idx}] has invalid target_index={target_idx}."
                    )
                queries.append(
                    cls.parse_query(
                        raw_query,
                        vlimit=task.vlimit,
                        space_dim=task.space_dim,
                        name=f"stage_queries[{stage.name}][{agent_idx}]",
                    )
                )
                target_indices.append(target_idx)
            queries_by_stage[stage.name] = tuple(queries)
            target_indices_by_stage[stage.name] = tuple(target_indices)
        cls.validate_assigned_stage_queries(task, queries_by_stage, target_indices_by_stage)
        return queries_by_stage, target_indices_by_stage

    @classmethod
    def stage_trajectories_from_payload(
        cls,
        task: SequentialMRMPTask,
        payload: dict[str, Any],
    ) -> dict[str, list[STTrajectory]]:
        raw_stages = payload.get("stage_trajectories")
        if not isinstance(raw_stages, dict):
            raise ValueError("MRMP payload does not contain stage_trajectories.")
        trajectories_by_stage: dict[str, list[STTrajectory]] = {}
        for stage in task.stages:
            raw_trajectories = raw_stages.get(stage.name)
            if not isinstance(raw_trajectories, list):
                raise ValueError(f"MRMP payload does not contain trajectories for stage {stage.name!r}.")
            trajectories = [cls.trajectory_from_payload(item) for item in raw_trajectories]
            if len(trajectories) != task.num_agents:
                raise ValueError(f"Stage {stage.name!r} contains {len(trajectories)} trajectories.")
            trajectories_by_stage[stage.name] = trajectories
        return trajectories_by_stage

    @classmethod
    def validate_stage_trajectories(
        cls,
        stage_name: str,
        queries: Sequence[MPQuery],
        trajectories: Sequence[STTrajectory],
    ) -> None:
        if len(queries) != len(trajectories):
            raise ValueError(f"Stage {stage_name!r} query/trajectory count mismatch.")
        for idx, (query, trajectory) in enumerate(zip(queries, trajectories)):
            start = np.asarray(trajectory.x0[:-1], dtype=float)
            goal = np.asarray(trajectory.xT[:-1], dtype=float)
            if not np.allclose(start, query.start, rtol=0.0, atol=1e-8):
                raise ValueError(
                    f"Stored trajectory {idx} for {stage_name!r} starts at {start}, expected {query.start}."
                )
            if not np.allclose(goal, query.goal, rtol=0.0, atol=1e-8):
                raise ValueError(
                    f"Stored trajectory {idx} for {stage_name!r} ends at {goal}, expected {query.goal}."
                )

    @classmethod
    def is_return_to_start_stage(cls, task: SequentialMRMPTask, stage: SequentialMRMPStage) -> bool:
        if not task.stages or stage.name != task.stages[-1].name:
            return False
        initial_positions = cls.initial_positions(task)
        return all(
            np.allclose(query.goal, initial_positions[agent_idx], rtol=0.0, atol=1e-8)
            for agent_idx, query in enumerate(stage.queries)
        )

    @classmethod
    def optimize_stage(
        cls,
        task: SequentialMRMPTask,
        stgcs: Any,
        trajectories: Sequence[STTrajectory],
        queries: Sequence[MPQuery],
    ) -> tuple[list[STTrajectory], GlobalTrajOptResult, GlobalTrajOptConfig, bool]:
        config = cls.global_trajopt_config(task, progress=False)
        result = GlobalTrajectoryOptimizer.optimize_sliding_window(
            trajectories,
            float(task.robot_radius),
            float(task.vlimit),
            config=config,
            window_span=cls.global_trajopt_window_span(task),
            stride=cls.global_trajopt_stride(task),
            cells=NativeGeometry.cells(stgcs),
            env=None,
        )
        if not result.sampled_pairwise_collision_free:
            raise RuntimeError("Global trajectory optimization produced a sampled pairwise collision.")
        continuous_pairwise_ok = cls.pairwise_collision_free(
            stgcs,
            result.trajectories,
            goal_stays=[query.is_stay for query in queries],
            robot_radius=float(task.robot_radius),
        )
        stage_end_time = max(float(trajectory.xT[-1]) for trajectory in result.trajectories)
        padded = [cls.pad_trajectory_to_time(trajectory, stage_end_time) for trajectory in result.trajectories]
        return padded, result, config, continuous_pairwise_ok

    @classmethod
    def global_trajopt_window_span(cls, task: SequentialMRMPTask) -> float:
        return GlobalTrajOptConfig.scaled_time(cls.GLOBAL_TRAJOPT_WINDOW_SPAN_FACTOR, task.robot_radius, task.vlimit)

    @classmethod
    def global_trajopt_stride(cls, task: SequentialMRMPTask) -> float:
        return GlobalTrajOptConfig.scaled_time(cls.GLOBAL_TRAJOPT_STRIDE_FACTOR, task.robot_radius, task.vlimit)

    @classmethod
    def global_trajopt_config(cls, task: SequentialMRMPTask, progress: bool) -> GlobalTrajOptConfig:
        return GlobalTrajOptConfig(
            sample_dt=GlobalTrajOptConfig.scaled_time(cls.GLOBAL_TRAJOPT_SAMPLE_DT_FACTOR, task.robot_radius, task.vlimit),
            clearance_margin=cls.GLOBAL_TRAJOPT_CLEARANCE_MARGIN,
            solver_max_iter=cls.GLOBAL_TRAJOPT_SOLVER_MAX_ITER,
            solver_eps_abs=cls.GLOBAL_TRAJOPT_SOLVER_EPS,
            solver_eps_rel=cls.GLOBAL_TRAJOPT_SOLVER_EPS,
            include_trajectory_knot_times=True,
            progress=bool(progress),
        )

    @classmethod
    def pairwise_collision_free(
        cls,
        stgcs: Any,
        trajectories: Sequence[STTrajectory],
        goal_stays: Sequence[bool],
        robot_radius: float,
    ) -> bool:
        return GlobalTrajectoryOptimizer.pairwise_collision_free(
            trajectories, robot_radius, goal_stays, cls.GLOBAL_TRAJOPT_COLLISION_TOLERANCE)

    @classmethod
    def mrmp_passthrough_stage_result_payload(
        cls,
        stage_name: str,
        trajectories: Sequence[STTrajectory],
        config: GlobalTrajOptConfig,
        continuous_pairwise_collision_free: bool,
        wall_time: float,
    ) -> dict[str, Any]:
        stage_end_time = max(float(trajectory.xT[-1]) for trajectory in trajectories)
        inspected = GlobalTrajectoryOptimizer.inspect(trajectories, config)
        sample_times = inspected.times
        return {
            "stage": stage_name,
            "status": "SUCCESS",
            "is_success": True,
            "runtime": 0.0,
            "wall_time": cls.finite_float_or_none(wall_time),
            "makespan": cls.finite_float_or_none(stage_end_time),
            "sample_dt": float(config.sample_dt),
            "num_sample_times": int(sample_times.size),
            "sampled_pairwise_collision_free": True,
            "continuous_pairwise_collision_free": bool(continuous_pairwise_collision_free),
            "min_pairwise_distance": float(inspected.min_pairwise_distance),
            "max_velocity_component": float(inspected.max_velocity_component),
            "solver_message": "mrmp_passthrough",
            "trajopt_skipped": True,
            "skip_reason": "return_to_start_stage",
        }

    @classmethod
    def trajopt_stage_result_payload(
        cls,
        stage_name: str,
        result: GlobalTrajOptResult,
        config: GlobalTrajOptConfig,
        stage_end_time: float,
        continuous_pairwise_collision_free: bool,
        wall_time: float,
    ) -> dict[str, Any]:
        return {
            "stage": stage_name,
            "status": "SUCCESS",
            "is_success": True,
            "runtime": cls.finite_float_or_none(result.runtime),
            "wall_time": cls.finite_float_or_none(wall_time),
            "makespan": cls.finite_float_or_none(stage_end_time),
            "sample_dt": float(config.sample_dt),
            "num_sample_times": int(result.times.size),
            "sampled_pairwise_collision_free": bool(result.sampled_pairwise_collision_free),
            "continuous_pairwise_collision_free": bool(continuous_pairwise_collision_free),
            "min_pairwise_distance": float(result.min_pairwise_distance),
            "max_velocity_component": float(result.max_velocity_component),
            "solver_message": result.solver_message,
        }

    @staticmethod
    def pad_trajectory_to_time(trajectory: STTrajectory, target_time: float) -> STTrajectory:
        if target_time < float(trajectory.xT[-1]) - 1e-9:
            raise ValueError(f"Cannot pad trajectory ending after target time {target_time:.4f}.")
        padded = trajectory.copy()
        if np.isclose(float(padded.xT[-1]), float(target_time)):
            return padded
        padded.vertex_path.append(padded.vertex_path[-1])
        padded.points.append(np.hstack([padded.xT, padded.xT[:-1], float(target_time)]))
        return padded

    @staticmethod
    def shift_trajectory_time(trajectory: STTrajectory, time_offset: float) -> STTrajectory:
        shifted = trajectory.copy()
        for point in shifted.points:
            point[shifted.dim - 1] += float(time_offset)
            point[-1] += float(time_offset)
        return shifted

    @staticmethod
    def finite_float_or_none(value: float | int | None) -> float | None:
        return SolutionIO.finite_float_or_none(value)

    @staticmethod
    def trajectory_to_viewer_json(trajectory: STTrajectory | None) -> dict[str, Any]:
        return SolutionIO.trajectory_to_viewer_json(trajectory)

    @classmethod
    def trajectory_to_payload(cls, trajectory: STTrajectory) -> dict[str, Any]:
        return {
            "dim": int(trajectory.dim - 1),
            "vertex_path": list(trajectory.vertex_path),
            "points": [np.asarray(point, dtype=float).tolist() for point in trajectory.points],
            "viewer": cls.trajectory_to_viewer_json(trajectory),
        }

    @staticmethod
    def trajectory_from_payload(payload: dict[str, Any]) -> STTrajectory:
        dim = int(payload["dim"])
        vertex_path = [str(vertex) for vertex in payload["vertex_path"]]
        points = [np.asarray(point, dtype=float) for point in payload["points"]]
        return STTrajectory(vertex_path, points, dim=dim)

    @classmethod
    def query_to_viewer_json(cls, query: MPQuery, agent_idx: int) -> dict[str, Any]:
        return {
            "agent_index": int(agent_idx),
            "start": np.asarray(query.start, dtype=float).tolist(),
            "goal": np.asarray(query.goal, dtype=float).tolist(),
            "t_start": float(query.t_start),
            "is_stay": bool(query.is_stay),
            "vlimit": float(query.vlimit),
        }

    @classmethod
    def assigned_query_to_payload(cls, query: MPQuery, agent_idx: int, target_idx: int) -> dict[str, Any]:
        payload = cls.query_to_viewer_json(query, agent_idx)
        payload["target_index"] = int(target_idx)
        return payload

    @classmethod
    def validate_assigned_stage_queries(
        cls,
        task: SequentialMRMPTask,
        queries_by_stage: dict[str, Sequence[MPQuery]],
        target_indices_by_stage: dict[str, Sequence[int]],
    ) -> None:
        expected_starts = cls.initial_positions(task)
        expected_target_order = list(range(task.num_agents))
        for stage in task.stages:
            queries = list(queries_by_stage.get(stage.name, ()))
            target_indices = [int(idx) for idx in target_indices_by_stage.get(stage.name, ())]
            if len(queries) != task.num_agents:
                raise ValueError(f"Assigned query count mismatch for stage {stage.name!r}.")
            if sorted(target_indices) != expected_target_order:
                raise ValueError(f"Assigned target indices for stage {stage.name!r} are not a permutation.")
            target_points = [np.asarray(query.goal, dtype=float) for query in stage.queries]
            for agent_idx, query in enumerate(queries):
                target_idx = target_indices[agent_idx]
                if not np.allclose(query.start, expected_starts[agent_idx], rtol=0.0, atol=1e-8):
                    raise ValueError(f"Assigned query {agent_idx} for {stage.name!r} has the wrong start.")
                if not np.allclose(query.goal, target_points[target_idx], rtol=0.0, atol=1e-8):
                    raise ValueError(f"Assigned query {agent_idx} for {stage.name!r} has the wrong goal.")
            expected_starts = [np.asarray(query.goal, dtype=float) for query in queries]

    @classmethod
    def solution_payload(
        cls,
        task: SequentialMRMPTask,
        phase: str,
        stage_trajectories: dict[str, list[STTrajectory]],
        stage_queries: dict[str, Sequence[MPQuery]],
        stage_assignment_target_indices: dict[str, Sequence[int]],
        full_trajectories: Sequence[STTrajectory],
        stage_results: Sequence[dict[str, Any]],
        runtime: float,
    ) -> dict[str, Any]:
        cls.validate_assigned_stage_queries(task, stage_queries, stage_assignment_target_indices)
        durations = [float(trajectory.duration) for trajectory in full_trajectories]
        planner_key = cls.planner_key_for_phase(task, phase)
        planner_name = cls.planner_name_for_phase(task, phase)
        global_config = cls.global_trajopt_config(task, progress=False) if phase == "trajopt" else None
        return {

            "ok": True,
            "instance_id": task.instance_id,
            "demo": cls.DEMO_KEY,
            "task_name": task.name,
            "phase": phase,
            "planner_key": planner_key,
            "planner_name": planner_name,
            "status": "SUCCESS",
            "is_success": True,
            "task_type": task.task_type,
            "assignment_strategy": cls.ASSIGNMENT_STRATEGY,
            "assignment_solver": None,
            "runtime": cls.finite_float_or_none(runtime),
            "cost": cls.finite_float_or_none(sum(durations)),
            "makespan": cls.finite_float_or_none(max(durations, default=0.0)),
            "num_agents": int(task.num_agents),
            "num_completed_agents": int(task.num_agents),
            "path_alpha": 0.98,
            "body_alpha": 0.82,
            "stage_names": list(task.stage_names),
            "env": cls.env_payload(task),
            "planner": cls.planner_metadata(task),
            "global_trajopt": None if phase != "trajopt" else {
                "backend": "stgcs_cplusplus",
                "sample_dt_factor": cls.GLOBAL_TRAJOPT_SAMPLE_DT_FACTOR,
                "window_span_factor": cls.GLOBAL_TRAJOPT_WINDOW_SPAN_FACTOR,
                "stride_factor": cls.GLOBAL_TRAJOPT_STRIDE_FACTOR,
                "window_span": cls.global_trajopt_window_span(task),
                "stride": cls.global_trajopt_stride(task),
                "velocity_gradient_weight": float(global_config.velocity_gradient_weight),
                "displacement_weight": float(global_config.displacement_weight),
                "displacement_jitter_weight": float(global_config.displacement_jitter_weight),
                "clearance_margin": float(global_config.clearance_margin),
                "solver_max_iter": cls.GLOBAL_TRAJOPT_SOLVER_MAX_ITER,
                "solver_eps": cls.GLOBAL_TRAJOPT_SOLVER_EPS,
            },
            "stage_results": list(stage_results),
            "stage_queries": {
                stage_name: [
                    cls.assigned_query_to_payload(
                        query,
                        agent_idx,
                        int(stage_assignment_target_indices[stage_name][agent_idx]),
                    )
                    for agent_idx, query in enumerate(stage_queries[stage_name])
                ]
                for stage_name in task.stage_names
            },
            "stage_trajectories": {
                stage_name: [
                    cls.trajectory_to_payload(trajectory)
                    for trajectory in stage_trajectories[stage_name]
                ]
                for stage_name in task.stage_names
            },
            "full_trajectories": [cls.trajectory_to_payload(trajectory) for trajectory in full_trajectories],
            "trajectories": [
                {
                    **cls.trajectory_to_viewer_json(trajectory),
                    "agent_index": agent_idx,
                }
                for agent_idx, trajectory in enumerate(full_trajectories)
            ],
        }

    @classmethod
    def env_payload(cls, task: SequentialMRMPTask) -> dict[str, Any]:
        env = cls.build_env(task)
        return {
            "name": task.name,
            "space_dim": int(task.space_dim),
            "cspace": "script-defined-cspace",
            "bounds": [
                np.asarray(env.lb, dtype=float).tolist(),
                np.asarray(env.ub, dtype=float).tolist(),
            ],
            "robot_radius": float(task.robot_radius),
        }

    @staticmethod
    def load_payload(path: Path) -> dict[str, Any]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {path}.")
        payload["output_path"] = str(path)
        return payload

    @classmethod
    def load_saved_solution_payload(
        cls,
        task: SequentialMRMPTask,
        path: Path,
        expected_phase: str,
    ) -> dict[str, Any] | None:
        if not Path(path).exists():
            return None
        try:
            payload = cls.load_payload(Path(path))
            cls.validate_solution_payload(task, payload, expected_phase=expected_phase)
        except Exception as exc:
            print(f"Skipping saved solution payload {path}: {exc}")
            return None
        return payload

    @classmethod
    def validate_solution_payload(
        cls,
        task: SequentialMRMPTask,
        payload: dict[str, Any],
        expected_phase: str,
    ) -> None:
        if not bool(payload.get("ok")):
            raise ValueError("Saved payload is not marked ok.")
        if payload.get("demo") != cls.DEMO_KEY:
            raise ValueError(f"Saved payload demo={payload.get('demo')!r}, expected {cls.DEMO_KEY!r}.")
        if payload.get("instance_id") != task.instance_id:
            raise ValueError("Saved payload belongs to a different demo instance.")
        if payload.get("phase") != expected_phase:
            raise ValueError(f"Saved payload phase is {payload.get('phase')!r}, expected {expected_phase!r}.")
        if payload.get("planner_key") != cls.planner_key_for_phase(task, expected_phase):
            raise ValueError("Saved payload planner key does not match this task.")
        if payload.get("planner_name") != cls.planner_name_for_phase(task, expected_phase):
            raise ValueError("Saved payload planner name does not match this task.")
        if payload.get("task_type") != task.task_type:
            raise ValueError("Saved payload task type does not match this task.")
        if int(payload.get("num_agents", -1)) != task.num_agents:
            raise ValueError("Saved payload agent count does not match this task.")
        if list(payload.get("stage_names", [])) != task.stage_names:
            raise ValueError("Saved payload stage sequence does not match this task.")
        if payload.get("planner") != cls.planner_metadata(task):
            raise ValueError("Saved payload planner metadata does not match this task.")
        queries_by_stage, _ = cls.stage_queries_from_payload(task, payload)
        trajectories_by_stage = cls.stage_trajectories_from_payload(task, payload)
        for stage in task.stages:
            cls.validate_stage_trajectories(stage.name, queries_by_stage[stage.name], trajectories_by_stage[stage.name])

    @classmethod
    def load_saved_solution_payloads(
        cls,
        task: SequentialMRMPTask,
        paths: dict[str, Path],
    ) -> list[dict[str, Any]]:
        payloads = []
        mrmp_payload = cls.load_saved_solution_payload(task, paths["mrmp_json"], expected_phase="mrmp")
        if mrmp_payload is not None:
            payloads.append(mrmp_payload)
        trajopt_payload = cls.load_saved_solution_payload(task, paths["trajopt_json"], expected_phase="trajopt")
        if trajopt_payload is not None:
            payloads.append(trajopt_payload)
        return payloads

    @staticmethod
    def write_json(output_path: Path, payload: dict[str, Any]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["output_path"] = str(output_path)

    @classmethod
    def independent_metrics(cls, task: SequentialMRMPTask) -> tuple[float, float]:
        per_agent = np.zeros(task.num_agents, dtype=float)
        current_positions = cls.initial_positions(task)
        for stage in task.stages:
            assignment = cls.assign_stage_queries(task, current_positions, stage)
            for agent_idx, query in enumerate(assignment.queries):
                per_agent[agent_idx] += float(np.linalg.norm(query.goal - query.start) / float(query.vlimit))
            current_positions = [np.asarray(query.goal, dtype=float) for query in assignment.queries]
        return float(per_agent.sum()), float(per_agent.max(initial=0.0))

    @classmethod
    def build_viewer_manifest(
        cls,
        task: SequentialMRMPTask,
        saved_solutions: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        env = cls.build_env(task)
        stgcs = cls.build_stgcs(task, env)
        independent_cost_sum, independent_makespan = cls.independent_metrics(task)
        first_stage_assignment = cls.assign_stage_queries(task, cls.initial_positions(task), task.stages[0])
        time_horizon = 0.0
        for solution in saved_solutions or []:
            value = solution.get("makespan")
            if value is not None and math.isfinite(float(value)):
                time_horizon = max(time_horizon, float(value))
        instance = {
            "instance_id": task.instance_id,
            "base_instance_id": f"base-{task.instance_id}",
            "problem_kind": "mrmp",
            "domain": "empty-square2d",
            "space_dim": int(task.space_dim),
            "spatial_seed": 0,
            "env_params": {
                "task_type": task.task_type,
                "assignment_strategy": cls.ASSIGNMENT_STRATEGY,
                "num_stages": len(task.stages),
                "stage_names": list(task.stage_names),
                "vlimit": float(task.vlimit),
                "tmax": float(task.tmax),
            },
            "traffic_family": "script-defined-sequential-demo",
            "traffic_tier": f"{task.num_agents}-robot-{len(task.stages)}-stage",
            "stgcs_num_vertices": int(stgcs.G.number_of_nodes()),
            "stgcs_num_edges": int(stgcs.G.number_of_edges()),
            "dimension": int(task.space_dim),
            "robot_radius": float(env.robot_radius),
            "bounds": {
                "min": np.asarray(env.lb, dtype=float).tolist(),
                "max": np.asarray(env.ub, dtype=float).tolist(),
            },
            "spatial_sets": [
                {"id": f"v{idx}", "vertices": np.asarray(poly, dtype=float).tolist()}
                for idx, poly in enumerate(task.cspace)
            ],
            "static_obstacles": [],
            "time_horizon": float(time_horizon),
            "time_range": [0.0, float(time_horizon)],
            "queries": [
                cls.query_to_viewer_json(query, agent_idx)
                for agent_idx, query in enumerate(first_stage_assignment.queries)
            ],
            "endpoint_markers": [],
            "hide_query_labels": True,
            "dynamic_obstacles": [],
            "num_agents": int(task.num_agents),
            "independent_cost_sum": float(independent_cost_sum),
            "independent_makespan": float(independent_makespan),
            "independent_runtime": 0.0,
            "num_conflicting_pairs": 0,
            "num_conflicting_agents": 0,
            "conflict_largest_component": 0,
            "conflict_density": 0.0,
        }
        return {

            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_kind": cls.DEMO_KEY,
            "instance_count": 1,
            "instances": [instance],
            "solutions": list(saved_solutions or []),
        }

    @classmethod
    def viewer_manifest_payload(
        cls,
        task: SequentialMRMPTask,
        paths: dict[str, Path],
        saved_solutions: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if saved_solutions is None:
            saved_solutions = cls.load_saved_solution_payloads(task, paths)
        return cls.build_viewer_manifest(task, saved_solutions=saved_solutions)

    @classmethod
    def run_pipeline(cls, task: SequentialMRMPTask, paths: dict[str, Path], *,
                     force_solve: bool = False, force_trajopt: bool = False) -> dict[str, Any]:
        mrmp_payload = None if force_solve else cls.load_saved_solution_payload(
            task, paths["mrmp_json"], expected_phase="mrmp")
        mrmp_recomputed = mrmp_payload is None
        if mrmp_recomputed:
            mrmp_payload = cls.solve_mrmp(task)
            cls.write_json(paths["mrmp_json"], mrmp_payload)
        else:
            print(f"Loaded MRMP solution: {paths['mrmp_json']}")

        trajopt_payload = None if (mrmp_recomputed or force_trajopt) else cls.load_saved_solution_payload(
            task, paths["trajopt_json"], expected_phase="trajopt")
        if trajopt_payload is None:
            trajopt_payload = cls.solve_trajopt_from_payload(task, mrmp_payload)
            cls.write_json(paths["trajopt_json"], trajopt_payload)
        else:
            print(f"Loaded trajopt solution: {paths['trajopt_json']}")

        return cls.viewer_manifest_payload(task, paths, saved_solutions=[mrmp_payload, trajopt_payload])

    @classmethod
    def query_target_configurations(cls, task: SequentialMRMPTask) -> list[tuple[str, list[np.ndarray]]]:
        configurations = [
            ("initial starts", [np.asarray(query.start, dtype=float) for query in task.stages[0].queries])
        ]
        for stage in task.stages:
            configurations.append((f"{stage.name} targets", [np.asarray(query.goal, dtype=float) for query in stage.queries]))
        return configurations

    @classmethod
    def plot_query_targets(
        cls,
        task: SequentialMRMPTask,
        output_path: Path | None = None,
        show: bool = True,
    ) -> None:
        import matplotlib.pyplot as plt

        env = cls.build_env(task)
        configurations = cls.query_target_configurations(task)
        num_panels = len(configurations)
        cols = min(3, num_panels)
        rows = int(math.ceil(num_panels / cols))
        fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=(4.0 * cols, 4.0 * rows))
        flat_axes = axes.ravel()

        for ax, (title, points) in zip(flat_axes, configurations):
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(float(env.lb[0]), float(env.ub[0]))
            ax.set_ylim(float(env.lb[1]), float(env.ub[1]))
            ax.grid(True, alpha=0.25)
            for poly in task.cspace:
                closed = np.vstack([poly, poly[0]])
                ax.plot(closed[:, 0], closed[:, 1], color="0.35", linewidth=1.0)
            points_array = np.asarray(points, dtype=float)
            ax.scatter(points_array[:, 0], points_array[:, 1], s=80, c="#1f77b4", edgecolors="black", zorder=3)
            for agent_idx, point in enumerate(points_array):
                ax.text(
                    float(point[0]),
                    float(point[1]),
                    str(agent_idx),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    weight="bold",
                    zorder=4,
                )

        for ax in flat_axes[num_panels:]:
            ax.axis("off")
        fig.suptitle(f"{task.name}: query target configurations", y=0.995)
        fig.tight_layout()
        if output_path is not None:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=200)
            print(f"Wrote query target plot: {target}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    @classmethod
    def planner_payload(cls, task: SequentialMRMPTask) -> dict[str, Any]:
        return {
            "ok": True,
            "default_budget": float(task.stage_timeout_secs),
            "planners": {
                "mrmp": [
                    {
                        "key": cls.mrmp_planner_key(task),
                        "name": cls.mrmp_planner_name(task),
                        "button_label": "MRMP",
                        "space_dims": [2],
                    },
                    {
                        "key": cls.trajopt_planner_key(task),
                        "name": cls.trajopt_planner_name(task),
                        "button_label": "Trajopt",
                        "space_dims": [2],
                    }
                ],
                "st_heuristic_ablation": [],
                "base": [],
            },
        }

    @classmethod
    def saved_solution_request(cls, task, payload, request):
        if request.get("instance_id") != task.instance_id:
            raise ValueError(f"Unknown instance_id: {request.get('instance_id')!r}")
        planner_key = str(request.get("planner_key", cls.mrmp_planner_key(task)))
        for solution in payload.get("solutions", []):
            if isinstance(solution, dict) and solution.get("planner_key") == planner_key:
                return solution
        raise ValueError(f"Unknown planner_key or missing saved solution: {planner_key!r}")

    @classmethod
    def serve_manifest(cls, task, payload, host, port):
        ViewerServer(payload, cls.planner_payload(task),
                     partial(cls.saved_solution_request, task, payload)).serve(host, port)
