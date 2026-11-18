"""Read, validate, and serialize saved ST, MRMP, and smoothed solutions."""
from __future__ import annotations

import math
from typing import Any, Dict, Sequence
from dataclasses import dataclass

import numpy as np

from experiments.manifests.mrmp import MRMPBenchmarkRecord
from stgcs_native.trajopt import GlobalTrajOptConfig, GlobalTrajOptResult
from stgcs_native.types import STTrajectory
from baselines.common import ShortestPathSolution
from experiments.manifests.st_planning import STHeuristicAblationRecord, STHeuristicAblationResultEntry

STORED_PWL_TIME_ORDER_TOLERANCE = 1e-6
STORED_PWL_POSITION_CONTINUITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ViewerSolutionRun:
    planner_key: str
    planner_name: str
    budget: float
    status: str
    entry: STHeuristicAblationResultEntry
    solution: STTrajectory | ShortestPathSolution | None


@dataclass(frozen=True)
class ViewerMRMPSolutionRun:
    planner_key: str
    planner_name: str
    budget: float
    status: str
    entry: Any
    solutions: Sequence[STTrajectory] | None


class SolutionIO:
    @staticmethod
    def validate_solution_payload(
        payload: dict[str, Any],
        record: MRMPBenchmarkRecord,
        expected_planner_key: str,
        budget: float,
    ) -> None:
        if not bool(payload.get("ok")):
            raise ValueError("Saved solution payload is not ok.")
        if str(payload.get("instance_id")) != record.instance_id:
            raise ValueError(
                f"Saved solution belongs to {payload.get('instance_id')!r}, expected {record.instance_id!r}."
            )
        if str(payload.get("planner_key")) != expected_planner_key:
            raise ValueError(
                f"Saved solution planner is {payload.get('planner_key')!r}, expected {expected_planner_key!r}."
            )
        if abs(float(payload.get("budget")) - float(budget)) > 1e-9:
            raise ValueError(f"Saved solution budget does not match {float(budget):g}.")
        if not bool(payload.get("is_success")):
            completed = int(payload.get("num_completed_agents", 0))
            raise ValueError(
                f"Saved MRMP solution is not successful: completed {completed}/{int(record.num_agents)} agents."
            )
        trajectories = payload.get("trajectories")
        if not isinstance(trajectories, list) or len(trajectories) != int(record.num_agents):
            count = 0 if not isinstance(trajectories, list) else len(trajectories)
            raise ValueError(f"Saved MRMP solution has {count} trajectories, expected {int(record.num_agents)}.")

    @staticmethod
    def trajectory_from_viewer_json(payload: dict[str, Any]) -> STTrajectory:
        dim = int(payload.get("dimension", 0))
        segments = payload.get("segments")
        if dim <= 0 or not isinstance(segments, list) or not segments:
            raise ValueError("Stored PWL trajectory must contain drawable segments.")

        vertex_path: list[str] = []
        points: list[np.ndarray] = []
        expected_spatial_shape = (dim,)
        last_time = -math.inf
        last_goal: np.ndarray | None = None
        for idx, segment in enumerate(segments):
            if not isinstance(segment, dict):
                raise ValueError("Stored PWL trajectory segment must be a JSON object.")
            start = np.asarray(segment["start"], dtype=float).reshape(-1)
            goal = np.asarray(segment["goal"], dtype=float).reshape(-1)
            if start.shape != expected_spatial_shape or goal.shape != expected_spatial_shape:
                raise ValueError(
                    f"Stored PWL trajectory segment has shape {start.shape}->{goal.shape}, "
                    f"expected {expected_spatial_shape}."
                )
            t_start = float(segment["t_start"])
            t_end = float(segment["t_end"])
            if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
                raise ValueError("Stored PWL trajectory contains a non-finite spatial coordinate.")
            if (
                not math.isfinite(t_start)
                or not math.isfinite(t_end)
                or t_end + STORED_PWL_TIME_ORDER_TOLERANCE < t_start
            ):
                raise ValueError("Stored PWL trajectory contains an invalid time interval.")
            if t_end < t_start:
                t_end = t_start
            if t_start + STORED_PWL_TIME_ORDER_TOLERANCE < last_time:
                raise ValueError("Stored PWL trajectory segments are not time ordered.")
            if t_start < last_time:
                if last_goal is None or not np.allclose(
                    start,
                    last_goal,
                    rtol=0.0,
                    atol=STORED_PWL_POSITION_CONTINUITY_TOLERANCE,
                ):
                    raise ValueError("Stored PWL trajectory has a tiny time overlap but is spatially discontinuous.")
                t_start = last_time
            elif t_start > last_time and last_goal is not None:
                # A saved wait can round its two copies of the same position one float apart.
                if np.array_equal(np.nextafter(start, last_goal), last_goal):
                    start = last_goal
            vertex_path.append(f"stored-pwl-segment-{idx}")
            points.append(np.hstack([start, t_start, goal, t_end]))
            last_time = t_end
            last_goal = goal
        return STTrajectory(vertex_path, points, dim=dim)

    @staticmethod
    def pwl_trajectories_from_solution_payload(
        payload: dict[str, Any],
        expected_num_agents: int,
    ) -> list[STTrajectory]:
        trajectory_payloads = payload.get("trajectories")
        if not isinstance(trajectory_payloads, list) or not trajectory_payloads:
            raise ValueError("Stored PWL solution payload does not contain trajectories.")

        trajectories = [SolutionIO.trajectory_from_viewer_json(trajectory_payload) for trajectory_payload in trajectory_payloads]
        if len(trajectories) != int(expected_num_agents):
            raise ValueError(
                f"Stored PWL trajectory count {len(trajectories)} does not match "
                f"num_agents={int(expected_num_agents)}."
            )
        return trajectories

    @staticmethod
    def finite_float_or_none(value: float | int | None) -> float | None:
        if value is None:
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return numeric

    @staticmethod
    def _trajectory_points(solution: STTrajectory | ShortestPathSolution | None) -> tuple[int, Sequence[Any]]:
        if solution is None:
            return 0, []
        if isinstance(solution, ShortestPathSolution):
            return int(solution.dim), solution.trajectory
        return int(solution.dim), solution.points

    @classmethod
    def trajectory_to_viewer_json(
        cls,
        solution: STTrajectory | ShortestPathSolution | None,
    ) -> Dict[str, Any]:
        dim, raw_points = cls._trajectory_points(solution)
        if dim <= 1 or not raw_points:
            return {
                "dimension": max(dim - 1, 0),
                "segments": [],
                "path": [],
                "time_range": [0.0, 0.0],
            }

        segments: list[Dict[str, Any]] = []
        path: list[list[float]] = []
        min_time = math.inf
        max_time = -math.inf
        for point in raw_points:
            values = np.asarray(point, dtype=float).reshape(-1)
            if values.size != 2 * dim:
                raise ValueError(
                    f"Solution segment has {values.size} values, expected {2 * dim} for dim={dim}."
                )
            start = values[: dim - 1]
            goal = values[dim: 2 * dim - 1]
            t_start = float(values[dim - 1])
            t_end = float(values[-1])
            if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
                raise ValueError("Solution contains a non-finite spatial coordinate.")
            if not math.isfinite(t_start) or not math.isfinite(t_end):
                raise ValueError("Solution contains a non-finite time coordinate.")
            segment = {
                "start": start.tolist(),
                "goal": goal.tolist(),
                "t_start": t_start,
                "t_end": t_end,
            }
            segments.append(segment)
            if not path:
                path.append(segment["start"])
            path.append(segment["goal"])
            min_time = min(min_time, t_start, t_end)
            max_time = max(max_time, t_start, t_end)

        if not math.isfinite(min_time) or not math.isfinite(max_time):
            min_time = max_time = 0.0
        return {
            "dimension": dim - 1,
            "segments": segments,
            "path": path,
            "time_range": [float(min_time), float(max_time)],
        }

    @classmethod
    def run_to_viewer_json(
        cls,
        record: STHeuristicAblationRecord,
        run: ViewerSolutionRun,
    ) -> Dict[str, Any]:
        return {
            "ok": True,
            "instance_id": record.instance_id,
            "planner_key": run.planner_key,
            "planner_name": run.planner_name,
            "budget": cls.finite_float_or_none(run.budget),
            "status": run.status,
            "is_success": bool(run.entry.is_success),
            "runtime": cls.finite_float_or_none(run.entry.runtime),
            "cost": cls.finite_float_or_none(run.entry.cost),
            "num_expanded_nodes": int(run.entry.num_expanded_nodes),
            "num_generated_nodes": int(run.entry.num_generated_nodes),
            "trajectory": cls.trajectory_to_viewer_json(run.solution if run.entry.is_success else None),
        }

    @classmethod
    def mrmp_run_to_viewer_json(
        cls,
        record: MRMPBenchmarkRecord,
        run: ViewerMRMPSolutionRun,
    ) -> Dict[str, Any]:
        trajectories = [
            {
                **cls.trajectory_to_viewer_json(solution),
                "agent_index": agent_idx,
            }
            for agent_idx, solution in enumerate([] if run.solutions is None else run.solutions)
        ]
        return {
            "ok": True,
            "instance_id": record.instance_id,
            "planner_key": run.planner_key,
            "planner_name": run.planner_name,
            "budget": cls.finite_float_or_none(run.budget),
            "status": run.status,
            "is_success": bool(run.entry.is_success),
            "runtime": cls.finite_float_or_none(run.entry.runtime),
            "cost": cls.finite_float_or_none(run.entry.sum_of_costs),
            "makespan": cls.finite_float_or_none(run.entry.makespan),
            "num_completed_agents": int(run.entry.num_completed_agents),
            "pbs_calls": int(run.entry.pbs_calls),
            "pbs_runtime": cls.finite_float_or_none(run.entry.pbs_runtime),
            "pbs_popped_nodes": int(run.entry.pbs_popped_nodes),
            "pbs_generated_children": int(run.entry.pbs_generated_children),
            "pbs_update_calls": int(run.entry.pbs_update_calls),
            "num_expanded_nodes": int(run.entry.pbs_popped_nodes),
            "num_generated_nodes": int(run.entry.pbs_generated_children),
            "trajectories": trajectories,
        }

    @classmethod
    def refined_solution_payload(
        cls,
        record: MRMPBenchmarkRecord,
        mrmp_payload: dict[str, Any],
        result: GlobalTrajOptResult,
        config: GlobalTrajOptConfig,
        *,
        trajopt_planner_key: str,
        trajopt_planner_name: str,
        window_span: float,
        stride: float,
        continuous_pairwise_ok: bool,
    ) -> dict[str, Any]:
        durations = [float(trajectory.duration) for trajectory in result.trajectories]
        payload = dict(mrmp_payload)
        payload.update(
            {
                "planner_key": trajopt_planner_key,
                "planner_name": trajopt_planner_name,
                "runtime": cls.finite_float_or_none(
                    float(mrmp_payload["runtime"]) + float(result.runtime)
                ),
                "cost": cls.finite_float_or_none(sum(durations)),
                "makespan": cls.finite_float_or_none(max(durations, default=math.inf)),
                "trajectories": [
                    {
                        **cls.trajectory_to_viewer_json(trajectory),
                        "agent_index": agent_idx,
                    }
                    for agent_idx, trajectory in enumerate(result.trajectories)
                ],
                "source_planner_key": mrmp_payload["planner_key"],
                "source_runtime": mrmp_payload["runtime"],
                "global_trajopt": {
                    "backend": "stgcs_cplusplus",
                    "qp_solver": "MOSEK",
                    "solver_threads": 1,
                    "mode": "sliding_window",
                    "sample_dt": float(config.sample_dt),
                    "window_span": float(window_span),
                    "stride": float(stride),
                    "min_sample_dt": float(config.min_sample_dt),
                    "velocity_gradient_weight": float(config.velocity_gradient_weight),
                    "displacement_weight": float(config.displacement_weight),
                    "displacement_jitter_weight": float(config.displacement_jitter_weight),
                    "clearance_margin": float(config.clearance_margin),
                    "fix_terminal_states": bool(config.fix_terminal_states),
                    "include_trajectory_knot_times": bool(config.include_trajectory_knot_times),
                    "solver_eps_abs": float(config.solver_eps_abs),
                    "solver_eps_rel": float(config.solver_eps_rel),
                    "constraint_numerical_allowance": .5*float(config.solver_eps_abs),
                    "include_interval_collision_constraints": bool(config.include_interval_collision_constraints),
                    "runtime": cls.finite_float_or_none(result.runtime),
                    "num_sample_times": int(result.times.size),
                    "sampled_pairwise_collision_free": bool(result.sampled_pairwise_collision_free),
                    "continuous_pairwise_collision_free": continuous_pairwise_ok,
                    "pairwise_collision_geometry": "disk",
                    "min_continuous_pairwise_distance": float(result.min_continuous_pairwise_distance),
                    "sampled_environment_collision_free": result.sampled_environment_collision_free,
                    "continuous_environment_collision_free": result.continuous_environment_collision_free,
                    "min_pairwise_distance": float(result.min_pairwise_distance),
                    "max_velocity_component": float(result.max_velocity_component),
                    "velocity_limit": float(record.queries[0].vlimit) if record.queries else None,
                    "solver_message": result.solver_message,
                },
            }
        )
        return payload

