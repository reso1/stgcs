"""Python I/O for the stgcs_trajopt executable."""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np

from .client import NativeClient
from .types import STTrajectory, Environment, Polyhedron, Sphere


@dataclass(frozen=True)
class GlobalTrajOptConfig:
    sample_dt: float = 0.02
    min_sample_dt: float = 1e-4
    velocity_gradient_weight: float = 0.05
    displacement_weight: float = 0.2
    displacement_jitter_weight: float = 1e-4
    clearance_margin: float = 1e-6
    spatial_containment_tolerance: float = 0.0
    spatial_containment_soft_margin: float = 0.0
    constraint_slack_penalty: float = 0.0
    constraint_slack_limit: float = math.inf
    include_interval_collision_constraints: bool = True
    fix_terminal_states: bool = True
    include_trajectory_knot_times: bool = True
    solver_max_iter: int = 100000
    solver_eps_abs: float = 1e-6
    solver_eps_rel: float = 1e-6
    solver_print_to_console: bool = False
    progress: bool = False
    round_time_decimals: int = 9

    @staticmethod
    def scaled_time(factor: float, robot_radius: float, velocity_limit: float) -> float:
        """Convert a duration in robot-radius/velocity units to seconds."""
        return float(factor) * (float(robot_radius) / float(velocity_limit))


@dataclass(frozen=True)
class GlobalTrajOptResult:
    trajectories: list[STTrajectory]
    times: np.ndarray
    original_positions: np.ndarray
    optimized_positions: np.ndarray
    displacements: np.ndarray
    runtime: float
    sampled_pairwise_collision_free: bool
    sampled_environment_collision_free: bool | None
    continuous_environment_collision_free: bool | None
    min_pairwise_distance: float
    max_velocity_component: float
    solver_message: str
    continuous_pairwise_collision_free: bool = False
    min_continuous_pairwise_distance: float = math.inf


class GlobalTrajectoryOptimizer:
    @classmethod
    def _request(cls, mode, trajectories, robot_radius=0.0, velocity_limit=1.0,
                 config=None, cells=(), env: Environment | None = None, extra=()):
        if not trajectories or any(path.size == 0 for path in trajectories):
            raise ValueError("Expected nonempty trajectories.")
        config = config or GlobalTrajOptConfig()
        dimension = int(trajectories[0].dim - 1)
        if any(int(path.dim - 1) != dimension for path in trajectories):
            raise ValueError("Trajectory dimension mismatch.")
        if any(cell.A.shape[1] != dimension + 1 for _, cell in cells):
            raise ValueError("ST-GCS dimension mismatch.")
        if env is not None:
            if int(env.dim) != dimension or not math.isclose(float(env.robot_radius), float(robot_radius), rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("Environment dimension or robot radius mismatch.")
        names = [name for name, _ in cells]
        indices = {name: index for index, name in enumerate(names)}
        for path in trajectories:
            for name in path.vertex_path:
                if name not in indices:
                    indices[name] = len(names)
                    names.append(name)
        words = [robot_radius, velocity_limit]
        for field in fields(config):
            value = getattr(config, field.name)
            words.append(int(value) if isinstance(value, bool) else value)
        words.extend([dimension, len(trajectories)])
        for path in trajectories:
            words.append(path.size)
            for name, point in zip(path.vertex_path, path.points):
                words.extend([indices[name], *np.asarray(point).ravel()])
        words.append(len(cells))
        for _, cell in cells:
            words.extend([len(cell.b), *np.asarray(cell.A).ravel(), *cell.b])
        words.append(int(env is not None))
        if env is not None:
            obstacles = env.obstacles
            words.extend([*env.lower, *env.upper, len(obstacles)])
            for obstacle in obstacles:
                if isinstance(obstacle, Sphere):
                    words.extend([1, *obstacle.center, obstacle.radius])
                elif isinstance(obstacle, Polyhedron):
                    words.extend([0, len(obstacle.b), *np.asarray(obstacle.A).ravel(), *obstacle.b])
                else:
                    raise TypeError(f"Unsupported static obstacle: {type(obstacle).__name__}")
            regions = env.regions
            words.append(len(regions))
            for region in regions:
                words.extend([len(region.b), *np.asarray(region.A).ravel(), *region.b])
        words.extend(extra)
        payload = NativeClient.request("stgcs_trajopt", mode, words, progress=config.progress)
        if "trajectories" in payload:
            payload["trajectories"] = [
                STTrajectory([names[index] for index in path["vertices"]],
                             [np.asarray(point) for point in path["points"]], dimension)
                for path in payload["trajectories"]]
        return payload

    @staticmethod
    def _result(payload):
        original = np.asarray(payload["original_positions"])
        optimized = np.asarray(payload["optimized_positions"])
        return GlobalTrajOptResult(
            trajectories=payload["trajectories"], times=np.asarray(payload["times"]),
            original_positions=original, optimized_positions=optimized,
            displacements=optimized-original, runtime=float(payload["runtime"]),
            sampled_pairwise_collision_free=payload["sampled_pairwise_collision_free"],
            sampled_environment_collision_free=payload["sampled_environment_collision_free"],
            continuous_environment_collision_free=payload["continuous_environment_collision_free"],
            min_pairwise_distance=(math.inf if payload["min_pairwise_distance"] is None else payload["min_pairwise_distance"]),
            max_velocity_component=payload["max_velocity_component"], solver_message=payload["solver_message"],
            continuous_pairwise_collision_free=payload["continuous_pairwise_collision_free"],
            min_continuous_pairwise_distance=(math.inf if payload["min_continuous_pairwise_distance"] is None
                                              else payload["min_continuous_pairwise_distance"]))

    @classmethod
    def optimize(cls, trajectories, robot_radius, velocity_limit, config=None, *, cells=(), env=None):
        return cls._result(cls._request("optimize", trajectories, robot_radius, velocity_limit,
                                       config, cells, env, extra=(0, 0)))

    @classmethod
    def optimize_sliding_window(cls, trajectories, robot_radius, velocity_limit, config=None,
                                *, window_span, stride, cells=(), env=None):
        if not 0 < stride < window_span or not math.isfinite(window_span):
            raise ValueError("Require finite 0 < stride < window_span.")
        return cls._result(cls._request("optimize", trajectories, robot_radius, velocity_limit,
                                       config, cells, env, extra=(window_span, stride)))

    @classmethod
    def smooth(cls, trajectories, robot_radius, velocity_limit, config=None,
               *, window_span, stride, goal_stays, collision_tolerance=1e-6,
               cells=(), env=None):
        """Prepare PWL paths, optimize sliding windows, and require disk clearance."""
        config = config or GlobalTrajOptConfig()
        reference = cls.reference_trajectories(
            trajectories, robot_radius, velocity_limit, config, cells=cells, env=env)
        result = cls.optimize_sliding_window(
            reference, robot_radius, velocity_limit, config,
            window_span=window_span, stride=stride, cells=cells, env=env)
        if not result.sampled_pairwise_collision_free:
            raise RuntimeError("Global trajectory optimization produced a sampled pairwise collision.")
        if not cls.disk_pairwise_collision_free(
                result.trajectories, robot_radius, goal_stays, tolerance=collision_tolerance):
            raise RuntimeError("Global trajectory optimization produced a continuous disk collision.")
        return result

    @classmethod
    def inspect(cls, trajectories, config=None, *, times=()):
        return cls._result(cls._request("inspect", trajectories, config=config, extra=(len(times), *times)))

    @classmethod
    def reference_trajectories(cls, trajectories, robot_radius, velocity_limit,
                              config=None, *, cells=(), env=None):
        return cls._request("reference", trajectories, robot_radius, velocity_limit,
                            config, cells, env)["trajectories"]

    @classmethod
    def pairwise_collision_free(cls, trajectories, robot_radius, goal_stays, tolerance=1e-4):
        if len(goal_stays) != len(trajectories):
            raise ValueError("Goal-stay count mismatch.")
        return cls._request("pairwise", trajectories, robot_radius,
                            extra=(*map(int, goal_stays), tolerance))["ok"]

    @classmethod
    def disk_pairwise_collision_free(cls, trajectories, robot_radius, goal_stays, tolerance=1e-6):
        if len(goal_stays) != len(trajectories):
            raise ValueError("Goal-stay count mismatch.")
        return cls._request("disk_pairwise", trajectories, robot_radius,
                            extra=(*map(int, goal_stays), tolerance))["ok"]

    @classmethod
    def environment_collision_free(cls, env, trajectories, tolerance=1e-9, *, cells=()):
        return cls._request("environment", trajectories, env.robot_radius,
                            cells=cells, env=env, extra=(tolerance,))["ok"]

    @classmethod
    def cspace_collision_free(cls, cells, trajectories, tolerance=1e-9):
        return cls._request("environment", trajectories, cells=cells, extra=(tolerance,))["ok"]
