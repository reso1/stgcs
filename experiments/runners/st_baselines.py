"""Adapters for the independent OMPL and Zeta-SIPP comparison algorithms."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

from baselines.ompl_strrt_star import OfficialOMPLSTRRTStar, OfficialOMPLSTRRTStarOptions
from baselines.common import ShortestPathSolution
from baselines.zeta_sipp import ZetaStarSIPPPlanner
from experiments.instance import Instance
from experiments.manifests.st_planning import STHeuristicAblationRecord
from experiments.planners.st_planning import STPerformanceComparisonConfig as Config
from experiments.native import STGCSPlannerClient


class BaselineWorker:
    @staticmethod
    def validate_zeta_solution(env, query, horizon, solution) -> None:
        """Independent continuous checks, without Zeta's grid or interval routines."""
        if not solution.is_success:
            return
        tolerance = 1e-7
        if (not np.isfinite(solution.cost) or solution.cost < 0
                or not np.isfinite(solution.time) or solution.time < 0):
            raise ValueError("Invalid Zeta solution cost or runtime")
        checker = OfficialOMPLSTRRTStar._dynamic_obstacle_collision_checker(
            env, query.t_start, horizon)
        previous, previous_time = np.asarray(query.start), query.t_start
        if not solution.trajectory:
            raise ValueError("Successful Zeta solution has no trajectory")
        for segment in solution.trajectory:
            segment = np.asarray(segment, dtype=float)
            if segment.shape != (2 * (env.dim + 1),) or not np.all(np.isfinite(segment)):
                raise ValueError("Invalid Zeta trajectory segment")
            a, ta = segment[:env.dim], segment[env.dim]
            b, tb = segment[env.dim + 1:-1], segment[-1]
            if (np.max(np.abs(a - previous)) > tolerance or abs(ta - previous_time) > tolerance
                    or tb < ta or tb > horizon + tolerance
                    or np.max(np.abs(b - a)) > query.vlimit * (tb - ta) + tolerance):
                raise ValueError("Zeta trajectory violates continuity, speed, or horizon")
            if (np.any(np.minimum(a, b) < env.lb - tolerance)
                    or np.any(np.maximum(a, b) > env.ub + tolerance)):
                raise ValueError("Zeta trajectory leaves the domain")
            if (not env.O_Static or str(env.name).lower().startswith("maze")):
                if not env.is_segment_in_CSpace(a, b):
                    raise ValueError("Zeta trajectory leaves the free-space union")
            if any(obstacle.is_colliding_lineseg(a, b, env.robot_radius) for obstacle in env.O_Static):
                raise ValueError("Zeta trajectory has a static collision")
            if checker is not None and checker(a, b, ta, tb):
                raise ValueError("Zeta trajectory has a continuous dynamic collision")
            previous, previous_time = b, tb
        if (np.max(np.abs(previous - query.goal)) > tolerance
                or abs(solution.cost - (previous_time - query.t_start)) > tolerance):
            raise ValueError("Zeta trajectory has an invalid goal or cost")
        if query.is_stay and checker is not None and checker(
                previous, previous, previous_time, horizon):
            raise ValueError("Zeta trajectory cannot remain at the goal")

    @classmethod
    def run(cls, record, base_root: Path, method: str, budget: float,
            solution_path: Path | None = None) -> dict:
        if method == "strrt_matched" and budget == 0:
            solution = ShortestPathSolution(False, float("inf"), 0.0, [], [])
            return {Config.ST_RRT_MATCHED_PLANNER: asdict(
                Config.entry_from_shortest_path_solution(solution, 0.0, 0.0))}
        instance = STGCSPlannerClient.base_instance(record, base_root)
        env = instance.env.copy()
        env.O_Dynamic = Instance.dynamic_obstacles_from_specs(record.dynamic_obstacles)
        query = record.query.to_query()
        seed = Config.seed_for_record(record, 0)
        started = time.perf_counter()
        if method in ("strrt", "strrt_matched"):
            planner = OfficialOMPLSTRRTStar(env, seed, query.vlimit)
            snapshots = planner.solve_with_snapshots(
                query.start, query.goal, t0=query.t_start,
                options=OfficialOMPLSTRRTStarOptions(max_runtime_in_secs=budget,
                    return_first_valid=False, time_upper_bound=record.tmax, stay_at_goal=query.is_stay))
            solutions = {Config.ST_RRT_FIRST_PLANNER: snapshots.first_solution,
                         Config.ST_RRT_FINAL_PLANNER: snapshots.final_solution}
            if method == "strrt_matched":
                solutions = {Config.ST_RRT_MATCHED_PLANNER: snapshots.final_solution}
        elif method in ("zeta", "zeta2r"):
            multiplier = 1 if method == "zeta" else 2
            planner = ZetaStarSIPPPlanner(env, seed, query.vlimit,
                                         cell_size=multiplier * env.robot_radius,
                                         runtime_limit_secs=budget)
            solution = planner.solve(query.start, query.goal, t_start=query.t_start,
                                     t_max=record.tmax, is_stay=query.is_stay)
            validation_started = time.perf_counter()
            validation_error = None
            try:
                cls.validate_zeta_solution(env, query, record.tmax, solution)
            except ValueError as error:
                validation_error = str(error)
            if solution_path is not None:
                solution_path.write_text(json.dumps(dict(
                    instance_id=record.instance_id, method=method, budget=budget,
                    cell_size=planner.cell_size, robot_half_extent=env.robot_radius,
                    grid_shape=(planner.last_projection.shape if planner.last_projection else None),
                    projection_runtime=(planner.last_projection.build_runtime
                                        if planner.last_projection else None),
                    success=solution.is_success, cost=solution.cost, runtime=solution.time,
                    status=planner.last_status,
                    validation=("invalid" if validation_error else
                                "continuous_square" if solution.is_success else "not_applicable"),
                    validation_error=validation_error,
                    validation_runtime=time.perf_counter() - validation_started,
                    trajectory=[np.asarray(s).tolist() for s in solution.trajectory],
                ), indent=2))
            if validation_error:
                raise ValueError(validation_error)
            solutions = {Config.ZETA_SIPP_PLANNER if multiplier == 1 else Config.ZETA_SIPP_2R_PLANNER: solution}
        else:
            raise ValueError(method)
        elapsed = time.perf_counter() - started
        return {name: asdict(Config.entry_from_shortest_path_solution(solution, elapsed, budget))
                for name, solution in solutions.items()}

    @classmethod
    def main(cls):
        record_path, base_root, method, budget = sys.argv[1:]
        record = STHeuristicAblationRecord.from_dict(json.loads(Path(record_path).read_text()))
        artifact = Path(record_path).with_suffix(f".{method}.solution.json")
        print(json.dumps(cls.run(record, Path(base_root), method, float(budget), artifact)), flush=True)


if __name__ == "__main__":
    BaselineWorker.main()
