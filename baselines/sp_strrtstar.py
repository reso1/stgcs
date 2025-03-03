from __future__ import annotations
from typing import List

import time
import numpy as np

from environment.instance import Instance
from environment.obstacle import ConcatDynamicSphere

from baselines.st_rrt_star.planner import STRRTStar, Options

from mrmp.graph import ShortestPathSolution


def sequential_planning(
    istc:Instance, tmax:float, vlimit:float, safe_radius:float,
    starts:List[np.ndarray], goals:List[np.ndarray], t0s:List[float],
    seed:int, timeout_secs:float, use_CSpace:bool
) -> List[ShortestPathSolution]:
    istc = istc.copy()
    solution = []
    P = Options.default()
    P.use_CSpace_sampling = use_CSpace
    P.max_iterations = int(1e9)
    P.max_runtime_in_secs = timeout_secs / len(starts)
    for start, goal, t_start in zip(starts, goals, t0s):
        print(f"Plan for agent {len(solution)}")
        planner = STRRTStar(istc, seed, vlimit)
        sol = planner.solve(start, goal, t0=t_start, t_max=tmax, P=P, robot_radius=safe_radius)
        if not sol.is_success:
            print(f"No solution found for agent {len(solution)}")
            return []
            
        solution.append(sol)
        istc.O_Dynamic.append(ConcatDynamicSphere.from_solution(sol, safe_radius=safe_radius))
    
    return solution
