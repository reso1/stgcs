from __future__ import annotations
from typing import List

import time
import numpy as np

from environment.instance import Instance
from environment.obstacle import ConcatDynamicSphere

from baselines.tprm.planner import TemporalPRM

from mrmp.graph import ShortestPathSolution


def sequential_planning(
    istc:Instance, tmax:float, vlimit:float, safe_radius:float,
    starts:List[np.ndarray], goals:List[np.ndarray], t0s:List[float],
    seed:int, timeout_secs:float, use_CSpace:bool
) -> List[ShortestPathSolution]:
    istc = istc.copy()
    solution = []
    
    for start, goal, t_start in zip(starts, goals, t0s):
        print(f"Plan for agent {len(solution)}")
        planner = TemporalPRM(istc, seed, vlimit, safe_radius)
        sol = planner.solve(start, goal, t_start, use_CSpace=use_CSpace, timeout_secs=timeout_secs/len(starts))
        if not sol.is_success:
            print(f"No solution found for agent {len(solution)}")
            return []
            
        solution.append(sol)
        istc.O_Dynamic.append(ConcatDynamicSphere.from_solution(sol, safe_radius=safe_radius))
    
    return solution
