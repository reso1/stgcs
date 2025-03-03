from __future__ import annotations
from typing import List, Dict, Tuple

import time
import numpy as np

from environment.instance import Instance
from mrmp.stgcs import STGCS, BASE_MAX_ROUNDED_PATHS, BASE_MAX_ROUNDING_TRIALS
from mrmp.ecd import ECDPair, reserve_solution
from mrmp.utils import timeit, make_hpolytope
from mrmp.graph import ShortestPathSolution


def randomized_prioritized_planning(
    istc:Instance, tmax:float, vlimit:float, safe_radius:float,
    starts:List[np.ndarray], goals:List[np.ndarray], t0s:List[float],
    seed:int, max_ordering_trials:int, timeout_secs:float, scaler_multiplier:float=1.0
) -> Tuple[List[ShortestPathSolution], int]:
    
    """ prioritized planner with randomized ordering """
    rng = np.random.RandomState(seed)
    num_agents = len(starts)
    ordering = [int(i) for i in range(num_agents)]

    visited = set()
    ts = time.perf_counter()
    while len(visited) < max_ordering_trials:
        rng.shuffle(ordering)
        ordering_tuple = tuple(ordering)
        if ordering_tuple in visited:
            continue
        
        print(f"-> PP: using total ordering: {ordering}. time elapsed={time.perf_counter() - ts}")
        visited.add(ordering_tuple)

        stgcs = STGCS.from_instance(
            istc, robot_radius=safe_radius, t0=0.0, tmax=tmax, vlimit=vlimit)
        
        sol, num_edges_stgcs = prioritized_planning(
            stgcs, ordering, safe_radius, starts, goals, t0s,
            timeout_secs = timeout_secs - (time.perf_counter() - ts),
            scaler_multiplier = scaler_multiplier
        )
        
        if sol != []:
            return [sol[_idx] for _idx in range(num_agents)], num_edges_stgcs
        
        if time.perf_counter() - ts > timeout_secs:
            break

    return [], -1


def sequential_planning(
    istc:Instance, tmax:float, vlimit:float, safe_radius:float,
    starts:List[np.ndarray], goals:List[np.ndarray], t0s:List[float],
    timeout_secs:float, scaler_multiplier:float=1.0
) -> Tuple[List[ShortestPathSolution], int]:
    
    ts = time.perf_counter()
    stgcs = STGCS.from_instance(
        istc, robot_radius=safe_radius, t0=0.0, tmax=tmax, vlimit=vlimit)
    
    ordering = [int(i) for i in range(len(starts))]
    sol, num_edges_stgcs = prioritized_planning(
        stgcs, ordering, safe_radius, starts, goals, t0s,
        timeout_secs = timeout_secs - (time.perf_counter() - ts),
        scaler_multiplier = scaler_multiplier
    )
    
    if sol != []:
        return [sol[_idx] for _idx in range(len(starts))], num_edges_stgcs

    return [], -1


def prioritized_planning(
    stgcs:STGCS, ordering:List[int], safe_radius:float,
    starts:List[np.ndarray], goals:List[np.ndarray], t0s:List[float],
    timeout_secs:float, scaler_multiplier:float
) -> Tuple[Dict[int, ShortestPathSolution], int]:
    ts = time.perf_counter()
    num_agents = len(starts)
    solution = {}
    for i in ordering:
        print(f"\tplanning for agent {i}, STGCS: |V|={stgcs.G.n_vertices}, |E|={stgcs.G.n_edges}")
        start, goal, t0 = starts[i], goals[i], t0s[i]
        scaler = np.clip(np.log(stgcs.G.n_edges), 1, 10) * scaler_multiplier
        sol = stgcs.solve(start, goal, t0, 
                            relaxation=True, 
                            max_rounded_paths = int(BASE_MAX_ROUNDED_PATHS * scaler),
                            max_rounding_trials = int(BASE_MAX_ROUNDING_TRIALS * scaler))
        if not sol.is_success:
            print(f"\t PP failed to find a solution for {i}")
            break
        if time.perf_counter() - ts > timeout_secs:
            print(f"\t PP timed out for {i}")
            break

        solution[i] = sol

        if len(solution) == num_agents:
            break
        
        stgcs = reserve_solution(stgcs, sol, safe_radius)

    if len(solution) == num_agents:
        return solution, stgcs.G.n_edges
    
    return [], -1

