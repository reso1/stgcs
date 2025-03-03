from __future__ import annotations
from typing import List, Tuple
from copy import deepcopy
from itertools import combinations, product

import time
import numpy as np
import networkx as nx

from pydrake.all import HPolyhedron
from environment.instance import Instance
from mrmp.stgcs import STGCS, BASE_MAX_ROUNDED_PATHS, BASE_MAX_ROUNDING_TRIALS
from mrmp.graph import ShortestPathSolution
from mrmp.interval import Interval
from mrmp.ecd import reserve as ecd_reserve, reconstruct_ECD_pairs
from mrmp.utils import timeit, make_hpolytope, draw_cuboid


class Node:
    
    def __init__(self, dg:nx.DiGraph) -> None:
        self.dg = dg
        self.sols: List[ShortestPathSolution] = []
        self._stgcs_num_edges: List[int] = [] # for debugging
    
    def __str__(self) -> str:
        return f"({', '.join([f'{u}<{v}' for u, v in self.dg.edges()])})"
        
    def update_plans(
        self, parent:Node, stgcs:STGCS, idx:int, safe_radius:float, tmin:float, tmax:float, 
        starts:List[np.ndarray], goals:List[np.ndarray], t0s:List[float], scaler_multiplier:float
    ) -> bool:
        num_agents = len(self.sols)
        replan_list = set([idx] + [j for j in range(num_agents) if nx.has_path(self.dg, idx, j)])
        for j in nx.topological_sort(self.dg.subgraph(replan_list)):
            high_priority_agents = list(self.dg.predecessors(j))
            replan = j == idx
            if not replan:
                for i in high_priority_agents: # check higher priority agents for conflicts
                    if collision_checking(parent.sols[i].trajectory, parent.sols[j].trajectory, safe_radius, tmin, tmax):
                        replan = True
                        break
            
            if replan:
                stgcs_reserved = stgcs.copy()
                for k in high_priority_agents:
                    ecd_pairs = reconstruct_ECD_pairs(stgcs_reserved, self.sols[k].trajectory)
                    stgcs_reserved = ecd_reserve(stgcs_reserved, ecd_pairs, safe_radius)
                scaler = np.clip(np.log(stgcs_reserved.G.n_edges), 1, 10) * scaler_multiplier
                sol = stgcs_reserved.solve(
                        starts[j], goals[j], t0s[j], 
                        relaxation=True,
                        max_rounded_paths = int(BASE_MAX_ROUNDED_PATHS * scaler), 
                        max_rounding_trials = int(BASE_MAX_ROUNDING_TRIALS * scaler),
                    )
                if not sol.is_success:
                    print(f"\t\u2713 Fails to update plan for {j} using STGCS: |V|={stgcs_reserved.G.n_vertices}, |E|={stgcs_reserved.G.n_edges}, |rounded_paths|={int(BASE_MAX_ROUNDED_PATHS * scaler)}")
                    return False
                self.sols[j] = sol
                self._stgcs_num_edges.append(stgcs_reserved.G.n_edges)
                print(f"\t\u2713 Succeeds to update plan for {j} using STGCS: |V|={stgcs_reserved.G.n_vertices}, |E|={stgcs_reserved.G.n_edges}, |rounded_paths|={int(BASE_MAX_ROUNDED_PATHS * scaler)}")

        return True

    def get_child(self, i:int, j:int) -> Node|None:
        G = self.dg.copy()
        G.add_edge(i, j)
        if nx.is_directed_acyclic_graph(G):
            child = Node(G)
            child.sols = [ShortestPathSolution(
                            is_success = sol.is_success, 
                            cost = sol.cost,
                            time = sol.time,
                            vertex_path = deepcopy(sol.vertex_path),
                            trajectory = deepcopy(sol.trajectory),
                            itvl = Interval(sol.itvl.start, sol.itvl.end),
                            dim = sol.dim) for sol in self.sols
                        ]
            return child

        return None

    def find_first_conflict(self, num_agents:int, safe_radius:float, tmin:float, tmax:float) -> Tuple[int|None, int|None]:
        for i, j in combinations(range(num_agents), 2):
            if (i, j) not in self.dg.edges and (j, i) not in self.dg.edges and \
                collision_checking(self.sols[i].trajectory, self.sols[j].trajectory, safe_radius, tmin, tmax):
                    return i, j

        return None, None


def PBS(
    istc:Instance, tmax:float, vlimit:float, safe_radius:float, 
    starts:List[np.ndarray], goals:List[np.ndarray], t0s:List[float],
    timeout_secs:float, scaler_multiplier:float=1.0,
) -> Tuple[List[ShortestPathSolution], float]:
    
    ts = time.perf_counter()
    T_MIN = 0.0
    num_agents = len(starts)
    
    stgcs = STGCS.from_instance(
        istc = istc, robot_radius = safe_radius,
        t0 = 0, tmax = tmax, vlimit = vlimit
    )

    G = nx.DiGraph()
    for i in range(num_agents):
        G.add_node(i)
    root = Node(G)
    scaler = np.clip(np.log(stgcs.G.n_edges), 1, 10) * scaler_multiplier

    print(f"\n-> PBS: initial STGCS: |V|={stgcs.G.n_vertices}, |E|={stgcs.G.n_edges}, |rounded_paths|={int(BASE_MAX_ROUNDED_PATHS * scaler)}")
    
    for start, goal, t0 in zip(starts, goals, t0s):
        sol = stgcs.solve(
            start, goal, t0, 
            relaxation=True,
            max_rounded_paths = int(BASE_MAX_ROUNDED_PATHS * scaler), 
            max_rounding_trials = int(BASE_MAX_ROUNDED_PATHS  * scaler),
        )
        if not sol.is_success:
            print("\n-> PBS:intial solution not found")
            return [], -1
        root.sols.append(sol)
        root._stgcs_num_edges.append(stgcs.G.n_edges)
    
    Stack = [root]
    while Stack != []:
        node = Stack.pop()
        ci, cj = node.find_first_conflict(num_agents, safe_radius, tmin=T_MIN, tmax=tmax)
        if ci is None:
            print(f"\nPBS: Successfully found a valid set of plans: {node}")
            return node.sols, np.mean(node._stgcs_num_edges)
        
        print(f"\n-> PBS: Current node = {str(node)}")
        print(f"-> PBS: Found conflict between {ci} and {cj}")

        for i, j in [(ci, cj), (cj, ci)]:
            child = node.get_child(i, j)
            if child:
                print(f"-> PBS: Update plan for child {child}.")
                if child.update_plans(node, stgcs, j, safe_radius, T_MIN, tmax, starts, goals, t0s, scaler_multiplier):
                    Stack.append(child)
    
        if time.perf_counter() - ts > timeout_secs:
            print(f"\n-> PBS: Timeout.")
            break
        
    print(f"\n-> PBS: Failed to find a path for the robot.")
    return [], -1


""" collision checking """


def collision_checking(
    pi_a:List[np.ndarray], pi_b:List[np.ndarray], 
    safe_radius:float, tmin:float, tmax:float
) -> bool:

    _a_t0 = [np.concatenate([pi_a[0][0:2], [tmin], pi_a[0][:3]])]
    _a_tf = [np.concatenate([pi_a[-1][-3:], pi_a[-1][-3:-1], [tmax]])]
    _b_t0 = [np.concatenate([pi_b[0][0:2], [tmin], pi_b[0][:3]])]
    _b_tf = [np.concatenate([pi_b[-1][-3:], pi_b[-1][-3:-1], [tmax]])]
    
    for xy_a, xy_b in product(_a_t0 + pi_a + _a_tf, _b_t0 + pi_b + _b_tf):
        xa, ya = xy_a[:3], xy_a[3:]
        xb, yb = xy_b[:3], xy_b[3:]
        if collision_checking_inner(xa, ya, xb, yb, safe_radius):
            return True

    return False


def collision_checking_inner(
    xa:np.ndarray, ya:np.ndarray, xb:np.ndarray, yb:np.ndarray, 
    safe_radius:float
) -> bool:
    """ return True if (xa, ya) and (xb, yb) are in collision (3d only) """

    itvl = Interval(xa[-1], ya[-1]).intersection(Interval(xb[-1], yb[-1]))
    if itvl is None:
        return False # no intersection in time
    
    P0, P1 = lerp(xa, ya, itvl.start), lerp(xa, ya, itvl.end)
    Q0, Q1 = lerp(xb, yb, itvl.start), lerp(xb, yb, itvl.end)
    # D(t) = ||(P0 - Q0) + t * (P1 - P0 - Q1 + Q0)||^2, t\in [0, 1]
    #      = || A + t*B||^2 >= r^2
    A = P0 - Q0
    B = P1 - P0 - Q1 + Q0
    # D'(t) = 2 * (A + t*B) * B = 0
    t = np.clip(- np.dot(A, B) / np.dot(B, B), 0, 1)
    min_dist = np.min([
        sqeuclidean(A + t * B), # at t \in [0, 1]
        sqeuclidean(A),         # at t = 0
        sqeuclidean(A + B)      # at t = 1
    ])

    return min_dist < safe_radius ** 2


def sqeuclidean(x:np.ndarray) -> float:
    return np.inner(x, x)


def lerp(x0:np.ndarray, xf:np.ndarray, t:float) -> np.ndarray:
    k = np.clip((t - x0[-1])/(xf[-1]-x0[-1]), 0, 1)
    return x0 + k * (xf - x0)
