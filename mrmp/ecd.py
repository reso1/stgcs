from __future__ import annotations
from typing import List, Tuple, Dict, Set
from itertools import combinations, product
from collections import defaultdict
from dataclasses import dataclass

from copy import deepcopy
import numpy as np
import networkx as nx

from pydrake.all import HPolyhedron

from mrmp.stgcs import STGCS
from mrmp.graph import ShortestPathSolution, Edge, Graph, Vertex
from mrmp.interval import Interval
from mrmp.utils import (
    timeit, crop_time_extruded, get_hpoly_bounds, squash_multi_points, find_space_time_intersecting_pts,
    draw_2d_space_set, draw_3d_space_time_set, draw_cuboid
)


""" reserve a shortest path solution class in STGCS, from which the solution is produced 
    (use it for prioritized planning) """


@dataclass
class ECDPair:
    v_name: str
    xp: np.ndarray
    xq: np.ndarray
    reserve_xp_to_tmin: bool
    reserve_xq_to_tmax: bool


# @timeit
def reserve(
    stgcs:STGCS, ecd_pairs:List[ECDPair], safe_radius:float, t_padding=1e-3,
) -> STGCS:
    """ reserve a shortest path solution class in STGCS, from which the solution is produced """
    new = stgcs.copy()
    adj_graph = nx.Graph()
    for u, V in stgcs.G._adjacency_list.items():
        adj_graph.add_edges_from([(u, v) for v in V])

    ##### DEBUG Code #####
    # import matplotlib.pyplot as plt
    # c = ['r', 'c', 'b', 'y', 'g', 'm', 'k']
    # ax = plt.gca()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ######################
    
    split_map = defaultdict(list)
    vertex_path = set([p.v_name for p in ecd_pairs])
    split_map.update({v:[v] for v in stgcs.G.vertex_names if v not in vertex_path})
    
    for item in ecd_pairs:
        v_name, xp, xq = item.v_name, item.xp, item.xq
        v = new.remove_vertex(v_name) # remove old vertex

        ##### DEBUG Code #####
        # draw_cuboid(ax, xp, xq, safe_radius * 0.6)
        # idx = 0
        ######################

        res = reserve_cuboid_2d(
            v, xp, xq, safe_radius, 
            rsv_to_t0 = item.reserve_xp_to_tmin, 
            rsv_to_tf = item.reserve_xq_to_tmax,
            t_padding = t_padding
        )
        
        # add new vertices
        for hpoly, itvl in res:
            new_v = new.add_vertex(hpoly, itvl)
            split_map[v_name].append(new_v.name)
            
            ##### DEBUG Code #####
            # draw_3d_space_time_set(hpoly, ax, fc=c[idx%len(c)])
            # ax.axis('off')
            # ax.set_aspect('equal')
            # idx += 1
            ######################
    
    """ updating edges """
    edge_key = lambda u_name, v_name: (u_name, v_name) if u_name < v_name else (v_name, u_name)
    edge_checked = set([edge_key(e.u, e.v) for e in new.G.edges.values()])

    # build new edges between previously neighboring vertices
    for old_u, old_v in adj_graph.edges:
        for u_name, v_name in product(split_map[old_u], split_map[old_v]):
            key = edge_key(u_name, v_name)
            if key not in edge_checked:
                edge_checked.add(key)
                new.try_build_edge(u_name, v_name)

    # build edges within the split map
    for v_name, new_verts in split_map.items():
        for u_name, v_name in combinations(new_verts, 2):
            key = edge_key(u_name, v_name)
            if key not in edge_checked:
                edge_checked.add(key)
                new.try_build_edge(u_name, v_name)

    # print(f"\nCreated |V|={new.G.n_vertices}, |E|={new.G.n_edges}")

    return new


def reserve_solution(
    stgcs:STGCS, sol:ShortestPathSolution, safe_radius:float, t_padding=1e-3
) -> STGCS:
    ecd_pairs = []
    for i, v_name in enumerate(sol.vertex_path):
        xp, xq = sol.trajectory[i][:3], sol.trajectory[i][-3:]
        rsv_to_t0 = i == 0
        rsv_to_tf = i == len(sol.vertex_path) - 1
        ecd_pairs.append(ECDPair(v_name, xp, xq, rsv_to_t0, rsv_to_tf))

    return reserve(stgcs, ecd_pairs, safe_radius, t_padding)

""" core functions """


def reserve_cuboid_2d(
    v:Vertex, xp:np.ndarray, xq:np.ndarray, halfsize:float, t_padding,
    rsv_to_t0=False, rsv_to_tf=False
) -> List[Tuple[HPolyhedron, Interval]]:
    
    ret: List[Tuple[HPolyhedron, Interval]] = []
    
    # project vertex set from (dim * num_pts) to dim
    hpoly = squash_multi_points(v.convex_set.set, dim=xp.shape[-1])
    t0, tf = v.itvl.start, v.itvl.end
    tp, tq = xp[-1], xq[-1]

    top = crop_time_extruded(hpoly, t0, tp)
    if rsv_to_t0:
        _x0 = np.hstack([xp[:-1], [t0]])
        top_sliced = slice_cuboid_2d(top, _x0, xp, halfsize)
        ret.extend(top_sliced)
    elif t_padding:
        _x0 = np.hstack([xp[:-1], [tp - t_padding]])
        top_sliced = slice_cuboid_2d(top, _x0, xp, halfsize)
        ret.extend(top_sliced)
    elif top is not None and not top.IsEmpty():
        ret.append((top, Interval(t0, tp)))
        
    bot = crop_time_extruded(hpoly, tq, tf)
    if rsv_to_tf:
        _xt = np.hstack([xq[:-1], [tf]])
        bot_sliced = slice_cuboid_2d(bot, xq, _xt, halfsize)
        ret.extend(bot_sliced)
    elif t_padding:
        _xt = np.hstack([xq[:-1], [tq + t_padding]])
        bot_sliced = slice_cuboid_2d(bot, xq, _xt, halfsize)
        ret.extend(bot_sliced)
    elif bot is not None and not bot.IsEmpty():
        ret.append((bot, Interval(tq, tf)))
    
    mid = crop_time_extruded(hpoly, tp, tq)
    ret.extend(slice_cuboid_2d(mid, xp, xq, halfsize))
    
    return ret


def slice_cuboid_2d(hpoly:HPolyhedron|None, xp:np.ndarray, xq:np.ndarray, halfsize:float) -> List[HPolyhedron]:
    ret = []
    for out_halfspace, in_halfspace in cuboid_side_halfspace_kd(xp, xq, halfsize):
        if hpoly is None or hpoly.IsEmpty():
            break

        region = hpoly.Intersection(out_halfspace)
        if not region.IsEmpty():
            hpoly = hpoly.Intersection(in_halfspace)
            itvl = Interval(*get_hpoly_bounds(region, dim=-1))
            ret.append((region, itvl))
    
    return ret


def cuboid_side_halfspace_kd(
    xp:np.ndarray, xq:np.ndarray, halfsize:float
) -> List[Tuple[HPolyhedron, HPolyhedron]]:
    p_facet_verts = np.array(                                       # 3 -- e2 -- 2                      
                    [[xp[0] - halfsize, xp[1] - halfsize, xp[2]],   # |          |
                     [xp[0] + halfsize, xp[1] - halfsize, xp[2]],   # e3         e1
                     [xp[0] + halfsize, xp[1] + halfsize, xp[2]],   # |          |
                     [xp[0] - halfsize, xp[1] + halfsize, xp[2]]])  # 0 -- e0 -- 1                                                                               
    q_facet_verts = np.array(
                    [[xq[0] - halfsize, xq[1] - halfsize, xq[2]],   
                     [xq[0] + halfsize, xq[1] - halfsize, xq[2]],
                     [xq[0] + halfsize, xq[1] + halfsize, xq[2]],
                     [xq[0] - halfsize, xq[1] + halfsize, xq[2]]])
    edges = [[0, 1], [1, 2], [2, 3], [0, 3]]

    center = (xp + xq) / 2
    halfspaces = []
    for simplex in edges:
        u, v = simplex
        # get plane (ax + by + cz + d = 0) from 3 points
        p1, p2, p3 = p_facet_verts[u], p_facet_verts[v], q_facet_verts[u]
        v1, v2 = p2 - p1, p3 - p1
        normal = np.cross(v1, v2)
        a, b, c = normal
        d = -np.dot(normal, p1)
        # if A·center > b then the halfspace without center is A·x <= b - eps; otherwise -A·x <= -(b - eps)
        A, b = np.array([[a, b, c]]), np.array([[-d]])
        if np.dot(center, A[0]) + d > 0:
            halfspaces.append((HPolyhedron(A, b), HPolyhedron(-A, -b)))
        else:
            halfspaces.append((HPolyhedron(-A, -b), HPolyhedron(A, b)))
    
    return halfspaces


""" reserve any piecewise linear trajectory in STGCS, where the trajectory is not necessarily resulte
    (use it for PBS) """


def reconstruct_ECD_pairs(stgcs:STGCS, trajectory:List[np.ndarray]) -> List[ECDPair]:
    """ reconstruct a dictionary of v -> (xp, xq) segment pairs, assuming: 
        - each set contains at most one segment; 
        - while the segment is not necessarily completely contained in any set """
    dim = 3
    OPEN = [(wp[:3], wp[-3:]) for wp in trajectory]
    ecd_pairs = []
    V = set(stgcs.G.vertex_names)
    x0, xt = trajectory[0][:3], trajectory[-1][-3:]
    
    while OPEN:
        xp, xq = OPEN.pop()
        itvl = Interval(xp[-1], xq[-1])
        # print(f"Checking {xp} -> {xq} with time interval {itvl}")
        for v_name in V:
            v = stgcs.G.vertices[v_name]
            if itvl.intersects(v.itvl): # only check vertices with overlapping time intervals
                hpoly = squash_multi_points(v.convex_set.set, dim=dim)
                itsc = find_space_time_intersecting_pts(hpoly, xp, xq, dim)
                if itsc is not None and not np.allclose(itsc[0], itsc[1]):
                    V.remove(v_name)
                    ecd_pairs.append(ECDPair(
                        v.name, itsc[0], itsc[1],
                        reserve_xp_to_tmin = np.allclose(itsc[0], x0),
                        reserve_xq_to_tmax = np.allclose(itsc[1], xt)
                    ))
                    if not np.allclose(xp, itsc[0]):
                        OPEN.append((xp, itsc[0]))
                    if not np.allclose(itsc[1], xq):
                        OPEN.append((itsc[1], xq))
                    break
    
    return ecd_pairs


def reconstruct_SSP(
    stgcs:STGCS, xp:np.ndarray, xq:np.ndarray, set_p_name:str, set_q_name:str
) -> Tuple[List[str], List[np.ndarray]]:
        """ reconstruct the vertex_path and trajectory in GCS sets,
            assuming each line segment is contained completely in one or multiple sets """
        
        if set_p_name == set_q_name:
            return [set_q_name], [np.hstack([xp, xq])]
        
        found = False
        vertex_path, trajectory, visited = [], [], set()
        dim = len(xp)
        stack, visited = [(set_p_name, xp)], set()
        while stack:
            vc, xc = stack.pop()
            visited.add(vc)
            
            itsc = find_space_time_intersecting_pts(
                hpoly = squash_multi_points(stgcs.G.vertices[vc].convex_set.set, dim),
                xp = xc, xq = xq, dim = dim
            )
            if itsc is not None:
                stack = [] # empty the stack since we have stepped forward
                vertex_path.append(vc)
                trajectory.append(np.hstack([itsc[0], itsc[1]]))
                
                if stgcs.G.is_neighbor(vc, set_q_name):
                    found = True
                    vertex_path.append(set_q_name)
                    trajectory.append(np.hstack([itsc[1], xq]))
                    break

                for vn in stgcs.G.successors(vc):
                    if vn not in visited:
                        stack.append((vn, itsc[1]))
        
        # assert found, "Failed to find a path from set_p to set_q"
        return vertex_path, trajectory
