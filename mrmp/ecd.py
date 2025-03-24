from __future__ import annotations
from typing import List, Tuple, Dict, Set
from itertools import combinations, product
from collections import defaultdict
from dataclasses import dataclass

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pydrake.all import HPolyhedron

from mrmp.stgcs import STGCS
from mrmp.graph import ShortestPathSolution, Edge, Graph, Vertex
from mrmp.interval import Interval, AABB
from mrmp.utils import (
    timeit, get_hpoly_bounds, squash_multi_points, find_space_time_intersecting_pts,
    draw_2d_set, draw_3d_set, draw_cuboid, draw_parallelpiped,
)


@dataclass
class ECDPair:
    bottom_halfspace: HPolyhedron
    top_halfspace: HPolyhedron
    mid_halfspaces: List[HPolyhedron]
    bounds: List[Interval]


def reserve(
    stgcs:STGCS, trajectory:List[np.ndarray], safe_radius:float, 
    x0_staying:bool=True, xt_staying:bool=True, debug:bool=False
) -> STGCS:
    new = stgcs.copy()
    adj_graph = nx.Graph()
    for u, V in stgcs.G._adjacency_list.items():
        adj_graph.add_edges_from([(u, v) for v in V])
    
    split_list = defaultdict(list)
    tmin = stgcs.t0 if x0_staying else None
    tmax = stgcs.tmax if xt_staying else None
    for ecd_pair in generate_all_ECD_pairs(stgcs, trajectory, safe_radius, tmin, tmax):
        for v_name in stgcs.G.vertex_names:
            v = stgcs.G.vertices[v_name]
            v_bounds = v.space_bounds + [v.itvl]
            if AABB(ecd_pair.bounds, v_bounds):
                split_list[v_name].append(ecd_pair)

    split_map = {v_name:set([v_name]) for v_name in stgcs.G.vertex_names}
    for v_name, ecd_pairs in split_list.items():
        v = new.remove_vertex(v_name)
        hpoly = squash_multi_points(v.convex_set.set, dim=stgcs.dim)
        for res in slice(hpoly, ecd_pairs, v.itvl.start, v.itvl.end):
            new_v = new.try_add_vertex(*res)
            if new_v is not None:
                split_map[v_name].add(new_v.name)
    
    new = update_edge(new, adj_graph, split_map)

    if debug:
        if stgcs.dim == 2:
            fig, ax = plt.subplots(1, 2)
            ax[0].axis("equal")
            stgcs.draw(ax[0])
            ax[1].axis("equal")
            for x in trajectory:
                xp, xq = x[:2], x[2:]
                ax[0].plot([xp[0], xq[0]], [xp[1], xq[1]], "-.k")
                draw_parallelpiped(ax[1], xp, xq, 0.8 * safe_radius)
            new.draw(ax[1], edges=True)
        elif stgcs.dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.axis("equal")
            stgcs.draw(ax)
            ax.axis("off")
            for x in trajectory:
                xp, xq = x[:3], x[3:]
                draw_cuboid(ax, xp, xq, 0.9 * safe_radius)
            new.draw(ax, edges=False)

        plt.show()
    
    return new


def update_edge(
    stgcs:STGCS, adj_graph:nx.DiGraph, split_map:Dict[str, List[str]]
) -> STGCS:
    """ update the edge between u and v in STGCS """
    edge_key = lambda u_name, v_name: (u_name, v_name) if u_name < v_name else (v_name, u_name)
    edge_checked = set([edge_key(e.u, e.v) for e in stgcs.G.edges.values()])

    # build new edges between previously neighboring vertices
    for old_u, old_v in adj_graph.edges:
        for u_name, v_name in product(split_map[old_u], split_map[old_v]):
            key = edge_key(u_name, v_name)
            if key not in edge_checked:
                edge_checked.add(key)
                stgcs.try_build_edge(u_name, v_name)

    # build edges within the split map
    for v_name, new_verts in split_map.items():
        for u_name, v_name in combinations(new_verts, 2):
            key = edge_key(u_name, v_name)
            if key not in edge_checked:
                edge_checked.add(key)
                stgcs.try_build_edge(u_name, v_name)
    
    return stgcs


def generate_all_ECD_pairs(
    stgcs:STGCS, trajectory:List[np.ndarray], safe_radius:float, 
    staying_tmin:float=None, staying_tmax:float=None
) -> List[ECDPair]:
    ret: List[ECDPair] = []
    
    dim = stgcs.dim
    if dim == 2:
        halfspace_func = parallelepiped_side_halfspace_1d
    elif dim == 3:
        halfspace_func = parallelepiped_side_halfspace_2d
    else:
        raise ValueError(f"Unsupported dimension {dim}")
    
    # add start/end staying trajectory points if necessary
    if staying_tmin is not None and trajectory[0][dim-1] != staying_tmin:
        x0t0 = np.hstack([trajectory[0][:dim-1], [staying_tmin], trajectory[0][:dim]]) 
        trajectory.insert(0, x0t0)
    
    if staying_tmax is not None and trajectory[-1][-1] != staying_tmax:
        xttf = np.hstack([trajectory[-1][-dim:], trajectory[-1][-dim:-1], [staying_tmax]])
        trajectory.append(xttf)
    
    # merge pieces with the same direction
    i = 0
    while i < len(trajectory) - 1:
        dx = trajectory[i][dim:] - trajectory[i][:dim]
        dx_next = trajectory[i+1][dim:] - trajectory[i+1][:dim]
        if np.allclose(dx/np.linalg.norm(dx), dx_next/np.linalg.norm(dx_next)):
            trajectory[i][dim:] = trajectory[i+1][dim:]
            trajectory.pop(i+1)
        else:
            i += 1

    for comp_x in trajectory: 
        xp, xq = comp_x[:dim], comp_x[-dim:]
        bounds = []
        for d in range(dim-1):
            left, right = (xp[d], xq[d]) if xp[d] < xq[d] else (xq[d], xp[d])
            bounds.append(Interval(left - safe_radius, right + safe_radius))
        bounds.append(Interval(xp[-1], xq[-1]))
        bot_hs, top_hs = time_cropping_bot_top(dim, xp[-1], xq[-1])
        mid_hs_list = halfspace_func(xp, xq, safe_radius)
        ret.append(ECDPair(bot_hs, top_hs, mid_hs_list, bounds))
    return ret


def slice(hpoly:HPolyhedron|None, ecd_pairs:List[ECDPair], tlow:float, thigh:float) -> List[HPolyhedron]:    
    # note: the ecd_pairs must be collected from a continuous piece-wise linear trajectory 
    #       otherwise the slicing would be incorrect 
    
    ret = []
    bot = ecd_pairs[0].bottom_halfspace.Intersection(hpoly)
    if bot is not None and not bot.IsEmpty():
        ret.append((bot, Interval(tlow, ecd_pairs[0].bounds[-1].start)))
    top = ecd_pairs[-1].top_halfspace.Intersection(hpoly)
    if top is not None and not top.IsEmpty():
        ret.append((top, Interval(ecd_pairs[-1].bounds[-1].end, thigh)))

    # merge pairs if they have the same direction
    sorted_pairs = sorted(ecd_pairs, key=lambda x: x.bounds[-1].start)


    for ecd_pair in sorted_pairs:
        mid_tlow, mid_thigh = ecd_pair.bounds[-1].start, ecd_pair.bounds[-1].end
        mid = time_cropping_mid(hpoly, mid_tlow, mid_thigh)
        
        for out_halfspace in ecd_pair.mid_halfspaces:
            if mid is None or mid.IsEmpty():
                break

            in_halfspace = HPolyhedron(-out_halfspace.A(), -out_halfspace.b())
            new_set = mid.Intersection(out_halfspace)
            if not new_set.IsEmpty():
                mid = mid.Intersection(in_halfspace)
                itvl = Interval(*get_hpoly_bounds(new_set, dim=-1))
                ret.append((new_set, itvl))

    if len(ret) == 1:
        # TODO: do not slice if the set is not split
        pass

    return ret


def time_cropping_bot_top(dim:int, t_low:float, t_high:float) -> List[HPolyhedron]:
    # [0, 0, 1] @ [x, y, t] <= t_low ----> t <= t_low
    bottom = HPolyhedron(
        A = np.hstack([np.zeros(dim-1), 1]).reshape(1, -1),
        b = np.array([[t_low]])
    )

    # [0, 0, -1] @ [x, y, t] <= -t_high ----> t >= t_high
    top = HPolyhedron(
        A = np.hstack([np.zeros(dim-1), -1]).reshape(1, -1),
        b = np.array([[-t_high]])
    )

    return [bottom, top]


def time_cropping_mid(hpoly:HPolyhedron, t_low:float, t_high:float) -> HPolyhedron|None:
    if t_low >= t_high:
        return None

    if t_low == -np.inf and t_high == np.inf:
        return hpoly
    
    cropping_halfspace = HPolyhedron(
        A = np.block([np.zeros((2, hpoly.ambient_dimension()-1)), np.array([[-1], [1]])]),
        b = np.array([[-t_low], [t_high]])
    )
    
    return hpoly.Intersection(cropping_halfspace)


def parallelepiped_side_halfspace_1d(
    xp:np.ndarray, xq:np.ndarray, halfsize:float,
) -> List[Tuple[HPolyhedron, HPolyhedron]]:
    p_line = np.array([[xp[0] - halfsize, xp[1]], [xp[0] + halfsize, xp[1]]])
    q_line = np.array([[xq[0] - halfsize, xq[1]], [xq[0] + halfsize, xq[1]]])

    center = (xp + xq) / 2
    halfspaces = []
    for p1, p2 in zip(p_line, q_line):
        vec = p2 - p1
        normal = np.hstack([vec[1], -vec[0]]).reshape(1, -1)
        c = -np.dot(normal, p1)
        A, b = normal, np.array([-c])
        if np.dot(center, A[0]) + c > 0:
            halfspaces.append(HPolyhedron(A, b))
        else:
            halfspaces.append(HPolyhedron(-A, -b))
    
    return halfspaces


def parallelepiped_side_halfspace_2d(
    xp:np.ndarray, xq:np.ndarray, halfsize:float,
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
            halfspaces.append(HPolyhedron(A, b))
        else:
            halfspaces.append(HPolyhedron(-A, -b))
    
    return halfspaces

