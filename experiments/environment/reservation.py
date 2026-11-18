from __future__ import annotations
from typing import Any, List, Tuple, Dict, Optional
from itertools import combinations, product
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from pydrake.all import HPolyhedron

from experiments.environment.space_time_graph import STGCS, STVertex
from experiments.environment.interval import Interval, AABB
from experiments.environment.geometry import get_hpoly_bounds

@dataclass
class ECDPair:
    bottom_halfspace: HPolyhedron
    top_halfspace: HPolyhedron
    mid_halfspaces: List[HPolyhedron]
    bounds: List[Interval] # spatial bounds + time bound

    @property
    def tube(self) -> HPolyhedron:
        outside = [*self.mid_halfspaces, self.bottom_halfspace, self.top_halfspace]
        return HPolyhedron(np.vstack([-h.A() for h in outside]), np.concatenate([-h.b() for h in outside]))


ECDPairCacheKey = Tuple[Any, ...]


def _ecd_pair_cache_key(
    stgcs: STGCS,
    trajectory: List[np.ndarray],
    safe_radius: float,
    staying_tmin: Optional[float],
    staying_tmax: Optional[float],
    eps: float,
) -> ECDPairCacheKey:
    return (
        int(stgcs.dimension),
        float(safe_radius),
        None if staying_tmin is None else float(staying_tmin),
        None if staying_tmax is None else float(staying_tmax),
        float(eps),
        tuple(
            tuple(float(value) for value in np.asarray(segment, dtype=float).ravel())
            for segment in trajectory
        ),
    )


def reserve(
    stgcs:STGCS, trajectory:List[np.ndarray], safe_radius:float, 
    x0_staying:bool=True, xt_staying:bool=True, eps:float=1e-9,
    ecd_pair_cache: Optional[Dict[ECDPairCacheKey, List[ECDPair]]]=None,
) -> STGCS:
    E = list(stgcs.G.edges)

    split_list = defaultdict(list)
    tmin = stgcs.t0 if x0_staying else None
    tmax = stgcs.tmax if xt_staying else None
    cache_key = None
    if ecd_pair_cache is not None:
        cache_key = _ecd_pair_cache_key(stgcs, trajectory, safe_radius, tmin, tmax, eps)
        ecd_pairs = ecd_pair_cache.get(cache_key)
    else:
        ecd_pairs = None
    if ecd_pairs is None:
        ecd_pairs = generate_all_ECD_pairs(stgcs, trajectory, safe_radius, tmin, tmax, eps=eps)
        if ecd_pair_cache is not None:
            assert cache_key is not None
            ecd_pair_cache[cache_key] = ecd_pairs
    for ecd_pair in ecd_pairs:
        tube = ecd_pair.tube
        for v_name in stgcs.G.nodes:
            vertex: STVertex = stgcs.get_vertex(v_name)
            if (AABB(ecd_pair.bounds, vertex.space_itvls + [vertex.time_itvl])
                    and vertex.st_hpoly.IntersectsWith(tube)):
                split_list[v_name].append(ecd_pair)
    if not split_list:
        return stgcs

    split_map = {v_name:[v_name] for v_name in stgcs.G.nodes}
    for v_name, ecd_pairs in split_list.items():
        v = stgcs.get_vertex(v_name)
        stgcs.remove_vertex_from_graph(v_name)
        for hpoly, time_itvl in slice(v.st_hpoly, ecd_pairs, v.time_itvl.start, v.time_itvl.end):
            new_v = stgcs.add_vertex(
                hpoly,
                time_itvl,
                parent=v,
                remove_redundancies=False,
            )
            if new_v is not None:
                split_map[v_name].append(new_v.name)
    
    stgcs = update_edge(stgcs, E, split_map)
    
    return stgcs


def update_edge(stgcs:STGCS, E:List[Tuple[str, str]], split_map:Dict[str, List[str]]) -> STGCS:
    """ update the edge between u and v in STGCS """
    edge_key = lambda u_name, v_name: (u_name, v_name) if u_name < v_name else (v_name, u_name)
    edge_checked = set([edge_key(u, v) for u, v in stgcs.G.edges])

    # build new edges between previously neighboring vertices
    for old_u, old_v in E:
        for u_name, v_name in product(split_map[old_u], split_map[old_v]):
            key = edge_key(u_name, v_name)
            if key not in edge_checked:
                edge_checked.add(key)
                stgcs.add_edges_bidir(u_name, v_name)

    # build edges within the split map
    for v_name, new_verts in split_map.items():
        for u_name, split_v_name in combinations(new_verts, 2):
            key = edge_key(u_name, split_v_name)
            if key not in edge_checked:
                edge_checked.add(key)
                stgcs.add_edges_bidir(u_name, split_v_name)
    
    return stgcs


def generate_all_ECD_pairs(
    stgcs:STGCS, trajectory:List[np.ndarray], safe_radius:float, 
    staying_tmin:float=None, staying_tmax:float=None, eps:float=1e-9
) -> List[ECDPair]:
    ret: List[ECDPair] = []
    traj = [np.asarray(segment, dtype=float).copy() for segment in trajectory]
    
    dim = stgcs.dimension + 1
    
    # add start/end staying trajectory points if necessary
    if staying_tmin is not None and traj[0][dim-1] != staying_tmin:
        x0t0 = np.hstack([traj[0][:dim-1], [staying_tmin], traj[0][:dim]]) 
        traj.insert(0, x0t0)
    
    if staying_tmax is not None and traj[-1][-1] != staying_tmax:
        xttf = np.hstack([traj[-1][-dim:], traj[-1][-dim:-1], [staying_tmax]])
        traj.append(xttf)
    
    # merge pieces with the same direction, and drop degenerate zero-length pieces
    i = 0
    while i < len(traj) - 1:
        dx = traj[i][dim:] - traj[i][:dim]
        dx_next = traj[i+1][dim:] - traj[i+1][:dim]
        n_dx = np.linalg.norm(dx)
        n_dx_next = np.linalg.norm(dx_next)
        if n_dx <= eps:
            traj.pop(i)
            if i > 0:
                i -= 1
            continue
        if n_dx_next <= eps:
            traj.pop(i + 1)
            continue

        if np.allclose(dx / n_dx, dx_next / n_dx_next):
            traj[i][dim:] = traj[i+1][dim:]
            traj.pop(i+1)
        else:
            i += 1

    for comp_x in traj: 
        xp, xq = comp_x[:dim], comp_x[-dim:]
        if np.linalg.norm(xq - xp) <= eps:
            continue
        bounds = []
        for d in range(dim-1):
            left, right = (xp[d], xq[d]) if xp[d] < xq[d] else (xq[d], xp[d])
            bounds.append(Interval(left - safe_radius, right + safe_radius))
        bounds.append(Interval(xp[-1], xq[-1]))
        bot_hs, top_hs = time_cropping_bot_top(dim, xp[-1], xq[-1])
        mid_hs_list = parallelotope_side_halfspace_kd(xp, xq, safe_radius, eps=eps)
        ret.append(ECDPair(bot_hs, top_hs, mid_hs_list, bounds))
    return ret


def slice(hpoly:HPolyhedron, ecd_pairs:List[ECDPair], tlow:float, thigh:float) -> List[Tuple[HPolyhedron, Interval]]:
    # note: the ecd_pairs must be collected from a continuous piece-wise linear trajectory 
    #       otherwise the slicing would be incorrect 
    st_dim = hpoly.ambient_dimension()
    ret = []
    bot = ecd_pairs[0].bottom_halfspace.Intersection(hpoly)
    if bot is not None and not bot.IsEmpty():
        ret.append((bot, Interval(tlow, ecd_pairs[0].bounds[-1].start)))
    top = ecd_pairs[-1].top_halfspace.Intersection(hpoly)
    if top is not None and not top.IsEmpty():
        ret.append((top, Interval(ecd_pairs[-1].bounds[-1].end, thigh)))

    sorted_pairs = sorted(ecd_pairs, key=lambda x: x.bounds[-1].start)

    # Spatial AABB filtering can omit legs between two visits to this set.
    # Those time spans are entirely free and must remain in the decomposition.
    for previous, following in zip(sorted_pairs, sorted_pairs[1:]):
        gap_start = max(tlow, previous.bounds[-1].end)
        gap_end = min(thigh, following.bounds[-1].start)
        if gap_start < gap_end:
            gap = time_cropping_mid(hpoly, gap_start, gap_end)
            if not gap.IsEmpty():
                ret.append((gap, Interval(gap_start, gap_end)))

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
                itvl = Interval(*get_hpoly_bounds(new_set, dim=st_dim-1))
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


def time_cropping_mid(hpoly:HPolyhedron, t_low:float, t_high:float) -> Optional[HPolyhedron]:
    if t_low >= t_high:
        return None

    if t_low == -np.inf and t_high == np.inf:
        return hpoly
    
    cropping_halfspace = HPolyhedron(
        A = np.block([np.zeros((2, hpoly.ambient_dimension()-1)), np.array([[-1], [1]])]),
        b = np.array([[-t_low], [t_high]])
    )
    
    return hpoly.Intersection(cropping_halfspace)


def parallelotope_verts_offset(dim:int) -> np.ndarray:
    signs = np.array(list(product([-1.0, 1.0], repeat=dim - 1)))
    return np.hstack([signs, np.zeros((signs.shape[0], 1))])


def parallelotope_side_halfspace_kd(
    xp:np.ndarray, xq:np.ndarray, apothem:float, eps:float=1e-9,
) -> List[HPolyhedron]:
    dim = xp.shape[0]
    if xq.shape[0] != dim:
        raise ValueError("xp and xq must have the same dimension")
    if dim < 2:
        raise ValueError(f"Unsupported dimension {dim}")

    space_dim = dim - 1
    dt = xq[-1] - xp[-1]
    if abs(dt) <= eps:
        raise ValueError("Expected a nonzero time interval for the trajectory segment")

    slope = (xq[:space_dim] - xp[:space_dim]) / dt
    intercept = xp[:space_dim] - slope * xp[-1]

    halfspaces = []
    for axis in range(space_dim):
        A_low = np.zeros((1, dim))
        A_low[0, axis] = 1.0
        A_low[0, -1] = -slope[axis]
        b_low = np.array([intercept[axis] - apothem])
        halfspaces.append(HPolyhedron(A_low, b_low))

        A_high = np.zeros((1, dim))
        A_high[0, axis] = -1.0
        A_high[0, -1] = slope[axis]
        b_high = np.array([-(intercept[axis] + apothem)])
        halfspaces.append(HPolyhedron(A_high, b_high))

    return halfspaces
