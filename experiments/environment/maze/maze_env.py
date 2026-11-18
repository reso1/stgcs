""" 
Generating maze env with convex C-Space regions merging to reduce the number of regions.
Source: https://github.com/reso1/MCFS/blob/master/baselines/TMSTCStar.py
ref: Lu, Junjie, et al. "TMSTC*: A path planning algorithm for minimizing turns in multi-robot coverage." IEEE Robotics and Automation Letters 8.8 (2023): 5275-5282.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from itertools import combinations, product
from collections import defaultdict

import numpy as np
import networkx as nx

from experiments.environment.env import Env
from experiments.environment.maze.rectangle import *
from experiments.environment.obstacle import StaticPolygon


def make_random_2d_maze(width:int, height:int, seed:int, robot_radius:float=0.25,
                        spatial_scale:float=1.0) -> Env:
    """ make a random maze using recursive division method """
    np_rng = np.random.RandomState(seed)
    map = np.zeros((width, height), dtype=int)

    def _recursive_func(r_idx_range:Tuple[int, int], c_idx_range:Tuple[int, int]) -> None:
        rw_start, rw_end = r_idx_range
        cw_start, cw_end = c_idx_range
        
        if rw_end - rw_start <= 2 or cw_end - cw_start <= 2:
            return
        
        # add horizontal wall
        row = np_rng.randint(rw_start + 1, rw_end - 1)
        map[row, cw_start:cw_end] = 1
        # add vertical wall
        col = np_rng.randint(cw_start + 1, cw_end - 1)
        map[rw_start:rw_end, col] = 1
        
        # add three holes in four wall segments
        segments = [(row+1, col, rw_end, col),        # top wall
                    (rw_start, col, row-1, col),      # bottom wall
                    (row, cw_start, row, col-1),      # left wall
                    (row, col+1, row, cw_end)         # right wall
                    ]
        
        for seg_idx in np_rng.choice(4, 3, replace=False):
            r_s, c_s, r_e, c_e = segments[seg_idx]
            if r_s == r_e:
                if c_s == c_e:
                    hole_c = c_s
                else:
                    hole_c = np_rng.randint(c_s, c_e)
                map[r_s, hole_c] = 0
            else:
                hole_r = np_rng.randint(r_s, r_e)
                map[hole_r, c_s] = 0

        _recursive_func((rw_start, row-1), (cw_start, col-1))
        _recursive_func((rw_start, row-1), (col+1, cw_end))
        _recursive_func((row+1, rw_end), (cw_start, col-1))
        _recursive_func((row+1, rw_end), (col+1, cw_end))

    _recursive_func((0, height), (0, width))

    for r, c in product(range(height), range(width)):
        if map[r][c] == 0:
            left_cell = map[r][c-1] if c-1 >= 0 else 1
            right_cell = map[r][c+1] if c+1 < width else 1
            top_cell = map[r-1][c] if r-1 >= 0 else 1
            bottom_cell = map[r+1][c] if r+1 < height else 1
            if left_cell + right_cell + top_cell + bottom_cell == 4:
                map[r][c] = 1 # fill isolated cell

    return make_2d_maze_env(map, robot_radius=robot_radius, spatial_scale=spatial_scale)


def make_2d_maze_env(occ_grid:np.ndarray, robot_radius:float=0.25,
                     spatial_scale:float=1.0) -> Env:
    if not np.isfinite(spatial_scale) or spatial_scale <= 0:
        raise ValueError("spatial_scale must be finite and positive.")
    width, height = occ_grid.shape
    R:Dict[tuple, Rectangle] = {}
    for xc, yc in product(range(width), range(height)):
        xn, yn = xc + 1, yc + 1
        if occ_grid[xc, yc] == 0:
            R[(xc, yc)] = Rectangle((xc, yc), (xn, yn))
            
    I, E, V, H = nx.DiGraph(), set(), set(), set()

    edge_key = lambda ra, rb: (ra.lowerleft, rb.lowerleft) if ra.lowerleft < rb.lowerleft else (rb.lowerleft, ra.lowerleft)

    for va, vb in combinations(R.keys(), 2):
        ra, rb = R[va], R[vb]
        if ra.lowerleft[0] == rb.lowerleft[0]:
            if ra.lowerleft[1] + ra.height == rb.lowerleft[1]:
                v = edge_key(ra, rb)
                E.add(v)
                V.add(v)
                I.add_node(v, bipartite=VERTICAL)
            if ra.lowerleft[1] == rb.lowerleft[1] + rb.height:
                v = edge_key(ra, rb)
                E.add(v)
                V.add(v)
                I.add_node(v, bipartite=VERTICAL)
        
        if ra.lowerleft[1] == rb.lowerleft[1]:
            if ra.lowerleft[0] + ra.width == rb.lowerleft[0]:
                v = edge_key(ra, rb)
                E.add(v)
                H.add(v)
                I.add_node(v, bipartite=HORIZONTAL)
            if ra.lowerleft[0] == rb.lowerleft[0] + rb.width:
                v = edge_key(ra, rb)
                E.add(v)
                H.add(v)
                I.add_node(v, bipartite=HORIZONTAL)
    
    I.graph["V"], I.graph["H"], I.graph["rect"] = V, H, R
    for h, v in product(H, V):
        if v[0] == h[0] or v[0] == h[1] or v[1] == h[0] or v[1] == h[1]:
            I.add_edge(h, v, capacity=1)
    
    rects = bipartite_min_vertex_cover(I, nx.Graph(E))
    rects = remove_redundant(occ_grid, rects)

    edges = set()
    for i, j in combinations(range(len(rects)), 2):
        ra, rb = rects[i], rects[j]
        if ra.intersects(rb):
            edges.add((i, j))

    print(f"Maze env generated from grid: {width}x{height}, "
          f"# of initial rects={len(R)}, "
          f"# of merged rects={len(rects)}")
    
    return Env(
        name=f"maze_{width}x{height}",
        CSpace=[rect.ply_ndarray * spatial_scale for rect in rects],
        robot_radius=robot_radius,
        OStatic=[StaticPolygon(obstacle.vertices * spatial_scale)
                 for obstacle in create_obstacle_blocks(occ_grid, robot_radius / spatial_scale)],
        edges = edges,
        domain_lb=np.array([0.0, 0.0]),
        domain_ub=spatial_scale * np.array([float(width), float(height)]),
    )


def bipartite_min_vertex_cover(I:nx.DiGraph, G:nx.Graph) -> List[Rectangle]:
    # find maximum independent set
    src, dst = "s", "t"
    I.add_edges_from([(src, h) for h in I.graph["H"]], capacity=1)
    I.add_edges_from([(v, dst) for v in I.graph["V"]], capacity=1)

    residual, H, V = nx.Graph(), set(), set()
    residual.add_nodes_from([src, dst])
    _, flow_dict = nx.maximum_flow(I, src, dst)
    for u, neighbors in flow_dict.items():
        for v, flow in neighbors.items():
            if flow == 0:
                residual.add_edge(u, v)
                if u != src and u != dst:
                    if I.nodes[u]["bipartite"] == HORIZONTAL:
                        H.add(u)
                    else:
                        V.add(u)
                if v != src and v != dst:
                    if I.nodes[v]["bipartite"] == HORIZONTAL:
                        H.add(v)
                    else:
                        V.add(v)
    
    C = set(I.graph["H"]) - set([h for h in H if not nx.has_path(residual, src, h)])
    D = set(I.graph["V"]) - set([v for v in V if nx.has_path(residual, src, v)])
    
    merged_grids = set()

    # first merge connected components in C and D
    merged_H = []
    for cc_H in nx.connected_components(nx.Graph(C)):
        merged_grids.update(cc_H)
        rect_list = [I.graph["rect"][h] for h in cc_H]
        merged_rect = Rectangle.merge_all(rect_list)
        merged_H.append(merged_rect)
    
    merged_V = []
    for cc_V in nx.connected_components(nx.Graph(D)):
        merged_grids.update(cc_V)
        rect_list = [I.graph["rect"][v] for v in cc_V]
        merged_rect = Rectangle.merge_all(rect_list)
        merged_V.append(merged_rect)

    return merged_H + merged_V + [I.graph["rect"][v] for v in G.nodes - merged_grids]


def remove_redundant(occ_grid:np.ndarray, rects:List[Rectangle]) -> List[Rectangle]:
    ret = []

    # grow each rectangle until hits obstacle
    while rects:
        r = rects.pop()
        left, right = r.lowerleft[0], r.lowerright[0]
        top, bottom = r.upperleft[1], r.lowerleft[1]
        # grow left
        while left - 1 >= 0 and all(occ_grid[left - 1, bottom:top] == 0):
            left -= 1
        # grow right
        while right < occ_grid.shape[0] and all(occ_grid[right, bottom:top] == 0):
            right += 1
        # grow bottom
        while bottom - 1 >= 0 and all(occ_grid[left:right, bottom - 1] == 0):
            bottom -= 1
        # grow top
        while top < occ_grid.shape[1] and all(occ_grid[left:right, top] == 0):
            top += 1
        
        ret.append(Rectangle((left, bottom), (right, top)))
        # remove redundant rectangles
        for rb in rects:
            if r.contains(rb):
                rects.remove(rb)
    
    for i, ra in enumerate(ret):
        for j, rb in enumerate(ret):
            if i != j and ra is not None and rb is not None and ra.contains(rb):
                ret[j] = None

    ret:List[Rectangle] = [r for r in ret if r is not None]
    
    # shrink 1D rectangles
    for idx, r in enumerate(ret):
        if r.width == 1:
            bottom, top = r.lowerleft[1], r.upperleft[1]
            other_rects = [r for k, r in enumerate(ret) if k != idx]
            while bottom < top - 1 and point_in_any_rectangle((r.lowerleft[0], bottom), other_rects):
                bottom += 1
            while bottom < top - 1 and point_in_any_rectangle((r.lowerleft[0], top - 1), other_rects):
                top -= 1
            ret[idx] = Rectangle((r.lowerleft[0], bottom), (r.lowerright[0], top)) 
        elif r.height == 1:
            left, right = r.lowerleft[0], r.lowerright[0]
            other_rects = [r for k, r in enumerate(ret) if k != idx]
            while left < right - 1 and point_in_any_rectangle((left, r.lowerleft[1]), other_rects):
                left += 1
            while left < right - 1 and point_in_any_rectangle((right - 1, r.lowerleft[1]), other_rects):
                right -= 1
            ret[idx] = Rectangle((left, r.lowerleft[1]), (right, r.upperleft[1]))

    return ret


def point_in_any_rectangle(pt:tuple, rects:List[Rectangle]) -> bool:
    for r in rects:
        if r.point_in_rect((pt[0]+0.5, pt[1]+0.5)):
            return True
    return False


def create_obstacle_blocks(occ_grid:np.ndarray, robot_radius:float) -> List[StaticPolygon]:
    G = nx.Graph()
    width, height = occ_grid.shape
    for xc, yc in product(range(width), range(height)):
        if occ_grid[xc, yc] == 1:
            G.add_node((xc, yc))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                xn, yn = xc + dx, yc + dy
                if 0 <= xn < width and 0 <= yn < height and occ_grid[xn, yn] == 1:
                    G.add_edge((xc, yc), (xn, yn))

    ret = []
    for cc in nx.connected_components(G):
        if len(cc) == 1:
            xc, yc = cc.pop()
            obs = StaticPolygon(np.array([
                [xc + robot_radius, yc + robot_radius],
                [xc + 1 - robot_radius, yc + robot_radius],
                [xc + 1 - robot_radius, yc + 1 - robot_radius],
                [xc + robot_radius, yc + 1 - robot_radius],
            ]))
            ret.append(obs)
        
        # try create a vertical wall
        V = set(cc)
        while V:
            v = V.pop()
            xc, yc = v
            vertical_wall = (xc, yc, yc)
            xc_top, xc_bottom = (xc, yc + 1), (xc, yc - 1)
            while xc_top in V:
                vertical_wall = (vertical_wall[0], vertical_wall[1], vertical_wall[2] + 1)
                V.remove(xc_top)
                xc_top = (xc_top[0], xc_top[1] + 1)
            while xc_bottom in V:
                vertical_wall = (vertical_wall[0], vertical_wall[1] - 1, vertical_wall[2])
                V.remove(xc_bottom)
                xc_bottom = (xc_bottom[0], xc_bottom[1] - 1)
            if vertical_wall[2] > vertical_wall[1]:
                obs = StaticPolygon(np.array([
                    [vertical_wall[0] + robot_radius, vertical_wall[1] + robot_radius],
                    [vertical_wall[0] + robot_radius, vertical_wall[2] + 1 - robot_radius],
                    [vertical_wall[0] + 1 - robot_radius, vertical_wall[2] + 1 - robot_radius],
                    [vertical_wall[0] + 1 - robot_radius, vertical_wall[1] + robot_radius],
                ]))
                ret.append(obs)

        # try create a horizontal wall
        V = set(cc)
        while V:
            v = V.pop()
            xc, yc = v
            horizontal_wall = (xc, yc, xc)
            yc_left, yc_right = (xc - 1, yc), (xc + 1, yc)
            while yc_left in V:
                horizontal_wall = (horizontal_wall[0] - 1, horizontal_wall[1], horizontal_wall[2])
                V.remove(yc_left)
                yc_left = (yc_left[0] - 1, yc_left[1])
            while yc_right in V:
                horizontal_wall = (horizontal_wall[0], horizontal_wall[1], horizontal_wall[2] + 1)
                V.remove(yc_right)
                yc_right = (yc_right[0] + 1, yc_right[1])
            if horizontal_wall[2] > horizontal_wall[0]:
                obs = StaticPolygon(np.array([
                    [horizontal_wall[0] + robot_radius, horizontal_wall[1] + robot_radius],
                    [horizontal_wall[2] + 1 - robot_radius, horizontal_wall[1] + robot_radius],
                    [horizontal_wall[2] + 1 - robot_radius, horizontal_wall[1] + 1 - robot_radius],
                    [horizontal_wall[0] + robot_radius, horizontal_wall[1] + 1 - robot_radius],
                ]))
                ret.append(obs)

    return ret
