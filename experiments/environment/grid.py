from __future__ import annotations

from itertools import combinations
import os

import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull
from tqdm import tqdm

from experiments.environment.env import Env
from experiments.environment.geometry import make_hpolytope
from experiments.environment.interval import AABB, Interval


class GridEnvironmentBuilder:
    ROBOT_RADIUS = 0.1

    @staticmethod
    def _progress_disabled() -> bool:
        return os.environ.get("STGCS_DISABLE_PROGRESS", "") == "1"

    @classmethod
    def random(cls, seed: int, grid_shape: tuple[int, ...], num_pts_per_ply: int,
               spatial_scale: float = 1.0) -> Env:
        if not np.isfinite(spatial_scale) or spatial_scale <= 0:
            raise ValueError("spatial_scale must be finite and positive.")
        rng = np.random.default_rng(seed)
        space_dim = len(grid_shape)
        num_cells = int(np.prod(grid_shape))
        cells = []
        cell_hpolys = []
        space_bounds = []
        graph = nx.Graph()
        disable_progress = cls._progress_disabled()

        for idx_1d in tqdm(range(num_cells), desc="Generating polygons...", disable=disable_progress):
            cell_idx = np.array(np.unravel_index(idx_1d, grid_shape), dtype=float)
            points = 1.5 * rng.random((num_pts_per_ply, space_dim))
            hull = ConvexHull(points)
            vertices = (points[hull.vertices] + cell_idx) * spatial_scale
            graph.add_node(len(cells))
            cells.append(vertices)
            cell_hpolys.append(make_hpolytope(vertices))
            lb = np.min(vertices, axis=0)
            ub = np.max(vertices, axis=0)
            space_bounds.append([Interval(l, u) for l, u in zip(lb, ub)])

        for uid, vid in tqdm(
            combinations(range(num_cells), 2),
            desc="Checking graph connectivity ...",
            total=num_cells * (num_cells - 1) // 2,
            disable=disable_progress,
        ):
            if AABB(space_bounds[uid], space_bounds[vid]) and cell_hpolys[uid].IntersectsWith(cell_hpolys[vid]):
                graph.add_edge(uid, vid)

        largest_cc = max(nx.connected_components(graph), key=len)
        cells = [cells[i] for i in largest_cc]

        shape_str = "x".join(str(v) for v in grid_shape)
        return Env(
            f"rnd_seed{seed}_grid_{shape_str}",
            cells,
            robot_radius=cls.ROBOT_RADIUS * spatial_scale,
            OStatic=[],
            ODynamic=[],
        )

    @classmethod
    def random_2d(cls, seed: int, n: int, m: int, num_pts_per_ply: int = 8,
                  spatial_scale: float = 1.0) -> Env:
        return cls.random(seed, (n, m), num_pts_per_ply, spatial_scale)

    @classmethod
    def random_3d(cls, seed: int, n: int, m: int, num_pts_per_ply: int = 16,
                  spatial_scale: float = 1.0) -> Env:
        return cls.random(seed, (n, m, m), num_pts_per_ply, spatial_scale)
