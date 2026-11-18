""" this environment is adopted from https://github.com/cvxgrp/fastpathplanning
    Reference: "Fast Path Planning Through Large Collections of Safe Boxes"; https://arxiv.org/abs/2305.01072 """

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np

from experiments.environment.env import Env
from stgcs_native.types import MPQuery


class VillagePlanningInstance(SimpleNamespace):
    """Pickle-friendly planning instance with the heuristic hook expected by base caches."""

    def compute_heuristics(self) -> None:
        from experiments.offline_heuristics import BaseOfflineHeuristicStore
        BaseOfflineHeuristicStore.compute(self, cost_table=False)



class VillageEnv:
    """Environment and instance construction helpers for the 3D multi-UAV demo."""

    BUILDING_CSPACE_MODEL = "building-footprint-complement"
    CSPACE_SIMPLIFICATION_MODEL = "axis-aligned-box-exact-merge-disjoint-rings"
    RANDOM_QUERY_SEED_OFFSET = 1_000_003
    RANDOM_QUERY_MIN_ENDPOINT_SEPARATION_FACTOR = 3.0
    RANDOM_QUERY_MIN_TRAVEL_DISTANCE_FACTOR = 20.0
    RANDOM_QUERY_MIN_TRAVEL_DISTANCE_DOMAIN_FACTOR = 0.2
    RANDOM_QUERY_BOUNDARY_PADDING_FACTOR = 0.0
    RANDOM_QUERY_MAX_ATTEMPTS = 20_000
    BUILDING_BODY_COLOR = [0.66, 0.66, 0.64, 0.88]
    BUILDING_ROOF_COLOR = [0.62, 0.06, 0.05, 0.92]
    BUILDING_WINDOW_COLOR = [0.20, 0.48, 0.86, 0.92]
    BUILDING_LINE_COLOR = [0.16, 0.16, 0.16, 0.95]
    TREE_TRUNK_COLOR = [0.50, 0.24, 0.08, 0.90]
    TREE_FOLIAGE_COLOR = [0.13, 0.58, 0.16, 0.84]
    BUSH_COLOR = [0.04, 0.42, 0.12, 0.78]

    @staticmethod
    def box_vertices(lower: list[float] | np.ndarray, upper: list[float] | np.ndarray) -> np.ndarray:
        l = np.asarray(lower, dtype=float)
        u = np.asarray(upper, dtype=float)
        return np.asarray(
            [
                [l[0], l[1], l[2]],
                [u[0], l[1], l[2]],
                [u[0], u[1], l[2]],
                [l[0], u[1], l[2]],
                [l[0], l[1], u[2]],
                [u[0], l[1], u[2]],
                [u[0], u[1], u[2]],
                [l[0], u[1], u[2]],
            ],
            dtype=float,
        )

    @staticmethod
    def box_faces() -> list[list[int]]:
        return [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ]

    @staticmethod
    def box_edges() -> list[list[int]]:
        return [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ]

    @classmethod
    def box_mesh(
        cls,
        box_id: str,
        lower: list[float] | np.ndarray,
        upper: list[float] | np.ndarray,
        box_type: str = "polygon",
        fill_color: list[float] | None = None,
        line_color: list[float] | None = None,
        material: str | None = None,
    ) -> dict[str, Any]:
        vertices = cls.box_vertices(lower, upper)
        payload = {
            "id": box_id,
            "type": box_type,
            "vertices": vertices.tolist(),
            "faces": cls.box_faces(),
            "edges": cls.box_edges(),
        }
        if fill_color is not None:
            payload["fill_color"] = [float(value) for value in fill_color]
        if line_color is not None:
            payload["line_color"] = [float(value) for value in line_color]
        if material is not None:
            payload["material_name"] = str(material)
        return payload

    @staticmethod
    def direction(rng: np.random.Generator) -> np.ndarray:
        directions = np.vstack((np.eye(2), -np.eye(2)))
        return directions[int(rng.integers(0, 4))]

    @classmethod
    def walk(cls, rng: np.random.Generator, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d1 = cls.direction(rng)
        starts = [np.zeros(2)]
        ends = [d1]
        blocks = [np.zeros(2), d1]
        for _ in range(steps):
            d2 = cls.direction(rng)
            blocks.append(blocks[-1] + d2)
            if np.all(d2 == d1):
                ends[-1] += d1
            else:
                starts.append(ends[-1])
                ends.append(starts[-1] + d2)
                d1 = d2
        return np.asarray(starts), np.asarray(ends), np.asarray(blocks)

    @classmethod
    def append_building_windows(
        cls,
        static_obstacles: list[dict[str, Any]],
        prefix: str,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> None:
        body_height = float(upper[2] - lower[2])
        if body_height <= 0.6:
            return

        row_count = max(1, min(4, int(math.floor(body_height))))
        z_centers = np.linspace(lower[2] + 0.45, upper[2] - 0.35, row_count)
        eps = 0.012

        def append_face_windows(
            face_axis: int,
            fixed_coord: float,
            span_axis: int,
            outward_sign: float,
        ) -> None:
            span = float(upper[span_axis] - lower[span_axis])
            count = max(1, min(4, int(math.floor(span))))
            centers = np.linspace(lower[span_axis] + span / (count + 1), upper[span_axis] - span / (count + 1), count)
            half_span = min(0.15, 0.24 * span / max(count, 1))
            half_z = min(0.16, 0.18 * body_height / row_count)
            for row_idx, z_center in enumerate(z_centers):
                for col_idx, span_center in enumerate(centers):
                    win_lower = lower.copy()
                    win_upper = upper.copy()
                    win_lower[span_axis] = span_center - half_span
                    win_upper[span_axis] = span_center + half_span
                    win_lower[2] = z_center - half_z
                    win_upper[2] = z_center + half_z
                    if outward_sign > 0:
                        win_lower[face_axis] = fixed_coord
                        win_upper[face_axis] = fixed_coord + eps
                    else:
                        win_lower[face_axis] = fixed_coord - eps
                        win_upper[face_axis] = fixed_coord
                    static_obstacles.append(
                        cls.box_mesh(
                            f"{prefix}-window-{face_axis}-{row_idx}-{col_idx}-{outward_sign:g}",
                            win_lower,
                            win_upper,
                            fill_color=cls.BUILDING_WINDOW_COLOR,
                            line_color=cls.BUILDING_LINE_COLOR,
                            material="window",
                        )
                    )

        append_face_windows(face_axis=0, fixed_coord=float(lower[0]), span_axis=1, outward_sign=-1.0)
        append_face_windows(face_axis=0, fixed_coord=float(upper[0]), span_axis=1, outward_sign=1.0)
        append_face_windows(face_axis=1, fixed_coord=float(lower[1]), span_axis=0, outward_sign=-1.0)
        append_face_windows(face_axis=1, fixed_coord=float(upper[1]), span_axis=0, outward_sign=1.0)

    @classmethod
    def building(
        cls,
        rng: np.random.Generator,
        i: int,
        j: int,
        steps: int,
        village_height: float,
        robot_radius: float,
        building_clearance_margin: float,
        static_obstacles: list[dict[str, Any]],
        building_cspace_obstacles: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        starts, ends, blocks = cls.walk(rng, steps)
        offset = np.asarray([i, j], dtype=float) + 0.5
        starts += offset
        ends += offset
        blocks += np.asarray([i, j], dtype=float)
        for segment_idx, (start, end) in enumerate(zip(starts, ends)):
            center = 0.5 * (start + end)
            extent = np.abs(end - start) + 1.0
            radius_xy = 0.5 * extent - (robot_radius + building_clearance_margin)
            lower = np.asarray([center[0] - radius_xy[0], center[1] - radius_xy[1], 0.0], dtype=float)
            upper = np.asarray([center[0] + radius_xy[0], center[1] + radius_xy[1], village_height], dtype=float)
            building_cspace_obstacles.append(
                (
                    lower[:2] - float(robot_radius),
                    upper[:2] + float(robot_radius),
                )
            )
            roof_height = min(0.10, 0.04 * village_height)
            body_upper = upper.copy()
            body_upper[2] = max(lower[2], upper[2] - roof_height)
            static_obstacles.append(
                cls.box_mesh(
                    f"building-{i}-{j}-{segment_idx}-body",
                    lower,
                    body_upper,
                    fill_color=cls.BUILDING_BODY_COLOR,
                    line_color=cls.BUILDING_LINE_COLOR,
                    material="building-body",
                )
            )
            static_obstacles.append(
                cls.box_mesh(
                    f"building-{i}-{j}-{segment_idx}-roof",
                    [lower[0], lower[1], body_upper[2]],
                    upper,
                    fill_color=cls.BUILDING_ROOF_COLOR,
                    line_color=cls.BUILDING_LINE_COLOR,
                    material="building-roof",
                )
            )
            cls.append_building_windows(
                static_obstacles,
                f"building-{i}-{j}-{segment_idx}",
                lower,
                body_upper,
            )
        return blocks

    @staticmethod
    def outer_boxes(
        i: int,
        j: int,
        x: float,
        y: float,
        radius: float,
        zmin: float,
        zmax: float,
    ) -> tuple[list[list[float]], list[list[float]]]:
        lower = [
            [i, j, zmin],
            [x + radius, j, zmin],
            [x - radius, j, zmin],
            [x - radius, y + radius, zmin],
        ]
        upper = [
            [x - radius, j + 1, zmax],
            [i + 1, j + 1, zmax],
            [x + radius, y - radius, zmax],
            [x + radius, j + 1, zmax],
        ]
        return lower, upper

    @classmethod
    def append_safe_boxes(
        cls,
        safe_boxes: list[np.ndarray],
        lower_list: list[list[float]],
        upper_list: list[list[float]],
        min_width: float = 1e-8,
    ) -> None:
        for lower, upper in zip(lower_list, upper_list):
            l = np.asarray(lower, dtype=float)
            u = np.asarray(upper, dtype=float)
            if np.all(u - l > min_width):
                safe_boxes.append(cls.box_vertices(l, u))

    @classmethod
    def append_open_cell_safe_box(
        cls,
        safe_boxes: list[np.ndarray],
        i: int,
        j: int,
        village_height: float,
    ) -> None:
        cls.append_safe_boxes(
            safe_boxes,
            [[i, j, 0.0]],
            [[i + 1, j + 1, village_height]],
        )

    @staticmethod
    def clipped_rect(
        lower_xy: np.ndarray,
        upper_xy: np.ndarray,
        cell_lower: np.ndarray,
        cell_upper: np.ndarray,
        min_width: float = 1e-8,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        lower = np.maximum(np.asarray(lower_xy, dtype=float), cell_lower)
        upper = np.minimum(np.asarray(upper_xy, dtype=float), cell_upper)
        if np.all(upper - lower > float(min_width)):
            return lower, upper
        return None

    @staticmethod
    def point_in_rect(point: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
        return bool(np.all(point >= lower) and np.all(point <= upper))

    @classmethod
    def append_building_cell_safe_boxes(
        cls,
        safe_boxes: list[np.ndarray],
        i: int,
        j: int,
        village_height: float,
        building_cspace_obstacles: list[tuple[np.ndarray, np.ndarray]],
        min_width: float = 1e-8,
    ) -> int:
        cell_lower = np.asarray([float(i), float(j)], dtype=float)
        cell_upper = cell_lower + 1.0
        clipped_obstacles: list[tuple[np.ndarray, np.ndarray]] = []
        x_coords = [cell_lower[0], cell_upper[0]]
        y_coords = [cell_lower[1], cell_upper[1]]
        for obstacle_lower, obstacle_upper in building_cspace_obstacles:
            clipped = cls.clipped_rect(
                obstacle_lower,
                obstacle_upper,
                cell_lower,
                cell_upper,
                min_width=min_width,
            )
            if clipped is None:
                continue
            lower, upper = clipped
            clipped_obstacles.append((lower, upper))
            x_coords.extend([float(lower[0]), float(upper[0])])
            y_coords.extend([float(lower[1]), float(upper[1])])

        if not clipped_obstacles:
            return 0

        xs = sorted(set(x_coords))
        ys = sorted(set(y_coords))
        count = 0
        for x0, x1 in zip(xs[:-1], xs[1:]):
            if x1 - x0 <= float(min_width):
                continue
            for y0, y1 in zip(ys[:-1], ys[1:]):
                if y1 - y0 <= float(min_width):
                    continue
                midpoint = np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=float)
                if any(cls.point_in_rect(midpoint, lower, upper) for lower, upper in clipped_obstacles):
                    continue
                cls.append_safe_boxes(
                    safe_boxes,
                    [[x0, y0, 0.0]],
                    [[x1, y1, village_height]],
                    min_width=min_width,
                )
                count += 1
        return count

    @staticmethod
    def safe_box_bounds(box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vertices = np.asarray(box, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Expected 3D safe-box vertices, got shape {vertices.shape}.")
        return np.min(vertices, axis=0), np.max(vertices, axis=0)

    @staticmethod
    def box_contains(
        outer: tuple[np.ndarray, np.ndarray],
        inner: tuple[np.ndarray, np.ndarray],
        tolerance: float = 1e-9,
    ) -> bool:
        outer_lower, outer_upper = outer
        inner_lower, inner_upper = inner
        return bool(
            np.all(outer_lower <= inner_lower + float(tolerance))
            and np.all(outer_upper >= inner_upper - float(tolerance))
        )

    @classmethod
    def mergeable_axis_aligned_box_bounds(
        cls,
        first: tuple[np.ndarray, np.ndarray],
        second: tuple[np.ndarray, np.ndarray],
        tolerance: float = 1e-9,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if cls.box_contains(first, second, tolerance):
            return first[0].copy(), first[1].copy()
        if cls.box_contains(second, first, tolerance):
            return second[0].copy(), second[1].copy()

        first_lower, first_upper = first
        second_lower, second_upper = second
        merge_dim = None
        for dim in range(first_lower.size):
            same_interval = (
                abs(float(first_lower[dim] - second_lower[dim])) <= float(tolerance)
                and abs(float(first_upper[dim] - second_upper[dim])) <= float(tolerance)
            )
            if same_interval:
                continue
            intervals_touch = (
                max(float(first_lower[dim]), float(second_lower[dim]))
                <= min(float(first_upper[dim]), float(second_upper[dim])) + float(tolerance)
            )
            if not intervals_touch or merge_dim is not None:
                return None
            merge_dim = dim

        lower = np.minimum(first_lower, second_lower)
        upper = np.maximum(first_upper, second_upper)
        return lower, upper

    @classmethod
    def simplify_axis_aligned_safe_boxes(cls, safe_boxes: list[np.ndarray]) -> list[np.ndarray]:
        bounds = [cls.safe_box_bounds(box) for box in safe_boxes]
        changed = True
        while changed:
            changed = False
            for i in range(len(bounds)):
                for j in range(i + 1, len(bounds)):
                    merged = cls.mergeable_axis_aligned_box_bounds(bounds[i], bounds[j])
                    if merged is None:
                        continue
                    bounds[i] = merged
                    bounds.pop(j)
                    changed = True
                    break
                if changed:
                    break
        return [cls.box_vertices(lower, upper) for lower, upper in bounds]

    @classmethod
    def tree(
        cls,
        rng: np.random.Generator,
        i: int,
        j: int,
        village_height: float,
        robot_radius: float,
        safe_boxes: list[np.ndarray],
        static_obstacles: list[dict[str, Any]],
    ) -> None:
        foliage_radius = 0.5 - robot_radius
        trunk_radius = 0.1
        low = np.asarray([i + 0.5, j + 0.5, 1.0], dtype=float)
        high = np.asarray([i + 0.5, j + 0.5, village_height - 0.5], dtype=float)
        center = rng.uniform(low=low, high=high)
        trunk_lower = [center[0] - trunk_radius, center[1] - trunk_radius, 0.0]
        trunk_upper = [center[0] + trunk_radius, center[1] + trunk_radius, center[2] - foliage_radius]
        foliage_lower = center - foliage_radius
        foliage_upper = center + foliage_radius
        static_obstacles.append(
            cls.box_mesh(
                f"tree-{i}-{j}-trunk",
                trunk_lower,
                trunk_upper,
                fill_color=cls.TREE_TRUNK_COLOR,
                line_color=cls.BUILDING_LINE_COLOR,
                material="tree-trunk",
            )
        )
        static_obstacles.append(
            cls.box_mesh(
                f"tree-{i}-{j}-foliage",
                foliage_lower,
                foliage_upper,
                fill_color=cls.TREE_FOLIAGE_COLOR,
                line_color=[0.05, 0.22, 0.06, 0.88],
                material="tree-foliage",
            )
        )
        lower, upper = cls.outer_boxes(
            i, j, center[0], center[1], trunk_radius + robot_radius, 0.0, center[2] - foliage_radius
        )
        lower.append([i, j, center[2] + foliage_radius])
        upper.append([i + 1, j + 1, village_height])
        cls.append_safe_boxes(safe_boxes, lower, upper)

    @classmethod
    def bush(
        cls,
        rng: np.random.Generator,
        i: int,
        j: int,
        village_height: float,
        robot_radius: float,
        safe_boxes: list[np.ndarray],
        static_obstacles: list[dict[str, Any]],
    ) -> None:
        radius = float(rng.uniform(low=0.1, high=0.35))
        height = min(4.0 * radius, village_height)
        center = np.asarray([i + 0.5, j + 0.5], dtype=float)
        obstacle_lower = [center[0] - radius, center[1] - radius, 0.0]
        obstacle_upper = [center[0] + radius, center[1] + radius, height]
        static_obstacles.append(
            cls.box_mesh(
                f"bush-{i}-{j}",
                obstacle_lower,
                obstacle_upper,
                fill_color=cls.BUSH_COLOR,
                line_color=[0.03, 0.18, 0.06, 0.86],
                material="bush",
            )
        )
        lower, upper = cls.outer_boxes(
            i, j, center[0], center[1], radius + robot_radius, 0.0, height + robot_radius
        )
        lower.append([i, j, height + robot_radius])
        upper.append([i + 1, j + 1, village_height])
        cls.append_safe_boxes(safe_boxes, lower, upper)

    @classmethod
    def build_village_geometry(
        cls,
        village_side: int,
        village_height: float,
        building_every: int,
        decoration_density: float,
        robot_radius: float,
        building_clearance_margin: float,
        seed: int,
    ) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
        rng = np.random.default_rng(seed)
        safe_boxes: list[np.ndarray] = []
        static_obstacles: list[dict[str, Any]] = []
        building_cspace_obstacles: list[tuple[np.ndarray, np.ndarray]] = []
        blocks: list[tuple[int, int]] = []
        building_anchors = 0
        for i in range(village_side):
            for j in range(village_side):
                if (i + 1) % building_every == 0 and (j + 1) % building_every == 0:
                    building_anchors += 1
                    block_array = cls.building(
                        rng,
                        i,
                        j,
                        steps=4,
                        village_height=village_height,
                        robot_radius=robot_radius,
                        building_clearance_margin=building_clearance_margin,
                        static_obstacles=static_obstacles,
                        building_cspace_obstacles=building_cspace_obstacles,
                    )
                    blocks.extend((int(block[0]), int(block[1])) for block in block_array)

        block_set = set(blocks)
        building_safe_boxes = 0
        for i, j in sorted(block_set):
            if 0 <= i < village_side and 0 <= j < village_side:
                building_safe_boxes += cls.append_building_cell_safe_boxes(
                    safe_boxes,
                    i,
                    j,
                    village_height,
                    building_cspace_obstacles,
                )

        decorated_cells = 0
        open_cells = 0
        for i in range(village_side):
            for j in range(village_side):
                if (i, j) in block_set:
                    continue
                if float(rng.random()) >= float(decoration_density):
                    cls.append_open_cell_safe_box(safe_boxes, i, j, village_height)
                    open_cells += 1
                    continue
                decorated_cells += 1
                if int(rng.integers(0, 2)) == 0:
                    cls.tree(
                        rng,
                        i,
                        j,
                        village_height,
                        robot_radius,
                        safe_boxes,
                        static_obstacles,
                    )
                else:
                    cls.bush(
                        rng,
                        i,
                        j,
                        village_height,
                        robot_radius,
                        safe_boxes,
                        static_obstacles,
                    )

        num_safe_boxes_before_simplification = len(safe_boxes)
        safe_boxes = cls.simplify_axis_aligned_safe_boxes(safe_boxes)
        num_safe_boxes_after_simplification = len(safe_boxes)
        metadata = {
            "num_safe_boxes": int(num_safe_boxes_after_simplification),
            "num_safe_boxes_before_simplification": int(num_safe_boxes_before_simplification),
            "num_safe_boxes_removed_by_simplification": int(
                num_safe_boxes_before_simplification - num_safe_boxes_after_simplification
            ),
            "num_static_obstacles": len(static_obstacles),
            "num_building_anchors": int(building_anchors),
            "num_blocked_cells": len(block_set),
            "num_building_safe_boxes": int(building_safe_boxes),
            "num_decorated_cells": int(decorated_cells),
            "num_open_cells": int(open_cells),
            "decoration_density": float(decoration_density),
            "building_clearance_margin": float(building_clearance_margin),
            "building_cspace_model": cls.BUILDING_CSPACE_MODEL,
            "cspace_simplification_model": cls.CSPACE_SIMPLIFICATION_MODEL,
        }
        return safe_boxes, static_obstacles, metadata

    @classmethod
    def random_query_rng(cls, seed: int) -> np.random.Generator:
        return np.random.default_rng(int(seed) + cls.RANDOM_QUERY_SEED_OFFSET)

    @classmethod
    def random_query_box_data(
        cls,
        safe_boxes: list[np.ndarray],
        robot_radius: float,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
        padding = float(robot_radius) * cls.RANDOM_QUERY_BOUNDARY_PADDING_FACTOR
        box_bounds: list[tuple[np.ndarray, np.ndarray]] = []
        volumes: list[float] = []
        for box in safe_boxes:
            lower, upper = cls.safe_box_bounds(box)
            padded_lower = lower + padding
            padded_upper = upper - padding
            if np.any(padded_upper <= padded_lower):
                padded_lower = lower
                padded_upper = upper
            extent = padded_upper - padded_lower
            if np.any(extent <= 0.0):
                continue
            box_bounds.append((padded_lower, padded_upper))
            volumes.append(float(np.prod(extent)))

        if not box_bounds:
            raise ValueError("Cannot sample random UAV queries because no nonempty safe boxes were generated.")
        weights = np.asarray(volumes, dtype=float)
        weights /= float(np.sum(weights))
        return box_bounds, weights

    @staticmethod
    def sample_safe_point(
        rng: np.random.Generator,
        box_bounds: list[tuple[np.ndarray, np.ndarray]],
        box_weights: np.ndarray,
    ) -> np.ndarray:
        box_idx = int(rng.choice(len(box_bounds), p=box_weights))
        lower, upper = box_bounds[box_idx]
        return rng.uniform(lower, upper)

    @staticmethod
    def point_far_enough(point: np.ndarray, existing: list[np.ndarray], min_distance: float) -> bool:
        if not existing:
            return True
        return all(float(np.linalg.norm(point - other)) >= float(min_distance) for other in existing)

    @classmethod
    def random_endpoint(
        cls,
        rng: np.random.Generator,
        box_bounds: list[tuple[np.ndarray, np.ndarray]],
        box_weights: np.ndarray,
        existing: list[np.ndarray],
        min_endpoint_distance: float,
    ) -> np.ndarray:
        for _ in range(cls.RANDOM_QUERY_MAX_ATTEMPTS):
            candidate = cls.sample_safe_point(rng, box_bounds, box_weights)
            if cls.point_far_enough(candidate, existing, min_endpoint_distance):
                return candidate
        raise RuntimeError("Failed to sample separated random UAV query endpoints.")

    @classmethod
    def random_goal(
        cls,
        rng: np.random.Generator,
        box_bounds: list[tuple[np.ndarray, np.ndarray]],
        box_weights: np.ndarray,
        start: np.ndarray,
        existing_goals: list[np.ndarray],
        min_endpoint_distance: float,
        min_travel_distance: float,
    ) -> np.ndarray:
        for _ in range(cls.RANDOM_QUERY_MAX_ATTEMPTS):
            candidate = cls.sample_safe_point(rng, box_bounds, box_weights)
            if float(np.linalg.norm(candidate - start)) < float(min_travel_distance):
                continue
            if cls.point_far_enough(candidate, existing_goals, min_endpoint_distance):
                return candidate
        raise RuntimeError("Failed to sample random UAV goals with the requested travel distance.")

    @classmethod
    def random_endpoints(
        cls,
        safe_boxes: list[np.ndarray],
        village_side: int,
        village_height: float,
        num_uavs: int,
        robot_radius: float,
        seed: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        box_bounds, box_weights = cls.random_query_box_data(safe_boxes, robot_radius)
        rng = cls.random_query_rng(seed)
        domain_diag = float(np.linalg.norm([float(village_side), float(village_side), float(village_height)]))
        min_endpoint_distance = float(robot_radius) * cls.RANDOM_QUERY_MIN_ENDPOINT_SEPARATION_FACTOR
        min_travel_distance = max(
            float(robot_radius) * cls.RANDOM_QUERY_MIN_TRAVEL_DISTANCE_FACTOR,
            domain_diag * cls.RANDOM_QUERY_MIN_TRAVEL_DISTANCE_DOMAIN_FACTOR,
        )
        starts: list[np.ndarray] = []
        goals: list[np.ndarray] = []
        endpoints: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(num_uavs):
            start = cls.random_endpoint(
                rng,
                box_bounds,
                box_weights,
                starts,
                min_endpoint_distance,
            )
            goal = cls.random_goal(
                rng,
                box_bounds,
                box_weights,
                start,
                goals,
                min_endpoint_distance,
                min_travel_distance,
            )
            starts.append(start)
            goals.append(goal)
            endpoints.append((start, goal))
        return endpoints

    @classmethod
    def build_queries(
        cls,
        safe_boxes: list[np.ndarray],
        village_side: int,
        village_height: float,
        num_uavs: int,
        robot_radius: float,
        vlimit: float,
        seed: int,
    ) -> list[dict[str, Any]]:
        queries = []
        for start, goal in cls.random_endpoints(
            safe_boxes,
            village_side,
            village_height,
            num_uavs,
            robot_radius,
            seed,
        ):
            queries.append(
                {
                    "start": start.tolist(),
                    "goal": goal.tolist(),
                    "t_start": 0.0,
                    "is_stay": True,
                    "vlimit": float(vlimit),
                }
            )
        return queries

    @classmethod
    def build_record(
        cls,
        safe_boxes: list[np.ndarray],
        village_side: int,
        village_height: float,
        building_every: int,
        decoration_density: float,
        building_clearance_margin: float,
        num_uavs: int,
        robot_radius: float,
        vlimit: float,
        seed: int,
        geometry_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        queries = cls.build_queries(
            safe_boxes,
            village_side,
            village_height,
            num_uavs,
            robot_radius,
            vlimit,
            seed,
        )
        num_agents = len(queries)
        durations = [
            float(np.linalg.norm(np.asarray(query["goal"]) - np.asarray(query["start"])) / vlimit)
            for query in queries
        ]
        pair_count = math.comb(num_agents, 2) if num_agents > 1 else 0
        base_instance_id = f"base-fastpathplanning-village3d-seed{seed:05d}"
        return {
            "instance_id": f"mrmp-{base_instance_id}-{num_agents}-robot-random",
            "base_instance_id": base_instance_id,
            "queries": queries,
            "num_agents": int(num_agents),
            "independent_cost_sum": float(sum(durations)),
            "independent_makespan": float(max(durations, default=0.0)),
            "independent_runtime": 0.0,
            "num_conflicting_pairs": int(pair_count),
            "num_conflicting_agents": int(num_agents if pair_count else 0),
            "conflict_largest_component": int(num_agents if pair_count else 0),
            "conflict_density": float(pair_count / math.comb(num_agents, 2)) if num_agents > 1 else 0.0,
            "stgcs_num_vertices": int(geometry_metadata["num_safe_boxes"]),
            "stgcs_num_edges": 0,
            "village_side": int(village_side),
            "village_height": float(village_height),
            "building_every": int(building_every),
            "decoration_density": float(decoration_density),
            "building_clearance_margin": float(building_clearance_margin),
            "building_cspace_model": cls.BUILDING_CSPACE_MODEL,
            "cspace_simplification_model": cls.CSPACE_SIMPLIFICATION_MODEL,
            "spatial_seed": int(seed),
            "robot_radius": float(robot_radius),
            "vlimit": float(vlimit),
        }

    @classmethod
    def query_to_mp_query(cls, query: dict[str, Any]) -> MPQuery:
        return MPQuery(
            start=np.asarray(query["start"], dtype=float),
            goal=np.asarray(query["goal"], dtype=float),
            t_start=float(query["t_start"]),
            is_stay=bool(query["is_stay"]),
            vlimit=float(query["vlimit"]),
        )

    @classmethod
    def build_planning_instance(
        cls,
        record: dict[str, Any],
        safe_boxes: list[np.ndarray],
        robot_radius: float,
        vlimit: float,
        tmax: float,
        build_heuristics: bool = True,
    ) -> SimpleNamespace:
        env = Env(
            name=record["base_instance_id"],
            CSpace=[np.asarray(box, dtype=float) for box in safe_boxes],
            robot_radius=float(robot_radius),
            domain_lb=np.asarray([0.0, 0.0, 0.0], dtype=float),
            domain_ub=np.asarray(
                [float(record["village_side"]), float(record["village_side"]), float(record["village_height"])],
                dtype=float,
            ),
        )
        stgcs = env.build_STGCS(t0=0.0, tmax=float(tmax), vlimit=float(vlimit))
        stgcs.make_leaves_roots()
        record["stgcs_num_edges"] = int(stgcs.G.number_of_edges())
        instance = VillagePlanningInstance(
            name=record["instance_id"],
            env=env,
            stgcs=stgcs,
        )
        if build_heuristics:
            instance.compute_heuristics()
        return instance
