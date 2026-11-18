from __future__ import annotations

from functools import partial
from visualization.viewer.server import ViewerServer
from visualization.viewer.planning import ViewerPlannerService

from datetime import datetime, timezone
from typing import Any

import numpy as np


class Village3DVisualizationMixin:
    """Viewer manifest and serving helpers for the 3D multi-UAV demo."""

    @classmethod
    def build_viewer_manifest(
        cls,
        record: dict[str, Any],
        safe_boxes: list[np.ndarray],
        static_obstacles: list[dict[str, Any]],
        robot_radius: float,
        geometry_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        bounds = {
            "min": [0.0, 0.0, 0.0],
            "max": [
                float(record["village_side"]),
                float(record["village_side"]),
                float(record["village_height"]),
            ],
        }
        instance = {
            "instance_id": record["instance_id"],
            "base_instance_id": record["base_instance_id"],
            "problem_kind": "mrmp",
            "domain": "fastpathplanning-village3d",
            "space_dim": 3,
            "spatial_seed": int(record["spatial_seed"]),
            "env_params": {
                "village_side": int(record["village_side"]),
                "village_height": float(record["village_height"]),
                "building_every": int(record["building_every"]),
                "decoration_density": float(record["decoration_density"]),
                "building_clearance_margin": float(record["building_clearance_margin"]),
                "building_cspace_model": str(record["building_cspace_model"]),
                "cspace_simplification_model": str(record["cspace_simplification_model"]),
                "source_repo": "cvxgrp/fastpathplanning",
            },
            "stgcs_num_vertices": int(record["stgcs_num_vertices"]),
            "stgcs_num_edges": int(record["stgcs_num_edges"]),
            "dimension": 3,
            "robot_radius": float(robot_radius),
            "agent_model": "uav",
            "uav_materials": cls.UAV_MATERIALS,
            "uav_visual_scale": cls.UAV_VISUAL_SCALE,
            "bounds": bounds,
            "spatial_sets": [
                {
                    "id": f"v{idx}",
                    "vertices": np.asarray(vertices, dtype=float).tolist(),
                    "faces": cls.box_faces(),
                    "edges": cls.box_edges(),
                }
                for idx, vertices in enumerate(safe_boxes)
            ],
            "static_obstacles": static_obstacles,
            "time_horizon": 0.0,
            "time_range": [0.0, 0.0],
            "queries": record["queries"],
            "dynamic_obstacles": [],
            "num_agents": int(record["num_agents"]),
            "independent_cost_sum": float(record["independent_cost_sum"]),
            "independent_makespan": float(record["independent_makespan"]),
            "independent_runtime": float(record["independent_runtime"]),
            "num_conflicting_pairs": int(record["num_conflicting_pairs"]),
            "num_conflicting_agents": int(record["num_conflicting_agents"]),
            "conflict_largest_component": int(record["conflict_largest_component"]),
            "conflict_density": float(record["conflict_density"]),
            "cspace_alpha": cls.CSPACE_ALPHA,
            "static_obstacle_alpha": cls.STATIC_OBSTACLE_ALPHA,
            "geometry_metadata": geometry_metadata,
        }
        return {

            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_kind": "mrmp",
            "instance_count": 1,
            "instances": [instance],
        }

    @classmethod
    def planner_options(cls) -> list[dict[str, Any]]:
        return [
            {
                "key": cls.MRMP_PLANNER_KEY,
                "name": cls.MRMP_PLANNER_NAME,
                "button_label": "MRMP planner",
                "space_dims": [3],
            },
            {
                "key": cls.PLANNER_KEY,
                "name": cls.PLANNER_NAME,
                "button_label": "MRMP + trajopt",
                "space_dims": [3],
            },
        ]

    @classmethod
    def planner_name(cls, planner_key: str) -> str:
        for option in cls.planner_options():
            if option["key"] == planner_key:
                return str(option["name"])
        raise ValueError(f"Unknown planner_key: {planner_key!r}")

    @classmethod
    def planner_payload(cls, solution_budget: float, window_alpha: float, epsilon: float) -> dict[str, Any]:
        return {
            "ok": True,
            "default_budget": float(solution_budget),
            "default_window_alpha": float(window_alpha),
            "default_window_beta": 1.0,
            "default_epsilon": float(epsilon),
            "planners": {
                "mrmp": cls.planner_options(),
                "st_heuristic_ablation": [],
                "base": [],
            },
        }

    @classmethod
    def solve_request(cls, record, base_record, solution_budget, window_alpha, epsilon, request):
        if request.get("instance_id") != record["instance_id"]:
            raise ValueError(f"Unknown instance_id: {request.get('instance_id')!r}")
        planner_key = str(request.get("planner_key", cls.PLANNER_KEY))
        cls.planner_name(planner_key)
        budget = ViewerPlannerService.positive_finite_float(request.get("budget", solution_budget), "Budget")
        alpha = ViewerPlannerService.positive_finite_float(request.get("window_alpha", window_alpha), "window_alpha")
        eps = ViewerPlannerService.positive_finite_float(request.get("epsilon", epsilon), "epsilon")
        return cls.solve_record(record.copy(), base_record, budget, alpha, eps, planner_key=planner_key)

    @classmethod
    def serve_manifest(cls, payload, record, base_record, host, port,
                       solution_budget, window_alpha, epsilon):
        server = ViewerServer(
            payload, cls.planner_payload(solution_budget, window_alpha, epsilon),
            partial(cls.solve_request, record, base_record, solution_budget, window_alpha, epsilon),
        )
        server.serve(host, port)
