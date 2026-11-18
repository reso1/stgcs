from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from experiments.environment.env import Env
from experiments.environment.uav_village import VillageEnv
from experiments.manifests.base import BaseBenchmarkRecord, load_manifest
from experiments.base_serialization import SerializedBaseInstanceStore
from experiments.instance import GridInstance, Instance, Iris2DInstance, MazeInstance, Simple2DInstance


class BaseInstanceFactory:
    @staticmethod
    def _info(message: str) -> None:
        print(f"[stgcs-base] {message}")

    @staticmethod
    def _build_empty_square2d_env(env_params: Dict[str, Any]) -> Env:
        square_size = float(env_params.get("square_size", 10.0))
        robot_radius = float(env_params.get("robot_radius", Simple2DInstance.DEFAULT_ROBOT_RADIUS))
        if square_size <= 0.0:
            raise ValueError(f"empty-square2d square_size must be positive, got {square_size}.")
        if robot_radius <= 0.0:
            raise ValueError(f"empty-square2d robot_radius must be positive, got {robot_radius}.")
        vertices = np.asarray(
            [
                [0.0, 0.0],
                [square_size, 0.0],
                [square_size, square_size],
                [0.0, square_size],
            ],
            dtype=float,
        )
        return Env(
            name="empty-square2d",
            CSpace=[vertices],
            robot_radius=robot_radius,
        )

    @classmethod
    def _build_empty_square2d_instance(
        cls,
        env_params: Dict[str, Any],
        spatial_seed: int,
        compute_heuristics: bool,
    ):
        vlimit = float(env_params.get("vlimit", 1.0))
        tmax = float(env_params.get("tmax", 100.0))
        env = cls._build_empty_square2d_env(env_params)
        instance = Simple2DInstance.from_env(
            seed=spatial_seed,
            env=env,
            tmax=tmax,
            vlimit=vlimit,
            compute_heuristics=compute_heuristics,
        )
        instance.env_params = dict(env_params)
        return instance

    @staticmethod
    def _build_sphere_ball_env(env_params: Dict[str, Any]) -> Env:
        ball_radius = float(env_params.get("ball_radius", 1.0))
        robot_radius = float(env_params.get("robot_radius", 0.025))
        if ball_radius <= 0.0:
            raise ValueError(f"sphere-ball ball_radius must be positive, got {ball_radius}.")
        if robot_radius <= 0.0:
            raise ValueError(f"sphere-ball robot_radius must be positive, got {robot_radius}.")
        if robot_radius >= ball_radius:
            raise ValueError(
                f"sphere-ball robot_radius must be smaller than ball_radius, got {robot_radius} >= {ball_radius}."
            )

        r = ball_radius
        vertices = np.asarray(
            [
                [-r, -r, -r],
                [r, -r, -r],
                [r, r, -r],
                [-r, r, -r],
                [-r, -r, r],
                [r, -r, r],
                [r, r, r],
                [-r, r, r],
            ],
            dtype=float,
        )
        return Env(
            name="sphere-ball",
            CSpace=[vertices],
            robot_radius=robot_radius,
            domain_lb=np.asarray([-r, -r, -r], dtype=float),
            domain_ub=np.asarray([r, r, r], dtype=float),
        )

    @classmethod
    def _build_sphere_ball_instance(
        cls,
        env_params: Dict[str, Any],
        spatial_seed: int,
        compute_heuristics: bool,
    ):
        vlimit = float(env_params.get("vlimit", 0.5))
        tmax = float(env_params.get("tmax", 100.0))
        if vlimit <= 0.0:
            raise ValueError(f"sphere-ball vlimit must be positive, got {vlimit}.")
        if tmax <= 0.0:
            raise ValueError(f"sphere-ball tmax must be positive, got {tmax}.")

        env = cls._build_sphere_ball_env(env_params)
        stgcs = Instance.build_stgcs_from_env(env, tmax=tmax, vlimit=vlimit)
        instance = Instance(
            f"SPHERE-BALL-S{spatial_seed}",
            spatial_seed,
            env,
            stgcs,
        )
        instance.env_params = dict(env_params)
        if compute_heuristics:
            instance.compute_heuristics()
        return instance

    @classmethod
    def _build_fastpathplanning_village3d_instance(
        cls,
        env_params: Dict[str, Any],
        spatial_seed: int,
        compute_heuristics: bool,
    ):
        village_side = int(env_params.get("village_side", 8))
        village_height = float(env_params.get("village_height", 3.0))
        building_every = int(env_params.get("building_every", 3))
        decoration_density = float(env_params.get("decoration_density", 0.35))
        robot_radius = float(env_params.get("robot_radius", 0.2))
        vlimit = float(env_params.get("vlimit", 1.0))
        tmax = float(env_params.get("tmax", 100.0))
        building_clearance_margin = float(env_params.get("building_clearance_margin", 1e-3))
        if vlimit <= 0.0:
            raise ValueError(f"fastpathplanning-village3d vlimit must be positive, got {vlimit}.")
        if tmax <= 0.0:
            raise ValueError(f"fastpathplanning-village3d tmax must be positive, got {tmax}.")

        safe_boxes, static_obstacles, geometry_metadata = VillageEnv.build_village_geometry(
            village_side=village_side,
            village_height=village_height,
            building_every=building_every,
            decoration_density=decoration_density,
            robot_radius=robot_radius,
            building_clearance_margin=building_clearance_margin,
            seed=int(spatial_seed),
        )
        record = VillageEnv.build_record(
            safe_boxes=safe_boxes,
            village_side=village_side,
            village_height=village_height,
            building_every=building_every,
            decoration_density=decoration_density,
            building_clearance_margin=building_clearance_margin,
            num_uavs=int(env_params.get("num_uavs", 48)),
            robot_radius=robot_radius,
            vlimit=vlimit,
            seed=int(spatial_seed),
            geometry_metadata=geometry_metadata,
        )
        instance = VillageEnv.build_planning_instance(
            record,
            safe_boxes,
            robot_radius=robot_radius,
            vlimit=vlimit,
            tmax=tmax,
            build_heuristics=compute_heuristics,
        )
        instance.name = record["base_instance_id"]
        instance.env_params = dict(env_params)
        instance.safe_boxes = safe_boxes
        instance.static_obstacles = static_obstacles
        instance.geometry_metadata = geometry_metadata
        return instance

    @staticmethod
    def build(
        domain: str,
        env_params: Dict[str, Any],
        spatial_seed: int,
        space_dim: int,
        compute_heuristics: bool = True,
    ):
        if domain == "grid":
            return GridInstance.generate_base(
                seed=spatial_seed,
                N=int(env_params["N"]),
                M=int(env_params["M"]),
                tmax=float(env_params.get("tmax", 100.0)),
                compute_heuristics=compute_heuristics,
                space_dim=space_dim,
                spatial_scale=float(env_params.get("spatial_scale", 1.0)),
            )
        if domain == "maze":
            vlimit = float(env_params.get("vlimit", 1.0))
            robot_radius = float(env_params.get("robot_radius", 0.25))
            return MazeInstance.generate_base(
                seed=spatial_seed,
                width=int(env_params["width"]),
                height=int(env_params["height"]),
                tmax=float(env_params.get("tmax", 100.0)),
                vlimit=vlimit,
                robot_radius=robot_radius,
                spatial_scale=float(env_params.get("spatial_scale", 1.0)),
                compute_heuristics=compute_heuristics,
            )
        if domain == "simple2d":
            vlimit = float(env_params.get("vlimit", 1.0))
            robot_radius = float(env_params.get("robot_radius", Simple2DInstance.DEFAULT_ROBOT_RADIUS))
            return Simple2DInstance.generate_base(
                seed=spatial_seed,
                tmax=float(env_params.get("tmax", 100.0)),
                vlimit=vlimit,
                robot_radius=robot_radius,
                compute_heuristics=compute_heuristics,
            )
        if domain == "empty-square2d":
            return BaseInstanceFactory._build_empty_square2d_instance(
                env_params=env_params,
                spatial_seed=spatial_seed,
                compute_heuristics=compute_heuristics,
            )
        if domain == "sphere-ball":
            return BaseInstanceFactory._build_sphere_ball_instance(
                env_params=env_params,
                spatial_seed=spatial_seed,
                compute_heuristics=compute_heuristics,
            )
        if domain == "fastpathplanning-village3d":
            return BaseInstanceFactory._build_fastpathplanning_village3d_instance(
                env_params=env_params,
                spatial_seed=spatial_seed,
                compute_heuristics=compute_heuristics,
            )
        if domain == "iris-2d":
            return Iris2DInstance.generate_base(
                seed=spatial_seed,
                tmax=float(env_params.get("tmax", 100.0)),
                env_params=env_params,
                compute_heuristics=compute_heuristics,
            )
        raise ValueError(f"Unsupported domain {domain!r}")

    @classmethod
    def load_cached_instance(cls, record: BaseBenchmarkRecord):
        return SerializedBaseInstanceStore.load_instance(record)

    @classmethod
    def write_cached_instance(cls, record: BaseBenchmarkRecord, instance) -> None:
        SerializedBaseInstanceStore.write_instance(record, instance)

    @staticmethod
    def _ensure_requested_heuristics(instance, compute_heuristics: bool):
        if compute_heuristics:
            instance.compute_heuristics()
        return instance

    @classmethod
    def from_record(cls, record: BaseBenchmarkRecord, compute_heuristics: bool = True):
        cached_instance = cls.load_cached_instance(record)
        if cached_instance is not None:
            return cls._ensure_requested_heuristics(cached_instance, compute_heuristics)

        if record.domain == "iris-2d" and record.env_params.get("region_generation") != "vcc-iris":
            raise ValueError(
                f"{record.instance_id}: legacy IRIS cache is missing; regenerate the base manifest "
                "with VCC/IRIS instead of rebuilding a different graph under the old record."
            )

        object_path = SerializedBaseInstanceStore.object_path(record)
        if object_path is not None:
            cls._info(
                f"{record.instance_id}: serialized base object missing at {object_path}; "
                "constructing online and saving it."
            )
        instance = cls.build(
            domain=record.domain,
            env_params=record.env_params,
            spatial_seed=record.spatial_seed,
            space_dim=record.space_dim,
            compute_heuristics=False,
        )
        cls.write_cached_instance(record, instance)
        return cls._ensure_requested_heuristics(instance, compute_heuristics)


class BaseManifestStore:
    _records_by_manifest: Dict[Path, Dict[str, BaseBenchmarkRecord]] = {}

    @staticmethod
    def manifest_path(source: str | Path, domain_key: str | None = None) -> Path:
        source_path = Path(source).resolve()
        if source_path.suffix == ".json":
            return source_path
        if domain_key is None:
            raise ValueError("domain_key is required when resolving a base manifest from a directory.")
        if source_path.name == domain_key:
            return source_path / "manifest.json"
        return source_path / domain_key / "manifest.json"

    @staticmethod
    def default_base_root(benchmark_manifest_path: str | Path) -> Path:
        manifest_path = Path(benchmark_manifest_path).resolve()
        if manifest_path.parent.name == "st_planning":
            return manifest_path.parent / "base"
        parts = manifest_path.parts
        for idx, part in enumerate(parts[:-1]):
            if part == "data" and idx + 1 < len(parts) and parts[idx + 1] == "instances":
                return Path(*parts[: idx + 1]) / "stgcs_base"
        if manifest_path.parent.parent.name == "mrmp":
            return manifest_path.parent.parent.parent / "stgcs_base"
        if manifest_path.parent.parent.name in {"base", "stgcs_base"}:
            return manifest_path.parent.parent
        raise ValueError(f"Unable to infer base root from {manifest_path}")

    @staticmethod
    def domain_key_for_benchmark_manifest(benchmark_manifest_path: str | Path) -> str:
        return Path(benchmark_manifest_path).resolve().parent.name

    @classmethod
    def manifest_path_for_benchmark(
        cls,
        benchmark_manifest_path: str | Path,
        base_root: str | Path | None = None,
    ) -> Path:
        domain_key = cls.domain_key_for_benchmark_manifest(benchmark_manifest_path)
        root = cls.default_base_root(benchmark_manifest_path) if base_root is None else Path(base_root).resolve()
        return cls.manifest_path(root, domain_key=domain_key)

    @classmethod
    def records_by_id(cls, manifest_path: str | Path) -> Dict[str, BaseBenchmarkRecord]:
        target = Path(manifest_path).resolve()
        if target not in cls._records_by_manifest:
            cls._records_by_manifest[target] = {
                record.instance_id: record
                for record in load_manifest(target)
            }
        return cls._records_by_manifest[target]

    @classmethod
    def record_by_id(cls, manifest_path: str | Path, instance_id: str) -> BaseBenchmarkRecord:
        try:
            return cls.records_by_id(manifest_path)[instance_id]
        except KeyError as exc:
            raise KeyError(f"Unknown base instance_id {instance_id!r} in {Path(manifest_path).resolve()}") from exc
