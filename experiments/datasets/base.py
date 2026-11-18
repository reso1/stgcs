from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

from experiments.environment.iris import Iris2DEnvBuilder

from experiments.base import BaseInstanceFactory
from experiments.manifests.base import BaseBenchmarkRecord, load_manifest, save_manifest


class ManifestLockError(RuntimeError):
    pass


class ManifestLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    @staticmethod
    def _pid_is_alive(pid_text: str) -> bool:
        try:
            pid = int(pid_text)
        except ValueError:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"{os.getpid()}\n"
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError as exc:
                owner = self.path.read_text().strip() if self.path.exists() else "unknown"
                if owner and not self._pid_is_alive(owner):
                    self.path.unlink(missing_ok=True)
                    continue
                raise ManifestLockError(
                    f"Manifest build already running for {self.path.parent.name}; lock held by pid {owner}."
                ) from exc
        os.write(self.fd, payload.encode("utf-8"))

    def release(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> "ManifestLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.release()


class BaseManifestBuilder:
    MAX_FAILED_ATTEMPTS = 1000

    @staticmethod
    def _load_existing_records(manifest_path: Path | None) -> list[BaseBenchmarkRecord]:
        if manifest_path is None:
            return []
        target = Path(manifest_path)
        if not target.exists():
            return []
        return load_manifest(target)

    @staticmethod
    def _supports_sampling(domain: str) -> bool:
        return domain in {"grid", "maze", "simple2d", "empty-square2d", "iris-2d"}

    @classmethod
    def _record(
        cls,
        instance_id: str,
        domain: str,
        space_dim: int,
        spatial_seed: int,
        env_params: Dict[str, Any],
        instance,
        manifest_path: Path | None = None,
    ) -> BaseBenchmarkRecord:
        return BaseBenchmarkRecord(
            instance_id=instance_id,
            domain=domain,
            space_dim=space_dim,
            spatial_seed=spatial_seed,
            env_params=dict(env_params),
            supports_sampling=cls._supports_sampling(domain),
            stgcs_num_vertices=int(instance.stgcs.G.number_of_nodes()),
            stgcs_num_edges=int(instance.stgcs.G.number_of_edges()),
            manifest_path=None if manifest_path is None else Path(manifest_path).resolve(),
        )


class GridBaseManifestBuilder(BaseManifestBuilder):
    DEFAULT_COUNT_PER_SIZE = 100

    @staticmethod
    def _instance_id(space_dim: int, size: int, seed: int, spatial_scale: float = 1.0) -> str:
        name = f"base-grid{space_dim}d-s{size}-seed{seed:05d}"
        return name if spatial_scale == 1.0 else name + "-scale" + f"{spatial_scale:g}".replace(".", "p")

    @classmethod
    def size_targets(
        cls,
        sizes: Sequence[int],
        count_per_size: int | None = None,
        count_total: int | None = None,
    ) -> Dict[int, int]:
        if count_total is not None:
            if count_total < 0:
                raise ValueError(f"Grid base total count must be non-negative, got {count_total}")
            base = count_total // len(sizes)
            remainder = count_total % len(sizes)
            targets = {int(size): base for size in sizes}
            for size in sizes[:remainder]:
                targets[int(size)] += 1
            return targets
        if count_per_size is None:
            count_per_size = cls.DEFAULT_COUNT_PER_SIZE
        if count_per_size < 0:
            raise ValueError(f"Grid base count per size must be non-negative, got {count_per_size}")
        return {int(size): int(count_per_size) for size in sizes}

    @classmethod
    def build_records(
        cls,
        space_dim: int,
        sizes: Sequence[int],
        count_per_size: int | None,
        count_total: int | None,
        seed_start: int,
        manifest_path: Path | None = None,
        spatial_scale: float = 1.0,
    ) -> list[BaseBenchmarkRecord]:
        records: list[BaseBenchmarkRecord] = []
        if not 0 < spatial_scale < float("inf"):
            raise ValueError("spatial_scale must be finite and positive.")
        size_targets = cls.size_targets(sizes=sizes, count_per_size=count_per_size, count_total=count_total)
        for size in sizes:
            accepted = 0
            failed_attempts = 0
            seed = int(seed_start)
            env_params = {"N": int(size), "M": int(size), "size": int(size)}
            if spatial_scale != 1.0:
                env_params["spatial_scale"] = float(spatial_scale)
            while accepted < int(size_targets[int(size)]):
                try:
                    instance = BaseInstanceFactory.build(
                        domain="grid",
                        env_params=env_params,
                        spatial_seed=seed,
                        space_dim=space_dim,
                        compute_heuristics=False,
                    )
                except Exception:
                    failed_attempts += 1
                    seed += 1
                    if failed_attempts >= cls.MAX_FAILED_ATTEMPTS:
                        raise RuntimeError(f"Unable to build enough grid{space_dim}d bases for size={size}.")
                    continue

                record = cls._record(
                    instance_id=cls._instance_id(space_dim, int(size), seed, spatial_scale),
                    domain="grid",
                    space_dim=space_dim,
                    spatial_seed=seed,
                    env_params=env_params,
                    instance=instance,
                    manifest_path=manifest_path,
                )
                BaseInstanceFactory.write_cached_instance(record, instance)
                records.append(record)
                accepted += 1
                seed += 1
                failed_attempts = 0
        return records


class MazeBaseManifestBuilder(BaseManifestBuilder):
    @staticmethod
    def _instance_id(width: int, height: int, seed: int, spatial_scale: float) -> str:
        scale = f"{spatial_scale:g}".replace(".", "p")
        return f"base-maze-{width}x{height}-seed{seed:05d}-scale{scale}"

    @classmethod
    def build_records(
        cls,
        count_total: int,
        width: int,
        height: int,
        seed_start: int,
        manifest_path: Path | None = None,
        spatial_scale: float = 0.25,
    ) -> list[BaseBenchmarkRecord]:
        records: list[BaseBenchmarkRecord] = []
        failed_attempts = 0
        seed = int(seed_start)
        if not 0 < spatial_scale < float("inf"):
            raise ValueError("spatial_scale must be finite and positive.")
        env_params = {"width": int(width), "height": int(height),
                      "spatial_scale": float(spatial_scale), "robot_radius": 0.25 * spatial_scale}
        while len(records) < int(count_total):
            try:
                instance = BaseInstanceFactory.build(
                    domain="maze",
                    env_params=env_params,
                    spatial_seed=seed,
                    space_dim=2,
                    compute_heuristics=False,
                )
            except Exception:
                failed_attempts += 1
                seed += 1
                if failed_attempts >= cls.MAX_FAILED_ATTEMPTS:
                    raise RuntimeError("Unable to build enough maze bases.")
                continue

            record = cls._record(
                instance_id=cls._instance_id(int(width), int(height), seed, spatial_scale),
                domain="maze",
                space_dim=2,
                spatial_seed=seed,
                env_params=env_params,
                instance=instance,
                manifest_path=manifest_path,
            )
            BaseInstanceFactory.write_cached_instance(record, instance)
            records.append(record)
            seed += 1
            failed_attempts = 0
        return records


class Iris2DBaseManifestBuilder(BaseManifestBuilder):
    MAX_STGCS_EDGES = 50

    @staticmethod
    def _instance_id(m: int, seed: int, square_size: float) -> str:
        size = f"{square_size:g}".replace(".", "p")
        return f"base-iris-2d-m{m}-s{size}-seed{seed:05d}"

    @staticmethod
    def save_progress(manifest_path: Path | None, records: Sequence[BaseBenchmarkRecord]) -> None:
        if manifest_path is not None:
            save_manifest(manifest_path, list(records))

    @classmethod
    def _validate_existing_records(
        cls,
        records: Sequence[BaseBenchmarkRecord],
        resolved_env_params: Dict[str, Any],
    ) -> None:
        seen_ids: set[str] = set()
        for record in records:
            if record.instance_id in seen_ids:
                raise ValueError(f"Duplicate iris-2d base instance_id {record.instance_id!r}.")
            seen_ids.add(record.instance_id)
            if record.domain != "iris-2d" or record.space_dim != 2:
                raise ValueError(
                    f"Existing record {record.instance_id!r} is not an iris-2d base record."
                )
            if record.env_params.get("region_generation") != "vcc-iris":
                raise ValueError(
                    "Existing iris-2d records were not generated with VCC/IRIS; "
                    "use a new output root or --replace to regenerate them."
                )
            if Iris2DEnvBuilder.normalize_env_params(record.env_params) != resolved_env_params:
                raise ValueError(
                    f"Existing record {record.instance_id!r} was built with different iris-2d env params."
                )

    @classmethod
    def build_records(
        cls,
        count_total: int,
        seed_start: int,
        env_params: Dict[str, Any],
        manifest_path: Path | None = None,
        replace_existing: bool = False,
        on_status: Callable[[str, Dict[str, Any]], None] | None = None,
    ) -> list[BaseBenchmarkRecord]:
        records = [] if replace_existing else cls._load_existing_records(manifest_path)
        failed_attempts = 0
        resolved_env_params = Iris2DEnvBuilder.normalize_env_params(env_params)
        cls._validate_existing_records(records, resolved_env_params)
        seed = max(
            [int(seed_start)]
            + [int(record.spatial_seed) + 1 for record in records]
        )
        if on_status is not None:
            on_status(
                "start",
                {
                    "accepted": len(records),
                    "target": int(count_total),
                    "seed": seed,
                    "env_params": resolved_env_params,
                },
            )
        if len(records) < int(count_total):
            cls.save_progress(manifest_path, records)
        while len(records) < int(count_total):
            if on_status is not None:
                on_status(
                    "attempt",
                    {
                        "accepted": len(records),
                        "target": int(count_total),
                        "seed": seed,
                    },
                )
            try:
                instance = BaseInstanceFactory.build(
                    domain="iris-2d",
                    env_params=resolved_env_params,
                    spatial_seed=seed,
                    space_dim=2,
                    compute_heuristics=False,
                )
            except Exception as exc:
                failed_attempts += 1
                failed_seed = seed
                seed += 1
                if on_status is not None:
                    on_status(
                        "skip",
                        {
                            "accepted": len(records),
                            "target": int(count_total),
                            "seed": failed_seed,
                            "failed_attempts": failed_attempts,
                            "reason": str(exc),
                        },
                    )
                if failed_attempts >= cls.MAX_FAILED_ATTEMPTS:
                    raise RuntimeError("Unable to build enough iris-2d bases.")
                continue

            stgcs_num_edges = int(instance.stgcs.G.number_of_edges())
            if stgcs_num_edges > cls.MAX_STGCS_EDGES:
                failed_attempts += 1
                rejected_seed = seed
                seed += 1
                if on_status is not None:
                    on_status(
                        "skip",
                        {
                            "accepted": len(records),
                            "target": int(count_total),
                            "seed": rejected_seed,
                            "failed_attempts": failed_attempts,
                            "reason": f"|E|={stgcs_num_edges} exceeds {cls.MAX_STGCS_EDGES}",
                        },
                    )
                if failed_attempts >= cls.MAX_FAILED_ATTEMPTS:
                    raise RuntimeError("Unable to build enough iris-2d bases.")
                continue

            record = cls._record(
                instance_id=cls._instance_id(
                    int(resolved_env_params["m"]),
                    seed,
                    float(resolved_env_params["square_size"]),
                ),
                domain="iris-2d",
                space_dim=2,
                spatial_seed=seed,
                env_params=resolved_env_params,
                instance=instance,
                manifest_path=manifest_path,
            )
            BaseInstanceFactory.write_cached_instance(record, instance)
            records.append(record)
            cls.save_progress(manifest_path, records)
            if on_status is not None:
                on_status(
                    "accepted",
                    {
                        "accepted": len(records),
                        "target": int(count_total),
                        "seed": seed,
                        "instance_id": record.instance_id,
                        "stgcs_num_vertices": record.stgcs_num_vertices,
                        "stgcs_num_edges": record.stgcs_num_edges,
                    },
                )
            seed += 1
            failed_attempts = 0
        return records


class BaseManifestCLI:
    DEFAULT_COUNT_TOTAL = 500

    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Build shared base ST-GCS manifests.")
        parser.add_argument("domain", choices=("grid2d", "grid3d", "maze", "iris-2d"))
        parser.add_argument("--output-root", type=Path, default=Path("data/stgcs_base"))
        parser.add_argument("--seed-start", type=int, default=0)
        parser.add_argument("--count-per-size", type=int, default=100)
        parser.add_argument("--count-total", type=int, default=None)
        parser.add_argument("--sizes", type=int, nargs="*", default=None)
        parser.add_argument("--width", type=int, default=20)
        parser.add_argument("--height", type=int, default=20)
        parser.add_argument("--maze-scale", type=float, default=0.25)
        parser.add_argument("--grid-scale", type=float, default=1.0)
        parser.add_argument("--m", type=int, default=Iris2DEnvBuilder.DEFAULT_PARAMS["m"])
        parser.add_argument("--square-size", type=float, default=Iris2DEnvBuilder.DEFAULT_PARAMS["square_size"])
        parser.add_argument("--min-obstacle-radius", type=float, default=Iris2DEnvBuilder.DEFAULT_PARAMS["min_obstacle_radius"])
        parser.add_argument("--max-obstacle-radius", type=float, default=Iris2DEnvBuilder.DEFAULT_PARAMS["max_obstacle_radius"])
        parser.add_argument("--iris-iterations", type=int, default=Iris2DEnvBuilder.DEFAULT_PARAMS["iteration_limit"])
        parser.add_argument("--coverage-threshold", type=float, default=0.8)
        parser.add_argument(
            "--iris-max-attempts", type=int, default=5000,
            help="Maximum candidate-point draws per VCC visibility round.",
        )
        parser.add_argument("--vcc-max-iterations", type=int, default=10)
        parser.add_argument("--visibility-samples", type=int, default=200)
        parser.add_argument("--minimum-clique-size", type=int, default=3)
        parser.add_argument("--coverage-samples", type=int, default=500)
        parser.add_argument(
            "--replace",
            action="store_true",
            help="For iris-2d, rebuild records from seed-start instead of resuming the existing manifest.",
        )
        return parser.parse_args()

    @staticmethod
    def _print_iris_status(event: str, payload: Dict[str, Any]) -> None:
        if event == "start":
            params = payload["env_params"]
            print(
                "[iris-2d] target={target} existing={accepted} start_seed={seed:05d} "
                "clearance={clearance:g} robot_radius={robot_radius:g} max_attempts={max_attempts}".format(
                    target=payload["target"],
                    accepted=payload["accepted"],
                    seed=payload["seed"],
                    clearance=float(params["clearance"]),
                    robot_radius=float(params["robot_radius"]),
                    max_attempts=int(params["max_attempts"]),
                ),
                flush=True,
            )
        elif event == "attempt":
            print(
                "[iris-2d] trying seed{seed:05d} ({accepted}/{target} accepted)".format(
                    seed=payload["seed"],
                    accepted=payload["accepted"],
                    target=payload["target"],
                ),
                flush=True,
            )
        elif event == "accepted":
            print(
                "[iris-2d] accepted seed{seed:05d} -> {instance_id} "
                "({accepted}/{target}; |V|={vertices}, |E|={edges})".format(
                    seed=payload["seed"],
                    instance_id=payload["instance_id"],
                    accepted=payload["accepted"],
                    target=payload["target"],
                    vertices=payload["stgcs_num_vertices"],
                    edges=payload["stgcs_num_edges"],
                ),
                flush=True,
            )
        elif event == "skip":
            reason = str(payload["reason"]).splitlines()[0]
            print(
                "[iris-2d] skipped seed{seed:05d} ({accepted}/{target} accepted): {reason}".format(
                    seed=payload["seed"],
                    accepted=payload["accepted"],
                    target=payload["target"],
                    reason=reason,
                ),
                flush=True,
            )

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        output_dir = Path(args.output_root).resolve() / args.domain
        manifest_path = output_dir / "manifest.json"
        lock_path = output_dir / ".build_manifest.lock"

        with ManifestLock(lock_path):
            if args.domain == "grid2d":
                records = GridBaseManifestBuilder.build_records(
                    space_dim=2,
                    sizes=(5,) if args.sizes is None else tuple(args.sizes),
                    count_per_size=int(args.count_per_size),
                    count_total=None if args.count_total is None else int(args.count_total),
                    seed_start=int(args.seed_start),
                    manifest_path=manifest_path,
                    spatial_scale=float(args.grid_scale),
                )
            elif args.domain == "grid3d":
                records = GridBaseManifestBuilder.build_records(
                    space_dim=3,
                    sizes=(1, 2, 3) if args.sizes is None else tuple(args.sizes),
                    count_per_size=int(args.count_per_size),
                    count_total=None if args.count_total is None else int(args.count_total),
                    seed_start=int(args.seed_start),
                    manifest_path=manifest_path,
                    spatial_scale=float(args.grid_scale),
                )
            elif args.domain == "maze":
                records = MazeBaseManifestBuilder.build_records(
                    count_total=cls.DEFAULT_COUNT_TOTAL if args.count_total is None else int(args.count_total),
                    width=int(args.width),
                    height=int(args.height),
                    spatial_scale=float(args.maze_scale),
                    seed_start=int(args.seed_start),
                    manifest_path=manifest_path,
                )
            else:
                records = Iris2DBaseManifestBuilder.build_records(
                    count_total=cls.DEFAULT_COUNT_TOTAL if args.count_total is None else int(args.count_total),
                    seed_start=int(args.seed_start),
                    env_params={
                        "square_size": float(args.square_size),
                        "m": int(args.m),
                        "min_obstacle_radius": float(args.min_obstacle_radius),
                        "max_obstacle_radius": float(args.max_obstacle_radius),
                        "iteration_limit": int(args.iris_iterations),
                        "coverage_threshold": float(args.coverage_threshold),
                        "max_attempts": int(args.iris_max_attempts),
                        "max_iterations": int(args.vcc_max_iterations),
                        "visibility_samples": int(args.visibility_samples),
                        "minimum_clique_size": int(args.minimum_clique_size),
                        "coverage_samples": int(args.coverage_samples),
                    },
                    manifest_path=manifest_path,
                    replace_existing=bool(args.replace),
                    on_status=cls._print_iris_status,
                )

            if args.domain != "iris-2d":
                save_manifest(manifest_path, records)
        print(f"Wrote {len(records)} base records to {manifest_path}")


if __name__ == "__main__":
    BaseManifestCLI.main()
