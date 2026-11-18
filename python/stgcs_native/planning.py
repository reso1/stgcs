"""ST and MRMP planning through the ST-GCS C++ executable."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import struct

import numpy as np

from .client import NativeClient
from .types import PlanningProblem, STTrajectory, MRMPResultEntry, SearchPlannerSpec
from tempfile import TemporaryDirectory, mkdtemp
from types import SimpleNamespace
import math


class NativeBackend(NativeClient):
    CACHE_MAGIC = b"STGCACHE"

    def __init__(self, executable: Path, work_dir: Path, cache_root: Path | None = None):
        self.executable = Path(executable).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cache_root = Path(cache_root).resolve() if cache_root is not None else self.work_dir / "cache"

    @staticmethod
    def write_problem(path: Path, problem: PlanningProblem, query, obstacles) -> None:
        regions = problem.regions
        words = [len(query.start), len(regions), problem.horizon, query.vlimit, query.t_start,
                 int(query.is_stay), *query.start, *query.goal]
        for region in regions:
            words.extend([len(region.b), *np.asarray(region.A).ravel(), *region.b])
        words.append(len(obstacles))
        for obstacle in obstacles:
            words.extend([obstacle["radius"] + problem.robot_radius, len(obstacle["segments"])])
            for segment in obstacle["segments"]:
                words.extend([*segment["start"], segment["t_start"],
                              *segment["goal"], segment["t_end"]])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(" ".join(map(str, words)) + "\n")

    @staticmethod
    def bind_cache(input_path: Path, base_instance_id: str | None, domain: str) -> None:
        for name in (base_instance_id, domain):
            if name is not None and (not name or Path(name).name != name or name in (".", "..")):
                raise ValueError(f"Invalid base cache name: {name}")
        input_path.with_suffix(".base.json").write_text(json.dumps(
            dict(base_instance_id=base_instance_id, domain=domain)))

    @staticmethod
    def base_input(input_path: Path) -> str:
        """Canonical base graph input, independent of queries and reservations."""
        words = input_path.read_text().split()
        dimension, count = int(words[0]), int(words[1])
        if dimension < 1 or count < 1:
            raise ValueError("Invalid base graph dimensions")
        base = [str(dimension), str(count), str(float(words[2])), str(float(words[3])),
                "0", "1", *(["0.0"] * (2 * dimension))]
        offset = 6 + 2 * dimension
        for _ in range(count):
            rows = int(words[offset])
            size = rows * (dimension + 1)
            values = words[offset + 1:offset + 1 + size]
            if rows < 1 or len(values) != size:
                raise ValueError("Truncated base region")
            base.extend([str(rows), *(str(float(value)) for value in values)])
            offset += 1 + size
        return " ".join([*base, "0"]) + "\n"

    def cache_path(self, input_path: Path, *, cost_table: bool = True) -> Path:
        record = json.loads(input_path.with_suffix(".base.json").read_text())
        filename = "heuristics.bin" if cost_table else "triplets.bin"
        if record["base_instance_id"] is None:
            return input_path.parent / filename
        return self.cache_root / record["domain"] / "heuristics" / record["base_instance_id"] / filename

    @classmethod
    def read_cache_tables(cls, path: Path) -> dict:
        with path.open("rb") as stream:
            header = stream.read(struct.calcsize("<8sBBdd"))
            if len(header) != struct.calcsize("<8sBBdd"):
                raise ValueError(f"Truncated native cache: {path}")
            magic, complete, has_table, h_tri, h_tab = struct.unpack("<8sBBdd", header)
            if magic != cls.CACHE_MAGIC or complete != 1 or has_table not in (0, 1):
                raise ValueError(f"Incomplete or invalid native cache: {path}")
            if not all(math.isfinite(t) and t >= 0 for t in (h_tri, h_tab)):
                raise ValueError(f"Invalid cache timings: {path}")
            tables = dict(h_tri=h_tri, h_tab=h_tab, has_table=bool(has_table))
            for name, width in (("pairs", 3), ("incoming", 3), ("sets", 2), ("table", 2), ("triplets", 3)):
                encoded_count = stream.read(8)
                if len(encoded_count) != 8:
                    raise ValueError(f"Truncated native cache: {path}")
                count, = struct.unpack("<Q", encoded_count)
                if count > 100000000:
                    raise ValueError(f"Invalid cache map size: {path}")
                encoding = "<" + "i" * width + "d"
                encoded = stream.read(count * struct.calcsize(encoding))
                if len(encoded) != count * struct.calcsize(encoding):
                    raise ValueError(f"Truncated native cache: {path}")
                entries = struct.iter_unpack(encoding, encoded)
                values = {tuple(row[:-1]): row[-1] for row in entries}
                if len(values) != count or any(not np.isfinite(v) or v < 0 for v in values.values()):
                    raise ValueError(f"Invalid cache map: {path}")
                tables[name] = values
            if stream.read(1):
                raise ValueError(f"Trailing cache data: {path}")
        return tables

    def inspect(self, input_path: Path) -> dict:
        result = self.run_process([str(self.executable), "inspect", str(input_path)],
                                  60, input_path.with_suffix(".inspect.log"))
        if not {"st_vertices", "st_edges"} <= result.keys():
            raise RuntimeError(f"Native graph inspection failed: {input_path.stem}")
        return result


    def cache(self, input_path: Path, timeout: float = 10000, *, cost_table: bool = True) -> dict:
        path = self.cache_path(input_path, cost_table=cost_table)
        timing = path.with_suffix(".json")
        base_path = path.with_suffix(".input")
        path.parent.mkdir(parents=True, exist_ok=True)
        base = self.base_input(input_path)
        with path.with_suffix(".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                if base_path.read_text() != base:
                    raise ValueError(f"Cached geometry differs from the requested base: {path}")
                self.read_cache_tables(path)
                data = json.loads(timing.read_text())
                if not data["complete"]:
                    raise ValueError(f"Incomplete native cache: {path}")
            else:
                base_path.write_text(base)
                temp = path.with_suffix(".tmp")
                command = [str(self.executable), "cache", str(base_path), str(temp), str(timeout)]
                if not cost_table:
                    command.append("0")
                data = self.run_process(command, timeout + 30,
                    self.work_dir / "cache-logs" / path.parent.name / f"{path.stem}.log")
                if not data.get("complete"):
                    raise RuntimeError(f"Incomplete native cache: {input_path.stem}")
                self.read_cache_tables(temp)
                data = {k: v for k, v in data.items() if k not in ("st_vertices", "st_edges")}
                temp.replace(path)
                receipt = timing.with_suffix(".json.tmp")
                receipt.write_text(json.dumps(data, indent=2))
                receipt.replace(timing)
        return {**data, **self.inspect(input_path)}


    def validate_witness(self, input_path: Path, segments) -> dict:
        path = self.work_dir / "witnesses" / input_path.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(len(segments)) + "\n" + "\n".join(
            " ".join(map(str, segment)) for segment in segments) + "\n")
        result = self.run_process(
            [str(self.executable), "validate", str(input_path), str(path)],
            60, path.with_suffix(".log"))
        if not result.get("valid"):
            raise RuntimeError(f"Native witness validation failed: {input_path.stem}")
        return result

    def command(self, input_path: Path, spec, budget: float, method: str, seed: int, rounded: int):
        dominance = next((name for name in spec.dominance_checks if name != "GUB"), "none")
        needs_cache = method == "search" and (spec.heuristic not in {"h_zero", "h_mot"}
                                             or spec.upper_bound_heuristic == "h_max")
        cache = str(self.cache_path(input_path)) if needs_cache else "-"
        return [str(self.executable), "plan", str(input_path), cache,
                spec.heuristic, dominance, str(spec.epsilon), str(budget), method, str(seed), str(rounded),
                spec.key_evaluation, spec.upper_bound_heuristic or "same"]

    def plan_single(self, problem, query, obstacles=(), *, spec=None, budget=60.0,
                    method="search", seed=0, rounded=100, base_id=None, domain="problems"):
        spec = spec or SearchPlannerSpec("h_mot", ("GUB",))
        with TemporaryDirectory(dir=self.work_dir) as staging:
            path = Path(staging) / "input.txt"
            self.write_problem(path, problem, query, obstacles)
            self.bind_cache(path, base_id, domain)
            if method == "search" and (spec.heuristic not in {"h_zero", "h_mot"}
                                       or spec.upper_bound_heuristic == "h_max"):
                self.cache(path)
            result = self.run_process(self.command(path, spec, budget, method, seed, rounded),
                                      None if math.isinf(budget) else budget + 30, Path(staging) / "plan.log")
        segments = result.get("segments", [])
        solution = (STTrajectory([f"native:{i}" for i in range(len(segments))],
                                 [np.asarray(segment).ravel() for segment in segments], len(query.start))
                    if result["success"] else None)
        return solution, result



class MRMPNativeBackend(NativeBackend):
    def prepare_multi(self, problem: PlanningProblem, queries, *, base_id=None, domain="problems",
                      compute_cache=True, cost_table=True):
        if not queries:
            raise ValueError("An MRMP instance needs at least one robot.")
        if any(not math.isclose(query.vlimit, problem.velocity_limit) for query in queries):
            raise ValueError("Native MRMP requires a shared velocity limit.")
        # The single-query fields are unused by the MRMP/cache commands. Canonical
        # values let n10 and n20 reuse the same native base-geometry heuristic cache.
        zero = np.zeros(len(queries[0].start))
        query = SimpleNamespace(start=zero, goal=zero, t_start=0, is_stay=True,
                                vlimit=problem.velocity_limit)
        inputs = self.work_dir / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        request_dir = Path(mkdtemp(prefix=f"{base_id or 'problem'}-", dir=inputs))
        path = request_dir / "base.txt"
        self.write_problem(path, problem, query, [])
        self.bind_cache(path, base_id, domain)
        if compute_cache:
            self.cache(path, cost_table=cost_table)
        words = [len(queries), problem.robot_radius]
        for item in queries:
            words.extend([item.t_start, int(item.is_stay), *item.start, *item.goal])
        content = " ".join(map(str, words)) + "\n"
        query_path = path.with_name("queries.txt")
        query_path.write_text(content)
        return path, query_path

    @staticmethod
    def trajectories(result, dimension):
        return [STTrajectory([f"native:{i}" for i in range(len(segments))],
                             [np.asarray(segment).ravel() for segment in segments], dimension)
                for segments in result.get("trajectories", [])]

    def plan_multi(self, problem, queries, spec, budget, child_rule,
                           window_span_factor=0.0, execution_horizon_factor=1.0,
                           dynamic_window_adjustment=False, method="pbs", *, base_id=None, domain="problems"):
        if method not in {"pbs", "pp"}:
            raise ValueError(f"Unknown MRMP coordination method: {method}")
        if spec.exact_astar or "GUB" not in spec.dominance_checks:
            raise ValueError("Native MRMP requires the Search + GUB specification.")
        dominance = [name for name in spec.dominance_checks if name != "GUB"]
        if len(dominance) > 1:
            raise ValueError("MRMP native adapter accepts at most one dominance policy.")
        cost_table = spec.heuristic != "h_mot_tri"
        path, query_path = self.prepare_multi(
            problem, queries, base_id=base_id, domain=domain, compute_cache=spec.heuristic not in {"h_zero", "h_mot"},
            cost_table=cost_table)
        if window_span_factor == math.inf:
            window_span_factor = 0.0
            dynamic_window_adjustment = False
        command = [str(self.executable), "mrmp", str(path), str(self.cache_path(path, cost_table=cost_table)),
                   spec.heuristic, dominance[0] if dominance else "none", str(spec.epsilon),
                   str(budget), method, child_rule, str(window_span_factor),
                   str(execution_horizon_factor), str(int(dynamic_window_adjustment)), str(query_path)]
        log = self.work_dir / "logs" / f"{path.parent.name}-{method}.log"
        result = self.run_process(command, budget + 30, log)
        within_budget = bool(result["success"]) and float(result["runtime"]) <= budget
        log.with_suffix(".json").write_text(json.dumps(
            dict(command=command, budget=budget, within_budget=within_budget, result=result), indent=2))
        trajectories = self.trajectories(result, len(queries[0].start))
        # A killed worker has no complete profile; keep missing telemetry explicit.
        unavailable = np.array([-1, math.nan])
        profile = result.get("profile", {})
        return trajectories, MRMPResultEntry.from_statistics(
            solutions=trajectories, queries=queries, is_success=within_budget,
            runtime=float(result["runtime"]),
            pbs_calls=int(result.get("coordination_calls", -1)),
            pbs_runtime=float(result.get("coordination_runtime", math.nan)),
            pbs_popped_nodes=int(result.get("expanded", -1)),
            pbs_generated_children=int(result.get("generated", -1)),
            pbs_update_calls=int(result.get("update_calls", -1)),
            mp_stats=np.array([result.get("low_level_calls", -1), result.get("low_level_runtime", math.nan)]),
            ecd_stats=np.array([result.get("reservation_calls", -1), result.get("reservation_runtime", math.nan)]),
            cc_stats=np.array([result.get("collision_calls", -1), result.get("collision_runtime", math.nan)]),
            mp_gcs_stats=profile.get("gcs", unavailable), mp_gub_stats=profile.get("gub", unavailable),
            mp_search_stats=profile.get("search", unavailable),
            mp_convex_restriction_stats=profile.get("cr", unavailable),
            mp_dominance_check_stats=profile.get("dc", unavailable),
            mp_dominance_convex_restriction_stats=profile.get("dc_cr", unavailable),
        )
