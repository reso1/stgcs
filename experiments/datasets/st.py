"""Generate shared ST instances with periodic obstacles across the spatial cover.

Run ``python -m experiments.datasets.st --help`` from the repo root.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from pydrake.all import RandomGenerator
from scipy.sparse.csgraph import shortest_path
from shapely import get_coordinates, line_merge
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.ops import nearest_points, unary_union

from experiments.base import BaseInstanceFactory
from experiments.environment.obstacle import ConcatDynamicSphere
from experiments.instance import Instance
from experiments.manifests.base import BaseBenchmarkRecord, load_manifest
from experiments.manifests.st_planning import (
    STHeuristicAblationRecord, STPlanningManifestStore, STQuerySpec,
)
from experiments.environment.geometry import HPolyhedronSampler
from experiments.environment.interval import Interval
from stgcs_native.types import MPQuery
from experiments.environment.collision import collision_geometries, collision_geometries_collide


@dataclass(frozen=True)
class STInstanceGenerator:
    """One seeded random generation policy for maze, grid and IRIS maps."""

    tmax: float = 100.0
    motion_duration: float = 20.0
    obstacle_speed: float = 0.25
    obstacle_radius: float = 0.1
    min_patrol_length: float = 0.5
    max_vertices: int = 400
    max_edges: int = 2500
    max_attempts: int = 20
    vlimit: float = 1.0
    num_obstacles: int = 10
    seed: int = 0
    distinct_start_sets: bool = False
    motion: str = "region-chord"

    def __post_init__(self) -> None:
        for name in ("tmax", "motion_duration", "vlimit", "obstacle_speed",
                     "obstacle_radius", "min_patrol_length"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.motion_duration > self.tmax:
            raise ValueError("Require motion_duration <= tmax.")
        for name in ("num_obstacles", "max_vertices", "max_edges", "max_attempts"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if self.motion not in ("global", "region-chord"):
            raise ValueError("motion must be 'global' or 'region-chord'.")

    def reference_routes(self, env):
        """Prefer routes visiting more patrol regions, reserving half the horizon for avoidance."""
        # Interior candidates also support a base map with only one region.
        points = np.vstack([0.5 * (vertices + np.mean(vertices, axis=0))
                            for vertices in env.C_Space])
        owners = np.repeat(np.arange(len(env.C_Space)), [len(vertices) for vertices in env.C_Space])
        if len(points) < 2:
            raise ValueError("Need two query points for a reference route.")
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
        routes, vertices, indices, predecessors = self.static_routes(env, points)
        order = np.lexsort((-distances.ravel(), -routes.ravel()))
        seen = set()
        available = self.exclusive_regions(env)
        candidates = []
        for index in order:
            i, j = np.unravel_index(index, distances.shape)
            duration = routes[i, j] / self.vlimit
            if distances[i, j] <= 1e-9 or not 0 < duration <= self.tmax / 2:
                continue
            pair = (int(owners[i]), int(owners[j]))
            if pair in seen:
                continue
            seen.add(pair)
            path = [int(indices[j])]
            while path[-1] != indices[i]:
                path.append(int(predecessors[indices[i], path[-1]]))
            waypoints = vertices[path[::-1]]
            times = np.r_[0.0, np.cumsum(np.max(np.abs(np.diff(waypoints, axis=0)), axis=1)) / self.vlimit]
            reference = [np.r_[a, ta, b, tb] for a, b, ta, tb in
                         zip(waypoints, waypoints[1:], times, times[1:])]
            query = MPQuery(points[i], points[j], 0.0, True, self.vlimit)
            route = LineString(waypoints)
            coverage = sum(route.distance(area) < (env.robot_radius + self.obstacle_radius) / 2
                           for _, area in available)
            candidates.append((coverage, duration, query, reference))
        for _, _, query, reference in sorted(candidates, key=lambda c: (-c[0], -c[1])):
            yield query, reference

    def waiting_time(self, reference, radius, obstacles):
        """Earliest safe uniform start delay on a 0.25 s grid, including endpoint stays."""
        for wait in np.arange(0.0, self.tmax - reference[-1][-1] + 1e-9, 0.25):
            delayed = [s + np.array([0, 0, wait, 0, 0, wait]) for s in reference]
            if wait > 0:
                delayed.insert(0, np.r_[reference[0][:3], reference[0][:2], wait])
            if not self.collides(delayed, radius, obstacles):
                return float(wait)
        raise ValueError("No collision-free delayed reference within the horizon.")

    @staticmethod
    def collision_geometry(segments, radius, horizon):
        return collision_geometries(segments, cc_to_t0=False, cc_to_tf=True, space_dim=2,
            robot_radius=radius, t0=0, tmax=horizon, tolerance=0.5e-6, collision_geometry_cache={})

    def collides(self, segments, radius, obstacles):
        geometry = self.collision_geometry(segments, radius, self.tmax)
        return any(collision_geometries_collide(geometry, obstacle) for obstacle in obstacles)

    @staticmethod
    def static_routes(env, points: np.ndarray):
        """Shortest visibility-graph distances in the static cover, using the L-infinity norm."""
        space = unary_union([Polygon(vertices).convex_hull for vertices in env.C_Space])
        vertices, inverse = np.unique(np.vstack([points, get_coordinates(space.boundary)]),
                                      axis=0, return_inverse=True)
        weights = np.full((len(vertices), len(vertices)), np.inf)
        np.fill_diagonal(weights, 0.0)
        for i, a in enumerate(vertices):
            for j in range(i):
                if space.covers(LineString([a, vertices[j]])):
                    weights[i, j] = weights[j, i] = np.max(np.abs(a - vertices[j]))
        routes, predecessors = shortest_path(weights, directed=False, return_predecessors=True)
        indices = inverse[:len(points)]
        return routes[np.ix_(indices, indices)], vertices, indices, predecessors

    @staticmethod
    def spatial_path(base: Instance, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Use BfS on the initial ST-GCS to connect globally sampled endpoints."""
        query = MPQuery(start, goal, base.stgcs.t0, False, base.stgcs.vlimit)
        from experiments.native import STGCSPlannerClient

        solution, _ = STGCSPlannerClient.solve_single(base.env, query, base.stgcs.tmax, budget=float("inf"))
        if solution is None:
            raise ValueError("BfS could not connect the sampled obstacle endpoints.")
        path = np.asarray([solution.x0[:-1],
                           *[solution.xB(i)[:-1] for i in range(solution.size)]])
        return path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9]]

    def periodic_obstacle(self, path: np.ndarray, radius: float, phase: float = 0.0) -> ConcatDynamicSphere:
        if len(path) < 2:
            raise ValueError("Need distinct points for obstacle motion.")
        cycle = np.vstack([path, path[-2::-1]])
        distances = np.linalg.norm(np.diff(cycle, axis=0), axis=1)
        if np.any(distances <= 0):
            raise ValueError("Consecutive motion waypoints must be distinct.")
        times = np.r_[0.0, np.cumsum(distances)] / self.obstacle_speed
        if not np.isfinite(phase):
            raise ValueError("Patrol phase must be finite.")
        phase %= times[-1]
        starts, goals, intervals = [], [], []
        for offset in np.arange(-phase, self.motion_duration, times[-1]):
            for a, b, t0, t1 in zip(cycle, cycle[1:], times, times[1:]):
                begin, end = max(0.0, offset + t0), min(offset + t1, self.motion_duration)
                if begin >= end:
                    continue
                starts.append(a + (b - a) * ((begin - offset - t0) / (t1 - t0)))
                goals.append(a + (b - a) * ((end - offset - t0) / (t1 - t0)))
                intervals.append(Interval(begin, end))
        return ConcatDynamicSphere(starts, goals, intervals, radius)

    def exclusive_regions(self, env):
        """Center positions whose inflated ECD footprint misses every other set."""
        if env.dim != 2:
            raise ValueError("Region-chord motion requires a 2D spatial cover.")
        # ECD uses an axis-aligned box with radius r_robot + r_obstacle.
        offsets = (env.robot_radius + self.obstacle_radius + 1e-6) * np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        regions = [Polygon(vertices).convex_hull for vertices in env.C_Space]
        inflated = [MultiPoint((vertices[:, None] + offsets).reshape(-1, 2)).convex_hull
                    for vertices in env.C_Space]
        available = []
        for index, region in enumerate(regions):
            exclusive = region.difference(unary_union([p for j, p in enumerate(inflated) if j != index]))
            if exclusive.area > 1e-12:
                available.append((index, exclusive))
        return available

    @staticmethod
    def region_chord(vertices, exclusive, anchor) -> np.ndarray:
        """A long clear chord through a reference-route encounter location."""
        reach = 2 * np.linalg.norm(np.ptp(vertices, axis=0))
        best = None
        # Chord endpoints change boundary edges at directions through polygon corners.
        for corner in np.unique(get_coordinates(exclusive), axis=0):
            direction = corner - anchor
            length = np.linalg.norm(direction)
            if length <= 1e-12:
                continue
            direction /= length
            clipped = line_merge(exclusive.intersection(
                LineString([anchor - reach * direction, anchor + reach * direction])))
            pieces = [clipped] if clipped.geom_type == "LineString" else list(clipped.geoms)
            for piece in pieces:
                if (piece.geom_type == "LineString" and piece.distance(Point(anchor)) < 1e-9
                        and (best is None or piece.length > best.length)):
                    best = piece
        if best is None:
            raise ValueError("Could not locate the sampled point's clear patrol segment.")
        return np.asarray([best.coords[0], best.coords[-1]])

    def populate_obstacles(self, base: Instance, reference, attempt: int = 0):
        """Spread encounters along the nominal route while protecting a delayed witness."""
        env = base.env.copy()
        remaining = list(range(len(env.C_Space)))
        if self.distinct_start_sets and self.num_obstacles > len(remaining):
            raise ValueError("Need at least one spatial set per obstacle for distinct starts.")
        seed = (base.seed + self.seed + 1000003 * attempt) % 2**32
        rng, drake_rng = np.random.RandomState(seed), RandomGenerator(seed)
        env.O_Dynamic = []
        placements, geometries = [], []
        if self.motion == "region-chord":
            regions = self.exclusive_regions(env)
            radius = env.robot_radius + self.obstacle_radius + 1e-6
            endpoints = [reference[0][:2], reference[-1][3:5]]
            protected = unary_union([box(*(p - radius), *(p + radius)) for p in endpoints])
            regions = [(i, area.difference(protected)) for i, area in regions]
            candidates = []
            duration = reference[-1][-1]
            # Protect a departure halfway through the remaining time, rounded to the wait grid.
            witness_wait = np.floor((self.tmax - duration) / 0.5) * 0.25
            witness = [s + np.array([0, 0, witness_wait, 0, 0, witness_wait]) for s in reference]
            witness.insert(0, np.r_[reference[0][:3], reference[0][:2], witness_wait])
            witness_geometry = self.collision_geometry(witness, env.robot_radius, self.tmax)
            step = duration / (8 * self.num_obstacles + 1)
            for t in np.linspace(0, duration, 8 * self.num_obstacles + 2)[1:-1]:
                t += rng.uniform(-0.25, 0.25) * step
                if t >= self.motion_duration:
                    continue
                leg = next(s for s in reference if s[2] <= t <= s[-1])
                point = leg[:2] + (leg[3:5] - leg[:2]) * ((t - leg[2]) / (leg[-1] - leg[2]))
                for index, exclusive in regions:
                    if exclusive.is_empty:
                        continue
                    # Move inward while keeping the reference inside the collision footprint.
                    nearby = exclusive.intersection(box(*(point - 0.9 * radius), *(point + 0.9 * radius)))
                    if nearby.is_empty or nearby.area <= 1e-12:
                        continue
                    center = Point(np.mean(env.C_Space[index], axis=0))
                    anchor = np.asarray(nearest_points(nearby, center)[0].coords[0])
                    path = self.region_chord(env.C_Space[index], exclusive, anchor)
                    if np.linalg.norm(path[1] - path[0]) >= self.min_patrol_length:
                        # Prefer motion opposing the reference direction.
                        if np.dot(path[1] - path[0], leg[3:5] - leg[:2]) > 0:
                            path = path[::-1].copy()
                        for directed_path in (path, path[::-1].copy()):
                            phase = np.linalg.norm(anchor - directed_path[0]) / self.obstacle_speed - t
                            obstacle = self.periodic_obstacle(directed_path, self.obstacle_radius, phase)
                            geometry = self.collision_geometry(
                                [np.r_[s.x0, s.itvl.start, s.xt, s.itvl.end] for s in obstacle.segments],
                                obstacle.radius, self.tmax)
                            if not collision_geometries_collide(witness_geometry, geometry):
                                placement = dict(region=index, route_time=float(t), encounter_time=float(t),
                                                 anchor=anchor.tolist(), patrol=directed_path.tolist(), phase=float(phase))
                                candidates.append((obstacle, geometry, placement))
                                break
                        break
            if len(candidates) < self.num_obstacles:
                raise ValueError("Not enough long region chords along the reference route.")
            used, encounter_times = set(), [0.0, duration]
            for _ in range(self.num_obstacles):
                choices = [c for c in candidates if not self.distinct_start_sets or c[2]['region'] not in used]
                if not choices:
                    raise ValueError("Not enough distinct patrol regions along the reference route.")
                chosen = max(choices, key=lambda c: (c[2]['region'] not in used,
                    min(abs(c[2]['route_time'] - t) for t in encounter_times)))
                candidates = [c for c in candidates if c is not chosen]
                obstacle, geometry, placement = chosen
                used.add(placement['region'])
                encounter_times.append(placement['route_time'])
                env.O_Dynamic.append(obstacle)
                geometries.append(geometry)
                placements.append(placement)
            wait = self.waiting_time(reference, env.robot_radius, geometries)
            return env, placements, wait
        for _ in range(self.num_obstacles):
            if self.distinct_start_sets:
                index = remaining.pop(rng.randint(len(remaining)))
                start = HPolyhedronSampler.uniform_sample(
                    env._CSpace_hpoly[index], drake_rng,
                    context=f"STInstanceGenerator[{env.name}]",
                )
                goal = env.sample_CSpace(rng, drake_rng)
            else:
                start, goal = env.sample_CSpace(rng, drake_rng, size=2)
            path = self.spatial_path(base, start, goal)
            if np.linalg.norm(np.diff(path, axis=0), axis=1).sum() < self.min_patrol_length:
                raise ValueError("Global patrol is shorter than min_patrol_length.")
            env.O_Dynamic.append(self.periodic_obstacle(path, self.obstacle_radius))
        geometries = [self.collision_geometry(
            [np.r_[s.x0, s.itvl.start, s.xt, s.itvl.end] for s in o.segments],
            o.radius, self.tmax) for o in env.O_Dynamic]
        return env, placements, self.waiting_time(reference, env.robot_radius, geometries)

    def generate(self, base: Instance):
        """Choose the route before its obstacles; bound graphs without screening BFS."""
        reason = "No layouts attempted"
        for attempt, (query, reference) in enumerate(self.reference_routes(base.env)):
            if attempt >= self.max_attempts:
                break
            try:
                env, placements, wait = self.populate_obstacles(base, reference, attempt)
                if wait <= 0:
                    raise ValueError("Reference route has no timed obstacle encounter.")
                stgcs = Instance.build_stgcs_from_env(env, tmax=self.tmax, vlimit=self.vlimit)
                vertices, edges = stgcs.G.number_of_nodes(), stgcs.G.number_of_edges()
                if vertices > self.max_vertices or edges > self.max_edges:
                    raise ValueError(f"Graph exceeds limits: {vertices} vertices, {edges} edges.")
                if not stgcs.has_path(query):
                    raise ValueError("No encoded graph path for the reference query.")
            except ValueError as exc:
                reason = str(exc)
                continue
            metrics = dict(reference_duration=float(reference[-1][-1]), reference_wait=wait,
                           reference_path=[p.tolist() for p in reference], wait_step=0.25,
                           placements=placements, policy="route-first", layout_attempt=attempt,
                           parameters=asdict(self))
            return Instance(base.name, base.seed, env, stgcs), query, metrics
        raise ValueError(f"{base.name}: no accepted layout in {self.max_attempts} attempts: {reason}")

    def build_record(self, base_record: BaseBenchmarkRecord) -> STHeuristicAblationRecord:
        base = BaseInstanceFactory.from_record(base_record, compute_heuristics=False)
        instance, query, metrics = self.generate(base)
        return STHeuristicAblationRecord(
            instance_id=f"st-route-first-{base_record.instance_id}-s{self.seed}",
            group="iris2d" if base_record.domain_key == "iris-2d" else base_record.domain_key,
            source_domain=base_record.domain_key,
            base_instance_id=base_record.instance_id,
            query=STQuerySpec.from_query(query),
            dynamic_obstacles=[Instance.dynamic_obstacle_to_spec(o) for o in instance.env.O_Dynamic],
            stgcs_num_vertices=instance.stgcs.G.number_of_nodes(),
            stgcs_num_edges=instance.stgcs.G.number_of_edges(),
            tmax=self.tmax,
            generation=metrics,
        )

    @classmethod
    def main(cls) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--base-root", type=Path, default=Path("data/instances/st_planning/base"))
        parser.add_argument("--domains", nargs="+", choices=("maze", "grid2d", "iris-2d"),
                            default=("maze", "grid2d", "iris-2d"))
        parser.add_argument("--count-per-domain", type=int, default=20)
        parser.add_argument("--workers", type=int, default=1)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--from-manifest", type=Path,
                            help="Reuse the base-map selection of an existing ST manifest.")
        parser.add_argument("--tmax", type=float, default=cls.tmax)
        parser.add_argument("--obstacle-speed", type=float, default=cls.obstacle_speed)
        parser.add_argument("--obstacle-radius", type=float, default=cls.obstacle_radius)
        parser.add_argument("--min-patrol-length", type=float, default=cls.min_patrol_length)
        parser.add_argument("--max-vertices", type=int, default=cls.max_vertices)
        parser.add_argument("--max-edges", type=int, default=cls.max_edges)
        parser.add_argument("--max-attempts", type=int, default=cls.max_attempts)
        parser.add_argument("--motion-duration", type=float, default=cls.motion_duration,
                            help="Time at which obstacles stop and remain at their final positions.")
        parser.add_argument("--vlimit", type=float, default=cls.vlimit)
        parser.add_argument("--num-obstacles", type=int, default=cls.num_obstacles)
        parser.add_argument("--distinct-start-sets", action="store_true",
                            help="Require distinct starting sets; global-motion goals remain unrestricted.")
        parser.add_argument("--motion", choices=("global", "region-chord"), default=cls.motion,
                            help="Global BfS routes or long chords clear of other sets, reusing regions as needed.")
        parser.add_argument("--seed", type=int, default=cls.seed)
        args = parser.parse_args()
        if args.count_per_domain <= 0 or args.workers <= 0:
            parser.error("--count-per-domain and --workers must be positive.")
        generator = cls(tmax=args.tmax, motion_duration=args.motion_duration,
                        obstacle_speed=args.obstacle_speed, obstacle_radius=args.obstacle_radius,
                        min_patrol_length=args.min_patrol_length, max_vertices=args.max_vertices,
                        max_edges=args.max_edges, max_attempts=args.max_attempts,
                        vlimit=args.vlimit, num_obstacles=args.num_obstacles, seed=args.seed,
                        distinct_start_sets=args.distinct_start_sets, motion=args.motion)
        previous = (STPlanningManifestStore.load_manifest(args.from_manifest)
                    if args.from_manifest is not None else None)
        selected_bases = []
        for domain in dict.fromkeys(args.domains):
            bases = load_manifest(args.base_root / domain / "manifest.json")
            if len(bases) < args.count_per_domain:
                raise ValueError(f"{domain}: fewer than {args.count_per_domain} base maps.")
            if previous is None:
                indices = np.random.default_rng(args.seed).choice(
                    len(bases), size=args.count_per_domain, replace=False,
                )
                selected = [bases[index] for index in indices]
            else:
                ids = [record.base_instance_id for record in previous if record.source_domain == domain]
                if len(ids) != args.count_per_domain or len(set(ids)) != len(ids):
                    raise ValueError(f"{domain}: source manifest must reference {args.count_per_domain} distinct base maps.")
                by_id = {base.instance_id: base for base in bases}
                selected = [by_id[instance_id] for instance_id in ids]
            selected_bases.extend(selected)
        records = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for record in pool.map(generator.build_record, selected_bases):
                records.append(record)
                print(f"{record.instance_id}: {record.stgcs_num_vertices} vertices, "
                      f"{record.stgcs_num_edges} directed edges", flush=True)
        STPlanningManifestStore.save_manifest(args.output, records)


if __name__ == "__main__":
    STInstanceGenerator.main()
