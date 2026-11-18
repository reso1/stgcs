"""Build a paired, size-stratified stress benchmark from the shared ST instances."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from dataclasses import replace
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.datasets.st import STInstanceGenerator
from experiments.instance import Instance
from experiments.manifests.st_planning import STPlanningManifestStore
from experiments.native import STGCSPlannerClient
from experiments.environment.collision import collision_geometries_collide


class STStressDataset:
    # Ten maps from each domain in each half-open band (last band includes 30,000).
    BOUNDS = (3000, 8400, 13800, 19200, 24600, 30001)

    @staticmethod
    def write_json(path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)


    @staticmethod
    def witness(record) -> list[np.ndarray]:
        wait = record.generation["reference_wait"]
        path = [np.asarray(s) + [0, 0, wait, 0, 0, wait]
                for s in record.generation["reference_path"]]
        return [np.r_[path[0][:2], 0, path[0][:3]], *path]


    @staticmethod
    def size(record):
        return record.stgcs_num_vertices + record.stgcs_num_edges

    @staticmethod
    def seed(identifier, attempt):
        # This fixed seed mapping preserves sampled instances; it is not a file/version check.
        return int.from_bytes(hashlib.sha256(f'st-stress-v1:{identifier}:{attempt}'.encode()).digest()[:8], 'little')

    @classmethod
    def schedule(cls, records):
        if (Counter(r.source_domain for r in records) != {'grid2d': 50, 'iris-2d': 50, 'maze': 50}
                or len({r.base_instance_id for r in records}) != 150):
            raise ValueError('Expected the 150 distinct spatial bases in the shared benchmark')
        jobs = []
        for domain in ('maze', 'grid2d', 'iris-2d'):
            group = sorted((r for r in records if r.source_domain == domain), key=lambda r: cls.seed(r.base_instance_id, 0))
            smallest = min(group, key=cls.size)
            group.remove(smallest)
            for i, record in enumerate([smallest, *group]):
                band = i % 5
                low, high = cls.BOUNDS[band], cls.BOUNDS[band + 1] - 1
                if domain == 'maze' and i == 0:
                    high = cls.BOUNDS[0] + 100
                if domain == 'grid2d' and i == 4:
                    low = cls.BOUNDS[-1] - 201
                jobs.append((record, band, low, high))
        return jobs

    @classmethod
    def patrols(cls, record, env, attempt):
        generator = STInstanceGenerator(tmax=record.tmax)
        witness = generator.collision_geometry(cls.witness(record), env.robot_radius, record.tmax)
        rng = np.random.default_rng(cls.seed(record.base_instance_id, attempt))
        # A chord of a convex host stays in the original map's collision-free space.
        # Draw its endpoints near different polygon sides, then randomize the patrol phase.
        while True:
            region = int(rng.integers(len(env.C_Space)))
            vertices = env.C_Space[region]
            i, j = rng.choice(len(vertices), 2, replace=False)
            u = rng.uniform(0.02, 0.98, (2, 1))
            ij = np.array([i, j]); chord = (1-u)*vertices[ij] + u*vertices[(ij+1) % len(vertices)]
            length = float(np.linalg.norm(chord[1]-chord[0]))
            if length < generator.min_patrol_length:
                continue
            phase = float(rng.uniform(0, 2*length/generator.obstacle_speed))
            obstacle = generator.periodic_obstacle(chord, generator.obstacle_radius, phase)
            legs = [np.r_[s.x0, s.itvl.start, s.xt, s.itvl.end] for s in obstacle.segments]
            if collision_geometries_collide(witness, generator.collision_geometry(legs, obstacle.radius, record.tmax)):
                continue
            yield Instance.dynamic_obstacle_to_spec(obstacle), dict(region=region, patrol=chord.tolist(), phase=phase)

    @classmethod
    def generate_one(cls, native, record, base_root, band, low, high):
        base = native.base_instance(record, base_root)
        iid = 'st-stress-' + record.base_instance_id
        path = native.work_dir / 'inputs' / (iid + '.txt')
        lower = max(low, cls.size(record) + 1)
        aim = (lower + high) / 2
        if lower > high:
            raise ValueError(f'{iid}: original graph exceeds requested band')
        for attempt in range(64):
            stream = cls.patrols(record, base.env, attempt)
            additions, placements = [], []
            measured = {0: cls.size(record)}
            growth_estimate = {"maze": 650, "grid2d": 900, "iris-2d": 2000}[record.source_domain]
            count = max(1, round((aim - cls.size(record)) / growth_estimate))
            for _ in range(16):
                while len(additions) < count:
                    obstacle, placement = next(stream)
                    additions.append(obstacle); placements.append(placement)
                obstacles = record.dynamic_obstacles + additions[:count]
                native.write_input(path, base.env, record.query, record.tmax, obstacles)
                try:
                    info = native.validate_witness(path, cls.witness(record))
                except RuntimeError:
                    log = native.work_dir / 'witnesses' / (iid + '.log')
                    if log.read_text().strip() not in ('Continuous witness collision', 'Witness not continuously covered by native ST cells'):
                        raise
                    break
                size = info['vertices'] + info['edges']
                if lower <= size <= high:
                    return cls.accept(record, additions[:count], placements[:count], info, band, low, high, attempt)
                measured[count] = size
                below = max(n for n, s in measured.items() if s < lower)
                above = [n for n, s in measured.items() if n > below and s > high]
                if above:
                    count = (below + min(above)) // 2
                else:
                    slope = max(100, (measured[below] - measured[0])/max(1, below))
                    count = below + max(1, min(max(2, below), round((aim - measured[below])/slope)))
                if count in measured or count > 160:
                    break
        raise RuntimeError(f'{iid}: no valid graph in {low}..{high}')

    @classmethod
    def accept(cls, record, additions, placements, info, band, low, high, attempt):
        obstacles = record.dynamic_obstacles + additions
        gen = record.generation
        parameters = {k: gen['parameters'][k] for k in
                      ('motion_duration', 'obstacle_speed', 'obstacle_radius', 'min_patrol_length', 'vlimit')}
        generation = dict(policy='route-first-stress', source_instance_id=record.instance_id,
                          parameters=dict(parameters, tmax=record.tmax, num_obstacles=len(obstacles)),
                          reference_duration=gen['reference_duration'], reference_wait=gen['reference_wait'],
                          reference_path=gen['reference_path'], placements=gen['placements']+placements,
                          source_obstacles=len(record.dynamic_obstacles), size_metric='vertices + directed edges',
                          size_band=band, acceptance_interval=[low, high],
                          stress_seed=cls.seed(record.base_instance_id, attempt), stress_attempt=attempt)
        result = replace(record, instance_id='st-stress-'+record.base_instance_id, generation=generation,
                         dynamic_obstacles=obstacles, stgcs_num_vertices=info['vertices'], stgcs_num_edges=info['edges'])
        return result, dict(info, instance_id=result.instance_id, source_instance_id=record.instance_id,
                            domain=record.source_domain, base_instance_id=record.base_instance_id, size=cls.size(result),
                            band=band, obstacle_count=len(obstacles))

    @classmethod
    def save_case(cls, job, native, base_root, work):
        parent, band, low, high = job
        saved = work / 'records' / (parent.base_instance_id + '.json')
        if not saved.exists():
            record, info = cls.generate_one(native, parent, base_root, band, low, high)
            cls.write_json(saved, dict(record=record.to_dict(), validation=info))
        return json.loads(saved.read_text())

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--source', type=Path, default=Path('data/instances/st_planning/manifest.json'))
        parser.add_argument('--base-root', type=Path, default=Path('data/instances/st_planning/base'))
        parser.add_argument('--output', type=Path, default=Path('data/instances/st_planning/st_planning_stress'))
        parser.add_argument('--work-dir', type=Path, required=True)
        parser.add_argument('--executable', type=Path, required=True)
        parser.add_argument('--workers', type=int, default=1)
        args = parser.parse_args()
        work = args.work_dir
        native = STGCSPlannerClient(args.executable, work/'native')
        source = STPlanningManifestStore.load_manifest(args.source); records, audit = [], []
        jobs = cls.schedule(source)
        generate = partial(cls.save_case, native=native, base_root=args.base_root, work=work)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for job, entry in zip(jobs, pool.map(generate, jobs), strict=True):
                parent, band, low, high = job
                record = type(parent).from_dict(entry['record'])
                info = entry['validation']
                count = len(parent.dynamic_obstacles)
                if (record.base_instance_id != parent.base_instance_id or record.query != parent.query
                        or record.tmax != parent.tmax or record.source_domain != parent.source_domain
                        or record.dynamic_obstacles[:count] != parent.dynamic_obstacles
                        or len(record.dynamic_obstacles) <= count or not low <= cls.size(record) <= high
                        or info['band'] != band or info['instance_id'] != record.instance_id):
                    raise ValueError(f'Invalid stress case for {parent.instance_id}')
                records.append(record)
                audit.append(info)
                print(f'{len(records)}/150 {record.instance_id}: {info["size"]} ({info["obstacle_count"]} obstacles)', flush=True)
        if (len({r.instance_id for r in records}) != 150 or any(sum(low<=cls.size(r)<high and r.source_domain==d
                for r in records)!=10 for low,high in zip(cls.BOUNDS,cls.BOUNDS[1:]) for d in ('grid2d','iris-2d','maze'))):
            raise ValueError('Unbalanced or incomplete benchmark')
        # Round-trip every saved record through the actual planning input adapter.
        with TemporaryDirectory(dir=work) as staging:
            reconstructed = STGCSPlannerClient(args.executable, Path(staging))
            for r, info in zip(records, audit):
                original = (native.work_dir / 'inputs' / f'{r.instance_id}.txt').read_bytes()
                path = reconstructed.prepare(r, args.base_root)
                if path.read_bytes() != original or not info['valid'] or cls.size(r) != info['size']:
                    raise ValueError('Stored graph or witness differs from validated record: ' + r.instance_id)
        if [r.to_dict() for r in STPlanningManifestStore.load_manifest(args.source)] != [r.to_dict() for r in source]:
            raise ValueError('Parent manifest changed during generation')
        STPlanningManifestStore.save_manifest(args.output/'manifest.json', records)
        cls.write_json(args.output/'validation.json',dict(source_manifest=str(args.source),
            size_metric='vertices + directed edges', band_lower_bounds=list(cls.BOUNDS[:-1]), maximum=cls.BOUNDS[-1]-1,
            base_root=str(args.base_root), instances=audit))


if __name__ == '__main__':
    STStressDataset.main()
