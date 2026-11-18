# Saved data

Inputs, reusable base geometry/heuristics, solution trajectories, and raw results.
The [experiment scripts](../experiments/README.md) generate and consume these
files; the [demos](../demos/README.md) load the saved solutions.

| Folder | Contents |
| --- | --- |
| `stgcs_base/<domain>/` | Base manifests, serialized geometry, and heuristic caches |
| `instances/st_planning/` | ST manifest, selected base instances, and `st_planning_stress/` |
| `instances/mrmp/performance_comparison/` | One `manifest_nN.json` per robot count |
| `instances/mrmp/demo_*/` | Large-demo query manifests |
| `solutions/demo_*/` | Saved MRMP and smoothed trajectories |
| `exp_res/st_planning/`, `exp_res/mrmp/` | Raw experiment CSVs and required manifests |

## Formats

- **Manifests:** JSON arrays of instance records; loaded through
  `experiments.manifests`.
- **Base instances:** `objects/<instance-id>.npy` stores a Python object array.
  Its `.yaml` sidecar contains JSON metadata. Use `SerializedBaseInstanceStore`
  through `BaseInstanceFactory` to read/write these objects.
- **Heuristics:** `heuristics/<base-id>/` contains binary tables, their native
  input, and computation metadata. These are reused across matching queries.
- **Solutions:** JSON includes instance/planner identifiers, timing and cost
  metrics, and timestamped trajectory segments. `SolutionIO` validates and
  converts trajectories; smoothing metadata is stored under `global_trajopt`.
- **Experiment results:** raw per-run CSV rows. Use `STResultStore` for ST rows
  and the MRMP runner/report readers for their result files.

Keep aggregate plots, tables, and run-status files outside `exp_res`; report
examples write to `build/figures`. The windowed-coordination result manifest
identifies the evaluated instances.

Solution folders use the same names as their corresponding MRMP instance
folders. `solutions/demo_r50_four_room/figure18/` holds the selected paper
trajectories separately from the demo's default saved solution.

To inspect the saved 50-robot solutions without planning or opening a viewer:

```bash
python -m demos.demo_r50_four_room --no-serve
```

To view a base domain, including its convex decomposition:

```bash
python -m visualization.viewer.server \
  --input data/stgcs_base/grid2d/manifest.json --limit 1
```
