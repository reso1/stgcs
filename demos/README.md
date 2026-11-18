# Demos

Python entry points for individual planning problems and their browser viewers.
Install the project and build the native tools as described in the
[root README](../README.md). Run commands from the repository root.

## Large-scale saved solutions

```bash
python -m demos.demo_r50_four_room
python -m demos.demo_r100_empty
python -m demos.demo_r32_sphere_exchange
python -m demos.demo_r48_village
```

Run one viewer at a time, or choose distinct ports with `--port`.

| Module | Instance | Planning budget | Viewer port |
| --- | --- | --- | --- |
| `demo_r50_four_room` | 50 robots, four rooms, query seed 3 | 120 s | 8765 |
| `demo_r100_empty` | 100 robots, empty square, query seed 3 | 120 s | 8765 |
| `demo_r32_sphere_exchange` | 32 UAVs exchanging antipodal positions | 120 s | 8765 |
| `demo_r48_village` | 48 UAVs, village seed 1000 | 300 s | 8780 |

Each script uses its matching folder under `data/instances/mrmp` and
`data/solutions`. Saved MRMP and smoothed solutions are reused when present;
missing solutions are computed. Displayed timings are the saved run's timings.

The four demos use epsilon 8, alpha 2, beta 1, dynamic windows, NC ordering,
and a 100-second planning horizon. The village uses `max(h_mot, h_tri)`;
the others use `h_max`. Smoothing samples every 0.10 s with 0.50 s windows
and a 0.25 s stride.

Common options:

| Option | Effect |
| --- | --- |
| `--no-serve` | Load/run and print results without starting the viewer |
| `--force-solve` | Recompute MRMP and refresh smoothing |
| `--force-trajopt` | Reuse MRMP and recompute smoothing |
| `--no-trajopt` | Show only the MRMP solution |
| `--solution-output PATH`, `--trajopt-output PATH` | Select the saved solution files |

For example, recompute into separate files:

```bash
python -m demos.demo_r50_four_room --force-solve --force-trajopt --no-serve \
  --solution-output build/demo-rerun/mrmp_solution.json \
  --trajopt-output build/demo-rerun/trajopt_solution.json
```

## Sequential tasks

```bash
python -m demos.demo_stgcs_rearrangement
python -m demos.demo_stgcs_formation_control
```

These run a sequence of MRMP stages, smooth the trajectories, and serve the
viewer on port 8766. They create solutions when no saved files exist. Both accept
`--force-solve`, `--force-trajopt`, `--no-serve`, and `--output-dir`.
To inspect targets without planning:

```bash
python -m demos.demo_stgcs_rearrangement targets --no-show-targets \
  --targets-output build/rearrangement-targets.png
```

## Single-instance planner comparison

```bash
python -m demos.demo_mrmp_benchmark
```

This compares planners on the 10-robot grid `s4/seed1` instance. Its current
budget is **30 s per planner**, whereas Figure 16 uses 120 s. Configuration is
in the module's `INSTANCE`, `BUDGET`, and `PLANNER_KEYS` constants. It saves
results under `data/solutions/benchmark_comparison` and opens port 8765.
Build the [optional OMPL backends](../baselines/README.md) for the complete
comparison; unavailable native baselines are skipped.

Dataset-backed demos use `MRMPExperiment` and the shared `stgcs_native` client.
Sequential tasks use `sequential_mrmp_utils.py`; browser serving and solution
serialization live in [visualization](../visualization/README.md).
