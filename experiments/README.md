# Experiments

Dataset generation, batch planning, and reports. Run commands from the repository
root after the [project setup](../README.md#setup).

| Location | Purpose |
| --- | --- |
| `datasets/` | Generate base geometry, ST queries, and MRMP manifests |
| `runners/` | Run planners, ablations, and heuristic precomputation |
| `reports/` | Read saved results and produce figures/tables |
| `manifests/`, `environment/` | Dataset records and geometry |
| `planners/` | Shared planner settings and experiment schedules |
| `native.py`, `mrmp.py` | Adapt dataset records to the shared Python/C++ API |
| `results.py`, `base_serialization.py` | Raw ST CSV and saved-base read/write helpers |

## Run experiments

Run one ST instance with BFS, writing results to a separate directory:

```bash
python -m experiments.runners.st \
  --executable stgcs_cplusplus/build/bin/stgcs_planner \
  --work-dir build/st-example/work --output-root build/st-example/results \
  --suite performance_comparison --planner delta-state \
  --performance-budget 30 --limit 1 --skip-scaling
```

Other ST suites are `heuristic_ablation`, `dominance_ablation`, and
`heuristic_inflation`. Use `--suite all` for the full schedule. The runner
rebuilds the selected native build before execution and resumes completed jobs
from its work directory.

Run one 10-robot MRMP instance with Windowed-PBS+BFS:

```bash
python -m experiments.runners.mrmp_comparison \
  data/instances/mrmp/performance_comparison/manifest_n10.json \
  --executable stgcs_cplusplus/build/bin/stgcs_planner \
  --work-dir build/mrmp-example/work --output-root build/mrmp-example/results \
  --planner wpbs --budget 120 --limit 1
```

Omit `--limit` to run the selected manifest. Omit the manifest argument to use
the robot-count manifests under `--manifest-root`. `--planner all` includes the
full comparison; [OMPL backends](../baselines/README.md) are needed for its native
baselines. MRMP runners resume rows already recorded at the selected budget.
`runners.mrmp` runs PBS expansion-rule ablations; `runners.mrmp_windows` runs
windowed-coordination ablations. Use each module's `--help` for its choices.

## Generate figures from saved results

```bash
python -m experiments.reports.st_comparison \
  --budget 30 --output-prefix build/figures/st_comparison
python -m experiments.reports.mrmp_comparison \
  --budget 120 --output-prefix build/figures/mrmp_comparison
```

These read `data/exp_res` and write PDF/PNG outputs. `--results-root` selects
another result directory. Set `--output-prefix` explicitly; most report defaults
point to `latex/figs` or `latex/tables`, which are output directories.

| Paper figure | Module under `experiments.reports` |
| --- | --- |
| 9: spatial domains | `spatial_domains` |
| 10: dominance ablation | `st_dominance` |
| 11: heuristic ablation | `st_heuristics` |
| 12: ST comparison | `st_comparison` |
| 13: ST trajectories | `st_trajectories` |
| 14: PBS expansion rules | `mrmp_pbs` |
| 15: coordination windows | `mrmp_windows` |
| 16: MRMP comparison | `mrmp_comparison` |
| 17: MRMP trajectories | `mrmp_trajectories` |
| 18: large 2D demos | `demo_large_2d` |
| 20: sphere exchange | `demo_sphere_3d` |
| B1: heuristic scaling/stress | `heuristic_scaling` |

The saved Fig. 14 results require
`--results-root data/exp_res/mrmp/pbs_node_expansion`; the module's default points
to a different run directory. `st_inflation` produces heuristic-inflation tables;
`mrmp_runtime` produces runtime breakdowns.

Qualitative trajectory reports have additional inputs: `st_trajectories` needs
the separately prepared IRIS example, and `mrmp_trajectories` needs a comparison
snapshot or `--replay-report` containing saved trajectories. Their normal replay
paths can run planners. The two large-demo reports read `data/solutions`.
Conceptual diagrams and real-robot photos are separate paper assets.

## Generate datasets and precompute heuristics

The existing datasets and heuristic caches are already saved. To inspect the
generation options:

```bash
python -m experiments.datasets.base --help
python -m experiments.datasets.st --help
python -m experiments.datasets.st_stress --help
python -m experiments.datasets.mrmp --help
```

MRMP performance datasets keep one `manifest_nN.json` per robot count. ST stress
instances live under `data/instances/st_planning/st_planning_stress`.
Use each generator's output argument when creating another dataset.

Explicitly prepare missing base heuristic tables with:

```bash
python -m experiments.runners.precompute \
  data/stgcs_base/grid2d/manifest.json --workers 1
```

This skips complete caches; `--force` recomputes them. Planner runs may also
prepare missing caches. Report-only commands above do not need precomputation.
See [data layout](../data/README.md) for stored formats and locations.
