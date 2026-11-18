# Baseline planners

Adapters for the planners compared against ST-GCS. They are selected through
the [experiment runners](../experiments/README.md) or
`python -m demos.demo_mrmp_benchmark`.

| Implementation | Purpose | Extra native dependency |
| --- | --- | --- |
| `ompl_strrt_star.py` | Single-robot ST-RRT* | OMPL ST-RRT* extension |
| `sp_strrtstar.py` | Prioritized multi-robot ST-RRT* | Same ST-RRT* extension |
| `ompl_kcbs.py` | Multi-robot K-CBS | K-CBS OMPL fork and extension |
| `zeta_sipp/`, `pbs_zeta_sipp.py` | Zeta*-SIPP and PBS + Zeta*-SIPP | None beyond project dependencies |
| `cb_gcs.py` | Coupled CBS/GCS baseline | Drake and solver from project setup |

## Native extensions

Build instructions:

- [ST-RRT*](ompl_strrt_star_native/README.md)
- [K-CBS](ompl_kcbs_native/README.md)

Both extensions are built for the active Python environment and placed in
`baselines/`. Check that they can be imported:

```bash
python -c "import baselines._ompl_strrt_star_native"
python -c "import baselines._ompl_kcbs_native"
```

The main `stgcs_cplusplus` build does not build these optional comparison
backends. Full comparisons need both; ST-GCS-only demos use the main C++ tools.
Use `python -m experiments.runners.mrmp_comparison --help` to see planner keys.
