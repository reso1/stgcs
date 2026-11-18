# Official OMPL ST-RRT* Native Backend

This optional extension builds `baselines._ompl_strrt_star_native`, which is used
by `baselines.ompl_strrt_star.OfficialOMPLSTRRTStar`.

It requires official OMPL headers/libraries and pybind11. This repository does
not vendor either dependency.

With OMPL installed, build from the repository root using the active Python
environment:

```bash
python -m pip install pybind11
cmake \
  -S baselines/ompl_strrt_star_native \
  -B build/ompl-strrt-star \
  -DPYBIND11_FINDPYTHON=ON \
  -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_PREFIX_PATH=/path/to/ompl \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/ompl-strrt-star -j 4
python -c "import baselines._ompl_strrt_star_native"
```

Replace `/path/to/ompl` with the OMPL installation prefix. CMake writes the
extension into `baselines/`. Build the main C++ tools using the
[C++ guide](../../stgcs_cplusplus/README.md), then run the ST-RRT* comparison:

```bash
python -m experiments.runners.st \
  --executable stgcs_cplusplus/build/bin/stgcs_planner \
  --work-dir build/strrt-example/work --output-root build/strrt-example/results \
  --suite performance_comparison --planner strrt \
  --performance-budget 30 --limit 1 --skip-scaling
```

Each OMPL call produces `OMPL ST-RRT*-C(first)` and
`OMPL ST-RRT*-C(final)` result rows.

The adapter only records exact OMPL solutions as successful rows. OMPL
approximate solutions are written as failures because they may not reach the
spatial goal.
By default, Python callers leave OMPL's time dimension unbounded so ST-RRT*
uses its own time-bound expansion strategy. A finite `time_upper_bound` remains
available for genuinely windowed queries. The native binding captures the first
exact OMPL solution through OMPL's intermediate-solution callback, then
continues until the runtime budget to report the final exact solution. The
local binding keeps OMPL's narrowed time bound after each solution but does not
run OMPL 1.7's destructive prune step, which can leave stale connection
pointers when planning continues after the first solution.

`FixedPrioritySTRRTStarPlanner` runs each low-level OMPL call in a forked child
process. OMPL's conditional sampler keeps process-global RNG state internally;
isolating each low-level call prevents earlier agents or earlier benchmark rows
from changing whether a later agent receives an exact first solution.

The extension implements a custom OMPL spatial state space with distance
`max(abs(dx_i) / vlimit_i)`, then wraps it in OMPL `SpaceTimeStateSpace` with
`vMax=1`. This preserves the velocity feasibility semantics used by this
codebase while running official `ompl::geometric::STRRTstar`.

The native motion validator owns static 2D Env geometry checks. Polygonal
obstacles are checked against the robot's swept L-infinity footprint, and
grid-style C-space cells use the same segment-cover test as `Env.is_segment_in_CSpace`.
Python callbacks are reserved for extra time-dependent constraints, such as
dynamic obstacles and fixed-priority reservations.
