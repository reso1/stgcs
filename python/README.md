# Python client

`stgcs_native` is the shared Python interface to the C++ executables. Install it
with `python -m pip install -e .` from the repository root, using the
[project environment](../README.md#setup).

| Module | Purpose |
| --- | --- |
| `client.py` | Executable discovery, subprocess execution, and result decoding |
| `types.py` | Planning problems, queries, trajectories, and result records |
| `planning.py` | `NativeBackend` for ST and `MRMPNativeBackend` for MRMP |
| `trajopt.py` | `GlobalTrajectoryOptimizer` and smoothing configuration/results |

The default binaries are `stgcs_cplusplus/build/bin/stgcs_planner` and
`stgcs_cplusplus/build/bin/stgcs_trajopt`. The client requires an existing build.
Override discovery with `STGCS_NATIVE_EXECUTABLE` and `STGCS_TRAJOPT_EXECUTABLE`.

## Plan a single query

This example plans inside a unit-square convex region with the motion heuristic:

```python
from pathlib import Path
import numpy as np
from stgcs_native import NativeBackend, NativeClient, PlanningProblem, Polyhedron, MPQuery

region = Polyhedron(
    A=np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]),
    b=np.array([1., 0., 1., 0.]),
)
problem = PlanningProblem(regions=[region], robot_radius=0.05,
                          horizon=10., velocity_limit=1.)
query = MPQuery(start=np.array([0.1, 0.1]), goal=np.array([0.9, 0.9]),
                t_start=0., is_stay=True, vlimit=1.)
client = NativeBackend(NativeClient.executable("stgcs_planner"), Path("build/client-example"))
trajectory, result = client.plan_single(problem, query, budget=5.)
print(result["success"], result["cost"])
```

`Polyhedron` uses `A x <= b`. Planning regions describe collision-free robot-center
space; account for the robot radius when constructing them. A returned
`STTrajectory` stores segments as `[start_position, start_time, goal_position,
goal_time]`. Failed solves return no trajectory and retain status information
in the result.

`MRMPNativeBackend.plan_multi` accepts multiple queries and coordination settings.
`GlobalTrajectoryOptimizer.smooth` refines saved piecewise-linear trajectories
using sliding windows and validates the resulting clearance.

## Dataset-backed callers

`experiments.native.STGCSPlannerClient` converts project geometry and manifests
to these request types. `experiments.mrmp.MRMPExperiment` provides the planning
and smoothing workflows used by the demos. Use these adapters for existing
project datasets; see [experiments](../experiments/README.md) and
[demos](../demos/README.md). Saved viewer JSON is handled by
`visualization.solution_io.SolutionIO`.
