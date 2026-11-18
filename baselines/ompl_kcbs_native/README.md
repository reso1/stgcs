# Official OMPL K-CBS Native Backend

This optional extension builds `baselines._ompl_kcbs_native`, used by
`baselines.ompl_kcbs.OfficialOMPLKCBS`.

Unlike ST-RRT*, K-CBS is not available in standard OMPL. Build this module
against the official K-CBS Multi-Robot-OMPL fork:

```bash
git clone https://github.com/aria-systems-group/Multi-Robot-OMPL.git /tmp/Multi-Robot-OMPL-KCBS
cmake -S /tmp/Multi-Robot-OMPL-KCBS -B /tmp/Multi-Robot-OMPL-KCBS-build -G Ninja \
  -DCMAKE_INSTALL_PREFIX=/tmp/Multi-Robot-OMPL-KCBS-install
cmake --build /tmp/Multi-Robot-OMPL-KCBS-build
cmake --install /tmp/Multi-Robot-OMPL-KCBS-build
```

Then build the adapter from the repository root using the active Python
environment:

```bash
python -m pip install pybind11
cmake \
  -S baselines/ompl_kcbs_native \
  -B build/ompl-kcbs \
  -DPYBIND11_FINDPYTHON=ON \
  -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_PREFIX_PATH=/tmp/Multi-Robot-OMPL-KCBS-install \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/ompl-kcbs -j 4
python -c "import baselines._ompl_kcbs_native"
```

CMake writes the extension into `baselines/`. Select `--planner kcbs` in the
[MRMP comparison runner](../../experiments/README.md#run-experiments).

The adapter instantiates the official `ompl::multirobot::control::KCBS`
planner. Each MRMP robot is modeled as a 2D disk with single-integrator
controls `x_dot = u`, `u_i in [-vlimit, vlimit]`. Static workspace validity is
checked inside this native extension from the `Env` polygon geometry: polygonal
obstacles are tested against the robot footprint, and grid-style environments
without explicit static obstacles test membership in the C-space polygon union.
Robot-robot collision is the disk-distance check used by the K-CBS demo.
Constraint-tree behavior remains inside the official OMPL fork.
