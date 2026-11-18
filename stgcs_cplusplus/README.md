# C++ core

ST-GCS graph construction, trajectory optimization, search, and multi-robot
coordination. Python demos and experiments call the executables through
[`stgcs_native`](../python/README.md).

## Build

Requirements: CMake 3.20+, a C++20 compiler, Drake's C++ SDK, `pkg-config`, and
`cddlib`. Optimization runs need a working MOSEK license.
The [root setup](../README.md#setup) installs the Python environment and cddlib.

From the repository root:

```bash
cmake -S stgcs_cplusplus -B stgcs_cplusplus/build \
  -Ddrake_DIR=/path/to/drake/lib/cmake/drake \
  -DCMAKE_BUILD_TYPE=Release
cmake --build stgcs_cplusplus/build -j 4
ctest --test-dir stgcs_cplusplus/build --output-on-failure
```

`drake_DIR` is the directory containing `drake-config.cmake`. CMake finds cddlib
through `pkg-config`; if it is installed outside the active environment, set
`PKG_CONFIG_PATH` to its `lib/pkgconfig` directory. Native tests include trajectory
optimization and therefore run the solver. Use `-DBUILD_TESTING=OFF` to omit them.

## Layout and targets

| Path/target | Purpose |
| --- | --- |
| `include/stgcs/` | Public C++ headers |
| `src/` | Implementation and the two CLI entry points |
| `tests/` | Native and Python integration tests |
| `stgcs::stgcs` | CMake target for the library |
| `build/bin/stgcs_planner` | ST/MRMP planning, graph inspection, validation, and heuristic caching |
| `build/bin/stgcs_trajopt` | Trajectory smoothing and collision checks |

`planner_main.cc` defines `PlannerCLI`, with `plan`, `mrmp`, `inspect`, `validate`,
and `cache` commands. `trajopt_main.cc` handles optimization and trajectory
checks. The Python client writes their text requests and decodes JSON results.

For a C++ application in a parent CMake project:

```cmake
add_subdirectory(stgcs_cplusplus)
add_executable(my_planner main.cc)
target_link_libraries(my_planner PRIVATE stgcs::stgcs)
```

For Python use, see the [client example](../python/README.md#plan-a-single-query).
Set `STGCS_NATIVE_EXECUTABLE` or `STGCS_TRAJOPT_EXECUTABLE` to use binaries from
another build directory.
