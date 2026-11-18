# ST-GCS

This repository implements Graphs of Space-Time Convex Sets (ST-GCS) for
spatiotemporal and multi-robot motion planning.

- Branch *master*: Jingtao Tang, Zining Mao, Lufan Yang, and Hang Ma. "Spatiotemporal and Multi-Robot Motion Planning on Graphs of Space-Time Convex Sets." [[paper]](https://arxiv.org/abs/2607.00444), [[project]](https://sites.google.com/view/stgcs)
- Branch *iros*: Jingtao Tang, Zining Mao, Lufan Yang, and Hang Ma. "Space-Time Graphs of Convex Sets for Multi-Robot Motion Planning." [[paper]](https://arxiv.org/abs/2503.00583), [[project]](https://sites.google.com/view/stgcs)

## Setup

Run commands from the repository root. Create the Python environment and install
this package in editable mode:

```bash
conda create -n stgcs -c conda-forge python=3.11 pip cmake pkg-config cddlib gmp
conda activate stgcs
CFLAGS="-I$CONDA_PREFIX/include" LDFLAGS="-L$CONDA_PREFIX/lib" python -m pip install -e .
```

Planning and trajectory optimization require the Drake C++ SDK and a working
MOSEK license. Build the native library and executables:

```bash
cmake -S stgcs_cplusplus -B stgcs_cplusplus/build \
  -Ddrake_DIR=/path/to/drake/lib/cmake/drake -DCMAKE_BUILD_TYPE=Release
cmake --build stgcs_cplusplus/build -j 4
```

Replace `/path/to/drake` with your SDK installation. See the
[C++ guide](stgcs_cplusplus/README.md) for dependencies and tests, and the
[baseline guide](baselines/README.md) for optional OMPL extensions.

## Quick start

Load the saved 50-robot solution and open the browser viewer:

```bash
python -m demos.demo_r50_four_room
```

Open `http://127.0.0.1:8765/`; stop the server with Ctrl-C. The four large-scale
demos reuse their saved MRMP and smoothed solutions. Add `--no-serve` to load and
validate them without starting a server. Missing solutions are computed.

```bash
python -m demos.demo_r100_empty
python -m demos.demo_r32_sphere_exchange
python -m demos.demo_r48_village
```

The village viewer uses port **8780**. See [demo usage](demos/README.md) for
rerun flags, sequential tasks, and the single-instance planner comparison.
That comparison currently uses a **30-second** budget; the paper's Figure 16
experiment uses 120 seconds.

Generate a comparison figure from the saved raw results:

```bash
python -m experiments.reports.mrmp_comparison \
  --budget 120 --output-prefix build/figures/mrmp_comparison
```

This reads existing CSVs and writes a PDF/PNG. The [experiments guide](experiments/README.md)
covers dataset generation, planner runs, and other reports.

## Repository layout

| Folder | Purpose and usage |
| --- | --- |
| [stgcs_cplusplus/](stgcs_cplusplus/README.md) | C++ library, planner/trajopt executables, and native tests |
| [python/stgcs_native/](python/README.md) | Shared Python client, request/result types, and executable discovery |
| [demos/](demos/README.md) | Runnable single-instance and large-scale demonstrations |
| [experiments/](experiments/README.md) | Dataset generators, experiment runners, and reports |
| [visualization/](visualization/README.md) | Browser viewer, ST instance editor, and GIF renderers |
| [baselines/](baselines/README.md) | Comparison planners and optional native OMPL bindings |
| [data/](data/README.md) | Instances, base geometry/heuristics, saved solutions, and raw experiment results |

Demos and experiments call the Python client, which runs `stgcs_planner` and
`stgcs_trajopt` from `stgcs_cplusplus/build/bin`. To select another build, set
`STGCS_NATIVE_EXECUTABLE` and `STGCS_TRAJOPT_EXECUTABLE`.

## BibTeX

```bibtex
@misc{tang2026searchbasedspatiotemporalmultirobotmotion,
  title={Spatiotemporal and Multi-Robot Motion Planning on Graphs of Space-Time Convex Sets},
  author={Jingtao Tang and Zining Mao and Lufan Yang and Hang Ma},
  year={2026},
  eprint={2607.00444},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2607.00444}
}
```

IROS version:

```bibtex
@misc{tang2025spacetimegraphsconvexsets,
  title={Space-Time Graphs of Convex Sets for Multi-Robot Motion Planning},
  author={Jingtao Tang and Zining Mao and Lufan Yang and Hang Ma},
  year={2025},
  eprint={2503.00583},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2503.00583},
}
```

## License

ST-GCS is released under GPL version 3. See `LICENSE.txt` for details.
