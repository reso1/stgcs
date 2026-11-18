# Visualization

Run commands from the repository root in the
[project Python environment](../README.md#setup).

```text
visualization/
  solution_io.py       saved trajectory reading, validation, and serialization
  animation/
    common.py          interpolation, playback timing, and GIF encoding
    st.py              one saved ST trajectory
    mrmp.py            saved MRMP or smoothed trajectories
  viewer/
    server.py          shared HTTP server and command-line entry point
    manifest.py        geometry and query export for base/ST/MRMP instances
    planning.py        viewer requests dispatched to the shared planning API
    village.py         village scene presentation
    editor.py          interactive ST instance editor
    assets.py          packaged HTML loading
    viewer.html
    editor.html
```

Generated experiment figures and tables are documented in
[experiments/reports](../experiments/README.md#generate-figures-from-saved-results).
The viewer reads the existing saved-solution JSON format. Loading and animating
saved solutions does not run planners or compute heuristics. The viewer's planner
buttons explicitly request a new solve through the shared Python API.

## View a saved solution

The demo scripts start this viewer automatically. To start it directly:

```bash
python -m visualization.viewer.server \
  --input data/instances/mrmp/demo_r50_four_room/manifest.json \
  --solution-json data/solutions/demo_r50_four_room/mrmp_solution.json \
  --solution-json data/solutions/demo_r50_four_room/trajopt_solution.json
```

Open `http://127.0.0.1:8765/`. Use `--instance-id` or `--limit` to select entries
from a larger manifest. The server also accepts base and ST manifests.

Export viewer geometry without serving it:

```bash
python -m visualization.viewer.manifest \
  --input data/instances/st_planning/manifest.json \
  --limit 1 --output /tmp/st-viewer.json
```

## Animate a saved solution

```bash
python -m visualization.animation.mrmp \
  --manifest data/instances/mrmp/demo_r50_four_room/manifest.json \
  --solution data/solutions/demo_r50_four_room/trajopt_solution.json \
  --output /tmp/four-room.gif --fps 20 --speed 4
```

`visualization.animation.st` accepts the same flags for a saved single-robot
solution returned by the viewer API (`trajectory`, `instance_id`, and planner
metrics). Both renderers support 2D scenes. `--speed` controls simulated seconds
per playback second. Use the browser viewer for 3D scenes.

## Edit an ST instance

```bash
python -m visualization.viewer.editor --output /tmp/custom-st/manifest.json
```

Open `http://127.0.0.1:8770/` to choose a base instance, edit a query and moving
obstacles, preview the geometry, and export the manifest.
