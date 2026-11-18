"""Animate a saved two-dimensional MRMP or trajectory-optimization solution."""
from __future__ import annotations
from visualization.solution_io import SolutionIO

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from PIL import Image

from experiments.base import BaseManifestStore
from experiments.manifests.mrmp import load_manifest

from experiments.mrmp import MRMPExperiment
from visualization.animation.common import GifAnimation


class MRMPSolutionGif:
    @classmethod
    def render(cls, manifest, solution, base_root, output, fps=20, speed=4.):
        payload = json.loads(solution.read_text())
        trajopt = payload.get("global_trajopt", {})
        disk = trajopt.get("pairwise_collision_geometry") == "disk"
        record, = [r for r in load_manifest(manifest) if r.instance_id == payload["instance_id"]]
        SolutionIO.validate_solution_payload(payload, record, payload["planner_key"], payload["budget"])
        if payload["tmax"] != record.tmax:
            raise ValueError("Saved trajectory horizon differs from its manifest")
        base_path = BaseManifestStore.manifest_path(base_root, domain_key=record.domain_key)
        base = BaseManifestStore.record_by_id(base_path, record.base_instance_id)
        instance, queries = MRMPExperiment.reconstruct_instance(record, base, compute_heuristics=False)
        env = instance.env
        if env.dim != 2:
            raise ValueError("This renderer expects a two-dimensional instance")
        paths = [np.asarray(SolutionIO.trajectory_from_viewer_json(p).points).reshape(-1, 2, 3)
                 for p in payload["trajectories"]]
        end = float(payload["makespan"])
        times = GifAnimation.times(0., end, fps, speed)
        for path, query in zip(paths, queries, strict=True):
            np.testing.assert_allclose(GifAnimation.position(path, 0), query.start, rtol=0, atol=1e-6)
            np.testing.assert_allclose(GifAnimation.position(path, end), query.goal, rtol=0, atol=1e-6)

        colors = plt.get_cmap("turbo")(np.linspace(.03, .97, len(paths)))
        fig, ax = plt.subplots(figsize=(6.6, 7.2), dpi=110, facecolor="white")
        fig.subplots_adjust(left=.045, right=.955, bottom=.10, top=.88)
        title = f"{len(paths)}-robot MRMP" + (" · trajopt" if trajopt else "")
        fig.text(.055, .952, title, fontsize=20, weight="bold", color="#203742")
        fig.text(.055, .912,
                 f"Windowed-PBS + BFS   ·   α={payload['window_alpha']:g}   ·   ε={payload['epsilon']:g}",
                 fontsize=11, color="#52636c")
        env.draw_static(ax, alpha=.75)
        ax.set(xlim=(env.lb[0]-.15, env.ub[0]+.15), ylim=(env.lb[1]-.15, env.ub[1]+.15),
               aspect="equal", xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_visible(False)
        robots, labels, trails = [], [], []
        for index, (color, query) in enumerate(zip(colors, queries, strict=True)):
            ax.plot(*query.goal, marker="x", ms=4, mew=.7, color=color, alpha=.45, zorder=2)
            trail, = ax.plot([], [], color=color, lw=1.1, alpha=.55, zorder=3)
            trails.append(trail)
            footprint = Circle((0, 0), env.robot_radius) if disk else Rectangle(
                (0, 0), 2*env.robot_radius, 2*env.robot_radius)
            footprint.set(facecolor=color, edgecolor="#263238", lw=.55, zorder=5)
            robots.append(ax.add_patch(footprint))
            ink = "white" if np.dot(color[:3], [.2126, .7152, .0722]) < .48 else "#132330"
            labels.append(ax.text(0, 0, str(index+1), fontsize=6.5, color=ink,
                                  ha="center", va="center", zorder=6))
        clock = fig.text(.055, .06, "", fontsize=11, color="#203742", family="monospace")
        counter = fig.text(.945, .06, "", ha="right", fontsize=11, color="#203742")
        geometry = "disk" if disk else "square"
        fig.text(.5, .025, f"{speed:g}× playback  ·  {geometry} footprints  ·  × marks goals",
                 ha="center", fontsize=9, color="#65757d")

        frames, palette = [], None
        for time in times:
            arrived = 0
            for path, query, robot, label, trail in zip(paths, queries, robots, labels, trails, strict=True):
                center = GifAnimation.position(path, time)
                if disk:
                    robot.set_center(center)
                else:
                    robot.set_xy(center-env.robot_radius)
                label.set_position(center)
                points = np.asarray([GifAnimation.position(path, t)
                                     for t in np.linspace(max(0., time-.6), time, 5)])
                trail.set_data(points[:, 0], points[:, 1])
                arrived += int(np.max(np.abs(center-query.goal)) <= 1e-6)
            clock.set_text(f"t = {time:5.2f} / {end:.2f} s")
            counter.set_text(f"At goal: {arrived}/{len(paths)}")
            fig.canvas.draw()
            frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
            if palette is None:
                palette = frame.quantize(colors=256)
            frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
        GifAnimation.save(frames, output, fps)
        plt.close(fig)
        with Image.open(output) as gif:
            preview = Image.new("RGB", (frames[0].width*2, frames[0].height*2), "white")
            for tile, index in enumerate((0, int(np.argmin(np.abs(np.array(times)-2.))), len(times)//2, len(times)-1)):
                gif.seek(index)
                preview.paste(gif.convert("RGB"), ((tile % 2)*gif.width, (tile // 2)*gif.height))
            preview.save(output.with_name(output.stem+"_preview.png"))
        metadata = dict(gif=str(output), source=str(solution),
            frames=len(frames),
            fps=fps, playback_speed=speed, physical_time_range=[0., end],
            start_hold_ms=500, end_hold_ms=1000, num_agents=len(paths),
            robot_geometry=geometry, robot_radius=env.robot_radius,
            all_frames_decoded=True, final_at_goal=arrived)
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
        return metadata

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--manifest", type=Path, required=True)
        parser.add_argument("--solution", type=Path, required=True)
        parser.add_argument("--base-root", type=Path, default=Path("data/stgcs_base"))
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--fps", type=int, default=20)
        parser.add_argument("--speed", type=float, default=4.)
        args = parser.parse_args()
        print(json.dumps(cls.render(args.manifest, args.solution, args.base_root, args.output,
                                   args.fps, args.speed), indent=2))


if __name__ == "__main__":
    MRMPSolutionGif.main()
