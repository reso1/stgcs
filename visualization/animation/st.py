"""Animate a saved two-dimensional ST solution with square collision footprints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from experiments.instance import Instance
from experiments.manifests.st_planning import STPlanningManifestStore
from experiments.native import STGCSPlannerClient
from visualization.animation.common import GifAnimation
from visualization.solution_io import SolutionIO


class STSolutionGif:
    @classmethod
    def render(cls, manifest, solution, base_root, output, fps=20, speed=4.):
        payload = json.loads(Path(solution).read_text())
        record, = [r for r in STPlanningManifestStore.load_manifest(manifest)
                   if r.instance_id == payload["instance_id"]]
        if not payload["ok"] or not payload["is_success"]:
            raise ValueError(f"No successful trajectory for {record.instance_id}")
        trajectory = SolutionIO.trajectory_from_viewer_json(payload["trajectory"])
        env = STGCSPlannerClient.base_instance(record, base_root).env
        if env.dim != 2 or trajectory.dim != 3:
            raise ValueError("This renderer expects a two-dimensional instance and trajectory")
        segments = np.asarray(trajectory.points).reshape(-1, 2, 3)
        start, end = float(trajectory.x0[-1]), float(trajectory.xT[-1])
        np.testing.assert_allclose(trajectory.x0[:2], record.query.start, rtol=0, atol=1e-6)
        np.testing.assert_allclose(trajectory.xT[:2], record.query.goal, rtol=0, atol=1e-6)
        times = GifAnimation.times(start, end, fps, speed)
        obstacles = Instance.dynamic_obstacles_from_specs(record.dynamic_obstacles)
        path = np.vstack([segments[0, 0], segments[:, 1]])
        fig, ax = plt.subplots(figsize=(6.6, 7.2), dpi=100)
        fig.subplots_adjust(left=.08, right=.97, bottom=.12, top=.86)
        for vertices in env.C_Space:
            ax.fill(*vertices.T, facecolor="#f8fafc", edgecolor="#cbd5e1", lw=.4)
        env.draw_static(ax, alpha=.75)
        ax.plot(*path[:, :2].T, color="#2563eb", alpha=.2, lw=2)
        trail, = ax.plot([], [], color="#2563eb", lw=2)
        for point, marker, color in ((record.query.start, "o", "#2563eb"),
                                     (record.query.goal, "*", "#16a34a")):
            ax.scatter(*point, marker=marker, color=color, s=80, zorder=5)
        robot = ax.add_patch(Rectangle((0, 0), 2*env.robot_radius, 2*env.robot_radius,
                                      facecolor="#2563eb", edgecolor="#172554", zorder=8))
        patches = [ax.add_patch(Rectangle((0, 0), 2*o.radius, 2*o.radius,
                                         facecolor="#f59e0b", edgecolor="#92400e", zorder=6))
                   for o in obstacles]
        ax.set(xlim=(env.lb[0]-.12, env.ub[0]+.12), ylim=(env.lb[1]-.12, env.ub[1]+.12),
               aspect="equal", xlabel="x", ylabel="y")
        fig.suptitle(f"{record.source_domain} · ST planning\nArrival {end:.3f} s", fontsize=13)
        fig.text(.5, .89, payload["planner_name"], ha="center", fontsize=9, color="#64748b")
        clock = ax.text(.02, .98, "", transform=ax.transAxes, va="top", fontsize=10,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.9), zorder=10)
        fig.text(.5, .035, f"{len(obstacles)} moving obstacles · square footprints · {speed:g}× playback",
                 ha="center", fontsize=9)
        frames = []
        for time in times:
            point = GifAnimation.position(segments, time)
            robot.set_xy(point-env.robot_radius)
            elapsed = np.vstack([path[path[:, -1] <= time, :2], point])
            trail.set_data(*elapsed.T)
            for obstacle, patch in zip(obstacles, patches, strict=True):
                patch.set_xy(obstacle.x(time)-obstacle.radius)
            clock.set_text(f"t = {time:5.2f} / {end:.2f} s")
            fig.canvas.draw()
            rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
            frame = rgb.quantize(colors=128) if not frames else rgb.quantize(palette=frames[0], dither=Image.Dither.NONE)
            frames.append(frame)
        plt.close(fig)
        GifAnimation.save(frames, output, fps)
        return dict(gif=str(output), source=str(solution), frames=len(frames),
                    playback_speed=speed, physical_time_range=[start, end])

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--manifest", type=Path, required=True)
        parser.add_argument("--solution", type=Path, required=True)
        parser.add_argument("--base-root", type=Path, default=Path("data/instances/st_planning/base"))
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--fps", type=int, default=20)
        parser.add_argument("--speed", type=float, default=4.)
        args = parser.parse_args()
        print(json.dumps(cls.render(args.manifest, args.solution, args.base_root, args.output,
                                   args.fps, args.speed), indent=2))


if __name__ == "__main__":
    STSolutionGif.main()
