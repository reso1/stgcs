"""Figure 18: selected 50/100-robot MRMP plans and disk trajectory optimization."""
from __future__ import annotations
from visualization.solution_io import SolutionIO

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, Rectangle

from experiments.base import BaseManifestStore
from experiments.manifests.mrmp import load_manifest
from experiments.native import NativeGeometry
from stgcs_native.trajopt import GlobalTrajectoryOptimizer

from experiments.mrmp import MRMPExperiment
from experiments.reports.mrmp_trajectories import Config, MRMPQualitativeComparison
from experiments.reports.common import _save_figure, plt


class Large2DDemoFigure:
    CASES = (
        ("mrmp-base-simple2d-seed00000-50-robot-random", "figure18", 50, 3,
         Path("data/instances/mrmp/demo_r50_four_room/manifest.json")),
        ("mrmp-base-empty-square2d-seed00000-100-robot-random", "", 100, 3,
         Path("data/instances/mrmp/demo_r100_empty/manifest.json")),
    )
    SOURCE_ROOT = Path("data/solutions")
    BUDGET, TMAX = 120., 100.
    LOW_LEVEL_SPEC = replace(Config.LOW_LEVEL_SPEC, epsilon=8.)
    SETTINGS = dict(window_alpha=2., window_beta=1.,
                    epsilon=LOW_LEVEL_SPEC.epsilon, child_expansion_rule="num_conflicts",
                    dynamic_window_adjustment=True)
    TRAJOPT_SETTINGS = dict(sample_dt=.1, window_span=.5, stride=.25,
        velocity_gradient_weight=.05, displacement_weight=.2, displacement_jitter_weight=2e-4,
        clearance_margin=0., solver_eps_abs=1e-6, solver_eps_rel=1e-6,
        fix_terminal_states=True, include_trajectory_knot_times=True,
        include_interval_collision_constraints=True, solver_threads=1,
        pairwise_collision_geometry="disk", qp_solver="MOSEK")

    @classmethod
    def load(cls, source_root, base_root):
        cases = []
        sources = []
        for name, subfolder, count, seed, manifest in cls.CASES:
            solution_dir = source_root / manifest.parent.name / subfolder
            record, = [r for r in load_manifest(manifest) if r.instance_id == name]
            if record.num_agents != count or record.tmax != cls.TMAX:
                raise ValueError(f"Figure 18 instance selection differs: {manifest}")
            base_manifest = base_root / record.domain_key / "manifest.json"
            base = BaseManifestStore.record_by_id(base_manifest, record.base_instance_id)
            instance, queries = MRMPExperiment.reconstruct_instance(record, base, compute_heuristics=False)
            sources.append(str(manifest))
            sources.append(str(base_manifest))
            stages = []
            for stage in ("mrmp", "trajopt"):
                path = solution_dir / f"{stage}_solution.json"
                payload = json.loads(path.read_text())
                if payload.get("tmax") != record.tmax:
                    raise ValueError(f"Solution time horizon differs from the instance: {path}")
                planner_key = "wc-wpbs-h-max-gub-delta-pos-num_conflicts"
                if stage == "trajopt":
                    planner_key += "-global-trajopt"
                SolutionIO.validate_solution_payload(payload, record, planner_key, cls.BUDGET)
                if any(payload.get(key) != value for key, value in cls.SETTINGS.items()):
                    raise ValueError(f"Solution does not use Figure 18 settings: {path}")
                if cls.LOW_LEVEL_SPEC.name not in payload["planner_name"]:
                    raise ValueError(f"Low-level BFS settings differ from Figure 18: {path}")
                if payload.get("backend") != "stgcs_cplusplus":
                    raise ValueError(f"Expected the native C++ planner: {path}")
                paths = [SolutionIO.trajectory_from_viewer_json(p) for p in payload["trajectories"]]
                if stage == "mrmp":
                    checked = MRMPQualitativeComparison.validate(dict(success=True,
                        runtime=payload["runtime"], sum_of_costs=payload["cost"], makespan=payload["makespan"],
                        trajectories=[np.asarray(p.points).tolist() for p in paths]),
                        Config.WINDOWED_PBS_KEY, instance.env, queries, record.tmax)
                else:
                    stats = payload["global_trajopt"]
                    if stats["backend"] != "stgcs_cplusplus" or not stats["sampled_pairwise_collision_free"]:
                        raise ValueError(f"Native trajectory optimization failed its sampled checks: {path}")
                    for key, value in cls.TRAJOPT_SETTINGS.items():
                        actual = stats.get(key)
                        matches = (actual is not None and np.isclose(actual, value, rtol=0, atol=1e-14)
                                   if isinstance(value, float) else actual == value)
                        if not matches:
                            raise ValueError(f"Shared trajectory optimization setting {key} differs: {path}")
                    maximum_velocity = 0.
                    for trajectory, query in zip(paths, queries, strict=True):
                        segments = np.asarray(trajectory.points)
                        if (not np.isfinite(segments).all()
                                or not np.allclose(segments[0, :2], query.start, rtol=0, atol=1e-6)
                                or not np.allclose(segments[-1, 3:5], query.goal, rtol=0, atol=1e-6)):
                            raise ValueError(f"Smoothed trajectory endpoints differ from the query: {path}")
                        dt = segments[:, 5] - segments[:, 2]
                        if (np.any(dt <= 0) or abs(segments[0, 2]-query.t_start) > 1e-6
                                or not np.allclose(segments[1:, :3], segments[:-1, 3:], rtol=0, atol=1e-6)):
                            raise ValueError(f"Smoothed trajectory timing or continuity failed: {path}")
                        velocity = float(np.max(np.abs(segments[:, 3:5]-segments[:, :2])/dt[:, None]))
                        if velocity > query.vlimit+1e-6:
                            raise ValueError(f"Smoothed trajectory exceeds its speed limit: {path}")
                        maximum_velocity = max(maximum_velocity, velocity)
                    if not np.allclose([sum(p.duration for p in paths), max(p.points[-1][-1] for p in paths)],
                                       [payload["cost"], payload["makespan"]], rtol=1e-8, atol=1e-6):
                        raise ValueError(f"Smoothed trajectory costs differ from recorded measurements: {path}")
                    pairwise = GlobalTrajectoryOptimizer.disk_pairwise_collision_free(
                        paths, instance.env.robot_radius, [q.is_stay for q in queries], tolerance=1e-6)
                    contained = GlobalTrajectoryOptimizer.cspace_collision_free(NativeGeometry.cells(instance.stgcs), paths, tolerance=1e-6)
                    if not contained:
                        raise ValueError(f"Smoothed trajectories leave free space: {path}")
                    if not pairwise or not stats["continuous_pairwise_collision_free"]:
                        raise ValueError(f"Continuous disk collision checks failed: {path}")
                    checked = dict(trajectories=len(paths), sampled_pairwise_collision_free=True,
                        continuous_disk_pairwise_collision_free=pairwise,
                        continuous_cspace_collision_free=contained,
                        continuous_pair_checks=count*(count-1)//2, tolerance=1e-6,
                        endpoint_and_continuity_checks=True, max_velocity_component=maximum_velocity)
                sources.append(str(path))
                stages.append(dict(stage=stage, payload=payload, paths=paths, validation=checked, source=str(path)))
            cases.append(dict(name=name, num_agents=count, seed=seed, instance_id=record.instance_id,
                              env=instance.env, queries=queries, stages=stages))
        return cases, sources

    @classmethod
    def draw(cls, cases, prefix):
        # Preserve the original panel size and typography in the 2x2 layout.
        # Robot markers and wall polygons are sized in workspace coordinates.
        panel = 7.2 * .988 / (4 + 3*.045)
        # Add room for labels without resizing the trajectory panels.
        width, height = 2.045*panel + .30, 2*panel + .43
        fig, axes = plt.subplots(2, 2, figsize=(width, height))
        for index, case in enumerate(cases):
            env = case["env"]
            colors = plt.get_cmap("jet")(np.linspace(0., 1., case["num_agents"]))
            radius = env.robot_radius
            for stage_index, stage in enumerate(case["stages"]):
                ax = axes[index, stage_index]
                for obstacle in env.O_Static:
                    ax.fill(*obstacle.vertices.T, facecolor="#484844", edgecolor="none", lw=0)
                for color, path, query in zip(colors, stage["paths"], case["queries"]):
                    segments = np.asarray(path.points)
                    points = np.vstack((segments[0, :2], segments[:, 3:5]))
                    ax.plot(points[:, 0], points[:, 1], color=color, lw=.55, zorder=3)
                    ax.add_patch(Circle(query.start, radius, fill=False,
                                        edgecolor=color, lw=.45, zorder=4))
                    ax.add_patch(Rectangle(np.asarray(query.goal)-radius, 2*radius, 2*radius,
                                           fill=False, edgecolor=color, lw=.45, zorder=4))
                ax.set(xlim=(env.lb[0]-radius, env.ub[0]+radius),
                       ylim=(env.lb[1]-radius, env.ub[1]+radius),
                       aspect="equal", xticks=[], yticks=[])
                payload = stage["payload"]
                method = "Windowed-PBS+BFS" if stage["stage"] == "mrmp" else "Trajectory optimization"
                runtime = payload["runtime"] if stage["stage"] == "mrmp" else payload["global_trajopt"]["runtime"]
                ax.set_title(f"{method} ({runtime:.2f}s)", fontsize=7, color="#333330", pad=4)
                if stage_index == 0:
                    ax.set_ylabel(f"{case['num_agents']}-robot MRMP", rotation=90,
                                  fontsize=7, color="#333330", labelpad=5)
                for spine in ax.spines.values():
                    spine.set_linewidth(1.1)
                    spine.set_edgecolor("#484844")
        fig.subplots_adjust(left=.28/width, right=1-.02/width,
                            bottom=.02/height, top=1-.19/height,
                            wspace=.045, hspace=.22/panel)
        _save_figure(fig, prefix, ("pdf", "png"))
        plt.close(fig)

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--source-root", type=Path, default=cls.SOURCE_ROOT)
        parser.add_argument("--base-root", type=Path, default=Path("data/stgcs_base"))
        parser.add_argument("--output-prefix", type=Path, default=Path("latex/figs/demo_large_2d"))
        args = parser.parse_args()
        cases, sources = cls.load(args.source_root, args.base_root)
        report = dict(planner_settings=cls.SETTINGS, low_level=cls.LOW_LEVEL_SPEC.name, tmax=cls.TMAX,
            planning_budget=cls.BUDGET, trajopt_settings=cls.TRAJOPT_SETTINGS, layout=[2, 2],
            sources=sources, cases=[])
        timings, makespans = [], []
        for case in cases:
            stages = []
            for stage in case["stages"]:
                p = stage["payload"]
                runtime = p["runtime"] if stage["stage"] == "mrmp" else p["global_trajopt"]["runtime"]
                stages.append(dict(stage=stage["stage"], runtime=runtime, total_runtime=p["runtime"],
                    sum_of_costs=p["cost"], makespan=p["makespan"], source=stage["source"],
                    validation=stage["validation"]))
            report["cases"].append(dict(name=case["name"], num_agents=case["num_agents"], seed=case["seed"],
                instance_id=case["instance_id"], workspace=[case["env"].lb.tolist(), case["env"].ub.tolist()],
                robot_radius=float(case["env"].robot_radius), marker_units="workspace",
                wall_polygons=[obstacle.vertices.tolist() for obstacle in case["env"].O_Static], stages=stages))
            timings.append("/".join(f"{stage['runtime']:.2f}" for stage in stages))
            makespans.append(f"{stages[-1]['makespan']:.1f}")
        args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        cls.draw(cases, args.output_prefix)
        args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2))
        caption = ("Large-scale 2D coordination with random start-goal pairs using Windowed-PBS+BFS "
            r"with $\alpha=2$, $\beta=1$, dynamic windows, NC ordering, $\epsilon=8$, $t_{\max}=100$, "
            "and a 120 s planning budget. The top row shows 50 robots in a four-room workspace "
            f"(seed {cases[0]['seed']}), and the bottom row shows 100 robots in a "
            rf"$10.5\times10.5$ empty workspace (seed {cases[1]['seed']}). "
            "Left: MRMP; right: disk trajectory optimization. Both cases use a 0.10 s sampling interval, "
            "0.50 s optimization windows, and a 0.25 s stride. "
            f"Planning/optimization runtimes are {timings[0]} s and {timings[1]} s, respectively; "
            f"optimized makespans are {makespans[0]} s and {makespans[1]} s. "
            "Hollow start circles have radius $r$, and hollow goal squares have side length $2r$, "
            "with $r=0.10$ and $r=0.15$ for 50 and 100 robots, respectively. Walls are drawn to scale.")
        args.output_prefix.with_suffix(".caption.tex").write_text("\\caption{" + caption + "}\n")
        print(json.dumps(report["cases"], indent=2), flush=True)


if __name__ == "__main__":
    Large2DDemoFigure.main()
