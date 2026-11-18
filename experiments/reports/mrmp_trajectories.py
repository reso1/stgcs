"""Selected favorable 16-robot examples from Figure 16's common-success cohort.

Panel measurements come from saved common-success replays, so their costs
describe the actual plotted trajectories. Use --replay-report to redraw them
without planning again.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from experiments.reports.style import configure_matplotlib_for_latex

configure_matplotlib_for_latex(backend="Agg")

import numpy as np
from shapely.geometry import MultiPoint, Polygon

from baselines.sp_strrtstar import FixedPrioritySTRRTStarPlanner
from baselines.ompl_kcbs import OfficialOMPLKCBS
from baselines.ompl_strrt_star import OfficialOMPLSTRRTStar
from experiments.base import BaseManifestStore
from experiments.manifests.mrmp import load_manifest, save_manifest
from experiments.planners.mrmp import MRMPPerformanceComparison
from experiments.mrmp import MRMPExperiment
from experiments.native import STGCSPlannerClient
from experiments.reports.mrmp_comparison import MRMPPerformanceComparisonReport
from experiments.reports.common import _save_figure, plt
from experiments.environment.collision import collision_checking
from stgcs_native.types import STTrajectory


class Config(MRMPPerformanceComparison):
    WINDOW_SPAN_FACTOR = 2.0


class MRMPQualitativeComparison:
    DOMAINS = {"grid2d": "rand", "maze": "maze", "iris-2d": "iris"}
    PLANNERS = (Config.WINDOWED_PBS_KEY, Config.FIXED_PP_ST_RRT_STAR_FINAL_KEY, Config.KCBS_KEY)
    LABELS = {Config.WINDOWED_PBS_KEY: "Ours", Config.FIXED_PP_ST_RRT_STAR_FINAL_KEY: r"PP+ST-RRT$^*$",
              Config.KCBS_KEY: "K-CBS"}
    BUDGET = 120.0
    NUM_AGENTS = 16
    WINDOW_SPAN_FACTOR = Config.WINDOW_SPAN_FACTOR
    METRICS = ("runtime", "sum_of_costs", "makespan")
    SELECTION = ("Figure 16 common successes across its metric panels, also successful for all three displayed planners; "
                 "prefer lower runtime, sum-of-costs, and makespan than both baselines, "
                 "then lower runtime than both, then the strongest minimum quality ratio.")

    @staticmethod
    def load_native(path, name):
        """Use an isolated native build in this process and its forked workers."""
        spec = importlib.util.spec_from_file_location(name, path.resolve())
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
        return module

    @classmethod
    def rank_candidates(cls, rows, domain):
        candidates = []
        for instance_id in sorted(set.intersection(*(set(v) for v in rows.values()))):
            entries = {key: values[instance_id] for key, values in rows.items()}
            if any(r["domain_key"] != domain or not r["is_success"] or r["budget"] != cls.BUDGET
                   or r["num_agents"] != cls.NUM_AGENTS for r in entries.values()):
                continue
            if any(not np.isfinite(r[field]) or r[field] <= 0
                   for r in entries.values() for field in cls.METRICS):
                raise ValueError(f"Invalid successful measurement: {instance_id}")
            ratios = {field: min(entries[p][field] / entries[cls.PLANNERS[0]][field]
                                 for p in cls.PLANNERS[1:]) for field in cls.METRICS}
            candidates.append(dict(instance_id=instance_id, ratios=ratios, benchmark=entries))
        return sorted(candidates, key=cls.selection_key, reverse=True)

    @staticmethod
    def selection_key(row):
        ratios = row["ratios"]
        return (all(v >= 1 for v in ratios.values()), ratios["runtime"] >= 1,
                min(ratios["sum_of_costs"], ratios["makespan"]), ratios["sum_of_costs"], ratios["runtime"])

    @classmethod
    def select(cls, results_root, records, comparison_report, overrides=None):
        overrides = overrides or {}
        if set(overrides) - set(cls.DOMAINS):
            raise ValueError("Unknown domain in selected-instance override")
        comparison = json.loads(comparison_report.read_text())
        if comparison["budget"] != cls.BUDGET or comparison["partial"]:
            raise ValueError("Figure 16 must contain the complete 120-second comparison")
        sources = [str(comparison_report)]
        selected = {}
        for domain in cls.DOMAINS:
            rows = MRMPPerformanceComparisonReport.rows_for_domain(comparison["rows"], domain)
            expected = {r.instance_id for r in records.values() if r.domain_key == domain}
            for key in rows:
                path = Config.result_path(results_root, cls.NUM_AGENTS, domain, key)
                sources.append(str(path))
                cohort = {r["instance_id"]: r for r in rows[key]
                          if r["budget"] == cls.BUDGET and r["num_agents"] == cls.NUM_AGENTS}
                if set(cohort) != expected:
                    raise ValueError(f"Incomplete 120-second cohort: {path}")
                actual = {r["instance_id"]: r for r in MRMPPerformanceComparisonReport._load_domain_rows(path, domain)
                          if r["budget"] == cls.BUDGET and r["num_agents"] == cls.NUM_AGENTS}
                if (set(actual) != expected or any(
                        actual[i][field] != cohort[i][field]
                        for i in expected for field in ("is_success", *cls.METRICS))):
                    raise ValueError(f"Measurements differ from the Figure 16 snapshot: {path}")
            contexts = {field: MRMPPerformanceComparisonReport.common_success_metric_context(
                rows, cls.BUDGET,
                MRMPPerformanceComparisonReport.planner_keys_for_panel(tuple(rows), field),
                cls.NUM_AGENTS, field) for field in cls.METRICS}
            common = set.intersection(*(ids for _, ids in contexts.values()))
            candidates = {key: {r["instance_id"]: r for r in rows[key] if r["instance_id"] in common}
                          for key in cls.PLANNERS}
            ranked = cls.rank_candidates(candidates, domain)
            if not ranked:
                raise ValueError(f"No Figure 16 common-success instance for all displayed planners: {domain}")
            choice = next((row for row in ranked if row["instance_id"] == overrides[domain]), None) if domain in overrides else ranked[0]
            if choice is None:
                raise ValueError(f"Selected instance is not a common success: {overrides[domain]}")
            selected[domain] = dict(choice, eligible_instances=len(ranked),
                figure16_common_success_ids=sorted(common),
                figure16_eligible_planners={field: planners for field, (planners, _) in contexts.items()},
                selection_source="explicit Figure 16 common-success example" if domain in overrides else "ranked Figure 16 common successes")
        return selected, sources

    @staticmethod
    def reconstruct(record, base_root):
        base = BaseManifestStore.record_by_id(base_root / record.domain_key / "manifest.json", record.base_instance_id)
        return MRMPExperiment.reconstruct_instance(record, base, compute_heuristics=False)

    @classmethod
    def solve(cls, key, instance, queries, backend):
        if key == Config.WINDOWED_PBS_KEY:
            solutions, entry = backend.run_with_solutions(instance, queries, Config.LOW_LEVEL_SPEC, cls.BUDGET,
                "num_conflicts", cls.WINDOW_SPAN_FACTOR, Config.EXECUTION_HORIZON_FACTOR,
                Config.DYNAMIC_WINDOW_ADJUSTMENT, "pbs")
        elif key == Config.FIXED_PP_ST_RRT_STAR_FINAL_KEY:
            snapshot = FixedPrioritySTRRTStarPlanner().solve(instance, queries, timeout_secs=cls.BUDGET).final_solution
            solutions = snapshot.solutions
            entry = MRMPExperiment._entry_from_fixed_priority_strrt_star_snapshot(snapshot, queries)
        else:
            solutions, entry = MRMPExperiment.run_ompl_kcbs_with_solutions(instance, queries, budget=cls.BUDGET,
                low_level_solve_time=Config.KCBS_LOW_LEVEL_SOLVE_TIME,
                propagation_step_size=Config.KCBS_PROPAGATION_STEP_SIZE,
                min_control_duration=Config.KCBS_MIN_CONTROL_DURATION,
                max_control_duration=Config.KCBS_MAX_CONTROL_DURATION, goal_tolerance=Config.KCBS_GOAL_TOLERANCE,
                goal_bias=Config.KCBS_GOAL_BIAS, intermediate_states=Config.KCBS_INTERMEDIATE_STATES,
                num_threads=Config.KCBS_NUM_THREADS, seed=Config.KCBS_SEED)
        return dict(success=bool(entry.is_success), **{field: float(getattr(entry, field)) for field in cls.METRICS},
                    trajectories=[np.asarray(s.points).tolist() for s in solutions] if solutions else [])

    @classmethod
    def validate(cls, result, key, env, queries, horizon, allow_kcbs_outside_workspace=False):
        if not result["success"] or len(result["trajectories"]) != len(queries):
            raise ValueError(f"Replay failed: {key}")
        tol = 1e-6
        paths = [np.asarray(p, dtype=float) for p in result["trajectories"]]
        obstacles = [Polygon(obstacle.vertices) for obstacle in env.O_Static]
        corners = max(0., env.robot_radius - tol) * np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        durations, arrivals, bounds_violations = [], [], []
        for index, (path, query) in enumerate(zip(paths, queries)):
            if path.ndim != 2 or path.shape[1] != 6 or len(path) == 0 or not np.isfinite(path).all():
                raise ValueError("Invalid trajectory shape or coordinates")
            a, b = path[:, :3], path[:, 3:]
            goal_tol = Config.KCBS_GOAL_TOLERANCE if key == Config.KCBS_KEY else tol
            if (np.max(np.abs(a[0, :2] - query.start)) > tol
                    or abs(a[0, 2] - query.t_start) > tol
                    or np.max(np.abs(b[-1, :2] - query.goal)) > goal_tol + tol
                    or (len(path) > 1 and np.max(np.abs(a[1:] - b[:-1])) > tol)):
                raise ValueError("Discontinuous trajectory or mismatched query")
            dt = b[:, 2] - a[:, 2]
            if np.any(dt < -tol) or np.any(np.abs(b[:, :2] - a[:, :2]) > query.vlimit * dt[:, None] + tol):
                raise ValueError("Trajectory violates time or speed bounds")
            lower, upper = np.minimum(a[:, :2], b[:, :2]).min(axis=0), np.maximum(a[:, :2], b[:, :2]).max(axis=0)
            if np.any(lower < np.asarray(env.lb) - tol) or np.any(upper > np.asarray(env.ub) + tol):
                if key != Config.KCBS_KEY or not allow_kcbs_outside_workspace:
                    raise ValueError("Trajectory violates workspace bounds")
                bounds_violations.append(dict(robot=index + 1, lower=lower.tolist(), upper=upper.tolist()))
            # K-CBS validates propagated states at its configured 0.1 s step;
            # its native PointGoalRegion also uses the infinity-norm tolerance.
            checks = [(point, point) for point in np.vstack((a, b))] if key == Config.KCBS_KEY else zip(a, b)
            for start, end in checks:
                # The swept square is a convex hull. This exact polygon test
                # avoids an LP solver's feasibility tolerance at wall contacts.
                swept = MultiPoint(np.vstack((start[:2] + corners, end[:2] + corners))).convex_hull
                if any(swept.intersects(obstacle) for obstacle in obstacles):
                    raise ValueError("Trajectory intersects a static obstacle")
                if not env.O_Static and not env.is_segment_in_CSpace(start[:2], end[:2], tol=tol):
                    raise ValueError("Trajectory leaves the free-space union")
            durations.append(b[-1, 2] - a[0, 2])
            arrivals.append(b[-1, 2])
        if not np.allclose([sum(durations), max(arrivals)],
                           [result["sum_of_costs"], result["makespan"]], rtol=1e-8, atol=tol):
            raise ValueError("Trajectory costs differ from reported measurements")
        if key == Config.KCBS_KEY:
            # Match DiskStateValidityChecker.areStatesValid in the benchmark
            # adapter, which uses Euclidean separation at propagated states.
            times = np.arange(0., max(arrivals) + tol, Config.KCBS_PROPAGATION_STEP_SIZE)
            sampled = []
            for path in paths:
                trajectory = STTrajectory(list(range(len(path))), list(path), 2)
                sampled.append(np.asarray([trajectory.lerp(t)[:2] for t in times]))
            for i, points in enumerate(sampled):
                for j in range(i):
                    if np.any(np.linalg.norm(points - sampled[j], axis=1) < 2 * env.robot_radius - tol):
                        raise ValueError(f"Sampled inter-robot collision: {i}, {j}")
            return dict(trajectories=len(paths), collision_model="sampled static squares and inter-robot disks",
                        collision_check_step=Config.KCBS_PROPAGATION_STEP_SIZE, tolerance=tol,
                        workspace_bounds_valid=not bounds_violations, workspace_bounds_violations=bounds_violations)
        # Match PP + ST-RRT*'s benchmark collision checker: its endpoint
        # reservations use the base horizon, without capping arrival times.
        pair_tolerance = 1e-4 if key == Config.FIXED_PP_ST_RRT_STAR_FINAL_KEY else tol
        pair_horizon = horizon if key == Config.FIXED_PP_ST_RRT_STAR_FINAL_KEY else max(horizon, max(arrivals))
        cache, profile = {}, {"cc": np.zeros(2)}
        for i, path in enumerate(paths):
            for j in range(i):
                if collision_checking(list(path), list(paths[j]), space_dim=2, robot_radius=env.robot_radius,
                        t0=0., tmax=pair_horizon, profiler=profile, collision_geometry_cache=cache,
                        tolerance=pair_tolerance, cc_to_tf_a=queries[i].is_stay, cc_to_tf_b=queries[j].is_stay):
                    raise ValueError(f"Inter-robot collision: {i}, {j}")
        return dict(trajectories=len(paths), collision_model="continuous squares",
                    continuous_pair_checks=int(profile["cc"][0]), tolerance=tol,
                    pairwise_tolerance=pair_tolerance, goal_stay_horizon=float(pair_horizon))

    @classmethod
    def draw(cls, selected, records, results, base_root, prefix, planners):
        fig, axes = plt.subplots(3, len(planners), figsize=(1.5 * len(planners), 5.0), squeeze=False)
        colors = plt.get_cmap("tab20" if cls.NUM_AGENTS > 10 else "tab10").colors
        for row, (domain, label) in enumerate(cls.DOMAINS.items()):
            instance, queries = cls.reconstruct(records[selected[domain]["instance_id"]], base_root)
            env = instance.env
            all_points = np.vstack([np.asarray(path).reshape(-1, 3)[:, :2]
                                    for key in planners for path in results[domain][key]["trajectories"]])
            pad = env.robot_radius + .06
            lower = np.minimum(env.lb, all_points.min(axis=0)) - pad
            upper = np.maximum(env.ub, all_points.max(axis=0)) + pad
            center, half_width = (lower + upper) / 2, max(upper - lower) / 2
            for col, key in enumerate(planners):
                ax = axes[row, col]
                if col == 0 or domain == "grid2d":
                    for region in env.C_Space:
                        ax.fill(*np.asarray(region).T, facecolor="#eeeeee", edgecolor="#d4d4d4", lw=.35)
                for obstacle in env.O_Static:
                    ax.fill(*obstacle.vertices.T, facecolor="black", edgecolor="black", alpha=.75, lw=.4)
                result = results[domain][key]
                for index, (segments, query) in enumerate(zip(result["trajectories"], queries)):
                    segments = np.asarray(segments)
                    path = np.vstack((segments[0, :3], segments[:, 3:]))
                    ax.plot(path[:, 0], path[:, 1], color=colors[index], lw=.8, zorder=5)
                    for point, marker in ((query.start, "o"), (query.goal, "s")):
                        ax.plot(*point, marker=marker, ms=3, mfc="white", mec=colors[index], mew=.7, zorder=6)
                caveat = r"$^\dagger$" if not result["validation"].get("workspace_bounds_valid", True) else ""
                metrics = "/".join(f"{result[field]:.1f}".removesuffix(".0") for field in cls.METRICS)
                ax.set_title(f"{cls.LABELS[key]}{caveat} ({metrics})", fontsize=7.5, pad=3)
                ax.set(xlim=(center[0]-half_width, center[0]+half_width), ylim=(center[1]-half_width, center[1]+half_width),
                       aspect="equal", xticks=[], yticks=[])
                if col == 0:
                    ax.set_ylabel(label, fontsize=10)
                for spine in ax.spines.values():
                    spine.set_linewidth(.7)
        fig.tight_layout(pad=.45, w_pad=.45, h_pad=.8)
        _save_figure(fig, prefix, ("pdf", "png"))
        plt.close(fig)

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--results-root", type=Path, default=Path(Config.DEFAULT_OUTPUT_ROOT))
        parser.add_argument("--comparison-report", type=Path, default=Path("latex/figs/mrmp_performance_comparison.json"),
                            help="Figure 16 snapshot defining the common-success selection cohort.")
        parser.add_argument("--manifest", type=Path, default=Path(Config.DEFAULT_MANIFEST_ROOT) / f"manifest_n{cls.NUM_AGENTS}.json")
        parser.add_argument("--base-root", type=Path, default=Path("data/stgcs_base"))
        parser.add_argument("--work-dir", type=Path, default=Path("build/fig17-mrmp-qualitative"))
        parser.add_argument("--executable", type=Path, default=Path("stgcs_cplusplus/build/bin/stgcs_planner"))
        parser.add_argument("--output-prefix", type=Path, default=Path("latex/figs/mrmp_qualitative_comparison"))
        parser.add_argument("--replay-report", type=Path,
                            help="Redraw a saved qualitative report without running planners.")
        parser.add_argument("--instance", action="append", default=[], metavar="DOMAIN=INSTANCE_ID",
                            help="Choose another common-success example after inspecting the trajectory replays.")
        parser.add_argument("--planners", nargs="+", choices=cls.PLANNERS, default=list(cls.PLANNERS),
                            help="Planner columns to include; every plotted replay must pass its validation checks.")
        parser.add_argument("--allow-kcbs-outside-workspace", action="store_true",
                            help="Show reported K-CBS paths that leave the workspace, with explicit panel caveats.")
        parser.add_argument("--kcbs-native", type=Path,
                            help="Isolated corrected K-CBS extension; rerun its panels in a separate cache.")
        parser.add_argument("--strrt-native", type=Path,
                            help="Isolated ST-RRT* extension for the trajectory replays.")
        args = parser.parse_args()
        if args.kcbs_native:
            cls.load_native(args.kcbs_native, OfficialOMPLKCBS._NATIVE_MODULE)
        if args.strrt_native:
            cls.load_native(args.strrt_native, OfficialOMPLSTRRTStar._NATIVE_MODULE)
        records = {r.instance_id: r for r in load_manifest(args.manifest)}
        if args.replay_report:
            report = json.loads(args.replay_report.read_text())
            selected, results = report["selected"], report["replays"]
            for domain, choice in selected.items():
                record = records[choice["instance_id"]]
                instance, queries = cls.reconstruct(record, args.base_root)
                for key in args.planners:
                    results[domain][key]["validation"] = cls.validate(
                        results[domain][key], key, instance.env, queries, record.tmax,
                        args.allow_kcbs_outside_workspace)
            cls.draw(selected, records, results, args.base_root, args.output_prefix, args.planners)
            print(f"Redrew {args.output_prefix}", flush=True)
            return
        selected, sources = cls.select(args.results_root, records, args.comparison_report,
                                      dict(item.split("=", 1) for item in args.instance))
        backend = STGCSPlannerClient(args.executable, args.work_dir / "native", args.work_dir / "heuristics")
        identity = dict(budget=cls.BUDGET, num_agents=cls.NUM_AGENTS,
                        low_level=Config.LOW_LEVEL_SPEC.name,
                        window_span_factor=cls.WINDOW_SPAN_FACTOR,
                        strrt_time_upper_bound=None)
        results = {}
        for domain, choice in selected.items():
            record = records[choice["instance_id"]]
            instance, queries = cls.reconstruct(record, args.base_root)
            results[domain] = {}
            for key in args.planners:
                print(f"Replay {domain}: {record.instance_id}, {key}", flush=True)
                output = args.work_dir / record.instance_id / f"{key}.json"
                expected = dict(identity, record=record.to_dict(), planner=key)
                if output.exists():
                    saved = json.loads(output.read_text())
                    if saved["identity"] != expected:
                        raise ValueError(f"Replay inputs changed; use a fresh work directory: {output}")
                    result = saved["result"]
                else:
                    result = cls.solve(key, instance, queries, backend)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(dict(identity=expected, result=result), indent=2))
                print({field: result[field] for field in ("success", *cls.METRICS)}, flush=True)
                result["validation"] = cls.validate(result, key, instance.env, queries, record.tmax,
                                                     args.allow_kcbs_outside_workspace)
                results[domain][key] = result
        report = dict(selection_policy=cls.SELECTION, selected=selected, sources=sources,
                      arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                      replay_identity=identity, replays=results)
        args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2))
        save_manifest(args.output_prefix.with_suffix(".manifest.json"), [records[s["instance_id"]] for s in selected.values()])
        cls.draw(selected, records, results, args.base_root, args.output_prefix, args.planners)
        print(f"Saved {args.output_prefix}", flush=True)


if __name__ == "__main__":
    MRMPQualitativeComparison.main()
