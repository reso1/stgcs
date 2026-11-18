"""Figure-16-style ST trajectories: B2, S1 (30 s final incumbent), and Z1.

Select rand/maze instances by obstacle spread and use the separate IRIS spread
example. Reuse saved solutions; titles describe those trajectories rather than
the original benchmark rows.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from experiments.reports.style import configure_matplotlib_for_latex

configure_matplotlib_for_latex(backend="Agg")

import numpy as np
from matplotlib.patches import Rectangle

from baselines.ompl_strrt_star import OfficialOMPLSTRRTStar, OfficialOMPLSTRRTStarOptions
from experiments.instance import Instance
from experiments.manifests.st_planning import STPlanningManifestStore
from experiments.planners.st_planning import STPerformanceComparisonConfig as Config
from experiments.reports.common import _save_figure, plt
from experiments.runners.st_baselines import BaselineWorker
from experiments.native import STGCSPlannerClient
from experiments.results import STResultStore as Store


class STQualitativeComparison:
    DOMAINS = {"grid2d": "rand", "maze": "maze", "iris-2d": "iris"}
    PLANNERS = {"B2": Config.DELTA_STATE_PLANNER,
                "S1": Config.ST_RRT_FINAL_PLANNER, "Z1": Config.ZETA_SIPP_PLANNER}
    LABELS = {"B2": "B2 (ours)", "S1": r"S1: ST-RRT$^*$", "Z1": r"Z1: Zeta$^*$-SIPP"}
    BUDGET = 30.0
    PATH_COLOR = "#2563aa"
    OBSTACLE_COLOR = "#d58935"

    @staticmethod
    def obstacle_spread(record):
        placements = record.generation["placements"]
        centers = np.asarray([p["patrol"] for p in placements], dtype=float).mean(axis=1)
        distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        return dict(occupied_regions=len({p["region"] for p in placements}),
                    patrol_center_spacing=float(distances.min(axis=1).mean()))

    @classmethod
    def rank_candidates(cls, rows, records, domain):
        candidates = []
        for instance_id in sorted(set.intersection(*(set(v) for v in rows.values()))):
            entries = {key: values[instance_id] for key, values in rows.items()}
            if any(r["source_domain"] != domain or not r["is_success"]
                   or r["budget"] != cls.BUDGET for r in entries.values()):
                continue
            if any(not np.isfinite(r[field]) or r[field] <= 0
                   for r in entries.values() for field in ("runtime", "cost")):
                raise ValueError(f"Invalid successful measurement: {instance_id}")
            speedup = min(entries[p]["runtime"] / entries["B2"]["runtime"] for p in ("S1", "Z1"))
            cost_ratio = min(entries[p]["cost"] / entries["B2"]["cost"] for p in ("S1", "Z1"))
            candidates.append(dict(instance_id=instance_id, **cls.obstacle_spread(records[instance_id]),
                                   minimum_speedup=speedup, minimum_cost_ratio=cost_ratio, benchmark=entries))
        return sorted(candidates, key=lambda r: (-r["occupied_regions"], -r["patrol_center_spacing"],
                                                -r["minimum_speedup"],
                                                -r["minimum_cost_ratio"], r["instance_id"]))

    @classmethod
    def select(cls, results_root, records):
        rows, sources = {}, []
        for key, planner in cls.PLANNERS.items():
            path = results_root / f"{planner}.csv"
            sources.append(str(path))
            entries = Store.load_result_rows(path)
            rows[key] = {r["instance_id"]: r for r in entries}
            if len(entries) != len(records) or len(rows[key]) != len(entries):
                raise ValueError(f"Incomplete or duplicated benchmark results: {path}")
        if any(set(v) != set(rows["B2"]) for v in rows.values()):
            raise ValueError("Planner instance sets differ")
        if set(records) != set(rows["B2"]):
            raise ValueError("Manifest and benchmark instance sets differ")
        for entries in rows.values():
            for instance_id, entry in entries.items():
                record = records[instance_id]
                if any(entry[key] != getattr(record, key) for key in (
                        "base_instance_id", "source_domain", "stgcs_num_vertices", "stgcs_num_edges")):
                    raise ValueError(f"Benchmark geometry differs from the manifest: {instance_id}")
        selected = {}
        for domain in cls.DOMAINS:
            ranked = cls.rank_candidates(rows, records, domain)
            if not ranked:
                raise ValueError(f"No common-success instance: {domain}")
            selected[domain] = {**ranked[0], "eligible_instances": len(ranked)}
        return selected, sources

    @classmethod
    def load_iris_example(cls, prefix):
        report = json.loads(prefix.with_suffix(".json").read_text())
        records = STPlanningManifestStore.load_manifest(prefix.with_suffix(".manifest.json"))
        if (len(records) != 1 or records[0].source_domain != "iris-2d"
                or report["budget_seconds"] != cls.BUDGET):
            raise ValueError("IRIS example must contain one instance with the benchmark budget")
        record = records[0]
        choice = report["selected"]["iris-2d"]
        results = report["replays"]["iris-2d"]
        if choice["instance_id"] != record.instance_id or set(results) != set(cls.PLANNERS):
            raise ValueError("IRIS example instance or planner results differ")
        for result in results.values():
            if (not result["success"] or not result.get("segments")
                    or result.get("validation") not in ("native_continuous", "continuous_square")
                    or not 0 < result["runtime"] <= cls.BUDGET
                    or not np.isfinite(result["cost"]) or result["cost"] <= 0):
                raise ValueError("IRIS example requires three validated successful trajectories")
        choice = dict(choice, selection_source="separately generated illustration",
            minimum_speedup=min(results[k]["runtime"] / results["B2"]["runtime"] for k in ("S1", "Z1")),
            minimum_cost_ratio=min(results[k]["cost"] / results["B2"]["cost"] for k in ("S1", "Z1")))
        return record, choice, report

    @classmethod
    def solve(cls, key, record, base_root, backend, input_path, output):
        if key == "B2":
            command = backend.command(input_path, Config.DELTA_STATE_SPEC, cls.BUDGET, "search", 0,
                                      Config.micp_rounding_paths(record.stgcs_num_edges))
            result = backend.run_process(command, cls.BUDGET + 30, output.with_suffix(".log"))
            if result.get("success"):
                witness = np.asarray(result["segments"]).reshape(-1, 6).tolist()
                backend.validate_witness(input_path, witness)
                result["validation"] = "native_continuous"
        elif key == "Z1":
            measured = BaselineWorker.run(record, base_root, "zeta", cls.BUDGET, output)
            result = json.loads(output.read_text())
            entry = measured[Config.ZETA_SIPP_PLANNER]
            result.update(success=entry["is_success"], runtime=entry["runtime"], cost=entry["cost"],
                          segments=np.asarray(result.pop("trajectory")).reshape(-1, 2, 3).tolist())
        else:
            env = STGCSPlannerClient.base_instance(record, base_root).env.copy()
            env.O_Dynamic = Instance.dynamic_obstacles_from_specs(record.dynamic_obstacles)
            query = record.query.to_query()
            started = time.perf_counter()
            snapshots = OfficialOMPLSTRRTStar(env, Config.seed_for_record(record, 0), query.vlimit).solve_with_snapshots(
                query.start, query.goal, query.t_start,
                OfficialOMPLSTRRTStarOptions(max_runtime_in_secs=cls.BUDGET, return_first_valid=False,
                    time_upper_bound=record.tmax, stay_at_goal=query.is_stay))
            solution = snapshots.final_solution
            entry = Config.entry_from_shortest_path_solution(solution, time.perf_counter() - started, cls.BUDGET)
            BaselineWorker.validate_zeta_solution(env, query, record.tmax, solution)
            result = dict(success=entry.is_success, runtime=entry.runtime, cost=entry.cost,
                          first_solution_runtime=snapshots.first_solution.time,
                          segments=np.asarray(solution.trajectory).reshape(-1, 2, 3).tolist(),
                          validation="continuous_square")
        if not result.get("success") or result["runtime"] > cls.BUDGET or not result.get("segments"):
            raise RuntimeError(f"Selected instance failed on replay: {record.instance_id}, {key}")
        return result

    @classmethod
    def draw(cls, selected, records, solutions, base_root, prefix):
        domains = [(domain, label) for domain, label in cls.DOMAINS.items() if domain in selected]
        height = 1.6 * len(domains)
        fig, axes = plt.subplots(len(domains), 3, figsize=(4.5, height), squeeze=False)
        for row, (domain, label) in enumerate(domains):
            record = records[selected[domain]["instance_id"]]
            env = STGCSPlannerClient.base_instance(record, base_root).env
            for col, key in enumerate(cls.PLANNERS):
                ax = axes[row, col]
                # Match Figure 16: show the GCS for ours, and rand's free-space
                # union for every planner; obstacle-based baselines need no IRIS cover.
                if key == "B2" or domain == "grid2d":
                    for region in env.C_Space:
                        ax.fill(*np.asarray(region).T, facecolor="#eeeeee", edgecolor="#d4d4d4", lw=.35)
                env.draw_static(ax, alpha=.75)
                for obstacle in record.dynamic_obstacles:
                    pieces = obstacle["segments"]
                    patrol = np.asarray([pieces[0]["start"], *[p["goal"] for p in pieces]])
                    # Each benchmark patrol is a line segment traversed periodically.
                    patrol = np.unique(patrol, axis=0)
                    ax.plot(*patrol.T, "--", color=cls.OBSTACLE_COLOR, alpha=.65, lw=.6)
                    center, radius = np.asarray(pieces[0]["start"]), obstacle["radius"]
                    ax.add_patch(Rectangle(center - radius, 2*radius, 2*radius,
                                           fc=cls.OBSTACLE_COLOR, ec=cls.OBSTACLE_COLOR, alpha=.28, lw=.5))
                result = solutions[domain][key]
                segments = np.asarray(result["segments"])
                path = np.vstack((segments[0, 0], segments[:, 1]))
                ax.plot(path[:, 0], path[:, 1], color=cls.PATH_COLOR, lw=1.2, zorder=5)
                for point, marker in ((record.query.start, "o"), (record.query.goal, "s")):
                    ax.plot(*point, marker=marker, ms=4, mfc="white", mec=cls.PATH_COLOR, mew=.9, zorder=6)
                arrival = result["cost"] + record.query.t_start
                metrics = "/".join(f"{value:.1f}".removesuffix(".0") for value in (result["runtime"], arrival))
                ax.set_title(f"{cls.LABELS[key]} ({metrics})", fontsize=8, pad=4)
                ax.set(xlim=(env.lb[0]-.1, env.ub[0]+.1), ylim=(env.lb[1]-.1, env.ub[1]+.1),
                       aspect="equal", xticks=[], yticks=[])
                if col == 0:
                    ax.set_ylabel(label, fontsize=10)
                for spine in ax.spines.values():
                    spine.set_linewidth(.7)
        fig.tight_layout(pad=.45, w_pad=.4, h_pad=.7)
        _save_figure(fig, prefix, ("pdf", "png"))
        plt.close(fig)

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--results-root", type=Path, default=Path("data/exp_res/st_planning/performance_comparison"))
        parser.add_argument("--manifest", type=Path, default=Path("data/instances/st_planning/manifest.json"))
        parser.add_argument("--base-root", type=Path, default=Path("data/instances/st_planning/base"))
        parser.add_argument("--executable", type=Path, default=Path("stgcs_cplusplus/build/bin/stgcs_planner"))
        parser.add_argument("--work-dir", type=Path, default=Path("build/st-qualitative-comparison"))
        parser.add_argument("--output-prefix", type=Path, default=Path("latex/figs/st_qualitative_comparison"))
        parser.add_argument("--domain", choices=cls.DOMAINS, help="Export one domain instead of all three")
        parser.add_argument("--iris-example", type=Path, default=Path("latex/figs/iris_spread_comparison"),
                            help="Prefix of the separate IRIS example's manifest and saved solutions")
        parser.add_argument("--rerun", action="store_true", help="Replace this figure's saved trajectory replays")
        args = parser.parse_args()
        records = {r.instance_id: r for r in STPlanningManifestStore.load_manifest(args.manifest)}
        selected, sources = cls.select(args.results_root, records)
        if args.domain:
            selected = {args.domain: selected[args.domain]}
        iris_example = None
        if "iris-2d" in selected:
            record, selected["iris-2d"], iris_example = cls.load_iris_example(args.iris_example)
            records[record.instance_id] = record
        backend = STGCSPlannerClient(args.executable, args.work_dir)
        solutions = {}
        for domain, choice in selected.items():
            record = records[choice["instance_id"]]
            print(f"Selected {domain}: {record.instance_id}; {choice['occupied_regions']} obstacle regions; "
                  f"mean nearest patrol-center distance {choice['patrol_center_spacing']:.3f}; "
                  f"minimum speedup {choice['minimum_speedup']:.3f}x; "
                  f"minimum baseline/B2 cost {choice['minimum_cost_ratio']:.3f}", flush=True)
            path = backend.prepare(record, args.base_root)
            backend.cache(path)
            identity = dict(record=record.to_dict(), budget=cls.BUDGET,
                            search_spec=asdict(Config.DELTA_STATE_SPEC))
            identity = json.loads(json.dumps(identity))
            solutions[domain] = {}
            for key in cls.PLANNERS:
                output = args.work_dir / f"{record.instance_id}-{key}.json"
                if output.exists() and not args.rerun:
                    saved = json.loads(output.read_text())
                    if saved.get("identity") != identity or saved.get("planner") != cls.PLANNERS[key]:
                        raise ValueError(f"Changed replay inputs; use --rerun: {output}")
                    result = saved["result"]
                else:
                    if domain == "iris-2d" and not args.rerun:
                        result = iris_example["replays"][domain][key]
                    else:
                        print(f"Replaying {domain} / {key} (one worker, {cls.BUDGET:g}s budget)", flush=True)
                        result = cls.solve(key, record, args.base_root, backend, path, output)
                    output.write_text(json.dumps(dict(identity=identity, planner=cls.PLANNERS[key], result=result), indent=2)+"\n")
                solutions[domain][key] = result
                print(f"{domain} {key}: runtime={result['runtime']:.3f}s cost={result['cost']:.3f}s", flush=True)
        cls.draw(selected, records, solutions, args.base_root, args.output_prefix)
        STPlanningManifestStore.save_manifest(args.output_prefix.with_suffix(".manifest.json"),
                                              [records[c["instance_id"]] for c in selected.values()])
        args.output_prefix.with_suffix(".json").write_text(json.dumps(dict(
            selection_policy="rand/maze: common successes ranked by obstacle regions and patrol-center spacing; "
                             "IRIS: separately generated max-two-per-region example selected by the user",
            budget_seconds=cls.BUDGET, title_metrics="replayed trajectories", sources=sources,
            manifest=str(args.manifest), selected=selected, replays=solutions,
            iris_example=str(args.iris_example) if iris_example is not None else None,
            note="Illustrative selected examples, not aggregate performance; S1 is the final 30 s incumbent."), indent=2)+"\n")


if __name__ == "__main__":
    STQualitativeComparison.main()
