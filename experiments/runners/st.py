"""Schedule ST experiments; all ST-GCS planning and cache computation is C++."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from experiments.manifests.st_planning import STHeuristicAblationResultEntry, STPlanningManifestStore
from experiments.planners.st_planning import STExperimentConfig, STPerformanceComparisonConfig as Config
from experiments.native import STGCSPlannerClient
from experiments.results import STResultStore as Store


class STSuite(STExperimentConfig):
    def __init__(self, args):
        self.args = args
        self.work = args.work_dir.resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.backend = STGCSPlannerClient(args.executable, self.work)
        records = STPlanningManifestStore.load_manifest(args.manifest)
        if len({record.instance_id for record in records}) != len(records):
            raise ValueError("Duplicate instance IDs in manifest")
        domains = sorted({record.source_domain for record in records})
        grouped = [[record for record in records if record.source_domain == domain] for domain in domains]
        self.records = [group[i] for i in range(max(map(len, grouped))) for group in grouped if i < len(group)]
        if args.limit:
            self.records = self.records[:args.limit]
        self.completed = 0
        self.interrupted = False
        self.state = "preparing"
        self.heuristic_budget = getattr(args, "heuristic_budget", self.DEFAULT_HEURISTIC_BUDGET)
        self.inflation_budget = getattr(args, "inflation_budget", self.DEFAULT_INFLATION_BUDGET)
        self.jobs = self.jobs(args.performance_budget, args.ablation_budget, args.suite, args.planner,
                              args.joint_motion, self.inflation_budget, self.heuristic_budget)
        for suite in getattr(args, "append_suite", ()) or ():
            if suite in {job[0] for job in self.jobs}:
                raise ValueError(f"Suite already scheduled: {suite}")
            self.jobs.extend(type(self).jobs(args.performance_budget, args.ablation_budget,
                                             suite, args.planner, args.joint_motion, self.inflation_budget,
                                             self.heuristic_budget))
        if getattr(args, "heuristic_dominance", "both") == "delta_set":
            self.jobs = [job for job in self.jobs if job[0] != "heuristic_ablation"
                         or "delta_set" in job[2].dominance_checks]
        first_planner = getattr(args, "first_planner", None)
        if first_planner is not None and first_planner not in {job[1] for job in self.jobs}:
            raise ValueError(f"Prioritized planner is not in this suite: {first_planner}")
        self.execution_jobs = sorted(self.jobs, key=lambda job: job[1] != first_planner)
        if any(job[-1] == "strrt_matched" for job in self.jobs):
            self.execution_jobs.sort(key=lambda job: not (
                job[0] == "performance_comparison" and job[1] == Config.ST_RRT_MATCHED_BUDGET_PLANNER))
        self.expected = len(self.records) * sum(2 if job[-1] == "strrt" else 1 for job in self.jobs)

    def status(self, **extra):
        data = dict(state=self.state, pid=os.getpid(), workers=self.args.workers,
                    completed_rows=self.completed, expected_rows=self.expected,
                    active=STGCSPlannerClient.active_processes(), updated_at=time.time(), **extra)
        path = self.work / "status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2))
        temporary.replace(path)

    def prepare(self, record):
        path = self.backend.prepare(record, self.args.base_root)
        record_path = self.work / "inputs" / f"{record.instance_id}.json"
        record_path.write_text(json.dumps(record.to_dict()))
        info = self.backend.cache(path, self.args.cache_timeout)
        # Selection caps belong to generation; benchmark the validated saved graph.
        expected = (record.stgcs_num_vertices, record.stgcs_num_edges)
        actual = (info["st_vertices"], info["st_edges"])
        if actual != expected:
            raise ValueError(f"{record.instance_id}: native graph size {actual} differs from manifest {expected}")
        return record

    def matched_budget_source(self, record, budget):
        path = self.args.output_root / "performance_comparison" / f"{Config.ST_RRT_MATCHED_BUDGET_PLANNER}.csv"
        rows = [row for row in Store.load_result_rows(path)
                if Store.result_key(row["instance_id"], row["budget"])
                == Store.result_key(record.instance_id, budget)]
        if len(rows) != 1 or not math.isfinite(rows[0]["runtime"]) or rows[0]["runtime"] < 0:
            raise ValueError(f"Matched ST-RRT* requires one valid B3 runtime for {record.instance_id}")
        return rows[0]

    def run_job(self, record, job):
        suite, name, spec, budget, method = job
        budget_source = self.matched_budget_source(record, budget) if method == "strrt_matched" else None
        runtime_budget = min(budget, budget_source["runtime"]) if budget_source is not None else budget
        log = self.work / "logs" / suite / record.instance_id / f"{name.replace('/', '_')}-{budget:g}s.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        path = self.work / "inputs" / f"{record.instance_id}.txt"
        if method in ("strrt", "strrt_matched", "zeta", "zeta2r"):
            command = [sys.executable, "-m", "experiments.runners.st_baselines",
                       str(path.with_suffix(".json")), str(self.args.base_root), method, str(runtime_budget)]
        else:
            command = self.backend.command(path, spec, budget, method,
                0, Config.micp_rounding_paths(record.stgcs_num_edges))
        receipt_path = log.with_suffix(".json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["command"] != command or receipt["instance"] != record.instance_id:
                raise ValueError(f"Mismatched result receipt: {receipt_path}")
            if method == "strrt_matched" and receipt.get("budget_source") != budget_source:
                raise ValueError(f"Changed B3 runtime source: {receipt_path}")
            result = receipt["result"]
        else:
            result = self.backend.run_process(command, runtime_budget + 30, log)
        if result.get("status") == "hard_timeout":
            result["runtime"] = runtime_budget
        elif method in ("search", "micp", "micpg") and result["runtime"] > budget:
            result.update(success=False, status="timeout", cost=None)
        receipt_path.write_text(json.dumps(dict(instance=record.instance_id,
            suite=suite, planner=name, budget=budget, runtime_budget=runtime_budget,
            budget_source=budget_source, command=command, result=result), indent=2))
        if method in ("strrt", "strrt_matched", "zeta", "zeta2r") and result.get("status") != "hard_timeout":
            return [(planner, STHeuristicAblationResultEntry(**entry)) for planner, entry in result.items()]
        entry = STHeuristicAblationResultEntry(
            bool(result["success"]), float(result["runtime"]),
            math.inf if result["cost"] is None else float(result["cost"]),
            int(result["expanded"]), int(result["generated"]))
        return [(planner, entry) for planner in Config.expand_planner_name(name)]

    @staticmethod
    def measurement_key(record, job):
        _, _, spec, budget, method = job
        return record.instance_id, spec, float(budget), method

    def run_measurements(self, pool, measurements):
        pending = {}
        for record, destinations in measurements:
            missing = []
            for job in destinations:
                suite, name, _, budget, _ = job
                outputs = Config.expand_planner_name(name)
                paths = [Store.planner_output_path(self.args.output_root / suite, planner) for planner in outputs]
                complete = sum(Store.result_key(record.instance_id, budget) in Store.completed_result_keys(path)
                               for path in paths)
                self.completed += complete
                if complete == len(paths):
                    continue
                missing.append(job)
            if missing:
                pending[pool.submit(self.run_job, record, destinations[0])] = (record, missing)
        self.status()
        for future in as_completed(pending):
            record, destinations = pending[future]
            try:
                entries = future.result()
            except BaseException:
                STGCSPlannerClient.stop()
                raise
            for suite, _, _, budget, _ in destinations:
                for name, entry in entries:
                    path = Store.planner_output_path(self.args.output_root / suite, name)
                    if Store.result_key(record.instance_id, budget) not in Store.completed_result_keys(path):
                        Store.append_result_row(path, name, record, budget, entry)
                        self.completed += 1
                        print(f"{self.completed}/{self.expected} {record.instance_id} {name} success={entry.is_success} runtime={entry.runtime:.3f}", flush=True)
            self.status()

    @staticmethod
    def input_record(record):
        values = record.to_dict()
        # Native preparation checks these derived counts separately.
        for key in ("stgcs_num_vertices", "stgcs_num_edges"):
            values.pop(key, None)
        return values

    def execute(self):
        metadata = dict(backend="stgcs_cplusplus", build_type="Release", workers=self.args.workers,
            output_root=str(self.args.output_root.resolve()),
            manifest=str(self.args.manifest.resolve()),
            executable=str(self.backend.executable),
            cache_root=str(self.backend.cache_root),
            instance_ids=[r.instance_id for r in self.records], instances=len(self.records), expected_rows=self.expected, performance_budget=self.args.performance_budget,
            jobs=[dict(suite=suite, planner=name, budget=budget, method=method,
                       **({"runtime_budget_policy": "min(suite_budget, B3 runtime for same instance)"}
                          if method == "strrt_matched" else {}),
                       spec=asdict(spec),
                       key_evaluation=spec.key_evaluation,
                       upper_bound_heuristic=spec.upper_bound_heuristic or "same")
                  for suite, name, spec, budget, method in self.jobs],
            execution_order=[job[1] for job in self.execution_jobs],
            ablation_budget=self.args.ablation_budget, cache_timeout=self.args.cache_timeout, seed_offset=0,
            heuristic_budget=self.heuristic_budget,
            inflation_budget=self.inflation_budget,
            external_baselines={"OMPL": "native OMPL", "Zeta-SIPP": "original Python baseline"},
            runtime_scope="C++ Planner::plan including query setup and GUB; excludes input, ECD and offline caches",
            statistics_scope="Both GUB prepass and main search; generated counts enqueued labels")
        metadata_path = self.work / "metadata.json"
        previous_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
        if previous_metadata:
            old_records = STPlanningManifestStore.load_manifest(self.work / "manifest.json")
            if [self.input_record(r) for r in old_records] != [self.input_record(r) for r in self.records]:
                raise ValueError("Cannot resume with changed instances; use a new work directory")
            previous_jobs = previous_metadata["jobs"]
            current_jobs = json.loads(json.dumps(metadata["jobs"]))
            if previous_jobs != current_jobs:
                retired = set(getattr(self.args, "retire_planner", ()) or ())
                retained = [job for job in previous_jobs if job["planner"] not in retired]
                existing_suites = {job["suite"] for job in previous_jobs}
                additions = current_jobs[len(retained):]
                if (current_jobs[:len(retained)] != retained
                        or any(job["planner"] in retired for job in current_jobs)
                        or (additions and (not getattr(self.args, "append_suite", None)
                            or any(job["suite"] in existing_suites for job in additions)))):
                    raise ValueError("Cannot resume with changed jobs; only append suites or explicitly retire planners")
                for job in additions:
                    for output in Config.expand_planner_name(job["planner"]):
                        if Store.load_result_rows(self.args.output_root / job["suite"] / f"{output}.csv"):
                            raise ValueError("Appended suite has result CSVs without matching run provenance")
        else:
            for suite, name, _, _, _ in self.jobs:
                for output in Config.expand_planner_name(name):
                    if Store.load_result_rows(self.args.output_root / suite / f"{output}.csv"):
                        raise ValueError("Existing result CSVs have no matching run provenance; use a new output directory")
        (self.work / "process.pid").write_text(str(os.getpid()))
        self.status()

        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            pending = {pool.submit(self.prepare, record): record for record in self.records}
            encoded_records = {}
            for future in as_completed(pending):
                try:
                    record = future.result()
                    encoded_records[record.instance_id] = record
                    print("Prepared", record.instance_id, flush=True)
                except BaseException:
                    STGCSPlannerClient.stop()
                    raise
                self.status(prepared=sum(f.done() and f.exception() is None for f in pending))
            self.records = [encoded_records[record.instance_id] for record in self.records]
            metadata_path.write_text(json.dumps(metadata, indent=2))
            self.args.output_root.mkdir(parents=True, exist_ok=True)
            STPlanningManifestStore.save_manifest(self.work / "manifest.json", self.records)
            self.state = "running"
            # Queue one planner across all domains before advancing to the next.
            measurements = {}
            for job in self.execution_jobs:
                for record in self.records:
                    key = self.measurement_key(record, job)
                    measurements.setdefault(key, (record, []))[1].append(job)
            # Finish and persist B3 before submitting any runtime-matched jobs.
            has_matched = any(job[-1] == "strrt_matched" for job in self.jobs)
            prerequisites, remaining = [], []
            for measurement in measurements.values():
                destinations = measurement[1]
                is_budget_source = any(job[0] == "performance_comparison"
                                       and job[1] == Config.ST_RRT_MATCHED_BUDGET_PLANNER
                                       for job in destinations)
                (prerequisites if has_matched and is_budget_source else remaining).append(measurement)
            self.run_measurements(pool, prerequisites)
            self.run_measurements(pool, remaining)
        if self.completed != self.expected:
            raise RuntimeError("Incomplete result count")
        if not self.args.skip_scaling and self.args.suite == "all":
            self.state = "scaling"
            self.status()
            command = [sys.executable, "-m", "experiments.runners.heuristic_scaling",
                       "--output-root", str(self.args.scaling_root), "--native-executable", str(self.backend.executable),
                       "--native-work-dir", str(self.work / "scaling")]
            # Scaling has its own per-candidate 10,000-second budget and stop rule.
            result = self.backend.run_process(command, 100 * 3 * (10000 + 30), self.work / "scaling.log")
            if result.get("status") == "hard_timeout":
                raise RuntimeError("Scaling study exceeded its full schedule")
        self.state = "complete"
        self.status()

    def report(self):
        """Render the four ST experiments after a complete shared sweep."""
        self.state = "plotting"
        self.status()
        for suite, module, stem, budget in (
            ("heuristic_ablation", "st_heuristic_factorial", "heur_ablation", self.heuristic_budget),
            ("dominance_ablation", "st_dominance_gaps", "dc_ablation", self.args.ablation_budget),
            ("heuristic_inflation", "st_inflation", "table_b1", self.inflation_budget),
            ("performance_comparison", "st_comparison", "st_performance_comparison", self.args.performance_budget),
        ):
            output = self.args.output_root / suite
            report_root = Path('latex/tables' if suite == 'heuristic_inflation' else 'latex/figs')
            command = [sys.executable, "-m", f"experiments.reports.{module}",
                       "--results-root", str(output), "--output-prefix", str(report_root / stem), "--budget", str(budget)]
            if suite == "heuristic_ablation":
                command += ["--dominance", getattr(self.args, "heuristic_dominance", "both")]
            if suite in ("dominance_ablation", "heuristic_inflation"):
                command += ["--manifest", str(self.work / "manifest.json")]
            result = self.backend.run_process(command, 300, self.work / f"{module}.log", expect_json=False)
            if result.get("status") == "hard_timeout":
                raise RuntimeError(f"Report timed out: {module}")
        if not self.args.skip_scaling:
            command = [sys.executable, "-m", "experiments.reports.heuristic_scaling",
                       "--scaling-manifest", str(self.args.scaling_root / "manifest.json")]
            self.backend.run_process(command, 300, self.work / "scaling-plot.log", expect_json=False)
        self.state = "complete"
        self.status()

    def stop(self, *_):
        self.interrupted = True
        STGCSPlannerClient.stop()

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--work-dir", type=Path, required=True)
        parser.add_argument("--executable", type=Path, required=True)
        parser.add_argument("--manifest", type=Path, default=Path("data/instances/st_planning/manifest.json"))
        parser.add_argument("--base-root", type=Path, default=Path("data/instances/st_planning/base"))
        parser.add_argument("--output-root", type=Path, default=Path("data/exp_res/st_planning"))
        parser.add_argument("--scaling-root", type=Path, default=Path("data/stgcs_base/heur_computation_time_scaling"))
        parser.add_argument("--workers", type=int, default=1,
                            help="Concurrent experiment workers (default: %(default)s)")
        parser.add_argument("--limit", type=int)
        parser.add_argument("--performance-budget", type=float, default=Config.DEFAULT_BUDGET)
        parser.add_argument("--ablation-budget", type=float, default=cls.DEFAULT_BUDGET,
                            help="Dominance-ablation runtime budget in seconds (default: %(default)s)")
        parser.add_argument("--heuristic-budget", type=float, default=cls.DEFAULT_HEURISTIC_BUDGET,
                            help="Heuristic-ablation runtime budget in seconds (default: %(default)s)")
        parser.add_argument("--inflation-budget", type=float, default=cls.DEFAULT_INFLATION_BUDGET,
                            help="Per-run budget for the delta_pos epsilon sweep (default: %(default)s)")
        parser.add_argument("--cache-timeout", type=float, default=10000)
        parser.add_argument("--suite", choices=("all", *cls.SUITES), default="all")
        parser.add_argument("--append-suite", choices=cls.SUITES, action="append", default=[],
                            help="Append a suite after --suite (repeatable); reuse matching measurements "
                                 "while preserving all existing jobs and provenance on resume")
        parser.add_argument("--retire-planner", action="append", default=[],
                            help="Explicitly retire an exact planner name from a saved schedule; preserve its old results")
        parser.add_argument("--first-planner",
                            help="Run this exact scheduled planner name first, then the rest; may change on resume")
        parser.add_argument("--heuristic-dominance", choices=("both", "delta_set"), default="both",
                            help="Fig. 10 dominance configurations; other suites are unaffected")
        parser.add_argument("--joint-motion", action="store_true",
                            help="Also include joint evaluation in performance comparison; "
                                 "heuristic ablation and inflation always include it")
        parser.add_argument("--planner", nargs="+", choices=Config.planner_cli_choices(),
                            help="Planners to include in performance comparison (default: all)")
        parser.add_argument("--skip-scaling", action="store_true")
        args = parser.parse_args()
        if args.workers < 1 or min(args.performance_budget, args.ablation_budget,
                                  args.heuristic_budget, args.inflation_budget, args.cache_timeout) <= 0:
            parser.error("Workers and time budgets must be positive")
        args.executable = STGCSPlannerClient.rebuild(args.executable)
        suite = cls(args)
        signal.signal(signal.SIGTERM, suite.stop)
        signal.signal(signal.SIGINT, suite.stop)
        try:
            suite.execute()
            if args.suite == "all":
                suite.report()
        except BaseException as error:
            STGCSPlannerClient.stop()
            suite.state = "stopped" if suite.interrupted else "failed"
            suite.status(error=str(error))
            raise


if __name__ == "__main__":
    STSuite.main()
