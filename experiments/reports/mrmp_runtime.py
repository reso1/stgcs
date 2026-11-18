from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from experiments.reports.style import configure_matplotlib_for_latex

configure_matplotlib_for_latex(backend="Agg")

from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, PercentFormatter

from experiments.manifests.base import BaseBenchmarkRecord, load_manifest as load_base_manifest
from experiments.manifests.mrmp import MRMPBenchmarkRecord, load_manifest
from experiments.planners.mrmp import MRMPPerformanceComparison
from experiments.mrmp import MRMPExperiment
from experiments.reports.mrmp_comparison import MRMPPerformanceComparisonReport
from experiments.reports.common import PlotPalette, _save_figure, plt


class MRMPWPBSRuntimeBreakdownReport:
    DEFAULT_RESULTS_ROOT = Path(MRMPPerformanceComparison.DEFAULT_OUTPUT_ROOT)
    DEFAULT_OUTPUT_PREFIX = Path("latex/figs/mrmp_wpbs_runtime_breakdown")
    DEFAULT_TABLE_OUTPUT = Path("latex/tables/mrmp_wpbs_runtime_breakdown_table.tex")
    DEFAULT_BASE_ROOT = Path("data/stgcs_base")
    PLANNER_KEY = MRMPPerformanceComparison.WINDOWED_PBS_KEY
    ROBOT_COUNT_VALUES = MRMPPerformanceComparison.num_agent_values()
    ROBOT_COUNT_TICK_LABELS = tuple(str(num_agents) for num_agents in ROBOT_COUNT_VALUES)
    COMPONENT_ORDER = (
        "search_procedure",
        "convex_restriction",
        "ecd",
        "others",
    )
    COMPONENT_LABELS = {
        "search_procedure": "Search",
        "convex_restriction": "Opt.",
        "ecd": "ECD",
        "others": "Others",
    }
    COMPONENT_COLORS = {
        "ecd": "#fac20a",
        "convex_restriction": PlotPalette.BLUE,
        "search_procedure": PlotPalette.HEURISTIC_INTERFACE_TO_SET_TABLE,
        "others": PlotPalette.BLACK,
    }
    COMPONENT_OVERFLOW_REL_TOL = 1e-6
    OTHER_WARNING_FRACTION = 0.20
    FIGURE_SIZE = (5.0, 3.0)
    RUNTIME_ROW_HEIGHT_RATIO = 3.0
    DIFFICULTY_ROW_HEIGHT_RATIO = 1.35
    BAR_WIDTH = 1.18
    DIFFICULTY_FIELD = "query_area_coverage"
    DIFFICULTY_LABEL = "query density"
    DIFFICULTY_COLOR = "#786666"
    DIFFICULTY_Y_LIMITS = (0.0, 0.85)
    DIFFICULTY_TICKS = (0.0, 0.4, 0.8)
    RUNTIME_Y_LIMITS = (0.1, 70.0)
    RUNTIME_Y_TICKS = (0.2, 1.0, 10.0, 60.0)
    RUNTIME_Y_TICK_LABELS = ("0.2", "1", "10", "60")
    FIGURE_FONT_SIZE = 9.0
    LEGEND_FONT_SIZE = FIGURE_FONT_SIZE
    X_TICK_LABEL_SIZE = FIGURE_FONT_SIZE
    X_AXIS_LABEL_SIZE = FIGURE_FONT_SIZE
    Y_AXIS_LABEL_SIZE = FIGURE_FONT_SIZE
    Y_TICK_LABEL_SIZE = FIGURE_FONT_SIZE
    Y_TICK_LABEL_ROTATION = 90.0

    @classmethod
    def domains_for_arg(cls, domain: str) -> tuple[str, ...]:
        return MRMPPerformanceComparisonReport.domains_for_arg(domain)

    @classmethod
    def load_rows(
        cls,
        results_root: str | Path,
        domains: Sequence[str],
        manifest_root: str | Path | None = None,
        base_root: str | Path | None = None,
    ) -> list[Dict[str, object]]:
        rows_by_planner = MRMPPerformanceComparisonReport.load_rows_by_planner(
            results_root,
            domains,
            (cls.PLANNER_KEY,),
        )
        rows = list(rows_by_planner.get(cls.PLANNER_KEY, ()))
        if manifest_root is not None:
            cls.attach_manifest_fields(
                rows,
                cls.load_manifest_records(manifest_root, domains),
                cls.load_base_records(base_root, domains) if base_root is not None else {},
            )
        return rows

    @classmethod
    def manifest_paths(
        cls,
        manifest_root: str | Path,
        domains: Sequence[str],
    ) -> tuple[Path, ...]:
        root = Path(manifest_root)
        if root.is_file():
            return (root.resolve(),)

        paths: list[Path] = []
        combined_path = root / "manifest.json"
        if combined_path.exists():
            paths.append(combined_path.resolve())
        paths.extend(path.resolve() for path in sorted(root.glob("manifest_n*.json")))
        for domain in domains:
            domain_path = root / str(domain) / "manifest.json"
            if domain_path.exists():
                paths.append(domain_path.resolve())
        return tuple(dict.fromkeys(paths))

    @classmethod
    def load_manifest_records(
        cls,
        manifest_root: str | Path,
        domains: Sequence[str],
    ) -> Dict[str, MRMPBenchmarkRecord]:
        requested_domains = set(str(domain) for domain in domains)
        records: Dict[str, MRMPBenchmarkRecord] = {}
        for manifest_path in cls.manifest_paths(manifest_root, domains):
            for record in load_manifest(manifest_path):
                if record.domain_key not in requested_domains:
                    continue
                records[record.instance_id] = record
        return records

    @classmethod
    def base_manifest_paths(
        cls,
        base_root: str | Path,
        domains: Sequence[str],
    ) -> tuple[Path, ...]:
        root = Path(base_root)
        if root.is_file():
            return (root.resolve(),)

        paths: list[Path] = []
        for domain in domains:
            domain_path = root / str(domain) / "manifest.json"
            if domain_path.exists():
                paths.append(domain_path.resolve())
        return tuple(dict.fromkeys(paths))

    @classmethod
    def load_base_records(
        cls,
        base_root: str | Path,
        domains: Sequence[str],
    ) -> Dict[str, BaseBenchmarkRecord]:
        requested_domains = set(str(domain) for domain in domains)
        records: Dict[str, BaseBenchmarkRecord] = {}
        for manifest_path in cls.base_manifest_paths(base_root, domains):
            for record in load_base_manifest(manifest_path):
                if record.domain_key not in requested_domains:
                    continue
                records[record.instance_id] = record
        return records

    @classmethod
    def attach_manifest_fields(
        cls,
        rows: Sequence[Dict[str, object]],
        records_by_instance_id: Mapping[str, MRMPBenchmarkRecord],
        base_records_by_instance_id: Mapping[str, BaseBenchmarkRecord],
    ) -> None:
        for row in rows:
            record = records_by_instance_id.get(str(row.get("instance_id", "")))
            if record is None:
                continue
            row["num_conflicting_pairs"] = record.num_conflicting_pairs
            row["num_conflicting_agents"] = record.num_conflicting_agents
            row["conflict_largest_component"] = record.conflict_largest_component
            base_record = base_records_by_instance_id.get(record.base_instance_id)
            if base_record is not None:
                row[cls.DIFFICULTY_FIELD] = cls.query_area_coverage_ratio(record, base_record)

    @staticmethod
    def cspace_bounds(base_record: BaseBenchmarkRecord) -> tuple[np.ndarray, np.ndarray]:
        env_params = base_record.env_params
        if base_record.domain == "grid":
            space_dim = int(base_record.space_dim)
            if space_dim < 2:
                raise ValueError(f"Grid CSpace dimension must be at least 2, got {space_dim}.")
            first_extent = float(env_params["N"]) + 0.5
            other_extent = float(env_params["M"]) + 0.5
            return np.zeros(space_dim, dtype=float), np.asarray(
                [first_extent, *([other_extent] * (space_dim - 1))],
                dtype=float,
            )
        if base_record.domain == "maze":
            return np.zeros(2, dtype=float), np.asarray(
                [float(env_params["width"]), float(env_params["height"])],
                dtype=float,
            )
        if base_record.domain == "iris-2d":
            square_size = float(env_params["square_size"])
            return np.zeros(2, dtype=float), np.asarray([square_size, square_size], dtype=float)
        raise ValueError(f"Unsupported MRMP CSpace coverage domain {base_record.domain!r}.")

    @classmethod
    @staticmethod
    def convex_hull_area(points: np.ndarray) -> float:
        if points.shape[0] < 3:
            return 0.0
        try:
            return float(ConvexHull(points).volume)
        except QhullError:
            return 0.0

    @classmethod
    def query_area_coverage_ratio(
        cls,
        record: MRMPBenchmarkRecord,
        base_record: BaseBenchmarkRecord,
    ) -> float:
        if not record.queries:
            return 0.0
        points = np.asarray(
            [query.start for query in record.queries] + [query.goal for query in record.queries],
            dtype=float,
        )
        cspace_lb, cspace_ub = cls.cspace_bounds(base_record)
        if points.shape[1] != cspace_lb.shape[0]:
            raise ValueError(
                f"Query dimension {points.shape[1]} does not match CSpace dimension {cspace_lb.shape[0]} "
                f"for {record.instance_id!r}."
            )
        if points.shape[1] != 2:
            raise ValueError(
                f"Query area coverage is defined only for 2D CSpace records, got dimension {points.shape[1]} "
                f"for {record.instance_id!r}."
            )
        cspace_extent = np.maximum(cspace_ub - cspace_lb, 1e-12)
        cspace_area = float(cspace_extent[0] * cspace_extent[1])
        query_area = cls.convex_hull_area(points)
        coverage = query_area / cspace_area
        return min(max(coverage, 0.0), 1.0)

    @classmethod
    def select_budget(cls, rows: Sequence[Mapping[str, object]], requested_budget: float | None) -> float:
        return MRMPPerformanceComparisonReport.select_budget(
            {cls.PLANNER_KEY: list(rows)},
            requested_budget,
        )

    @staticmethod
    def _same_budget(row: Mapping[str, object], budget: float) -> bool:
        return math.isclose(float(row["budget"]), float(budget))

    @staticmethod
    def _finite_nonnegative(value: object) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            return 0.0
        return numeric

    @classmethod
    def row_has_detailed_low_level_profile(cls, row: Mapping[str, object]) -> bool:
        return (bool(row.get("has_disjoint_breakdown_profile", False))
                and MRMPExperiment.profile_values_available(row))

    @classmethod
    def row_components(cls, row: Mapping[str, object]) -> Dict[str, float]:
        if not cls.row_has_detailed_low_level_profile(row):
            instance_id = str(row.get("instance_id", "<unknown>"))
            budget = float(row.get("budget", math.nan))
            raise ValueError(
                "Windowed-PBS+BFS runtime breakdown requires finite, nonnegative low-level profiler values. "
                f"Row {instance_id!r} at budget={budget:g} only has aggregate mp_runtime, "
                "which overlaps convex-restriction time."
            )
        runtime = cls._finite_nonnegative(row["runtime"])
        pbs_runtime = cls._finite_nonnegative(row["pbs_runtime"])
        ecd = cls._finite_nonnegative(row["ecd_runtime"])
        convex_restriction = cls._finite_nonnegative(row["mp_convex_restriction_runtime"])
        collision_checking = cls._finite_nonnegative(row["cc_runtime"])
        gcs_construction = cls._finite_nonnegative(row["mp_gcs_runtime"])
        dominance_check = max(
            cls._finite_nonnegative(row["mp_dominance_check_runtime"])
            - cls._finite_nonnegative(row["mp_dominance_convex_restriction_runtime"]),
            0.0,
        )
        low_level_gub = cls._finite_nonnegative(row["mp_gub_runtime"])
        low_level_search = cls._finite_nonnegative(row["mp_search_runtime"])
        low_level_search_total = low_level_gub + low_level_search
        bfs_overhead = max(
            low_level_search_total
            - convex_restriction
            - dominance_check,
            0.0,
        )
        pbs_overhead = max(
            pbs_runtime
            - ecd
            - collision_checking
            - gcs_construction
            - low_level_gub
            - low_level_search,
            0.0,
        )
        search_procedure = bfs_overhead + pbs_overhead
        raw_other = (
            runtime
            - ecd
            - convex_restriction
            - search_procedure
        )
        overflow_tolerance = max(1e-9, cls.COMPONENT_OVERFLOW_REL_TOL * max(runtime, 1.0))
        if raw_other < -overflow_tolerance:
            instance_id = str(row.get("instance_id", "<unknown>"))
            raise ValueError(
                "Windowed-PBS+BFS runtime breakdown components exceed total runtime for "
                f"row {instance_id!r}: runtime={runtime:.6g}, "
                f"components={runtime - raw_other:.6g}."
            )
        others = max(raw_other, 0.0)
        return {
            "ecd": ecd,
            "convex_restriction": convex_restriction,
            "search_procedure": search_procedure,
            "others": others,
        }

    @classmethod
    def row_difficulty(cls, row: Mapping[str, object]) -> float | None:
        if cls.DIFFICULTY_FIELD not in row:
            return None
        value = cls._finite_nonnegative(row[cls.DIFFICULTY_FIELD])
        return min(value, 1.0)

    @classmethod
    def aggregate_by_num_agents(
        cls,
        rows: Sequence[Mapping[str, object]],
        budget: float,
    ) -> tuple[
        Dict[int, Dict[str, float]],
        Dict[int, float],
        Dict[int, float],
        Dict[int, int],
    ]:
        component_rows: Dict[int, list[Dict[str, float]]] = defaultdict(list)
        difficulty_rows: Dict[int, list[float]] = defaultdict(list)
        runtime_rows: Dict[int, list[float]] = defaultdict(list)
        for row in rows:
            if not cls._same_budget(row, budget):
                continue
            num_agents = int(row["num_agents"])
            components = cls.row_components(row)
            component_rows[num_agents].append(components)
            runtime_rows[num_agents].append(sum(components[component] for component in cls.COMPONENT_ORDER))
            difficulty = cls.row_difficulty(row)
            if difficulty is not None:
                difficulty_rows[num_agents].append(difficulty)

        medians_by_num_agents: Dict[int, Dict[str, float]] = {}
        difficulty_medians_by_num_agents: Dict[int, float] = {}
        runtime_medians_by_num_agents: Dict[int, float] = {}
        counts_by_num_agents: Dict[int, int] = {}
        for num_agents in cls.ROBOT_COUNT_VALUES:
            components = component_rows.get(num_agents, [])
            if not components:
                continue
            counts_by_num_agents[num_agents] = len(components)
            medians_by_num_agents[num_agents] = {
                component: statistics.median(row[component] for row in components)
                for component in cls.COMPONENT_ORDER
            }
            difficulty_values = difficulty_rows.get(num_agents, [])
            if difficulty_values:
                difficulty_medians_by_num_agents[num_agents] = statistics.median(difficulty_values)
            runtime_values = runtime_rows.get(num_agents, [])
            if runtime_values:
                runtime_medians_by_num_agents[num_agents] = statistics.median(runtime_values)
        return (
            medians_by_num_agents,
            difficulty_medians_by_num_agents,
            runtime_medians_by_num_agents,
            counts_by_num_agents,
        )

    @classmethod
    def num_detailed_rows(cls, rows: Sequence[Mapping[str, object]], budget: float) -> int:
        return sum(
            1
            for row in rows
            if cls._same_budget(row, budget) and cls.row_has_detailed_low_level_profile(row)
        )

    @classmethod
    def plot_figure(
        cls,
        medians_by_num_agents: Mapping[int, Mapping[str, float]],
        difficulty_medians_by_num_agents: Mapping[int, float],
        output_prefix: Path,
        formats: Sequence[str],
    ) -> None:
        if not medians_by_num_agents:
            raise ValueError("No Windowed-PBS+BFS runtime rows are available to plot.")
        x_values = np.array(
            [float(num_agents) for num_agents in cls.ROBOT_COUNT_VALUES if num_agents in medians_by_num_agents],
            dtype=float,
        )
        missing_difficulty = [
            int(num_agents)
            for num_agents in x_values
            if int(num_agents) not in difficulty_medians_by_num_agents
        ]
        if missing_difficulty:
            raise ValueError(
                "Windowed-PBS+BFS runtime breakdown difficulty row requires query area coverage fields "
                f"for robot counts {missing_difficulty}. Pass --manifest-root and --base-root."
            )
        fig, (runtime_ax, difficulty_ax) = plt.subplots(
            2,
            1,
            figsize=cls.FIGURE_SIZE,
            sharex=True,
            gridspec_kw={
                "height_ratios": (cls.RUNTIME_ROW_HEIGHT_RATIO, cls.DIFFICULTY_ROW_HEIGHT_RATIO),
                "hspace": 0.16,
            },
        )
        bottoms = np.zeros(len(x_values), dtype=float)
        for component in cls.COMPONENT_ORDER:
            heights = np.array(
                [float(medians_by_num_agents[int(num_agents)][component]) for num_agents in x_values],
                dtype=float,
            )
            runtime_ax.bar(
                x_values,
                heights,
                bottom=bottoms,
                width=cls.BAR_WIDTH,
                color=cls.COMPONENT_COLORS[component],
                edgecolor="white",
                linewidth=0.35,
                label=cls.COMPONENT_LABELS[component],
            )
            bottoms += heights

        difficulty_values = np.array(
            [float(difficulty_medians_by_num_agents[int(num_agents)]) for num_agents in x_values],
            dtype=float,
        )
        difficulty_ax.bar(
            x_values,
            difficulty_values,
            width=cls.BAR_WIDTH,
            color=cls.DIFFICULTY_COLOR,
            edgecolor="white",
            linewidth=0.35,
        )

        cls._style_axes(runtime_ax, x_values, bottoms)
        cls._style_difficulty_axis(difficulty_ax, x_values, difficulty_values)
        runtime_ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.225, 1.0),
            ncol=2,
            frameon=False,
            fontsize=cls.LEGEND_FONT_SIZE,
            borderaxespad=0.0,
            handlelength=1.2,
            handletextpad=0.45,
            columnspacing=2,
        )
        fig.subplots_adjust(left=0.13, right=0.995, bottom=0.13, top=0.82, hspace=0.24)
        _save_figure(fig, output_prefix, formats)

    @classmethod
    def _style_axes(cls, ax: plt.Axes, x_values: np.ndarray, total_values: np.ndarray) -> None:
        ax.set_xlim(min(x_values) - 1.0, max(x_values) + 1.0)
        ax.xaxis.set_major_locator(FixedLocator(tuple(float(value) for value in cls.ROBOT_COUNT_VALUES)))
        ax.set_xticklabels(cls.ROBOT_COUNT_TICK_LABELS)
        ax.set_ylabel("median runtime (s)", fontsize=cls.Y_AXIS_LABEL_SIZE)
        ax.tick_params(axis="x", labelbottom=False)
        del total_values
        ax.set_yscale("log")
        ax.set_ylim(cls.RUNTIME_Y_LIMITS)
        ax.yaxis.set_major_locator(FixedLocator(cls.RUNTIME_Y_TICKS))
        ax.yaxis.set_major_formatter(FixedFormatter(cls.RUNTIME_Y_TICK_LABELS))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(
            axis="y",
            labelsize=cls.Y_TICK_LABEL_SIZE,
            labelleft=True,
            labelrotation=cls.Y_TICK_LABEL_ROTATION,
        )
        ax.grid(axis="y", alpha=0.24, linewidth=0.7)
        ax.set_axisbelow(True)

    @classmethod
    def _style_difficulty_axis(
        cls,
        ax: plt.Axes,
        x_values: np.ndarray,
        difficulty_values: np.ndarray,
    ) -> None:
        ax.set_xlim(min(x_values) - 1.0, max(x_values) + 1.0)
        ax.xaxis.set_major_locator(FixedLocator(tuple(float(value) for value in cls.ROBOT_COUNT_VALUES)))
        ax.set_xticklabels(cls.ROBOT_COUNT_TICK_LABELS)
        ax.set_xlabel("robots", fontsize=cls.X_AXIS_LABEL_SIZE)
        ax.set_ylabel(cls.DIFFICULTY_LABEL, fontsize=cls.Y_AXIS_LABEL_SIZE)
        ax.set_ylim(cls.DIFFICULTY_Y_LIMITS)
        ax.yaxis.set_major_locator(FixedLocator(cls.DIFFICULTY_TICKS))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.tick_params(axis="x", labelsize=cls.X_TICK_LABEL_SIZE)
        ax.tick_params(
            axis="y",
            labelsize=cls.Y_TICK_LABEL_SIZE,
            labelleft=True,
            labelrotation=cls.Y_TICK_LABEL_ROTATION,
        )
        ax.grid(axis="y", alpha=0.20, linewidth=0.6)
        ax.set_axisbelow(True)

    @classmethod
    def format_summary_with_difficulty(
        cls,
        medians_by_num_agents: Mapping[int, Mapping[str, float]],
        difficulty_medians_by_num_agents: Mapping[int, float],
    ) -> str:
        lines = ["robots,total,query_area_coverage,others_fraction," + ",".join(cls.COMPONENT_ORDER)]
        for num_agents in cls.ROBOT_COUNT_VALUES:
            components = medians_by_num_agents.get(num_agents)
            if components is None:
                continue
            total = sum(float(components[component]) for component in cls.COMPONENT_ORDER)
            others_fraction = 0.0 if total <= 0.0 else float(components["others"]) / total
            difficulty = difficulty_medians_by_num_agents.get(num_agents, math.nan)
            values = [f"{float(components[component]):.6g}" for component in cls.COMPONENT_ORDER]
            lines.append(
                f"{num_agents},{total:.6g},{float(difficulty):.6g},{others_fraction:.6g},"
                + ",".join(values)
            )
        return "\n".join(lines)

    @classmethod
    def component_portions(cls, components: Mapping[str, float]) -> Dict[str, float]:
        total = sum(float(components[component]) for component in cls.COMPONENT_ORDER)
        if total <= 0.0:
            return {component: 0.0 for component in cls.COMPONENT_ORDER}
        return {
            component: float(components[component]) / total
            for component in cls.COMPONENT_ORDER
        }

    @staticmethod
    def _format_runtime_number(value: float) -> str:
        if value < 10.0:
            return f"{value:.2f}"
        if value < 100.0:
            return f"{value:.1f}"
        return f"{value:.0f}"

    @classmethod
    def _format_runtime(cls, value: float) -> str:
        return f"{cls._format_runtime_number(value)}s"

    @staticmethod
    def _format_percent(value: float) -> str:
        return f"{100.0 * value:.1f}\\%"

    @classmethod
    def build_latex_table(
        cls,
        medians_by_num_agents: Mapping[int, Mapping[str, float]],
        difficulty_medians_by_num_agents: Mapping[int, float],
        runtime_medians_by_num_agents: Mapping[int, float],
        budget: float,
    ) -> str:
        budget_phrase = (
            "without a runtime limit"
            if math.isinf(float(budget))
            else f"at budget {float(budget):g}s"
        )
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            (
                "\\caption{Windowed-PBS+BFS runtime breakdown "
                f"{budget_phrase}. Runtime and query density are medians; "
                "shares use median component runtimes.}"
            ),
            "\\label{tab:mrmp-wpbs-runtime-breakdown}",
            "\\small",
            "\\setlength{\\tabcolsep}{2.5pt}",
            "\\begin{tabular}{r|c|c|cccc}",
            "\\toprule",
            (
                "\\multicolumn{1}{c|}{\\multirow{2}{*}{$n$}} & "
                "\\multicolumn{1}{c|}{\\multirow{2}{*}{\\begin{tabular}[c]{@{}c@{}}Query\\\\Density\\end{tabular}}} & "
                "\\multicolumn{1}{c|}{\\multirow{2}{*}{\\begin{tabular}[c]{@{}c@{}}Runtime\\\\Total\\end{tabular}}} & "
                "\\multicolumn{4}{|c}{Runtime Breakdown} \\\\"
            ),
            "\\cline{4-7}",
            "\\multicolumn{1}{c|}{} & \\multicolumn{1}{c|}{} & \\multicolumn{1}{c|}{} & "
            "\\rule[-0.7ex]{0pt}{2.8ex}Search & Opt. & ECD & Others \\\\",
            "\\midrule",
        ]
        for num_agents in cls.ROBOT_COUNT_VALUES:
            components = medians_by_num_agents.get(num_agents)
            if components is None:
                continue
            total = sum(float(components[component]) for component in cls.COMPONENT_ORDER)
            portions = cls.component_portions(components)
            difficulty = difficulty_medians_by_num_agents.get(num_agents, math.nan)
            runtime_median = runtime_medians_by_num_agents.get(num_agents, total)
            row = [
                str(num_agents),
                "" if not math.isfinite(float(difficulty)) else cls._format_percent(float(difficulty)),
                cls._format_runtime(float(runtime_median)),
                cls._format_percent(portions["search_procedure"]),
                cls._format_percent(portions["convex_restriction"]),
                cls._format_percent(portions["ecd"]),
                cls._format_percent(portions["others"]),
            ]
            lines.append(" & ".join(row) + " \\\\")
        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def write_latex_table(table_source: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table_source, encoding="utf-8")
        print(f"Saved LaTeX table to {output_path}")

    @classmethod
    def large_other_entries(
        cls,
        medians_by_num_agents: Mapping[int, Mapping[str, float]],
    ) -> list[tuple[int, float, float]]:
        entries: list[tuple[int, float, float]] = []
        for num_agents in cls.ROBOT_COUNT_VALUES:
            components = medians_by_num_agents.get(num_agents)
            if components is None:
                continue
            total = sum(float(components[component]) for component in cls.COMPONENT_ORDER)
            if total <= 0.0:
                continue
            others = float(components["others"])
            others_fraction = others / total
            if others_fraction > cls.OTHER_WARNING_FRACTION:
                entries.append((num_agents, others, others_fraction))
        return entries


class MRMPWPBSRuntimeBreakdownReportCLI:
    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Plot a component-wise Windowed-PBS+BFS runtime breakdown over MRMP robot counts."
        )
        parser.add_argument(
            "--results-root",
            type=Path,
            default=MRMPWPBSRuntimeBreakdownReport.DEFAULT_RESULTS_ROOT,
        )
        parser.add_argument(
            "--output-prefix",
            type=Path,
            default=MRMPWPBSRuntimeBreakdownReport.DEFAULT_OUTPUT_PREFIX,
        )
        parser.add_argument(
            "--manifest-root",
            type=Path,
            default=Path(MRMPPerformanceComparison.DEFAULT_MANIFEST_ROOT),
        )
        parser.add_argument(
            "--base-root",
            type=Path,
            default=MRMPWPBSRuntimeBreakdownReport.DEFAULT_BASE_ROOT,
        )
        parser.add_argument(
            "--table-output",
            type=Path,
            default=MRMPWPBSRuntimeBreakdownReport.DEFAULT_TABLE_OUTPUT,
        )
        parser.add_argument("--formats", nargs="+", default=("png", "pdf"))
        parser.add_argument("--budget", type=float, default=None)
        parser.add_argument(
            "--domain",
            choices=("all", *MRMPPerformanceComparisonReport.DOMAINS),
            default="all",
        )
        return parser.parse_args()

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        domains = MRMPWPBSRuntimeBreakdownReport.domains_for_arg(args.domain)
        rows = MRMPWPBSRuntimeBreakdownReport.load_rows(
            args.results_root,
            domains,
            manifest_root=args.manifest_root,
            base_root=args.base_root,
        )
        budget = MRMPWPBSRuntimeBreakdownReport.select_budget(rows, args.budget)
        try:
            (
                medians_by_num_agents,
                difficulty_medians_by_num_agents,
                runtime_medians_by_num_agents,
                counts_by_num_agents,
            ) = (
                MRMPWPBSRuntimeBreakdownReport.aggregate_by_num_agents(
                    rows,
                    budget,
                )
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if not medians_by_num_agents:
            raise RuntimeError(f"No Windowed-PBS+BFS rows found at budget={budget:g}.")
        detailed_rows = MRMPWPBSRuntimeBreakdownReport.num_detailed_rows(rows, budget)
        if detailed_rows == 0:
            raise RuntimeError(
                "Detailed low-level profiler columns are absent or zero; refusing to draw an overlapping "
                "breakdown from aggregate mp_runtime. Rerun Windowed-PBS+BFS performance-comparison rows "
                "with the current profiler schema first."
            )
        print(
            f"Plotting Windowed-PBS+BFS runtime breakdown at budget={budget:g}; "
            f"rows by robot count: {counts_by_num_agents}"
        )
        print(
            MRMPWPBSRuntimeBreakdownReport.format_summary_with_difficulty(
                medians_by_num_agents,
                difficulty_medians_by_num_agents,
            )
        )
        large_other = MRMPWPBSRuntimeBreakdownReport.large_other_entries(medians_by_num_agents)
        if large_other:
            formatted = ", ".join(
                f"{num_agents} robots: {others:.3g}s ({100.0 * fraction:.1f}%)"
                for num_agents, others, fraction in large_other
            )
            print(
                "Warning: Others is larger than "
                f"{100.0 * MRMPWPBSRuntimeBreakdownReport.OTHER_WARNING_FRACTION:.0f}% "
                f"for {formatted}."
            )
        MRMPWPBSRuntimeBreakdownReport.plot_figure(
            medians_by_num_agents,
            difficulty_medians_by_num_agents,
            args.output_prefix,
            args.formats,
        )
        table_source = MRMPWPBSRuntimeBreakdownReport.build_latex_table(
            medians_by_num_agents,
            difficulty_medians_by_num_agents,
            runtime_medians_by_num_agents,
            budget,
        )
        MRMPWPBSRuntimeBreakdownReport.write_latex_table(table_source, args.table_output)


if __name__ == "__main__":
    MRMPWPBSRuntimeBreakdownReportCLI.main()
