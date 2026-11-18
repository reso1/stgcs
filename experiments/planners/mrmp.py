from __future__ import annotations

from dataclasses import dataclass
import math
import re
from pathlib import Path
from typing import Mapping, Tuple

from stgcs_native.types import DEFAULT_CHILD_EXPANSION_MODE, ChildExpansionMode
from stgcs_native.trajopt import GlobalTrajOptConfig


from stgcs_native.types import SearchPlannerSpec


@dataclass(frozen=True)
class MRMPPlannerSettings:
    child_expansion_rules: Mapping[str, ChildExpansionMode]
    dynamic_window_adjustment: bool


@dataclass(frozen=True)
class TrajoptSettings:
    sample_dt_factor: float
    window_span_factor: float
    stride_factor: float
    velocity_gradient_weight: float
    displacement_weight: float
    displacement_jitter_weight: float
    solver_max_iter: int
    solver_eps: float
    clearance_margin: float
    include_trajectory_knot_times: bool
    collision_tolerance: float

    def window_span(self, robot_radius: float, velocity_limit: float) -> float:
        return GlobalTrajOptConfig.scaled_time(self.window_span_factor, robot_radius, velocity_limit)

    def stride(self, robot_radius: float, velocity_limit: float) -> float:
        return GlobalTrajOptConfig.scaled_time(self.stride_factor, robot_radius, velocity_limit)

    def config(self, robot_radius: float, velocity_limit: float) -> GlobalTrajOptConfig:
        return GlobalTrajOptConfig(
            sample_dt=GlobalTrajOptConfig.scaled_time(self.sample_dt_factor, robot_radius, velocity_limit),
            velocity_gradient_weight=float(self.velocity_gradient_weight),
            displacement_weight=float(self.displacement_weight),
            displacement_jitter_weight=float(self.displacement_jitter_weight),
            clearance_margin=float(self.clearance_margin),
            solver_max_iter=int(self.solver_max_iter),
            solver_eps_abs=float(self.solver_eps),
            solver_eps_rel=float(self.solver_eps),
            include_trajectory_knot_times=bool(self.include_trajectory_knot_times),
            progress=True,
        )


@dataclass(frozen=True)
class PBSExpansionRuleSpec:
    key: str
    label: str
    child_expansion_mode: ChildExpansionMode

    def planner_name(self, low_level_spec: SearchPlannerSpec) -> str:
        return f"PBS-{self.label} + {low_level_spec.name}"


class PBSExpansionAblation:
    LOW_LEVEL_SPEC = SearchPlannerSpec("h_max", ("GUB", "delta_pos"), epsilon=8.0)
    DEFAULT_BUDGET = 60.0
    DEFAULT_MANIFEST_PATH = "data/instances/mrmp/performance_comparison/manifest_n10.json"
    DEFAULT_RULE_KEY = "num_conflicts"
    RULES: Tuple[PBSExpansionRuleSpec, ...] = (
        PBSExpansionRuleSpec("lazy", "Lazy", ChildExpansionMode.LAZY),
        PBSExpansionRuleSpec("soc", "SOC", ChildExpansionMode.SOC),
        PBSExpansionRuleSpec("makespan", "Makespan", ChildExpansionMode.MAKESPAN),
        PBSExpansionRuleSpec("num_conflicts", "NumConflicts", ChildExpansionMode.NUM_CONFLICTS),
    )
    DEFAULT_OUTPUT_ROOT = "data/exp_res/mrmp/pbs_node_expansion_eps8_60s"
    DEFAULT_MANIFEST_ROOT = "data/instances/mrmp/pbs_node_expansion"

    @classmethod
    def rule_by_key(cls, key: str) -> PBSExpansionRuleSpec:
        for rule in cls.RULES:
            if rule.key == key:
                return rule
        raise KeyError(f"Unknown PBS child-expansion rule {key!r}")

    @classmethod
    def default_rule(cls) -> PBSExpansionRuleSpec:
        return cls.rule_by_key(cls.DEFAULT_RULE_KEY)

    @classmethod
    def planner_name(cls, rule: PBSExpansionRuleSpec) -> str:
        return rule.planner_name(cls.LOW_LEVEL_SPEC)

    @classmethod
    def required_heuristics(cls) -> set[str]:
        return {cls.LOW_LEVEL_SPEC.heuristic}


@dataclass(frozen=True)
class WindowedCoordinationSpec:
    window_span_factor: float
    dynamic_window_adjustment: bool
    child_expansion_mode: ChildExpansionMode = DEFAULT_CHILD_EXPANSION_MODE
    execution_horizon_factor: float = 1.0

    @staticmethod
    def factor_label(window_span_factor: float) -> str:
        return f"{float(window_span_factor):g}"

    @staticmethod
    def factor_key(window_span_factor: float) -> str:
        return WindowedCoordinationSpec.factor_label(window_span_factor).replace(".", "p")

    @property
    def mode_key(self) -> str:
        return "dynamic" if self.dynamic_window_adjustment else "fixed"

    @property
    def key(self) -> str:
        base_key = f"{self.mode_key}_{self.factor_key(self.window_span_factor)}"
        if math.isclose(float(self.execution_horizon_factor), 1.0):
            return base_key
        return f"{base_key}_beta{self.factor_key(self.execution_horizon_factor)}"

    @property
    def label(self) -> str:
        mode_label = "Dynamic" if self.dynamic_window_adjustment else "Fixed"
        label = f"{mode_label} {self.factor_label(self.window_span_factor)}"
        if math.isclose(float(self.execution_horizon_factor), 1.0):
            return label
        return f"{label} beta={self.factor_label(self.execution_horizon_factor)}"

    def planner_name(self, low_level_spec: SearchPlannerSpec) -> str:
        rule = PBSExpansionAblation.default_rule()
        if self.window_span_factor == math.inf:
            return rule.planner_name(low_level_spec)
        return f"WC-{self.label} + wPBS-{rule.label} + {low_level_spec.name}"


class WindowedCoordinationAblation:
    LOW_LEVEL_SPEC = PBSExpansionAblation.LOW_LEVEL_SPEC
    DEFAULT_OUTPUT_ROOT = "data/exp_res/mrmp/windowed_coordination"
    DEFAULT_MANIFEST_ROOT = "data/instances/mrmp/windowed_coordination"
    DOMAINS: Tuple[str, ...] = ("grid2d", "maze", "iris-2d")
    DEFAULT_SPAN_FACTORS: Tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, math.inf)
    DEFAULT_MODES: Tuple[str, ...] = ("dynamic",)

    @classmethod
    def spec(
        cls,
        mode: str,
        window_span_factor: float,
        execution_horizon_factor: float = 1.0,
    ) -> WindowedCoordinationSpec:
        if mode not in cls.DEFAULT_MODES:
            raise KeyError(f"Unknown windowed coordination mode {mode!r}")
        return WindowedCoordinationSpec(
            window_span_factor=float(window_span_factor),
            dynamic_window_adjustment=mode == "dynamic",
            child_expansion_mode=PBSExpansionAblation.default_rule().child_expansion_mode,
            execution_horizon_factor=float(execution_horizon_factor),
        )

    @classmethod
    def specs(
        cls,
        window_span_factors: Tuple[float, ...] | list[float] | None = None,
        modes: Tuple[str, ...] | list[str] | None = None,
    ) -> Tuple[WindowedCoordinationSpec, ...]:
        factors = cls.DEFAULT_SPAN_FACTORS if window_span_factors is None else tuple(float(v) for v in window_span_factors)
        mode_values = cls.DEFAULT_MODES if modes is None else tuple(modes)
        return tuple(cls.spec(mode, factor) for factor in factors for mode in mode_values)

    @classmethod
    def planner_name(cls, spec: WindowedCoordinationSpec) -> str:
        return spec.planner_name(cls.LOW_LEVEL_SPEC)

    @classmethod
    def spec_from_planner_name(cls, name: str) -> WindowedCoordinationSpec:
        rule = PBSExpansionAblation.default_rule()
        if name == rule.planner_name(cls.LOW_LEVEL_SPEC):
            return WindowedCoordinationSpec(math.inf, True, rule.child_expansion_mode)
        match = re.fullmatch(r"WC-(Dynamic|Fixed) (\S+?)(?: beta=(\S+))? \+ .*", name)
        if match is None:
            raise ValueError(f"Unknown windowed planner: {name}")
        mode, alpha, beta = match.groups()
        spec = WindowedCoordinationSpec(float(alpha), mode == 'Dynamic', rule.child_expansion_mode,
                                       1.0 if beta is None else float(beta))
        if (not math.isfinite(spec.window_span_factor) or spec.window_span_factor <= 0
                or not math.isfinite(spec.execution_horizon_factor) or spec.execution_horizon_factor <= 0
                or cls.planner_name(spec) != name):
            raise ValueError(f"Invalid windowed planner: {name}")
        return spec

    @classmethod
    def required_heuristics(cls) -> set[str]:
        return {cls.LOW_LEVEL_SPEC.heuristic}


class MRMPPerformanceComparison:
    LOW_LEVEL_SPEC = SearchPlannerSpec("h_max", ("GUB", "delta_pos"), epsilon=8.0)
    DEFAULT_OUTPUT_ROOT = "data/exp_res/mrmp/performance_comparison"
    DEFAULT_MANIFEST_ROOT = "data/instances/mrmp/performance_comparison"
    DOMAINS: Tuple[str, ...] = ("grid2d", "maze", "iris-2d")
    MIN_NUM_AGENTS = 2
    MAX_NUM_AGENTS = 20
    NUM_AGENTS_STEP = 2
    PBS_KEY = "pbs"
    PP_BFS_KEY = "pp-bfs"
    WINDOWED_PBS_KEY = "wpbs"
    WINDOWED_PP_KEY = "wpp"
    PBS_ZETA_SIPP_KEY = "pbs-zeta-sipp"
    FIXED_PP_ST_RRT_STAR_KEY = "fixed-pp-st-rrt-star"
    FIXED_PP_ST_RRT_STAR_FIRST_KEY = "fixed-pp-st-rrt-star-first"
    FIXED_PP_ST_RRT_STAR_FINAL_KEY = "fixed-pp-st-rrt-star-final"
    FIXED_PP_ST_RRT_STAR_OUTPUT_KEYS: Tuple[str, ...] = (
        FIXED_PP_ST_RRT_STAR_FIRST_KEY,
        FIXED_PP_ST_RRT_STAR_FINAL_KEY,
    )
    CB_GCS_KEY = "cb-gcs"
    KCBS_KEY = "kcbs"
    ALL_PLANNER_KEY = "all"
    SEARCH_BASED_PLANNER_KEYS: Tuple[str, ...] = (PBS_KEY, PP_BFS_KEY, WINDOWED_PBS_KEY, WINDOWED_PP_KEY)
    BASELINE_PLANNER_KEYS: Tuple[str, ...] = (
        *FIXED_PP_ST_RRT_STAR_OUTPUT_KEYS,
        PBS_ZETA_SIPP_KEY,
        CB_GCS_KEY,
        KCBS_KEY,
    )
    PLANNER_KEYS: Tuple[str, ...] = SEARCH_BASED_PLANNER_KEYS + BASELINE_PLANNER_KEYS
    DEFAULT_PLANNER_KEYS: Tuple[str, ...] = (
        PBS_KEY, PP_BFS_KEY, *FIXED_PP_ST_RRT_STAR_OUTPUT_KEYS, PBS_ZETA_SIPP_KEY, CB_GCS_KEY,
    )
    PLANNER_SELECTION_KEYS: Tuple[str, ...] = (
        ALL_PLANNER_KEY,
        FIXED_PP_ST_RRT_STAR_KEY,
        *PLANNER_KEYS,
    )
    WINDOW_SPAN_FACTOR = 2.0
    EXECUTION_HORIZON_FACTOR = 1.0
    DYNAMIC_WINDOW_ADJUSTMENT = True
    CHILD_EXPANSION_MODE = ChildExpansionMode.NUM_CONFLICTS
    CHILD_EXPANSION_LABEL = "NumConflicts"
    CB_GCS_TIME_HORIZON = 25.0
    CB_GCS_TIME_STEP = 0.5
    MICP_ROUNDED_PATH_SCALE = 1000.0
    ZETA_SIPP_CELL_SIZE_MULTIPLIER = 1.0
    KCBS_LOW_LEVEL_SOLVE_TIME = 1.0
    KCBS_PROPAGATION_STEP_SIZE = 0.1
    KCBS_MIN_CONTROL_DURATION = 1
    KCBS_MAX_CONTROL_DURATION = 10
    KCBS_GOAL_TOLERANCE = 0.05
    KCBS_GOAL_BIAS = 0.05
    KCBS_INTERMEDIATE_STATES = True
    KCBS_NUM_THREADS = 4
    KCBS_SEED = 0

    @classmethod
    def num_agent_values(cls) -> Tuple[int, ...]:
        return tuple(range(cls.MIN_NUM_AGENTS, cls.MAX_NUM_AGENTS + 1, cls.NUM_AGENTS_STEP))

    @classmethod
    def result_directory(cls, output_root: str | Path, num_agents: int, domain_key: str) -> Path:
        if num_agents not in cls.num_agent_values():
            raise ValueError(f"Unsupported MRMP robot count {num_agents}.")
        if domain_key not in cls.DOMAINS:
            raise ValueError(f"Unknown MRMP performance-comparison domain {domain_key!r}.")
        return Path(output_root) / f"{num_agents}-robot" / domain_key / "results"

    @staticmethod
    def _safe_result_name(planner_name: str) -> str:
        return planner_name.replace("/", "_")

    @classmethod
    def result_path(cls, output_root: str | Path, num_agents: int, domain_key: str, planner_key: str) -> Path:
        safe_name = cls._safe_result_name(cls.planner_name(planner_key))
        return cls.result_directory(output_root, num_agents, domain_key) / f"{safe_name}.csv"

    @classmethod
    def planner_name(cls, planner_key: str) -> str:
        low_level_name = cls.low_level_spec(planner_key).name
        alpha = cls.WINDOW_SPAN_FACTOR
        beta = cls.EXECUTION_HORIZON_FACTOR
        if planner_key == cls.PBS_KEY:
            return f"PBS-{cls.CHILD_EXPANSION_LABEL} + {low_level_name}"
        if planner_key == cls.PP_BFS_KEY:
            return f"PP(order=index) + {low_level_name}"
        if planner_key == cls.WINDOWED_PBS_KEY:
            return f"Windowed PBS(alpha={alpha:g},beta={beta:g},rule=NC) + {low_level_name}"
        if planner_key == cls.WINDOWED_PP_KEY:
            return f"Windowed PP(alpha={alpha:g},beta={beta:g},order=index) + {low_level_name}"
        if planner_key == cls.FIXED_PP_ST_RRT_STAR_FIRST_KEY:
            return "Fixed PP + OMPL ST-RRT*(first)"
        if planner_key == cls.FIXED_PP_ST_RRT_STAR_FINAL_KEY:
            return "Fixed PP + OMPL ST-RRT*(final)"
        if planner_key == cls.PBS_ZETA_SIPP_KEY:
            return f"PBS-{cls.CHILD_EXPANSION_LABEL} + Zeta*-SIPP"
        if planner_key == cls.CB_GCS_KEY:
            return "CB-GCS(paper T-GCS)"
        if planner_key == cls.KCBS_KEY:
            return "OMPL K-CBS"
        raise KeyError(f"Unknown MRMP performance-comparison planner {planner_key!r}")

    @classmethod
    def low_level_spec(cls, planner_key: str) -> SearchPlannerSpec:
        return cls.LOW_LEVEL_SPEC

    @classmethod
    def planner_label(cls, planner_key: str) -> str:
        return cls.planner_name(planner_key)

    @classmethod
    def expand_planner_key(cls, planner_key: str) -> Tuple[str, ...]:
        if planner_key == cls.FIXED_PP_ST_RRT_STAR_KEY:
            return cls.FIXED_PP_ST_RRT_STAR_OUTPUT_KEYS
        return (planner_key,)

    @classmethod
    def required_heuristics(cls) -> set[str]:
        return {cls.LOW_LEVEL_SPEC.heuristic}

    @classmethod
    def micp_rounding_paths(cls, edge_count: int) -> int:
        num_edges = int(edge_count)
        if num_edges <= 1:
            raise ValueError(f"MICP(g) requires |E| > 1 to set rounded paths, got |E|={num_edges}.")
        return int(math.ceil(cls.MICP_ROUNDED_PATH_SCALE * math.log(num_edges)))


HEURISTIC_SPECS: Tuple[SearchPlannerSpec, ...] = (
    SearchPlannerSpec("h_zero", ("GUB",)),
    SearchPlannerSpec("h_mot", ("GUB",)),
    SearchPlannerSpec("h_tri", ("GUB",)),
    SearchPlannerSpec("h_tab", ("GUB",)),
    SearchPlannerSpec("h_max", ("GUB",)),
)

SEARCH_GROUPS = {
    "heuristic": HEURISTIC_SPECS,
}
