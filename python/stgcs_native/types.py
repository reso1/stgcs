"""Problem data and result containers shared by Python callers of the C++ tools."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import math
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Polyhedron:
    A: np.ndarray
    b: np.ndarray


@dataclass(frozen=True)
class Sphere:
    center: np.ndarray
    radius: float


@dataclass(frozen=True)
class Environment:
    lower: np.ndarray
    upper: np.ndarray
    robot_radius: float
    obstacles: Sequence[Sphere | Polyhedron] = ()
    regions: Sequence[Polyhedron] = ()

    @property
    def dim(self) -> int:
        return len(self.lower)


@dataclass(frozen=True)
class PlanningProblem:
    regions: Sequence[Polyhedron]
    robot_radius: float
    horizon: float
    velocity_limit: float


class STTrajectory:
    ZERO_DURATION_SPATIAL_TOL = 1e-7

    def __init__(self, vertex_path:List[str], points:List[np.ndarray], dim:int):
        assert len(vertex_path) == len(points), f"|vertex_path| != |points|"
        self.vertex_path = vertex_path
        self.dim = dim + 1 # include time dimension
        self.points = self._normalize_points(points, self.dim)

    @classmethod
    def _normalize_points(cls, points:List[np.ndarray], state_dim:int) -> List[np.ndarray]:
        return [
            cls._normalize_point(point, state_dim)
            for point in points
        ]

    @classmethod
    def _normalize_point(cls, point:np.ndarray, state_dim:int) -> np.ndarray:
        point_arr = np.asarray(point)
        dt = point_arr[-1] - point_arr[state_dim - 1]
        if dt != 0.0:
            return point

        spatial_delta = np.linalg.norm(point_arr[state_dim:-1] - point_arr[:state_dim - 1])
        if spatial_delta == 0.0 or spatial_delta > cls.ZERO_DURATION_SPATIAL_TOL:
            return point

        normalized = np.array(point_arr, dtype=float, copy=True)
        normalized[state_dim:] = normalized[:state_dim]
        return normalized
    
    def copy(self) -> STTrajectory:
        return STTrajectory(deepcopy(self.vertex_path), deepcopy(self.points), self.dim - 1)

    @property
    def size(self) -> int:
        return len(self.points)
    
    @property
    def x0(self) -> np.ndarray:
        return self.xA(0)

    @property
    def xT(self) -> np.ndarray:
        return self.xB(self.size - 1)
    
    @property
    def duration(self) -> float:
        return self.xT[-1] - self.x0[-1]

    def lerp(self, t:float) -> np.ndarray:
        if t <= self.x0[-1]:
            return self.points[0][:self.dim]

        if t >= self.xT[-1]:
            return self.points[-1][-self.dim:]

        idx = self.find_segment_index(t)
        xa, xb = self.points[idx][:self.dim], self.points[idx][-self.dim:]
        k = (t - xa[-1]) / (xb[-1] - xa[-1])
        return xa * (1 - k) + xb * k

    def find_segment_index(self, t:float) -> int:
        if t <= self.x0[-1]:
            return 0

        if t >= self.xT[-1]:
            return self.size - 1

        for idx in range(self.size):
            xa, xb = self.points[idx][:self.dim], self.points[idx][-self.dim:]
            if round(t - xa[-1], 6) >= 0 and round(t - xb[-1], 6) <= 0:
                break
        
        assert round(t - xa[-1], 6) >= 0 and round(t - xb[-1], 6) <= 0, f"t: {t} not in [{xa[-1]}, {xb[-1]}]"
        return idx
    
    def get_chunk(self, t0:float=None, tf:float=None) -> STTrajectory:
        t0 = self.x0[-1] if t0 is None else t0
        tf = self.xT[-1] if tf is None else tf
        
        # find segment indices for t0 and tf
        idx_0 = self.find_segment_index(t0)
        idx_f = self.find_segment_index(tf)
        if idx_0 == idx_f:
            vertex_path = [self.vertex_path[idx_0]]
            points = [np.hstack([self.lerp(t0), self.lerp(tf)])]
        elif idx_0 == idx_f - 1:
            vertex_path = [self.vertex_path[idx_0], self.vertex_path[idx_f]]
            points = [np.hstack([self.lerp(t0), self.xB(idx_0)]),
                        np.hstack([self.xA(idx_f), self.lerp(tf)])]
        else:
            vertex_path = [self.vertex_path[idx_0]]
            points = [np.hstack([self.lerp(t0), self.xB(idx_0)])]
            for idx in range(idx_0 + 1, idx_f):
                vertex_path.append(self.vertex_path[idx])
                points.append(self.points[idx])
            vertex_path.append(self.vertex_path[idx_f])
            points.append(np.hstack([self.xA(idx_f), self.lerp(tf)]))

        return STTrajectory(vertex_path, points, self.dim - 1)

    def append_chunk(self, chunk:STTrajectory) -> None:
        assert self.size == 0 or np.allclose(self.xT, chunk.x0), "Chunks are not contiguous."
        self.vertex_path.extend(chunk.vertex_path)
        self.points.extend(chunk.points)

    @staticmethod
    def from_chunks(chunks:List[STTrajectory]) -> STTrajectory:
        traj = chunks[0].copy()
        for chunk in chunks[1:]:
            traj.append_chunk(chunk)
        return traj

    def xA(self, idx:int) -> np.ndarray:
        return self.points[idx][:self.dim]
    
    def xB(self, idx:int) -> np.ndarray:
        return self.points[idx][self.dim:]


@dataclass(frozen=True)
class MRMPResultEntry:
    is_success: bool
    runtime: float
    sum_of_costs: float
    makespan: float
    num_completed_agents: int
    pbs_calls: int
    pbs_runtime: float
    pbs_popped_nodes: int
    pbs_generated_children: int
    pbs_update_calls: int
    mp_calls: int
    mp_runtime: float
    ecd_calls: int
    ecd_runtime: float
    cc_calls: int
    cc_runtime: float
    mp_gcs_calls: int = 0
    mp_gcs_runtime: float = 0.0
    mp_gub_calls: int = 0
    mp_gub_runtime: float = 0.0
    mp_search_calls: int = 0
    mp_search_runtime: float = 0.0
    mp_convex_restriction_calls: int = 0
    mp_convex_restriction_runtime: float = 0.0
    mp_dominance_check_calls: int = 0
    mp_dominance_check_runtime: float = 0.0
    mp_dominance_convex_restriction_calls: int = 0
    mp_dominance_convex_restriction_runtime: float = 0.0

    @staticmethod
    def _num_completed_agents(solutions: Sequence[object], queries: Sequence[object]) -> int:
        completed = 0
        for solution, query in zip(solutions, queries):
            if getattr(solution, "size", 0) == 0:
                continue
            if np.allclose(solution.xT[:-1], query.goal, atol=1e-6):
                completed += 1
        return completed

    @staticmethod
    def _solution_metrics(solutions: Sequence[object]) -> Tuple[float, float]:
        return (
            float(sum(solution.duration for solution in solutions)),
            float(max(solution.duration for solution in solutions)),
        )

    @classmethod
    def from_statistics(
        cls,
        solutions: Sequence[object] | None,
        queries: Sequence[object],
        is_success: bool,
        runtime: float,
        pbs_calls: int,
        pbs_runtime: float,
        pbs_popped_nodes: int,
        pbs_generated_children: int,
        pbs_update_calls: int,
        mp_stats: np.ndarray,
        ecd_stats: np.ndarray,
        cc_stats: np.ndarray,
        mp_gcs_stats: np.ndarray | None = None,
        mp_gub_stats: np.ndarray | None = None,
        mp_search_stats: np.ndarray | None = None,
        mp_convex_restriction_stats: np.ndarray | None = None,
        mp_dominance_check_stats: np.ndarray | None = None,
        mp_dominance_convex_restriction_stats: np.ndarray | None = None,
    ) -> MRMPResultEntry:
        mp_gcs_stats = np.zeros(2, dtype=float) if mp_gcs_stats is None else np.asarray(mp_gcs_stats, dtype=float)
        mp_gub_stats = np.zeros(2, dtype=float) if mp_gub_stats is None else np.asarray(mp_gub_stats, dtype=float)
        mp_search_stats = (
            np.zeros(2, dtype=float) if mp_search_stats is None else np.asarray(mp_search_stats, dtype=float)
        )
        mp_convex_restriction_stats = (
            np.zeros(2, dtype=float)
            if mp_convex_restriction_stats is None
            else np.asarray(mp_convex_restriction_stats, dtype=float)
        )
        mp_dominance_check_stats = (
            np.zeros(2, dtype=float)
            if mp_dominance_check_stats is None
            else np.asarray(mp_dominance_check_stats, dtype=float)
        )
        mp_dominance_convex_restriction_stats = (
            np.zeros(2, dtype=float)
            if mp_dominance_convex_restriction_stats is None
            else np.asarray(mp_dominance_convex_restriction_stats, dtype=float)
        )
        if is_success:
            if solutions is None:
                raise ValueError("Successful MRMP coordination must return solutions.")
            sum_of_costs, makespan = cls._solution_metrics(solutions)
            completed_agents = len(queries)
        else:
            sum_of_costs = math.inf
            makespan = math.inf
            completed_agents = cls._num_completed_agents([] if solutions is None else solutions, queries)
        return cls(
            is_success=bool(is_success),
            runtime=float(runtime),
            sum_of_costs=sum_of_costs,
            makespan=makespan,
            num_completed_agents=completed_agents,
            pbs_calls=int(pbs_calls),
            pbs_runtime=float(pbs_runtime),
            pbs_popped_nodes=int(pbs_popped_nodes),
            pbs_generated_children=int(pbs_generated_children),
            pbs_update_calls=int(pbs_update_calls),
            mp_calls=int(mp_stats[0]),
            mp_runtime=float(mp_stats[1]),
            ecd_calls=int(ecd_stats[0]),
            ecd_runtime=float(ecd_stats[1]),
            cc_calls=int(cc_stats[0]),
            cc_runtime=float(cc_stats[1]),
            mp_gcs_calls=int(mp_gcs_stats[0]),
            mp_gcs_runtime=float(mp_gcs_stats[1]),
            mp_gub_calls=int(mp_gub_stats[0]),
            mp_gub_runtime=float(mp_gub_stats[1]),
            mp_search_calls=int(mp_search_stats[0]),
            mp_search_runtime=float(mp_search_stats[1]),
            mp_convex_restriction_calls=int(mp_convex_restriction_stats[0]),
            mp_convex_restriction_runtime=float(mp_convex_restriction_stats[1]),
            mp_dominance_check_calls=int(mp_dominance_check_stats[0]),
            mp_dominance_check_runtime=float(mp_dominance_check_stats[1]),
            mp_dominance_convex_restriction_calls=int(mp_dominance_convex_restriction_stats[0]),
            mp_dominance_convex_restriction_runtime=float(mp_dominance_convex_restriction_stats[1]),
        )


@dataclass
class WindowedCoordinationReturn:
    success: bool = False
    solutions: List[STTrajectory] = field(default_factory=list)
    mp: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    gcs: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    gub: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    search: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    cr: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    dc: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    dc_cr: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    ecd: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    cc: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    wpbs: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    wpp: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    runtime: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    pbs_popped_nodes: int = 0
    pbs_generated_children: int = 0
    pbs_update_calls: int = 0

    def __repr__(self):
        status = "success" if self.success else "partial-or-failed"
        return "WindowedCoordinationReturn:\n" + \
                f"\tStatus: {status}\n" + \
                f"\t# trajectories: {len(self.solutions)}\n" + \
                f"\tRuntime: {self.runtime[0]:.4f} secs\n" + \
                f"\tWPBS: # calls={self.wpbs[0]:.0f}, cum-time={self.wpbs[1]:.4f} secs\n" + \
                f"\tWPP: # calls={self.wpp[0]:.0f}, cum-time={self.wpp[1]:.4f} secs\n" + \
                f"\tWPBS nodes: popped={self.pbs_popped_nodes}, children={self.pbs_generated_children}, updates={self.pbs_update_calls}\n" + \
                f"\tMP: # calls={self.mp[0]:.0f}, cum-time={self.mp[1]:.4f} secs\n" + \
                f"\tGCS build: # calls={self.gcs[0]:.0f}, cum-time={self.gcs[1]:.4f} secs\n" + \
                f"\tGUB: # calls={self.gub[0]:.0f}, cum-time={self.gub[1]:.4f} secs\n" + \
                f"\tSearch: # calls={self.search[0]:.0f}, cum-time={self.search[1]:.4f} secs\n" + \
                f"\tConvex restriction: # calls={self.cr[0]:.0f}, cum-time={self.cr[1]:.4f} secs\n" + \
                f"\tDominance check: # calls={self.dc[0]:.0f}, cum-time={self.dc[1]:.4f} secs\n" + \
                f"\tDominance CR: # calls={self.dc_cr[0]:.0f}, cum-time={self.dc_cr[1]:.4f} secs\n" + \
                f"\tECD: # calls={self.ecd[0]:.0f}, cum-time={self.ecd[1]:.4f} secs\n" + \
                f"\tCC: # calls={self.cc[0]:.0f}, cum-time={self.cc[1]:.4f} secs\n"

class STPlanStatus(str, Enum):
    SUCCESS = "success"
    GUB_FALLBACK = "gub_fallback"
    FAIL = "fail"

    @classmethod
    def from_solution(cls, sol: Optional[STTrajectory]) -> STPlanStatus:
        return cls.SUCCESS if sol is not None else cls.FAIL

@dataclass
class MPQuery:
    start: np.ndarray
    goal: np.ndarray
    t_start: float
    is_stay: bool
    vlimit: float

    def __repr__(self):
        dim = self.start.shape[0]
        x_start = ", ".join([f"{self.start[i]:.4f}" for i in range(dim)])
        p_goal = ", ".join([f"{self.goal[i]:.4f}" for i in range(dim)])
        return f"MPQuery(xs=[{x_start}, {self.t_start:.4f}], pg=[{p_goal}], stay={self.is_stay}, vlimit={self.vlimit})"

@dataclass
class MRMPQuery(Sequence[MPQuery]):
    queries: List[MPQuery]

    @classmethod
    def from_queries(cls, queries: MRMPQuery | Sequence[MPQuery]) -> MRMPQuery:
        if isinstance(queries, cls):
            return queries
        return cls(list(queries))

    def copy(self) -> MRMPQuery:
        return MRMPQuery(deepcopy(self.queries))

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self) -> Iterator[MPQuery]:
        return iter(self.queries)

    def __getitem__(self, index: int | slice) -> MPQuery | List[MPQuery]:
        return self.queries[index]

    def __repr__(self) -> str:
        return f"MRMPQuery(num_agents={len(self.queries)})"

class ChildExpansionMode(IntEnum):
    LAZY = 0    # use in large-scale instances with a large number of robots (e.g., > 50).
    SOC = 1
    MAKESPAN = 2
    NUM_CONFLICTS = 3

DEFAULT_CHILD_EXPANSION_MODE = ChildExpansionMode.NUM_CONFLICTS

@dataclass(frozen=True)
class SearchPlannerSpec:
    heuristic: str
    dominance_checks: Tuple[str, ...]
    epsilon: float = 1.0
    exact_astar: bool = False
    key_evaluation: str = "separate"
    upper_bound_heuristic: str | None = None

    @staticmethod
    def epsilon_label(epsilon: float) -> str:
        value = float(epsilon)
        if math.isclose(value, 1.0):
            return ""
        return f",eps={value:g}"

    @property
    def name(self) -> str:
        if self.exact_astar:
            return "Search(h_tri+Astar)"
        epsilon_label = self.epsilon_label(self.epsilon)
        evaluation_label = ",joint" if self.key_evaluation == "joint" else ""
        prepass_label = f",gub={self.upper_bound_heuristic}" if self.upper_bound_heuristic else ""
        if self.heuristic == "h_zero":
            return f"Search({'+'.join(self.dominance_checks)}{epsilon_label}{prepass_label})"
        return f"Search({self.heuristic}+{'+'.join(self.dominance_checks)}{epsilon_label}{evaluation_label}{prepass_label})"
