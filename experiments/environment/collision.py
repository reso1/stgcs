from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from pydrake.all import HPolyhedron, MathematicalProgram

from stgcs_native.types import STTrajectory


@dataclass(frozen=True)
class CollisionGeometry:
    segments: np.ndarray
    half_extent: float
    lb: np.ndarray
    ub: np.ndarray

    def __len__(self) -> int:
        return self.segments.shape[0]


CollisionGeometryCacheEntry = Tuple[List[np.ndarray], Tuple[np.ndarray, ...], CollisionGeometry]
CollisionAgentCacheKey = Tuple[object, ...]
CollisionPairCacheKey = Tuple[CollisionAgentCacheKey, CollisionAgentCacheKey, bool, float]


def is_point_colliding(hpoly: HPolyhedron, point: np.ndarray, robot_radius: float) -> bool:
    point = np.asarray(point, dtype=float).reshape(-1)
    dim = hpoly.ambient_dimension()
    if point.shape[0] != dim:
        raise ValueError(f"Point dimension {point.shape[0]} does not match hpoly dimension {dim}")

    inflated_point = HPolyhedron.MakeBox(point - robot_radius, point + robot_radius)
    return hpoly.IntersectsWith(inflated_point)


def is_lineseg_colliding(
    hpoly: HPolyhedron,
    xp: np.ndarray,
    xq: np.ndarray,
    robot_radius: float,
) -> bool:
    xp = np.asarray(xp, dtype=float).reshape(-1)
    xq = np.asarray(xq, dtype=float).reshape(-1)
    prog = MathematicalProgram()
    dim = hpoly.ambient_dimension()
    if xp.shape[0] != dim or xq.shape[0] != dim:
        raise ValueError(f"Segment dimensions {(xp.shape[0], xq.shape[0])} do not match hpoly dimension {dim}")
    if np.allclose(xp, xq):
        return is_point_colliding(hpoly, xp, robot_radius)

    x = prog.NewContinuousVariables(dim, "x")
    lam = prog.NewContinuousVariables(1, "lam")

    prog.AddLinearConstraint(
        A=hpoly.A(),
        lb=-np.inf * np.ones_like(hpoly.b()),
        ub=hpoly.b(),
        vars=x,
    )

    # x_seg(lam) = xp + lam * (xq - xp), lam in [0, 1]
    prog.AddLinearConstraint(
        A=np.ones((1, 1)),
        lb=np.array([0]),
        ub=np.array([1]),
        vars=lam,
    )

    delta = xq - xp
    A = np.zeros((2 * dim, dim + 1))
    b = np.zeros(2 * dim)
    for axis in range(dim):
        # x_axis - lam * delta_axis <= xp_axis + radius
        A[2 * axis, axis] = 1.0
        A[2 * axis, -1] = -delta[axis]
        b[2 * axis] = xp[axis] + robot_radius

        # -x_axis + lam * delta_axis <= radius - xp_axis
        A[2 * axis + 1, axis] = -1.0
        A[2 * axis + 1, -1] = delta[axis]
        b[2 * axis + 1] = robot_radius - xp[axis]

    prog.AddLinearConstraint(
        A=A,
        lb=-np.inf * np.ones_like(b),
        ub=b,
        vars=np.hstack([x, lam]),
    )

    from experiments.environment.geometry import solver, solver_options

    return solver.Solve(prog, solver_options=solver_options).is_success()


def collision_checking(
    pi_a: List[np.ndarray],
    pi_b: List[np.ndarray],
    *,
    space_dim: int,
    robot_radius: float,
    t0: float,
    tmax: float,
    profiler: Dict[str, np.ndarray],
    collision_geometry_cache: Dict[Tuple[Any, ...], CollisionGeometryCacheEntry],
    cc_to_t0: bool = True,
    cc_to_tf: bool = True,
    tolerance: float = 1e-4,
    cc_to_t0_a: bool | None = None,
    cc_to_t0_b: bool | None = None,
    cc_to_tf_a: bool | None = None,
    cc_to_tf_b: bool | None = None,
) -> bool:
    ts = time.perf_counter()
    geometries_a = collision_geometries(
        pi_a,
        cc_to_t0=cc_to_t0 if cc_to_t0_a is None else cc_to_t0_a,
        cc_to_tf=cc_to_tf if cc_to_tf_a is None else cc_to_tf_a,
        space_dim=space_dim,
        robot_radius=robot_radius,
        t0=t0,
        tmax=tmax,
        tolerance=tolerance,
        collision_geometry_cache=collision_geometry_cache,
    )
    geometries_b = collision_geometries(
        pi_b,
        cc_to_t0=cc_to_t0 if cc_to_t0_b is None else cc_to_t0_b,
        cc_to_tf=cc_to_tf if cc_to_tf_b is None else cc_to_tf_b,
        space_dim=space_dim,
        robot_radius=robot_radius,
        t0=t0,
        tmax=tmax,
        tolerance=tolerance,
        collision_geometry_cache=collision_geometry_cache,
    )

    return _record_collision_check(
        collision_geometries_collide(geometries_a, geometries_b),
        ts,
        profiler,
    )


def cached_node_pair_collision(
    node: Any,
    agent_i: int,
    agent_j: int,
    points_by_agent: Dict[int, List[np.ndarray]],
    goal_stay_by_agent: Dict[int, bool],
    geometry_cache: Dict[int, CollisionGeometry],
    *,
    space_dim: int,
    robot_radius: float,
    t0: float,
    tmax: float,
    profiler: Dict[str, np.ndarray],
    collision_geometry_cache: Dict[Tuple[Any, ...], CollisionGeometryCacheEntry],
    collision_pair_cache: Dict[CollisionPairCacheKey, bool],
    cc_to_t0: bool,
    tolerance: float,
) -> bool:
    ts = time.perf_counter()
    key = _collision_pair_cache_key(
        node,
        agent_i,
        agent_j,
        points_by_agent,
        goal_stay_by_agent,
        space_dim=space_dim,
        robot_radius=robot_radius,
        t0=t0,
        tmax=tmax,
        cc_to_t0=cc_to_t0,
        tolerance=tolerance,
    )
    cached = collision_pair_cache.get(key)
    if cached is not None:
        return _record_collision_check(cached, ts, profiler)

    colliding = collision_geometries_collide(
        _node_agent_collision_geometry(
            agent_i,
            points_by_agent,
            goal_stay_by_agent,
            geometry_cache,
            space_dim=space_dim,
            robot_radius=robot_radius,
            t0=t0,
            tmax=tmax,
            collision_geometry_cache=collision_geometry_cache,
            cc_to_t0=cc_to_t0,
            tolerance=tolerance,
        ),
        _node_agent_collision_geometry(
            agent_j,
            points_by_agent,
            goal_stay_by_agent,
            geometry_cache,
            space_dim=space_dim,
            robot_radius=robot_radius,
            t0=t0,
            tmax=tmax,
            collision_geometry_cache=collision_geometry_cache,
            cc_to_t0=cc_to_t0,
            tolerance=tolerance,
        ),
    )
    collision_pair_cache[key] = _record_collision_check(colliding, ts, profiler)
    return colliding


def _collision_pair_cache_key(
    node: Any,
    agent_i: int,
    agent_j: int,
    points_by_agent: Dict[int, List[np.ndarray]],
    goal_stay_by_agent: Dict[int, bool],
    *,
    space_dim: int,
    robot_radius: float,
    t0: float,
    tmax: float,
    cc_to_t0: bool,
    tolerance: float,
) -> CollisionPairCacheKey:
    key_i = _collision_agent_cache_key(
        node,
        agent_i,
        points_by_agent[int(agent_i)],
        bool(goal_stay_by_agent[int(agent_i)]),
        space_dim=space_dim,
        robot_radius=robot_radius,
        t0=t0,
        tmax=tmax,
    )
    key_j = _collision_agent_cache_key(
        node,
        agent_j,
        points_by_agent[int(agent_j)],
        bool(goal_stay_by_agent[int(agent_j)]),
        space_dim=space_dim,
        robot_radius=robot_radius,
        t0=t0,
        tmax=tmax,
    )
    if key_j < key_i:
        key_i, key_j = key_j, key_i
    return (key_i, key_j, bool(cc_to_t0), float(tolerance))


def _collision_agent_cache_key(
    node: Any,
    agent_idx: int,
    points: List[np.ndarray],
    goal_stay: bool,
    *,
    space_dim: int,
    robot_radius: float,
    t0: float,
    tmax: float,
) -> CollisionAgentCacheKey:
    if points:
        start_time = float(points[0][space_dim])
        finish_time = float(points[-1][-1])
    else:
        start_time = t0
        finish_time = t0
    try:
        solution_cache_serial = node.solution_cache_serial(agent_idx)
    except IndexError:
        solution_cache_serial = id(points)
    return (
        int(solution_cache_serial),
        len(points),
        start_time,
        finish_time,
        bool(goal_stay),
        int(space_dim),
        float(t0),
        float(tmax),
        float(robot_radius),
    )


def _node_agent_collision_geometry(
    agent_idx: int,
    points_by_agent: Dict[int, List[np.ndarray]],
    goal_stay_by_agent: Dict[int, bool],
    geometry_cache: Dict[int, CollisionGeometry],
    *,
    space_dim: int,
    robot_radius: float,
    t0: float,
    tmax: float,
    collision_geometry_cache: Dict[Tuple[Any, ...], CollisionGeometryCacheEntry],
    cc_to_t0: bool,
    tolerance: float,
) -> CollisionGeometry:
    if agent_idx not in geometry_cache:
        geometry_cache[agent_idx] = collision_geometries(
            points_by_agent[agent_idx],
            cc_to_t0=cc_to_t0,
            cc_to_tf=bool(goal_stay_by_agent[agent_idx]),
            space_dim=space_dim,
            robot_radius=robot_radius,
            t0=t0,
            tmax=tmax,
            tolerance=tolerance,
            collision_geometry_cache=collision_geometry_cache,
        )
    return geometry_cache[agent_idx]


def _record_collision_check(
    colliding: bool,
    started_at: float,
    profiler: Dict[str, np.ndarray],
) -> bool:
    profiler["cc"] += np.array([1, time.perf_counter() - started_at])
    return colliding


def collision_geometries_collide(
    geometries_a: CollisionGeometry,
    geometries_b: CollisionGeometry,
) -> bool:
    if len(geometries_a) == 0 or len(geometries_b) == 0:
        return False

    for idx_a, idx_b in _time_overlapping_segment_pairs(geometries_a, geometries_b):
        spatial_bounds_overlap = np.all(
            (geometries_a.lb[idx_a, :-1] <= geometries_b.ub[idx_b, :-1])
            & (geometries_b.lb[idx_b, :-1] <= geometries_a.ub[idx_a, :-1])
        )
        if not spatial_bounds_overlap:
            continue
        if _parallelotope_occupancies_intersect(
            geometries_a.segments[idx_a],
            geometries_b.segments[idx_b],
            geometries_a.half_extent,
            geometries_b.half_extent,
        ):
            return True
    return False


def _time_overlapping_segment_pairs(
    geometries_a: CollisionGeometry,
    geometries_b: CollisionGeometry,
    eps: float = 1e-9,
) -> Iterable[Tuple[int, int]]:
    b_start = 0
    b_time_low = geometries_b.lb[:, -1]
    b_time_high = geometries_b.ub[:, -1]
    a_time_bounds = zip(geometries_a.lb[:, -1], geometries_a.ub[:, -1])
    for idx_a, (a_time_low, a_time_high) in enumerate(a_time_bounds):
        while (
            b_start < len(geometries_b)
            and b_time_high[b_start] < a_time_low - eps
        ):
            b_start += 1

        idx_b = b_start
        while (
            idx_b < len(geometries_b)
            and b_time_low[idx_b] <= a_time_high + eps
        ):
            if b_time_high[idx_b] >= a_time_low - eps:
                yield idx_a, idx_b
            idx_b += 1


def _parallelotope_occupancies_intersect(
    segment_a: np.ndarray,
    segment_b: np.ndarray,
    half_extent_a: float,
    half_extent_b: float,
    eps: float = 1e-9,
) -> bool:
    pa, qa = segment_a
    pb, qb = segment_b
    if qa[-1] < pa[-1]:
        pa, qa = qa, pa
    if qb[-1] < pb[-1]:
        pb, qb = qb, pb

    t_low = max(pa[-1], pb[-1])
    t_high = min(qa[-1], qb[-1])
    if t_low > t_high + eps:
        return False

    slope_a, intercept_a = _segment_centerline_affine(pa, qa, eps)
    slope_b, intercept_b = _segment_centerline_affine(pb, qb, eps)
    distance_limit = half_extent_a + half_extent_b

    for rel_slope, rel_intercept in zip(slope_a - slope_b, intercept_a - intercept_b):
        if abs(rel_slope) <= eps:
            if abs(rel_intercept) > distance_limit + eps:
                return False
            continue

        boundary_a = (-distance_limit - rel_intercept) / rel_slope
        boundary_b = (distance_limit - rel_intercept) / rel_slope
        axis_low = min(boundary_a, boundary_b)
        axis_high = max(boundary_a, boundary_b)
        t_low = max(t_low, axis_low)
        t_high = min(t_high, axis_high)
        if t_low > t_high + eps:
            return False

    return True


def _segment_centerline_affine(
    p: np.ndarray,
    q: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    dt = q[-1] - p[-1]
    if dt == 0.0:
        zero_duration_spatial_tol = max(eps, STTrajectory.ZERO_DURATION_SPATIAL_TOL)
        if not np.allclose(p[:-1], q[:-1], atol=zero_duration_spatial_tol, rtol=0.0):
            raise ValueError("Zero-duration collision segment has different spatial endpoints")
        return np.zeros_like(p[:-1]), p[:-1]

    slope = (q[:-1] - p[:-1]) / dt
    return slope, p[:-1] - slope * p[-1]


def _parallelotope_bounds(
    p: np.ndarray,
    q: np.ndarray,
    half_extent: float,
) -> Tuple[np.ndarray, np.ndarray]:
    spatial_low = np.minimum(p[:-1], q[:-1]) - half_extent
    spatial_high = np.maximum(p[:-1], q[:-1]) + half_extent
    return (
        np.concatenate([spatial_low, [min(p[-1], q[-1])]]),
        np.concatenate([spatial_high, [max(p[-1], q[-1])]]),
    )


def _collision_geometry_cache_key(
    pi: List[np.ndarray],
    cc_to_t0: bool,
    cc_to_tf: bool,
    *,
    space_dim: int,
    robot_radius: float,
    t0: float,
    tmax: float,
    tolerance: float,
) -> Tuple[Any, ...]:
    return (
        id(pi),
        len(pi),
        tuple(id(seg) for seg in pi),
        bool(cc_to_t0),
        bool(cc_to_tf),
        int(space_dim),
        float(t0),
        float(tmax),
        float(robot_radius),
        float(tolerance),
    )


def collision_geometries(
    pi: List[np.ndarray],
    *,
    cc_to_t0: bool,
    cc_to_tf: bool,
    space_dim: int,
    robot_radius: float,
    t0: float,
    tmax: float,
    tolerance: float,
    collision_geometry_cache: Dict[Tuple[Any, ...], CollisionGeometryCacheEntry],
) -> CollisionGeometry:
    key = _collision_geometry_cache_key(
        pi,
        cc_to_t0,
        cc_to_tf,
        space_dim=space_dim,
        robot_radius=robot_radius,
        t0=t0,
        tmax=tmax,
        tolerance=tolerance,
    )
    cached_entry = collision_geometry_cache.get(key)
    if cached_entry is not None:
        return cached_entry[2]

    half_extent = abs(robot_radius - tolerance)
    segments: List[np.ndarray] = []
    lower_bounds: List[np.ndarray] = []
    upper_bounds: List[np.ndarray] = []
    for seg in collision_segments(pi, cc_to_t0=cc_to_t0, cc_to_tf=cc_to_tf, space_dim=space_dim, t0=t0, tmax=tmax):
        p, q = seg[: space_dim + 1], seg[space_dim + 1 :]
        lb, ub = _parallelotope_bounds(p, q, half_extent)
        segments.append(np.vstack([p, q]))
        lower_bounds.append(lb)
        upper_bounds.append(ub)

    if segments:
        segments_arr = np.stack(segments)
        lb_arr = np.vstack(lower_bounds)
        ub_arr = np.vstack(upper_bounds)
        order = np.argsort(lb_arr[:, -1], kind="stable")
        geometries = CollisionGeometry(
            segments_arr[order],
            half_extent,
            lb_arr[order],
            ub_arr[order],
        )
    else:
        geometries = CollisionGeometry(
            np.empty((0, 2, space_dim + 1), dtype=float),
            half_extent,
            np.empty((0, space_dim + 1), dtype=float),
            np.empty((0, space_dim + 1), dtype=float),
        )

    collision_geometry_cache[key] = (pi, tuple(pi), geometries)
    return geometries


def collision_segments(
    pi: List[np.ndarray],
    *,
    cc_to_t0: bool,
    cc_to_tf: bool,
    space_dim: int,
    t0: float,
    tmax: float,
) -> List[np.ndarray]:
    cc_pi = pi.copy()
    if cc_to_t0:
        t0_seg = [
            np.concatenate([
                pi[0][0:space_dim],
                [t0],
                pi[0][: space_dim + 1],
            ])
        ]
        cc_pi = t0_seg + cc_pi
    if cc_to_tf:
        tf_seg = [
            np.concatenate([
                pi[-1][-(space_dim + 1) :],
                pi[-1][-(space_dim + 1) : -1],
                [tmax],
            ])
        ]
        cc_pi = cc_pi + tf_seg
    return cc_pi
