from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from experiments.environment.interval import Interval


@dataclass
class ShortestPathSolution:
    is_success: bool
    cost: float
    time: float
    vertex_path: List[str]
    trajectory: List
    itvl: Interval = field(default_factory=lambda: Interval(0.0, 0.0))
    dim: int = 0


def _point_segment_dist_squared(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
    segment = seg_end - seg_start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        diff = point - seg_start
        return float(np.dot(diff, diff))
    t = float(np.dot(point - seg_start, segment)) / denom
    t = min(1.0, max(0.0, t))
    closest = seg_start + t * segment
    diff = point - closest
    return float(np.dot(diff, diff))


def min_dist_squared(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Return the minimum squared distance between two line segments."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)

    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))

    if a <= eps and c <= eps:
        diff = p0 - q0
        return float(np.dot(diff, diff))
    if a <= eps:
        return _point_segment_dist_squared(p0, q0, q1)
    if c <= eps:
        return _point_segment_dist_squared(q0, p0, p1)

    denom = a * c - b * b
    s_numer = 0.0
    s_denom = denom
    t_numer = 0.0
    t_denom = denom

    if denom <= eps:
        s_numer = 0.0
        s_denom = 1.0
        t_numer = e
        t_denom = c
    else:
        s_numer = b * e - c * d
        t_numer = a * e - b * d
        if s_numer < 0.0:
            s_numer = 0.0
            t_numer = e
            t_denom = c
        elif s_numer > s_denom:
            s_numer = s_denom
            t_numer = e + b
            t_denom = c

    if t_numer < 0.0:
        t_numer = 0.0
        if -d < 0.0:
            s_numer = 0.0
        elif -d > a:
            s_numer = s_denom
        else:
            s_numer = -d
            s_denom = a
    elif t_numer > t_denom:
        t_numer = t_denom
        if -d + b < 0.0:
            s_numer = 0.0
        elif -d + b > a:
            s_numer = s_denom
        else:
            s_numer = -d + b
            s_denom = a

    sc = 0.0 if abs(s_numer) <= eps else s_numer / s_denom
    tc = 0.0 if abs(t_numer) <= eps else t_numer / t_denom
    delta = w + sc * u - tc * v
    return float(np.dot(delta, delta))
