from __future__ import annotations
from abc import abstractmethod, ABC
from typing import List, Tuple, Set
from itertools import product, combinations
from collections import defaultdict
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle

import numpy as np
import networkx as nx

from pydrake.all import HPolyhedron, VPolytope

from mrmp.interval import Interval
from mrmp.utils import (
    find_space_time_intersecting_pts, is_lineseg_colliding, squash_multi_points,
    draw_cylinder, draw_3d_set, timeit
)

from mrmp.stgcs import STGCS
from mrmp.ecd import ECDPair, reserve as ecd_reserve
from mrmp.graph import ShortestPathSolution


""" Static Obstacles """


class StaticObstacle(ABC):
    
    @abstractmethod
    def is_colliding(self, point:np.ndarray, robot_radius:float) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_colliding_lineseg(self, p:np.ndarray, q:np.ndarray, robot_radius:float) -> bool:
        raise NotImplementedError

    @abstractmethod
    def draw(self, ax:Axes, color='k') -> None:
        raise NotImplementedError


class StaticSphere(StaticObstacle):

    def __init__(self, pos:np.ndarray, radius:float,) -> None:
        self.pos, self.radius = pos, radius
    
    def is_colliding(self, point:np.ndarray, robot_radius:float) -> bool:
        return np.linalg.norm(self.pos - point) <= self.radius + robot_radius
    
    def is_colliding_lineseg(self, p:np.ndarray, q:np.ndarray, robot_radius:float) -> bool:
        if self.is_colliding(p, robot_radius) or self.is_colliding(q, robot_radius):
            return True
        
        p2m1 = q - p
        p1mc = p - self.pos

        dot = np.dot(p1mc, p2m1)
        squared_norm = np.linalg.norm(p2m1) ** 2
        t = -1 * (dot / squared_norm)
        if t < 0:
            t = 0
        elif t > 1:
            t = 1
        closest = p + p2m1 * t
        return np.linalg.norm(closest - self.pos) <= self.radius + robot_radius

    def draw(self, ax:Axes, color='k') -> None:
        circle = Circle(self.pos, self.radius, color=color)
        ax.add_artist(circle)
    

class StaticPolygon(StaticObstacle):

    def __init__(self, vertices:np.ndarray) -> None:
        self.vertices = vertices
        self.hpoly = HPolyhedron(VPolytope(vertices.T))
    
    def is_colliding(self, point:np.ndarray, robot_radius:float) -> bool:
        assert len(point) == 2
        for xc, yc in [[-1, -1], [1, -1], [1, 1], [-1, 1]]:
            if self.hpoly.PointInSet(point + robot_radius * np.array([xc, yc])):
                return True
        return False

    def is_colliding_lineseg(self, p:np.ndarray, q:np.ndarray, robot_radius:float) -> bool:
        return is_lineseg_colliding(self.hpoly, p, q, robot_radius)

    def draw(self, ax:Axes, alpha=0.8, color='k') -> None:
        ax.fill(self.vertices[:, 0], self.vertices[:, 1], alpha=alpha, fc=color, ec='black')

    def draw_with_time(self, ax:Axes3D, tmax:float, color='k', alpha:float=1.0) -> None:
        hpoly = self.hpoly.CartesianProduct(HPolyhedron.MakeBox([0], [tmax]))
        draw_3d_set(hpoly, ax, alpha=alpha, fc=color)


""" Dynamic Obstacles (assuming uniform speed) """


class DynamicObstacle(ABC):
    x0: np.ndarray
    xt: np.ndarray
    velocity: np.ndarray
    itvl: Interval
    
    @abstractmethod
    def collision_intervals(self, point:np.ndarray, robot_radius:float) -> List[Interval]:
        raise NotImplementedError
    
    @abstractmethod
    def is_colliding_lineseg(self, p:np.ndarray, q:np.ndarray, tp:float, tq:float, robot_radius:float) -> bool:
        raise NotImplementedError

    @abstractmethod
    def x(self, t:float) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def reserve(self, stgcs:STGCS, robot_radius:float, reserve_first_to_t0:bool, reserve_last_to_tf:bool) -> STGCS:
        raise NotImplementedError
    
    @abstractmethod
    def draw(self, ax:Axes3D, *args) -> None:
        raise NotImplementedError


class DynamicSphere(DynamicObstacle):

    def __init__(self, x0:np.ndarray, xt:np.ndarray, radius:float, itvl:Interval) -> None:
        self.x0, self.xt, self.radius, self.itvl = x0, xt, radius, itvl
        self.velocity = (xt - x0) / self.itvl.duration
    
    def collision_intervals(self, point:np.ndarray, robot_radius:float) -> List[Interval]:
        ret = []
        rr = self.radius + robot_radius
        vel_magnitude = np.linalg.norm(self.velocity)
        x0_collision = np.linalg.norm(self.x0 - point) < rr
        xt_collision = np.linalg.norm(self.xt - point) < rr
        if x0_collision and self.itvl.start != 0:
            ret.append(Interval(0.0, self.itvl.start))
        if xt_collision and self.itvl.end != np.inf:
            ret.append(Interval(self.itvl.end, np.inf))

        # solve ||q - f(t)||^2 = r^2   | q = point
        # f(t) = p + v * t             | p = self.x0, v = self.velocity
        a = vel_magnitude ** 2  # a > 0
        b = 2 * np.dot(self.velocity, self.x0 - point)
        c = np.linalg.norm(self.x0 - point) ** 2 - rr ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant >= 0:
            t1 = (-b - np.sqrt(discriminant)) / (2 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2 * a)
            intersection = self.itvl.intersection(Interval(t1, t2))
            if intersection is not None:
                ret.append(intersection)
        
        return ret

    def is_colliding_lineseg(self, p:np.ndarray, q:np.ndarray, tp:float, tq:float, robot_radius:float) -> bool:
        """ assuming obstacle disappears outside the interval """
        itvl = self.itvl.intersection(Interval(tp, tq))
        if itvl is None:
            return False
        
        P0, Q0 = self.x(itvl.start), lerp(p, q, tp, tq, itvl.start)
        P1, Q1 = self.x(itvl.end), lerp(p, q, tp, tq, itvl.end)
        # D(t) = ||(P0 - Q0) + t * (P1 - P0 - Q1 + Q0)||^2, t\in [0, 1]
        #      = || A + t*B||^2 >= r^2
        A = P0 - Q0
        B = P1 - P0 - Q1 + Q0
        # D'(t) = 2 * (A + t*B) * B = 0
        t = np.clip(- np.dot(A, B) / np.dot(B, B), 0, 1)
        min_dist = np.min([
            np.linalg.norm(A + t * B),
            np.linalg.norm(A),          # at t=0
            np.linalg.norm(A + B)       # at t=1
        ])

        return min_dist <= self.radius + robot_radius
        
    def x(self, t:float) -> np.ndarray:
        if t < self.itvl.start:
            return self.x0
        if t > self.itvl.end:
            return self.xt
        
        return self.x0 + self.velocity * (t - self.itvl.start)

    def reserve(self, stgcs:STGCS, robot_radius:float, reserve_first_to_t0:bool=True, reserve_last_to_tf:bool=True) -> STGCS:
        trajectory = [np.hstack([self.x0, self.itvl.start, self.xt, self.itvl.end])]
        return ecd_reserve(stgcs, trajectory, robot_radius + self.radius, reserve_first_to_t0, reserve_last_to_tf)
 
    def draw(self, ax:Axes3D) -> None:
        p = np.hstack([self.x0, self.itvl.start])
        q = np.hstack([self.xt, self.itvl.end])
        draw_cylinder(ax, p, q, self.radius)


class ConcatDynamicSphere(DynamicObstacle):
    
    def __init__(self, X0:List[np.ndarray], Xt:List[np.ndarray], itvls:List[Interval], radius:float) -> None:
        self.x0, self.xt, self.itvl, self.radius = X0[0], Xt[-1], Interval(itvls[0].start, itvls[-1].end), radius
        self.segments: List[DynamicSphere] = []
        for x0, xt, itvl in zip(X0, Xt, itvls):
            self.segments.append(DynamicSphere(x0, xt, radius, itvl))

    @staticmethod
    def from_solution(sol:ShortestPathSolution, radius:float) -> ConcatDynamicSphere:
        X0, Xt, itvls = [], [], []

        for i in range(len(sol.trajectory)):
            X0.append(sol.trajectory[i][: sol.dim-1])
            Xt.append(sol.trajectory[i][sol.dim:-1])
            itvls.append(Interval(sol.trajectory[i][sol.dim-1], sol.trajectory[i][-1]))

        return ConcatDynamicSphere(X0, Xt, itvls, radius)

    def collision_intervals(self, point:np.ndarray, robot_radius:float) -> List[Interval]:
        ret = []
        for seg in self.segments:
            ret.extend(seg.collision_intervals(point, robot_radius))
        return ret
    
    def is_colliding_lineseg(self, p, q, tp, tq, robot_radius) -> bool:
        itvl = Interval(tp, tq)
        start_col = DynamicSphere(self.x0, self.x0, self.radius, Interval(0, self.itvl.start))
        end_col = DynamicSphere(self.xt, self.xt, self.radius, Interval(self.itvl.end, 1e9))
        if start_col.is_colliding_lineseg(p, q, tp, tq, robot_radius) or \
           end_col.is_colliding_lineseg(p, q, tp, tq, robot_radius):
            return True

        for seg in self.segments:
            if seg.itvl.intersects(itvl) and seg.is_colliding_lineseg(p, q, tp, tq, robot_radius):
                return True
        return False
    
    def x(self, t:float) -> np.ndarray:
        for seg in self.segments:
            if t <= seg.itvl.end:
                return seg.x(t)
        return self.segments[-1].x(t)
    
    def reserve(self, stgcs, robot_radius, reserve_first_to_t0=True, reserve_last_to_tf=True):
        trajectory = []
        for seg in self.segments:
            trajectory.append(np.hstack([seg.x0, seg.itvl.start, seg.xt, seg.itvl.end]))
        return ecd_reserve(stgcs, trajectory, robot_radius + self.radius, reserve_first_to_t0, reserve_last_to_tf)

    def draw(self, ax:Axes3D) -> None:
        for seg in self.segments:
            seg.draw(ax)


def lerp(p:np.ndarray, q:np.ndarray, tp:float, tq:float, t:float) -> np.ndarray:
    if t <= tp:
        return p
    if t >= tq:
        return q
    return p + (q - p) * ((t - tp) / (tq - tp))

