from __future__ import annotations
from abc import abstractmethod, ABC
from typing import List
import numpy as np

from pydrake.all import HPolyhedron, VPolytope

from experiments.environment.space_time_graph import STGCS
from experiments.environment.reservation import reserve as ecd_reserve
from stgcs_native.types import STTrajectory
from experiments.environment.interval import Interval
from experiments.environment.collision import is_lineseg_colliding, is_point_colliding


""" Static Obstacles """

class StaticObstacle(ABC):
    
    @abstractmethod
    def is_colliding(self, point:np.ndarray, robot_radius:float) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_colliding_lineseg(self, p:np.ndarray, q:np.ndarray, robot_radius:float) -> bool:
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


class StaticPolygon(StaticObstacle):

    def __init__(self, vertices:np.ndarray) -> None:
        self.vertices = vertices
        self.hpoly = HPolyhedron(VPolytope(vertices.T))
    
    def is_colliding(self, point:np.ndarray, robot_radius:float) -> bool:
        return is_point_colliding(self.hpoly, point, robot_radius)

    def is_colliding_lineseg(self, p:np.ndarray, q:np.ndarray, robot_radius:float) -> bool:
        return is_lineseg_colliding(self.hpoly, p, q, robot_radius)


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
    

class DynamicSphere(DynamicObstacle):

    def __init__(self, x0:np.ndarray, xt:np.ndarray, radius:float, itvl:Interval) -> None:
        self.x0, self.xt, self.radius, self.itvl = x0, xt, radius, itvl
        if self.itvl.duration <= 1e-12:
            self.velocity = np.zeros_like(xt - x0, dtype=float)
        else:
            self.velocity = (xt - x0) / self.itvl.duration
    
    def collision_intervals(self, point:np.ndarray, robot_radius:float) -> List[Interval]:
        ret = []
        rr = self.radius + robot_radius
        vel_magnitude = np.linalg.norm(self.velocity)
        x0_collision = np.linalg.norm(self.x0 - point) <= rr
        xt_collision = np.linalg.norm(self.xt - point) <= rr
        if x0_collision and self.itvl.start != 0:
            ret.append(Interval(0.0, self.itvl.start))
        if xt_collision and self.itvl.end != np.inf:
            ret.append(Interval(self.itvl.end, np.inf))

        if vel_magnitude <= 1e-12:
            if x0_collision:
                ret.append(Interval(self.itvl.start, self.itvl.end))
            return ret

        # Solve over the active motion phase using tau = t - self.itvl.start.
        # The obstacle position during the interval is x0 + v * tau.
        a = vel_magnitude ** 2  # a > 0
        b = 2 * np.dot(self.velocity, self.x0 - point)
        c = np.linalg.norm(self.x0 - point) ** 2 - rr ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant >= 0:
            tau1 = (-b - np.sqrt(discriminant)) / (2 * a)
            tau2 = (-b + np.sqrt(discriminant)) / (2 * a)
            intersection = Interval(tau1, tau2).intersection(Interval(0.0, self.itvl.duration))
            if intersection is not None:
                ret.append(Interval(
                    intersection.start + self.itvl.start,
                    intersection.end + self.itvl.start,
                ))
        
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
        denom = np.dot(B, B)
        if denom <= 1e-12:
            min_dist = np.linalg.norm(A)
        else:
            t = np.clip(- np.dot(A, B) / denom, 0, 1)
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
 

class ConcatDynamicSphere(DynamicObstacle):
    reserve_start = True
    reserve_goal = True
    
    def __init__(self, X0:List[np.ndarray], Xt:List[np.ndarray], itvls:List[Interval], radius:float) -> None:
        self.x0, self.xt, self.itvl, self.radius = X0[0], Xt[-1], Interval(itvls[0].start, itvls[-1].end), radius
        self.segments: List[DynamicSphere] = []
        for x0, xt, itvl in zip(X0, Xt, itvls):
            self.segments.append(DynamicSphere(x0, xt, radius, itvl))

    @classmethod
    def from_solution(cls, sol:STTrajectory, radius:float, *,
                      reserve_start:bool=True, reserve_goal:bool=True) -> ConcatDynamicSphere:
        X0, Xt, itvls = [], [], []
        for i in range(len(sol.points)):
            X0.append(sol.xA(i)[:-1])
            Xt.append(sol.xB(i)[:-1])
            itvls.append(Interval(sol.xA(i)[-1], sol.xB(i)[-1]))

        obstacle = cls(X0, Xt, itvls, radius)
        obstacle.reserve_start = bool(reserve_start)
        obstacle.reserve_goal = bool(reserve_goal)
        return obstacle

    def collision_intervals(self, point:np.ndarray, robot_radius:float) -> List[Interval]:
        ret = []
        for seg in self.segments:
            # Each leg is active only on its own interval; intermediate
            # waypoints are not occupied forever before/after that leg.
            for interval in seg.collision_intervals(point, robot_radius):
                clipped = interval.intersection(seg.itvl)
                if clipped is not None:
                    ret.append(clipped)
        radius = self.radius + robot_radius
        if self.reserve_start and self.itvl.start > 0 and np.linalg.norm(point - self.x0) <= radius:
            ret.append(Interval(0.0, self.itvl.start))
        if self.reserve_goal and np.linalg.norm(point - self.xt) <= radius:
            ret.append(Interval(self.itvl.end, np.inf))
        return ret
    
    def is_colliding_lineseg(self, p, q, tp, tq, robot_radius) -> bool:
        itvl = Interval(tp, tq)
        start_col = DynamicSphere(self.x0, self.x0, self.radius, Interval(0, self.itvl.start))
        end_col = DynamicSphere(self.xt, self.xt, self.radius, Interval(self.itvl.end, 1e9))
        if (self.reserve_start and start_col.is_colliding_lineseg(p, q, tp, tq, robot_radius)) or \
           (self.reserve_goal and end_col.is_colliding_lineseg(p, q, tp, tq, robot_radius)):
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


def lerp(p:np.ndarray, q:np.ndarray, tp:float, tq:float, t:float) -> np.ndarray:
    if t <= tp:
        return p
    if t >= tq:
        return q
    return p + (q - p) * ((t - tp) / (tq - tp))
