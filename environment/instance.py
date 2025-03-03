from __future__ import annotations
from typing import List
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy.spatial import ConvexHull
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

from environment.obstacle import StaticObstacle, DynamicObstacle, DynamicSphere, ConcatDynamicSphere
from mrmp.graph import ShortestPathSolution
from mrmp.interval import Interval
from mrmp.utils import make_hpolytope

from pydrake.all import (
    HPolyhedron, RandomGenerator,
)



class Instance:

    def __init__(self, name, CSpace: List[np.ndarray], OStatic: List[StaticObstacle] = [], ODynamic: List[DynamicObstacle] = []) -> None:
        self.name = name
        self.C_Space = CSpace       # list of polyhedrons defined by vertices w/o colliding O_Static
        self.O_Static = OStatic
        self.O_Dynamic = ODynamic
        self.lb: np.ndarray = np.min(np.vstack(CSpace), axis=0)
        self.ub: np.ndarray = np.max(np.vstack(CSpace), axis=0)
        self.dim: int = len(self.lb)
        self._CSpace_hpoly: List[HPolyhedron] = [make_hpolytope(C) for C in CSpace]
    
    def copy(self) -> Instance:
        return Instance(self.name, deepcopy(self.C_Space), deepcopy(self.O_Static), deepcopy(self.O_Dynamic))
    
    def animate_2d(self, ax:Axes, sols:List[ShortestPathSolution]=[], dt:float=0.02, save_anim:bool=False) -> None:
        if sols != []:
            Pi = []
            tmax = max([s.itvl.end for s in sols])
            T = np.arange(0, tmax + dt, dt)
            for sol in sols:
                traj = np.array([sol.lerp(t) for t in T])
                Pi.append(traj)
        else:
            Pi = []

        ax.set_aspect('equal')

        self.draw_static(ax)

        anim = _animate_func_2d(ax, self.lb, self.ub, Pi, self.O_Dynamic, dt=dt)
        if save_anim:
            anim.save(f"{self.name}.mp4", writer='ffmpeg', fps=1/dt, dpi=1000)
        plt.show()
    
    def draw_static(self, ax:Axes, alpha=0.8, draw_CSpace:bool=False) -> None:
        for obs in self.O_Static:
            obs.draw(ax, alpha=alpha)
        
        bounding_box_verts = np.array([
            [self.lb[0], self.lb[1]],
            [self.ub[0], self.lb[1]],
            [self.ub[0], self.ub[1]],
            [self.lb[0], self.ub[1]],
        ])
        for u, v in zip(bounding_box_verts, np.roll(bounding_box_verts, 1, axis=0)):
            ax.plot([u[0], v[0]], [u[1], v[1]], '-k')
        
        if draw_CSpace:
            colors = matplotlib.cm.get_cmap("Pastel2")
            for i, C in enumerate(self.C_Space):
                ax.fill(C[:, 0], C[:, 1], alpha=alpha, fc=colors(i/len(self.C_Space)), ec='black')

    def collision_checking_seg(self, p:np.ndarray, q:np.ndarray, tp:float, tq:float, robot_radius:float) -> bool:
        # collision checking w/ static obstacles
        for o in self.O_Static:
            if o.is_colliding_lineseg(p, q, robot_radius):
                return True
        
        if tp > tq:
            p, q, tp, tq = q, p, tq, tp

        # collision checking w/ dynamic obstacles
        for o in self.O_Dynamic:
            if isinstance(o, DynamicSphere):
                start_occ = DynamicSphere(o.x0, o.x0, o.radius, Interval(0, o.itvl.start))
                end_occ = DynamicSphere(o.xt, o.xt, o.radius, Interval(o.itvl.end, 1e9))
                if o.is_colliding_lineseg(p, q, tp, tq, robot_radius) or \
                start_occ.is_colliding_lineseg(p, q, tp, tq, robot_radius) or \
                end_occ.is_colliding_lineseg(p, q, tp, tq, robot_radius):
                    return True
            elif isinstance(o, ConcatDynamicSphere):
                if o.is_colliding_lineseg(p, q, tp, tq, robot_radius):
                    return True

        return False

    def sample_CSpace(self, np_rng:np.random.RandomState, drake_rng:RandomGenerator) -> np.ndarray:
        idx = np_rng.randint(0, len(self.C_Space))
        return self._CSpace_hpoly[idx].UniformSample(drake_rng)
    
    def sample_bounding_box(self, np_rng:np.random.RandomState) -> np.ndarray:
        return np_rng.uniform(self.lb, self.ub)


def _animate_func_2d(
    ax: Axes,
    lb: np.ndarray,
    ub: np.ndarray,
    trajectories: List[np.ndarray],
    ODynamic: List[DynamicObstacle] = [],
    dt:float = 0.02,
    labels: List[str] = None,
    colors: List[str] = None,
    interval: int = 30,
    robot_tail_length: int = 30,
    obs_tail_length: int = 10,
) -> FuncAnimation:
    
    k = len(trajectories)
    
    if colors is None:
        colors = plt.cm.rainbow(np.linspace(0, 1, k))
    
    if labels is None:
        labels = [f"Trajectory {i+1}" for i in range(k)]
    
    x_min, y_min = lb
    x_max, y_max = ub

    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    

    # draw dynamic obstacles
    obs_pos = []
    obs_markers:List[Circle] = []
    obs_footprints: List[Circle] = [None] * (len(ODynamic) * obs_tail_length)
    for i, obs in enumerate(ODynamic):
        c = ax.add_patch(Circle(xy=obs.x0, radius=obs.radius, color='k', fill=True))
        obs_markers.append(c)
        obs_pos.append(obs.x0)
        for j in range(obs_tail_length):
            idx = i * obs_tail_length + j
            c = ax.add_patch(Circle(xy=obs.x0, radius=obs.radius, color='k', fill=False, alpha = 1 - (j / obs_tail_length)))
            obs_footprints[idx] = c
    
    # draw robot trajectories
    footprint_itvl = int(interval / robot_tail_length)
    footprints, points, texts = [None] * (robot_tail_length * k), [None] * k, [None] * k
    robot_size = 6
    for i in range(k):
        for j in range(robot_tail_length):
            idx = i * robot_tail_length + j
            footprints[idx], = ax.plot([], [], '.', color=colors[i], markersize=robot_size, mfc='none', alpha = 1 - (j / robot_tail_length))
        
        points[i], = ax.plot([], [], '.', color=colors[i], markersize=robot_size, mfc=colors[i])
        texts[i] = ax.text(0, 0, '', color='k', fontsize=12)    

    # animation function
    def animate(frame):
        for i, obs_marker in enumerate(obs_markers):
            obs_pos[i] = ODynamic[i].x(frame * dt)
            obs_marker.set_center(obs_pos[i])
            for j in range(obs_tail_length-1, -1, -1):
                idx = i * obs_tail_length + j
                prev_frame = max(0, frame - footprint_itvl * j)
                obs_footprints[idx].set_center(ODynamic[i].x(prev_frame * dt))

        for i, (trajectory, point, text) in enumerate(zip(trajectories, points, texts)):
            if frame < len(trajectory):
                point.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
                text.set_position((trajectory[frame, 0], trajectory[frame, 1]))
                text.set_text(i)
                
                for j in range(robot_tail_length-1, -1, -1):
                    idx = i * robot_tail_length + j
                    prev_frame = max(0, frame - footprint_itvl * j)
                    x, y = trajectory[prev_frame, 0], trajectory[prev_frame, 1]
                    footprints[idx].set_data([x], [y])
            
        return footprints + points + texts + obs_markers + obs_footprints
    
    fig = plt.gcf()
    if trajectories == []:
        n_frames = 1000
    else:
        n_frames = max(len(traj) for traj in trajectories)
    anim = FuncAnimation(
        fig, animate, frames=n_frames,
        interval=interval, blit=True
    )
    
    plt.grid(False)
    plt.tight_layout()
    
    return anim

