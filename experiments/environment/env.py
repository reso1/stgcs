from __future__ import annotations
from typing import List, Tuple, Optional
from itertools import combinations
import os

import logging
from tqdm import tqdm
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from copy import deepcopy
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull, QhullError

from pydrake.all import HPolyhedron, RandomGenerator

from experiments.environment.obstacle import (
    StaticObstacle, StaticPolygon, StaticSphere,
    DynamicObstacle, DynamicSphere, ConcatDynamicSphere)

from experiments.environment.space_time_graph import STGCS
from experiments.environment.interval import Interval
from experiments.environment.geometry import HPolyhedronSampler, make_hpolytope, time_extruded

logger = logging.getLogger(__name__)


def _progress_disabled() -> bool:
    return os.environ.get("STGCS_DISABLE_PROGRESS", "") == "1"


class Env:
    CSPACE_SEGMENT_TOL = 1e-9

    def __init__(
        self, name, CSpace:List[np.ndarray], robot_radius:float, 
        OStatic:List[StaticObstacle]=[], ODynamic:List[DynamicObstacle]=[],
        edges:List[Tuple[int, int]]=[],
        domain_lb:np.ndarray | None=None,
        domain_ub:np.ndarray | None=None,
    ) -> None:
        self.name = name
        self.C_Space = CSpace       # list of polyhedrons defined by vertices w/o colliding O_Static
        self.robot_radius = robot_radius
        self.O_Static = OStatic
        self.O_Dynamic = ODynamic
        self.edges = edges
        cspace_lb = np.min(np.vstack(CSpace), axis=0)
        cspace_ub = np.max(np.vstack(CSpace), axis=0)
        self.lb: np.ndarray = np.asarray(cspace_lb if domain_lb is None else domain_lb, dtype=float)
        self.ub: np.ndarray = np.asarray(cspace_ub if domain_ub is None else domain_ub, dtype=float)
        if self.lb.shape != cspace_lb.shape or self.ub.shape != cspace_ub.shape:
            raise ValueError(
                f"Domain bounds must match CSpace dimension {cspace_lb.shape[0]}, "
                f"got lb shape {self.lb.shape} and ub shape {self.ub.shape}."
            )
        self.dim: int = len(self.lb)
        self._CSpace_hpoly: List[HPolyhedron] = [make_hpolytope(C) for C in CSpace]

    @staticmethod
    def _segment_parameter_interval_in_hpoly(
        hpoly: HPolyhedron,
        p: np.ndarray,
        q: np.ndarray,
        tol: float,
    ) -> Optional[Tuple[float, float]]:
        direction = q - p
        lo, hi = 0.0, 1.0
        A = np.asarray(hpoly.A(), dtype=float)
        b = np.asarray(hpoly.b(), dtype=float).reshape(-1)
        for normal, bound in zip(A, b):
            slope = float(normal @ direction)
            rhs = float(bound) + tol - float(normal @ p)
            if abs(slope) <= tol:
                if rhs < 0.0:
                    return None
                continue
            alpha = rhs / slope
            if slope > 0.0:
                hi = min(hi, alpha)
            else:
                lo = max(lo, alpha)
            if lo > hi + tol:
                return None

        if hi < -tol or lo > 1.0 + tol:
            return None
        lo = max(0.0, lo)
        hi = min(1.0, hi)
        if lo > hi + tol:
            return None
        return lo, hi

    @staticmethod
    def _intervals_cover_unit(intervals: List[Tuple[float, float]], tol: float) -> bool:
        covered_until = 0.0
        for lo, hi in sorted(intervals):
            if hi < covered_until - tol:
                continue
            if lo > covered_until + tol:
                return False
            covered_until = max(covered_until, hi)
            if covered_until >= 1.0 - tol:
                return True
        return covered_until >= 1.0 - tol

    def is_segment_in_CSpace(self, p: np.ndarray, q: np.ndarray, tol: float = CSPACE_SEGMENT_TOL) -> bool:
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        intervals = []
        for hpoly in self._CSpace_hpoly:
            interval = self._segment_parameter_interval_in_hpoly(hpoly, p, q, tol)
            if interval is not None:
                intervals.append(interval)
        return self._intervals_cover_unit(intervals, tol)

    def copy(self) -> Env:
        return Env(
            self.name,
            deepcopy(self.C_Space),
            self.robot_radius,
            deepcopy(self.O_Static),
            deepcopy(self.O_Dynamic),
            deepcopy(self.edges),
            domain_lb=deepcopy(self.lb),
            domain_ub=deepcopy(self.ub),
        )
    
    def animate_1d(self, ax:Axes, sols:list=[], dt:float=0.02, draw_CSpace=False, save_anim:bool=False) -> None:
        tmax = 5
        if sols != []:
            Pi = []
            tmax = max([s.itvl.end for s in sols])
            T = np.arange(0, tmax + dt, dt)
            for sol in sols:
                traj = np.array([np.hstack([sol.lerp(t)[0], 0]) for t in T])
                Pi.append(traj)
        else:
            Pi = []

        ax.set_aspect('equal')

        anim = _animate_func_2d(ax, self.robot_radius, np.hstack([self.lb, -0.5]), np.hstack([self.ub, 0.5]), Pi, self.O_Dynamic, dt=dt)
        
        if save_anim:
            anim.save(f"{self.name}.mp4", writer='ffmpeg', fps=1/dt, dpi=1000)
        plt.show()

    def animate_2d(self, ax:Axes, sols:list=[], dt:float=0.02, draw_CSpace=False, CSpace_text=True, save_anim:bool=False) -> None:
        if sols != [] and sols is not None:
            Pi = []
            tmax = max([traj.xT[-1] for traj in sols])
            T = np.arange(0, tmax + dt, dt)
            for sol in sols:
                traj = np.array([sol.lerp(t) for t in T])
                Pi.append(traj)
        else:
            Pi = []

        ax.set_aspect('equal')

        self.draw_static(ax, draw_CSpace=draw_CSpace, CSpace_text=CSpace_text)

        anim = _animate_func_2d(ax, self.robot_radius, self.lb, self.ub, Pi, self.O_Dynamic, dt=dt)
        if save_anim:
            anim.save(f"{self.name}.mp4", writer='ffmpeg', fps=1/dt, dpi=1000,
                      progress_callback=lambda i, n: print(f'{self.name}: saving frame {i}/{n}'))
        plt.show()

    def animate_3d(
        self,
        sols:list=[],
        dt:float=0.05,
        draw_CSpace:bool=False,
        save_anim:bool=False,
        show:bool=True,
    ) -> go.Figure:
        if self.dim != 3:
            raise ValueError(f"animate_3d only supports 3D environments, got dim={self.dim}")

        if sols is not None and sols != []:
            trajectories = []
            tmax = max([traj.xT[-1] for traj in sols])
            T = np.arange(0, tmax + dt, dt)
            for sol in sols:
                trajectories.append(np.array([sol.lerp(t)[:self.dim] for t in T]))
        else:
            trajectories = []
            T = np.array([0.0])

        if self.O_Dynamic:
            dynamic_tmax = max(
                [obs.itvl.end for obs in self.O_Dynamic if np.isfinite(obs.itvl.end)],
                default=0.0,
            )
            if dynamic_tmax > T[-1]:
                T = np.arange(0.0, dynamic_tmax + dt, dt)
                if trajectories:
                    trajectories = [np.array([sol.lerp(t)[:self.dim] for t in T]) for sol in sols]

        fig = _animate_func_3d_plotly(
            env=self,
            trajectories=trajectories,
            times=T,
            draw_CSpace=draw_CSpace,
        )

        if save_anim:
            fig.write_html(f"{self.name}_3d.html", include_plotlyjs=True, auto_play=False)
        if show:
            fig.show()
        return fig
    
    def draw_static(self, ax:Axes, alpha=0.8, draw_CSpace:bool=False, CSpace_text:bool=True) -> None:
        for obs in self.O_Static:
            if isinstance(obs, StaticPolygon):
                ax.fill(obs.vertices[:, 0], obs.vertices[:, 1], alpha=alpha, fc='k', ec='black')
            elif isinstance(obs, StaticSphere):
                ax.add_artist(Circle(obs.pos, obs.radius, color='k'))
            else:
                raise TypeError(f"Unsupported static obstacle type {type(obs)!r}")
        
        bounding_box_verts = np.array([
            [self.lb[0] - self.robot_radius, self.lb[1] - self.robot_radius],
            [self.ub[0] + self.robot_radius, self.lb[1] - self.robot_radius],
            [self.ub[0] + self.robot_radius, self.ub[1] + self.robot_radius],
            [self.lb[0] - self.robot_radius, self.ub[1] + self.robot_radius],
        ])
        for u, v in zip(bounding_box_verts, np.roll(bounding_box_verts, 1, axis=0)):
            ax.plot([u[0], v[0]], [u[1], v[1]], '-k')
        
        if draw_CSpace:
            colors = matplotlib.cm.get_cmap("Pastel2")
            for i, C in enumerate(self.C_Space):
                center = np.mean(C, axis=0)
                ax.fill(C[:, 0], C[:, 1], alpha=alpha, fc=colors(i/len(self.C_Space)), ec='black')
                if CSpace_text:
                    ax.text(center[0], center[1], f"v{i}")

    def collision_checking_seg(self, p:np.ndarray, q:np.ndarray, tp:float, tq:float) -> bool:
        # collision checking w/ static obstacles
        for o in self.O_Static:
            if o.is_colliding_lineseg(p, q, self.robot_radius):
                return True

        # Grid-style environments encode free space by C_Space cells instead
        # of explicit static obstacles.
        if not self.O_Static and not self.is_segment_in_CSpace(p, q):
            return True
        
        if tp > tq:
            p, q, tp, tq = q, p, tq, tp

        # collision checking w/ dynamic obstacles
        for o in self.O_Dynamic:
            if isinstance(o, DynamicSphere):
                start_occ = DynamicSphere(o.x0, o.x0, o.radius, Interval(0, o.itvl.start))
                end_occ = DynamicSphere(o.xt, o.xt, o.radius, Interval(o.itvl.end, 1e9))
                if o.is_colliding_lineseg(p, q, tp, tq, self.robot_radius) or \
                start_occ.is_colliding_lineseg(p, q, tp, tq, self.robot_radius) or \
                end_occ.is_colliding_lineseg(p, q, tp, tq, self.robot_radius):
                    return True
            elif isinstance(o, ConcatDynamicSphere):
                if o.is_colliding_lineseg(p, q, tp, tq, self.robot_radius):
                    return True

        return False

    def sample_CSpace(self, np_rng:np.random.RandomState=None, drake_rng:RandomGenerator=None, 
                      size:int=1, seed:int=0) -> np.ndarray:
        if np_rng is None:
            assert seed is not None
            np_rng = np.random.RandomState(seed=seed)
        if drake_rng is None:
            assert seed is not None
            drake_rng = RandomGenerator(seed)
            
        if size == 1:
            idx = np_rng.randint(0, len(self.C_Space))
            return HPolyhedronSampler.uniform_sample(
                self._CSpace_hpoly[idx],
                drake_rng,
                context=f"Env.sample_CSpace[{self.name}]",
            )
        else:
            samples = []
            for _ in range(size):
                idx = np_rng.randint(0, len(self.C_Space))
                sample = HPolyhedronSampler.uniform_sample(
                    self._CSpace_hpoly[idx],
                    drake_rng,
                    context=f"Env.sample_CSpace[{self.name}]",
                )
                samples.append(sample)
            return np.array(samples)
    
    def sample_bounding_box(self, np_rng:np.random.RandomState) -> np.ndarray:
        return np_rng.uniform(self.lb, self.ub)

    def sample_static_obstacle_free_space(self, np_rng:np.random.RandomState) -> np.ndarray:
        sample = self.sample_bounding_box(np_rng)
        while any(obstacle.is_colliding(sample, self.robot_radius) for obstacle in self.O_Static):
            sample = self.sample_bounding_box(np_rng)
        return sample

    def build_STGCS(self, t0:float=0, tmax:float=1e2, vlimit:float=1.0, dt:float=1e-6) -> STGCS:
        stgcs = STGCS(self.C_Space.copy(), t0, tmax, vlimit, dt)
        disable_progress = _progress_disabled()

        num_sets = len(self._CSpace_hpoly)
        for idx in tqdm(range(num_sets), desc="Building ST-GCS vertices", disable=disable_progress):
            t_set = time_extruded(self._CSpace_hpoly[idx], t0, tmax)
            lb = np.min(self.C_Space[idx], axis=0)
            ub = np.max(self.C_Space[idx], axis=0)
            space_bounds = [Interval(l, u) for l, u in zip(lb, ub)]
            stgcs.add_vertex(t_set, Interval(t0, tmax), space_bounds)

        if self.edges != []:
            for i, j in self.edges:
                stgcs.add_edges_bidir(f"v{i}", f"v{j}")
        else:
            num_total_checks = num_sets * (num_sets - 1) // 2
            node_names = list(stgcs.G.nodes)
            for u_name, v_name in tqdm(
                combinations(node_names, 2),
                desc="Building ST-GCS edges",
                total=num_total_checks,
                disable=disable_progress,
            ):
                stgcs.add_edges_bidir(u_name, v_name)
        
        for obs in self.O_Dynamic:
            stgcs = obs.reserve(stgcs, self.robot_radius, True, True)

        logger.debug(f"STGCS built with {stgcs.G.number_of_nodes()} vertices and {stgcs.G.number_of_edges()} edges")
        return stgcs


def _animate_func_2d(
    ax: Axes,
    robot_radius: float,
    lb: np.ndarray,
    ub: np.ndarray,
    trajectories: List[np.ndarray],
    ODynamic: List[DynamicObstacle] = [],
    dt:float = 0.02,
    labels: List[str] = None,
    colors: List[str] = None,
    interval: int = 30,
    robot_tail_length: int = 30,
    obs_tail_length: int = 15,
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
    footprints, robots, texts = [None] * (robot_tail_length * k), [None] * k, [None] * k

    for i in range(k):
        for j in range(robot_tail_length):
            idx = i * robot_tail_length + j
            footprints[idx] = ax.add_patch(Circle(xy=trajectories[i][0], radius=robot_radius, color=colors[i], fill=False, alpha = 1 - (j / robot_tail_length)))
            # footprints[idx], = ax.plot([], [], '.', color=colors[i], markersize=robot_size, mfc='none', alpha = 1 - (j / robot_tail_length))
        
        robots[i] = ax.add_patch(Circle(xy=trajectories[i][0], radius=robot_radius, color=colors[i], fill=True))
        # robots[i], = ax.plot([], [], '.', color=colors[i], markersize=robot_size, mfc=colors[i])
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

        for i, (trajectory, point, text) in enumerate(zip(trajectories, robots, texts)):
            if frame < len(trajectory):
                point.set_center(trajectory[frame])
                # point.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
                text.set_position((trajectory[frame, 0], trajectory[frame, 1]))
                text.set_text(i)
                
                for j in range(robot_tail_length-1, -1, -1):
                    idx = i * robot_tail_length + j
                    prev_frame = max(0, frame - footprint_itvl * j)
                    # x, y = trajectory[prev_frame, 0], trajectory[prev_frame, 1]
                    # footprints[idx].set_data([x], [y])
                    footprints[idx].set_center(trajectory[prev_frame])
            
        return footprints + robots + texts + obs_markers + obs_footprints
    
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


def _animate_func_3d_plotly(
    env: Env,
    trajectories: List[np.ndarray],
    times: np.ndarray,
    draw_CSpace: bool = False,
) -> go.Figure:
    colors = [
        "#0B6E4F", "#C84C09", "#1D4E89", "#A23B72",
        "#4D6CFA", "#6B8E23", "#B22222", "#008B8B",
    ]

    static_traces: List[go.BaseTraceType] = []
    static_traces.extend(_plotly_box_wireframe(env.lb, env.ub))

    if draw_CSpace:
        for idx, cspace in enumerate(env.C_Space):
            static_traces.extend(_plotly_convex_set_traces(
                cspace,
                color=colors[idx % len(colors)],
                opacity=0.12,
                name=f"CSpace {idx}",
                showlegend=(idx == 0),
            ))

    for idx, obs in enumerate(env.O_Static):
        if isinstance(obs, StaticPolygon):
            static_traces.extend(_plotly_convex_set_traces(
                obs.vertices,
                color="#4A4A4A",
                opacity=0.30,
                name="Static obstacle",
                showlegend=(idx == 0),
            ))
        elif isinstance(obs, StaticSphere):
            static_traces.append(_plotly_sphere_mesh_trace(
                center=obs.pos,
                radius=obs.radius,
                color="#4A4A4A",
                opacity=0.8,
                name="Static obstacle",
                showlegend=(idx == 0),
            ))

    max_frames = max(len(times), max((len(traj) for traj in trajectories), default=0), 1)

    frame_traces = []
    for frame_idx in range(max_frames):
        t = times[min(frame_idx, len(times) - 1)]
        traces: List[go.BaseTraceType] = []

        for obs_idx, obs in enumerate(env.O_Dynamic):
            pos = obs.x(t)
            traces.append(_plotly_sphere_mesh_trace(
                center=pos,
                radius=obs.radius,
                color="#111111",
                opacity=0.85,
                name=f"Dynamic obstacle {obs_idx}",
                showlegend=False,
            ))

        for traj_idx, traj in enumerate(trajectories):
            curr = traj[min(frame_idx, len(traj) - 1)]
            traces.append(go.Scatter3d(
                x=traj[:min(frame_idx + 1, len(traj)), 0],
                y=traj[:min(frame_idx + 1, len(traj)), 1],
                z=traj[:min(frame_idx + 1, len(traj)), 2],
                mode="lines",
                line=dict(color=colors[traj_idx % len(colors)], width=6),
                name=f"Trajectory {traj_idx}",
                showlegend=False,
            ))
            traces.append(_plotly_sphere_mesh_trace(
                center=curr,
                radius=env.robot_radius,
                color=colors[traj_idx % len(colors)],
                opacity=0.95,
                name=f"Robot {traj_idx}",
                showlegend=False,
            ))

        frame_traces.append(go.Frame(
            data=traces,
            name=f"{frame_idx}",
            traces=list(range(len(static_traces), len(static_traces) + len(traces))),
        ))

    initial_dynamic = frame_traces[0].data if frame_traces else []
    fig = go.Figure(data=static_traces + list(initial_dynamic), frames=frame_traces)

    fig.update_layout(
        title=f"{env.name} 3D animation",
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="x", range=[env.lb[0], env.ub[0]]),
            yaxis=dict(title="y", range=[env.lb[1], env.ub[1]]),
            zaxis=dict(title="z", range=[env.lb[2], env.ub[2]]),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True}],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                },
            ],
            "direction": "left",
            "x": 0.0,
            "y": 1.05,
        }],
        sliders=[{
            "currentvalue": {"prefix": "Frame: "},
            "steps": [
                {
                    "label": str(idx),
                    "method": "animate",
                    "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                }
                for idx, frame in enumerate(frame_traces)
            ],
        }],
    )

    return fig


def _plotly_box_wireframe(lb: np.ndarray, ub: np.ndarray) -> List[go.Scatter3d]:
    corners = np.array([
        [lb[0], lb[1], lb[2]],
        [ub[0], lb[1], lb[2]],
        [ub[0], ub[1], lb[2]],
        [lb[0], ub[1], lb[2]],
        [lb[0], lb[1], ub[2]],
        [ub[0], lb[1], ub[2]],
        [ub[0], ub[1], ub[2]],
        [lb[0], ub[1], ub[2]],
    ])
    edge_indices = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    traces = []
    for idx, (u, v) in enumerate(edge_indices):
        traces.append(go.Scatter3d(
            x=[corners[u, 0], corners[v, 0]],
            y=[corners[u, 1], corners[v, 1]],
            z=[corners[u, 2], corners[v, 2]],
            mode="lines",
            line=dict(color="#888888", width=3),
            name="Bounds",
            showlegend=(idx == 0),
        ))
    return traces


def _plotly_sphere_mesh_trace(
    center: np.ndarray,
    radius: float,
    color: str,
    opacity: float,
    name: str,
    showlegend: bool,
    n_theta: int = 16,
    n_phi: int = 10,
) -> go.Mesh3d:
    vertices, faces = _sphere_mesh(np.asarray(center, dtype=float), radius, n_theta=n_theta, n_phi=n_phi)
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
        flatshading=True,
        name=name,
        showlegend=showlegend,
    )


def _sphere_mesh(
    center: np.ndarray,
    radius: float,
    n_theta: int = 16,
    n_phi: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    if center.shape != (3,):
        raise ValueError(f"Expected 3D center, got shape {center.shape}")
    if radius < 0:
        raise ValueError("Sphere radius must be nonnegative")
    if n_theta < 3 or n_phi < 3:
        raise ValueError("Sphere mesh resolution must be at least 3 in both directions")

    vertices = [np.array([0.0, 0.0, 1.0])]
    theta_vals = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    for phi_idx in range(1, n_phi - 1):
        phi = np.pi * phi_idx / (n_phi - 1)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        for theta in theta_vals:
            vertices.append(np.array([
                sin_phi * np.cos(theta),
                sin_phi * np.sin(theta),
                cos_phi,
            ]))

    south_idx = len(vertices)
    vertices.append(np.array([0.0, 0.0, -1.0]))
    vertices = np.array(vertices, dtype=float)

    faces = []
    first_ring_start = 1
    for theta_idx in range(n_theta):
        nxt = (theta_idx + 1) % n_theta
        faces.append([0, first_ring_start + theta_idx, first_ring_start + nxt])

    num_inner_rings = n_phi - 2
    for ring_idx in range(num_inner_rings - 1):
        ring_start = 1 + ring_idx * n_theta
        next_ring_start = ring_start + n_theta
        for theta_idx in range(n_theta):
            nxt = (theta_idx + 1) % n_theta
            v00 = ring_start + theta_idx
            v01 = ring_start + nxt
            v10 = next_ring_start + theta_idx
            v11 = next_ring_start + nxt
            faces.append([v00, v10, v01])
            faces.append([v01, v10, v11])

    last_ring_start = 1 + (num_inner_rings - 1) * n_theta
    for theta_idx in range(n_theta):
        nxt = (theta_idx + 1) % n_theta
        faces.append([last_ring_start + theta_idx, south_idx, last_ring_start + nxt])

    return center + radius * vertices, np.array(faces, dtype=int)


def _plotly_convex_set_traces(
    vertices: np.ndarray,
    color: str,
    opacity: float,
    name: str,
    showlegend: bool,
) -> List[go.BaseTraceType]:
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected 3D vertices, got shape {vertices.shape}")

    try:
        hull = ConvexHull(vertices)
        simplices = hull.simplices
        return [go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=simplices[:, 0],
            j=simplices[:, 1],
            k=simplices[:, 2],
            color=color,
            opacity=opacity,
            name=name,
            showlegend=showlegend,
        )]
    except QhullError:
        return [go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode="markers",
            marker=dict(color=color, size=4, opacity=opacity),
            name=name,
            showlegend=showlegend,
        )]
