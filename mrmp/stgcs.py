from __future__ import annotations
from typing import List, Tuple, Dict, Set
from itertools import combinations, product
from copy import deepcopy

from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D

import numpy as np
import networkx as nx

from pydrake.all import (
    Binding, HPolyhedron, VPolytope, Constraint,
    Point as DrakePoint,
    LinearConstraint, LinearEqualityConstraint,
    L2NormCost,
)

from mrmp.geometry.polyhedron import Polyhedron
from mrmp.geometry.point import Point
from mrmp.graph import ShortestPathSolution, Edge, Graph, Vertex
from mrmp.interval import Interval, AABB
from mrmp.utils import (
    timeit, time_extruded, get_hpoly_bounds, make_hpolytope,
    squash_multi_points, draw_3d_set, draw_2d_set
)


GCS_SOURCE_NAME = "source"
GCS_TARGET_NAME = "target"
BASE_MAX_ROUNDED_PATHS = 1000
BASE_MAX_ROUNDING_TRIALS = 1000


class STGCS:

    def __init__(
        self, dim:int, order:int=0, t0:float=0, tmax:float=1e2, vlimit:float=1.0, dt:float=1e-6
    ) -> None:
        self.order, self.num_pts_per_verts = order, order + 2
        self.dim, self.t0, self.tmax, self.vlimit, self.dt = dim, t0, tmax, vlimit, dt
        self._vertex_names = []

        self.G = Graph()

        # define constraint that time must be increasing for each vertex
        A_vmax = np.hstack([
             np.eye(self.dim-1),  vlimit * np.ones((self.dim-1, 1)),
            -np.eye(self.dim-1), -vlimit * np.ones((self.dim-1, 1))])
        A_vmin = np.hstack([
            -np.eye(self.dim-1),  vlimit * np.ones((self.dim-1, 1)),
             np.eye(self.dim-1), -vlimit * np.ones((self.dim-1, 1))])
        A_dt = np.array([0] * (self.dim-1) + [1] + [0] * (self.dim-1) + [-1])
        A_time = np.vstack([A_vmax, A_vmin, A_dt])
        b_time = np.hstack([np.zeros(self.dim-1), np.zeros(self.dim-1), -dt])
        self.time_constraint = LinearConstraint(A_time, -np.inf*np.ones_like(b_time), b_time)
        
        self.time_cost = L2NormCost(A_dt.reshape(1, -1), np.zeros(1))

        # define the continuity constraint for each edge
        A_cont = np.hstack([
            np.zeros((self.dim, self.dim)), np.eye(self.dim),
            -np.eye(self.dim), np.zeros((self.dim, self.dim))])
        b_cont = np.zeros(self.dim)
        self.cont_constraint = LinearEqualityConstraint(A_cont, b_cont)
        
    def copy(self) -> STGCS:
        ret = STGCS(self.dim, t0=self.t0, tmax=self.tmax, vlimit=self.vlimit, dt=self.dt)
        ret._vertex_names = deepcopy(self._vertex_names)

        for v_name, v in self.G.vertices.items():
            ret.G.add_vertex(v, name=v_name)

        for e_name, e in self.G.edges.items():
            ret.G.add_edge(e)
        
        return ret
    
    @staticmethod
    def from_env(
        env, t0:float=0, tmax:float=1e2, vlimit:float=1.0, dt:float=1e-6
    ) -> STGCS:
        sets = [make_hpolytope(V) for V in env.C_Space]
        dim = 1 + sets[0].ambient_dimension()

        stgcs = STGCS(dim, t0=t0, tmax=tmax, vlimit=vlimit, dt=dt)

        for i, s in enumerate(sets):
            t_set = time_extruded(s, t0, tmax)
            lb = np.min(env.C_Space[i], axis=0)
            ub = np.max(env.C_Space[i], axis=0)
            space_bounds = [Interval(l, u) for l, u in zip(lb, ub)]
            stgcs.add_vertex(t_set, Interval(t0, tmax), space_bounds)

        for u_name, v_name in combinations(stgcs.G.vertex_names, 2):
            stgcs.try_build_edge(u_name, v_name)
        
        for obs in env.O_Dynamic:
            stgcs = obs.reserve(stgcs, env.robot_radius, True, True)

        return stgcs

    def add_vertex(self, hpoly:HPolyhedron, itvl:Interval, space_bounds:List[Interval]) -> Vertex:
        name = self.get_new_vertex_name()
        comp_hpoly = hpoly.CartesianPower(self.num_pts_per_verts)
        ply = Polyhedron(comp_hpoly.A(), comp_hpoly.b(), should_compute_vertices=False)
        vertex = Vertex(ply, costs=[self.time_cost], constraints=[self.time_constraint], name=name, itvl=itvl)
        vertex.space_bounds = space_bounds
        self.G.add_vertex(vertex, name=name)
        return vertex

    def try_add_vertex(self, hpoly:HPolyhedron, itvl:Interval, tol:float=1e-6) -> Vertex|None:
        if itvl.duration <= tol:
            return

        space_bounds = []
        dims = [int(_i) for _i in range(self.dim - 1)]
        for lb, ub in zip(*get_hpoly_bounds(hpoly, dim=dims)):
            if ub - lb <= tol:
                return
            space_bounds.append(Interval(lb, ub))

        return self.add_vertex(hpoly, itvl, space_bounds)
        
    def add_edge(self, u_name:str, v_name:str) -> None:
        self.G.add_edge(Edge(u_name, v_name, constraints=[self.cont_constraint]))
        assert len(self.G._gcs.Edges()) == self.G.n_edges

    def remove_vertex(self, name:str) -> Vertex:
        vertex = self.G.vertices[name]
        self._vertex_names.append(name)
        self.G.remove_vertex(name)
        assert len(self.G._gcs.Edges()) == self.G.n_edges
        return vertex

    def get_new_vertex_name(self) -> str:
        if len(self._vertex_names) == 0:
            return f"v{self.G.n_vertices}"
        
        return self._vertex_names.pop()
    
    def try_build_edge(self, u_name:str, v_name:str) -> None:
        if u_name == v_name or u_name not in self.G.vertices or v_name not in self.G.vertices:
            return
        
        u, v = self.G.vertices[u_name], self.G.vertices[v_name]
        if not u.convex_set.set.IntersectsWith(v.convex_set.set):
            return
        
        if np.allclose(u.itvl.end, v.itvl.start) and u.itvl.start <= v.itvl.end:
            self.add_edge(u_name, v_name)
        elif np.allclose(v.itvl.end, u.itvl.start) and v.itvl.start <= u.itvl.end:
            self.add_edge(v_name, u_name)
        else:
            self.add_edge(u_name, v_name)
            self.add_edge(v_name, u_name)

    def build_source_vertex(self, source:np.ndarray, t_start:float) -> str:
        # assume source is contained in only one convex set of a vertex
        V_src = []
        source = np.tile(np.hstack([source, t_start]), self.num_pts_per_verts)
        for v_name, v in self.G.vertices.items():
            if v.convex_set.set.PointInSet(source):
                A_src = np.eye(self.dim * self.num_pts_per_verts)
                src_cstr = LinearEqualityConstraint(A_src, source)
                src = Vertex(Point(source), constraints=[src_cstr], name=GCS_SOURCE_NAME)
                V_src.append((src, v_name))
        
        if len(V_src) == 0:
            return None
        
        if len(V_src) > 1:
            print("Multiple source vertices found")
        
        self.G.add_vertex(V_src[0][0], name=GCS_SOURCE_NAME)
        self.add_edge(GCS_SOURCE_NAME, V_src[0][1])
        
        return GCS_SOURCE_NAME

    def build_target_vertex(self, goal:np.ndarray) -> List[str]:
        candidates: List[Tuple[Vertex, str]] = []
        target = HPolyhedron.MakeBox(lb=np.hstack([goal, self.t0]), ub=np.hstack([goal, self.tmax]))
        target = target.CartesianPower(self.num_pts_per_verts)
        target_tmax = np.tile(np.hstack([goal, self.tmax]), self.num_pts_per_verts)
        
        v_target_tmax = None
        intersections: Dict[str, HPolyhedron] = {}
        for v_name, v in self.G.vertices.items():
            v_hpoly = v.convex_set.set
            if isinstance(v_hpoly, HPolyhedron):
                itsc = target.Intersection(v_hpoly)
                if itsc.IsEmpty():
                    itsc = None
            elif isinstance(v_hpoly, DrakePoint):
                v_point = v_hpoly.x()
                itsc = HPolyhedron.MakeBox(lb=v_point, ub=v_point)
                if not target.PointInSet(v_point):
                    itsc = None
            else:
                itsc = None
            
            if itsc is not None:
                A_dst = np.block([
                    [np.eye(self.dim - 1), np.zeros((self.dim - 1, 1 + self.dim * (self.num_pts_per_verts - 1)))],
                    [np.zeros((self.dim - 1, self.dim * (self.num_pts_per_verts - 1))), np.eye(self.dim - 1), np.zeros((self.dim - 1, 1))],
                    [0] * (self.dim - 1) + [1] + [0] * self.dim * (self.num_pts_per_verts - 2) + [0] * (self.dim - 1) + [-1]
                ])
                tar_csts = LinearEqualityConstraint(A_dst, np.hstack([np.tile(goal, self.num_pts_per_verts), 0]))
                name = f"sub-{GCS_TARGET_NAME}-{len(candidates)}"
                v_target = Vertex(Polyhedron(target.A(), target.b(), should_compute_vertices=False),
                              constraints=[tar_csts], name=name)
                candidates.append((v_target, v_name))
                intersections[name] = itsc
                if v.convex_set.set.PointInSet(target_tmax):
                    v_target_tmax = v_target
        
        if len(candidates) == 0 or v_target_tmax is None:
            print("No target vertices found connecting to (goal, tmax)")
            return
        
        T = nx.Graph()
        T.add_nodes_from([vtar.name for vtar, _ in candidates])
        for (vtar1, _), (vtar2, _) in combinations(candidates, 2):
            if intersections[vtar1.name].IntersectsWith(intersections[vtar2.name]):
                T.add_edge(vtar1.name, vtar2.name)
        
        # create a hyper target vertex
        hyper_target = Vertex(convex_set = Point(target_tmax), name=GCS_TARGET_NAME)
        self.G.add_vertex(hyper_target, name=GCS_TARGET_NAME)
        
        # connect the hyper target to all target vertices
        valid = []
        for vtar, vconn_name in candidates:
            if nx.has_path(T, vtar.name, v_target_tmax.name):
                valid.append(vtar.name)
                self.G.add_vertex(vtar, name=vtar.name)
                self.add_edge(vconn_name, vtar.name)
                self.G.add_edge(Edge(vtar.name, GCS_TARGET_NAME, costs=[], constraints=[]))
        
        if len(valid) == 0:
            print("No target vertices found connecting to (goal, tmax)")
            self.G.remove_vertex(GCS_TARGET_NAME)
            return
        
        return [GCS_TARGET_NAME] + valid
    
    def solve(
        self, start:np.ndarray, goal:np.ndarray, t_start:float, 
        relaxation:bool=True, max_rounded_paths:int=1000, max_rounding_trials:int=1000
    ) -> ShortestPathSolution:
        
        failure_ret = ShortestPathSolution(False, -1, -1, [], [])
        
        src_name  = self.build_source_vertex(start, t_start)
        tar_names = self.build_target_vertex(goal)

        if src_name is None or tar_names is None or len(tar_names) <= 1:
            print("Failed to build source or target vertices")
            return failure_ret
        
        self.G.set_source(GCS_SOURCE_NAME)
        self.G.set_target(GCS_TARGET_NAME)

        if not self.G.check_feasiblity():
            print("STGCS solving failed: No feasible path found")
            return failure_ret
       
        if relaxation:
            sol = self.G.solve_shortest_path(
                max_rounded_paths = max_rounded_paths,
                max_rounding_trials = max_rounding_trials,
            )
        else:
            sol = self.G.solve_shortest_path_optimally()

        self.G._source_name = None
        self.G._target_name = None
        for v_name in [src_name] + tar_names:
            self.G.remove_vertex(v_name)
        
        if not sol.is_success:
            print("Failed to find a solution")
            return failure_ret
        
        sol.vertex_path = sol.vertex_path[1:-2]
        sol.trajectory = sol.trajectory[1:-2]
        sol.itvl = Interval(t_start, sol.trajectory[-1][-1])
        sol.cost = sol.itvl.duration
        sol.dim = self.dim
        
        return sol

    def draw(self, ax:Axes|Axes3D, edges=False, set_labels=True, bounds:List[Interval]=[]) -> None:
        if isinstance(ax, Axes3D):
            assert self.dim == 3, "3D plotting is only supported for 3D STGCS"
            c = ['r', 'c', 'b', 'y', 'g', 'm']
            for i, v in enumerate(self.G.vertices.values()):
                if v.name[0] != 'v' or (bounds != [] and not AABB(v.space_bounds + [v.itvl], bounds)):
                    continue
                hpoly = squash_multi_points(v.convex_set.set, dim=self.dim)
                center = np.mean(VPolytope(hpoly).vertices(), axis=1)
                draw_3d_set(hpoly, ax, fc=c[i%len(c)], alpha=0.2)
                if set_labels:
                    ax.text(center[0], center[1], center[2], v.name, color='k')
            
            if edges:
                for e in self.G.edges.values():
                    u, v = self.G.vertices[e.u], self.G.vertices[e.v]
                    if u.name[0] != 'v' or v.name[0] != 'v':
                        continue

                    u_hpoly = squash_multi_points(u.convex_set.set, dim=self.dim)
                    v_hpoly = squash_multi_points(v.convex_set.set, dim=self.dim)
                    P = VPolytope(u_hpoly).vertices()
                    Q = VPolytope(v_hpoly).vertices()
                    cp, cq = np.mean(P, axis=1), np.mean(Q, axis=1)
                    ax.quiver(cp[0], cp[1], cp[2], cq[0] - cp[0], cq[1] - cp[1], cq[2] - cp[2], color='r', arrow_length_ratio=0.1)
        
        elif isinstance(ax, Axes):
            assert self.dim == 2, "2D plotting is only supported for 2D STGCS"
            c = ['r', 'c', 'b', 'y', 'g', 'm', 'k']
            for i, v in enumerate(self.G.vertices.values()):
                if v.name[0] != 'v':
                    continue
                hpoly = squash_multi_points(v.convex_set.set, dim=self.dim)
                center = np.mean(VPolytope(hpoly).vertices(), axis=1)
                draw_2d_set(hpoly, ax, color=c[i%len(c)])
                if set_labels:
                    ax.text(center[0], center[1], v.name, color='k')

            if edges:
                for e in self.G.edges.values():
                    u, v = self.G.vertices[e.u], self.G.vertices[e.v]
                    if u.name[0] != 'v' or v.name[0] != 'v':
                        continue

                    u_hpoly = squash_multi_points(u.convex_set.set, dim=self.dim)
                    v_hpoly = squash_multi_points(v.convex_set.set, dim=self.dim)
                    P = VPolytope(u_hpoly).vertices()
                    Q = VPolytope(v_hpoly).vertices()
                    cp, cq = np.mean(P, axis=1), np.mean(Q, axis=1)
                    ax.arrow(cp[0], cp[1], cq[0] - cp[0], cq[1] - cp[1], color='r', width=0.1)
            
        else:
            raise ValueError("Invalid axis type")

    def draw_with_solution(self, ax:Axes3D, sol:ShortestPathSolution, set_labels=True, bounds:List[Interval]=[]) -> None:
        c = ['r', 'c', 'b', 'y', 'g', 'm', 'k']
        for i, (v_name, wp) in enumerate(zip(sol.vertex_path, sol.trajectory)):
            v = self.G.vertices[v_name]
            xp, xq = wp[:3], wp[3:]
            if v_name[0] != 'v' or (bounds != [] and not AABB(v.space_bounds + [v.itvl], bounds)):
                continue

            ax.plot([xp[0], xq[0]], [xp[1], xq[1]], [xp[2], xq[2]], f'-ok', linewidth=2)
            hpoly = squash_multi_points(v.convex_set.set, dim=self.dim)
            center = np.mean(VPolytope(hpoly).vertices(), axis=1)
            draw_3d_set(hpoly, ax, fc=c[i%len(c)], alpha=0.25)
            if set_labels:
                ax.text(center[0], center[1], center[2], v_name, color='k')
