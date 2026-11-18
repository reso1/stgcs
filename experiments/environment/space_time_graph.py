from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from itertools import combinations


import numpy as np
import networkx as nx

from pydrake.all import HPolyhedron, Point as DrakePoint

GCS_SOURCE_NAME = "source"
GCS_TARGET_NAME = "target"

from experiments.environment.interval import Interval, AABB
from experiments.environment.geometry import (
    remove_hpoly_redundancies, get_hpoly_bounds, make_hpolytope
)


import logging
logger = logging.getLogger(__name__)


class STVertex:

    def __init__(
        self,
        name: str,
        st_hpoly: HPolyhedron,
        time_itvl: Interval,
        space_itvls: List[Interval],
        parent: Optional[STVertex] = None,
        target_parent_list: List[STVertex]=[]
    ) -> None:
        self.name: str = name
        self.st_hpoly: HPolyhedron = st_hpoly
        self.time_itvl: Interval = time_itvl
        self.space_itvls: List[Interval] = space_itvls
        self.parent: Optional[STVertex] = parent
        self.depth: int = self.parent.depth + 1 if self.parent is not None else 0
        self.target_parent_list: List[STVertex] = target_parent_list
        assert self.parent is None or len(self.target_parent_list) == 0, \
            "Either parent or target_parent_list should be set, not both."

    @property
    def root(self) -> STVertex|List[STVertex]:
        if self.target_parent_list: # special case for target vertex
            return [tp.root for tp in self.target_parent_list]
        
        if self.parent is None:
            return self
        else:
            return self.parent.root
        
    @property
    def root_name(self) -> List[str]:
        if self.target_parent_list: # special case for target vertex
            return [tp.root.name for tp in self.target_parent_list]
        
        if self.parent is None:
            return [self.name]
        else:
            return [self.parent.root.name]


class STGCS:

    def __init__(
        self, spatial_sets:List[np.ndarray], 
        t0:float=0, tmax:float=1e2, vlimit:float=1.0, dt:float=1e-6
    ) -> None:
        
        self.spatial_sets = spatial_sets
        self.t0, self.tmax, self.vlimit, self.dt = t0, tmax, vlimit, dt

        self._spatial_hpolys = [make_hpolytope(r) for r in spatial_sets]
        self.dimension = self._spatial_hpolys[0].ambient_dimension()

        self.source: Optional[STVertex] = None
        self.source_comp: Optional[np.ndarray] = None
        self.target: Optional[STVertex] = None


        self.G = nx.DiGraph()
        self._st_vertices: Dict[str, STVertex] = {}
        self._dummy_targets: Dict[str, STVertex] = {}
        self._next_vertex_index: int = 0



    def copy(self) -> STGCS:
        ret = STGCS(
            spatial_sets = self.spatial_sets,
            t0 = self.t0, tmax = self.tmax,
            vlimit = self.vlimit, dt = self.dt
        )
        for v_name in self.G.nodes:
            ret.G.add_node(v_name, **self.G.nodes[v_name])
        for u_name, v_name in self.G.edges:
            ret.G.add_edge(u_name, v_name, **self.G.edges[u_name, v_name])
        ret._st_vertices = {name: vertex for name, vertex in self._st_vertices.items()}
        ret.source = self.source
        ret.source_comp = self.source_comp
        ret.target = self.target
        ret._dummy_targets = {name: vertex for name, vertex in self._dummy_targets.items()}
        ret._next_vertex_index = self._next_vertex_index
        return ret

    def get_vertex(self, name:str) -> Optional[STVertex]:
        if name == GCS_SOURCE_NAME:
            return self.source
        elif name == GCS_TARGET_NAME:
            return self.target
        elif name in self._dummy_targets:
            return self._dummy_targets[name]
    
        return self._st_vertices[name]

    def add_vertex(
        self, cvx_set:HPolyhedron, time_bound:Interval, space_bounds:List[Interval]=None,
        size_filter_tol:float=1e-6, name:str=None, parent:STVertex=None,
        remove_redundancies: bool = True,
    ) -> Optional[STVertex]:
        if time_bound.duration <= size_filter_tol:
            return
        
        if cvx_set.ambient_dimension() == self.dimension + 1:
            st_hpoly = remove_hpoly_redundancies(cvx_set) if remove_redundancies else cvx_set
        else:
            raise "cvx_set must be a time-extruded set"

        if space_bounds is None:
            space_bounds = []
            dims = [int(_i) for _i in range(self.dimension)]
            for lb, ub in zip(*get_hpoly_bounds(st_hpoly, dim=dims)):
                if ub - lb <= size_filter_tol:
                    return
                space_bounds.append(Interval(lb, ub))
        
        if name is None:
            name = f"v{self._next_vertex_index}"
            self._next_vertex_index += 1
        
        vertex = STVertex(name, st_hpoly, time_bound, space_bounds, parent)
        
        self.G.add_node(name)
        assert name not in self._st_vertices, f"Vertex {name} already exists!"
        self._st_vertices[name] = vertex

        return vertex

    def add_edges_bidir(self, u_name:str, v_name:str) -> None:
        if u_name == v_name or u_name not in self.G or v_name not in self.G:
            return
        
        u, v = self.get_vertex(u_name), self.get_vertex(v_name)

        if AABB(u.space_itvls, v.space_itvls) and \
           u.st_hpoly.IntersectsWith(v.st_hpoly):
            if np.allclose(u.time_itvl.end, v.time_itvl.start) and u.time_itvl.start <= v.time_itvl.end:
                self._add_edge(u_name, v_name)
            elif np.allclose(v.time_itvl.end, u.time_itvl.start) and v.time_itvl.start <= u.time_itvl.end:
                self._add_edge(v_name, u_name)
            else:
                self._add_edge(u_name, v_name)
                self._add_edge(v_name, u_name)
    
    def remove_vertex_from_graph(self, name:str) -> STVertex:
        """ remove a vertex from the graph and return the removed vertex.
            Note that the vertex is not deleted from the _st_vertices dictionary."""
        vertex = self.get_vertex(name)
        if name in self.G.nodes:
            self.G.remove_node(name)
        return vertex

    def has_path(self, mp_query:object) -> bool:
        """Check directed query connectivity, without certifying motion feasibility."""
        self._clean_source_targets()
        try:
            if not self._init_source_vertex(mp_query.start, mp_query.t_start):
                return False
            earliest = mp_query.t_start + np.max(abs(mp_query.goal - mp_query.start) / self.vlimit)
            return (self._init_target_vertex(mp_query.goal, earliest, mp_query.is_stay, emit_warnings=False)
                    and nx.has_path(self.G, GCS_SOURCE_NAME, GCS_TARGET_NAME))
        finally:
            self._clean_source_targets()

    def validate_query(self, mp_query:object) -> Tuple[bool, Optional[str]]:
        self._clean_source_targets()
        try:
            found_src = self._init_source_vertex(mp_query.start, mp_query.t_start)
            if not found_src:
                start_state = np.hstack([mp_query.start, mp_query.t_start]).tolist()
                return False, (
                    f"Start state {start_state} is not contained in any ST set according to the real STGCS source check."
                )

            t_earliest_arrival = mp_query.t_start + np.max(abs(mp_query.goal - mp_query.start) / self.vlimit)
            if t_earliest_arrival > self.tmax + 1e-9:
                return False, (
                    f"Goal {mp_query.goal.tolist()} is unreachable by tmax={self.tmax:.4f}. "
                    f"The real STGCS target check requires arrival no earlier than {t_earliest_arrival:.4f}."
                )

            found_tar = self._init_target_vertex(
                mp_query.goal,
                t_earliest_arrival,
                mp_query.is_stay,
                emit_warnings=False,
            )
            if not found_tar:
                return False, self._describe_invalid_target(mp_query.goal, t_earliest_arrival, mp_query.is_stay)

            return True, None
        finally:
            self._clean_source_targets()







    def clear_query_vertices(self) -> None:
        self._clean_source_targets()

    def make_leaves_roots(self) -> None:
        roots = list(self.leaf_vertices)
        self._st_vertices = {vertex.name: vertex for vertex in roots}

    def _add_edge(self, tail_name:str, head_name:str) -> None:
        self.G.add_edge(tail_name, head_name)


    def _init_source_vertex(self, source:np.ndarray, t_start:float) -> bool:
        # assume source is contained in only one convex set of a vertex; 
        # otherwise, it uses the first one found
        source_found = False
        self.source_comp = np.hstack([source, t_start])
        for v_name in self.G.nodes:
            v = self.get_vertex(v_name)
            if v.st_hpoly.PointInSet(self.source_comp):
                self.source = STVertex(GCS_SOURCE_NAME, DrakePoint(self.source_comp), 
                                  time_itvl = Interval(t_start, t_start), 
                                  space_itvls = [Interval(p, p) for p in source],
                                  parent=v)
                self.G.add_node(GCS_SOURCE_NAME)
                self._add_edge(GCS_SOURCE_NAME, v_name)
                source_found = True
                break
        
        return source_found

    def _planning_vertex_names(self) -> List[str]:
        """Return real ST vertices, excluding temporary query vertices."""
        return [
            v_name
            for v_name in self.G.nodes
            if v_name not in {GCS_SOURCE_NAME, GCS_TARGET_NAME}
            and v_name not in self._dummy_targets
        ]

    def _describe_invalid_target(self, goal:np.ndarray, t_earliest_arrival:float, is_stay:bool) -> str:
        goal = np.asarray(goal, dtype=float).reshape(-1)
        if t_earliest_arrival > self.tmax + 1e-9:
            return (
                f"Goal {goal.tolist()} is unreachable by tmax={self.tmax:.4f}. "
                f"The real STGCS target check requires arrival no earlier than {t_earliest_arrival:.4f}."
            )

        target = HPolyhedron.MakeBox(lb=np.hstack([goal, t_earliest_arrival]), ub=np.hstack([goal, self.tmax]))
        anytime_target = HPolyhedron.MakeBox(lb=np.hstack([goal, self.t0]), ub=np.hstack([goal, self.tmax]))
        target_tmax = np.hstack([goal, self.tmax])
        candidate_names: List[str] = []
        anytime_names: List[str] = []
        intersections: Dict[str, HPolyhedron] = {}
        v_target_tmax = None

        for v_name in self._planning_vertex_names():
            v_hpoly = self.get_vertex(v_name).st_hpoly
            itsc = None
            anytime_itsc = None
            if isinstance(v_hpoly, HPolyhedron):
                itsc = target.Intersection(v_hpoly)
                if itsc.IsEmpty():
                    itsc = None
                anytime_itsc = anytime_target.Intersection(v_hpoly)
                if anytime_itsc.IsEmpty():
                    anytime_itsc = None
            elif isinstance(v_hpoly, DrakePoint):
                v_point = v_hpoly.x()
                if target.PointInSet(v_point):
                    itsc = HPolyhedron.MakeBox(lb=v_point, ub=v_point)
                if anytime_target.PointInSet(v_point):
                    anytime_itsc = HPolyhedron.MakeBox(lb=v_point, ub=v_point)

            if anytime_itsc is not None:
                anytime_names.append(v_name)
            if itsc is not None:
                candidate_names.append(v_name)
                intersections[v_name] = itsc
                if v_hpoly.PointInSet(target_tmax):
                    v_target_tmax = v_name

        if len(candidate_names) == 0:
            if len(anytime_names) > 0:
                return (
                    f"Goal {goal.tolist()} intersects ST sets only before the earliest reachable time "
                    f"{t_earliest_arrival:.4f}. The real STGCS target check uses goal over "
                    f"[{t_earliest_arrival:.4f}, {self.tmax:.4f}]."
                )
            return (
                f"Goal {goal.tolist()} is not contained in any ST set over the valid arrival window "
                f"[{t_earliest_arrival:.4f}, {self.tmax:.4f}]."
            )

        if not is_stay:
            return (
                f"Goal {goal.tolist()} failed the real STGCS target initialization over "
                f"[{t_earliest_arrival:.4f}, {self.tmax:.4f}]."
            )

        if v_target_tmax is None:
            return (
                f"Stay target is invalid: goal {goal.tolist()} is not contained in any ST set at tmax={self.tmax:.4f}."
            )

        T = nx.Graph()
        T.add_nodes_from(candidate_names)
        for name_1, name_2 in combinations(candidate_names, 2):
            if intersections[name_1].IntersectsWith(intersections[name_2]):
                T.add_edge(name_1, name_2)

        valid = [v_name for v_name in candidate_names if nx.has_path(T, v_name, v_target_tmax)]
        if len(valid) == 0:
            return (
                f"Stay target is invalid: goal {goal.tolist()} is not continuously supported by intersecting ST sets "
                f"through tmax={self.tmax:.4f}."
            )

        return (
            f"Goal {goal.tolist()} failed the real STGCS target initialization over "
            f"[{t_earliest_arrival:.4f}, {self.tmax:.4f}]."
        )

    def _init_target_vertex(
        self,
        goal:np.ndarray,
        t_earliest_arrival:float,
        is_stay:bool,
        emit_warnings: bool = True,
    ) -> bool:
        if t_earliest_arrival > self.tmax + 1e-9:
            if emit_warnings:
                logger.warning("Earliest target arrival exceeds tmax")
            return False
        candidates: List[Tuple[str, str]] = []
        target = HPolyhedron.MakeBox(lb=np.hstack([goal, t_earliest_arrival]), ub=np.hstack([goal, self.tmax]))
        target_tmax = np.hstack([goal, self.tmax])
        v_target_tmax = None    # record the vertex containing the goal position at tmax
        intersections: Dict[str, HPolyhedron] = {}
        for v_name in self._planning_vertex_names():
            v_hpoly = self.get_vertex(v_name).st_hpoly
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
                v_tar_name = f"sub-{GCS_TARGET_NAME}-{len(candidates)}"
                candidates.append((v_tar_name, v_name))   # (dummy_target, residing_vertex)
                intersections[v_tar_name] = itsc
                if v_hpoly.PointInSet(target_tmax):
                    v_target_tmax = v_tar_name
        
        if len(candidates) == 0:
            if emit_warnings:
                logger.warning("No target vertices found")
            return False
        
        if not is_stay:
            valid = candidates
        elif v_target_tmax is None:
            if emit_warnings:
                logger.warning("No target vertex found at tmax; cannot create a stay target")
            return False
        else:
            # filter candidates to only those that can reach the target at tmax
            T = nx.Graph()
            T.add_nodes_from([name for name, _ in candidates])
            for (name_1, _), (name_2, _) in combinations(candidates, 2):
                if intersections[name_1].IntersectsWith(intersections[name_2]):
                    T.add_edge(name_1, name_2)

            valid = [(vtar, v_residing) for vtar, v_residing in candidates if nx.has_path(T, vtar, v_target_tmax)]
                    
        if len(valid) == 0:
            if emit_warnings:
                logger.warning("No valid target vertices")
            return False

        self.target = STVertex(GCS_TARGET_NAME, target,
                                time_itvl = Interval(t_earliest_arrival, self.tmax), 
                                space_itvls = [Interval(p, p) for p in goal],
                                target_parent_list=[self.get_vertex(_v[1]) for _v in valid])
        self.G.add_node(GCS_TARGET_NAME)

        # add edges: (residing-vertex -> dummy-target) and (dummy-target -> hypertarget)
        # dummy-target vertices are used to connect the residing vertices to the hypertarget and constrain the target
        # this allows for multiple paths to the hypertarget, which can be useful for relaxation
        for v_dummy_tar, v_residing in valid:
            self._dummy_targets[v_dummy_tar] = STVertex(v_dummy_tar, target, 
                                    time_itvl = Interval(t_earliest_arrival, self.tmax), 
                                    space_itvls = [Interval(p, p) for p in goal],
                                    parent=self.get_vertex(v_residing))
            self.G.add_node(v_dummy_tar)
            self._add_edge(v_residing, v_dummy_tar)
            self._add_edge(v_dummy_tar, GCS_TARGET_NAME)
        
        return True

    def _clean_source_targets(self) -> None:
        if not (self.source is None and self.target is None and len(self._dummy_targets) == 0):
            logger.debug("Source and target vertices already set; Unset them first")
            self.remove_vertex_from_graph(GCS_SOURCE_NAME)
            self.remove_vertex_from_graph(GCS_TARGET_NAME)
            for v_name in self._dummy_targets.keys():
                self.remove_vertex_from_graph(v_name)
                
            self._dummy_targets.clear()
            self.source, self.target, self.source_comp = None, None, None

    @property
    def graph_size_string(self) -> str:
        return f"|V|={self.G.number_of_nodes()} |E|={self.G.number_of_edges()}"

    @property
    def leaf_vertices(self) -> List[STVertex]:
        return [self.get_vertex(v_name) for v_name in self.G.nodes]
