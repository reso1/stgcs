""" Temporal PRM (temporal_prm.h/.cpp) """

from typing import List, Dict

import time
import numpy as np

from mrmp.interval import Interval

from pydrake.all import RandomGenerator

from environment.instance import Instance
from environment.obstacle import DynamicSphere, ConcatDynamicSphere
from baselines.tprm.temporal_graph import TemporalGraph, TemporalGraphNode, TemporalGraphEdge

from mrmp.graph import ShortestPathSolution
from mrmp.pbs import collision_checking_inner


class TemporalPRM:

    def __init__(self, istc:Instance, seed:int, v:float, robot_radius:float, cost_edge_threshold:float=0.25) -> None:
        self.istc, self.seed = istc, seed
        self.np_rng = np.random.RandomState(seed)
        self.drake_rng = RandomGenerator(seed)
        self.graph = TemporalGraph(v)
        self.robot_radius = robot_radius
        self.cost_edge_threshold = cost_edge_threshold
    
    def build(self, num_nodes:int, use_CSpace:bool=True, timeout_secs:float=np.inf) -> None:
        ts = time.perf_counter()
        for _ in range(num_nodes):
            if time.perf_counter() - ts > timeout_secs:
                break

            if use_CSpace:
                sample = self.istc.sample_CSpace(self.np_rng, self.drake_rng)
            else:
                is_blocked = True
                sample = self.istc.sample_bounding_box(self.np_rng)
                while is_blocked:
                    is_blocked = False
                    for obstacle in self.istc.O_Static:
                        if obstacle.is_colliding(sample, self.robot_radius):
                            is_blocked = True
                            sample = self.istc.sample_bounding_box(self.np_rng)
                            break

            node = TemporalGraphNode(sample, self.get_time_availability(sample))
            i = self.graph.add_node(node)

            # build edge with other nodes
            for j in range(self.graph.num_nodes - 1):
                other = self.graph.get_node(j)
                cost = np.linalg.norm(node.pos - other.pos)
                if cost > self.cost_edge_threshold:
                    continue

                is_blocked = False
                for obstacle in self.istc.O_Static:
                    if obstacle.is_colliding_lineseg(node.pos, other.pos, self.robot_radius):
                        is_blocked = True
                        break
                
                # issue: collision along edge is not checked
                if not is_blocked:
                    self.graph.add_edge(TemporalGraphEdge(i, j))

    def add_sample(self, use_CSpace:bool=True) -> None:
        if use_CSpace:
            sample = self.istc.sample_CSpace(self.np_rng, self.drake_rng)
        else:
            is_blocked = True
            sample = self.istc.sample_bounding_box(self.np_rng)
            while is_blocked:
                is_blocked = False
                for obstacle in self.istc.O_Static:
                    if obstacle.is_colliding(sample, self.robot_radius):
                        is_blocked = True
                        sample = self.istc.sample_bounding_box(self.np_rng)
                        break

        node = TemporalGraphNode(sample, self.get_time_availability(sample))
        i = self.graph.add_node(node)

        # build edge with other nodes
        for j in range(self.graph.num_nodes - 1):
            other = self.graph.get_node(j)
            cost = np.linalg.norm(node.pos - other.pos)
            if cost > self.cost_edge_threshold:
                continue

            is_blocked = False
            for obstacle in self.istc.O_Static:
                if obstacle.is_colliding_lineseg(node.pos, other.pos, self.robot_radius):
                    is_blocked = True
                    break
            
            # issue: collision along edge is not checked
            if not is_blocked:
                self.graph.add_edge(TemporalGraphEdge(i, j))

    def get_time_availability(self, pos:np.ndarray) -> List[Interval]:
        if len(self.istc.O_Dynamic) == 0:
            return [Interval(0.0, np.inf)]
        
        hit_infos = []
        for obs in self.istc.O_Dynamic:
            hit_infos.extend(obs.collision_intervals(pos, self.robot_radius))
        
        if len(hit_infos) == 0:
            return [Interval(0.0, np.inf)]
        
        time_availabilities = [Interval(0.0, np.inf)]
        
        def _apply_obstacle_hit_to_interval(hi:Interval, itvl:Interval) -> bool:
            if hi.start > itvl.start and hi.end < itvl.end:
                itvl.end = hi.start
                time_availabilities.append(Interval(hi.end, itvl.end))
            elif hi.start > itvl.start and hi.start < itvl.end:
                itvl.end = hi.start
            elif hi.end >itvl.start and hi.end < itvl.end:
                itvl.start = hi.end
            elif hi.start < itvl.start and hi.end > itvl.end:
                pass
            elif abs(hi.start - itvl.start) < 1e-6 and abs(hi.end - itvl.end) < 1e-6:
                return False
            return True
        
        for avai in time_availabilities:
            for hit in hit_infos:
                if not _apply_obstacle_hit_to_interval(hit, avai):
                    return []
        
        return time_availabilities

    def recompute_time_availability(self) -> None:
        for i in range(self.graph.num_nodes):
            node = self.graph.get_node(i)
            node.time_availabilities = self.get_time_availability(node.pos)

    def solve(self, start:np.ndarray, goal:np.ndarray, time_start:float, use_CSpace:bool, timeout_secs:float) -> ShortestPathSolution:
        ts = time.perf_counter()
        while time.perf_counter() - ts < timeout_secs:
            self.add_sample(use_CSpace)

        return self.try_solve(start, goal, time_start, timeout_secs)

    def try_solve(self, start:np.ndarray, goal:np.ndarray, time_start:float, timeout_secs:float) -> ShortestPathSolution:
        ts = time.perf_counter()
        closest_start_id = self.graph.get_closest_node(start)
        closest_goal_id = self.graph.get_closest_node(goal)

        start_node = self.graph.get_node(closest_start_id)
        goal_node = self.graph.get_node(closest_goal_id)

        failure_ret = ShortestPathSolution(False, -1.0, -1.0, [], [])
        for obstacle in self.istc.O_Static:
            if obstacle.is_colliding_lineseg(goal, goal_node.pos, self.robot_radius):
                return failure_ret
        
        tmp_start = TemporalGraphNode(start, self.get_time_availability(start))
        if not tmp_start.is_active_at(time_start):
            return failure_ret
        
        time_to_closest_start = np.max(np.abs(start - start_node.pos) / self.graph.movement_speed)
        path = self.graph.get_shortest_path(closest_start_id, closest_goal_id, time_start + time_to_closest_start, 
                                            self.edge_collision_checking, timeout_secs)
        if len(path) == 0:
            return failure_ret
        
        tmp_goal = TemporalGraphNode(goal, self.get_time_availability(goal))
        time_from_closest_goal = np.max(np.abs(goal - goal_node.pos) / self.graph.movement_speed)
        if not tmp_goal.is_active_at(path[-1].time + time_from_closest_goal):
            return failure_ret
        
        x_last = np.hstack([start, time_start])
        trajectory = []
        for entry in path[1:]:
            x = np.hstack([self.graph.get_node(entry.node_id).pos, entry.time])
            trajectory.append(np.hstack([x_last, x]))
            x_last = x

        trajectory.append(np.hstack([trajectory[-1][3:], goal, path[-1].time + time_from_closest_goal]))

        sol = ShortestPathSolution(True, trajectory[-1][-1] - time_start, time.perf_counter() - ts, [], trajectory)
        sol.itvl = Interval(time_start, trajectory[-1][-1])
        sol.dim = self.istc.dim + 1
        return sol

    def edge_collision_checking(self, xp:np.ndarray, xq:np.ndarray) -> bool:
        # edge collision checking to fix the issue
        for obs in self.istc.O_Dynamic:
            segments = []
            if isinstance(obs, DynamicSphere):
                x0 = np.hstack([obs.x0, obs.itvl.start])
                xt = np.hstack([obs.xt, obs.itvl.end])
                segments = [(x0, xt)]
            elif isinstance(obs, ConcatDynamicSphere):
                segments = []
                for seg in obs.segments:
                    x0 = np.hstack([seg.x0, seg.itvl.start])
                    xt = np.hstack([seg.xt, seg.itvl.end])
                    segments.append((x0, xt))
            else:
                raise NotImplementedError("Unknown dynamic obstacle type")

            for seg in segments:
                x0, xt = seg
                if collision_checking_inner(x0, xt, xp, xq, self.robot_radius + obs.radius):
                    return True
        
        return False
