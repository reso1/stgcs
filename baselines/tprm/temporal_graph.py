""" Temporal Graph (temporal_graph.h/.cpp) """

from typing import List, Dict, Callable
from dataclasses import dataclass
from collections import defaultdict
import heapq

import time
import numpy as np
from matplotlib.axes import Axes

from mrmp.interval import Interval



@dataclass
class GraphPathResultEntry:
    node_id: int
    time: float


class TemporalGraphNode:

    def __init__(self, pos:np.ndarray, time_availabilities: List[Interval]) -> None:
        self.pos, self.time_availabilities = pos, time_availabilities

    def is_active_at(self, time:float) -> bool:
        if time < 0.0:
            raise ValueError("Time must be non-negative")
        for itvl in self.time_availabilities:
            if time >= itvl.start and time <= itvl.end:
                return True
        return False


class TemporalGraphEdge:

    def __init__(self, node_a:int, node_b:int) -> None:
        self.node_a, self.node_b = node_a, node_b

    def get_other_node_id(self, node_id:int) -> int:
        if node_id == self.node_a:
            return self.node_b
        elif node_id == self.node_b:
            return self.node_a
        else:
            raise ValueError("node id not found in edge")


class TemporalGraph:
    
    def __init__(self, v:float) -> None:
        self._nodes:List[TemporalGraphNode] = []
        self._edges:List[TemporalGraphEdge] = []
        self.movement_speed = v

    def add_node(self, node:TemporalGraphNode) -> int:
        self._nodes.append(node)
        return len(self._nodes) - 1

    def add_edge(self, edge:TemporalGraphEdge) -> int:
        self._edges.append(edge)
        return len(self._edges) - 1

    def get_node(self, node_id:int) -> TemporalGraphNode:
        if node_id < 0 or node_id >= self.num_nodes:
            raise ValueError("node id out of bounds")
        return self._nodes[node_id]

    def get_edge(self, edge_id:int) -> TemporalGraphEdge:
        if edge_id < 0 or edge_id >= self.num_edges:
            raise ValueError("Edge id out of bounds")
        return self._edges[edge_id]

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    def get_closest_node(self, pos:np.ndarray) -> int:
        return int(np.argmin([np.linalg.norm(node.pos - pos) for node in self._nodes]))
    
    def get_shortest_path(self, start_node_id:int, end_node_id:int, start_time:float, ecc_func:Callable, timeout_secs:float) -> List[GraphPathResultEntry]:
        if start_node_id < 0 or start_node_id >= self.num_nodes:
            raise ValueError("Start node id out of bounds")
        if end_node_id < 0 or end_node_id >= self.num_nodes:
            raise ValueError("End node id out of bounds")
        
        edge_buckets: Dict[int, List[int]] = defaultdict(list)
        for edge_id, edge in enumerate(self._edges):
            edge_buckets[edge.node_a].append(edge_id)
            edge_buckets[edge.node_b].append(edge_id)

        goal_node = self.get_node(end_node_id)
        heur = lambda node_id: np.linalg.norm(self.get_node(node_id).pos - goal_node.pos)
        OPEN, OPEN_set, CLOSED = [], set(), set()
        predecessor_map: Dict[int, int] = {}
        g_costs: Dict[int, float] = defaultdict(lambda: np.inf)
        f_costs: Dict[int, float] = defaultdict(lambda: np.inf)
        arrival_time: Dict[int, float] = defaultdict(lambda: np.inf)
        g_costs[start_node_id] = 0.0
        f_costs[start_node_id] = heur(start_node_id)
        arrival_time[start_node_id] = start_time
        
        ts = time.perf_counter()
        OPEN.append((f_costs[start_node_id], start_node_id))
        OPEN_set.add(start_node_id)
        while len(OPEN) > 0:
            f, current_id = heapq.heappop(OPEN)
            OPEN_set.remove(current_id)
            if current_id == end_node_id or time.perf_counter() - ts > timeout_secs:
                break

            CLOSED.add(current_id)
            for edge_id in edge_buckets[current_id]:
                successor_id = self.get_edge(edge_id).get_other_node_id(current_id)
                current_time = arrival_time[current_id]
                dx = self.get_node(current_id).pos - self.get_node(successor_id).pos
                edge_time = max([abs(d / self.movement_speed) for d in dx])
                xp = np.hstack([self.get_node(current_id).pos, current_time])
                xq = np.hstack([self.get_node(successor_id).pos, current_time + edge_time])
                if self._nodes[successor_id].is_active_at(current_time + edge_time) and not ecc_func(xp, xq):
                    if successor_id in CLOSED:
                        continue

                    tentative_g = g_costs[current_id] + edge_time
                    if successor_id in OPEN_set and tentative_g >= g_costs[successor_id]:
                        continue

                    predecessor_map[successor_id] = current_id
                    g_costs[successor_id] = tentative_g
                    f_costs[successor_id] = g_costs[successor_id] + heur(successor_id)
                    arrival_time[successor_id] = current_time + edge_time

                    if successor_id not in OPEN_set:
                        heapq.heappush(OPEN, (f_costs[successor_id], successor_id))
                        OPEN_set.add(successor_id)

        if arrival_time[end_node_id] == np.inf:
            return []
        
        path = []
        current_id = end_node_id
        while current_id != start_node_id:
            path.append(GraphPathResultEntry(current_id, arrival_time[current_id]))
            current_id = predecessor_map[current_id]
        path.append(GraphPathResultEntry(start_node_id, start_time))

        return path[::-1]

    def draw(self, ax: Axes) -> None:
        for edge in self._edges:
            node_a = self.get_node(edge.node_a)
            node_b = self.get_node(edge.node_b)
            ax.plot([node_a.pos[0], node_b.pos[0]], [node_a.pos[1], node_b.pos[1]], '.k-')

