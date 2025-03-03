from __future__ import annotations
from typing import Tuple, List, Dict, Set
from matplotlib.axes import Axes
from collections import defaultdict
import numpy as np

from enum import IntEnum

from environment.instance import Instance
from baselines.st_rrt_star.state import State, lerp, cost_func, distance, time_to_reach


class GrowState(IntEnum):
    TRAPPED = 0
    ADVANCED = 1
    REACHED = 2
    CONNECTED = 3


class TreeNode:
    
    def __init__(self, state:State, parent:TreeNode=None, is_goal:bool=False) -> None:
        self.state = state
        self.parent: TreeNode = parent
        self.is_dummy = False
        self.is_goal = is_goal
    
    @staticmethod
    def dummy() -> TreeNode:
        node = TreeNode(State(np.inf, -1.0))
        node.is_dummy = True
        return node
    
    def __eq__(self, other:TreeNode) -> bool:
        return self.state == other.state
    
    def __hash__(self) -> int:
        if self.is_dummy:
            return hash("Dummy")
        return self.state.__hash__()

    def __repr__(self) -> str:
        if self.is_dummy:
            return f"TreeNode.Dummy"
        if self.parent is None:
            parent_str = "None"
        elif self.parent.is_dummy:
            parent_str = "Dummy"
        else:
            parent_str = f"{self.parent.state.pos.round(3)}" if self.parent is not None else ""
        return f"TreeNode(x={self.state.pos.round(3)}, t={self.state.time:.3f}, parent={parent_str})"


class Tree:
    START_TREE = 0
    GOAL_TREE = 1
    UNINITIALIZED = 2

    def __init__(self):
        self._root: TreeNode = None
        self._nodes: List[TreeNode] = []
        self._type: int = Tree.UNINITIALIZED
        self._explored: Set[State] = set()
        self._children: Dict[TreeNode, List[TreeNode]] = defaultdict(list)

    @staticmethod
    def init_start_tree(x0:State) -> Tree:
        tree = Tree()
        tree._root = tree.add_node(x0, parent=None, is_goal=False)
        tree._type = Tree.START_TREE
        return tree
    
    @staticmethod
    def init_goal_tree() -> Tree:
        tree = Tree()
        tree._root = TreeNode.dummy()
        tree._type = Tree.GOAL_TREE
        return tree

    def add_goal_state(self, xg:State) -> TreeNode:
        assert self._type == Tree.GOAL_TREE
        return self.add_node(xg, parent=self._root, is_goal=True)

    def add_node(self, x:State, parent:TreeNode, is_goal:bool) -> TreeNode:
        node = TreeNode(x, parent=parent, is_goal=is_goal)
        self._nodes.append(node)
        self._explored.add(x)
        if parent is not None:
            self._children[parent].append(node)
            
        return node

    def remove_recursively(self, node:TreeNode) -> List[TreeNode]:
        """ recursively remove node and all its children """
        assert self._type == Tree.START_TREE # never use it for the goal tree
        removed = []
        if self._type == Tree.START_TREE:
            for child in self._children[node]:
                removed += self.remove_recursively(child)

            if node in self._nodes:
                self._nodes.remove(node)
                del self._children[node]
                removed.append(node)
        
        return removed

    def nearest_neighbor(self, x:State, vlimit:float, lambda_nn:float) -> TreeNode:
        x_nearest = (np.inf, None)
        for node in self._nodes:
            c = cost_func(node.state, x, vlimit, lambda_nn, is_start_tree = self._type == Tree.START_TREE)
            if c < x_nearest[0]:
                x_nearest = (c, node)
        return x_nearest[1]
    
    def extend(self, x:State, istc:Instance, epsilon:float, vlimit:float, lambda_nn:float, robot_radius:float) -> Tuple[GrowState, TreeNode|None]:
        node_near = self.nearest_neighbor(x, vlimit, lambda_nn)
        if node_near is None:
            return GrowState.TRAPPED, None
        
        x_near = node_near.state
        dist = distance(x_near, x, vlimit, is_start_tree = self._type == Tree.START_TREE)
        if dist < epsilon and not istc.collision_checking_seg(x_near.pos, x.pos, x_near.time, x.time, robot_radius):
            node = self.add_node(x, parent=node_near, is_goal=False)
            return GrowState.REACHED, node

        x_new = lerp(x_near, x, epsilon / np.linalg.norm(x.pos - x_near.pos))
        if istc.collision_checking_seg(x_near.pos, x_new.pos, x_near.time, x_new.time, robot_radius):
            return GrowState.TRAPPED, None
        elif x_new in self._explored:
            return GrowState.REACHED, None
        else:
            node_new = self.add_node(x_new, parent=node_near, is_goal=False)
            return GrowState.ADVANCED, node_new
    
    def debug_check(self) -> bool:
        for _parent, _children in self._children.items():
            for _child in _children:
                assert _child.parent == _parent
                if self._type == Tree.START_TREE:
                    assert _child.state.time > _parent.state.time
                elif self._type == Tree.GOAL_TREE and not _parent.is_dummy:
                    assert _child.state.time < _parent.state.time

    def rewire(self, node_new:TreeNode, istc:Instance, vlimit:float, robot_radius:float) -> int:
        # assert self._type == Tree.GOAL_TREE
        # check each node in the GOAL_TREE
        # rewire if node.time < node_new.time < node.parent.time (shortcutting node<-node.parent to node<-node_new)
        rewire_count = 0
        for node in self._nodes:
            if not node.parent.is_dummy and node.state.time < node_new.state.time < node.parent.state.time: 
                t = time_to_reach(node_new.state, node.state, vlimit, is_start_tree = False)
                if t != np.inf and not istc.collision_checking_seg(node_new.state.pos, node.state.pos, node_new.state.time, node.state.time, robot_radius):
                    self._children[node.parent].remove(node)
                    self._children[node_new].append(node)
                    node.parent = node_new
                    rewire_count += 1

        return rewire_count

    def connect(self, x:State, Tother:Tree, vlimit:float, istc:Instance, epsilon:float, lambda_nn:float, robot_radius:float) -> Tuple[GrowState, List[State]]:
        while True:
            gs, node = self.extend(x, istc, epsilon, vlimit, lambda_nn, robot_radius)
            if gs != GrowState.ADVANCED:
                return GrowState.TRAPPED, []
            path = self.check_solution(node, Tother, vlimit, istc, lambda_nn, robot_radius)
            if path != []:
                return GrowState.CONNECTED, path
    
    def check_solution(self, node:TreeNode, Tother:Tree, vlimit:float, istc:Instance, lambda_nn:float, robot_radius:float) -> List[State]:
        # check if node (in current tree) can be connected to the other tree
        node_near = Tother.nearest_neighbor(node.state, vlimit, lambda_nn)
        x_near = node_near.state
        t = time_to_reach(x_near, node.state, vlimit, is_start_tree = Tother._type == Tree.START_TREE)
        if t != np.inf and not istc.collision_checking_seg(x_near.pos, node.state.pos, x_near.time, node.state.time, robot_radius):
            pi_this = self.reconstruct_path(node)
            pi_other = Tother.reconstruct_path(node_near)
            return pi_this + pi_other if self._type == Tree.START_TREE else pi_other + pi_this
        
        return []

    def reconstruct_path(self, node:TreeNode) -> List[State]:
        path = []
        while node is not None and not node.is_dummy:
            path.append(node.state)
            node = node.parent

        return path[::-1] if self._type == Tree.START_TREE else path

    def draw_2d(self, ax:Axes, color:str, with_time:bool=False) -> None:
        if not self._root.is_dummy:
            if with_time:
                ax.plot([self._root.state.pos[0]], [self._root.state.pos[1]], [self._root.state.time], f'{color}x')
            else:
                ax.plot([self._root.state.pos[0]], [self._root.state.pos[1]], f'{color}x')
        for node in self._nodes:
            if node.parent is not None and not node.parent.is_dummy:
                parent_state = node.parent.state
                if with_time:
                    ax.plot([node.state.pos[0], parent_state.pos[0]], [node.state.pos[1], parent_state.pos[1]], [node.state.time, parent_state.time], f'.{color}--')
                else:
                    ax.plot([node.state.pos[0], parent_state.pos[0]], [node.state.pos[1], parent_state.pos[1]], f'.{color}--')
            else:
                if with_time:
                    ax.plot([node.state.pos[0]], [node.state.pos[1]], [node.state.time], f'.{color}--')
                else:
                    ax.plot([node.state.pos[0]], [node.state.pos[1]], f'.{color}--')
