from __future__ import annotations
from typing import Tuple, Set, List
from dataclasses import dataclass

import time
import numpy as np
from tqdm import tqdm

from environment.instance import Instance

from pydrake.all import RandomGenerator

from baselines.st_rrt_star.state import State, time_to_reach
from baselines.st_rrt_star.tree import Tree, TreeNode, GrowState

from mrmp.graph import ShortestPathSolution
from mrmp.interval import Interval


@dataclass
class Options:
    p_goal: float = 0.1
    range_factor:float = 2.0
    initial_batch_size:int = 512
    sample_ratio: int = 4
    epsilon: float = 0.1
    max_runtime_in_secs: float = np.inf
    max_iterations: int = int(1e3)
    lambda_nn: float = 0.2 # weight of space distance for nearest neighbor search
    use_CSpace_sampling: bool = True
    return_first_valid: bool = False

    @staticmethod
    def default() -> Options:
        return Options()


class BoundVariables:

    def __init__(self, P:Options) -> None:
        self.time_range = P.range_factor
        self.new_time_range = P.range_factor
        self.batch_size = P.initial_batch_size
        self.samples_in_batch = 0
        self.total_samples = 0
        self.batch_probability = 1.0
        self.goals: Set[TreeNode] = set()
        self.new_goals: Set[State] = set()

class STRRTStar:

    def __init__(self, istc:Instance, seed:int, vlimit:float) -> None:
        self.istc, self.seed, self.vlimit = istc, seed, vlimit
        self.np_rng = np.random.RandomState(seed)
        self.drake_rng = RandomGenerator(seed)
        
    def solve(self, start_pos:np.ndarray, goal_pos:np.ndarray, t0:float, t_max:float, P:Options, robot_radius:float) -> ShortestPathSolution:
        ######## DEBUG ########
        # plt.ion()
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #######################
        
        sol_opt = None
        x_start = State(start_pos, t0)
        Ta, Tb = Tree.init_start_tree(x_start), Tree.init_goal_tree()
        B = BoundVariables(P)
        ts = time.perf_counter()
        for n_iter in range(P.max_iterations):
            T_start, T_goal = (Ta, Tb) if Ta._type == Tree.START_TREE else (Tb, Ta)
            B = self.update_goal_region(B, P, t_max)
            if len(T_goal._nodes) == 0 or P.p_goal >= self.np_rng.random():
                B = self.sample_goal(x_start, goal_pos, T_goal, t_max, B)

            x_rand = self.sample_conditionally(x_start, B, P, ts)
            if x_rand is None or time.perf_counter() - ts > P.max_runtime_in_secs:
                break

            gs, node_new = Ta.extend(x_rand, self.istc, P.epsilon, self.vlimit, P.lambda_nn, robot_radius)
            if gs != GrowState.TRAPPED:
                B.samples_in_batch += 1
                B.total_samples += 1
                if Ta._type == Tree.GOAL_TREE:
                    num_rewired = Ta.rewire(node_new, self.istc, self.vlimit, robot_radius)
    
                gs_conn, sol = Tb.connect(node_new.state, Ta, self.vlimit, self.istc, P.epsilon, P.lambda_nn, robot_radius)
                if gs_conn == GrowState.CONNECTED:
                    t_max, sol_opt = self.update_solution(sol_opt, t_max, sol, T_start, T_goal, B, P, robot_radius)
                    if P.return_first_valid:
                        break 

            ######## DEBUG ########
            # ax.clear()
            # ax.axis("off")
            # Ta.draw_2d(ax, 'r', with_time=True)
            # Tb.draw_2d(ax, 'b', with_time=True)
            # for obs in self.istc.O_Static:
            #     obs.draw_with_time(ax, tmax=t_max)
            # for obs in self.istc.O_Dynamic:
            #     obs.draw(ax)
            # if sol_opt is not None:
            #     for x, y in zip(sol_opt[:-1], sol_opt[1:]):
            #         ax.plot([x.pos[0], y.pos[0]], [x.pos[1], y.pos[1]], [x.time, y.time], '-k')
            # plt.draw()
            # plt.waitforbuttonpress(0.1)
            #######################

            Ta, Tb = Tb, Ta # Swap trees Ta and Tb

        ######## DEBUG ########
        # plt.ioff()
        # plt.show()
        #######################

        if sol_opt is not None:
            x_last = np.hstack([sol_opt[0].pos, sol_opt[0].time])
            trajectory = []
            for state in sol_opt[1:]:
                x = np.hstack([state.pos, state.time])
                trajectory.append(np.hstack([x_last, x]))
                x_last = x

            return ShortestPathSolution(
                    is_success = True, 
                    cost = sol_opt[-1].time - sol_opt[0].time, 
                    time = time.perf_counter() - ts, 
                    vertex_path = [], 
                    trajectory = trajectory,
                    itvl = Interval(sol_opt[0].time, sol_opt[-1].time),
                    dim = self.istc.dim + 1
                )
        
        return ShortestPathSolution(False, -1.0, -1.0, [], [])

    def update_goal_region(self, B:BoundVariables, P:Options, t_max:float) -> BoundVariables:
        if t_max == np.inf and B.samples_in_batch == B.batch_size:
            B.time_range = B.new_time_range
            B.new_time_range *= P.range_factor
            B.batch_size = int( (P.range_factor - 1) * B.total_samples / P.sample_ratio )
            B.batch_probability = (1 - P.sample_ratio) / P.range_factor
            B.goals = set.union(B.goals, B.new_goals)
            B.new_goals = set()
            B.samples_in_batch = 0
        return B

    def sample_goal(self, x_start:State, goal_pos:np.ndarray, T_goal:Tree, t_max:float, B:BoundVariables) -> BoundVariables:
        t_min = self.travel_time(goal_pos, x_start.pos)
        sample_old_batch = self.np_rng.random() <= B.batch_probability
        if t_max != np.inf:
            t_lb, t_ub = t_min, t_max
        elif sample_old_batch:
            t_lb, t_ub = t_min, t_min * B.time_range
        else:
            t_lb = t_min * B.time_range
            t_ub = t_min * B.new_time_range
        
        if t_ub > t_lb:
            t = self.np_rng.uniform(t_lb, t_ub)
            goal_node = T_goal.add_goal_state(State(goal_pos, t))
            if sample_old_batch:
                B.goals.add(goal_node)
            else:
                B.new_goals.add(goal_node)
        
        return B

    def sample_conditionally(self, x_start:State, B:BoundVariables, P:Options, ts:float) -> State|None:
        t_lb = t_ub = 0.0
        while t_lb >= t_ub:
            if P.use_CSpace_sampling:
                sample = self.istc.sample_CSpace(self.np_rng, self.drake_rng)
            else:
                sample = self.istc.sample_bounding_box(self.np_rng)
            t_min = x_start.time + self.travel_time(sample, x_start.pos)
            if self.np_rng.random() < B.batch_probability:
                t_lb, t_ub = t_min, self.max_valid_time(sample, B.goals)
            else:
                t_lb = max(t_min, self.max_valid_time(sample, B.goals))
                t_ub = self.max_valid_time(sample, B.new_goals)
            
            if time.perf_counter() - ts > P.max_runtime_in_secs:
                return

        t = self.np_rng.uniform(t_lb, t_ub)
        return State(sample, t)
            
    def max_valid_time(self, pos:np.ndarray, goals:Set[TreeNode]) -> float:
        ret = -np.inf
        for goal in goals:
            t = self.travel_time(pos, goal.state.pos)
            if goal.state.time - t > ret:
                ret = goal.state.time - t
        return ret

    def prune_start_tree(self, t_max:float, T_start:Tree, B:BoundVariables) -> None:
        # prune start tree (delete nodes that can't reach any goal within t_max)
        unchecked, num_pruned = set(T_start._nodes), 0
        while unchecked != set():
            node = unchecked.pop()
            time_to_nearest_goal = np.inf
            for goal_node in B.goals:
                if node.state.time < goal_node.state.time:
                    dt = time_to_reach(node.state, goal_node.state, self.vlimit, is_start_tree=True)
                    if node.state.time + dt <= t_max and dt < time_to_nearest_goal:
                        time_to_nearest_goal = dt

            # recursively remove node and its children if it can't reach any goal within t_max
            if time_to_nearest_goal == np.inf:
                pruned = T_start.remove_recursively(node)
                unchecked = unchecked - set(pruned)
                num_pruned += len(pruned)
        
        # print(f"Pruned {num_pruned} nodes from the start tree")

    def prune_goal_tree(
        self, t_max:float, T_start:Tree, T_goal:Tree, lambda_nn:float, B:BoundVariables, robot_radius:float
    ) -> List[State]:
        num_pruned, invalid_goals = 0, set()
        rewiring_list: List[TreeNode] = []
        
        # check invalid goals with time > t_max
        for goal_node in B.goals:
            if goal_node.state.time > t_max:
                num_pruned += 1
                rewiring_list.extend(T_goal._children[goal_node])
                T_goal._nodes.remove(goal_node)
                invalid_goals.add(goal_node)
                if goal_node in T_goal._children:
                    del T_goal._children[goal_node]
        
        # remove invalid goals from the set of goals
        B.goals = B.goals - invalid_goals

        T_goal.debug_check()
        
        # try to rewire descendants to a valid goal, record the best solution if connectable to T_start
        num_rewired, sol_opt = 0, (np.inf, [])
        while rewiring_list:
            node = rewiring_list.pop(0)
            rewired = False
            for tentative_goal_node in sorted(B.goals, key=lambda _n:_n.state.time):
                x, xg = node.state, tentative_goal_node.state
                dt = time_to_reach(xg, x, self.vlimit, is_start_tree=False)
                if dt + xg.time <= t_max and not self.istc.collision_checking_seg(x.pos, xg.pos, x.time, xg.time, robot_radius):
                    num_rewired += 1
                    node.parent = tentative_goal_node
                    T_goal._children[tentative_goal_node].append(node)
                    path = T_goal.check_solution(node, T_start, self.vlimit, self.istc, lambda_nn, robot_radius)
                    if path != [] and path[-1].time < sol_opt[0]:
                        sol_opt = (path[-1].time, path)
                    rewired = True
                    break

            if not rewired:
                num_pruned += 1
                rewiring_list.extend(T_goal._children[node])
                T_goal._nodes.remove(node)
                if node in T_goal._children:
                    del T_goal._children[node]
        
        # print(f"Pruned {num_pruned} nodes and rewired {num_rewired} from the goal tree")
        return sol_opt[1]

    def update_solution(
        self, sol_opt:List[State]|None, t_max:float, sol:List[State], T_start:Tree, T_goal:Tree, B:BoundVariables, P:Options, robot_radius:float
    ) -> Tuple[float, List[State]]: 
        if sol_opt is None or sol[-1].time < sol_opt[-1].time:
            # print(f"Found a better solution with arrival time {sol[-1].time}")
            sol_opt, t_max, B.batch_probability = sol, sol[-1].time, 1.0
            self.prune_start_tree(t_max, T_start, B)
            sol_rewired = self.prune_goal_tree(t_max, T_start, T_goal, P.lambda_nn, B, robot_radius)
            if sol_rewired != []:
                t_max, sol_opt = self.update_solution(sol_opt, t_max, sol_rewired, T_start, T_goal, B, P, robot_radius)
        
        return t_max, sol_opt

    def travel_time(self, x:np.ndarray, y:np.ndarray) -> float:
        # return the max time among all dimensions
        return max([abs(x[i] - y[i]) / self.vlimit for i in range(self.istc.dim)])
    