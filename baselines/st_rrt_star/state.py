from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# d(x1, x2) = lambda * |x2 - x1| + (1 - lambda) * |t2 - t1|

@dataclass
class State:
    pos: np.ndarray
    time: float

    def __hash__(self) -> int:
        return hash( tuple(self.pos) + (self.time,) )

    def __eq__(self, other:State) -> bool:
        return np.all(self.pos == other.pos) and self.time == other.time


def lerp(x1:State, x2:State, k:float) -> State:
    k = np.clip(k, 0, 1)
    return State(
        pos  = x1.pos * (1 - k) + x2.pos * k,
        time = x1.time * (1 - k) + x2.time * k
    )


def cost_func(x1:State, x2:State, v_limit:np.ndarray, p_lambda:float, is_start_tree:bool) -> float:
    # d(x1, x2) = lambda * |x2 - x1| + (1 - lambda) * |t2 - t1|
    if (is_start_tree and x1.time < x2.time) or (not is_start_tree and x1.time > x2.time):
        dt = abs(x2.time - x1.time)
        vel = np.abs(x2.pos - x1.pos) / dt
        if np.all(vel <= v_limit):
            return p_lambda * np.linalg.norm(x2.pos - x1.pos) + (1 - p_lambda) * dt
    
    return np.inf


def distance(x1:State, x2:State, v_limit:np.ndarray, is_start_tree:bool) -> float:
    return cost_func(x1, x2, v_limit, p_lambda=1.0, is_start_tree=is_start_tree)


def time_to_reach(x1:State, x2:State, v_limit:np.ndarray, is_start_tree:bool) -> float:
    return cost_func(x1, x2, v_limit, p_lambda=0.0, is_start_tree=is_start_tree)
