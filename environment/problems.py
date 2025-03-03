from typing import List, Tuple, Dict
from itertools import product
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

import os, sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


from pydrake.all import RandomGenerator

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from environment.instance import Instance
    from environment import examples as ex
    from environment.obstacle import DynamicSphere
    from mrmp.interval import Interval

except:
    raise ImportError("You should run this script from the root directory")


@dataclass
class MRMP:
    istc: Instance
    tmax: float
    vlimit: float
    seed: int
    robot_radius: float
    starts: List[np.ndarray]
    goals: List[np.ndarray]
    T0s: List[float]


def random_objectives(istc:Instance, seed:int, num_agents:int, dist_tol:float=1.0, sep_tol:float=0.5) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    rng = np.random.RandomState(seed)
    drake_rng = RandomGenerator(seed)

    ret = []
    while len(ret) < num_agents:
        start = istc.sample_CSpace(rng, drake_rng)
        goal = istc.sample_CSpace(rng, drake_rng)
        t0 = rng.uniform(0, 2)
        valid = np.linalg.norm(start - goal) > dist_tol
        if valid:
            for s, g, _ in ret:
                if np.linalg.norm(start - s) < sep_tol or np.linalg.norm(goal - g) < sep_tol:
                    valid = False
                    break
        if valid:
            ret.append((start, goal, t0))

    return ret


def empty2d_problem_set(istc:Instance, tmax:float=50, vlimit:float=0.5, robot_radius:float=0.01) -> Dict[int, List[MRMP]]:
    assert istc.name == "empty2d"
    problems = defaultdict(list)
    Lb, Ub = np.zeros(2), np.array([1/2, 1/2])
    for num_agents, seed in tqdm(product(range(1, 11), range(12)), total=10*12):
        istc_new = istc.copy()
        starts, goals, T0s = zip(*random_objectives(istc_new, seed, num_agents, 0.3, 0.15))
        starts, goals, T0s = list(starts), list(goals), list(T0s)
        mrmp = MRMP(istc_new, tmax, vlimit, seed, robot_radius, starts, goals, T0s)
        problems[num_agents].append(mrmp)
    
    return problems


def simple2d_problem_set(istc:Instance, tmax:float=50, vlimit:float=1.0, robot_radius:float=0.1) -> Dict[int, List[MRMP]]:
    
    def _sample(kdtree, idx:int, drake_rng):
        pts = istc._CSpace_hpoly[idx].UniformSample(drake_rng)
        dist, _ = kdtree.query(pts)
        while dist <= 0.5:
            pts = istc._CSpace_hpoly[idx].UniformSample(drake_rng)
            dist, _ = kdtree.query(pts)
        return pts
    
    assert istc.name == "simple2d"
    problems = defaultdict(list)
    for num_agents, seed in tqdm(product(range(1, 11), range(12)), total=10*12):
        istc_new = istc.copy()
        starts, goals, T0s = zip(*random_objectives(istc_new, seed, num_agents))
        starts, goals, T0s = list(starts), list(goals), list(T0s)
        kdtree = KDTree(starts + goals)
        rng = np.random.RandomState(seed)
        drake_rng = RandomGenerator(seed)
        for i in range(4):
            itvl = Interval(rng.uniform(0, 1), rng.uniform(3, 5))
            radius = rng.uniform(0.05, 0.2)
            start = _sample(kdtree, i, drake_rng)
            goal = _sample(kdtree, i, drake_rng)
            istc_new.O_Dynamic.append(DynamicSphere(start, goal, radius, itvl))

        mrmp = MRMP(istc_new, tmax, vlimit, seed, robot_radius, starts, goals, T0s)
        problems[num_agents].append(mrmp)
    
    return problems


def complex2d_problem_set(istc:Instance, tmax:float=50, vlimit:float=1.0, robot_radius:float=0.1) -> Dict[int, List[MRMP]]:
    assert istc.name == "complex2d"
    problems = defaultdict(list)
    for num_agents, seed in tqdm(product(range(1, 11), range(12)), total=10*12):
        istc_new = istc.copy()
        starts, goals, T0s = zip(*random_objectives(istc_new, seed, num_agents))
        starts, goals, T0s = list(starts), list(goals), list(T0s)

        mrmp = MRMP(istc_new, tmax, vlimit, seed, robot_radius, starts, goals, T0s)
        problems[num_agents].append(mrmp)
    
    return problems


if __name__ == "__main__":
    # empty 2D
    ps = empty2d_problem_set(istc=ex.EMPTY2D)
    pickle.dump(ps, open(f"{ex.EMPTY2D.name}.ps", "wb"))

    # simple 2D
    ps = simple2d_problem_set(istc=ex.SIMPLE2D)
    pickle.dump(ps, open(f"{ex.SIMPLE2D.name}.ps", "wb"))

    # # complex 2D
    ps = complex2d_problem_set(istc=ex.COMPLEX2D)
    pickle.dump(ps, open(f"{ex.COMPLEX2D.name}.ps", "wb"))
