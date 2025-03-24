import numpy as np
import matplotlib.pyplot as plt

import time
import os, sys, pickle
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from mrmp.pbs import PBS
    from baselines.rp_stgcs import sequential_planning as SP_STGCS, randomized_prioritized_planning as RP_STGCS
    from mrmp.utils import make_hpolytope
    from environment import examples as ex
    from environment.problems import random_objectives
except:
    raise ImportError("You should run this script from the root directory")


if __name__ == "__main__":
    env = ex.COMPLEX2D
    vlimit = 1
    tf = 50

    starts = [
        np.array([0.0, 0.5]), np.array([0.2, 3.0]), np.array([0.2, 2.5]), np.array([1.2, 0.5]),
        np.array([3.5, 0.2]), np.array([0.1, 4.9]), np.array([1.5, 4.8]), np.array([4.9, 0.1]),
        np.array([4.8, 2.2]), np.array([4.0, 2.8]), np.array([4.9, 4.9]), np.array([4.8, 4.0]),
        np.array([3.0, 4.7]), np.array([2.0, 2.5]), np.array([2.35, 4.0]), np.array([3.5, 4.0]),
        np.array([2.0, 0.1]), np.array([1.1, 4.0]), np.array([2.0, 1.2]), np.array([3.8, 4.8]), 
    ]
    goals = [
        np.array([0.3, 2.0]), np.array([0.0, 4.5]), np.array([1.5, 2.5]), np.array([1.5, 1.2]),
        np.array([0.1, 1.5]), np.array([1.1, 4.0]), np.array([2.0, 2.5]), np.array([2.0, 1.0]),
        np.array([3.0, 4.8]), np.array([2.5, 4.9]), np.array([3.0, 0.5]), np.array([4.5, 0.5]),
        np.array([2.5, 0.2]), np.array([0.1, 0.1]), np.array([4.9, 4.8]), np.array([0.2, 4.9]),
        np.array([4.9, 2.5]), np.array([0.1, 2.5]), np.array([4.0, 2.5]), np.array([1.5, 2.0]), 
    ]

    T0s = np.random.uniform(0, 1, len(starts))
    sets = [make_hpolytope(V) for V in env.C_Space]
    
    ts = time.perf_counter()
    sol_pbs_stgcs, _ = PBS(env, tf, vlimit, starts, goals, T0s, 150, scaler_multiplier=3.0)
    print("STGCS solution time", time.perf_counter() - ts)
    print("SoC", sum([p.cost for p in sol_pbs_stgcs]), "makespan", max([p.itvl.end for p in sol_pbs_stgcs]))

    fig, ax = plt.subplots()
    env.animate_2d(ax, sol_pbs_stgcs, draw_CSpace=True, save_anim=True)
    # plt.show()

