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
    seed = 0
    vlimit = 1
    tf = 30
    num_agents = 12

    starts, goals, T0s = zip(*random_objectives(env, seed, num_agents))
    sets = [make_hpolytope(V) for V in env.C_Space]
    
    ts = time.perf_counter()
    sol_pbs_stgcs, _ = PBS(env, tf, vlimit, starts, goals, T0s, 150, scaler_multiplier=3)
    print("STGCS solution time", time.perf_counter() - ts)
    print("SoC", sum([p.cost for p in sol_pbs_stgcs]), "makespan", max([p.itvl.end for p in sol_pbs_stgcs]))


    fig, ax = plt.subplots()
    env.animate_2d(ax, sol_pbs_stgcs, draw_CSpace=True, save_anim=True)
    # plt.show()

