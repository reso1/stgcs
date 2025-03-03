import numpy as np
import matplotlib.pyplot as plt

import time
import os, sys, pickle
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from mrmp.pbs import PBS
    from baselines.rp_stgcs import sequential_planning as SP_STGCS, randomized_prioritized_planning as RP_STGCS
    from baselines.sp_strrtstar import sequential_planning as SP_STRRTStar
    from mrmp.utils import make_hpolytope
    from environment import examples as ex
    from environment.problems import random_objectives
except:
    raise ImportError("You should run this script from the root directory")


if __name__ == "__main__":
    istc = ex.SIMPLE2D
    seed = 1
    vlimit = 1
    tf = 50
    num_agents = 16
    robot_radius = 0.1

    starts, goals, T0s = zip(*random_objectives(istc, seed, num_agents))
    sets = [make_hpolytope(V) for V in istc.C_Space]
    
    ts = time.perf_counter()
    sol_pbs_stgcs, _ = PBS(istc, tf, vlimit, robot_radius, starts, goals, T0s, 150, scaler_multiplier=2)
    # sol_pbs_stgcs = SP_STRRTStar(istc, tmax=tf, vlimit=vlimit, safe_radius=robot_radius, starts=starts, goals=goals, t0s=T0s, seed=seed, timeout_secs=150, use_CSpace=True)
    print("STGCS solution time", time.perf_counter() - ts)
    print("SoC", sum([p.cost for p in sol_pbs_stgcs]), "makespan", max([p.itvl.end for p in sol_pbs_stgcs]))

    fig, ax = plt.subplots()
    istc.animate_2d(ax, sol_pbs_stgcs, save_anim=True)
    # plt.show()

