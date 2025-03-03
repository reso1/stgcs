import numpy as np
import matplotlib.pyplot as plt

import time
import os, sys, pickle
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from mrmp.pbs import PBS
    from baselines.rp_stgcs import sequential_planning as SP_STGCS, randomized_prioritized_planning as RP_STGCS
    from baselines.tprm.planner import TemporalPRM
    from baselines.st_rrt_star.planner import STRRTStar, Options
    from mrmp.stgcs import STGCS
    from mrmp.utils import make_hpolytope
    from environment import examples as ex
except:
    raise ImportError("You should run this script from the root directory")


if __name__ == "__main__":
    istc = ex.Complex2D_4DynamicSPHERE
    seed = 0
    vlimit = 1
    tf = 50
    robot_radius = 0.1

    starts = [np.array([0.1, 0.5]), np.array([0.1, 4.9]), np.array([3.5, 0.2]), np.array([3.5, 4.9])]
    goals = [np.array([4.75, 4]), np.array([4.75, 2.2]), np.array([2.5, 4.9]), np.array([1.5, 0.5])]
    T0s =  [0.0, 0.0, 0.0, 0.0]
    sets = [make_hpolytope(V) for V in istc.C_Space]
    
    ts = time.perf_counter()
    # sol_pbs_stgcs, _ = PBS(istc, tf, vlimit, robot_radius, starts, goals, T0s, 150, scaler_multiplier=5)
    sol_pbs_stgcs, _ = RP_STGCS(istc, tf, vlimit, robot_radius, starts, goals, T0s, seed=seed, max_ordering_trials=1000, timeout_secs=150, scaler_multiplier=5)
    print("STGCS solution time", time.perf_counter() - ts)
    print("SoC", sum([p.cost for p in sol_pbs_stgcs]), "makespan", max([p.itvl.end for p in sol_pbs_stgcs]))


    fig, ax = plt.subplots()
    istc.animate_2d(ax, sol_pbs_stgcs, save_anim=True)
    plt.show()

