import numpy as np
import matplotlib.pyplot as plt

import time
import os, sys, pickle
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from mrmp.pbs import PBS
    from baselines.rp_stgcs import randomized_prioritized_planning as PP_STGCS, sequential_planning as SP_STGCS
    from baselines.sp_strrtstar import sequential_planning as SP_STRRTSTAR
    from baselines.sp_tprm import sequential_planning as SP_TPRM
    from mrmp.stgcs import STGCS
    from mrmp.utils import make_hpolytope
    from environment import examples as ex
except:
    raise ImportError("You should run this script from the root directory")


if __name__ == "__main__":
    env = ex.EMPTY2D_4DynamicSPHERE
    seed = 0
    vlimit = 0.5
    tf = 50

    starts = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
    goals = [np.array([1.0, 1.0]), np.array([0.0, 0.0])]
    T0s =  [0.0, 0.0, 0.0, 0.0]
    sets = [make_hpolytope(V) for V in env.C_Space]
    
    ts = time.perf_counter()
    # sol, _ = PBS(env, tf, vlimit, starts, goals, T0s, 150, scaler_multiplier=3)
    sol = SP_STRRTSTAR(env, tf, vlimit, starts, goals, T0s, seed=seed, timeout_secs=150, use_CSpace=True)
    print("solution time", time.perf_counter() - ts)
    print("SoC", sum([p.cost for p in sol]), "makespan", max([p.itvl.end for p in sol]))

    fig, ax = plt.subplots()
    env.animate_2d(ax, sol, draw_CSpace=True, save_anim=True)
    plt.show()

