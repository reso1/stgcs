import numpy as np
import matplotlib.pyplot as plt

import time
import os, sys, pickle
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from mrmp.pbs import PBS
    from baselines.sp_strrtstar import sequential_planning as SP_STRRTStar
    from baselines.sp_tprm import sequential_planning as SP_TPRM
    from environment import examples as ex
except:
    raise ImportError("You should run this script from the root directory")


if __name__ == "__main__":
    env = ex.SIMPLE2D_8DynamicSPHERE
    seed = 0
    vlimit = 1
    tmax = 20

    starts = [np.array([0.0, 0.0]), np.array([4.0, 4.0]), np.array([0.0, 4.0]), np.array([4.0, 0.0])]
    goals  = [np.array([4.0, 4.0]), np.array([0.0, 0.0]), np.array([4.0, 0.0]), np.array([0.0, 4.0])]
    T0s = [0.0, 0.0, 0.0, 0.0]
    
    ts = time.perf_counter()
    # sol, _ = PBS(env, tmax, vlimit, starts, goals, T0s, 150, scaler_multiplier=5)
    sol = SP_STRRTStar(env, tmax=tmax, vlimit=vlimit, starts=starts, goals=goals, t0s=T0s, seed=seed, timeout_secs=150, use_CSpace=True)
    print("Solution time", time.perf_counter() - ts)
    print("SoC", sum([p.cost for p in sol]), "makespan", max([p.itvl.end for p in sol]))

    fig, ax = plt.subplots()
    env.animate_2d(ax, sol, draw_CSpace=True, save_anim=True)
    plt.show()

