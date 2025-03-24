import numpy as np
import matplotlib.pyplot as plt

import time
import os, sys, pickle
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from mrmp.pbs import PBS
    from baselines.rp_stgcs import sequential_planning as SP_STGCS  
    from baselines.sp_strrtstar import sequential_planning as SP_STRRTStar
    from environment import examples as ex
except:
    raise ImportError("You should run this script from the root directory")


if __name__ == "__main__":
    env = ex.SIMPLE2D
    seed = 0
    vlimit = 1
    tmax = 50

    starts = [np.array([1.2, 0.5]), 
              np.array([0.5, 1.2]), 
              np.array([2.7, 3.5]), 
              np.array([3.5, 2.7]), 
              np.array([0.5, 2.6]), 
              np.array([1.3, 3.5]), 
              np.array([3.5, 1.5]), 
              np.array([2.6, 0.5])
              ]
    goals  = [np.array([2.7, 3.5]), 
              np.array([3.5, 2.7]), 
              np.array([1.2, 0.5]), 
              np.array([0.5, 1.2]), 
              np.array([3.5, 1.5]), 
              np.array([2.6, 0.5]), 
              np.array([0.5, 2.6]), 
              np.array([1.3, 3.5])
              ]
    T0s = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    
    ts = time.perf_counter()
    sol, _ = PBS(env, tmax, vlimit, starts, goals, T0s, 150, scaler_multiplier=3.0)
    # sol = SP_STRRTStar(env, tmax=tmax, vlimit=vlimit, starts=starts, goals=goals, t0s=T0s, seed=seed, timeout_secs=150, use_CSpace=True)
    print("Solution time", time.perf_counter() - ts)
    print("SoC", sum([p.cost for p in sol]), "makespan", max([p.itvl.end for p in sol]))
    
    fig, ax = plt.subplots()
    env.animate_2d(ax, sol, draw_CSpace=True, save_anim=True)
    plt.show()


