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
    istc = ex.SIMPLE2D_8DynamicSPHERE
    seed = 0
    vlimit = 1
    tmax = 10
    robot_radius = 0.1

    starts = [np.array([0.0, 0.0]), np.array([4.0, 4.0]), np.array([0.0, 4.0]), np.array([4.0, 0.0])]
    goals  = [np.array([4.0, 4.0]), np.array([0.0, 0.0]), np.array([4.0, 0.0]), np.array([0.0, 4.0])]
    T0s = [0.0, 0.0, 0.0, 0.0]
    
    sol_pbs_stgcs, _ = PBS(istc, tmax, vlimit, robot_radius, starts, goals, T0s, 600, scaler_multiplier=5)
    # sol_sp_strrtstar = SP_STRRTStar(istc, tmax=tmax, vlimit=vlimit, safe_radius=robot_radius, starts=starts, goals=goals, t0s=T0s, seed=seed, timeout_secs=150, use_CSpace=True)

    fig, ax = plt.subplots()
    istc.animate_2d(ax, sol_pbs_stgcs, save_anim=False)
    plt.show()

