import matplotlib.pyplot as plt
import numpy as np
import time

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from baselines.tprm.planner import TemporalPRM
    from baselines.st_rrt_star.planner import STRRTStar, Options
    from mrmp.stgcs import STGCS
    from mrmp.utils import make_hpolytope
    from environment import examples as ex
except:
    raise ImportError("You should run this script from the root directory")


if __name__ == "__main__":
    env = ex.SIMPLE2D_4DynamicSPHERE

    seed = 0
    tf = 20
    time_scaler = 1/tf
    start, goal = np.array([0.0, 0.0]), np.array([4.0, 4.0])
    t_start = 0
    vlimit = 1.0
    sets = [make_hpolytope(V) for V in env.C_Space]

    # ST-GCS
    ts = time.perf_counter()
    stgcs = STGCS.from_env(env, vlimit=vlimit, t0=0.0, tmax=tf)
    
    sol_stgcs = stgcs.solve(
        start, goal, t_start=t_start, 
        relaxation=True, max_rounded_paths=5000, max_rounding_trials=5000)
    
    print("STGCS solution time", time.perf_counter() - ts)
    print("solution cost", sol_stgcs.cost, "arrival time", sol_stgcs.itvl.end)

    # ST-RRT*
    ts = time.perf_counter()
    strrtstar = STRRTStar(env, seed=seed, vlimit=vlimit)
    P = Options.default()
    P.max_iterations = int(1e9)
    P.max_runtime_in_secs = 150
    sol_strrtstar = strrtstar.solve(start, goal, t0=t_start, t_max=tf, P=P)
    print("ST-RRT* solution time:", time.perf_counter() - ts)
    print("trajectory cost:", sol_strrtstar.cost, "arrival time:", sol_strrtstar.itvl.end)

    fig, ax = plt.subplots()
    env.animate_2d(ax, [sol_strrtstar, sol_stgcs], draw_CSpace=True, save_anim=True)
    plt.show()
