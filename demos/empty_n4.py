import numpy as np
import matplotlib.pyplot as plt

import time
import os, sys, pickle
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from mrmp.pbs import PBS
    from baselines.rp_stgcs import sequential_planning as SP_STGCS, randomized_prioritized_planning as RP_STGCS, sequential_planning as SP_STGCS
    from baselines.sp_strrtstar import sequential_planning as SP_STRRTSTAR
    from baselines.sp_tprm import sequential_planning as SP_TPRM
    from mrmp.stgcs import STGCS
    from mrmp.utils import make_hpolytope
    from environment import examples as ex
except:
    raise ImportError("You should run this script from the root directory")


if __name__ == "__main__":
    istc = ex.EMPTY2D
    seed = 0
    vlimit = 0.2
    tf = 50
    robot_radius = 0.05

    starts = [np.array([0.5, 0.0]), np.array([0.5, 1.0]), np.array([0.0, 0.5]), np.array([1.0, 0.5])]
    goals = [np.array([0.5, 1.0]), np.array([0.5, 0.0]), np.array([1.0, 0.5]), np.array([0.0, 0.5])]
    T0s =  [0.0, 0.0, 0.0, 0.0]
    sets = [make_hpolytope(V) for V in istc.C_Space]
    
    ts = time.perf_counter()
    """
    STGCS solution time 5.366207082988694
    SoC 19.9999999947294 makespan 5.0
    """
    sol, _ = PBS(istc, tf, vlimit, robot_radius, starts, goals, T0s, 150, scaler_multiplier=5)
    """
    STGCS solution time 20.64687433396466
    SoC 20.124999667911936 makespan 5.124999963203342
    """
    # sol, _ = RP_STGCS(istc, tf, vlimit, robot_radius, starts, goals, T0s, seed=seed, max_ordering_trials=1000, timeout_secs=150, scaler_multiplier=5)
    """
    solution time 150.01767820795067
    SoC 20.000000003726115 makespan 5.000000002592516
    """
    # sol = SP_STRRTSTAR(istc, tf, vlimit, robot_radius, starts, goals, T0s, seed=seed, timeout_secs=150, use_CSpace=True)

    print("solution time", time.perf_counter() - ts)
    print("SoC", sum([p.cost for p in sol]), "makespan", max([p.itvl.end for p in sol]))

    fig, ax = plt.subplots()
    istc.animate_2d(ax, sol, save_anim=True)
    plt.show()

