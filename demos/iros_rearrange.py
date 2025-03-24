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
    from environment.env import Env, _animate_func_2d
except:
    raise ImportError("You should run this script from the root directory")


I = [np.array([0.5, 0.0]), np.array([0.5, 1/7]), np.array([0.5, 2/7]), np.array([0.5, 3/7]), 
     np.array([0.5, 4/7]), np.array([0.5, 5/7]), np.array([0.5, 6/7]), np.array([0.5, 1.0])]

R = [np.array([0.3, 0.1]), np.array([0.77, 0.1]), np.array([0.3, 0.5]), np.array([0.55, 0.5]), 
     np.array([0.3, 0.7]), np.array([0.68, 0.7]), np.array([0.3, 0.9]), np.array([0.55, 0.9])]

O = [np.array([0.5, 0.23]), np.array([0.3, 0.3]),np.array([0.7, 0.3]),  np.array([0.77, 0.5]), 
     np.array([0.23, 0.5]), np.array([0.3, 0.7]), np.array([0.7, 0.7]), np.array([0.5, 0.77])]

S = [np.array([0.5, 0.2]), np.array([0.38, 0.25]), np.array([0.63, 0.3]), np.array([0.55, 0.45]),
     np.array([0.45, 0.55]), np.array([0.37, 0.7]), np.array([0.62, 0.75]), np.array([0.5, 0.8])]

T0s =  np.zeros(8)
env = ex.EMPTY2D
env.robot_radius = 0.05

if __name__ == "__main__":
    vlimit = 0.2
    tf = 50
    robot_radius = 0.1

    t0 = 0
    Pi_total = [list() for _ in range(8)]
    for starts, goals in [(I, R), (R, O), (O, S)]:
        sets = [make_hpolytope(V) for V in env.C_Space]
        ts = time.perf_counter()
        sol, _ = PBS(env, tf, vlimit, starts, goals, T0s, 150, scaler_multiplier=5)
        
        T = np.linspace(0, max([p.cost for p in sol]), 100)
        Pi = []
        for p in sol:
            Pi.append(np.array([p.lerp(t) for t in T]))
        
        tmax = 0
        for i, pi in enumerate(Pi):
            Pi_total[i].extend([np.hstack([x[:2], x[2]+t0]) for x in pi])
            tmax = max(tmax, pi[-1][2])
        
        t0 += tmax
        
    Pi = [np.array(pi) for pi in Pi_total]

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    
    dt = 0.02
    anim = _animate_func_2d(ax, env.robot_radius, env.lb, env.ub, Pi, dt=dt)
    anim.save(f"rearange.mp4", writer='ffmpeg', fps=1/dt, dpi=1000)
    plt.show()

