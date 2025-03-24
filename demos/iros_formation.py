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
    from environment.env import Env, _animate_func_2d
except:
    raise ImportError("You should run this script from the root directory")


I_b = [np.array([0.45, -0.05]), np.array([0.55, -0.05]), np.array([0.55, 1.05]), np.array([0.45, 1.05])]
I = [np.array([0.5, 0.0]), np.array([0.5, 1/7]), np.array([0.5, 2/7]), np.array([0.5, 3/7]), 
     np.array([0.5, 4/7]), np.array([0.5, 5/7]), np.array([0.5, 6/7]), np.array([0.5, 1.0])]

R_b = np.array([1, 0]) + [np.array([0.25, 0.05]), np.array([0.25, 0.95]),
                          np.array([0.6, 0.95]), np.array([0.73, 0.75]), np.array([0.73, 0.65]),
                          np.array([0.55, 0.42]), np.array([0.83, 0.1]), np.array([0.75, 0.05]),
                          np.array([0.42, 0.42]), np.array([0.35, 0.42]), np.array([0.35, 0.05])]

R_bb = np.array([1.48, 0.7]) + 0.12*np.array([[np.cos(rad), np.sin(rad)] for rad in np.linspace(0, 2*np.pi, 30)])

R = np.array([1.0, 0.0]) + [np.array([0.3, 0.1]), np.array([0.77, 0.1]), np.array([0.3, 0.5]), np.array([0.55, 0.5]), 
                            np.array([0.3, 0.7]), np.array([0.68, 0.7]), np.array([0.3, 0.9]), np.array([0.55, 0.9])]

O_b = np.array([2.5, 0.5]) + 0.35*np.array([[np.cos(rad), np.sin(rad)] for rad in np.linspace(0, 2*np.pi, 30)])
O_bb = np.array([2.5, 0.5]) + 0.2*np.array([[np.cos(rad), np.sin(rad)] for rad in np.linspace(0, 2*np.pi, 30)])
                            

O = np.array([2.0, 0.0]) + [np.array([0.5, 0.23]), np.array([0.3, 0.3]),np.array([0.7, 0.3]),  np.array([0.77, 0.5]), 
                            np.array([0.23, 0.5]), np.array([0.3, 0.7]), np.array([0.7, 0.7]), np.array([0.5, 0.77])]

S_b = np.array([3.0, 0.0]) + [np.array([0.5, 0.87]), np.array([0.3, 0.7]), np.array([0.45, 0.45]), np.array([0.56, 0.3]),
                              np.array([0.46, 0.25]), np.array([0.34, 0.325]), np.array([0.32, 0.23]), np.array([0.5, 0.13]),
                              np.array([0.69, 0.26]), np.array([0.69, 0.32]), np.array([0.61, 0.45]), np.array([0.43, 0.7]),
                              np.array([0.5, 0.74]), np.array([0.62, 0.68]), np.array([0.68, 0.75]), np.array([0.62, 0.82])]

S = np.array([3.0, 0.0]) + [np.array([0.5, 0.2]), np.array([0.38, 0.25]), np.array([0.63, 0.3]), np.array([0.55, 0.45]),
                            np.array([0.45, 0.55]), np.array([0.37, 0.7]), np.array([0.62, 0.75]), np.array([0.5, 0.8])]

T0s =  np.zeros(8)
env = Env(name="formation", robot_radius=0.05, CSpace = [np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 1.0], [0.0, 1.0]])])


if __name__ == "__main__":
    vlimit = 0.2
    tf = 50

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
    for x in [I_b, R_b, R_bb, O_b, O_bb, S_b]:
        for i in range(len(x)):
            if i == len(x) - 1:
                ax.plot([x[i][0], x[0][0]], [x[i][1], x[0][1]], '-k')
            else:
                ax.plot([x[i][0], x[i+1][0]], [x[i][1], x[i+1][1]], '-k')

    for arr in [I, R, O, S]:
        for x in arr:
            ax.plot(x[0], x[1], 'ok', markersize=8, mfc='none')

    dt = 0.02
    anim = _animate_func_2d(ax, env.robot_radius, env.lb, env.ub, Pi, dt=dt)
    anim.save(f"{env.name}.mp4", writer='ffmpeg', fps=1/dt, dpi=1000)
    plt.show()

