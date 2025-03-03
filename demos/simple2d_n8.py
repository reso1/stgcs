import numpy as np
import matplotlib.pyplot as plt

import time
import os, sys, pickle
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from mrmp.pbs import PBS
    from baselines.sp_strrtstar import sequential_planning as SP_STRRTStar
    from environment import examples as ex
except:
    raise ImportError("You should run this script from the root directory")


if __name__ == "__main__":
    istc = ex.SIMPLE2D
    seed = 0
    vlimit = 1
    tmax = 20
    robot_radius = 0.1

    starts = [np.array([1.2, 0.5]), np.array([0.5, 1.2]), np.array([2.7, 3.5]), np.array([3.5, 2.7]), 
              np.array([0.5, 2.6]), np.array([1.3, 3.5]), np.array([3.5, 1.5]), np.array([2.6, 0.5])]
    goals  = [np.array([2.7, 3.5]), np.array([3.5, 2.7]), np.array([1.2, 0.5]), np.array([0.5, 1.2]), 
              np.array([3.5, 1.5]), np.array([2.6, 0.5]), np.array([0.5, 2.6]), np.array([1.3, 3.5])]
    T0s = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    
    sol_pbs_stgcs, _ = PBS(istc, tmax, vlimit, robot_radius, starts, goals, T0s, 150)
    sol_sp_strrtstar = SP_STRRTStar(istc, tmax=tmax, vlimit=vlimit, safe_radius=robot_radius, starts=starts, goals=goals, t0s=T0s, seed=seed, timeout_secs=150, use_CSpace=True)
    
    tmax = max(max([s.itvl.end for s in sol_pbs_stgcs]),
               max([s.itvl.end for s in sol_sp_strrtstar]))
    T = np.linspace(0, tmax, 100)
    Pi_pbs_stgcs, Pi_sp_strrtstar = [], []
    for i, s in enumerate(sol_pbs_stgcs):
        pi = [s.lerp(t) for t in T]
        Pi_pbs_stgcs.append(pi)
    for i, s in enumerate(sol_sp_strrtstar):
        pi = [s.lerp(t) for t in T]
        Pi_sp_strrtstar.append(pi)


    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.cm.get_cmap('tab10')
    for obs in istc.O_Static:
        obs.draw_with_time(ax, tmax=0.01, alpha=0.3)
    
    SoC, makespan = 0, 0
    for i, pi in enumerate(Pi_pbs_stgcs):
        pi = np.array(pi)
        cost = pi[-1,-1]
        SoC += cost
        makespan = max(makespan, cost)
        ax.plot(pi[:, 0], pi[:, 1], pi[:, 2], 'o-', ms=3, color=cmap(i))
        ax.plot(pi[0][0], pi[0][1], pi[0][2], 's', ms=6.5, color=cmap(i))
        ax.plot([pi[0][0], pi[0][0]], [pi[0][1], pi[0][1]], [0, pi[0][2]], '--k', alpha=0.5)
        ax.plot(pi[-1][0], pi[-1][1], pi[-1][2], '*', ms=10, color=cmap(i))
        ax.plot([pi[-1][0], pi[-1][0]], [pi[-1][1], pi[-1][1]], [0, pi[-1][2]], '--k', alpha=0.5)
    
    texts = [r'$s_0=g_1$', r'$s_1=g_0$', r'$s_2=g_3$', r'$s_3=g_2$', r'$s_4=g_5$', r'$s_5=g_4$', r'$s_6=g_7$', r'$s_7=g_6$']
    for i, x in enumerate(starts):
        ax.plot(x[0], x[1], 0, 'x', ms=8, mec='k')
        ax.text(x[0]+0.1, x[1]-0.15, 0, texts[i], fontsize=12, ha='center', va='top')

    ax.plot([0, 4, 4, 0, 0], [0, 0, 4, 4, 0], [0, 0, 0, 0, 0], '--k')
    ax.set_box_aspect([1, 1, 0.6])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    ax.set_zticks([0, 3, 6, 9])
    ax.axis('off')
    ax.view_init(elev=25, azim=-50)
    ax.set_title(f'PBS+ST-GCS (SoC:{SoC}, Makespan:{makespan})')
    # fig.savefig('multirobot_pbs.png', dpi=1000, transparent=True)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.cm.get_cmap('tab10')
    for obs in istc.O_Static:
        obs.draw_with_time(ax, tmax=0.01, alpha=0.3)
    
    SoC, makespan = 0, 0
    for i, pi in enumerate(Pi_sp_strrtstar):
        pi = np.array(pi)
        cost = pi[-1,-1]
        SoC += cost
        makespan = max(makespan, cost)
        ax.plot(pi[:, 0], pi[:, 1], pi[:, 2], 'o-', ms=3, color=cmap(i), alpha=0.5)
        ax.plot(pi[0][0], pi[0][1], pi[0][2], 's', ms=6.5, color=cmap(i))
        ax.plot([pi[0][0], pi[0][0]], [pi[0][1], pi[0][1]], [0, pi[0][2]], '--k', alpha=0.5)
        ax.plot(pi[-1][0], pi[-1][1], pi[-1][2], '*', ms=10, color=cmap(i))
        ax.plot([pi[-1][0], pi[-1][0]], [pi[-1][1], pi[-1][1]], [0, pi[-1][2]], '--k', alpha=0.5)

    for i, x in enumerate(starts):
        ax.plot(x[0], x[1], 0, 'x', ms=8, mec='k')
        ax.text(x[0]+0.1, x[1]-0.15, 0, texts[i], fontsize=12, ha='center', va='top')
    
    ax.plot([0, 4, 4, 0, 0], [0, 0, 4, 4, 0], [0, 0, 0, 0, 0], '--k')
    ax.set_box_aspect([1, 1, 0.6])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    ax.set_zticks([0, 3, 6, 9])
    ax.axis('off')
    ax.view_init(elev=25, azim=-50)
    ax.set_title(f'ST-GCS+STRRT* (SoC:{SoC}, Makespan:{makespan})')
    # fig.savefig('multirobot_strrtstar.png', dpi=1000, transparent=True)
    plt.show()

