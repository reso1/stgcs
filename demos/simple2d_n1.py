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

    istc = ex.SIMPLE2D_4DynamicSPHERE

    seed = 0
    tf = 10
    time_scaler = 1/tf
    start, goal = np.array([0.0, 0.0]), np.array([4.0, 4.0])
    t_start = 0
    vlimit = 1.0
    robot_radius = 0.1
    sets = [make_hpolytope(V) for V in istc.C_Space]

    # ST-GCS
    ts = time.perf_counter()
    stgcs = STGCS.from_space_sets(sets=sets, vlimit=vlimit, t0=0.0, tmax=tf)
    
    for obs in istc.O_Dynamic:
        stgcs = obs.reserve(stgcs, robot_radius)
    
    sol_stgcs = stgcs.solve(
        start, goal, t_start=t_start, 
        relaxation=True, max_rounded_paths=4000, max_rounding_trials=4000)
    
    print("STGCS solution time", time.perf_counter() - ts)
    print("solution cost", sol_stgcs.cost, "arrival time", sol_stgcs.itvl.end)

    # ST-RRT*
    ts = time.perf_counter()
    strrtstar = STRRTStar(istc, seed=seed, vlimit=vlimit)
    P = Options.default()
    P.max_iterations = int(1e9)
    P.max_runtime_in_secs = 30
    sol_strrtstar = strrtstar.solve(start, goal, t0=t_start, t_max=tf, P=P, robot_radius=robot_radius)
    print("ST-RRT* solution time:", time.perf_counter() - ts)
    print("trajectory cost:", sol_strrtstar.cost, "arrival time:", sol_strrtstar.itvl.end)

    T = np.linspace(0, max(sol_strrtstar.itvl.end, sol_stgcs.itvl.end), 100)
    Pi = []
    for i, sol in enumerate([sol_strrtstar, sol_stgcs]):
        pi = [sol.lerp(t) for t in T]
        Pi.append(pi)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    for obs in istc.O_Static:
        obs.draw_with_time(ax, tmax=8.5, alpha=0.3)
        
    for obs in istc.O_Dynamic:
        pi = [np.hstack([obs.x(t), t]) for t in T[::5]]
        ax.plot([p[0] for p in pi], [p[1] for p in pi], [p[2] for p in pi], '-ok', alpha=0.3, ms=obs.radius * 120)
    
    plt.plot([4, 4], [4, 4], [0, 9], '--k')
    ax.plot([0, 4, 4, 0, 0], [0, 0, 4, 4, 0], [0, 0, 0, 0, 0], '--k')
    
    colors = ['b', 'r']
    names = [r'ST-RRT$^*$', r'ST-GCS']
    for i, pi in enumerate(Pi):
        pi = pi[::2]
        X, Y, Z = zip(*pi)
        ax.plot(X, Y, Z, f'-o{colors[i]}', markersize=4, label=names[i])
        ax.plot([pi[0][0]], [pi[0][1]], [pi[0][2]], f's{colors[i]}', markersize=6, alpha=0.5)
        ax.plot([pi[-1][0]], [pi[-1][1]], [pi[-1][2]], f'*{colors[i]}', markersize=10, alpha=1.0)
    
    ax.legend(ncol=1, frameon=True, fontsize=15, columnspacing=0.5)
    ax.plot([4], [4], [0], 'Xk', markersize=8)
    ax.text(0.02, 0, 0.2, r"start", fontdict={"fontsize": 15})
    ax.text(4.03, 4, -0.1, r"goal", fontdict={"fontsize": 15})
    ax.axis('off')
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_box_aspect(aspect=(1,1,0.75))
    ax.view_init(elev=77, azim=-88, roll=2)
    # fig.savefig("dynamic_obstacles.png", dpi=1000, transparent=True)

    fig, ax = plt.subplots()
    istc.animate_2d(ax, [sol_strrtstar, sol_stgcs], save_anim=False)
    plt.show()
