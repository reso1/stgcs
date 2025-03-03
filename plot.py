import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from copy import deepcopy

import environment.examples as ex


def plot_stats():
    methods = ["SP+TPRM-C", "SP+ST-RRTStar-C", "SP+STGCS", "PP+STGCS", "PBS+STGCS"]
    colors = {"PBS+STGCS": "k", "PP+STGCS": "b", "SP+STGCS": "g", "SP+TPRM-C": "c", "SP+ST-RRTStar-C": "r"}
    lines = {"PBS+STGCS": "-", "PP+STGCS": "--", "SP+STGCS": ":", "SP+TPRM-C": "--", "SP+ST-RRTStar-C": "-"}
    markers = {"PBS+STGCS": "s", "PP+STGCS": "X", "SP+STGCS": "+", "SP+TPRM-C": "^", "SP+ST-RRTStar-C": "o"}
    labels = {"PBS+STGCS": "PBS+ST-GCS", "PP+STGCS": "RP+ST-GCS", "SP+STGCS":"SP+ST-GCS", "SP+TPRM-C": r"SP+T-PRM", "SP+ST-RRTStar-C": r"SP+ST-RRT$^*$"}

    num_cols, nrows = 5, 3
    widths = [1.8] * num_cols
    heights = [2.0] * nrows
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    legend_fontsize = 10
    axis_fontsize = 9
    title_fontsize = 10
    fig, axes = plt.subplots(ncols=num_cols, nrows=nrows, constrained_layout=True, gridspec_kw=gs_kw, figsize=(2.0*num_cols, sum(heights)))

    def subplot(istc_name:str, row:int):
        # get success rates and successful seeds for each method
        data, valid_seeds = {}, {}
        for method in methods:
            raw_data = pickle.load(open(f"data/{istc_name}-{method}.pkl", "rb"))
            n_robots_list = list(raw_data.keys())
            valid_seeds[method] = {}
            success_rate = {}
            for n_robots, stats in raw_data.items():
                n_succeeds = 0
                valid_seeds[method][n_robots] = []
                for seed, val in enumerate(stats):
                    succ = val[0] != []
                    n_succeeds += int(succ)
                    if succ:
                        valid_seeds[method][n_robots].append(seed)
                    
                success_rate[n_robots] = n_succeeds / len(stats)

            data[method] = {"success_rate": success_rate, "soc": {}, "makespan": {}, "runtime": {}}

        # remove methods that have no successful seeds for any n_robots
        valid_methods = {}
        for n_robots in n_robots_list:
            valid_methods[n_robots] = deepcopy(methods)
            for method in methods:
                if data[method]["success_rate"][n_robots] < 1/4:
                    valid_methods[n_robots].remove(method)
        
        # get the stats for methods have suceeeds for at least one seed
        SC, MS, RT = {}, {}, {}
        for n_robots in n_robots_list:
            seeds = set(valid_seeds[valid_methods[n_robots][0]][n_robots])
            for method in valid_methods[n_robots]:
                seeds = seeds.intersection(valid_seeds[method][n_robots])
            
            if seeds == set():
                continue

            soc, makespan, runtime = {}, {}, {}
            for method in valid_methods[n_robots]:
                raw_data = pickle.load(open(f"data/{istc_name}-{method}.pkl", "rb"))
                soc_total, makespan_total, runtime_total = [], [], []
                for seed in seeds:
                    sum_cost, max_cost = 0, 0
                    for pi in raw_data[n_robots][seed][0]:
                        cost = pi[-1][-1] - pi[0][2]
                        sum_cost += cost
                        max_cost = max(max_cost, cost)
                    
                    soc_total.append(sum_cost)
                    makespan_total.append(max_cost)
                    runtime_total.append(raw_data[n_robots][seed][1])
                
                soc[method] = [np.mean(soc_total), np.min(soc_total), np.max(soc_total)]
                makespan[method] = [np.mean(makespan_total), np.min(makespan_total), np.max(makespan_total)]
                runtime[method] = [np.mean(runtime_total), np.min(runtime_total), np.max(runtime_total)]
            
            SC[n_robots] = soc
            MS[n_robots] = makespan
            RT[n_robots] = runtime
        
        for method in methods:
            soc, makespan, runtime = {}, {}, {}
            for n_robots in n_robots_list:
                if n_robots not in SC or method not in SC[n_robots]:
                    continue
                soc[n_robots] = SC[n_robots][method]
                makespan[n_robots] = MS[n_robots][method]
                runtime[n_robots] = RT[n_robots][method]
            data[method]["soc"] = soc
            data[method]["makespan"] = makespan
            data[method]["runtime"] = runtime

        # c1: map
        ax = axes[row, 0]
        if row == 0:
            istc, name = ex.EMPTY2D, "empty"
        elif row == 1:
            istc, name = ex.SIMPLE2D, "simple"
        elif row == 2:
            istc, name = ex.COMPLEX2D, "complex"

        istc.draw_static(ax, alpha=1.0, draw_CSpace=True)
        ax.axis("off")
        ax.axis("equal")
        ax.set_title(f"{name}", fontsize=title_fontsize)

        # c2: success rates
        ax = axes[row, 1]
        ax.grid(True)
        ax.set_title(f"Success Rates", fontsize=title_fontsize)
        for method in methods:
            N = list(data[method]["success_rate"].keys())
            success_rate = np.array([data[method]["success_rate"][n] for n in N])
            ax.plot(np.array(N, dtype=int), success_rate, label=labels[method], color=colors[method], linestyle=lines[method], marker=markers[method], mfc="none")
        y_labels = [0.0, 0.33, 0.67, 1.0]
        ax.set_yticks(y_labels)
        ax.set_yticklabels([f"{_val:.0%}" for _val in y_labels], fontsize=axis_fontsize, rotation=60)
        ax.tick_params(axis='y', which='major', pad=0)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('center')
        ax.set_xticks([1, 4, 7, 10])

        # c5: runtime
        ax = axes[row, 2]
        ax.grid(True)
        ax.set_title(f"Runtime (secs.)", fontsize=title_fontsize)
        
        for method in methods:
            N = list(data[method]["runtime"].keys())
            runtime_mean = np.array([max(0, data[method]["runtime"][n][0]) for n in N])
            ax.plot(np.array(N, dtype=int), runtime_mean, label=labels[method], color=colors[method], linestyle=lines[method], marker=markers[method], mfc="none")
        
        ax.set_yscale('log')
        ax.set_xticks([1, 4, 7, 10])
        ax.tick_params(axis='y', labelrotation=60, pad=-2)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('center')

        # c3: sum of costs
        ax = axes[row, 3]
        ax.set_title(f"Sum of Costs", fontsize=title_fontsize)
        ax.grid(True)
        ax.set_xticks([1, 4, 7, 10])

        # def forward(x):
        #     return x**(1/2)
        # def inverse(x):
        #     return x**2
        # ax.set_yscale('function', functions=(forward, inverse))
        # if istc_name == "empty2d":
        #     a, b, c = 2, 10, 22
        # elif istc_name == "simple2d":
        #     a, b, c = 4, 25, 80
        # elif istc_name == "complex2d":
        #     a, b, c = 5, 30, 45
        # setup_two_part_plot(ax, a, b, c, ratio=0.9, num_ticks_lower=4, num_ticks_upper=1)
        for method in methods:
            N = list(data[method]["soc"].keys())
            soc_mean = np.array([data[method]["soc"][n][0] for n in N])
            ax.plot(np.array(N, dtype=int), soc_mean, label=labels[method], color=colors[method], linestyle=lines[method], marker=markers[method], mfc="none")
        
        # c4: makespan
        ax = axes[row, 4]
        ax.grid(True)
        ax.set_xticks([1, 4, 7, 10])
        ax.set_title(f"Makespan", fontsize=title_fontsize)
        # if istc_name == "empty2d":
        #     a, b, c, ratio, n_low, n_up = 1, 3, 9, 0.9, 4, 1
        # elif istc_name == "simple2d":
        #     a, b, c, ratio, n_low, n_up = 1, 6, 32, 0.9, 4, 1
        # elif istc_name == "complex2d":
        #     a, b, c, ratio, n_low, n_up = 4, 5, 9, 0.1, 1, 4
        # setup_two_part_plot(ax, a, b, c, ratio=ratio, num_ticks_lower=n_low, num_ticks_upper=n_up)
        for method in methods:
            N = list(data[method]["makespan"].keys())
            makespan_mean = np.array([data[method]["makespan"][n][0] for n in N])
            ax.plot(np.array(N, dtype=int), makespan_mean, label=labels[method], color=colors[method], linestyle=lines[method], marker=markers[method], mfc="none")
            # ax.fill_between(np.array(N, dtype=int), makespan_mean - makespan_std, makespan_mean + makespan_std, alpha=0.2, color=colors[method])
        
        ax.set_yscale('log', base=2)
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # if name == "empty":
        #     ax.set_yticks([1.2, 1.7, 2.4])
        #     ax.set_yticklabels([1.2, 1.7, 2.4])
        # elif name == "simple":
        #     ax.set_yticks([3.0, 8.0, 22])
        #     ax.set_yticklabels([3.0, 8.0, 22])
        # elif name == "complex":
        #     ax.set_yticks([4.3, 5.8, 8.0])
        #     ax.set_yticklabels([4.3, 5.8, 8.0])

    subplot("empty2d", 0)
    subplot("simple2d", 1)
    subplot("complex2d", 2)
    
    handles, labels = axes[1][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", fontsize=legend_fontsize, bbox_to_anchor=(0.5, 1.08), frameon=True, columnspacing=6, ncols=len(methods), prop={'size': legend_fontsize})
    
    plt.savefig("stats.pdf", bbox_inches='tight', dpi=500)
    

def plot_constrained_sampling():
    methods = ["SP+ST-RRTStar", "SP+ST-RRTStar-C", "SP+TPRM", "SP+TPRM-C"]
    colors = {"SP+TPRM": "b", "SP+ST-RRTStar": "c", "SP+TPRM-C": "g", "SP+ST-RRTStar-C": "r"}
    lines = {"SP+TPRM": "--", "SP+ST-RRTStar": ":", "SP+TPRM-C": "-", "SP+ST-RRTStar-C": "-"}
    markers = {"SP+TPRM": "X", "SP+ST-RRTStar": "+", "SP+TPRM-C": "^", "SP+ST-RRTStar-C": "o"}
    labels = {"SP+TPRM": r"SP+T-PRM(u)", "SP+ST-RRTStar": r"SP+ST-RRT$^*$(u)", "SP+TPRM-C": r"SP+T-PRM", "SP+ST-RRTStar-C": r"SP+ST-RRT$^*$"}

    num_cols, nrows = 3, 1
    widths = [1.8] * num_cols
    heights = [1.8] * nrows
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    legend_fontsize = 8.8
    axis_fontsize = 8
    title_fontsize = 10
    fig, axes = plt.subplots(ncols=num_cols, nrows=nrows, constrained_layout=True, gridspec_kw=gs_kw, figsize=(2.0*num_cols*0.8, sum(heights)*0.8))

    def subplot(istc_name:str):
        # get success rates and successful seeds for each method
        data, valid_seeds = {}, {}
        for method in methods:
            raw_data = pickle.load(open(f"data/{istc_name}-{method}.pkl", "rb"))
            n_robots_list = list(raw_data.keys())
            valid_seeds[method] = {}
            success_rate = {}
            for n_robots, stats in raw_data.items():
                n_succeeds = 0
                valid_seeds[method][n_robots] = []
                for seed, val in enumerate(stats):
                    succ = val[0] != []
                    n_succeeds += int(succ)
                    if succ:
                        valid_seeds[method][n_robots].append(seed)
                    
                success_rate[n_robots] = n_succeeds / len(stats)

            data[method] = {"success_rate": success_rate, "soc": {}, "makespan": {}, "runtime": {}}

        # remove methods that have no successful seeds for any n_robots
        valid_methods = {}
        for n_robots in n_robots_list:
            valid_methods[n_robots] = deepcopy(methods)
            for method in methods:
                if valid_seeds[method][n_robots] == []:
                    valid_methods[n_robots].remove(method)
        
        # get the stats for methods have suceeeds for at least one seed
        SC, MS, RT = {}, {}, {}
        for n_robots in n_robots_list:
            seeds = set(valid_seeds[valid_methods[n_robots][0]][n_robots])
            for method in valid_methods[n_robots]:
                seeds = seeds.intersection(valid_seeds[method][n_robots])
            
            if seeds == set():
                continue

            soc, makespan, runtime = {}, {}, {}
            for method in valid_methods[n_robots]:
                raw_data = pickle.load(open(f"data/{istc_name}-{method}.pkl", "rb"))
                soc_total, makespan_total, runtime_total = [], [], []
                for seed in seeds:
                    sum_cost, max_cost = 0, 0
                    for pi in raw_data[n_robots][seed][0]:
                        cost = pi[-1][-1] - pi[0][2]
                        sum_cost += cost
                        max_cost = max(max_cost, cost)
                    
                    soc_total.append(sum_cost)
                    makespan_total.append(max_cost)
                    runtime_total.append(raw_data[n_robots][seed][1])
                
                soc[method] = [np.mean(soc_total), np.min(soc_total), np.max(soc_total)]
                makespan[method] = [np.mean(makespan_total), np.min(makespan_total), np.max(makespan_total)]
                runtime[method] = [np.mean(runtime_total), np.min(runtime_total), np.max(runtime_total)]
            
            SC[n_robots] = soc
            MS[n_robots] = makespan
            RT[n_robots] = runtime
        
        for method in methods:
            soc, makespan, runtime = {}, {}, {}
            for n_robots in n_robots_list:
                if n_robots not in SC or method not in SC[n_robots]:
                    continue
                soc[n_robots] = SC[n_robots][method]
                makespan[n_robots] = MS[n_robots][method]
                runtime[n_robots] = RT[n_robots][method]
            data[method]["soc"] = soc
            data[method]["makespan"] = makespan
            data[method]["runtime"] = runtime

        # c1: success rates
        ax = axes[0]
        ax.grid(True)
        ax.set_title(f"Success Rates", fontsize=title_fontsize)
        ax.set_xticks([1, 4, 7, 10])
        for method in methods:
            N = list(data[method]["success_rate"].keys())
            success_rate = np.array([data[method]["success_rate"][n] for n in N])
            ax.plot(np.array(N, dtype=int), success_rate, label=labels[method], color=colors[method], linestyle=lines[method], marker=markers[method], mfc="none")
        y_labels = [0.0, 0.33, 0.67, 1.0]
        ax.set_yticks(y_labels)
        ax.set_yticklabels([f"{_val:.0%}" for _val in y_labels], fontsize=axis_fontsize, rotation=60)
        ax.tick_params(axis='y', which='major', pad=0)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('center')

        # c3: makespan
        ax = axes[2]
        ax.grid(True)
        ax.set_title(f"Makespan", fontsize=title_fontsize)
        ax.set_xticks([1, 4, 7, 10])
        for method in methods:
            N = list(data[method]["makespan"].keys())
            makespan_mean = np.array([data[method]["makespan"][n][0] for n in N])
            makespan_std = np.array([data[method]["makespan"][n][1] for n in N])
            ax.plot(np.array(N, dtype=int), makespan_mean, label=labels[method], color=colors[method], linestyle=lines[method], marker=markers[method], mfc="none")
            # ax.fill_between(np.array(N, dtype=int), makespan_mean - makespan_std, makespan_mean + makespan_std, alpha=0.2, color=colors[method])
        
        ax.set_yscale('log', base=2)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # MIN, MAX = ax.get_ylim()
        # y_labels = np.linspace(max(0, MIN), MAX, 4)
        # ax.set_yticks(y_labels)
        # ax.set_yticklabels([f"{_val:.1f}" for _val in y_labels], fontsize=axis_fontsize)
        ax.set_yticks([3, 9, 30])
        ax.set_yticklabels([3, 9, 30])

        # c4: sum of costs
        ax = axes[1]
        ax.grid(True)
        ax.set_title(f"Sum of Costs", fontsize=title_fontsize)
        ax.set_xticks([1, 4, 7, 10])
        for method in methods:
            N = list(data[method]["soc"].keys())
            soc_mean = np.array([data[method]["soc"][n][0] for n in N])
            ax.plot(np.array(N, dtype=int), soc_mean, label=labels[method], color=colors[method], linestyle=lines[method], marker=markers[method], mfc="none")

        # ax.set_yscale('log', base=2)
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_yticks([0, 33, 66, 99])
        ax.set_yticks([0, 33, 66, 99])
        # ax.set_ylim(MIN - 0.1 * (MAX-MIN), MAX + 0.2 * (MAX-MIN))
        # y_labels = np.linspace(max(0, MIN), MAX, 4)
        # ax.set_yticks(y_labels)
        # ax.set_yticklabels([f"{_val:.1f}" for _val in y_labels], fontsize=axis_fontsize)
  
    subplot("simple2d")
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", fontsize=legend_fontsize, bbox_to_anchor=(0.5, 1.22), frameon=True, columnspacing=0.4, ncols=len(methods), prop={'size': legend_fontsize})
    
    # plt.show()
    plt.savefig("sampling.pdf", bbox_inches='tight', dpi=500)


def plot_graph_size():
    methods = ["SP", "PP", "PBS"]
    colors = {"PBS": "k", "PP": "b", "SP": "r"}
    lines = {"PBS": "-", "PP": "--", "SP": ":"}
    markers = {"PBS": "s", "PP": "+", "SP": "o"}
    labels = {"PBS": "PBS+ST-GCS", "PP": "RP+ST-GCS", "SP":"SP+ST-GCS"}

    num_cols, nrows = 3, 1
    widths = [1.8] * num_cols
    heights = [1.8] * nrows
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    legend_fontsize = 8.8
    title_fontsize = 10
    fig, axes = plt.subplots(ncols=num_cols, nrows=nrows, constrained_layout=True, gridspec_kw=gs_kw, figsize=(2.0*num_cols*0.8, sum(heights)*0.8))

    def subplot(istc_name:str, i:int):
        # get success rates and successful seeds for each method
        data = {}
        for method in methods:
            raw_data = pickle.load(open(f"data/{istc_name}.gs", "rb"))
            data[method] = {"graph_size": {}}
            for n_robots, stats in raw_data[method].items():
                stats = np.array(stats)
                stats = np.delete(stats, np.where(stats == -1))
                data[method]["graph_size"][n_robots] = np.mean(stats)

        ax = axes[i]
        ax.grid(True)
        ax.set_title(istc_name[:-2], fontsize=title_fontsize)
        ax.set_xticks([1, 4, 7, 10])
        for method in methods:
            N, mean = [], []
            for n, E in data[method]["graph_size"].items():
                if not np.isnan(E):
                    N.append(n)
                    mean.append(max(1, E))
            ax.plot(np.array(N, dtype=int), mean, label=labels[method], color=colors[method], linestyle=lines[method], marker=markers[method], mfc="none", alpha=0.7)
        
        ax.set_yscale('log', base=2)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if i == 0:
            ax.set_yticks([5, 90, 2000])
            ax.set_yticklabels(['5', '90', '2k'])
        elif i == 1:
            ax.set_yticks([400, 900, 2000])
            ax.set_yticklabels(['400', '900', '2k'])
        elif i == 2:
            ax.set_yticks([80, 400, 2000])
            ax.set_yticklabels(['80', '400', '2k'])

        ax.tick_params(axis='y', labelrotation=60, pad=-2)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('center')

    subplot("empty2d", 0)
    subplot("simple2d", 1)
    subplot("complex2d", 2)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", fontsize=legend_fontsize, bbox_to_anchor=(0.5, 1.22), frameon=True, columnspacing=5, ncols=len(methods), prop={'size': legend_fontsize})

    fig.savefig("graph_size.pdf", bbox_inches='tight', dpi=500)


plot_stats()
# plot_constrained_sampling()
# plot_graph_size()

