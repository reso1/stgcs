from typing import List, Tuple, Dict
from collections import defaultdict
import time
import argparse
import os, sys

import pickle
import numpy as np
import matplotlib.pyplot as plt

from environment.problems import MRMP

from baselines.st_rrt_star.planner import STRRTStar, Options
from baselines.rp_stgcs import randomized_prioritized_planning as PP_STGCS, sequential_planning as SP_STGCS
from baselines.sp_strrtstar import sequential_planning as SP_STRRTStar
from baselines.sp_tprm import sequential_planning as SP_TPRM

from mrmp.pbs import PBS


TIMEOUT = 150.0


def exp_PBS_STGCS(ps: Dict[int, List[MRMP]]) -> None:
    res, num_suceeds, num_problems = defaultdict(list), 0, 0
    for num_agents, problems in ps.items():
        for seed, p in enumerate(problems):
            ts = time.perf_counter()
            solution, _ = PBS(p.istc, p.tmax, p.vlimit, p.robot_radius, p.starts, p.goals, p.T0s, timeout_secs=TIMEOUT)
            time_elapsed = time.perf_counter() - ts
            res[num_agents].append(([sol.trajectory for sol in solution], time_elapsed))
            num_problems += 1
            if solution != []:
                num_suceeds += 1

        # input(f"Finished {num_agents} agents. Success rate: {num_suceeds / len(problems)}. Press enter to continue.")
    
    print(f"Success rate: {num_suceeds / (len(ps) * len(problems))}")
    pickle.dump(res, open(f"{ps[1][0].istc.name}-PBS+STGCS.pkl", "wb"))


def exp_PP_STGCS(ps: Dict[int, List[MRMP]]) -> None:
    res, num_suceeds, num_problems = defaultdict(list), 0, 0
    for num_agents, problems in ps.items():
        for seed, p in enumerate(problems):
            ts = time.perf_counter()
            solution, _ = PP_STGCS(p.istc, p.tmax, p.vlimit, p.robot_radius, p.starts, p.goals, p.T0s, seed=seed, max_ordering_trials=1000, timeout_secs=TIMEOUT)
            time_elapsed = time.perf_counter() - ts
            res[num_agents].append(([sol.trajectory for sol in solution], time_elapsed))
            num_problems += 1
            if solution != []:
                num_suceeds += 1

        # input(f"Finished {num_agents} agents. Success rate: {num_suceeds / len(problems)}. Press enter to continue.")
    
    print(f"Success rate: {num_suceeds / (len(ps) * len(problems))}")
    pickle.dump(res, open(f"{ps[1][0].istc.name}-PP+STGCS.pkl", "wb"))


def exp_SP_STGCS(ps: Dict[int, List[MRMP]]) -> None:
    res, num_suceeds, num_problems = defaultdict(list), 0, 0
    for num_agents, problems in ps.items():
        for seed, p in enumerate(problems):
            ts = time.perf_counter()
            solution, _ = SP_STGCS(p.istc, p.tmax, p.vlimit, p.robot_radius, p.starts, p.goals, p.T0s, TIMEOUT)
            time_elapsed = time.perf_counter() - ts
            res[num_agents].append(([sol.trajectory for sol in solution], time_elapsed))
            num_problems += 1
            if solution != []:
                num_suceeds += 1

            print("# agents:", num_agents, "# succeeds:", num_suceeds, "# problems:", num_problems)

        # input(f"Finished {num_agents} agents. Success rate: {num_suceeds / len(problems)}. Press enter to continue.")
    
    print(f"Success rate: {num_suceeds / (len(ps) * len(problems))}")
    pickle.dump(res, open(f"{ps[1][0].istc.name}-SP+STGCS.pkl", "wb"))


def exp_SP_TPRM_C(ps: Dict[int, List[MRMP]]) -> None:
    res, num_suceeds, num_problems = defaultdict(list), 0, 0
    for num_agents, problems in ps.items():
        for seed, p in enumerate(problems):
            ts = time.perf_counter()
            solution = SP_TPRM(p.istc, p.tmax, p.vlimit, p.robot_radius, p.starts, p.goals, p.T0s, seed, TIMEOUT, True)
            time_elapsed = time.perf_counter() - ts
            res[num_agents].append(([sol.trajectory for sol in solution], time_elapsed))
            num_problems += 1
            if solution != []:
                num_suceeds += 1

            print("# agents:", num_agents, "# succeeds:", num_suceeds, "# problems:", num_problems)

        # input(f"Finished {num_agents} agents. Success rate: {num_suceeds / len(problems)}. Press enter to continue.")
    
    print(f"Success rate: {num_suceeds / (len(ps) * len(problems))}")
    pickle.dump(res, open(f"{ps[1][0].istc.name}-SP+TPRM-C.pkl", "wb"))


def exp_SP_TPRM(ps: Dict[int, List[MRMP]]) -> None:
    res, num_suceeds, num_problems = defaultdict(list), 0, 0
    for num_agents, problems in ps.items():
        for seed, p in enumerate(problems):
            ts = time.perf_counter()
            solution = SP_TPRM(p.istc, p.tmax, p.vlimit, p.robot_radius, p.starts, p.goals, p.T0s, seed, TIMEOUT, False)
            time_elapsed = time.perf_counter() - ts
            res[num_agents].append(([sol.trajectory for sol in solution], time_elapsed))
            num_problems += 1
            if solution != []:
                num_suceeds += 1

            print("# agents:", num_agents, "# succeeds:", num_suceeds, "# problems:", num_problems)

        # input(f"Finished {num_agents} agents. Success rate: {num_suceeds / len(problems)}. Press enter to continue.")
    
    print(f"Success rate: {num_suceeds / (len(ps) * len(problems))}")
    pickle.dump(res, open(f"{ps[1][0].istc.name}-SP+TPRM.pkl", "wb"))


def exp_SP_STRRTStar(ps: Dict[int, List[MRMP]]) -> None:
    res, num_suceeds, num_problems = defaultdict(list), 0, 0
    for num_agents, problems in ps.items():
        for seed, p in enumerate(problems):
            ts = time.perf_counter()
            solution = SP_STRRTStar(p.istc, p.tmax, p.vlimit, p.robot_radius, p.starts, p.goals, p.T0s, seed, TIMEOUT, use_CSpace=False)
            time_elapsed = time.perf_counter() - ts
            res[num_agents].append(([sol.trajectory for sol in solution], time_elapsed))
            num_problems += 1
            if solution != []:
                num_suceeds += 1

            print("# agents:", num_agents, "# succeeds:", num_suceeds, "# problems:", num_problems)

        # input(f"Finished {num_agents} agents. Success rate: {num_suceeds / len(problems)}. Press enter to continue.")
    
    print(f"Success rate: {num_suceeds / (len(ps) * len(problems))}")
    pickle.dump(res, open(f"{ps[1][0].istc.name}-SP+ST-RRTStar.pkl", "wb"))


def exp_SP_STRRTStar_C(ps: Dict[int, List[MRMP]]) -> None:
    res, num_suceeds, num_problems = defaultdict(list), 0, 0
    for num_agents, problems in ps.items():
        for seed, p in enumerate(problems):
            ts = time.perf_counter()
            solution = SP_STRRTStar(p.istc, p.tmax, p.vlimit, p.robot_radius, p.starts, p.goals, p.T0s, seed, TIMEOUT, use_CSpace=True)
            time_elapsed = time.perf_counter() - ts
            res[num_agents].append(([sol.trajectory for sol in solution], time_elapsed))
            num_problems += 1
            if solution != []:
                num_suceeds += 1

            print("# agents:", num_agents, "# succeeds:", num_suceeds, "# problems:", num_problems)

        # input(f"Finished {num_agents} agents. Success rate: {num_suceeds / len(problems)}. Press enter to continue.")
    
    print(f"Success rate: {num_suceeds / (len(ps) * len(problems))}")
    pickle.dump(res, open(f"{ps[1][0].istc.name}-SP+ST-RRTStar-C.pkl", "wb"))


def exp_STGCS_graph_size(ps: Dict[int, List[MRMP]]) -> None:
    graph_size = {"SP": defaultdict(list), "PP": defaultdict(list), "PBS": defaultdict(list)}
    for num_agents, problems in ps.items():
        for seed, p in enumerate(problems):
            solution, num_edges = SP_STGCS(p.istc, p.tmax, p.vlimit, p.robot_radius, p.starts, p.goals, p.T0s, TIMEOUT)
            graph_size["SP"][num_agents].append(num_edges)
            solution, num_edges = PP_STGCS(p.istc, p.tmax, p.vlimit, p.robot_radius, p.starts, p.goals, p.T0s, seed, max_ordering_trials=1000, timeout_secs=TIMEOUT)
            graph_size["PP"][num_agents].append(num_edges)
            solution, num_edges = PBS(p.istc, p.tmax, p.vlimit, p.robot_radius, p.starts, p.goals, p.T0s, timeout_secs=TIMEOUT)
            graph_size["PBS"][num_agents].append(num_edges)

    pickle.dump(graph_size, open(f"data/{ps[1][0].istc.name}.gs", "wb"))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-problem_set", type=str)
    argparser.add_argument("-method", type=str)
    args = argparser.parse_args()

    ps: Dict[int, List[MRMP]] = pickle.load(open(f"data/{args.problem_set}.ps", "rb"))
    if args.method == "PBS+STGCS":
        exp_PBS_STGCS(ps)
    elif args.method == "PP+STGCS":
        exp_PP_STGCS(ps)
    elif args.method == "SP+STGCS":
        exp_SP_STGCS(ps)
    elif args.method == "SP+STRRTStar":
        exp_SP_STRRTStar(ps)
    elif args.method == "SP+STRRTStar-C":
        exp_SP_STRRTStar_C(ps)
    elif args.method == "SP+TPRM":
        exp_SP_TPRM(ps)
    elif args.method == "SP+TPRM-C":
        exp_SP_TPRM_C(ps)
    elif args.method == "STGCS_graph_size":
        exp_STGCS_graph_size(ps)
    else:
        raise ValueError(f"Invalid method: {args.method}")
