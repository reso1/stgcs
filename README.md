# ST-GCS
This repository implements the Space-Time Graphs of Convex Sets (ST-GCS) from the following paper:

- *Jingtao Tang, Zining Mao, Lufan Yang, and Hang Ma. "Space-Time Graphs of Convex Sets for Multi-Robot Motion Planning." [[paper]](https://arxiv.org/abs/2503.00583), [[project]](https://sites.google.com/view/stgcs)


## Installation
- Python dependencies can be installed via `pip install -r requirements.txt`
- [Mosek](https://www.mosek.com/) solver should be installed, which is used in the [Drake](https://drake.mit.edu/) library for the GCS program solving. 
- [Optional] You can also use Gurobi solver for Drake, however, building Drake from source is required. See detailed installation guidance [here](https://drake.mit.edu/installation.html).


## File Structure
- baselines/
  - st_rrt_star/: the directory containing the [ST-RRT*](https://arxiv.org/abs/2203.02176) algorithm.
  - tprm/: the directory containing the [T-PRM](https://ieeexplore.ieee.org/document/9981739) algorithm.
  - rp_stgcs.py: the random-prioritized planner (RP) and sequential planner (SP) for Multi-Robot Motion Planning (MRMP), with ST-GCS as the low-level single-robot planner.
  - sp_strrtstar.py: the sequential planner (SP) for MRMP with ST-RRT* as the low-level single-robot planner.
  - sp_tprm.py: the sequential planner (SP) for MRMP with T-PRM as the low-level single-robot planner.

- data: the directory containing the problem instances and experiment results.
- demos: a collection of demonstrations for space-time single-robot motion planning and MRMP.
- environment:
  - examples.py: a collection of instance examples.
  - instances.py: the instance class of the environment.
  - obstacle.py: a simple implementation of static obstacles and dynamic obstacles.
  - problems.py: code for generating random problems.

- mrmp/
  - ecd.py: Exact Convex Decomposition implementation.
  - graph.py: the graph of convex sets class, adapted from [GCS*](https://github.com/shaoyuancc/large_gcs).
  - interval.py: a simple class representing an interval.
  - pbs.py: [Prioritized-Based Search](https://aaai.org/ojs/index.php/AAAI/article/view/4758/4636) implementation, adapted from the [conflict solver](https://github.com/reso1/LS-MCPP) for multi-robot coverage path planning.
  - stgcs.py: the Space-Time Graphs of Convex Sets class.
  - utils.py: a collection of utility functions. 

- exp.py: the experiment runner
- plot.py: plot functions for the experiments

## BibTex:
```
@misc{tang2025spacetimegraphsconvexsets,
      title={Space-Time Graphs of Convex Sets for Multi-Robot Motion Planning}, 
      author={Jingtao Tang and Zining Mao and Lufan Yang and Hang Ma},
      year={2025},
      eprint={2503.00583},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2503.00583}, 
}
```

## License
ST-GCS is released under the GPL version 3. See LICENSE.txt for further details.

