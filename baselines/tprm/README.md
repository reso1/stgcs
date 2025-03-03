This is an Python translation of the official C++ [Code](https://github.com/VIS4ROB-lab/t_prm) for the [paper](https://ieeexplore.ieee.org/iel7/9981026/9981028/09981739.pdf).

> **An issue fixed for the original T-PRM implmentation**: It has no dynamic collision checking for the motion along edges but only calculting the "safe time intervals" on each sampled configuration.
Therefore sometimes agents would "pass through" obstacles by connecting two configurations that are not colliding w/ dynmiac obstacles.


