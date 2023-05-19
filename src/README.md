# Original Source Code

Code Source: [https://github.com/yrlu/irl-imitation](https://github.com/yrlu/irl-imitation)

* We modified to fit in Python3.10
* modified `mdp/value_iteration.py` 
    * if `deterministic`: original max value iteration
    * if not `deterministic`: Apporximate Value Iteration in [Maximum Entropy Deep Inverse Reinforcement Learning](https://arxiv.org/abs/1507.04888)
    * add `alpha` argument: temperature
* modified `maxent_irl.py`
    * add return last time step's `policy`
    * add temperature argument `alpha` for `value_iteration`
* modified `maxent_irl_gridworld.py`

