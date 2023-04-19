# Project - Evolution of agents

This project focuses on the evolution of agents using the [Evolution Gym](https://evolutiongym.github.io/Evolution Gym) suite. To get started
with evogym, see the [documentation](https://evolutiongym.github.io/).

[Neuroevolution in evogym notebook](https://github.com/d9w/evolution/project/evogym.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d9w/evolution/blob/master/project/evogym.ipynb)

You will need to evolve movement policies for three tasks independently:

+ Walker-v0 (easy)
+ Thrower-v0 (medium)
+ Climb-v2 (hard) 

You have a budget of 10000 **evaluations** for evolution (for example, a
population of 10 for 1000 generations).  Each evaluation can have 500 maximum
steps, but you are encouraged to reduce this and the total number of
evaluations while making algorithm decisions to have faster results.  You
choose the evolutionary algorithm, gene representation, and evolutionary
hyperparameters, but you must demonstrate that you only used the allocated
evaluation budget. The goal is to obtain the best score independently on each
task. Scores should be shown in your final presentation as the average best
score over **at least 2 independent evolutions**.

For each task, points will be allocated to teams in the following
manner:

+ 1st place: 5 points
+ 2nd place: 4 points
+ 3rd place: 3 points
+ 4th place: 2 points
+ 5th place: 1 point

Project presentations will take place on Friday, May 5th.

You can use the code provided during class for your evolutionary algorithms, and you can also use any code online. Some popular libraries are:

+ [cmaes](https://github.com/CyberAgentAILab/cmaes)
+ [pycma](https://github.com/CMA-ES/pycma)
+ [pymoo](https://pymoo.org/)
+ [pyribs](https://pyribs.org/)
+ [neat-python](https://github.com/CodeReclaimers/neat-python)
+ [gplearn](https://github.com/trevorstephens/gplearn)
+ [pycgp](https://github.com/scussatb/pyCGP)
+ [DEAP](https://github.com/DEAP/deap)
