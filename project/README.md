# Project - Evolution of agents

The goal of this project is to get some experience running and comparing
evolutionary algorithms. We will be competing on the [MinAtar
benchmark](https://github.com/kenjyoung/MinAtar), specifically on the Breakout
and Freeway games. 

You will work in teams of 4-5 on a specific algorithm. You can optimize the
hyperparameters of this method and propose algorihtm enhancements for the
MinAtar environments in order to maximize your team's final score. We will
compare all 5 algorithms on the two games using a total maximum budget of 50000
evaluations for a single evolution. You should run the evolutionary method at
least 3 times to give an idea of final average and fitness.

The possible algorithms are:
+ CGP (Cartesian Genetic Programming): [implementations](https://www.cartesiangp.com/resources), [Julia](https://github.com/d9w/CartesianGeneticProgramming.jl/)
+ NEAT: [implementations](http://eplex.cs.ucf.edu/neat_software/), [Python](https://github.com/d9w/prettyNEAT), [Julia](https://github.com/TemplierPaul/NeuroEvolution.jl)
+ MAP-Elites: [Python](https://github.com/resibots/pymap_elites), [tutorial](https://github.com/jbmouret/map_elites_tutorial)
+ Canonical ES for neuroevolution: [Python](https://github.com/PatrykChrabaszcz/Canonical_ES_Atari)
+ CMA-ES for neuroevolution: [pycma](https://github.com/CMA-ES/pycma), [estool](https://github.com/hardmaru/estool)
+ TPG (Tangled Problem Graphs): [Python](https://github.com/Ryan-Amaral/PyTPG)

The deliverable for this project is a presentation of 15 minutes. Your team should present:
+ an overview of the algorithm
+ how the algorithm is applied to MinAtar
+ a study of hyperparameter impact on performance
+ presentation of final results: evolution, agent video

Here is a presentation explaining the evolution of agents and showing an example:
  + [notebook](https://github.com/d9w/evolution/blob/master/project/Project.ipynb)
  + [video](https://youtu.be/ByGsyRRvYuk)

