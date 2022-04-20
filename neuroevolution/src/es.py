import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
from multiprocessing import Pool

from .agent import Agent
from .model import Network
from .env import *


def ES(config):
    # Get network dims
    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg = {**config, **cfg}  # Merge configs

    # Update weights
    mu = cfg["mu"]
    w = np.array([np.log(mu + 0.5) - np.log(i)
                  for i in range(1, mu + 1)])
    w /= np.sum(w)

    env = make_env(cfg["env_name"], robot=cfg["robot"])

    # Center of the distribution
    elite = Agent(Network, cfg)
    elite.fitness = -np.inf
    theta = elite.genes
    d = len(theta)

    fits = []
    total_evals = []

    bar = tqdm(range(cfg["generations"]))
    for gen in bar:
        population = []
        for i in range(cfg["lambda"]):
            genes = theta + np.random.randn(len(theta)) * cfg["sigma"]
            ind = Agent(Network, cfg, genes=genes)
            # ind.fitness = evaluate(ind, env, max_steps=cfg["max_steps"])
            population.append(ind)

        with Pool(processes=len(population)) as pool:
            pop_fitness = pool.starmap(mp_eval, [(a, cfg) for a in population])

        for i in range(len(population)):
            population[i].fitness = pop_fitness[i]

        # sort by fitness
        inv_fitnesses = [- f for f in pop_fitness]
        # indices from highest fitness to lowest
        idx = np.argsort(inv_fitnesses)

        step = np.zeros(d)
        for i in range(mu):
            # update step
            step = step + w[i] * (population[idx[i]].genes - theta)
        # update theta
        theta = theta + step * cfg["lr"]

        if pop_fitness[idx[0]] > elite.fitness:
            elite.genes = population[idx[0]].genes
            elite.fitness = pop_fitness[idx[0]]

        fits.append(elite.fitness)
        total_evals.append(len(population) * (gen+1))

        bar.set_description(f"Best: {elite.fitness}")

    env.close()
    # plt.plot(total_evals, fits)
    # plt.xlabel("Evaluations")
    # plt.ylabel("Fitness")
    # plt.show()
    d = {
        "elite": elite,
        "fitness": fits,
        "evals": total_evals
    }
    return d
