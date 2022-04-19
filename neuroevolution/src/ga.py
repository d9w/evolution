import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool

from .agent import Agent
from .model import Network
from .env import *


def one_plus_lambda(config):
    # Get network dims
    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg = {**config, **cfg}  # Merge configs

    env = make_env(cfg["env_name"], robot=cfg["robot"])

    # Elite
    elite = Agent(Network, cfg)

    elite.fitness = evaluate(elite, env, max_steps=cfg["max_steps"])

    fits = []
    total_evals = []

    bar = tqdm(range(cfg["generations"]))
    for gen in bar:
        population = [Agent(Network, cfg, genes=a.mutate_ga())
                      for _ in range(cfg["lambda"])]

        with Pool(processes=len(population)) as pool:
            pop_fitness = pool.starmap(mp_eval, [(a, cfg) for a in population])

        # pop_fitness = [evaluate(a, env, max_steps=cfg["max_steps"]) for a in population]

        best = np.argmax(pop_fitness)
        best_fit = pop_fitness[best]
        if best_fit > elite.fitness:
            elite.genes = population[best].genes
            elite.fitness = best_fit
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
