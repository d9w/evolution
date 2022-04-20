import json
from .agent import Agent
from .model import Network
from .env import *


def save_solution(result, cfg, name="solution.json"):
    save_cfg = get_cfg(cfg["env_name"], robot=cfg["robot"])
    for i in ["env_name", "robot"]:
        assert i in cfg, f"{i} not in config"
        save_cfg[i] = cfg[i]
    save_cfg["robot"] = cfg["robot"].tolist()
    save_cfg["genes"] = result["elite"].genes.tolist()
    save_cfg["elite_fitness"] = float(result["elite"].fitness)
    save_cfg["fitness"] = result["fitness"]
    save_cfg["evals"] = result["evals"]

    # save
    with open(name, "w") as f:
        json.dump(save_cfg, f)
    print(f"Saved solution to {name}")
    return save_cfg


def load_solution(name="solution.json"):
    with open(name, "r") as f:
        cfg = json.load(f)
    cfg["robot"] = np.array(cfg["robot"])
    cfg["genes"] = np.array(cfg["genes"])
    a = Agent(Network, cfg, genes=cfg["genes"])
    a.fitness = cfg["elite_fitness"]
    return a, cfg


def benchmark(name="solution.json", render=True):
    a, cfg = load_solution(name=name)
    env = make_env(cfg["env_name"], robot=cfg["robot"])
    a.fitness = evaluate(a, env, render=render)
    env.close()
    return a.fitness
