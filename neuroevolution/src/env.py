import numpy as np
import gym
import evogym.envs
from evogym import sample_robot
from evogym.utils import get_full_connectivity


def make_env(env_name, seed=None, robot=None, **kwargs):
    if robot is None:  # Random robot
        robot, connections = sample_robot((5, 5))
    connections = get_full_connectivity(robot)
    env = gym.make(env_name, body=robot)
    if seed is not None:
        env.seed(seed)
    return env


def evaluate(agent, env, max_steps=1000, render=False):
    obs = env.reset()
    reward = 0
    steps = 0
    done = False
    while not done and steps < max_steps:
        if render:
            env.render()
        action = agent.act(obs)
        obs, r, done, _ = env.step(action)
        reward += r
        steps += 1
    return reward


def get_cfg(env_name, robot=None):
    env = make_env(env_name, robot=robot)
    cfg = {
        "n_in": env.observation_space.shape[0],
        "h_size": 32,
        "n_out": env.action_space.shape[0],
    }
    env.close()
    return cfg


def mp_eval(a, cfg):
    env = make_env(cfg["env_name"], robot=cfg["robot"])
    fit = evaluate(a, env, max_steps=cfg["max_steps"])
    env.close()
    return fit
