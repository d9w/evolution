import numpy as np
import time

from src.es import ES
# from src.ga import GA
from src.load_save import save_solution, benchmark

walker = np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3]
])

config = {
    "env_name": "Walker-v0",
    "robot": walker,
    "generations": 1,
    "lambda": 10,  # Population size
    "mu": 5,  # Parents pop size
    "sigma": 0.1,  # mutation std
    "lr": 1,  # Learning rate
    "max_steps": 500,
}

result = ES(config)
filename = f"solution_{time.strftime('%Y%m%d_%H%M%S')}.json"
save_solution(result, config, filename)

benchmark(filename)
