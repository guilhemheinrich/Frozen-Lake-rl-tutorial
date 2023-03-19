import random
import gymnasium as gym
import numpy as np

from SARSA import SARSA
from Q_learning import Q_learning

from src.V2.Classes.Policy import Policy
from src.V2.Classes.Agent import Agent
from src.V2.Functions.run import run_static

seed = 0
epoch_number = 100
test_epoch = 100
grid_steps = 3
mail = {
    "epsilon": {
        "start": 0.1,
        "end": 0.98,
        "steps": grid_steps,
    },
    "beta": {
        "start": 0.1,
        "end": 0.98,
        "steps": grid_steps,
    },
    "gamma": {
        "start": 0.1,
        "end": 0.98,
        "steps": grid_steps,
    },
}
result = []
desc=["SFFF", "FHFH", "FFFH", "HFFG"] # Same as the map called "4*4"

environment = gym.make('FrozenLake-v1', desc=desc, is_slippery=True, render_mode="rgb_array")
for epsilon in np.linspace(mail["epsilon"]["start"], mail["epsilon"]["end"], mail["epsilon"]["steps"]):
    for alpha in np.linspace(mail["beta"]["start"], mail["beta"]["end"], mail["beta"]["steps"]):
        for gamma in np.linspace(mail["gamma"]["start"], mail["gamma"]["end"], mail["gamma"]["steps"]):
            options = {
                "epoch_number": epoch_number, 
                "epsilon": epsilon,
                "alpha": alpha,
                "gamma": gamma
            }
            for algo in [SARSA, Q_learning]:
                random.seed(seed)
                environment.reset(seed=seed)
                q_sa = algo(environment, **options)
                deterministic_policy = Policy.buildOptimalPolicyFrom(q_sa)
                success = 0
                for epoch in range(test_epoch):
                    test_agent = Agent(deterministic_policy)
                    run_static(environment = environment, agent = test_agent)
                    success += test_agent.current_state_index == (environment.observation_space.n - 1) # type: ignore
                result.append({
                    "algorithm": algo.__name__,
                    "success": success / test_epoch,
                    "q_sa": q_sa,
                    "epsilon": epsilon,
                    "alpha": alpha,
                    "gamma": gamma
                })
            pass

best_success = result[0]
result.sort(reverse=True, key = lambda element: element["success"])
for test in result:
    if test["success"] > best_success["success"]:
        best_success = test
    pass

# to make plot interactive 
import matplotlib
%matplotlib


from src.V2.Functions.plot3d_heatmap import draw_3D
draw_3D(
    result,
    x_key="epsilon",
    y_key="alpha",
    z_key="gamma",
    scalar_key="success"
    )
