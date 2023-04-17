import random
import gymnasium as gym

from Q_learning_controlled import Q_learning
from src.V2.Classes.Policy import Policy as Policy_V2
from src.V2.Classes.Agent import Agent
from src.V2.Functions.run import run_static

desc=["SFFF", "FHFH", "FFFH", "HFFG"] # Same as the map called "4*4"

environment = gym.make('FrozenLake-v1', desc=desc, is_slippery=True, render_mode="rgb_array")

test_epoch = 1000
options = {
    "warmup_epoch": 3000,
    "maximum_epoch": 150000,
    "epsilon": 0.2,
    "alpha": 0.1,
    "gamma": 0.99
}

q_sa = Q_learning(environment, **options)

deterministic_policy = Policy_V2.buildOptimalPolicyFrom(q_sa)
success = 0
for epoch in range(test_epoch):
    test_agent_v1 = Agent(deterministic_policy, initial_state_index = 0)
    run_static(environment, test_agent_v1)
    success += test_agent_v1.current_state_index == (environment.observation_space.n - 1) 
print("Success rate: " + str(success/test_epoch))
