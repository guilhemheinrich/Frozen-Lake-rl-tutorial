import random
import gymnasium as gym

from SARSA import SARSA as V2
from Frozen_Lake_SARSA import SARSA as V1



seed = 3
options = {
    "epoch_number": 5000, 
    "epsilon": 0.1,
    "alpha": 0.1,
    "gamma": 0.99
}
test_epoch = 100
desc=["SFFF", "FHFH", "FFFH", "HFFG"] # Same as the map called "4*4"

environment = gym.make('FrozenLake-v1', desc=desc, is_slippery=True, render_mode="rgb_array")

print("Learning phase")
print("V1\n")
random.seed(seed)
environment.reset(seed=seed)
q_sa1 = V1(environment, **options).value
print("\n\nV2\n")
random.seed(seed)
environment.reset(seed=seed)
q_sa2 = V2(environment, **options)

print("\n\nTest phase")
success = 0
print("V1\n")
from src.Functions.Run import FrozenLake_parameters, run
from src.Classes.Policy import Policy as PolicyV1
from src.Classes.PolicyAgent import PolicyAgent
random.seed(seed)
environment.reset(seed=seed)
deterministic_policy = PolicyV1.buildOptimalPolicyFrom(q_sa1)
for epoch in range(test_epoch):
    test_agent_v1 = PolicyAgent(deterministic_policy, initial_state_index = 0)
    # frozenLake_parameters = {'desc' : ["SFFF", "FHFH", "FFFH", "HFFG"], 'is_slippery'  : True}
    frozenLake_parameters: FrozenLake_parameters = {'desc' : ["SFFF", "FHFH", "FFFH", "HFFG"], 'is_slippery'  : True}
    run(frozenLake_parameters, test_agent_v1)
    success += test_agent_v1.current_state_index == (environment.observation_space.n - 1) # type: ignore
print("Success rate: " + str(success/test_epoch))

print("\n\nV2\n")
from src.V2.Classes.Policy import Policy as PolicyV2
from src.V2.Classes.Agent import Agent as AgentV2
from src.V2.Functions.run import run_static
success = 0
# random.seed(seed)
# environment.reset(seed=seed)
deterministic_policy = PolicyV2.buildOptimalPolicyFrom(q_sa2)
for epoch in range(test_epoch):
    test_agent_v2 = AgentV2(deterministic_policy)
    run_static(environment = environment, agent = test_agent_v2)
    success += test_agent_v2.current_state_index == (environment.observation_space.n - 1) # type: ignore
print("Success rate: " + str(success/test_epoch))
