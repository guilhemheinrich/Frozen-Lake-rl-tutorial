import random
import gymnasium as gym

from SARSA import SARSA as SARSA_V2
from Frozen_Lake_Qlearning import Q_learning as Q_learning_V1
from Q_learning import Q_learning as Q_learning_V2
from Frozen_Lake_SARSA import SARSA as SARSA_V1

from src.Functions.Run import FrozenLake_parameters, run
from src.Classes.Policy import Policy as Policy_V1
from src.Classes.PolicyAgent import PolicyAgent

from src.V2.Classes.Policy import Policy as Policy_V2
from src.V2.Classes.Agent import Agent
from src.V2.Functions.run import run_static

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
print("SARSA V1")
random.seed(seed)
environment.reset(seed=seed)
q_sa1 = SARSA_V1(environment, **options)
print("\nSARSA V2")
random.seed(seed)
environment.reset(seed=seed)
q_sa2 = SARSA_V2(environment, **options)
print("\nQ learning V1")
random.seed(seed)
environment.reset(seed=seed)
q_sa3 = Q_learning_V1(environment, **options)
print("\nQ learning V2")
random.seed(seed)
environment.reset(seed=seed)
q_sa4 = Q_learning_V2(environment, **options)


frozenLake_parameters: FrozenLake_parameters = {'desc' : ["SFFF", "FHFH", "FFFH", "HFFG"], 'is_slippery'  : True}

print("\nTest phase")
success = 0
print("SARSA V1")

random.seed(seed)
environment.reset(seed=seed)
deterministic_policy = Policy_V1.buildOptimalPolicyFrom(q_sa1)
for epoch in range(test_epoch):
    test_agent_v1 = PolicyAgent(deterministic_policy, initial_state_index = 0)
    run(frozenLake_parameters, test_agent_v1)
    success += test_agent_v1.current_state_index == (environment.observation_space.n - 1) # type: ignore
print("check random seed: " + str(random.random()))
print("Success rate: " + str(success/test_epoch))

print("\nSARSA V2")

success = 0
random.seed(seed)
environment.reset(seed=seed)
deterministic_policy = Policy_V2.buildOptimalPolicyFrom(q_sa2)
for epoch in range(test_epoch):
    test_agent_SARSA_V2 = Agent(deterministic_policy)
    run_static(environment = environment, agent = test_agent_SARSA_V2)
    success += test_agent_SARSA_V2.current_state_index == (environment.observation_space.n - 1) # type: ignore
print("check random seed: " + str(random.random()))
print("Success rate: " + str(success/test_epoch))

print("\nQ learning V1")

success = 0
random.seed(seed)
environment.reset(seed=seed)
deterministic_policy = Policy_V1.buildOptimalPolicyFrom(q_sa3)
frozenLake_parameters: FrozenLake_parameters = {'desc' : ["SFFF", "FHFH", "FFFH", "HFFG"], 'is_slippery'  : True}
for epoch in range(test_epoch):
    test_agent_QL_v1 = PolicyAgent(deterministic_policy, initial_state_index = 0)
    run(frozenLake_parameters, agent = test_agent_QL_v1)
    success += test_agent_QL_v1.current_state_index == (environment.observation_space.n - 1) # type: ignore
print("check random seed: " + str(random.random()))
print("Success rate: " + str(success/test_epoch))

print("\nQ learning V2")
success = 0
random.seed(seed)
environment.reset(seed=seed)
deterministic_policy = Policy_V2.buildOptimalPolicyFrom(q_sa4)
for epoch in range(test_epoch):
    test_agent_QL_v2 = Agent(deterministic_policy)
    run_static(environment = environment, agent = test_agent_QL_v2)
    success += test_agent_QL_v2.current_state_index == (environment.observation_space.n - 1) # type: ignore
print("check random seed: " + str(random.random()))
print("Success rate: " + str(success/test_epoch))
