from src.Classes_v2.Policy import Policy
from src.Classes_v2.Agent import Agent
import random
import numpy as np


def next(policy, state_index):
    assert state_index >= 0 < policy.state_dimension
    rand = random.random()
    action_subset = [0.25 * i for i in range(4)]
    # See https://stackoverflow.com/questions/16243955/numpy-first-occurrence-of-value-greater-than-existing-value
    return(int(np.searchsorted(np.cumsum(action_subset), rand)))

policy = Policy(next, 16, 4)

policy.pick_next(2)

def update_SARSA(policy: Policy, q_matrix, state_index: int, action_index: int, next_state: int, next_action: int,  reward:float = 0.0, alpha = 0.5, gamma = 0.5):
    # Q[s, a] := Q[s, a] + α[r + γQ(s', a') - Q(s, a)]
    q_matrix[state_index, action_index] = q_matrix[state_index, action_index] + alpha * (reward + gamma * q_matrix[next_state, next_action] - q_matrix[state_index, action_index])

def update_test(policy: Policy, state, action):
    print(state)
    print(action)
    print('test')
agent = Agent(policy, update_test)
agent.update("hello", "you")


