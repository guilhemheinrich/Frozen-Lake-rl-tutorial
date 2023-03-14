import gymnasium as gym
from gymnasium.utils.play import play
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from PIL import Image
import numpy as np
from IPython.display import display # to display images
import matplotlib.pyplot as plt

from typing import TypeVar, List, Tuple, TypedDict
from src.Classes.Policy import Policy
from src.Classes.PolicyAgent import PolicyAgent

def check(row_index: int, col_index: int) -> None:
    assert 0 <= row_index < nrow
    assert 0 <= col_index < ncol

def stateToCoordinate(index: int) -> Tuple[int, int]:
    row_index = index // ncol
    col_index = index % ncol
    check(row_index, col_index)
    return (row_index, col_index)

def displayGame(environment, action = -1):
    match action:
        case 0:
            print('LEFT !')
        case 1:
            print('DOWN !')
        case 2:
            print('RIGHT !')
        case 3:
            print('UP !')
        case _:
            print('Start')
    rgb_array = environment.render()
    image = Image.fromarray(rgb_array)
    display(image)

desc=["SFFF", "FHFH", "FFFH", "HFFG"] # Same as the map called "4*4"
dimension_desc = (4, 4)
nrow = 4
ncol = 4
environment = gym.make('FrozenLake-v1', desc=desc, is_slippery=True, render_mode="rgb_array")


from src.Classes.QMatrix import QMatrix
from src.Classes.QAgent import Q_algo, QAgent
from src.Classes.QAgent import QAgent

def SARSA(environment, epoch_number = 10000, epsilon = 0.1, alpha = 0.1, gamma = 0.99):
    # Get the action space dimension
    # See https://stackoverflow.com/a/62306159
    # In our case, it's a standard discrete space
    action_space_length: int = environment.action_space.n

    # Get the observation space
    environment_space_length: int = environment.observation_space.n
    # Statistics
    win_count = 0
    win_ratio = []
    truncated_total = 0

    # Generic Initialisation
    Q_sa = QMatrix(
        np.zeros((environment_space_length, action_space_length)),
        epsilon,
        gamma, 
        alpha)

    for epoch in range(epoch_number):
        stop = False
        environment.reset()
        SARSA_agent = QAgent(Q_sa, initial_state_index = 0, algo = Q_algo.SARSA)
        current_state = SARSA_agent.current_state_index
        action = SARSA_agent.pickNextAction(current_state)
        while not stop:
            '''
            While Agent is not stopped
            Agent perform the action
            Agent pick an action from its state and its policy
            Agent update its trajectory: action taken, new state, new reward
            Update the stop condition if Goal reached or terminated
            '''
            observation, reward, terminated, truncated, info = environment.step(action)
            next_action = SARSA_agent.pickNextAction(observation)
            SARSA_agent.current_state_index = observation    
            Q_sa.update_SARSA(current_state, action, observation, next_action, float(reward))
            current_state = observation
            action = next_action
            if reward == 1: win_count += 1
            stop = terminated or truncated
        if epoch % 100 == 0: 
            # print("Done epoch " + str(epoch))
            # print("Win ration is " + str(win_count/100))
            win_ratio.append(win_count/100)
            win_count = 0
    return Q_sa

# # Parameters
# running_agent = 10000
# show = (running_agent == 1) and True # We only show if wanted and there is only one iteration
# epsilon = 0.1
# alpha = 0.1
# gamma = 0.99

# # Statistics
# win_count = 0
# win_ratio = []
# truncated_total = 0

# # Generic Initialisation
# Q_sa = QMatrix(
#     np.zeros((environment_space_length, action_space_length)),
#     epsilon,
#     gamma, 
#     alpha)

# for epoch in range(running_agent):
#     stop = False
#     environment.reset()
#     SARSA_agent = QAgent(Q_sa, initial_state_index = 0, algo = Q_algo.SARSA)
#     current_state = SARSA_agent.current_state_index
#     action = SARSA_agent.pickNextAction(current_state)
#     while not stop:
#         '''
#         While Agent is not stopped
#         Agent perform the action
#         Agent pick an action from its state and its policy
#         Agent update its trajectory: action taken, new state, new reward
#         Update the stop condition if Goal reached or terminated
#         '''
#         observation, reward, terminated, truncated, info = environment.step(action)
#         next_action = SARSA_agent.pickNextAction(observation)
#         SARSA_agent.current_state_index = observation
#         # qsa_footprint = np.sum(Q_sa.value)      
#         Q_sa.update_SARSA(current_state, action, observation, next_action, reward)
#         # if qsa_footprint != np.sum(Q_sa.value): displayGame(environment)
#         current_state = observation
#         action = next_action
#         if reward == 1: win_count += 1
#         stop = terminated or truncated
#     if epoch % 100 == 0: 
#         # print("Done epoch " + str(epoch))
#         # print("Win ration is " + str(win_count/100))
#         win_ratio.append(win_count/100)
#         win_count = 0

# print(Q_sa.value)
# %matplotlib inline
# plt.plot(win_ratio)
# half = int(len(win_ratio) / 2)
# # Get the mean only after the second half, to account for the warmup
# print("Mean success on second half: " + str(np.mean(win_ratio[- half:])))
# print("Mean truncated on all episode: " + str(truncated_total/running_agent))


# Test with the optimal policy

show = True

# deterministic_policy = np.zeros(((environment_space_length, action_space_length)))
# for state_index in range(Q_sa.value.shape[0]):
#     action_max = np.argmax(Q_sa.value[state_index])
#     deterministic_policy[state_index, action_max] = 1.0
Q_sa = SARSA(environment, epoch_number = 10000, epsilon = 0.1, alpha = 0.1, gamma = 0.99)
deterministic_policy = Policy.buildOptimalPolicyFrom(Q_sa.value)

policy_agent = PolicyAgent(deterministic_policy, initial_state_index = 0)

from src.Functions.Run import FrozenLake_parameters, run

# frozenLake_parameters = {'desc' : ["SFFF", "FHFH", "FFFH", "HFFG"], 'is_slippery'  : True}
frozenLake_parameters: FrozenLake_parameters = {'desc' : ["SFFF", "FHFH", "FFFH", "HFFG"], 'is_slippery'  : True}

run(frozenLake_parameters, policy_agent)

# stop = False
# environment.reset()
# current_state = policy_agent.current_state_index
# if show: displayGame(environment)
# while not stop:
#     '''
#     While Agent is not stopped
#     Agent perform the action
#     Agent pick an action from its state and its policy
#     Agent update its trajectory: action taken, new state, new reward
#     Update the stop condition if Goal reached or terminated
#     '''
#     next_action_index = policy_agent.pickNextAction()
#     observation, reward, terminated, truncated, info = environment.step(next_action_index)
#     if show: displayGame(environment, next_action_index)
#     policy_agent.nextStep(observation, next_action_index, float(reward))
#     stop = terminated or truncated