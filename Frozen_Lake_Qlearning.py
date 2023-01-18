import gymnasium as gym
from gymnasium.utils.play import play
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from PIL import Image
import numpy as np
from IPython.display import display # to display images
import matplotlib.pyplot as plt

from typing import TypeVar, List, Tuple

def displayGame(environment):
    rgb_array = environment.render()
    image = Image.fromarray(rgb_array)
    display(image)

desc=["SFFF", "FHFH", "FFFH", "HFFG"] # Same as the map called "4*4"

# desc = generate_random_map(3)
environment = gym.make('FrozenLake-v1', desc=desc, is_slippery=True, render_mode="rgb_array")
# Get the action space dimension
# See https://stackoverflow.com/a/62306159
# In our case, it's a standard discrete space
action_space_length: int = environment.action_space.n

# Get the observation space
environment_space_length: int = environment.observation_space.n

from src.QMatrix import QMatrix
from src.QAgent import Q_algo, QAgent
from src.QAgent import QAgent

# Parameters
running_agent = 10000
show = (running_agent == 1) and True # We only show if wanted and there is only one iteration
epsilon = 0.1
alpha = 0.1
gamma = 0.99

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

for epoch in range(running_agent):
    stop = False
    initial_state, info = environment.reset()
    Qlearn_agent = QAgent(Q_sa, initial_state_index = initial_state, algo = Q_algo.Qlearning)
    current_state = Qlearn_agent.current_state_index
    if show: displayGame(environment)
    while not stop:
        '''
        While Agent is not stopped
        Agent perform the action
        Agent pick an action from its state and its policy
        Agent update its trajectory: action taken, new state, new reward
        Update the stop condition if Goal reached or terminated
        '''
        next_action = Qlearn_agent.pickNextAction(current_state)
        observation, reward, terminated, truncated, info = environment.step(next_action)
        new_reward = reward
        # if reward != 1 and terminated:
        #     new_reward = -1
        if show: displayGame(environment)
        # qsa_footprint = np.sum(Q_sa.value)      
        Q_sa.update_Qlearning(current_state, next_action, observation, new_reward)
        # if qsa_footprint != np.sum(Q_sa.value): displayGame(environment)
        current_state = observation
        if reward == 1: win_count += 1
        stop = terminated or truncated
        if truncated: truncated_total += 1
    if epoch % 100 == 0: 
        # print("Done epoch " + str(epoch))
        # print("Win ration is " + str(win_count/100))
        win_ratio.append(win_count/100)
        win_count = 0

displayGame(environment)
%matplotlib inline
plt.plot(win_ratio)
half = int(len(win_ratio) / 2)
# Get the mean only after the second half, to account for the warmup
print("Mean success on second half: " + str(np.mean(win_ratio[- half:])))
print("Mean truncated on all episode: " + str(truncated_total/running_agent))