import gymnasium as gym
from gymnasium.utils.play import play
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from PIL import Image
import numpy as np
from IPython.display import display # to display images
import matplotlib.pyplot as plt
from typing import TypeVar, List, Tuple

# Utility functions

def displayGame(environment):
    rgb_array = environment.render()
    image = Image.fromarray(rgb_array)
    display(image)
    pass

desc=["SFFF", "FHFH", "FFFH", "HFFG"] # Same as the map called "4*4"
dimension_desc = (4, 4)
nrow = 4
ncol = 4
environment = gym.make('FrozenLake-v1', desc=desc, is_slippery=True, render_mode="rgb_array")

# random_policy = np.array()

# Get the action space dimension
# See https://stackoverflow.com/a/62306159
# In our case, it's a standard discrete space
action_space_length: int = environment.action_space.n

# Get the observation space
environment_space_length: int = environment.observation_space.n

# Build our random policy

from src.Policy import Policy
from src.PolicyAgent import PolicyAgent
random_policy_object = Policy(np.full((environment_space_length, action_space_length), 1 / action_space_length))

# Parameters
running_agent = 50000
show = (running_agent == 1) and True # We only show if wanted and there is only one iteration


# Statistics
win_count = 0
win_ratio = []
truncated_total = 0

# Initialisation



for epoch in range(running_agent):
    stop = False
    environment.reset()
    random_agent = PolicyAgent(random_policy_object, initial_state_index = 0)

    if show: displayGame(environment)
    while not stop:
        '''
        While Agent is not stopped
        Agent pick an action from its state and its policy
        Agent perform the action
        Agent update its trajectory: action taken, new state, new reward
        Update the stop condition if Goal reached or terminated
        '''
        next_action_index = random_agent.pickNextAction()
        observation, reward, terminated, truncated, info = environment.step(next_action_index)
        random_agent.nextStep(observation, next_action_index, float(reward))
        stop = terminated or truncated
        if reward == 1: win_count += 1
        stop = terminated or truncated
    if epoch % 100 == 0: 
        print("Done epoch " + str(epoch))
        print("Win ration is " + str(win_count/100))
        win_ratio.append(win_count/100)
        win_count = 0


%matplotlib inline
plt.plot(win_ratio)
half = int(len(win_ratio) / 2)
# Get the mean only after the second half, to account for the warmup
print("Mean success on second half: " + str(np.mean(win_ratio[- half:])))
print("Mean truncated on all episode: " + str(truncated_total/running_agent))
