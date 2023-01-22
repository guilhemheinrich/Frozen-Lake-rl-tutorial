import gymnasium as gym
from gymnasium.utils.play import play
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from PIL import Image
import numpy as np
from IPython.display import display # to display images

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


from src.Policy import Policy
from src.PolicyAgent import PolicyAgent
from src.Trajectory import Trajectory



# Parameters
running_agent = 5000
# Build our random policy
random_policy_object = Policy(np.full((environment_space_length, action_space_length), 1 / action_space_length))

# Observation List
trajectories: List[Trajectory] = []

# Generic Initialisation

for epoch in range(running_agent):
    stop = False
    environment.reset()
    random_agent = PolicyAgent(random_policy_object, initial_state_index = 0)
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
    # 
    trajectories.append(random_agent.trajectory)

# enriched_trajectories = [trajectory.enrich() for trajectory in trajectories]
enriched_trajectories = []
for trajectory in trajectories:
    enriched_trajectoriy = trajectory.enrich()
    enriched_trajectories.append(enriched_trajectoriy)

[print(trajectory.steps[0].reward) for trajectory in enriched_trajectories]
for trajectory in enriched_trajectories:
    reward_total = sum([step.reward for step in trajectory.steps])
    if reward_total > 1:
        # print(reward_total)
        pass


success_indexes: List[int] = []
index = 0
for trajectory in trajectories:
    if trajectory.steps[-1].reward == 1:
        success_indexes.append(index)
    index += 1

successful_trajectory = enriched_trajectories[success_indexes[1]]
print(str(trajectories[success_indexes[1]]))
print(str(successful_trajectory))


