import gymnasium as gym
from gymnasium.utils.play import play
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from PIL import Image
import numpy as np
from IPython.display import display # to display images

# from nptyping import NDArray, Shape, Int
from typing import TypeVar, List, Tuple
# https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
# map_name "4*4"
# "4x4":[
#     "SFFF",
#     "FHFH",
#     "FFFH",
#     "HFFG"
#     ]
# Actions
# 0: LEFT
# 1: DOWN
# 2: RIGHT
# 3: UP

environment = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="rgb_array")
# Playing this interesting game, buggy
# mapping = {
#     (ord('q'),): 0, # LEFT
#     (ord('s'),): 1, # DOWN
#     (ord('d'),): 2, # RIGHT
#     (ord('z'),): 3  # UP
#     }
# play(environment, keys_to_action=mapping)

# Utility functions
# The coordinate (0, 0) correspond to the top left start cell and the goal to the bottom right
# It follow the same convention as the observation space

def check(row_index: int, col_index: int) -> None:
    assert 0 <= row_index < nrow
    assert 0 <= col_index < ncol

def stateToCoordinate(index: int) -> Tuple[int, int]:
    row_index = index // ncol
    col_index = index % ncol
    check(row_index, col_index)
    return (row_index, col_index)


def coordinateToIndex(row_index: int, col_index: int) -> int:
    check(row_index, col_index)
    #! There is a typo in the doc, the agent current position is at current_row * ncols + current_col
    return row_index * ncol + col_index

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
environment.reset()
rgb_array = environment.render()
Image.fromarray(rgb_array)

action = 0
observation, reward, terminated, truncated, info = environment.step(action)
rgb_array = environment.render()
Image.fromarray(rgb_array)

# random_policy = np.array()

# Get the action space dimension
# See https://stackoverflow.com/a/62306159
# In our case, it's a standard discrete space
action_space_length: int = environment.action_space.n

# Get the observation space
environment_space_length: int = environment.observation_space.n

# Build our random policy

def possible_action(row_index: int, col_index: int) -> List[bool]:
    check(row_index, col_index)
    possible_action = [True, True, True, True]
    # Actions
    # 0: LEFT
    # 1: DOWN
    # 2: RIGHT
    # 3: UP

    if (row_index == 0):
        possible_action[3] = False
    elif (row_index == nrow - 1):
        possible_action[1] = False
    else:
        pass

    if (col_index == 0):
        possible_action[0] = False
    elif (col_index == ncol - 1):
        possible_action[2] = False
    else:
        pass

    return(possible_action)

random_policy = np.zeros((environment_space_length, action_space_length))

for row_index in range(nrow):
    for col_index in range(ncol):
        # https://stackoverflow.com/a/18713494
        # We use np.array to allow addition and division element wise / scalar division / boolean cast to int
        # And speed, which may be required at some point
        valid_actions = np.array(possible_action(row_index, col_index))
        random_policy[coordinateToIndex(row_index, col_index)] = valid_actions / sum(valid_actions)


from src.Classes.Policy import Policy
from src.Classes.Agent import Agent
random_policy_object = Policy(random_policy)


# Initialisation

random_agent = Agent(random_policy_object, initial_state_index = 0)

stop = False
environment.reset()
displayGame(environment)

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
    # Render the game
    displayGame(environment)


