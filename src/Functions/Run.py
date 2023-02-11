import gymnasium as gym
from gymnasium.utils.play import play
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from PIL import Image
import numpy as np
from IPython.display import display # to display images
import matplotlib.pyplot as plt

from typing import NamedTuple, Protocol, List, TypedDict
from src.Classes.Policy import Policy
from src.Classes.PolicyAgent import PolicyAgent
from src.Classes.QAgent import QAgent

class FrozenLake_parameters(TypedDict):
    desc: List[str]
    is_slippery: bool

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

def run(frozenLake_parameters: FrozenLake_parameters, agent: PolicyAgent):
    environment = gym.make('FrozenLake-v1', desc=frozenLake_parameters['desc'], is_slippery=frozenLake_parameters['is_slippery'], render_mode="rgb_array")
    stop = False
    environment.reset()
    displayGame(environment)
    while not stop:
        '''
        While Agent is not stopped
        Agent perform the action
        Agent pick an action from its state and its policy
        Agent update its trajectory: action taken, new state, new reward
        Update the stop condition if Goal reached or terminated
        '''
        next_action_index = agent.pickNextAction()
        observation, reward, terminated, truncated, info = environment.step(next_action_index)
        displayGame(environment, next_action_index)
        agent.nextStep(observation, next_action_index, float(reward))
        stop = terminated or truncated