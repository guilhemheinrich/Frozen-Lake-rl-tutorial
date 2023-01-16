from typing import TypeVar, Generic, List

import numpy as np

State = TypeVar('State') 
Action = TypeVar('Action')
class Step(object):
    def __init__(self, state_index: int, action_index: int, reward:float = 0):
        self.state = state_index
        self.action = action_index
        self.reward = reward
        pass

class Trajectory(Generic[State, Action]):
    def __init__(self):
        self.trajectory: List[Step] = []
        # self.trajectory.append(Step(state_index, action_index))

    def append(self, state_index: int, action_index: int, reward: float = 0):
        self.trajectory.append(Step(state_index, action_index, reward))
    
    # def append(self, step):
    #     self.trajectory.append(step)
