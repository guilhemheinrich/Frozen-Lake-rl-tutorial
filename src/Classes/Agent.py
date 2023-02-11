import numpy as np
from abc import ABC, abstractmethod
from src.Classes.Trajectory import Trajectory
from src.Classes.Policy import Policy

   
class Agent(ABC):

    def __init__(self, initial_state_index: int = 0 ):
        self.trajectory: Trajectory = Trajectory()
        self.current_state_index = initial_state_index
        pass
    
    @abstractmethod
    def pickNextAction(self):
        pass

    def nextStep(self, state_index: int, action_index: int, reward:float = 0):
        self.trajectory.append(state_index, action_index, reward)
        self.current_state_index = state_index
        pass
    
