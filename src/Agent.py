import numpy as np
from src.Trajectory import Trajectory
from src.Policy import Policy
from src.Policy import Policy
   
class Agent(object):

    def __init__(self, policy: Policy, initial_state_index: int = 0 ):
        self.policy = policy
        self.trajectory: Trajectory = Trajectory()
        self.current_state_index = initial_state_index
        pass

    def pickNextAction(self):
        return self.policy.pickNext(self.current_state_index)

    def nextStep(self, state_index: int, action_index: int, reward:float = 0):
        self.trajectory.append(state_index, action_index, reward)
        self.current_state_index = state_index
        pass
    
