import numpy as np
from enum import Enum
from src.QMatrix import QMatrix
from src.Agent import Agent


class Q_algo(Enum):
    SARSA = 'SARSA'
    Qlearning = 'Qlearning'

class QAgent(Agent):

    def __init__(self, q_matrix: QMatrix, algo: Q_algo, initial_state_index: int = 0):
        super().__init__( initial_state_index )
        self.q_matrix = q_matrix
        self.algo = algo
        self.ongoing_action = None
        pass

    def update(self, ongoing_action, next_state, reward):
        self.current_state_index = next_state
        match self.algo:
            case Q_algo.SARSA:
                self.q_matrix.update_SARSA(self.current_state_index, ongoing_action, next_state, reward)
            case Q_algo.Qlearning:
                self.q_matrix.update_Qlearning(self.current_state_index, ongoing_action, next_state, reward)


    def pickNextAction(self, current_state_index):
        self.ongoing_action = self.q_matrix.pickNext(current_state_index)
        return self.ongoing_action

    def nextStep(self, state_index: int, action_index: int, reward:float = 0):
        self.trajectory.append(state_index, action_index, reward)
        self.current_state_index = state_index
        pass
    
