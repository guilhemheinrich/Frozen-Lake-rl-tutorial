from nptyping import NDArray, Shape, Int, Float
import random
import numpy as np

class QMatrix_MC(object):

    def __init__(self, initial_qmatrix = np.zeros((10, 3), float)):
        self.value = initial_qmatrix
        self.state_dimension = initial_qmatrix.shape[0]
        self.action_dimension = initial_qmatrix.shape[1]
        self.incremental_counter = np.zeros((initial_qmatrix.shape), int)

    def increment(self, state, action, value):
        increment = self.incremental_counter[state, action]
        old_value = self.value[state, action]
        self.incremental_counter[state, action] += 1
        self.value[state, action]= (increment * old_value + value) / self.incremental_counter[state, action]

    


