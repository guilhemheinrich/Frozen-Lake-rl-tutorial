from nptyping import NDArray, Shape, Int, Float
import random
import numpy as np
# from typing import TypeVar, Generic
# StateDimension = TypeVar('StateDimension')
# ActionDimension = TypeVar('ActionDimension')

# TODO ajouter la epsilon greedy policy
class Policy(object):

    def __init__(self, policy_array: NDArray[Shape['StateDimension, ActionDimension'], Float]):
        self.policy_array = policy_array
        self.state_dimension = policy_array.shape[0]
        self.action_dimension = policy_array.shape[1]
        pass

    def pickNext(self, state_index: int)-> int:
        assert state_index >= 0 < self.state_dimension
        rand = random.random()
        action_subset = self.policy_array[state_index, ]
        # See https://stackoverflow.com/questions/16243955/numpy-first-occurrence-of-value-greater-than-existing-value
        return(int(np.searchsorted(np.cumsum(action_subset), rand)))
    
    @staticmethod
    def buildOptimalPolicyFrom(Q_sa: NDArray[Shape['StateDimension, ActionDimension'], Float]):
        state_dimension = Q_sa.shape[0]
        action_dimension = Q_sa.shape[1]
        deterministic_policy = np.zeros(((state_dimension, action_dimension)))
        for state_index in range(Q_sa.shape[0]):
            action_max = np.argmax(Q_sa[state_index])
            deterministic_policy[state_index, action_max] = 1.0
        return Policy(deterministic_policy)
        



