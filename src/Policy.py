from nptyping import NDArray, Shape, Int, Float
import random
import numpy as np
# from typing import TypeVar, Generic
# StateDimension = TypeVar('StateDimension')
# ActionDimension = TypeVar('ActionDimension')

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
        # print(np.searchsorted(np.cumsum(action_subset), rand))
        # print(int(np.argmax(rand < np.cumsum(action_subset))))
        return(int(np.searchsorted(np.cumsum(action_subset), rand)))



