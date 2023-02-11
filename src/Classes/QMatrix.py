from nptyping import NDArray, Shape, Int, Float
import random
import numpy as np

# TODO déplacer la logique de sélection (pickNext) dans la class Policy
# TODO i.e epsilon n'a rien à faire la

# TODO Sous classer la QMatrix, et son update, en SARSA, Q-Learning et MC
class QMatrix(object):

    def __init__(self, initial_qmatrix = np.zeros((10, 3), float), epsilon = 0.2, gamma = 0.9, alpha = 0.7):
        self.value = initial_qmatrix
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.state_dimension = initial_qmatrix.shape[0]
        self.action_dimension = initial_qmatrix.shape[1]

    # TODO déplacer la logique de sélection (pickNext) dans la class Policy     
    def pickNext(self, state_index: int)-> int:
        assert state_index >= 0 < self.state_dimension
        rand = random.random() # pick a number in [0, 1)
        if (rand < self.epsilon):
            # pick uniformly between all action with probability epsilon
            return(random.randint(0, self.action_dimension - 1))
        else:
            # pick uniformly between all action with maximum Qvalue with probability 1 - epsilon
            action_state_value = self.value[state_index, ]
            maximum_value = max(action_state_value)        
            possible_action_index = [i for i, j in enumerate(action_state_value) if j == maximum_value]
            random_between_maximum_value = random.randint(0, len(possible_action_index) - 1)
            action = possible_action_index[random_between_maximum_value]
            return(action)

    def update_SARSA(self, state_index: int, action_index: int, next_state: int, next_action: int,  reward:float = 0.0):
        # Q[s, a] := Q[s, a] + α[r + γQ(s', a') - Q(s, a)]
        self.value[state_index, action_index] = self.value[state_index, action_index] + self.alpha * (reward + self.gamma * self.value[next_state, next_action] - self.value[state_index, action_index])


    def update_Qlearning(self, state_index, action_index, next_state, reward: float = 0):
        # Q[s, a] := Q[s, a] + α[r + γ . argmax_a {Q(s', a')} - Q(s, a)]
        next_action = np.argmax(self.value[state_index, ])
        self.value[state_index, action_index] = self.value[state_index, action_index] + self.alpha * (reward + self.gamma * self.value[next_state, next_action] - self.value[state_index, action_index])

