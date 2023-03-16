import gymnasium as gym
import numpy as np

from src.V2.Classes.Policy import Policy
from src.V2.Classes.Agent import Agent
from src.V2.Functions.epsilon_greedy_policy_factory import make_epsilon_greedy_policy

desc=["SFFF", "FHFH", "FFFH", "HFFG"] # Same as the map called "4*4"
environment = gym.make('FrozenLake-v1', desc=desc, is_slippery=True, render_mode="rgb_array")

def Q_learning(environment, epsilon = 0.1, alpha = 0.1, gamma = 0.99, epoch_number = 8000):
    # Get the observation space & the action space
    environment_space_length: int = environment.observation_space.n # type: ignore
    action_space_length: int = environment.action_space.n # type: ignore
    Q_sa = np.zeros((environment_space_length, action_space_length))

    epsilon_greedy_policy = Policy(
        make_epsilon_greedy_policy(epsilon = epsilon, Q_sa = Q_sa),
        environment_space_length,
        action_space_length
    )
    def update_Qlearning(policy: Policy, state_index, action_index, next_state, reward: float = 0):
        # Q[s, a] := Q[s, a] + α[r + γ . argmax_a {Q(s', a')} - Q(s, a)]
        best_next_action = np.argmax(Q_sa[state_index, ])
        Q_sa[state_index, action_index] = Q_sa[state_index, action_index] + alpha * (reward + gamma * Q_sa[next_state, best_next_action] - Q_sa[state_index, action_index])



    for epoch in range(epoch_number):
        learning_agent = Agent(epsilon_greedy_policy, update_Qlearning)
        learning_agent.current_state_index = environment.reset()[0] # Initial state
        stop = False
        action = learning_agent.pick_next()
        while not stop:
            '''
            While Agent is not stopped
            Agent perform the action
            Agent pick an action from its state and its policy
            Agent update its trajectory: action taken, new state, new reward
            Update the stop condition if Goal reached or terminated
            '''
            observation, reward, terminated, truncated, info = environment.step(action)
            next_action_index = learning_agent.pick_next(observation)
            learning_agent.trajectory.append(observation, next_action_index, float(reward))
            learning_agent.update(
                learning_agent.current_state_index,
                action,
                observation,
                reward)
            learning_agent.current_state_index = observation
            action = next_action_index
            stop = terminated or truncated
    return Q_sa






