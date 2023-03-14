from __future__ import annotations # See https://stackoverflow.com/a/42845998
from typing import Any, Callable, Concatenate, Generic, ParamSpec, Protocol
import numpy as np
from abc import ABC, abstractmethod
from src.Classes_v2.Trajectory import Trajectory
from src.Classes_v2.Policy import Policy

# See https://github.com/microsoft/pyright/issues/3482
P = ParamSpec('P')
class Agent(Generic[P]):
    def __init__(self, policy: Policy, update_method: Callable[Concatenate[Policy, P], None], initial_state_index: int = 0):
        self.trajectory: Trajectory = Trajectory()
        self.current_state_index = initial_state_index
        self.policy = policy
        self._update_method = update_method
        pass

    def pick_next(self, state_index: int):
        return self.policy.pick_next(state_index)

    def append(self, state_index: int, action_index: int, reward:float = 0):
        self.trajectory.append(state_index, action_index, reward)
        self.current_state_index = state_index

    def update(self, *args: P.args, **kwargs: P.kwargs):
        return self._update_method(self.policy, *args, **kwargs)
