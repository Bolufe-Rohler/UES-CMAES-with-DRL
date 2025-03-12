from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any

import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
import cma
from ues_cmaes_rl import ues_cmaes_rl
import functions


class Env_Fit100_Act100(py_environment.PyEnvironment):
    def __init__(self, func_num, dim, minimum):
        super().__init__()  # supposedly not needed
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=9, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(100,), dtype=np.float64, name='observation')
        # self._state = 0
        self._actions_count = 0
        self._episode_ended = False
        self._fun_num = func_num
        self._dim = dim
        self._cec_benchmark = functions.CEC_functions(dim)
        self._max_episodes = 10
        self._max_evals = dim * 1000  # per episode, with 10 episodes it is double as many FEs as usual
        self._best_fitness = None
        self._minimum = minimum

    def obj_function(self, X):
        if len(X.shape) > 1:
            return self._cec_benchmark.Y_matrix(X, self._fun_num)
        else:
            return self._cec_benchmark.Y(X, self._fun_num)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._actions_count = 0
        self._episode_ended = False
        # Call ues_cmaes from random start
        # save current state and observations (state are the actual solution vectors)
        # observations are their fitness
        self._state, self._observation, best_f = ues_cmaes_rl(self.obj_function, dim=self._dim, max_eval=self._max_evals,
                                                      bound=100, state_size=100, start_point=None)
        self._best_fitness = None
        return ts.restart(self._observation)
        # np.array([self._observation], dtype=np.float32))  ##### WHAT THIS? ...this is initial observations

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        self._actions_count += 1
        if self._actions_count == self._max_episodes:
            self._episode_ended = True


        adjusted_action = action*10
        self._state, self._observation, best_f = ues_cmaes_rl(self.obj_function, dim=self._dim,
                                                              max_eval=self._max_evals, bound=100, state_size=100,
                                                              start_point=self._state[adjusted_action])
        if self._best_fitness is None:
            reward = self._minimum - best_f
            self._best_fitness = best_f
        else:
            reward = max(self._best_fitness - best_f, 0)  # no penalty in reward
            self._best_fitness = min(self._best_fitness, best_f)

        if self._episode_ended:
            return ts.termination(self._observation, reward)
        else:
            return ts.transition(self._observation, reward, discount=1.0)

    # supposedly not needed
    def get_info(self) -> types.NestedArray:
        pass

    def get_state(self) -> Any:
        pass

    def set_state(self, state: Any) -> None:
        pass
