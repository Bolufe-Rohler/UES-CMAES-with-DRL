from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any
import math
import random

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from ues_cmaes_X import ues_cmaes_X
import functions


class EnvBenchmarkComboEval(py_environment.PyEnvironment):
    """
    Evaluation environment for FinalCombo + standard reward / stagnation reward setup.

    IMPORTANT:
    - Unlike the old benchmark env, this one DOES NOT randomize the function in _reset().
    - It keeps the function fixed so each benchmark function can be evaluated properly.
    """

    def __init__(self, func_num, dim, minimum, median_error, state_size=20,
                 episodes=10, evals_per_restart=None):
        super().__init__()

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=11, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(65,), dtype=np.float64, name='observation'
        )

        self._fun_num = func_num
        self._dim = dim
        self._cec_benchmark = functions.CEC_functions(dim)

        self._max_episodes = episodes
        self._max_evals = evals_per_restart if evals_per_restart is not None else dim * 1000
        self._state_size = state_size

        self._episode_ended = False
        self._actions_count = 0
        self._best_fitness = None
        self._state = None
        self._observation = None

        self._minimum = minimum
        self._median_error = median_error

        self._stagnation_count = 0
        self._tau = 3
        self._gamma = 0.1

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._actions_count = 0
        self._episode_ended = False
        self._best_fitness = None
        self._stagnation_count = 0

        init_decision = {
            'FE': 0.9,
            'range': 1,
            'gamma': 1,
            'sigma': 0.1,
            'alpha': 0.1,
            'cma_pop': 15,
            'iters': 30
        }

        self._state, self._observation, best_f = ues_cmaes_X(
            self.obj_function,
            dim=self._dim,
            max_eval=self._max_evals,
            bound=100,
            state_size=self._state_size,
            start_point=None,
            decisions=init_decision
        )

        return ts.restart(self._observation)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self._actions_count += 1
        if self._actions_count == self._max_episodes:
            self._episode_ended = True

        decisions = self._map_action_to_decisions(int(action))

        self._state, self._observation, best_f = ues_cmaes_X(
            self.obj_function,
            dim=self._dim,
            max_eval=self._max_evals,
            bound=100,
            state_size=self._state_size,
            start_point=self._state[self._state_size - 1],
            decisions=decisions
        )

        if self._actions_count == 1 and self._best_fitness is None:
            self._best_fitness = best_f
            reward = -best_f
            self._stagnation_count = 0

        elif best_f < self._best_fitness:
            improvement = self._best_fitness - best_f
            if self._stagnation_count >= self._tau:
                reward = improvement - self._gamma
            else:
                reward = improvement

            self._best_fitness = best_f
            self._stagnation_count = 0

        else:
            reward = 0.0
            self._stagnation_count += 1

        if math.isinf(reward) or math.isnan(reward):
            reward = 0.0

        if self._episode_ended:
            return ts.termination(self._observation, reward)
        else:
            return ts.transition(self._observation, reward, discount=1.0)

    def obj_function(self, X):
        if len(X.shape) > 1:
            return self._cec_benchmark.Y_matrix(X, self._fun_num)
        else:
            return self._cec_benchmark.Y(X, self._fun_num)

    def _map_action_to_decisions(self, action):
        if action == 0:
            return {'FE': 0.9, 'range': 1, 'gamma': 1, 'sigma': 0.1, 'alpha': 0.1, 'cma_pop': 15, 'iters': 30}
        elif action == 1:
            return {'FE': 0.9, 'range': 0, 'gamma': 2, 'sigma': 10, 'alpha': 0.1, 'cma_pop': 30, 'iters': 30}
        elif action == 2:
            return {'FE': 0.9, 'range': 2, 'gamma': 2, 'sigma': 0.1, 'alpha': 0.05, 'cma_pop': 15, 'iters': 40}
        elif action == 3:
            return {'FE': 0.5, 'range': 3, 'gamma': 1, 'sigma': 10, 'alpha': 0.1, 'cma_pop': 45, 'iters': 50}
        elif action == 4:
            return {'FE': 0.5, 'range': 0, 'gamma': 2, 'sigma': 10, 'alpha': 0.1, 'cma_pop': 15, 'iters': 30}
        elif action == 5:
            return {'FE': 0.9, 'range': 0, 'gamma': 3, 'sigma': 10, 'alpha': 0.1, 'cma_pop': 15, 'iters': 30}
        elif action == 6:
            return {'FE': 0.9, 'range': 0, 'gamma': 1, 'sigma': 1, 'alpha': 0.1, 'cma_pop': 30, 'iters': 40}
        elif action == 7:
            return {'FE': 0.9, 'range': 0, 'gamma': 3, 'sigma': 0.1, 'alpha': 0.1, 'cma_pop': 15, 'iters': 40}
        elif action == 8:
            return {'FE': 0.5, 'range': 4, 'gamma': 1, 'sigma': 10, 'alpha': 0.1, 'cma_pop': 30, 'iters': 30}
        elif action == 9:
            return {'FE': 0.5, 'range': 4, 'gamma': 2, 'sigma': 1, 'alpha': 0.1, 'cma_pop': 15, 'iters': 40}
        elif action == 10:
            return {'FE': 0.5, 'range': 1, 'gamma': 1, 'sigma': 1, 'alpha': 0.1, 'cma_pop': 45, 'iters': 50}
        else:
            return {'FE': 0.5, 'range': 2, 'gamma': 2, 'sigma': 10, 'alpha': 0.05, 'cma_pop': 15, 'iters': 30}

    def get_info(self) -> types.NestedArray:
        return {}

    def get_state(self) -> Any:
        return None

    def set_state(self, state: Any) -> None:
        pass