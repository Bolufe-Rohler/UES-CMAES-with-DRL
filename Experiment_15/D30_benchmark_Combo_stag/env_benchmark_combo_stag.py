from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any

import tensorflow as tf
import numpy as np
import random
import math
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
from ues_cmaes_X import ues_cmaes_X
import functions


class Env_benchmark_combo_stag(py_environment.PyEnvironment):
    def __init__(self, func_num, dim, minimum, median_error):
        super().__init__()

        # Action & Observation Specs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=11, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(65,), dtype=np.float64, name='observation')

        # Environment Settings
        self._fun_num = func_num
        self._dim = dim
        self._cec_benchmark = functions.CEC_functions(dim)
        self._max_episodes = 10
        self._max_evals = dim * 1000
        self._state_size = 20
        self._episode_ended = False
        self._actions_count = 0

        # Track Best Fitness
        self._best_fitness = None

        # (Optional) References, retained if needed elsewhere
        self._minimum = minimum
        self._median_error = median_error

        # Stagnation Tracking
        self._stagnation_count = 0
        self._tau = 3  # Number of consecutive non-improving steps allowed
        self._gamma = 0.1  # Penalty applied once stagnation_count >= tau

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # Reset counters and flags
        self._actions_count = 0
        self._episode_ended = False
        self._best_fitness = None
        self._stagnation_count = 0

        # (Optional) Randomize the function each reset
        self._fun_num = random.randint(1, 28)

        # Run one initial UES-CMAES call (arbitrary initial decisions)
        init_decision = {
            'FE': 0.9, 'range': 1, 'gamma': 1,
            'sigma': 0.1, 'alpha': 0.1, 'cma_pop': 15, 'iters': 30
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

        # The agent's first "reward" will be handled in _step.
        return ts.restart(self._observation)

    def _step(self, action):
        if self._episode_ended:
            # If the episode is over, restart
            return self.reset()

        # Count the step; episode ends if we hit max_episodes
        self._actions_count += 1
        if self._actions_count == self._max_episodes:
            self._episode_ended = True

        # Map action to decisions
        decisions = self._map_action_to_decisions(action)

        # Run UES-CMAES from the last best state
        self._state, self._observation, best_f = ues_cmaes_X(
            self.obj_function,
            dim=self._dim,
            max_eval=self._max_evals,
            bound=100,
            state_size=self._state_size,
            start_point=self._state[self._state_size - 1],
            decisions=decisions
        )

        # -----------------------------
        # Compute the Reward
        # -----------------------------

        # 1) If this is the very first call to _step for the episode:
        if self._actions_count == 1 and self._best_fitness is None:
            self._best_fitness = best_f
            reward = -best_f  # negativity of error
            self._stagnation_count = 0

        # 2) If we have a stored best_fitness, check for improvement:
        elif best_f < self._best_fitness:
            # If improved, reward = (old_best - new_best)
            improvement = self._best_fitness - best_f
            # Apply stagnation penalty if needed
            if self._stagnation_count >= self._tau:
                reward = improvement - self._gamma
            else:
                reward = improvement

            # Update best and reset stagnation
            self._best_fitness = best_f
            self._stagnation_count = 0

        # 3) No improvement
        else:
            reward = 0.0
            self._stagnation_count += 1

        # Check for invalid reward
        if math.isinf(reward) or math.isnan(reward):
            reward = 0.0

        # End or Transition
        if self._episode_ended:
            return ts.termination(self._observation, reward)
        else:
            return ts.transition(self._observation, reward, discount=1.0)

    def obj_function(self, X):
        """Wrapper for the chosen CEC function."""
        if len(X.shape) > 1:
            return self._cec_benchmark.Y_matrix(X, self._fun_num)
        else:
            return self._cec_benchmark.Y(X, self._fun_num)

    def _map_action_to_decisions(self, action):
        """Maps discrete actions [0..11] to UES-CMAES parameter combos."""
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

    # Not strictly required by tf_agents, left as placeholders
    def get_info(self) -> types.NestedArray:
        pass

    def get_state(self) -> Any:
        pass

    def set_state(self, state: Any) -> None:
        pass
