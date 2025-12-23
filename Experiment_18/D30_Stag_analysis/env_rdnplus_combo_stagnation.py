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


class Env_Rdnplus_combo_stagnation(py_environment.PyEnvironment):
    def __init__(self, func_num=None, dim=30, tau=3, gamma=0.1):
        """
        func_num : int or None
            If None, a random CEC'13 function in [1, 28] is sampled at each reset().
            If an integer, that function index is used for all episodes (useful for per-function sensitivity).
        dim : int
            Problem dimension.
        tau : int
            Stagnation threshold (consecutive non-improving restarts before penalty kicks in).
        gamma : float
            Penalty applied when stagnation_count >= tau.
        """
        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=11, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(65,), dtype=np.float64, name='observation')

        # Environment configuration
        self._state_size = 20
        self._max_episodes = 10
        self._dim = dim
        self._cec_benchmark = functions.CEC_functions(dim)
        self._max_evals = dim * 1000  # per episode
        self._episode_ended = False
        self._actions_count = 0

        # Best fitness tracking
        self._best_fitness = None

        # Stagnation-related parameters
        self._stagnation_count = 0
        self._tau = tau       # Threshold of consecutive non-improving restarts
        self._gamma = gamma   # Penalty applied when stagnation_count >= tau

        # Function index (None → randomized in reset)
        self._fun_num = func_num

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._actions_count = 0
        self._episode_ended = False
        self._best_fitness = None
        self._stagnation_count = 0

        # Choose function:
        # - If func_num was given in __init__, keep it fixed.
        # - Otherwise, sample a random one in [1, 28].
        if self._fun_num is None:
            self._fun_num = random.randint(1, 28)

        # Initial decisions (arbitrary, just to start the environment)
        init_decision = {
            'FE': 0.9, 'range': 1, 'gamma': 1,
            'sigma': 0.1, 'alpha': 0.1, 'cma_pop': 15, 'iters': 30
        }

        # Run one round of UES-CMAES
        self._state, self._observation, best_f = ues_cmaes_X(
            self.obj_function,
            dim=self._dim,
            max_eval=self._max_evals,
            bound=100,
            state_size=self._state_size,
            start_point=None,
            decisions=init_decision
        )

        # First-step reward baseline = -error(best_fitness)
        self._best_fitness = best_f

        return ts.restart(self._observation)

    def _step(self, action):
        if self._episode_ended:
            # If the episode is already over, start a new one
            return self.reset()

        # Episode step count
        self._actions_count += 1
        if self._actions_count >= self._max_episodes:
            self._episode_ended = True

        # Map the chosen action to a parameter configuration
        decisions = self._map_action_to_decisions(action)

        # Run UES-CMAES from the last "best" state
        self._state, self._observation, best_f = ues_cmaes_X(
            self.obj_function,
            dim=self._dim,
            max_eval=self._max_evals,
            bound=100,
            state_size=self._state_size,
            start_point=self._state[self._state_size - 1],
            decisions=decisions
        )

        # ========================
        #    Compute Reward
        # ========================

        # 1) First action (episode start):
        if self._actions_count == 1:
            # Reward is just -best_f (matching the "if t=1" case).
            reward = -self._best_fitness

        # 2) Improvement case:
        elif best_f < self._best_fitness:
            # Check if stagnation penalty applies
            if self._stagnation_count >= self._tau:
                reward = (self._best_fitness - best_f) - self._gamma
            else:
                reward = self._best_fitness - best_f
            # Update best_fitness and reset stagnation
            self._best_fitness = best_f
            self._stagnation_count = 0

        # 3) No improvement:
        else:
            reward = 0.0
            self._stagnation_count += 1

        # Check for NaN or Inf in the reward
        if math.isinf(reward) or math.isnan(reward):
            reward = 0.0

        # If this was the last episode step, terminate; otherwise transition
        if self._episode_ended:
            return ts.termination(self._observation, reward)
        else:
            return ts.transition(self._observation, reward, discount=1.0)

    def obj_function(self, X):
        """Wrapper over the CEC benchmark function."""
        if len(X.shape) > 1:
            return self._cec_benchmark.Y_matrix(X, self._fun_num)
        else:
            return self._cec_benchmark.Y(X, self._fun_num)

    def _map_action_to_decisions(self, action):
        """Maps discrete action IDs [0..11] to UES-CMAES parameter sets."""
        if action == 0:
            decisions = {'FE': 0.9, 'range': 1, 'gamma': 1, 'sigma': 0.1, 'alpha': 0.1, 'cma_pop': 15, 'iters': 30}
        elif action == 1:
            decisions = {'FE': 0.9, 'range': 0, 'gamma': 2, 'sigma': 10, 'alpha': 0.1, 'cma_pop': 30, 'iters': 30}
        elif action == 2:
            decisions = {'FE': 0.9, 'range': 2, 'gamma': 2, 'sigma': 0.1, 'alpha': 0.05, 'cma_pop': 15, 'iters': 40}
        elif action == 3:
            decisions = {'FE': 0.5, 'range': 3, 'gamma': 1, 'sigma': 10, 'alpha': 0.1, 'cma_pop': 45, 'iters': 50}
        elif action == 4:
            decisions = {'FE': 0.5, 'range': 0, 'gamma': 2, 'sigma': 10, 'alpha': 0.1, 'cma_pop': 15, 'iters': 30}
        elif action == 5:
            decisions = {'FE': 0.9, 'range': 0, 'gamma': 3, 'sigma': 10, 'alpha': 0.1, 'cma_pop': 15, 'iters': 30}
        elif action == 6:
            decisions = {'FE': 0.9, 'range': 0, 'gamma': 1, 'sigma': 1, 'alpha': 0.1, 'cma_pop': 30, 'iters': 40}
        elif action == 7:
            decisions = {'FE': 0.9, 'range': 0, 'gamma': 3, 'sigma': 0.1, 'alpha': 0.1, 'cma_pop': 15, 'iters': 40}
        elif action == 8:
            decisions = {'FE': 0.5, 'range': 4, 'gamma': 1, 'sigma': 10, 'alpha': 0.1, 'cma_pop': 30, 'iters': 30}
        elif action == 9:
            decisions = {'FE': 0.5, 'range': 4, 'gamma': 2, 'sigma': 1, 'alpha': 0.1, 'cma_pop': 15, 'iters': 40}
        elif action == 10:
            decisions = {'FE': 0.5, 'range': 1, 'gamma': 1, 'sigma': 1, 'alpha': 0.1, 'cma_pop': 45, 'iters': 50}
        else:
            decisions = {'FE': 0.5, 'range': 2, 'gamma': 2, 'sigma': 10, 'alpha': 0.05, 'cma_pop': 15, 'iters': 30}

        return decisions

    # Not strictly required by tf_agents, left as placeholders
    def get_info(self) -> types.NestedArray:
        pass

    def get_state(self) -> Any:
        pass

    def set_state(self, state: Any) -> None:
        pass
