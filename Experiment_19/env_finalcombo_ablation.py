from __future__ import absolute_import, division, print_function

from typing import Any, Dict, Optional
import math
import random

import numpy as np
import tensorflow as tf  

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from ues_cmaes_X import ues_cmaes_X
import functions


class Env_FinalCombo_Ablation(py_environment.PyEnvironment):
    """
    A unified FinalCombo environment that can reproduce:
      - Exp 10 "standard reward" behavior
      - Exp 12 "stagnation-aware reward" behavior

    Plus, it supports ablations:
      - state_size (number of checkpoints) -> affects observation dimension (3*state_size + 5)
      - optional penalty for non-improving restarts

    Notes:
      - Action space: 12 discrete actions [0..11] mapping to parameter tuples (FinalCombo)
      - Episode length: 10 restarts (steps)
      - Each reset chooses a random function index in [1..28] (same as your previous envs)
    """

    def __init__(
        self,
        dim: int,
        minimums: Optional[list] = None,
        func_num: int = 1,  # kept for compatibility; env will randomize in reset()
        state_size: int = 20,
        include_single_run_features: bool = True,
        reward_mode: str = "standard",  # "standard" | "stagnation"
        tau: int = 3,
        stagnation_penalty: float = 0.1,  # (your "gamma" in the stagnation reward)
        penalty_lambda: float = 0.0,  # penalty for non-improvement steps (0 disables)
        randomize_function_each_episode: bool = True,
        max_episodes: int = 10,
        seed: Optional[int] = None,
    ):
        super().__init__()

        if reward_mode not in ("standard", "stagnation"):
            raise ValueError("reward_mode must be 'standard' or 'stagnation'")

        self._dim = dim
        self._cec_benchmark = functions.CEC_functions(dim)

        self._randomize_function_each_episode = randomize_function_each_episode
        self._fun_num = func_num

        # Observation engineering configuration
        self._state_size = int(state_size)
        self._include_single_run_features = bool(include_single_run_features)

        self._base_obs_dim = 3 * self._state_size
        self._single_obs_dim = 5 if self._include_single_run_features else 0
        self._obs_dim = self._base_obs_dim + self._single_obs_dim

        # Environment horizon / evaluations
        self._max_episodes = int(max_episodes)
        self._max_evals = self._dim * 1000  # per step, consistent with your existing code

        # Reward configuration
        self._reward_mode = reward_mode
        self._minimums = minimums  # needed for "standard" reward first-step scaling
        self._minimum = None

        # Stagnation mechanism (only used if reward_mode == "stagnation")
        self._stagnation_count = 0
        self._tau = int(tau)
        self._stagnation_penalty = float(stagnation_penalty)

        # Optional penalty for no-improvement steps (addresses reviewer point about "ineffective restarts")
        self._penalty_lambda = float(penalty_lambda)

        # Episode bookkeeping
        self._actions_count = 0
        self._episode_ended = False
        self._best_fitness = None

        # Seeding
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # TF-Agents specs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=11, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._obs_dim,), dtype=np.float64, name="observation"
        )

        # State/obs placeholders (filled on reset/step)
        self._state = None
        self._observation = None

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def obj_function(self, X):
        if len(X.shape) > 1:
            return self._cec_benchmark.Y_matrix(X, self._fun_num)
        return self._cec_benchmark.Y(X, self._fun_num)

    def _reset(self):
        self._actions_count = 0
        self._episode_ended = False
        self._best_fitness = None

        # Reset stagnation bookkeeping
        self._stagnation_count = 0

        # Randomize benchmark function per episode (your original behavior)
        if self._randomize_function_each_episode:
            self._fun_num = random.randint(1, 28)

        # For standard reward, we need a function minimum (your fDeltas list)
        if self._reward_mode == "standard":
            if self._minimums is None:
                raise ValueError("minimums list is required when reward_mode='standard'")
            self._minimum = self._minimums[self._fun_num - 1]

        # Initial decision (same as Exp10/Exp12)
        init_decision = {
            "FE": 0.9,
            "range": 1,
            "gamma": 1,
            "sigma": 0.1,
            "alpha": 0.1,
            "cma_pop": 15,
            "iters": 30,
        }

        self._state, self._observation, best_f = ues_cmaes_X(
            self.obj_function,
            dim=self._dim,
            max_eval=self._max_evals,
            bound=100,
            state_size=self._state_size,
            start_point=None,
            decisions=init_decision,
            include_single_run_features=self._include_single_run_features,  # see note below
        )

        self._best_fitness = best_f

        # reset() returns observation only (reward is issued on the first _step(), consistent with TF-Agents)
        return ts.restart(self._observation)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self._actions_count += 1
        if self._actions_count >= self._max_episodes:
            self._episode_ended = True

        decisions = self._map_action_to_decisions(int(action))

        self._state, self._observation, best_f = ues_cmaes_X(
            self.obj_function,
            dim=self._dim,
            max_eval=self._max_evals,
            bound=100,
            state_size=self._state_size,
            start_point=self._state[self._state_size - 1],
            decisions=decisions,
            include_single_run_features=self._include_single_run_features,  # see note below
        )

        reward = self._compute_reward(best_f)

        if math.isinf(reward) or math.isnan(reward):
            reward = 0.0

        if self._episode_ended:
            return ts.termination(self._observation, reward)
        return ts.transition(self._observation, reward, discount=1.0)

    def _compute_reward(self, best_f: float) -> float:
        """
        Standard (Exp10):
          - first step: minimum - best_f
          - later: max(best_prev - best_f, 0)

        Stagnation (Exp12 style):
          - first step: -best_fitness (fitness from reset rollout)
          - later:
              if improved:
                  reward = (prev_best - best_f) - stagnation_penalty if stagnation_count>=tau
                  else (prev_best - best_f)
                  reset stagnation_count
              else:
                  reward = 0 (or -penalty_lambda if enabled)
                  stagnation_count += 1
        """
        # First action in the episode:
        if self._actions_count == 1:
            if self._reward_mode == "standard":
                # Exp10 behavior
                r = float(self._minimum - best_f)
                self._best_fitness = best_f
                return r

            # stagnation mode: match your Exp12 style
            r = float(-self._best_fitness)
            # keep best fitness tracking coherent
            self._best_fitness = min(self._best_fitness, best_f)
            return r

        # Subsequent actions
        if best_f < self._best_fitness:
            improvement = float(self._best_fitness - best_f)

            if self._reward_mode == "stagnation":
                if self._stagnation_count >= self._tau:
                    improvement -= self._stagnation_penalty
                self._stagnation_count = 0

            self._best_fitness = best_f
            return improvement

        # No improvement
        if self._reward_mode == "stagnation":
            self._stagnation_count += 1

        if self._penalty_lambda > 0.0:
            return -self._penalty_lambda

        return 0.0

    def _map_action_to_decisions(self, action: int) -> Dict[str, float]:
        # Same mapping as your Exp10/Exp12 envs
        if action == 0:
            return {"FE": 0.9, "range": 1, "gamma": 1, "sigma": 0.1, "alpha": 0.1, "cma_pop": 15, "iters": 30}
        if action == 1:
            return {"FE": 0.9, "range": 0, "gamma": 2, "sigma": 10, "alpha": 0.1, "cma_pop": 30, "iters": 30}
        if action == 2:
            return {"FE": 0.9, "range": 2, "gamma": 2, "sigma": 0.1, "alpha": 0.05, "cma_pop": 15, "iters": 40}
        if action == 3:
            return {"FE": 0.5, "range": 3, "gamma": 1, "sigma": 10, "alpha": 0.1, "cma_pop": 45, "iters": 50}
        if action == 4:
            return {"FE": 0.5, "range": 0, "gamma": 2, "sigma": 10, "alpha": 0.1, "cma_pop": 15, "iters": 30}
        if action == 5:
            return {"FE": 0.9, "range": 0, "gamma": 3, "sigma": 10, "alpha": 0.1, "cma_pop": 15, "iters": 30}
        if action == 6:
            return {"FE": 0.9, "range": 0, "gamma": 1, "sigma": 1, "alpha": 0.1, "cma_pop": 30, "iters": 40}
        if action == 7:
            return {"FE": 0.9, "range": 0, "gamma": 3, "sigma": 0.1, "alpha": 0.1, "cma_pop": 15, "iters": 40}
        if action == 8:
            return {"FE": 0.5, "range": 4, "gamma": 1, "sigma": 10, "alpha": 0.1, "cma_pop": 30, "iters": 30}
        if action == 9:
            return {"FE": 0.5, "range": 4, "gamma": 2, "sigma": 1, "alpha": 0.1, "cma_pop": 15, "iters": 40}
        if action == 10:
            return {"FE": 0.5, "range": 1, "gamma": 1, "sigma": 1, "alpha": 0.1, "cma_pop": 45, "iters": 50}
        return {"FE": 0.5, "range": 2, "gamma": 2, "sigma": 10, "alpha": 0.05, "cma_pop": 15, "iters": 30}

    # tf_agents placeholders (optional)
    def get_info(self) -> types.NestedArray:
        pass

    def get_state(self) -> Any:
        pass

    def set_state(self, state: Any) -> None:
        pass
