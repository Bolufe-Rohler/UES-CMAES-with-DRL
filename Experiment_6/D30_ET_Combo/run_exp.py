from __future__ import absolute_import, division, print_function
# from uescmaes_env import UescmaesEnv
from env_rdnplus_comboXS import Env_Rdnplus_comboXS
import numpy as np

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.specs import array_spec
from tf_agents.policies import scripted_py_policy

import pickle
import matplotlib
import matplotlib.pyplot as plt
import time


def compute_returns(environment, policy, num_episodes=10):
    total_return = []
    total_fitness = []
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0
        step = 0
        while not time_step.is_last():
            action = np.array(policy[step], dtype=np.int32)
            time_step = environment.step(action)
            episode_return += time_step.reward
            step += 1
        total_return.append(episode_return.numpy().tolist()[0])
        total_fitness.append(environment.pyenv.envs[0]._best_fitness)

    # avg_return = total_return / num_episodes
    # avg_fitness = total_fitness / num_episodes
    return total_return, total_fitness


start_time = time.time()
policies = []
policies.append([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
policies.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
policies.append([3, 3, 3, 3, 3, 4, 4, 4, 4, 4])  # global
policies.append([1, 1, 1, 3, 3, 3, 3, 10, 10, 10])  # uniform converge
policies.append([1, 10, 1, 10, 1, 10, 1, 10, 1, 10])  # policy_expl_expl
policies.append([4, 2, 10, 4, 2, 10, 4, 2, 10, 10])  # 3-step expl-expl
policies.append(np.random.randint(0, 11, size=10))  # policy_random

# EXPERIMENT PARAMETERS
fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]
num_eval_episodes = 20
dim = 30

for func_num in range(14, 21):
    all_returns = []
    all_fitness = []
    experiment = "ET_cnvCombo"
    results_file_reward = f"{experiment}_rewards(f{func_num}).csv"
    results_file_fitness = f"{experiment}_fitness(f{func_num}).csv"

    for policy in policies:
        environment = Env_Rdnplus_comboXS(func_num, dim=dim, minimum=fDeltas[func_num - 1])
        eval_env = tf_py_environment.TFPyEnvironment(
            environment)  # and evaluation (Random Policy requires a TF environment)

        returns = []
        fitness = []
        try:
            returns, fitness = compute_returns(eval_env, policy, num_eval_episodes)
        except:
            pass
        finally:
            all_returns.append(returns)
            all_fitness.append(fitness)

    np.savetxt(results_file_reward, all_returns, delimiter=", ", fmt='% s')
    np.savetxt(results_file_fitness, all_fitness, delimiter=", ", fmt='% s')

print(f"--- Execution took {(time.time() - start_time) / 3600} hours ---")
