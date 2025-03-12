from env_benchmark_combo_norm import Env_benchmark_combo_norm
import functions
import os
import time
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
from tf_agents.trajectories import time_step as ts

start_time = time.time()

save_dir = os.getcwd()
policy_dir = os.path.join(save_dir, 'policy')
policy = tf.saved_model.load(policy_dir)

dim = 30
runs = 30

# EXPERIMENT PARAMETERS
f_deltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600, -500, -400, -300, -200, -100,
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]


f_median_errors = [2.565e-04, 2507.7, 8.8812, 11.345, 0.000335, 1.01402, 0.02984, 20.337, 1.8678, 0.004882, 2.9848,
                   3.3497, 4.6666, 245.85, 212.93, 0.03878, 13.267, 13.6609, 0.4489, 2.6383, 400.15, 368.77, 485.011,
                   207.34, 209.92, 210.71, 436.71, 300.0]

results = np.zeros((28, 30))

for fun in range(1, 29):

    print(f"For function {fun} ...")
    environment = Env_benchmark_combo_norm(fun, dim=dim, minimum=f_deltas[fun - 1],
                                           median_error=f_median_errors[fun-1])
    eval_py_env = environment  # and evaluation...
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    for r in range(runs):
        time_step = eval_env.reset()
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = eval_env.step(action_step.action)
        results[fun-1, r] = eval_py_env._best_fitness - f_deltas[fun-1]
        print(results[fun-1, r] )

    np.savetxt(f"combo_norm_results(103000).csv", results, delimiter=", ", fmt='% s')

print(f"--- Execution took {(time.time() - start_time) / 3600} hours ---")
