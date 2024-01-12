from __future__ import absolute_import, division, print_function
# from uescmaes_env import UescmaesEnv
from env_rdnplus_comboXS import Env_Rdnplus_comboXS
import numpy
import pydot
from tensorflow.keras.utils import plot_model

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
from tf_agents.policies import policy_saver

import pickle
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import csv


def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    total_fitness = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
        total_fitness += environment.pyenv.envs[0]._best_fitness

    avg_return = total_return / num_episodes
    avg_fitness = total_fitness / num_episodes
    return avg_return, avg_fitness


# EXPERIMENT PARAMETERS
fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

dim = 30
experiment = "DQN_ComboStd"
loss_file = f"{experiment}_loss).csv"
results_file_reward = f"{experiment}_returns.csv"
results_file_fitness = f"{experiment}_fitness.csv"
figure_file_rewards = f"{experiment}_plot.png"
figure_file_fitness = f"{experiment}_fit_plot.png"


# Execution parameters
num_iterations = 200000

initial_collect_steps = 100
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000

batch_size = 64
learning_rate = 1e-3
log_interval = 200
func_num = 1

num_eval_episodes = 10
eval_interval = 500

# Creating environments
environment = Env_Rdnplus_comboXS(func_num, dim=dim, minimum=fDeltas)  # fDeltas[func_num - 1])
train_py_env = environment  # Python environments for training
eval_py_env = environment  # and evaluation...

train_env = tf_py_environment.TFPyEnvironment(train_py_env)  # TF environments for training
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)  # and evaluation (Random Policy requires a TF environment)

# Creating network
fc_layer_params = (100, 75, 50)
action_tensor_spec = tensor_spec.from_spec(environment.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(dense_layers + [q_values_layer])



# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))



# train_step_counter = tf.Variable(0)

# According to the TF tutorial this old version has to be used for checkpoints
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
global_step = tf.compat.v1.train.get_or_create_global_step()

# Creating agent
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=global_step)

agent.initialize()



# Visualize the architecture using plot_model
plot_model(q_net), to_file='dqn_agent_architecture.png', show_shapes=True, expand_nested=True)

# Display the architecture in the console
q_net.summary()
