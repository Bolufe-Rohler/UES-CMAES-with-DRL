from __future__ import absolute_import, division, print_function

import argparse
import os
import time
import csv
import random

import numpy as np
import reverb
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.policies import policy_saver

from env_finalcombo_ablation import Env_FinalCombo_Ablation


def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_in", distribution="truncated_normal"
        ),
    )


def compute_avg_return(environment, policy, num_episodes=10):
    """
    Returns:
      avg_return: average sum of rewards per episode
      avg_fitness: average of internal env best fitness at end of episode
    """
    total_return = 0.0
    total_fitness = 0.0

    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += float(time_step.reward)

        total_return += episode_return
        total_fitness += float(environment.pyenv.envs[0]._best_fitness)

    return total_return / num_episodes, total_fitness / num_episodes


def save_list_csv(path, values):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for v in values:
            writer.writerow([v])


def parse_args():
    p = argparse.ArgumentParser()

    # Experiment naming / output
    p.add_argument("--experiment", type=str, default="DQN_FinalCombo_Ablation")
    p.add_argument("--out_dir", type=str, default="results")

    # Agent choice
    p.add_argument("--agent", type=str, default="dqn", choices=["dqn", "ddqn"])

    # Reproducibility
    p.add_argument("--seed", type=int, default=0)

    # Env config
    p.add_argument("--dim", type=int, default=30)
    p.add_argument("--state_size", type=int, default=20)
    p.add_argument("--include_single_run_features", type=int, default=1)
    p.add_argument("--reward_mode", type=str, default="standard", choices=["standard", "stagnation"])
    p.add_argument("--tau", type=int, default=3)
    p.add_argument("--stagnation_penalty", type=float, default=0.1)
    p.add_argument("--penalty_lambda", type=float, default=0.0)

    # Training config
    p.add_argument("--num_iterations", type=int, default=200000)
    p.add_argument("--initial_collect_steps", type=int, default=100)
    p.add_argument("--collect_steps_per_iteration", type=int, default=1)
    p.add_argument("--replay_buffer_max_length", type=int, default=100000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--log_interval", type=int, default=200)
    p.add_argument("--num_eval_episodes", type=int, default=10)

    return p.parse_args()


# Debug safety: stop early with context if batch has NaN/Inf
def _assert_finite_batch(experience):
    flat = tf.nest.flatten(experience)
    for t in flat:
        if t is None:
            continue
        if hasattr(t, "dtype") and t.dtype is not None and t.dtype.is_floating:
            tf.debugging.assert_all_finite(t, "Non-finite values found in experience batch")


def main():
    args = parse_args()
    start_time = time.time()

    # Global seeding (important for stability experiments)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # fDeltas: same as Exp10 (CEC13 30D minima offsets)
    fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
               -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
               700, 800, 900, 1000, 1100, 1200, 1300, 1400]

    run_name = (
        f"{args.experiment}_{args.agent}"
        f"_seed{args.seed}"
        f"_dim{args.dim}"
        f"_ss{args.state_size}"
        f"_single{args.include_single_run_features}"
        f"_reward{args.reward_mode}"
        f"_tau{args.tau}"
        f"_sp{args.stagnation_penalty}"
        f"_pl{args.penalty_lambda}"
    )
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    loss_file = os.path.join(out_dir, "loss.csv")
    returns_file = os.path.join(out_dir, "returns.csv")
    fitness_file = os.path.join(out_dir, "fitness.csv")

    policy_dir = os.path.join(out_dir, "policy")
    checkpoint_dir = os.path.join(out_dir, "checkpoint")
    os.makedirs(policy_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # IMPORTANT FIX: create separate *Python* env instances for training & eval
    # -------------------------------------------------------------------------
    train_py_env = Env_FinalCombo_Ablation(
        dim=args.dim,
        minimums=fDeltas,
        state_size=args.state_size,
        include_single_run_features=bool(args.include_single_run_features),
        reward_mode=args.reward_mode,
        tau=args.tau,
        stagnation_penalty=args.stagnation_penalty,
        penalty_lambda=args.penalty_lambda,
        seed=args.seed,
    )

    eval_py_env = Env_FinalCombo_Ablation(
        dim=args.dim,
        minimums=fDeltas,
        state_size=args.state_size,
        include_single_run_features=bool(args.include_single_run_features),
        reward_mode=args.reward_mode,
        tau=args.tau,
        stagnation_penalty=args.stagnation_penalty,
        penalty_lambda=args.penalty_lambda,
        seed=args.seed + 10000,
    )

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    # -------------------------------------------------------------------------

    # Network
    fc_layer_params = (100, 75, 50)
    action_tensor_spec = tensor_spec.from_spec(train_py_env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2),
    )
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate)
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # -------------------------
    # DQN vs Double DQN switch
    # -------------------------
    AgentCls = dqn_agent.DqnAgent if args.agent == "dqn" else dqn_agent.DdqnAgent

    agent = AgentCls(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=global_step,
        gradient_clipping=1.0,   # stability safeguard
    )
    agent.initialize()

    # Replay buffer (Reverb)
    table_name = "uniform_table"
    replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=args.replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature,
    )
    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=2,
        local_server=reverb_server,
    )
    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay_buffer.py_client,
        table_name,
        sequence_length=2,
    )

    # Initial random collection
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
    init_driver = py_driver.PyDriver(
        train_py_env,
        py_tf_eager_policy.PyTFEagerPolicy(random_policy, use_tf_function=True),
        [rb_observer],
        max_steps=args.initial_collect_steps,
    )
    init_driver.run(train_py_env.reset())

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=args.batch_size,
        num_steps=2,
    ).prefetch(3)
    iterator = iter(dataset)

    agent.train = common.function(agent.train)

    collect_driver = py_driver.PyDriver(
        train_py_env,
        py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=args.collect_steps_per_iteration,
    )

    # Saver + checkpoint
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)

    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step,
    )
    train_checkpointer.initialize_or_restore()

    # Metrics storage
    returns = []
    fitness = []
    losses = []

    # Baseline eval before training
    avg_return, avg_fitness = compute_avg_return(eval_env, agent.policy, args.num_eval_episodes)
    returns.append(float(avg_return))
    fitness.append(float(avg_fitness))
    save_list_csv(returns_file, returns)
    save_list_csv(fitness_file, fitness)

    time_step = train_py_env.reset()

    for _ in range(args.num_iterations):
        time_step, _ = collect_driver.run(time_step)
        experience, _ = next(iterator)

        _assert_finite_batch(experience)

        train_loss = agent.train(experience).loss
        step = int(agent.train_step_counter.numpy())

        if step % args.log_interval == 0:
            print(f"step={step} loss={train_loss}")
            losses.append(float(train_loss))
            save_list_csv(loss_file, losses)

        if step % args.eval_interval == 0:
            avg_return, avg_fitness = compute_avg_return(eval_env, agent.policy, args.num_eval_episodes)
            print(f"step={step} avg_return={avg_return} avg_fitness={avg_fitness}")
            returns.append(float(avg_return))
            fitness.append(float(avg_fitness))
            save_list_csv(returns_file, returns)
            save_list_csv(fitness_file, fitness)

            train_checkpointer.save(global_step)
            tf_policy_saver.save(policy_dir)

    final_step = int(agent.train_step_counter.numpy())
    print("FINAL_STEP", final_step)
    print(f"--- Execution took {(time.time() - start_time) / 3600.0:.3f} hours ---")
    print(f"Results saved under: {out_dir}")


if __name__ == "__main__":
    main()