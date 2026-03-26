from __future__ import absolute_import, division, print_function

import argparse
import csv
import os
import re
import time

import numpy as np
import tensorflow as tf

from tf_agents.environments import tf_py_environment

from env_finalcombo_ablation import Env_FinalCombo_Ablation


FDeltas_CEC13_30D = [
    -1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
    -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
    700, 800, 900, 1000, 1100, 1200, 1300, 1400
]


RUN_DIR_RE = re.compile(
    r"""
    ^(?P<exp>.+)
    _seed(?P<seed>\d+)
    _dim(?P<dim>\d+)
    _ss(?P<ss>\d+)
    _single(?P<single>\d+)
    _reward(?P<reward>[A-Za-z0-9_]+)
    _tau(?P<tau>[-+0-9.eE]+)
    _sp(?P<sp>[-+0-9.eE]+)
    _pl(?P<pl>[-+0-9.eE]+)
    $
    """,
    re.VERBOSE,
)


def scalarize(x):
    if hasattr(x, "numpy"):
        x = x.numpy()
    x = np.asarray(x)
    return float(x.reshape(-1)[0])


def mean(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def std(values):
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return float(var ** 0.5)


def median(values):
    if not values:
        return 0.0
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def discover_policy_runs(results_root, exp_prefix=None):
    runs = []

    if not os.path.isdir(results_root):
        raise RuntimeError(f"results_root does not exist or is not a directory: {results_root}")

    for name in sorted(os.listdir(results_root)):
        run_dir = os.path.join(results_root, name)
        policy_dir = os.path.join(run_dir, "policy")
        saved_model_pb = os.path.join(policy_dir, "saved_model.pb")

        if not os.path.isdir(run_dir):
            continue
        if not os.path.isdir(policy_dir):
            continue
        if not os.path.isfile(saved_model_pb):
            continue
        if exp_prefix is not None and not name.startswith(exp_prefix):
            continue

        m = RUN_DIR_RE.match(name)
        if not m:
            print(f"[WARN] Skipping unexpected folder name: {name}")
            continue

        meta = m.groupdict()
        meta["run_name"] = name
        meta["run_dir"] = run_dir
        meta["policy_dir"] = policy_dir
        meta["seed"] = int(meta["seed"])
        meta["dim"] = int(meta["dim"])
        meta["ss"] = int(meta["ss"])
        meta["single"] = int(meta["single"])
        meta["tau"] = int(float(meta["tau"]))
        meta["sp"] = float(meta["sp"])
        meta["pl"] = float(meta["pl"])
        runs.append(meta)

    return runs


def make_eval_env(meta):
    py_env = Env_FinalCombo_Ablation(
        dim=meta["dim"],
        minimums=FDeltas_CEC13_30D,
        state_size=meta["ss"],
        include_single_run_features=bool(meta["single"]),
        reward_mode=meta["reward"],
        tau=meta["tau"],
        stagnation_penalty=meta["sp"],
        penalty_lambda=meta["pl"],
        randomize_function_each_episode=False,
        max_episodes=10,
        seed=meta["seed"],
    )
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    return py_env, tf_env


def run_one_episode(py_env, tf_env, policy, fun_num, episode_seed):
    py_env.set_eval_context(fun_num=fun_num, episode_seed=episode_seed)

    time_step = tf_env.reset()
    episode_return = 0.0
    action_history = []

    while not bool(time_step.is_last()):
        action_step = policy.action(time_step)
        action = int(scalarize(action_step.action))
        action_history.append(action)

        time_step = tf_env.step(action_step.action)
        episode_return += scalarize(time_step.reward)

    best_fitness = float(py_env._best_fitness)
    return episode_return, best_fitness, action_history


def summarize_by_group(rows, group_keys, value_key, output_names):
    grouped = {}

    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(float(row[value_key]))

    summary_rows = []
    for key, values in grouped.items():
        out = {}
        for i, k in enumerate(group_keys):
            out[k] = key[i]
        out[output_names["mean"]] = mean(values)
        out[output_names["std"]] = std(values)
        out[output_names["min"]] = min(values) if values else 0.0
        out[output_names["max"]] = max(values) if values else 0.0
        out[output_names["median"]] = median(values)
        summary_rows.append(out)

    return summary_rows


def evaluate_one_run(meta, runs_per_function, out_dir, base_seed=12345):
    run_name = meta["run_name"]
    policy_dir = meta["policy_dir"]

    print("=" * 80)
    print(f"Evaluating {run_name}")
    print(f"Policy dir: {policy_dir}")

    policy = tf.saved_model.load(policy_dir)
    py_env, tf_env = make_eval_env(meta)

    records = []
    start_all = time.time()

    for fun_num in range(1, 29):
        print(f"  F{fun_num:02d}")

        for rep in range(runs_per_function):
            episode_seed = base_seed + fun_num * 100000 + rep

            t0 = time.time()
            episode_return, best_fitness, action_history = run_one_episode(
                py_env=py_env,
                tf_env=tf_env,
                policy=policy,
                fun_num=fun_num,
                episode_seed=episode_seed,
            )
            elapsed = time.time() - t0

            row = {
                "run_name": run_name,
                "reward_mode": meta["reward"],
                "train_seed": meta["seed"],
                "function": fun_num,
                "rep": rep,
                "episode_seed": episode_seed,
                "best_fitness": best_fitness,
                "episode_return": episode_return,
                "wall_seconds": elapsed,
            }

            for i in range(10):
                row[f"action_step_{i+1}"] = action_history[i] if i < len(action_history) else ""

            records.append(row)

    total_hours = (time.time() - start_all) / 3600.0
    print(f"Finished {run_name} in {total_hours:.3f} hours")

    raw_fieldnames = [
        "run_name", "reward_mode", "train_seed", "function", "rep",
        "episode_seed", "best_fitness", "episode_return", "wall_seconds"
    ] + [f"action_step_{i+1}" for i in range(10)]

    raw_path = os.path.join(out_dir, f"{run_name}_raw.csv")
    write_csv(raw_path, records, raw_fieldnames)

    summary_rows = summarize_by_group(
        rows=records,
        group_keys=["run_name", "reward_mode", "train_seed", "function"],
        value_key="best_fitness",
        output_names={
            "mean": "fitness_mean",
            "std": "fitness_std",
            "min": "fitness_min",
            "max": "fitness_max",
            "median": "fitness_median",
        }
    )

    summary_fieldnames = [
        "run_name", "reward_mode", "train_seed", "function",
        "fitness_mean", "fitness_std", "fitness_min", "fitness_max", "fitness_median"
    ]
    summary_path = os.path.join(out_dir, f"{run_name}_summary.csv")
    write_csv(summary_path, summary_rows, summary_fieldnames)

    return records, summary_rows


def build_master_outputs(all_raw_rows, out_dir):
    raw_fieldnames = [
        "run_name", "reward_mode", "train_seed", "function", "rep",
        "episode_seed", "best_fitness", "episode_return", "wall_seconds"
    ] + [f"action_step_{i+1}" for i in range(10)]

    write_csv(os.path.join(out_dir, "all_runs_raw.csv"), all_raw_rows, raw_fieldnames)

    per_run_function = summarize_by_group(
        rows=all_raw_rows,
        group_keys=["run_name", "reward_mode", "train_seed", "function"],
        value_key="best_fitness",
        output_names={
            "mean": "fitness_mean",
            "std": "fitness_std",
            "min": "fitness_min",
            "max": "fitness_max",
            "median": "fitness_median",
        }
    )
    per_run_function_fieldnames = [
        "run_name", "reward_mode", "train_seed", "function",
        "fitness_mean", "fitness_std", "fitness_min", "fitness_max", "fitness_median"
    ]
    write_csv(
        os.path.join(out_dir, "all_runs_per_function_summary.csv"),
        per_run_function,
        per_run_function_fieldnames
    )

    reward_mode_function = summarize_by_group(
        rows=per_run_function,
        group_keys=["reward_mode", "function"],
        value_key="fitness_mean",
        output_names={
            "mean": "seed_avg_mean",
            "std": "seed_avg_std",
            "min": "seed_avg_min",
            "max": "seed_avg_max",
            "median": "seed_avg_median",
        }
    )
    reward_mode_function_fieldnames = [
        "reward_mode", "function",
        "seed_avg_mean", "seed_avg_std", "seed_avg_min", "seed_avg_max", "seed_avg_median"
    ]
    write_csv(
        os.path.join(out_dir, "reward_mode_function_summary.csv"),
        reward_mode_function,
        reward_mode_function_fieldnames
    )

    overall_reward_mode = summarize_by_group(
        rows=reward_mode_function,
        group_keys=["reward_mode"],
        value_key="seed_avg_mean",
        output_names={
            "mean": "overall_mean",
            "std": "overall_std",
            "min": "overall_min",
            "max": "overall_max",
            "median": "overall_median",
        }
    )
    overall_reward_mode_fieldnames = [
        "reward_mode",
        "overall_mean", "overall_std", "overall_min", "overall_max", "overall_median"
    ]
    write_csv(
        os.path.join(out_dir, "overall_reward_mode_summary.csv"),
        overall_reward_mode,
        overall_reward_mode_fieldnames
    )

    print("\nSaved:")
    print(f"  {os.path.join(out_dir, 'all_runs_raw.csv')}")
    print(f"  {os.path.join(out_dir, 'all_runs_per_function_summary.csv')}")
    print(f"  {os.path.join(out_dir, 'reward_mode_function_summary.csv')}")
    print(f"  {os.path.join(out_dir, 'overall_reward_mode_summary.csv')}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results_root",
        type=str,
        required=True,
        help="Root folder containing the trained run folders."
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="eval_exp20_results",
        help="Where evaluation CSV files will be written."
    )
    p.add_argument(
        "--exp_prefix",
        type=str,
        default="DQN_FinalCombo_Exp20aRewardAbl_clip1_lr1e4",
        help="Only evaluate folders starting with this prefix."
    )
    p.add_argument(
        "--runs_per_function",
        type=int,
        default=51,
        help="Independent evaluation runs per benchmark function."
    )
    p.add_argument(
        "--base_seed",
        type=int,
        default=12345,
        help="Base seed used to generate episode seeds."
    )
    p.add_argument(
        "--only_reward",
        type=str,
        default="",
        choices=["", "standard", "normalized", "stagnation"],
        help="Optional: evaluate only one reward mode."
    )
    p.add_argument(
        "--only_run_name",
        type=str,
        default="",
        help="Optional: evaluate only this exact run folder name."
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = discover_policy_runs(args.results_root, exp_prefix=args.exp_prefix)

    if args.only_reward:
        runs = [r for r in runs if r["reward"] == args.only_reward]

    if args.only_run_name:
        runs = [r for r in runs if r["run_name"] == args.only_run_name]

    if not runs:
        raise RuntimeError("No valid policy folders found after applying filters.")

    print("Discovered runs:")
    for r in runs:
        print(f"  {r['run_name']}")

    all_raw_rows = []
    for meta in runs:
        raw_rows, _ = evaluate_one_run(
            meta=meta,
            runs_per_function=args.runs_per_function,
            out_dir=args.out_dir,
            base_seed=args.base_seed,
        )
        all_raw_rows.extend(raw_rows)

    build_master_outputs(all_raw_rows, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()