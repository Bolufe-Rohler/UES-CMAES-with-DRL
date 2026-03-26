from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import time
import argparse
import random

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment

from env_benchmark_combo_eval import EnvBenchmarkComboEval


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def evaluate_policy_on_function(policy, run_name, fun, runs, dim, state_size,
                                f_delta, f_median_error):
    results = np.zeros(runs, dtype=np.float64)

    for r in range(runs):
        seed = 100000 * fun + r
        set_all_seeds(seed)

        environment = EnvBenchmarkComboEval(
            func_num=fun,
            dim=dim,
            minimum=f_delta,
            median_error=f_median_error,
            state_size=state_size,
            episodes=10,
            evals_per_restart=dim * 1000
        )
        eval_py_env = environment
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        time_step = eval_env.reset()
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = eval_env.step(action_step.action)

        results[r] = eval_py_env._best_fitness - f_delta
        print(f"[{run_name}] F{fun:02d} run {r+1:02d}/{runs:02d}: {results[r]:.10e}", flush=True)

    return results


def save_raw_results_csv(results_matrix, out_path):
    np.savetxt(out_path, results_matrix, delimiter=",", fmt="%.10e")


def save_summary_csv(results_matrix, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["function", "mean", "std", "median", "min", "max"])
        for i in range(results_matrix.shape[0]):
            row = results_matrix[i, :]
            writer.writerow([
                i + 1,
                float(np.mean(row)),
                float(np.std(row, ddof=1)) if len(row) > 1 else 0.0,
                float(np.median(row)),
                float(np.min(row)),
                float(np.max(row))
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--state_size", type=int, default=20)
    args = parser.parse_args()

    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    f_deltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600, -500, -400, -300, -200, -100,
                100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]

    f_median_errors = [2.565e-04, 2507.7, 8.8812, 11.345, 0.000335, 1.01402, 0.02984, 20.337, 1.8678, 0.004882,
                       2.9848, 3.3497, 4.6666, 245.85, 212.93, 0.03878, 13.267, 13.6609, 0.4489, 2.6383,
                       400.15, 368.77, 485.011, 207.34, 209.92, 210.71, 436.71, 300.0]

    print("=" * 80)
    print(f"Evaluating run: {args.run_name}")
    print(f"Policy dir: {args.policy_dir}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)

    policy = tf.saved_model.load(args.policy_dir)
    results = np.zeros((28, args.runs), dtype=np.float64)

    for fun in range(1, 29):
        print(f"\n[{args.run_name}] Starting function F{fun:02d}")
        results[fun - 1, :] = evaluate_policy_on_function(
            policy=policy,
            run_name=args.run_name,
            fun=fun,
            runs=args.runs,
            dim=args.dim,
            state_size=args.state_size,
            f_delta=f_deltas[fun - 1],
            f_median_error=f_median_errors[fun - 1]
        )

        raw_path = os.path.join(args.output_dir, f"{args.run_name}_raw.csv")
        summary_path = os.path.join(args.output_dir, f"{args.run_name}_summary.csv")
        save_raw_results_csv(results, raw_path)
        save_summary_csv(results, summary_path)

    overall_mean = float(np.mean(results))
    overall_median = float(np.median(results))

    with open(os.path.join(args.output_dir, f"{args.run_name}_global_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "overall_mean", "overall_median"])
        writer.writerow([args.run_name, overall_mean, overall_median])

    elapsed_hours = (time.time() - start_time) / 3600.0
    print(f"\nFinished {args.run_name}")
    print(f"Overall mean error:   {overall_mean:.10e}")
    print(f"Overall median error: {overall_median:.10e}")
    print(f"--- Total execution took {elapsed_hours:.4f} hours ---")


if __name__ == "__main__":
    main()