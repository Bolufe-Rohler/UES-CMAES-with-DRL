
import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def read_csv_dict(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mean(vals):
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def rank_dict(score_dict):
    items = sorted(score_dict.items(), key=lambda kv: kv[1])  # lower is better
    ranks = {}
    current_rank = 1
    for i, (k, v) in enumerate(items):
        if i > 0 and v != items[i - 1][1]:
            current_rank = i + 1
        ranks[k] = current_rank
    return ranks


def build_reward_mode_table(rows):
    table = {}
    reward_modes = set()
    functions = set()
    for r in rows:
        mode = r["reward_mode"]
        fun = int(r["function"])
        val = float(r["seed_avg_mean"])
        reward_modes.add(mode)
        functions.add(fun)
        table.setdefault(fun, {})[mode] = val
    return table, sorted(reward_modes), sorted(functions)


def build_seed_level_table(rows):
    data = defaultdict(lambda: defaultdict(list))
    reward_modes = set()
    functions = set()
    for r in rows:
        mode = r["reward_mode"]
        fun = int(r["function"])
        val = float(r["fitness_mean"])
        reward_modes.add(mode)
        functions.add(fun)
        data[fun][mode].append(val)
    return data, sorted(reward_modes), sorted(functions)


def write_wide_csv(table, reward_modes, functions, out_path):
    rows = []
    for fun in functions:
        row = {"function": fun}
        for mode in reward_modes:
            row[mode] = table.get(fun, {}).get(mode, "")
        rows.append(row)
    write_csv(out_path, rows, ["function"] + reward_modes)


def write_latex_table(table, reward_modes, functions, out_path):
    with open(out_path, "w") as f:
        cols = "l" + "c" * len(reward_modes)
        f.write("\\begin{tabular}{" + cols + "}\n")
        f.write("\\toprule\n")
        f.write("Function & " + " & ".join(m.capitalize() for m in reward_modes) + " \\\\\n")
        f.write("\\midrule\n")
        for fun in functions:
            vals = []
            best = None
            if fun in table:
                best = min(table[fun].values())
            for mode in reward_modes:
                v = table.get(fun, {}).get(mode, None)
                if v is None:
                    vals.append("")
                else:
                    s = f"{v:.6g}"
                    if best is not None and v == best:
                        s = "\\textbf{" + s + "}"
                    vals.append(s)
            f.write(f"F{fun} & " + " & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def write_summary_stats(seed_table, reward_modes, functions, out_path):
    wins = {m: 0 for m in reward_modes}
    avg_rank_sum = {m: 0.0 for m in reward_modes}
    count = 0

    for fun in functions:
        scores = {}
        for mode in reward_modes:
            vals = seed_table[fun].get(mode, [])
            if vals:
                scores[mode] = mean(vals)
        if not scores:
            continue

        best_mode = min(scores, key=scores.get)
        wins[best_mode] += 1

        ranks = rank_dict(scores)
        for mode, rk in ranks.items():
            avg_rank_sum[mode] += rk
        count += 1

    rows = []
    for mode in reward_modes:
        rows.append({
            "reward_mode": mode,
            "wins": wins[mode],
            "average_rank": avg_rank_sum[mode] / count if count else float("nan")
        })

    write_csv(out_path, rows, ["reward_mode", "wins", "average_rank"])


def make_paired_difference_plot(seed_table, mode_a, mode_b, functions, out_path):
    xs, ys = [], []
    for fun in functions:
        vals_a = seed_table[fun].get(mode_a, [])
        vals_b = seed_table[fun].get(mode_b, [])
        n = min(len(vals_a), len(vals_b))
        for i in range(n):
            xs.append(fun)
            ys.append(vals_a[i] - vals_b[i])

    plt.figure(figsize=(10, 5))
    plt.axhline(0, linewidth=1)
    plt.scatter(xs, ys)
    plt.xlabel("Function")
    plt.ylabel(f"{mode_a} - {mode_b}")
    plt.title(f"Paired Differences: {mode_a} vs {mode_b}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_boxplot(seed_table, reward_modes, functions, out_path):
    data, labels = [], []
    for mode in reward_modes:
        vals = []
        for fun in functions:
            vals.extend(seed_table[fun].get(mode, []))
        if vals:
            data.append(vals)
            labels.append(mode)

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels)
    plt.ylabel("Fitness mean")
    plt.title("Distribution of function-level means by reward mode")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=".")
    ap.add_argument("--out_dir", default="processed_exp20")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    reward_rows = read_csv_dict(os.path.join(args.input_dir, "reward_mode_function_summary.csv"))
    all_rows = read_csv_dict(os.path.join(args.input_dir, "all_runs_per_function_summary.csv"))

    reward_table, reward_modes, functions = build_reward_mode_table(reward_rows)
    seed_table, seed_modes, seed_functions = build_seed_level_table(all_rows)

    write_wide_csv(
        reward_table, reward_modes, functions,
        os.path.join(args.out_dir, "reward_mode_comparison_wide.csv")
    )
    write_latex_table(
        reward_table, reward_modes, functions,
        os.path.join(args.out_dir, "reward_mode_comparison_table.tex")
    )
    write_summary_stats(
        seed_table, seed_modes, seed_functions,
        os.path.join(args.out_dir, "reward_mode_wins_ranks.csv")
    )

    if len(seed_modes) >= 2:
        make_paired_difference_plot(
            seed_table, seed_modes[0], seed_modes[1], seed_functions,
            os.path.join(args.out_dir, f"paired_diff_{seed_modes[0]}_vs_{seed_modes[1]}.png")
        )

    if len(seed_modes) >= 3:
        make_paired_difference_plot(
            seed_table, "standard", "normalized", seed_functions,
            os.path.join(args.out_dir, "paired_diff_standard_vs_normalized.png")
        )
        make_paired_difference_plot(
            seed_table, "standard", "stagnation", seed_functions,
            os.path.join(args.out_dir, "paired_diff_standard_vs_stagnation.png")
        )
        make_paired_difference_plot(
            seed_table, "normalized", "stagnation", seed_functions,
            os.path.join(args.out_dir, "paired_diff_normalized_vs_stagnation.png")
        )

    make_boxplot(
        seed_table, seed_modes, seed_functions,
        os.path.join(args.out_dir, "reward_mode_boxplot.png")
    )

    print(f"Done. Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
