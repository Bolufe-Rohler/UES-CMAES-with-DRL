import pandas as pd
import numpy as np


def format_scientific(x: float) -> str:
    return "{:.3e}".format(x)


def format_mean_std(mean: float, std: float) -> str:
    return f"${format_scientific(mean)} \\pm {format_scientific(std)}$"


def safe_rank(values):
    """
    Average rank, lower is better.
    For two methods:
      best -> rank 1
      worst -> rank 2
      tie -> 1.5 each
    """
    arr = np.array(values, dtype=float)
    order = arr.argsort()
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1)

    # Handle ties with average rank
    unique_vals = {}
    for i, v in enumerate(arr):
        unique_vals.setdefault(v, []).append(i)

    for _, idxs in unique_vals.items():
        if len(idxs) > 1:
            avg_rank = ranks[idxs].mean()
            ranks[idxs] = avg_rank

    return ranks


def main():
    dqn_file = "exp21_dqn_per_function_across_seeds.csv"
    ddqn_file = "exp21_ddqn_per_function_across_seeds.csv"

    dqn = pd.read_csv(dqn_file)
    ddqn = pd.read_csv(ddqn_file)

    required_cols = {"function", "mean_across_seeds", "std_across_seeds"}
    if not required_cols.issubset(dqn.columns):
        raise ValueError(f"DQN file missing required columns: {required_cols - set(dqn.columns)}")
    if not required_cols.issubset(ddqn.columns):
        raise ValueError(f"DDQN file missing required columns: {required_cols - set(ddqn.columns)}")

    # Merge by function to be safe
    merged = dqn[["function", "mean_across_seeds", "std_across_seeds"]].merge(
        ddqn[["function", "mean_across_seeds", "std_across_seeds"]],
        on="function",
        suffixes=("_dqn", "_ddqn"),
        how="inner"
    ).sort_values("function").reset_index(drop=True)

    rows = []
    dqn_wins = 0
    ddqn_wins = 0
    ties = 0

    dqn_ranks = []
    ddqn_ranks = []

    for _, row in merged.iterrows():
        f = int(row["function"])

        dqn_mean = float(row["mean_across_seeds_dqn"])
        dqn_std = float(row["std_across_seeds_dqn"])

        ddqn_mean = float(row["mean_across_seeds_ddqn"])
        ddqn_std = float(row["std_across_seeds_ddqn"])

        dqn_text = format_mean_std(dqn_mean, dqn_std)
        ddqn_text = format_mean_std(ddqn_mean, ddqn_std)

        # Determine winner
        if np.isclose(dqn_mean, ddqn_mean, atol=1e-12):
            ties += 1
        elif ddqn_mean < dqn_mean:
            ddqn_text = "\\textbf{" + ddqn_text + "}"
            ddqn_wins += 1
        else:
            dqn_text = "\\textbf{" + dqn_text + "}"
            dqn_wins += 1

        # Average rank per function
        ranks = safe_rank([dqn_mean, ddqn_mean])
        dqn_ranks.append(ranks[0])
        ddqn_ranks.append(ranks[1])

        rows.append((f, dqn_text, ddqn_text))

    dqn_overall_mean = merged["mean_across_seeds_dqn"].mean()
    ddqn_overall_mean = merged["mean_across_seeds_ddqn"].mean()

    dqn_overall_median = merged["mean_across_seeds_dqn"].median()
    ddqn_overall_median = merged["mean_across_seeds_ddqn"].median()

    dqn_avg_rank = np.mean(dqn_ranks)
    ddqn_avg_rank = np.mean(ddqn_ranks)

    # Build LaTeX table
    print()
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\caption{Comparison of DQN and Double DQN restart policies on the benchmark functions ($D=30$). Values correspond to mean $\\pm$ standard deviation across three training seeds (30 optimization runs per function). Lower values are better. Best result per row is shown in bold.}")
    print("\\label{tab:dqn_vs_ddqn_per_function}")
    print("\\begin{tabular}{ccc}")
    print("\\hline")
    print("Function & DQN & Double DQN \\\\")
    print("\\hline")

    for f, dqn_text, ddqn_text in rows:
        print(f"F{f} & {dqn_text} & {ddqn_text} \\\\")

    print("\\hline")
    print(f"Wins / Losses / Ties & {dqn_wins} / {ddqn_wins} / {ties} & {ddqn_wins} / {dqn_wins} / {ties} \\\\")
    print(f"Average rank & {dqn_avg_rank:.3f} & {ddqn_avg_rank:.3f} \\\\")
    print(f"Overall mean & {format_scientific(dqn_overall_mean)} & {format_scientific(ddqn_overall_mean)} \\\\")
    print(f"Overall median & {format_scientific(dqn_overall_median)} & {format_scientific(ddqn_overall_median)} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table*}")
    print()

    # Also print a compact text summary
    print("Summary:")
    print(f"  DQN wins   : {dqn_wins}")
    print(f"  DDQN wins  : {ddqn_wins}")
    print(f"  Ties       : {ties}")
    print(f"  DQN avg rank   : {dqn_avg_rank:.3f}")
    print(f"  DDQN avg rank  : {ddqn_avg_rank:.3f}")
    print(f"  DQN overall mean   : {dqn_overall_mean:.6f}")
    print(f"  DDQN overall mean  : {ddqn_overall_mean:.6f}")
    print(f"  DQN overall median : {dqn_overall_median:.6f}")
    print(f"  DDQN overall median: {ddqn_overall_median:.6f}")


if __name__ == "__main__":
    main()