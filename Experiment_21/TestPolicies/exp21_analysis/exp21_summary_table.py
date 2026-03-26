import pandas as pd
import numpy as np


def read_key_value_csv(path):
    df = pd.read_csv(path)
    return dict(zip(df["metric"], df["value"]))


def fmt_float(x, digits=2):
    return f"{float(x):.{digits}f}"


def main():
    overall = read_key_value_csv("exp21_overall_summary.csv")
    wins = read_key_value_csv("exp21_win_loss_counts.csv")
    wilcox = read_key_value_csv("exp21_wilcoxon.csv")

    dqn_mean = float(overall["dqn_overall_mean_of_function_means"])
    ddqn_mean = float(overall["ddqn_overall_mean_of_function_means"])

    dqn_median = float(overall["dqn_overall_median_of_function_means"])
    ddqn_median = float(overall["ddqn_overall_median_of_function_means"])

    dqn_wins = int(float(wins["dqn_wins"]))
    ddqn_wins = int(float(wins["ddqn_wins"]))
    ties = int(float(wins["ties"]))

    p_value = wilcox["p_value"]
    try:
        p_value = float(p_value)
        p_text = f"{p_value:.4f}"
    except Exception:
        p_text = str(p_value)

    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Summary comparison between DQN and Double DQN under the proposed restart-level environment. Results are based on per-function averages across three training seeds. Lower mean and median errors are better.}")
    print("\\label{tab:dqn_ddqn_summary}")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("Metric & DQN & Double DQN \\\\")
    print("\\hline")

    dqn_mean_txt = fmt_float(dqn_mean)
    ddqn_mean_txt = fmt_float(ddqn_mean)
    if ddqn_mean < dqn_mean:
        ddqn_mean_txt = f"\\textbf{{{ddqn_mean_txt}}}"
    else:
        dqn_mean_txt = f"\\textbf{{{dqn_mean_txt}}}"
    print(f"Overall mean error & {dqn_mean_txt} & {ddqn_mean_txt} \\\\")

    dqn_median_txt = fmt_float(dqn_median)
    ddqn_median_txt = fmt_float(ddqn_median)
    if ddqn_median < dqn_median:
        ddqn_median_txt = f"\\textbf{{{ddqn_median_txt}}}"
    else:
        dqn_median_txt = f"\\textbf{{{dqn_median_txt}}}"
    print(f"Overall median error & {dqn_median_txt} & {ddqn_median_txt} \\\\")

    print(f"Wins / Losses / Ties & {dqn_wins} / {ddqn_wins} / {ties} & {ddqn_wins} / {dqn_wins} / {ties} \\\\")
    print(f"Wilcoxon $p$-value & \\multicolumn{{2}}{{c}}{{{p_text}}} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    main()