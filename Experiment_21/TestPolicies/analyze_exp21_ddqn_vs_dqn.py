from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def discover_summary_files(input_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Returns:
        dqn_files, ddqn_files
    """
    summary_files = sorted(input_dir.rglob("*_summary.csv"))

    dqn_files = []
    ddqn_files = []

    for f in summary_files:
        name = f.name.lower()
        if "_ddqn_" in name:
            ddqn_files.append(f)
        elif "_dqn_" in name:
            dqn_files.append(f)

    return dqn_files, ddqn_files


def extract_seed(path: Path) -> int:
    """
    Extracts seed number from filenames like:
    ..._seed0_..._summary.csv
    """
    name = path.stem
    marker = "_seed"
    idx = name.find(marker)
    if idx == -1:
        raise ValueError(f"Could not find seed in filename: {path.name}")

    start = idx + len(marker)
    digits = []
    while start < len(name) and name[start].isdigit():
        digits.append(name[start])
        start += 1

    if not digits:
        raise ValueError(f"Could not parse seed in filename: {path.name}")

    return int("".join(digits))


def load_summary_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = {"function", "mean"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    df = df.copy()
    df["function"] = df["function"].astype(int)
    df["mean"] = df["mean"].astype(float)
    return df


def build_seed_table(files: List[Path], algo_name: str) -> pd.DataFrame:
    """
    Builds a table:
        function | seed0 | seed1 | seed2
    from the summary CSVs, using the 'mean' column.
    """
    if not files:
        raise ValueError(f"No files found for algorithm: {algo_name}")

    merged = None

    for f in sorted(files, key=extract_seed):
        seed = extract_seed(f)
        df = load_summary_csv(f)[["function", "mean"]].rename(columns={"mean": f"seed{seed}"})

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="function", how="inner")

    if merged is None:
        raise ValueError(f"Failed to build seed table for {algo_name}")

    merged = merged.sort_values("function").reset_index(drop=True)
    return merged


def add_aggregate_columns(df: pd.DataFrame) -> pd.DataFrame:
    seed_cols = [c for c in df.columns if c.startswith("seed")]
    out = df.copy()
    out["mean_across_seeds"] = out[seed_cols].mean(axis=1)
    out["std_across_seeds"] = out[seed_cols].std(axis=1, ddof=1)
    out["median_across_seeds"] = out[seed_cols].median(axis=1)
    out["best_seed_value"] = out[seed_cols].min(axis=1)   # lower is better
    out["worst_seed_value"] = out[seed_cols].max(axis=1)  # higher is worse
    return out


def compare_algorithms(dqn_df: pd.DataFrame, ddqn_df: pd.DataFrame, tol: float = 1e-12) -> pd.DataFrame:
    """
    Compares DQN vs DDQN using mean_across_seeds per function.
    Lower is better.
    """
    merged = dqn_df[["function", "mean_across_seeds"]].merge(
        ddqn_df[["function", "mean_across_seeds"]],
        on="function",
        suffixes=("_dqn", "_ddqn"),
        how="inner"
    )

    outcomes = []
    diffs = []

    for _, row in merged.iterrows():
        dqn_val = float(row["mean_across_seeds_dqn"])
        ddqn_val = float(row["mean_across_seeds_ddqn"])
        diff = dqn_val - ddqn_val  # positive means DDQN is better
        diffs.append(diff)

        if math.isclose(dqn_val, ddqn_val, abs_tol=tol):
            outcomes.append("tie")
        elif ddqn_val < dqn_val:
            outcomes.append("ddqn_win")
        else:
            outcomes.append("dqn_win")

    merged["difference_dqn_minus_ddqn"] = diffs
    merged["winner"] = outcomes
    return merged


def compute_win_loss_counts(compare_df: pd.DataFrame) -> Dict[str, int]:
    counts = compare_df["winner"].value_counts().to_dict()
    return {
        "ddqn_wins": counts.get("ddqn_win", 0),
        "dqn_wins": counts.get("dqn_win", 0),
        "ties": counts.get("tie", 0),
        "num_functions": int(len(compare_df)),
    }


def run_wilcoxon(compare_df: pd.DataFrame) -> Dict[str, object]:
    """
    Paired Wilcoxon signed-rank test on the 28 per-function averages.
    We compare:
        DQN mean_across_seeds  vs  DDQN mean_across_seeds
    Lower is better.
    """
    x = compare_df["mean_across_seeds_dqn"].to_numpy(dtype=float)
    y = compare_df["mean_across_seeds_ddqn"].to_numpy(dtype=float)

    result = {
        "scipy_available": SCIPY_AVAILABLE,
        "n_pairs": int(len(x)),
        "test_used": "wilcoxon signed-rank",
        "alternative": "two-sided",
        "statistic": None,
        "p_value": None,
        "note": ""
    }

    if not SCIPY_AVAILABLE:
        result["note"] = "SciPy not available; Wilcoxon test was not run."
        return result

    try:
        test = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        result["statistic"] = float(test.statistic)
        result["p_value"] = float(test.pvalue)
        result["note"] = "OK"
    except Exception as e:
        result["note"] = f"Wilcoxon failed: {repr(e)}"

    return result


def save_key_value_csv(data: Dict[str, object], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in data.items():
            writer.writerow([k, v])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Exp21 DQN vs DDQN summary CSVs.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the 6 *_summary.csv files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exp21_analysis",
        help="Directory where analysis CSVs will be written."
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dqn_files, ddqn_files = discover_summary_files(input_dir)

    print("Discovered files:")
    print(f"  DQN : {len(dqn_files)}")
    for f in dqn_files:
        print(f"    - {f.name}")
    print(f"  DDQN: {len(ddqn_files)}")
    for f in ddqn_files:
        print(f"    - {f.name}")
    print()

    if len(dqn_files) == 0 or len(ddqn_files) == 0:
        raise RuntimeError("Could not find both DQN and DDQN summary files.")

    dqn_seed_table = build_seed_table(dqn_files, "DQN")
    ddqn_seed_table = build_seed_table(ddqn_files, "DDQN")

    dqn_agg = add_aggregate_columns(dqn_seed_table)
    ddqn_agg = add_aggregate_columns(ddqn_seed_table)

    compare_df = compare_algorithms(dqn_agg, ddqn_agg)
    win_loss = compute_win_loss_counts(compare_df)
    wilcoxon_result = run_wilcoxon(compare_df)

    # Overall aggregate means from the 28 per-function averages
    overall = {
        "dqn_overall_mean_of_function_means": float(dqn_agg["mean_across_seeds"].mean()),
        "ddqn_overall_mean_of_function_means": float(ddqn_agg["mean_across_seeds"].mean()),
        "dqn_overall_median_of_function_means": float(dqn_agg["mean_across_seeds"].median()),
        "ddqn_overall_median_of_function_means": float(ddqn_agg["mean_across_seeds"].median()),
    }

    # Save outputs
    dqn_agg.to_csv(output_dir / "exp21_dqn_per_function_across_seeds.csv", index=False)
    ddqn_agg.to_csv(output_dir / "exp21_ddqn_per_function_across_seeds.csv", index=False)
    compare_df.to_csv(output_dir / "exp21_dqn_vs_ddqn_per_function_comparison.csv", index=False)
    save_key_value_csv(win_loss, output_dir / "exp21_win_loss_counts.csv")
    save_key_value_csv(wilcoxon_result, output_dir / "exp21_wilcoxon.csv")
    save_key_value_csv(overall, output_dir / "exp21_overall_summary.csv")

    # Console summary
    print("=== Overall summary ===")
    print(f"DQN  overall mean of per-function means : {overall['dqn_overall_mean_of_function_means']:.6f}")
    print(f"DDQN overall mean of per-function means : {overall['ddqn_overall_mean_of_function_means']:.6f}")
    print(f"DQN  overall median of per-function means: {overall['dqn_overall_median_of_function_means']:.6f}")
    print(f"DDQN overall median of per-function means: {overall['ddqn_overall_median_of_function_means']:.6f}")
    print()

    print("=== Win/loss counts ===")
    print(f"DDQN wins: {win_loss['ddqn_wins']}")
    print(f"DQN wins : {win_loss['dqn_wins']}")
    print(f"Ties     : {win_loss['ties']}")
    print(f"Functions: {win_loss['num_functions']}")
    print()

    print("=== Wilcoxon signed-rank test ===")
    if wilcoxon_result["scipy_available"] and wilcoxon_result["p_value"] is not None:
        print(f"Statistic: {wilcoxon_result['statistic']:.6f}")
        print(f"p-value  : {wilcoxon_result['p_value']:.6g}")
    else:
        print(wilcoxon_result["note"])

    print()
    print(f"Saved analysis files to: {output_dir}")


if __name__ == "__main__":
    main()