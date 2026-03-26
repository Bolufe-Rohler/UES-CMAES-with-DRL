
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

RUN_RE = re.compile(
    r'^DQN_FinalCombo_Exp20aRewardAbl_clip1_lr1e4_seed(?P<seed>\d+)_dim(?P<dim>\d+)_ss(?P<ss>\d+)_single(?P<single>\d+)_reward(?P<reward>[A-Za-z0-9_]+)_tau(?P<tau>[-+0-9.eE]+)_sp(?P<sp>[-+0-9.eE]+)_pl(?P<pl>[-+0-9.eE]+)_summary\.csv$'
)

def mean(vals):
    return sum(vals)/len(vals) if vals else 0.0

def std(vals):
    if len(vals) <= 1:
        return 0.0
    m = mean(vals)
    return (sum((v-m)**2 for v in vals)/(len(vals)-1))**0.5

def median(vals):
    if not vals:
        return 0.0
    vals = sorted(vals)
    n = len(vals)
    if n % 2 == 1:
        return vals[n//2]
    return 0.5*(vals[n//2-1] + vals[n//2])

def read_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))

def write_rows(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=".")
    ap.add_argument("--out_dir", default="rebuilt_eval")
    ap.add_argument("--pattern", default="DQN_FinalCombo_Exp20aRewardAbl_clip1_lr1e4*_summary.csv")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        raise RuntimeError("No per-run summary files found.")

    all_rows = []
    for f in files:
        base = os.path.basename(f)
        m = RUN_RE.match(base)
        if not m:
            print(f"[WARN] Skipping unexpected file name: {base}")
            continue
        rows = read_rows(f)
        if not rows:
            continue
        # keep rows as-is, but normalize type formatting
        for r in rows:
            r["train_seed"] = int(r["train_seed"])
            r["function"] = int(r["function"])
            r["fitness_mean"] = float(r["fitness_mean"])
            r["fitness_std"] = float(r["fitness_std"])
            r["fitness_min"] = float(r["fitness_min"])
            r["fitness_max"] = float(r["fitness_max"])
            r["fitness_median"] = float(r["fitness_median"])
            all_rows.append(r)

    if not all_rows:
        raise RuntimeError("No valid rows found in per-run summaries.")

    # 1) rebuilt all_runs_per_function_summary.csv
    all_fieldnames = [
        "run_name", "reward_mode", "train_seed", "function",
        "fitness_mean", "fitness_std", "fitness_min", "fitness_max", "fitness_median"
    ]
    out1 = os.path.join(args.out_dir, "all_runs_per_function_summary.csv")
    write_rows(out1, all_rows, all_fieldnames)

    # 2) reward_mode_function_summary.csv aggregated across the 3 seeds
    grouped = defaultdict(list)
    for r in all_rows:
        grouped[(r["reward_mode"], r["function"])].append(r["fitness_mean"])

    reward_mode_rows = []
    for (reward_mode, function), vals in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        reward_mode_rows.append({
            "reward_mode": reward_mode,
            "function": function,
            "seed_avg_mean": mean(vals),
            "seed_avg_std": std(vals),
            "seed_avg_min": min(vals),
            "seed_avg_max": max(vals),
            "seed_avg_median": median(vals),
        })

    out2 = os.path.join(args.out_dir, "reward_mode_function_summary.csv")
    write_rows(
        out2,
        reward_mode_rows,
        ["reward_mode", "function", "seed_avg_mean", "seed_avg_std", "seed_avg_min", "seed_avg_max", "seed_avg_median"]
    )

    # 3) overall_reward_mode_summary.csv aggregated across the 28 function means
    grouped2 = defaultdict(list)
    for r in reward_mode_rows:
        grouped2[r["reward_mode"]].append(r["seed_avg_mean"])

    overall_rows = []
    for reward_mode, vals in sorted(grouped2.items()):
        overall_rows.append({
            "reward_mode": reward_mode,
            "overall_mean": mean(vals),
            "overall_std": std(vals),
            "overall_min": min(vals),
            "overall_max": max(vals),
            "overall_median": median(vals),
        })

    out3 = os.path.join(args.out_dir, "overall_reward_mode_summary.csv")
    write_rows(
        out3,
        overall_rows,
        ["reward_mode", "overall_mean", "overall_std", "overall_min", "overall_max", "overall_median"]
    )

    print(f"Read {len(files)} per-run summary files")
    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")
    print(f"Wrote: {out3}")
    print(f"Rows in rebuilt all_runs_per_function_summary.csv: {len(all_rows)}")
    print(f"Rows in rebuilt reward_mode_function_summary.csv: {len(reward_mode_rows)}")
    print(f"Rows in rebuilt overall_reward_mode_summary.csv: {len(overall_rows)}")

if __name__ == "__main__":
    main()
