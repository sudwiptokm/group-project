"""
Aggregate evaluation metrics across algorithms and seeds -> comparison table.

Reads the per-episode CSVs sumo-rl writes during evaluation
(logs/eval_<algo>_seed<seed>_conn*_ep*.csv), time-averages each run over the
episode, then reports mean +/- std across seeds per algorithm and ranks them.

Lower waiting time / more throughput = better. The ranking key is mean waiting
time (system_mean_waiting_time), the headline efficiency metric.

Workflow:
    # train each algo on several seeds
    for a in dqn qrdqn ppo a2c; do
      for s in 0 1 2; do python train.py --algo $a --seed $s --steps 100000; done
    done
    # evaluate each trained model on held-out seeds
    for a in dqn qrdqn ppo a2c; do
      for s in 42 43 44; do
        python train.py --algo $a --eval models/${a}_seed0.zip --seed $s
      done
    done
    # build the table
    python compare.py
"""

import argparse
import glob
import os

import pandas as pd

from algos import ALGOS

# episode-averaged columns we surface (present in every sumo-rl eval CSV)
METRICS = [
    "system_mean_waiting_time",
    "system_total_stopped",
    "system_mean_speed",
    "system_total_waiting_time",
]
RANK_KEY = "system_mean_waiting_time"  # lower is better


def _run_means(logs_dir: str, algo: str) -> pd.DataFrame:
    """One row per eval run for `algo`, each metric time-averaged over the episode."""
    pattern = os.path.join(logs_dir, f"eval_{algo}_seed*_conn*_ep*.csv")
    rows = []
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path)
        row = {m: df[m].mean() for m in METRICS if m in df.columns}
        row["run"] = os.path.basename(path)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="logs")
    args = p.parse_args()

    summary = []
    for algo in ALGOS:
        runs = _run_means(args.logs, algo)
        if runs.empty:
            continue
        agg = {"algo": algo, "n_runs": len(runs)}
        for m in METRICS:
            if m in runs.columns:
                agg[f"{m}_mean"] = runs[m].mean()
                agg[f"{m}_std"] = runs[m].std(ddof=0)
        summary.append(agg)

    if not summary:
        print("no eval CSVs found — run evaluations first "
              "(python train.py --algo <a> --eval models/<a>_seed0.zip --seed 42)")
        return

    table = pd.DataFrame(summary).sort_values(f"{RANK_KEY}_mean")

    # pretty "mean ± std" columns
    print("\n=== Algorithm comparison (held-out eval, mean ± std over runs) ===\n")
    display = pd.DataFrame({"algo": table["algo"], "runs": table["n_runs"]})
    for m in METRICS:
        mc, sc = f"{m}_mean", f"{m}_std"
        if mc in table.columns:
            display[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(table[mc], table[sc])]
    print(display.to_string(index=False))

    best = table.iloc[0]["algo"]
    print(f"\nbest by {RANK_KEY} (lower=better): {best.upper()}")

    out = os.path.join(args.logs, "comparison.csv")
    table.to_csv(out, index=False)
    print(f"full table -> {out}")


if __name__ == "__main__":
    main()
