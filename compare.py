"""
Aggregate evaluation metrics across algorithms and seeds -> comparison table.

Reads the per-episode CSVs written during evaluation, time-averages each run
over the episode, then reports mean ± std per (scenario, λ, algo) group and
ranks within each group by mean waiting time.

Filename conventions
--------------------
RL algos:    logs/eval_<algo>_<scenario>_lam<LAM>_seed<seed>_conn<N>_ep<M>.csv
             where <LAM> is the λ float with the dot removed: 0.0→"00", 0.5→"05", 1.0→"10"
Fixed-time:  logs/eval_fixedtime_<scenario>_seed<seed>_conn<N>_ep<M>.csv  (NO lam fragment)

Lower waiting time / more throughput = better. The ranking key is mean waiting
time (system_mean_waiting_time), the headline efficiency metric.

Workflow:
    # train each algo on several seeds / lambdas
    python train.py --algo dqn --scenario peak --lam 0.5 --seed 0 --steps 100000
    # run fixed-time baseline
    python baseline.py --scenario peak --seed 42
    # build the table
    python compare.py
"""

import argparse
import glob
import os

import pandas as pd

# episode-averaged columns we surface (present in every sumo-rl eval CSV)
METRICS = [
    "system_mean_waiting_time",
    "system_total_stopped",
    "system_mean_speed",
    "system_total_waiting_time",
    "system_safety_brake",     # vulnerability-weighted emergency braking (SafetyLoggingEnv)
    "system_safety_exposure",  # vulnerability-weighted intersection exposure
    "system_safety_total",     # brake + exposure = raw safety penalty
]
RANK_KEY = "system_mean_waiting_time"  # lower is better


def _run_means(logs_dir: str, entity: str, scenario: str, lam=None) -> pd.DataFrame:
    """One row per eval run for `entity`/`scenario`/`lam`, each metric time-averaged.

    Parameters
    ----------
    logs_dir : str
        Directory containing eval CSVs.
    entity : str
        Algorithm name (e.g. "dqn") or "fixedtime".
    scenario : str
        Scenario name (e.g. "peak", "offpeak").
    lam : str or None
        Lambda tag string (e.g. "00", "05", "10") for RL algos.
        None for the fixed-time baseline (no lam fragment in filename).
    """
    if lam is None:
        # fixed-time: eval_fixedtime_<scenario>_seed*.csv (with _conn*_ep* suffix)
        pattern = os.path.join(logs_dir, f"eval_{entity}_{scenario}_seed*.csv")
    else:
        # RL algo: eval_<algo>_<scenario>_lam<lam>_seed*.csv
        pattern = os.path.join(logs_dir, f"eval_{entity}_{scenario}_lam{lam}_seed*.csv")

    rows = []
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path)
        row = {m: df[m].mean() for m in METRICS if m in df.columns}
        row["run"] = os.path.basename(path)
        rows.append(row)
    return pd.DataFrame(rows)


def _summarise(df: pd.DataFrame, entity: str, scenario: str, lam: str) -> dict:
    """Return a single summary dict for one (entity, scenario, lam) group."""
    row = {"scenario": scenario, "lam": lam, "algo": entity, "n_runs": len(df)}
    for m in METRICS:
        if m in df.columns:
            row[f"{m}_mean"] = df[m].mean()
            row[f"{m}_std"] = df[m].std()
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="logs")
    args = p.parse_args()

    scenarios = ["peak", "offpeak"]
    lambdas = ["00", "05", "10"]   # filename lambda tags
    algos = ["dqn", "qrdqn", "ppo", "a2c"]

    rows = []
    for scenario in scenarios:
        # fixed-time baseline is lambda-independent — load once per scenario
        base = _run_means(args.logs, "fixedtime", scenario)
        for lam in lambdas:
            for algo in algos:
                df = _run_means(args.logs, algo, scenario, lam=lam)
                if df.empty:
                    continue
                rows.append(_summarise(df, algo, scenario, lam))
            # always add fixed-time row to every (scenario, lam) group for direct comparison
            if not base.empty:
                rows.append(_summarise(base, "fixedtime", scenario, lam))

    if not rows:
        print("no eval CSVs found yet")
        return

    out_df = pd.DataFrame(rows).sort_values(
        ["scenario", "lam", f"{RANK_KEY}_mean"],
        ascending=[True, True, True],
    )

    # pretty print with mean ± std columns
    print("\n=== Algorithm comparison grouped by (scenario, λ) "
          f"— ranked by {RANK_KEY} (lower=better) ===\n")
    display_cols = ["scenario", "lam", "algo", "n_runs"]
    display = out_df[display_cols].copy()
    for m in METRICS:
        mc, sc = f"{m}_mean", f"{m}_std"
        if mc in out_df.columns:
            display[m] = [
                f"{a:.3f} ± {b:.3f}" if pd.notna(b) else f"{a:.3f}"
                for a, b in zip(out_df[mc], out_df[sc])
            ]
    print(display.to_string(index=False))

    os.makedirs(args.logs, exist_ok=True)
    csv_path = os.path.join(args.logs, "comparison.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"\nfull numeric table -> {csv_path}")


if __name__ == "__main__":
    main()
