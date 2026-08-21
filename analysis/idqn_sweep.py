"""IDQN training/eval at one explicit floor, reduced to delay-per-completed-trip
and paired against both analysis/corridor_sweep.csv's green_wave rows and
analysis/ippo_sweep.csv's ippo rows.

Same methodology as analysis/ippo_sweep.py (tripinfo reduction, seed set
42-51, resumable "reuse what's on disk" design) applied to the true-
independent DQN driver (train_corridor_dqn.py) instead of the parameter-
shared IPPO one.

    python -m analysis.idqn_sweep --scenario corridor_peak --min-green 10 \
        --seeds 42 43 44 --lam 0.5 --steps 100000
"""
import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd

import train_corridor_dqn as tcd
from analysis.tripinfo import reduce_tripinfo
from env_common import CORRIDOR_SCENARIOS, tripinfo_path

OUT_CSV = os.path.join(REPO, "analysis", "idqn_sweep.csv")
CORRIDOR_SWEEP_CSV = os.path.join(REPO, "analysis", "corridor_sweep.csv")
IPPO_SWEEP_CSV = os.path.join(REPO, "analysis", "ippo_sweep.csv")

os.environ.setdefault("TIME_TO_TELEPORT", "300")


def run_one(scenario: str, seed: int, min_green: int, lam: float, steps: int,
           force: bool = False) -> dict:
    """Train (if any of the 3 agent checkpoints is missing) + eval one seed,
    reduced to the ranking metric. Resumable: existing model/tripinfo files
    are reused."""
    paths = [tcd._model_path(a, scenario, lam, seed, min_green, steps)
             for a in tcd.CORRIDOR_TS_IDS]
    if force or not all(os.path.exists(p) for p in paths):
        t0 = time.monotonic()
        tcd.train(scenario, lam, seed, steps, min_green)
        took = time.monotonic() - t0
    else:
        took = float("nan")
    tcd.evaluate(scenario, lam, seed, min_green, steps, tripinfo=True)
    trip = tripinfo_path(f"logs/eval_idqn_{tcd._tag(scenario, lam, seed, min_green, steps)}")
    row = reduce_tripinfo(trip)
    return {
        "controller": "idqn", "scenario": scenario, "seed": seed,
        "min_green": min_green, "delay_per_trip": row["trip_time_loss_mean"],
        "trips": row["trips_completed"], "wall_s": took,
    }


def sweep(scenario: str, seeds, min_green: int, lam: float, steps: int,
         force: bool = False) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        rows.append(run_one(scenario, seed, min_green, lam, steps, force))
        r = rows[-1]
        took = "reused" if pd.isna(r["wall_s"]) else f"{r['wall_s']:.0f}s"
        print(f"[{len(rows)}/{len(seeds)}] idqn seed{seed} "
              f"delay/trip={r['delay_per_trip']:7.1f}s trips={r['trips']:5d}  ({took})",
              flush=True)
    return pd.DataFrame(rows)


def paired_vs(idqn_df: pd.DataFrame, bar_df: pd.DataFrame, bar_name: str) -> dict:
    """idqn - bar_df per seed, paired. Both dataframes must be one (scenario,
    min_green) already -- raises if they disagree, the same cross-scenario
    guard ippo_sweep.paired_vs_green_wave relies on. bar_name labels which
    reference (green_wave / ippo) this call compares against."""
    i_scen = set(idqn_df["scenario"])
    b_scen = set(bar_df["scenario"])
    if i_scen != b_scen or len(i_scen) != 1:
        raise ValueError(f"scenario mismatch: idqn={i_scen} {bar_name}={b_scen}")
    wide = pd.merge(
        idqn_df[["seed", "delay_per_trip"]].rename(columns={"delay_per_trip": "idqn"}),
        bar_df[["seed", "delay_per_trip"]].rename(columns={"delay_per_trip": bar_name}),
        on="seed", how="inner")
    d = wide["idqn"] - wide[bar_name]
    return {
        "scenario": idqn_df["scenario"].iloc[0], "vs": bar_name,
        "mean": float(d.mean()),
        "sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
        "wins": int((d < 0).sum()), "n": int(len(d)),
    }


def load_green_wave_bar(scenario: str, min_green: int) -> pd.DataFrame:
    """green_wave rows already in analysis/corridor_sweep.csv for this
    (scenario, min_green)."""
    df = pd.read_csv(CORRIDOR_SWEEP_CSV)
    return df[(df["controller"] == "green_wave") & (df["scenario"] == scenario) &
              (df["min_green"] == min_green)]


def load_ippo_bar(scenario: str, min_green: int) -> pd.DataFrame:
    """ippo rows already in analysis/ippo_sweep.csv for this
    (scenario, min_green) -- SP4's result, the second reference this sweep
    pairs against."""
    df = pd.read_csv(IPPO_SWEEP_CSV)
    return df[(df["scenario"] == scenario) & (df["min_green"] == min_green)]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=list(CORRIDOR_SCENARIOS))
    p.add_argument("--min-green", type=int, required=True)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")

    df = sweep(args.scenario, args.seeds, args.min_green, args.lam, args.steps, args.force)
    if os.path.exists(OUT_CSV):
        prior = pd.read_csv(OUT_CSV)
        df = pd.concat([prior, df]).drop_duplicates(
            subset=["scenario", "seed", "min_green"], keep="last")
    df.to_csv(OUT_CSV, index=False)

    this_run = df[(df["scenario"] == args.scenario) & (df["min_green"] == args.min_green)]

    gw_bar = load_green_wave_bar(args.scenario, args.min_green)
    if gw_bar.empty:
        print(f"no green_wave rows in {CORRIDOR_SWEEP_CSV} for "
              f"{args.scenario}/mg{args.min_green} -- cannot pair")
    else:
        result = paired_vs(this_run, gw_bar, "green_wave")
        print(f"\nidqn - green_wave, {args.scenario} mg{args.min_green}: "
              f"{result['mean']:+.2f} +/- {result['sd']:.2f} s, "
              f"idqn wins {result['wins']}/{result['n']}")

    ippo_bar = load_ippo_bar(args.scenario, args.min_green)
    if ippo_bar.empty:
        print(f"no ippo rows in {IPPO_SWEEP_CSV} for "
              f"{args.scenario}/mg{args.min_green} -- cannot pair")
    else:
        result = paired_vs(this_run, ippo_bar, "ippo")
        print(f"idqn - ippo, {args.scenario} mg{args.min_green}: "
              f"{result['mean']:+.2f} +/- {result['sd']:.2f} s, "
              f"idqn wins {result['wins']}/{result['n']}")
