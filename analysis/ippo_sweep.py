"""IPPO training/eval at one explicit floor, reduced to delay-per-completed-trip
and paired against analysis/corridor_sweep.csv's green_wave rows.

This is corridor_sweep.py's methodology applied to a learned controller instead
of a reference: same tripinfo reduction (docs/FINDINGS_2026-08-12.md section 1),
same seed set (42-51, so the pairing lines up with the rows corridor_sweep.py
already produced), same "resumable, reuse what's on disk" design so an
interrupted local run picks back up.

    python -m analysis.ippo_sweep --scenario corridor_peak --min-green 10 \
        --seeds 42 43 44 45 46 47 48 49 50 51 --lam 0.5 --steps 100000
"""
import argparse
import os
import sys
import time
from typing import Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd

import train_corridor as tc
from analysis.tripinfo import reduce_tripinfo
from env_common import CORRIDOR_SCENARIOS, tripinfo_path

OUT_CSV = os.path.join(REPO, "analysis", "ippo_sweep.csv")
CORRIDOR_SWEEP_CSV = os.path.join(REPO, "analysis", "corridor_sweep.csv")

os.environ.setdefault("TIME_TO_TELEPORT", "300")


def run_one(scenario: str, seed: int, min_green: int, lam: float, steps: int,
           force: bool = False) -> dict:
    """Train (if no checkpoint exists) + eval one seed, reduced to the ranking
    metric. Resumable: an existing model/tripinfo file is reused."""
    model_path = f"models/ippo_{tc._tag(scenario, lam, seed, min_green, steps)}.pt"
    if force or not os.path.exists(model_path):
        t0 = time.monotonic()
        tc.train(scenario, lam, seed, steps, min_green)
        took = time.monotonic() - t0
    else:
        took = float("nan")
    tc.evaluate(model_path, scenario, lam, seed, min_green, steps, tripinfo=True)
    trip = tripinfo_path(f"logs/eval_ippo_{tc._tag(scenario, lam, seed, min_green, steps)}")
    row = reduce_tripinfo(trip)
    return {
        "controller": "ippo", "scenario": scenario, "seed": seed,
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
        print(f"[{len(rows)}/{len(seeds)}] ippo seed{seed} "
              f"delay/trip={r['delay_per_trip']:7.1f}s trips={r['trips']:5d}  ({took})",
              flush=True)
    return pd.DataFrame(rows)


def paired_vs_green_wave(ippo_df: pd.DataFrame, baseline_df: pd.DataFrame) -> dict:
    """ippo - green_wave per seed, paired. Both dataframes must be one
    (scenario, min_green) already -- raises if they disagree, the same
    cross-scenario-pairing guard corridor_sweep.paired_diffs relies on."""
    i_scen = set(ippo_df["scenario"])
    b_scen = set(baseline_df["scenario"])
    if i_scen != b_scen or len(i_scen) != 1:
        raise ValueError(f"scenario mismatch: ippo={i_scen} baseline={b_scen}")
    wide = pd.merge(
        ippo_df[["seed", "delay_per_trip"]].rename(columns={"delay_per_trip": "ippo"}),
        baseline_df[["seed", "delay_per_trip"]].rename(columns={"delay_per_trip": "green_wave"}),
        on="seed", how="inner")
    d = wide["ippo"] - wide["green_wave"]
    return {
        "scenario": ippo_df["scenario"].iloc[0],
        "mean": float(d.mean()),
        "sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
        "wins": int((d < 0).sum()),
        "n": int(len(d)),
    }


# same survivorship-bias tolerance as corridor_sweep.COMPLETION_TOLERANCE:
# two controllers whose completed-trip counts differ by more than this at the
# same floor are not being ranked on the same population of vehicles
COMPLETION_TOLERANCE = 0.02


def completion_gap(ippo_df: pd.DataFrame, baseline_df: pd.DataFrame,
                   tolerance: float = COMPLETION_TOLERANCE) -> Optional[dict]:
    """Survivorship guard for the ippo/green_wave pairing -- the analogue of
    corridor_sweep.completion_gaps for this file's controller/baseline shape.

    delay_per_trip is delay per COMPLETED trip. A controller that jams an
    approach until hundreds of vehicles never finish is scored on the
    survivors and can look better than one that cleared everybody -- the
    survivorship bias behind this project's withdrawn headline (see
    corridor_sweep.completion_gaps' docstring). Both dataframes must already
    be one (scenario, min_green), the same guard paired_vs_green_wave uses.

    Returns None if mean trips completed are within `tolerance` of each
    other, else a dict describing the spread.
    """
    i_scen = set(ippo_df["scenario"])
    b_scen = set(baseline_df["scenario"])
    if i_scen != b_scen or len(i_scen) != 1:
        raise ValueError(f"scenario mismatch: ippo={i_scen} baseline={b_scen}")
    trips = {
        "ippo": float(ippo_df["trips"].mean()),
        "green_wave": float(baseline_df["trips"].mean()),
    }
    hi = max(trips.values())
    if hi <= 0:
        return None
    spread = (hi - min(trips.values())) / hi
    if spread <= tolerance:
        return None
    return {
        "scenario": ippo_df["scenario"].iloc[0],
        "spread": spread,
        "trips": trips,
    }


def load_green_wave_bar(scenario: str, min_green: int) -> pd.DataFrame:
    """green_wave rows already in analysis/corridor_sweep.csv for this
    (scenario, min_green) -- the bar this sweep pairs IPPO against."""
    df = pd.read_csv(CORRIDOR_SWEEP_CSV)
    return df[(df["controller"] == "green_wave") & (df["scenario"] == scenario) &
              (df["min_green"] == min_green)]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True,
                   choices=list(CORRIDOR_SCENARIOS))
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

    bar = load_green_wave_bar(args.scenario, args.min_green)
    if bar.empty:
        print(f"no green_wave rows in {CORRIDOR_SWEEP_CSV} for "
              f"{args.scenario}/mg{args.min_green} -- cannot pair")
    else:
        this_run = df[(df["scenario"] == args.scenario) &
                      (df["min_green"] == args.min_green)]
        result = paired_vs_green_wave(this_run, bar)
        print(f"\nippo - green_wave, {args.scenario} mg{args.min_green}: "
              f"{result['mean']:+.2f} +/- {result['sd']:.2f} s, "
              f"ippo wins {result['wins']}/{result['n']}")

        gap = completion_gap(this_run, bar)
        if gap is not None:
            counts = "  ".join(f"{c}={n:.0f}" for c, n in gap["trips"].items())
            print(f"\n!!! survivorship warning: ippo and green_wave completed "
                  f"different numbers of trips at this floor.")
            print(f"    delay per COMPLETED trip favours whoever finished fewer "
                  f"of them.")
            print(f"    {gap['spread'] * 100:.1f}% spread   {counts}")
