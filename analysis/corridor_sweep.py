"""Phase 1: calibrate the corridor's non-RL references before training anything.

This is the corridor's version of analysis/actuated.py + analysis/static_timing.py,
and it exists for the same reason those did. At the single intersection every
learned policy was trained and scored against references that had never been
calibrated, at an action-space floor that a controller with perfect queue
information could not win at either. The result was a headline that had to be
withdrawn. The corridor inherits both baselines and the same floor knob, so it
gets calibrated FIRST.

The floor is not a detail for either reference:

  * green_wave requests a phase change on every decision step, so nothing sets
    its green duration except `min_green`. Sweeping the floor IS sweeping the
    green-wave plan; a single uncalibrated point is not a baseline.
  * max_pressure is reactive, so the floor is a genuine constraint on how fast
    it may respond -- exactly the knob the actuated sweep found to be worth a
    factor of 5.6 at the single junction.

What this answers, before a single training step is spent:

  1. What is the best each reference can do? That is the bar IPPO must clear,
     and it is currently unknown.
  2. Is there a gap between the best fixed-offset plan (green_wave) and the best
     reactive one (max_pressure)? A reactive controller beating a fixed plan is
     evidence that this corridor rewards adaptation at all. If nothing beats the
     fixed plan here, the corridor is the single intersection again and IPPO
     should not be trained until the demand has structure worth adapting to.

Ranking metric is delay per completed trip from tripinfo, never the in-network
average -- see docs/FINDINGS_2026-08-12.md section 1.

    python analysis/corridor_sweep.py --seeds 42 43 44 --min-greens 10 20 30 45 60 75 90
    python analysis/corridor_sweep.py --seeds 42 43 44 45 46 47 48 49 50 51 --min-greens 30 45 60
"""
import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import pandas as pd

import corridor_baseline as cb
from analysis.tripinfo import reduce_tripinfo
from env_common import tripinfo_path

OUT_CSV = os.path.join(REPO, "analysis", "corridor_sweep.csv")

# Teleporting off makes junction deadlock an absorbing state, which turns a
# jammed episode's metrics into a clock rather than a measurement. 300 is SUMO's
# own default and what every corrected single-intersection run used.
os.environ.setdefault("TIME_TO_TELEPORT", "300")


def _stem(controller: str, scenario: str, seed: int, min_green: int) -> str:
    return f"logs/eval_{controller}_{scenario}_seed{seed}_mg{min_green}"


def run_one(controller: str, scenario: str, seed: int, min_green: int,
            force: bool = False) -> dict:
    """One episode, reduced to the ranking metric. Resumable by design.

    Long runs on this machine are interrupted routinely, so an existing
    tripinfo file is reused rather than recomputed -- the sweep can be
    re-invoked until it is complete without repeating finished work.
    """
    stem = _stem(controller, scenario, seed, min_green)
    trip = tripinfo_path(stem)
    if force or not os.path.exists(trip):
        t0 = time.monotonic()
        cb.run(scenario, controller, seed, min_green=min_green, tripinfo=True)
        took = time.monotonic() - t0
    else:
        took = float("nan")          # reused, not re-run
    row = reduce_tripinfo(trip)
    return {
        "controller": controller,
        "scenario": scenario,
        "seed": seed,
        "min_green": min_green,
        "delay_per_trip": row["trip_time_loss_mean"],
        "trips": row["trips_completed"],
        "wall_s": took,
    }


def sweep(controllers, scenario, seeds, min_greens, force=False) -> pd.DataFrame:
    rows = []
    total = len(controllers) * len(seeds) * len(min_greens)
    done = 0
    for mg in min_greens:
        for controller in controllers:
            for seed in seeds:
                rows.append(run_one(controller, scenario, seed, mg, force))
                done += 1
                r = rows[-1]
                took = "reused" if np.isnan(r["wall_s"]) else f"{r['wall_s']:.0f}s"
                print(f"[{done}/{total}] {controller:13s} mg{mg:<3d} seed{seed} "
                      f"delay/trip={r['delay_per_trip']:7.1f}s "
                      f"trips={r['trips']:5d}  ({took})", flush=True)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    """Per-floor means, and the paired reactive-vs-fixed difference per floor.

    Paired on the seed: both controllers see the same demand realisation, so the
    per-seed difference removes the demand variance that made the single
    intersection's unpaired comparisons unresolvable.
    """
    print("\n=== delay per completed trip (s), mean +/- sd over seeds ===")
    piv = df.pivot_table(index="min_green", columns="controller",
                         values="delay_per_trip", aggfunc=["mean", "std", "count"])
    print(piv.round(1).to_string())

    print("\n=== paired: max_pressure - green_wave, per floor ===")
    print("negative = the reactive controller wins, i.e. this corridor rewards adaptation")
    for mg, g in df.groupby("min_green"):
        wide = g.pivot_table(index="seed", columns="controller",
                             values="delay_per_trip")
        if not {"green_wave", "max_pressure"}.issubset(wide.columns):
            continue
        wide = wide.dropna()
        if wide.empty:
            continue
        d = wide["max_pressure"] - wide["green_wave"]
        wins = int((d < 0).sum())
        sd = d.std(ddof=1) if len(d) > 1 else float("nan")
        print(f"  mg {mg:>3d}: {d.mean():+7.1f} +/- {sd:5.1f} s   "
              f"max_pressure wins {wins}/{len(d)} seeds")

    best = df.groupby(["controller", "min_green"])["delay_per_trip"].mean()
    print("\n=== best floor per controller (the bar IPPO must clear) ===")
    for controller in sorted(df["controller"].unique()):
        s = best[controller]
        print(f"  {controller:13s} best mg={s.idxmin():<3d} "
              f"delay/trip={s.min():.1f}s   (worst mg={s.idxmax()}, {s.max():.1f}s)")


def main():
    if "SUMO_HOME" not in os.environ:
        sys.exit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="corridor_peak",
                   choices=["corridor_peak", "corridor_offpeak"])
    p.add_argument("--controllers", nargs="+", default=list(cb.CONTROLLERS),
                   choices=list(cb.CONTROLLERS))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--min-greens", type=int, nargs="+",
                   default=[10, 20, 30, 45, 60, 75, 90])
    p.add_argument("--force", action="store_true",
                   help="re-run episodes that already have a tripinfo file")
    p.add_argument("--report-only", action="store_true",
                   help="re-print the report from the existing CSV, running nothing")
    args = p.parse_args()

    if args.report_only:
        report(pd.read_csv(OUT_CSV))
        return

    df = sweep(args.controllers, args.scenario, args.seeds, args.min_greens,
               args.force)
    if os.path.exists(OUT_CSV):
        old = pd.read_csv(OUT_CSV)
        keys = ["controller", "scenario", "seed", "min_green"]
        df = pd.concat([old, df]).drop_duplicates(subset=keys, keep="last")
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(df)} rows)")
    report(df)


if __name__ == "__main__":
    main()
