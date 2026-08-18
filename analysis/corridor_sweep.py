"""Phase 1: calibrate the corridor's non-RL references before training anything.

This is the corridor's version of analysis/actuated.py + analysis/static_timing.py,
and it exists for the same reason those did. At the single intersection every
learned policy was trained and scored against references that had never been
calibrated, at an action-space floor that a controller with perfect queue
information could not win at either. The result was a headline that had to be
withdrawn. The corridor inherits both baselines and the same floor knob, so it
gets calibrated FIRST.

The floor is not a detail for either reference:

  * green_wave's phase duration IS the floor (rounded up to the decision grid),
    so sweeping the floor is sweeping the green-wave plan itself; a single
    uncalibrated point is not a baseline.
  * max_pressure is reactive, so the floor is a genuine constraint on how fast
    it may respond -- exactly the knob the actuated sweep found to be worth a
    factor of 5.6 at the single junction.

Neither controller acts on the floor directly: a change is honoured at the first
decision step at or after min_green + yellow_time. floor_aliases() reports that
mapping so the curve is read on the axis the controllers actually see.

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
import corridor_control as cc
from analysis.tripinfo import reduce_tripinfo
from env_common import (CORRIDOR_DELTA_TIME, CORRIDOR_SCENARIOS,
                        CORRIDOR_YELLOW_TIME, tripinfo_path)

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


def floor_aliases(min_greens, yellow_time: int = None,
                  delta_time: int = None) -> dict:
    """{effective switching interval (s): [floors that produce it]}.

    min_green is not the quantity either controller acts on. A phase change is
    honoured at the first decision step at or after min_green + yellow_time, so
    what governs both references is that sum rounded up to the decision grid --
    which is exactly corridor_control.plan_phase_seconds, the green_wave plan's
    phase duration.

    Any entry with more than one floor is a floor the sweep is sampling twice
    under two names. Reporting those as separate points overstates the
    resolution of the floor curve, and a "best floor" chosen among them is
    arbitrary between equals.
    """
    yellow_time = CORRIDOR_YELLOW_TIME if yellow_time is None else yellow_time
    delta_time = CORRIDOR_DELTA_TIME if delta_time is None else delta_time
    out = {}
    for mg in sorted(min_greens):
        out.setdefault(cc.plan_phase_seconds(int(mg), yellow_time, delta_time),
                       []).append(int(mg))
    return out


def best_floors(df: pd.DataFrame) -> dict:
    """{(scenario, controller): (best min_green, its mean delay per trip)}."""
    means = df.groupby(["scenario", "controller", "min_green"])["delay_per_trip"].mean()
    out = {}
    for (scenario, controller), s in means.groupby(level=[0, 1]):
        s = s.droplevel([0, 1])
        out[(scenario, controller)] = (int(s.idxmin()), float(s.min()))
    return out


def paired_diffs(df: pd.DataFrame) -> list:
    """max_pressure - green_wave per (scenario, floor), paired on the seed.

    Paired on the seed: both controllers see the same demand realisation, so the
    per-seed difference removes the demand variance that made the single
    intersection's unpaired comparisons unresolvable. Pairing NEVER crosses
    scenarios -- seed 42 under corridor_peak and seed 42 under corridor_tidal
    are different demands and differencing them means nothing.
    """
    rows = []
    for (scenario, mg), g in df.groupby(["scenario", "min_green"]):
        wide = g.pivot_table(index="seed", columns="controller",
                             values="delay_per_trip")
        if not {"green_wave", "max_pressure"}.issubset(wide.columns):
            continue
        wide = wide.dropna()
        if wide.empty:
            continue
        d = wide["max_pressure"] - wide["green_wave"]
        rows.append({
            "scenario": scenario,
            "min_green": int(mg),
            "mean": float(d.mean()),
            "sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
            "wins": int((d < 0).sum()),
            "n": int(len(d)),
        })
    return rows


# two controllers whose completed-trip counts differ by more than this at the
# same floor are not being ranked on the same population of vehicles
COMPLETION_TOLERANCE = 0.02


def completion_gaps(df: pd.DataFrame) -> list:
    """Floors where the controllers completed materially different trip counts.

    delay_per_trip is delay per COMPLETED trip. A controller that jams an
    approach until hundreds of vehicles never finish is scored on the survivors
    and can look better than one that cleared everybody -- the survivorship bias
    behind this project's withdrawn headline. corridor_tidal loads the dominant
    direction past what an even green split can discharge on purpose, so this is
    an expected condition there, not a hypothetical one.
    """
    gaps = []
    for (scenario, mg), g in df.groupby(["scenario", "min_green"]):
        trips = g.groupby("controller")["trips"].mean()
        if len(trips) < 2 or trips.max() <= 0:
            continue
        spread = (trips.max() - trips.min()) / trips.max()
        if spread > COMPLETION_TOLERANCE:
            gaps.append({
                "scenario": scenario,
                "min_green": int(mg),
                "spread": float(spread),
                "trips": {c: float(n) for c, n in trips.items()},
            })
    return gaps


def report(df: pd.DataFrame) -> None:
    """Per-floor means and paired differences, one block per scenario.

    Everything here is grouped by scenario first. The CSV accumulates every
    scenario ever swept, and a table that pools them reports a mean over two
    different demands and a best floor that belongs to neither.
    """
    diffs = paired_diffs(df)
    best = best_floors(df)
    gaps = completion_gaps(df)

    for scenario, sdf in df.groupby("scenario"):
        print(f"\n################ {scenario} ################")
        print("\n=== delay per completed trip (s), mean +/- sd over seeds ===")
        piv = sdf.pivot_table(index="min_green", columns="controller",
                              values="delay_per_trip",
                              aggfunc=["mean", "std", "count"])
        print(piv.round(1).to_string())

        aliases = floor_aliases(sorted(sdf["min_green"].unique()))
        print("\n=== floor -> effective switching interval (s) ===")
        print("    what the signal acts on is min_green + yellow rounded up to the "
              "decision grid,")
        print("    so this, not the floor label, is the curve's x-axis and its "
              "resolution limit.")
        line = "   " + "  ".join(f"mg {mg:>2d} -> {secs}s"
                                 for secs in sorted(aliases)
                                 for mg in aliases[secs])
        print(line)
        collapsed = {s: f for s, f in aliases.items() if len(f) > 1}
        if collapsed:
            print("    !!! aliased floors -- these are one point sampled under several "
                  "names:")
            for secs, floors in sorted(collapsed.items()):
                print(f"        {secs}s <- mg {floors}")

        print("\n=== trips completed, mean over seeds ===")
        print(sdf.pivot_table(index="min_green", columns="controller",
                              values="trips", aggfunc="mean").round(0).to_string())

        here = [g for g in gaps if g["scenario"] == scenario]
        if here:
            print("\n!!! survivorship warning: these floors rank the controllers on "
                  "different populations of vehicles.")
            print("    delay per COMPLETED trip favours whoever finished fewer of them.")
            for g in sorted(here, key=lambda g: g["min_green"]):
                counts = "  ".join(f"{c}={n:.0f}" for c, n in sorted(g["trips"].items()))
                print(f"    mg {g['min_green']:>3d}: {g['spread'] * 100:.1f}% spread   {counts}")

        print("\n=== paired: max_pressure - green_wave, per floor ===")
        print("negative = the reactive controller wins, i.e. this scenario rewards adaptation")
        for r in sorted((r for r in diffs if r["scenario"] == scenario),
                        key=lambda r: r["min_green"]):
            print(f"  mg {r['min_green']:>3d}: {r['mean']:+7.1f} +/- {r['sd']:5.1f} s   "
                  f"max_pressure wins {r['wins']}/{r['n']} seeds")

        print("\n=== best floor per controller (the bar IPPO must clear) ===")
        worst = sdf.groupby(["controller", "min_green"])["delay_per_trip"].mean()
        for controller in sorted(sdf["controller"].unique()):
            mg, delay = best[(scenario, controller)]
            w = worst[controller]
            print(f"  {controller:13s} best mg={mg:<3d} delay/trip={delay:.1f}s   "
                  f"(worst mg={w.idxmax()}, {w.max():.1f}s)")


def main():
    if "SUMO_HOME" not in os.environ:
        sys.exit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="corridor_peak",
                   choices=list(CORRIDOR_SCENARIOS))
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
