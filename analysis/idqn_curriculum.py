"""SP11: does a magnitude-diverse training curriculum close IDQN's
corridor_offpeak generalization gap SP6 found?

SP6 (docs/FINDINGS_2026-08-22-sp6-idqn-demand-shift.md) trained IDQN once on
corridor_peak and found its zero-shot gap to green_wave more than triples on
corridor_offpeak (+11.26s vs the in-distribution +3.09s) -- a demand
*magnitude* the checkpoint never saw during training, not a structural shift
(corridor_tidal/corridor_skew's gaps held). This module trains a second IDQN
checkpoint across train_corridor_dqn.CURRICULUM_ROUTES -- a 5-point demand-
magnitude curriculum spanning corridor_offpeak's 0.5x to corridor_peak's
1.5x, randomized per episode (train_corridor_dqn.train_curriculum) -- and
evaluates it zero-shot on corridor_offpeak and corridor_peak, paired against
the same green_wave/max_pressure rows in analysis/corridor_sweep.csv SP6
used, to see whether seeing a range of magnitudes during training closes,
narrows, or leaves unchanged the offpeak gap.

No structural-shift scenarios (corridor_tidal/corridor_skew) are re-tested
here -- SP6 already found those gaps held for the peak-only checkpoint, and
this curriculum only varies demand magnitude, not shape, so there is no
reason to expect that result to move; re-running it would not test this
curriculum's hypothesis.

    python -m analysis.idqn_curriculum --train
    python -m analysis.idqn_curriculum --eval
    python -m analysis.idqn_curriculum --train --eval --seeds 42
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd

import train_corridor_dqn as tcd
from analysis.idqn_zeroshot import load_baseline, paired_gap
from analysis.tripinfo import reduce_tripinfo
from env_common import tripinfo_path

OUT_CSV = os.path.join(REPO, "analysis", "idqn_curriculum.csv")

SEEDS = (42, 43, 44)
LAM = 0.5
STEPS = 100_000
MIN_GREEN = 10
EVAL_SCENARIOS = ("corridor_offpeak", "corridor_peak")

# SP6's peak-only-trained idqn zero-shot reference, same seeds/lam/min_green
# -- analysis/idqn_zeroshot.csv and
# docs/FINDINGS_2026-08-22-sp6-idqn-demand-shift.md's results table. This is
# the "before" arm of the comparison; not re-measured here (SP6 already
# measured it on the exact same checkpoints/seeds this project still has).
PEAK_ONLY_GAP = {
    "corridor_offpeak": {"mean": 11.26, "sd": 0.42, "wins": 0, "n": 3},
    "corridor_peak": {"mean": 3.09, "sd": 0.37, "wins": 0, "n": 3},
}

os.environ.setdefault("TIME_TO_TELEPORT", "300")


def train_all(seeds=SEEDS, force: bool = False) -> None:
    """Train one curriculum checkpoint per seed, skipping seeds whose
    checkpoint already exists on disk (resumable, same convention
    idqn_zeroshot.run_one's tripinfo check follows -- checked via C1's file
    since train_curriculum() writes all 3 agents' checkpoints atomically
    within one call, so C1 present implies C2/C3 are too)."""
    for seed in seeds:
        path0 = tcd._model_path(tcd.CORRIDOR_TS_IDS[0], tcd.CURRICULUM_TAG, LAM,
                                seed, MIN_GREEN, STEPS)
        if force or not os.path.exists(path0):
            print(f"[train] curriculum seed {seed} ...", flush=True)
            tcd.train_curriculum(LAM, seed, STEPS, MIN_GREEN)
        else:
            print(f"[train] seed {seed} checkpoint exists, skipping ({path0})")


def run_one(eval_scenario: str, seed: int, force: bool = False) -> dict:
    """Zero-shot eval: load the curriculum-trained seed's checkpoint, run it
    on eval_scenario's demand. Resumable -- an existing tripinfo file for
    this exact (eval_scenario, seed) is reused."""
    stem = tcd._eval_out_stem(tcd.CURRICULUM_TAG, eval_scenario, LAM, seed,
                              MIN_GREEN, STEPS)
    trip = tripinfo_path(stem)
    if force or not os.path.exists(trip):
        tcd.evaluate(tcd.CURRICULUM_TAG, LAM, seed, MIN_GREEN, STEPS, tripinfo=True,
                     eval_scenario=eval_scenario)
    row = reduce_tripinfo(trip)
    return {
        "scenario": eval_scenario, "seed": seed, "min_green": MIN_GREEN,
        "delay_per_trip": row["trip_time_loss_mean"], "trips": row["trips_completed"],
    }


def eval_sweep(scenarios=EVAL_SCENARIOS, seeds=SEEDS, force: bool = False) -> pd.DataFrame:
    rows = []
    total = len(scenarios) * len(seeds)
    for scenario in scenarios:
        for seed in seeds:
            rows.append(run_one(scenario, seed, force))
            r = rows[-1]
            print(f"[{len(rows)}/{total}] curriculum idqn zero-shot {scenario} "
                  f"seed{seed} delay/trip={r['delay_per_trip']:7.1f}s "
                  f"trips={r['trips']:5d}", flush=True)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    for scenario, g in df.groupby("scenario"):
        print(f"\n################ {scenario} ################")
        gw = load_baseline("green_wave", scenario)
        mp = load_baseline("max_pressure", scenario)
        if gw.empty:
            print(f"  [!] no green_wave baseline for {scenario}/mg{MIN_GREEN} -- cannot pair")
            continue
        gap = paired_gap(g, gw)
        print(f"  curriculum idqn - green_wave: {gap['mean']:+.2f} +/- {gap['sd']:.2f} s, "
              f"wins {gap['wins']}/{gap['n']}")
        ref = PEAK_ONLY_GAP.get(scenario)
        if ref:
            print(f"  peak-only idqn  - green_wave (SP6): {ref['mean']:+.2f} +/- "
                  f"{ref['sd']:.2f} s, wins {ref['wins']}/{ref['n']}")
            moved = ref["mean"] - gap["mean"]
            pct = 100.0 * moved / ref["mean"] if ref["mean"] else float("nan")
            direction = "closed/narrowed" if moved > 0 else "widened/worsened"
            print(f"  curriculum vs peak-only: {moved:+.2f} s ({pct:+.0f}%) -- {direction}")
        if not mp.empty:
            mean_idqn = g["delay_per_trip"].mean()
            mean_mp = mp["delay_per_trip"].mean()
            print(f"  curriculum idqn absolute: {mean_idqn:.2f}s | "
                  f"green_wave: {gw['delay_per_trip'].mean():.2f}s | "
                  f"max_pressure: {mean_mp:.2f}s")


def main():
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--scenarios", nargs="+", default=list(EVAL_SCENARIOS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.train:
        train_all(args.seeds, args.force)

    if args.eval:
        df = eval_sweep(args.scenarios, args.seeds, args.force)
        if os.path.exists(OUT_CSV):
            prior = pd.read_csv(OUT_CSV)
            df = pd.concat([prior, df]).drop_duplicates(subset=["scenario", "seed"], keep="last")
        df.to_csv(OUT_CSV, index=False)
        print(f"\nwrote {OUT_CSV} ({len(df)} rows)")
        report(df)


if __name__ == "__main__":
    main()
