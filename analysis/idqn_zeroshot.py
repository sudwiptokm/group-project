"""SP6: IDQN zero-shot demand-shift generalization.

Evaluates the 9 existing corridor_peak-trained IDQN checkpoints (SP5) on
demand scenarios they never trained on -- corridor_offpeak, corridor_tidal,
corridor_skew -- and pairs each shifted scenario's result against the
existing green_wave/max_pressure rows in analysis/corridor_sweep.csv.

No training happens here. See
docs/superpowers/specs/2026-08-22-sp6-idqn-demand-shift-design.md.

    python -m analysis.idqn_zeroshot
    python -m analysis.idqn_zeroshot --scenarios corridor_tidal --seeds 42
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd

import train_corridor_dqn as tcd
from analysis.tripinfo import reduce_tripinfo
from env_common import CORRIDOR_SCENARIOS, tripinfo_path

CORRIDOR_SWEEP_CSV = os.path.join(REPO, "analysis", "corridor_sweep.csv")
OUT_CSV = os.path.join(REPO, "analysis", "idqn_zeroshot.csv")

TRAIN_SCENARIO = "corridor_peak"
SHIFTED_SCENARIOS = ("corridor_offpeak", "corridor_tidal", "corridor_skew")
SEEDS = (42, 43, 44)
LAM = 0.5
STEPS = 100_000
MIN_GREEN = 10

os.environ.setdefault("TIME_TO_TELEPORT", "300")

# SP5's in-distribution reference: idqn - green_wave on corridor_peak, same
# seeds, same checkpoints (docs/FINDINGS_2026-08-21-sp5-idqn-vs-corrected-bar.md).
INDIST_GAP = {"scenario": TRAIN_SCENARIO, "mean": 3.09, "sd": 0.37, "wins": 0, "n": 3}


def run_one(eval_scenario: str, seed: int, force: bool = False) -> dict:
    """Zero-shot eval: load the corridor_peak-trained seed's checkpoint, run
    it on eval_scenario's demand. Resumable -- an existing tripinfo file for
    this exact (eval_scenario, seed) is reused."""
    stem = tcd._eval_out_stem(TRAIN_SCENARIO, eval_scenario, LAM, seed, MIN_GREEN, STEPS)
    trip = tripinfo_path(stem)
    if force or not os.path.exists(trip):
        tcd.evaluate(TRAIN_SCENARIO, LAM, seed, MIN_GREEN, STEPS, tripinfo=True,
                     eval_scenario=eval_scenario)
    row = reduce_tripinfo(trip)
    return {
        "scenario": eval_scenario, "seed": seed, "min_green": MIN_GREEN,
        "delay_per_trip": row["trip_time_loss_mean"], "trips": row["trips_completed"],
    }


def zeroshot_sweep(scenarios=SHIFTED_SCENARIOS, seeds=SEEDS, force: bool = False) -> pd.DataFrame:
    rows = []
    total = len(scenarios) * len(seeds)
    for scenario in scenarios:
        for seed in seeds:
            rows.append(run_one(scenario, seed, force))
            r = rows[-1]
            print(f"[{len(rows)}/{total}] idqn zero-shot {scenario} seed{seed} "
                  f"delay/trip={r['delay_per_trip']:7.1f}s trips={r['trips']:5d}",
                  flush=True)
    return pd.DataFrame(rows)


def load_baseline(controller: str, scenario: str, min_green: int = MIN_GREEN) -> pd.DataFrame:
    """green_wave/max_pressure rows already in analysis/corridor_sweep.csv for
    this (controller, scenario, min_green)."""
    df = pd.read_csv(CORRIDOR_SWEEP_CSV)
    return df[(df["controller"] == controller) & (df["scenario"] == scenario) &
              (df["min_green"] == min_green)]


def paired_gap(idqn_df: pd.DataFrame, bar_df: pd.DataFrame) -> dict:
    """idqn - bar_df per seed, paired. Both dataframes must be one scenario --
    raises if they disagree, same cross-scenario guard
    analysis.idqn_sweep.paired_vs enforces."""
    i_scen = set(idqn_df["scenario"])
    b_scen = set(bar_df["scenario"])
    if i_scen != b_scen or len(i_scen) != 1:
        raise ValueError(f"scenario mismatch: idqn={i_scen} bar={b_scen}")
    wide = pd.merge(
        idqn_df[["seed", "delay_per_trip"]].rename(columns={"delay_per_trip": "idqn"}),
        bar_df[["seed", "delay_per_trip"]].rename(columns={"delay_per_trip": "bar"}),
        on="seed", how="inner")
    d = wide["idqn"] - wide["bar"]
    return {
        "scenario": idqn_df["scenario"].iloc[0],
        "mean": float(d.mean()),
        "sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
        "wins": int((d < 0).sum()), "n": int(len(d)),
    }


def report(zeroshot_df: pd.DataFrame) -> None:
    print(f"\n=== reference: idqn in-distribution, {INDIST_GAP['scenario']} (SP5) ===")
    print(f"  gap vs green_wave: {INDIST_GAP['mean']:+.2f} +/- {INDIST_GAP['sd']:.2f} s, "
          f"idqn wins {INDIST_GAP['wins']}/{INDIST_GAP['n']}")

    for scenario, g in zeroshot_df.groupby("scenario"):
        print(f"\n################ {scenario} (zero-shot) ################")
        gw = load_baseline("green_wave", scenario)
        mp = load_baseline("max_pressure", scenario)
        if gw.empty:
            print(f"  [!] no green_wave baseline for {scenario}/mg{MIN_GREEN} -- cannot pair")
            continue
        gap = paired_gap(g, gw)
        print(f"  idqn - green_wave: {gap['mean']:+.2f} +/- {gap['sd']:.2f} s, "
              f"idqn wins {gap['wins']}/{gap['n']}  "
              f"(in-distribution reference: {INDIST_GAP['mean']:+.2f}s)")
        if not mp.empty:
            mp_wide = pd.merge(
                mp[["seed", "delay_per_trip"]].rename(columns={"delay_per_trip": "mp"}),
                gw[["seed", "delay_per_trip"]].rename(columns={"delay_per_trip": "gw"}),
                on="seed", how="inner")
            mp_delta = float((mp_wide["mp"] - mp_wide["gw"]).mean())
            print(f"  max_pressure - green_wave (same scenario, for the "
                  f"'harder for everyone' check): {mp_delta:+.2f}s")
        else:
            print(f"  [!] no max_pressure baseline for {scenario}/mg{MIN_GREEN}")


def main():
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", default=list(SHIFTED_SCENARIOS),
                   choices=list(CORRIDOR_SCENARIOS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    df = zeroshot_sweep(args.scenarios, args.seeds, args.force)
    if os.path.exists(OUT_CSV):
        prior = pd.read_csv(OUT_CSV)
        df = pd.concat([prior, df]).drop_duplicates(subset=["scenario", "seed"], keep="last")
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(df)} rows)")
    report(df)


if __name__ == "__main__":
    main()
