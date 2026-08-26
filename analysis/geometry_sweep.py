"""SP13: does the idqn-beats-green_wave flip (SP8/SP9/SP10) scale smoothly
with spacing asymmetry, or does it jump at some threshold? SP8-SP10 only
sampled 3 discrete geometries, two of them (irregular/irregular2, 600m/100m
and 100m/600m nominal) on a different total arterial span (700m) than the
regular net (400m) -- confounding asymmetry with overall corridor length.

This holds the C1-to-C3 span fixed at 400m (same as corridor.net.xml) and
sweeps the asymmetry ratio r = nominal C1-C2 length / 400 across 8 points:

    r = 0.50 (corridor.net.xml, regular, reused)
        0.55, 0.60, 0.65, 0.70 (new, analysis/build_geometry_sweep_nets.py)
        0.75 (corridor_irregular3.net.xml, SP10, reused)
        0.80, 0.90 (new)

Denser in [0.50, 0.75] because SP10's own irregular3 (r=0.75) already shows
idqn ahead of green_wave while the regular net (r=0.50) shows the opposite --
the flip threshold lies somewhere in that gap and this sweep resolves it.

Zero-shot only, no training: idqn's existing corridor_peak checkpoints (SP5,
seeds 42-44 -- no others exist) are evaluated on each geometry exactly as
SP8/SP10 did. green_wave/max_pressure get seeds 42-46 (5 seeds, no checkpoint
needed). Reuses cached tripinfo for r=0.50 and r=0.75 (already on disk from
SP4 and SP10) rather than re-running them -- same stem convention as
analysis/irregular_net_compare2.py, so the existing files are picked up
automatically.

Deliberately bypasses compare.py -- same reason irregular_net_compare2.py
gives: non-default net_file isn't a tag dimension compare.py's glob knows
about.

    python -m analysis.build_geometry_sweep_nets   # once, builds the 6 new nets
    python -m analysis.geometry_sweep
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd

import corridor_baseline as cb
import train_corridor_dqn as tcd
from analysis.tripinfo import reduce_tripinfo
from env_common import tripinfo_path

OUT_CSV = os.path.join(REPO, "analysis", "geometry_sweep.csv")

SCENARIO = "corridor_peak"
LAM = 0.5
STEPS = 100_000
MIN_GREEN = 10

# ratio -> net file. 0.50/0.75 reuse existing nets; the rest were built by
# build_geometry_sweep_nets.py.
RATIO_NETS = {
    0.50: "corridor.net.xml",
    0.55: "corridor_geom220.net.xml",
    0.60: "corridor_geom240.net.xml",
    0.65: "corridor_geom260.net.xml",
    0.70: "corridor_geom280.net.xml",
    0.75: "corridor_irregular3.net.xml",
    0.80: "corridor_geom320.net.xml",
    0.90: "corridor_geom360.net.xml",
}
BASELINE_SEEDS = (42, 43, 44, 45, 46)   # green_wave/max_pressure -- no checkpoint needed
IDQN_SEEDS = (42, 43, 44)               # only seeds with a corridor_peak checkpoint on disk

os.environ.setdefault("TIME_TO_TELEPORT", "300")


def _run_baseline(controller: str, ratio: float, net_file: str, seed: int,
                  force: bool) -> dict:
    stem = (f"logs/eval_{controller}_{SCENARIO}_seed{seed}_mg{MIN_GREEN}"
            + (f"_net{net_file.removesuffix('.net.xml').removeprefix('corridor_')}"
               if net_file != "corridor.net.xml" else ""))
    trip = tripinfo_path(stem)
    if force or not os.path.exists(trip):
        cb.run(SCENARIO, controller, seed, min_green=MIN_GREEN, tripinfo=True,
              net_file=net_file)
    row = reduce_tripinfo(trip)
    return {"controller": controller, "ratio": ratio, "seed": seed,
            "delay_per_trip": row["trip_time_loss_mean"], "trips": row["trips_completed"]}


def _run_idqn(ratio: float, net_file: str, seed: int, force: bool) -> dict:
    """Zero-shot: corridor_peak-trained IDQN checkpoint (SP5) on this geometry."""
    stem = tcd._eval_out_stem(SCENARIO, SCENARIO, LAM, seed, MIN_GREEN, STEPS,
                              net_file=net_file)
    trip = tripinfo_path(stem)
    if force or not os.path.exists(trip):
        tcd.evaluate(SCENARIO, LAM, seed, MIN_GREEN, STEPS, tripinfo=True,
                    eval_scenario=SCENARIO, net_file=net_file)
    row = reduce_tripinfo(trip)
    return {"controller": "idqn", "ratio": ratio, "seed": seed,
            "delay_per_trip": row["trip_time_loss_mean"], "trips": row["trips_completed"]}


def run_all(force: bool = False) -> pd.DataFrame:
    rows = []
    for controller in ("green_wave", "max_pressure"):
        for ratio, net_file in RATIO_NETS.items():
            for seed in BASELINE_SEEDS:
                rows.append(_run_baseline(controller, ratio, net_file, seed, force))
                r = rows[-1]
                print(f"[{controller}/r{ratio:.2f}] seed{seed} "
                      f"delay/trip={r['delay_per_trip']:7.2f}s trips={r['trips']:5d}",
                      flush=True)
    for ratio, net_file in RATIO_NETS.items():
        for seed in IDQN_SEEDS:
            rows.append(_run_idqn(ratio, net_file, seed, force))
            r = rows[-1]
            print(f"[idqn/r{ratio:.2f}] seed{seed} delay/trip={r['delay_per_trip']:7.2f}s "
                  f"trips={r['trips']:5d}", flush=True)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    print(f"\n=== {SCENARIO}, mg{MIN_GREEN}: delay/trip vs spacing-asymmetry ratio ===")
    summary = df.groupby(["controller", "ratio"])["delay_per_trip"].agg(["mean", "std", "count"])
    print(summary.to_string(float_format=lambda x: f"{x:6.2f}"))

    print("\n--- idqn - green_wave, paired by seed (negative = idqn wins) ---")
    wide = df[df["controller"].isin(["idqn", "green_wave"])].pivot_table(
        index=["ratio", "seed"], columns="controller", values="delay_per_trip")
    for ratio, g in wide.groupby(level="ratio"):
        g = g.dropna()
        if g.empty:
            continue
        delta = g["idqn"] - g["green_wave"]
        winner = "idqn" if delta.mean() < 0 else "green_wave"
        print(f"  r={ratio:.2f}: {delta.mean():+6.2f}s +/- {delta.std():5.2f}s  "
              f"(n={len(g)}, {winner} ahead)")

    print("\n--- flip threshold (linear interpolation on the mean gap) ---")
    means = df[df["controller"].isin(["idqn", "green_wave"])].groupby(
        ["ratio", "controller"])["delay_per_trip"].mean().unstack()
    means = means.dropna().sort_index()
    gap = means["idqn"] - means["green_wave"]  # negative once idqn is ahead
    crossed = False
    for (r0, g0), (r1, g1) in zip(gap.items(), list(gap.items())[1:]):
        if (g0 > 0) != (g1 > 0):
            r_star = r0 + (r1 - r0) * (0 - g0) / (g1 - g0)
            print(f"  crosses between r={r0:.2f} (gap {g0:+.2f}s) and "
                  f"r={r1:.2f} (gap {g1:+.2f}s) -> interpolated flip at r~{r_star:.3f}")
            crossed = True
    if not crossed:
        print("  no sign change found across the sampled ratios")


def main():
    import argparse
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    df = run_all(args.force)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(df)} rows)")
    report(df)


if __name__ == "__main__":
    main()
