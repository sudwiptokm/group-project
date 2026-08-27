"""SP13c follow-up: with span=400 (1 crossing), span=550 (3 crossings,
close to span=700's own locations), and span=700 (3 crossings) sampled
(docs/FINDINGS_2026-08-26-sp13-geometry-dose-response.md,
docs/FINDINGS_2026-08-27-sp13b-span700-confound.md,
docs/FINDINGS_2026-08-27-sp13c-span550.md), the 1-to-3-crossing transition
is bracketed to (400m, 550m]. This adds a fourth point, 450m, near the low
end of that bracket, to narrow it further.

Same 8 ratio points as every other span in this series (r = 0.50, 0.55,
0.60, 0.65, 0.70, 0.75, 0.80, 0.90), same corridor_peak/min_green=10
zero-shot protocol, same seed sets -- only the net files differ
(analysis/build_geometry_sweep_nets_span450.py's span=450 builds, all new;
none of the other spans' nets are reusable here since C2's absolute
position differs for every r at a different span).

Deliberately bypasses compare.py -- same reason geometry_sweep.py gives.

    python -m analysis.build_geometry_sweep_nets_span450   # once
    python -m analysis.geometry_sweep_span450
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

OUT_CSV = os.path.join(REPO, "analysis", "geometry_sweep_span450.csv")

SCENARIO = "corridor_peak"
LAM = 0.5
STEPS = 100_000
MIN_GREEN = 10

RATIO_NETS = {
    0.50: "corridor_geom450_225.net.xml",
    0.55: "corridor_geom450_248.net.xml",
    0.60: "corridor_geom450_270.net.xml",
    0.65: "corridor_geom450_292.net.xml",
    0.70: "corridor_geom450_315.net.xml",
    0.75: "corridor_geom450_338.net.xml",
    0.80: "corridor_geom450_360.net.xml",
    0.90: "corridor_geom450_405.net.xml",
}
BASELINE_SEEDS = (42, 43, 44, 45, 46)
IDQN_SEEDS = (42, 43, 44)

os.environ.setdefault("TIME_TO_TELEPORT", "300")


def _run_baseline(controller: str, ratio: float, net_file: str, seed: int,
                  force: bool) -> dict:
    stem = (f"logs/eval_{controller}_{SCENARIO}_seed{seed}_mg{MIN_GREEN}"
            + f"_net{net_file.removesuffix('.net.xml').removeprefix('corridor_')}")
    trip = tripinfo_path(stem)
    if force or not os.path.exists(trip):
        cb.run(SCENARIO, controller, seed, min_green=MIN_GREEN, tripinfo=True,
              net_file=net_file)
    row = reduce_tripinfo(trip)
    return {"controller": controller, "ratio": ratio, "seed": seed,
            "delay_per_trip": row["trip_time_loss_mean"], "trips": row["trips_completed"]}


def _run_idqn(ratio: float, net_file: str, seed: int, force: bool) -> dict:
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
    print(f"\n=== {SCENARIO}, mg{MIN_GREEN}, span=450: delay/trip vs ratio ===")
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
    gap = means["idqn"] - means["green_wave"]
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
