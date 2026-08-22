"""Does breaking the corridor's regular 200m/200m arterial spacing change
which controller wins?

green_wave's offset is computed exactly (cc.green_wave_offsets), so uneven
block lengths don't break it outright -- but a single offset per signal can
only truly progress ONE direction; equal spacing lets it serve both
directions' shared through-phase well by accident. corridor_irregular.net.xml
(C1@0, C2@600, C3@700, vs the regular C1@0, C2@200, C3@400) stresses exactly
that: same topology, same demand, same controllers, only the geometry
changes. See corridor_irregular.nod.xml's own comment for the reasoning.

No training happens here -- IDQN's existing corridor_peak checkpoints (SP5)
are evaluated zero-shot on the new geometry, same posture as SP6's
demand-shift eval and SP7's incident eval. green_wave/max_pressure regular-net
reference numbers are read from analysis/corridor_sweep.csv (already computed,
SP4); only the irregular-net runs are new.

Deliberately bypasses compare.py: this is a new axis (network geometry) that
compare.py's glob (`eval_{entity}_{scenario}_seed*.csv`) has no tag for, and
retrofitting one risks the exact glob-confound bug class SP6 and SP7 both hit
and had to fix twice. Reading each run's tripinfo XML directly by its exact
path sidesteps that class of bug entirely, at the cost of not feeding
compare.py's other reports -- an acceptable trade for a single-purpose probe.

    python -m analysis.irregular_net_compare
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

CORRIDOR_SWEEP_CSV = os.path.join(REPO, "analysis", "corridor_sweep.csv")
OUT_CSV = os.path.join(REPO, "analysis", "irregular_net_compare.csv")

NET_REGULAR = "corridor.net.xml"
NET_IRREGULAR = "corridor_irregular.net.xml"
SCENARIO = "corridor_peak"
SEEDS = (42, 43, 44)
LAM = 0.5
STEPS = 100_000
MIN_GREEN = 10

os.environ.setdefault("TIME_TO_TELEPORT", "300")


def _regular_reference(controller: str) -> pd.DataFrame:
    """green_wave/max_pressure's already-computed regular-net corridor_peak
    numbers (SP4), filtered to this eval's seeds -- avoids re-running a
    baseline whose regular-net result is already on disk."""
    df = pd.read_csv(CORRIDOR_SWEEP_CSV)
    df = df[(df["controller"] == controller) & (df["scenario"] == SCENARIO) &
            (df["min_green"] == MIN_GREEN) & (df["seed"].isin(SEEDS))]
    return df[["seed", "delay_per_trip", "trips"]].sort_values("seed")


def _run_irregular_baseline(controller: str, seed: int, force: bool) -> dict:
    """One green_wave/max_pressure run on the irregular net."""
    stem = (f"logs/eval_{controller}_{SCENARIO}_seed{seed}_mg{MIN_GREEN}"
            f"_net{NET_IRREGULAR.removesuffix('.net.xml').removeprefix('corridor_')}")
    trip = tripinfo_path(stem)
    if force or not os.path.exists(trip):
        cb.run(SCENARIO, controller, seed, min_green=MIN_GREEN, tripinfo=True,
              net_file=NET_IRREGULAR)
    row = reduce_tripinfo(trip)
    return {"controller": controller, "net": "irregular", "seed": seed,
            "delay_per_trip": row["trip_time_loss_mean"], "trips": row["trips_completed"]}


def _run_irregular_idqn(seed: int, force: bool) -> dict:
    """Zero-shot: corridor_peak-trained IDQN checkpoint on the irregular net."""
    stem = tcd._eval_out_stem(SCENARIO, SCENARIO, LAM, seed, MIN_GREEN, STEPS,
                              net_file=NET_IRREGULAR)
    trip = tripinfo_path(stem)
    if force or not os.path.exists(trip):
        tcd.evaluate(SCENARIO, LAM, seed, MIN_GREEN, STEPS, tripinfo=True,
                    eval_scenario=SCENARIO, net_file=NET_IRREGULAR)
    row = reduce_tripinfo(trip)
    return {"controller": "idqn", "net": "irregular", "seed": seed,
            "delay_per_trip": row["trip_time_loss_mean"], "trips": row["trips_completed"]}


def _run_regular_idqn(seed: int, force: bool) -> dict:
    """IDQN's in-distribution regular-net corridor_peak number (SP5's own
    checkpoints/tripinfo -- idqn has no row in corridor_sweep.csv, which only
    holds the non-RL baselines, per that file's own design)."""
    stem = tcd._eval_out_stem(SCENARIO, SCENARIO, LAM, seed, MIN_GREEN, STEPS)
    trip = tripinfo_path(stem)
    if force or not os.path.exists(trip):
        tcd.evaluate(SCENARIO, LAM, seed, MIN_GREEN, STEPS, tripinfo=True,
                    eval_scenario=SCENARIO)
    row = reduce_tripinfo(trip)
    return {"controller": "idqn", "net": "regular", "seed": seed,
            "delay_per_trip": row["trip_time_loss_mean"], "trips": row["trips_completed"]}


def run_all(seeds=SEEDS, force: bool = False) -> pd.DataFrame:
    rows = []
    for controller in ("green_wave", "max_pressure"):
        ref = _regular_reference(controller)
        for _, r in ref.iterrows():
            rows.append({"controller": controller, "net": "regular", "seed": int(r["seed"]),
                        "delay_per_trip": r["delay_per_trip"], "trips": int(r["trips"])})
        for seed in seeds:
            rows.append(_run_irregular_baseline(controller, seed, force))
            r = rows[-1]
            print(f"[{controller}/irregular] seed{seed} delay/trip={r['delay_per_trip']:7.2f}s "
                  f"trips={r['trips']:5d}", flush=True)
    for seed in seeds:
        rows.append(_run_regular_idqn(seed, force))
        rows.append(_run_irregular_idqn(seed, force))
        print(f"[idqn/regular]   seed{seed} delay/trip={rows[-2]['delay_per_trip']:7.2f}s "
              f"trips={rows[-2]['trips']:5d}", flush=True)
        print(f"[idqn/irregular] seed{seed} delay/trip={rows[-1]['delay_per_trip']:7.2f}s "
              f"trips={rows[-1]['trips']:5d}", flush=True)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    print(f"\n=== {SCENARIO}, mg{MIN_GREEN}, seeds {list(SEEDS)}: regular vs irregular spacing ===")
    summary = df.groupby(["controller", "net"])["delay_per_trip"].agg(["mean", "std"])
    print(summary.to_string(float_format=lambda x: f"{x:6.2f}"))

    print("\n--- per-controller shift (irregular - regular), paired by seed ---")
    for controller, g in df.groupby("controller"):
        wide = g.pivot(index="seed", columns="net", values="delay_per_trip")
        if "regular" not in wide or "irregular" not in wide:
            continue
        delta = wide["irregular"] - wide["regular"]
        print(f"  {controller:13s}: {delta.mean():+6.2f}s +/- {delta.std():5.2f}s  "
              f"(regular {wide['regular'].mean():.2f}s -> irregular {wide['irregular'].mean():.2f}s)")

    print("\n--- who wins on irregular spacing? (lower delay/trip is better) ---")
    irr = df[df["net"] == "irregular"].groupby("controller")["delay_per_trip"].mean()
    for controller, delay in irr.sort_values().items():
        print(f"  {controller:13s}: {delay:6.2f}s")


def main():
    import argparse
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    df = run_all(args.seeds, args.force)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(df)} rows)")
    report(df)


if __name__ == "__main__":
    main()
