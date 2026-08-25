"""Does SP8's idqn-beats-green_wave flip generalize beyond its one 578m/78m
asymmetric-spacing sample? (SP10, follow-up to SP8's
analysis/irregular_net_compare.py and docs/FINDINGS_2026-08-22-sp8-irregular-
spacing.md.)

SP8 built corridor_irregular.net.xml (C1@0, C2@600, C3@700 nominal -> 578m/78m
realised edge length after junction-shape shrinkage) and found idqn beats
green_wave zero-shot there, reversing the regular-net (200m/200m) ranking.
SP8's own caveat flagged this could be an artifact of that one specific
asymmetry rather than asymmetric spacing generally. This script tests two
more variants, same topology/demand/controllers, only the spacing changes:

  corridor_irregular2.net.xml -- REVERSE skew of SP8's (78m C1-C2, 578m C2-C3
  realised; SP8 was 578m/78m). Tests whether the effect is direction-
  dependent -- e.g. an artifact of the long block landing on C2-C3
  specifically, rather than of asymmetry per se.

  corridor_irregular3.net.xml -- MODERATE asymmetry, same skew direction as
  SP8 (long block first) but 278m/78m realised instead of 578m/78m. Tests
  whether idqn's win margin scales with how extreme the asymmetry is, or is
  closer to a step function (any asymmetry flips it) or a threshold (only
  large asymmetry flips it).

See each variant's own .nod.xml comment for the exact reasoning.

No training happens here -- IDQN's existing corridor_peak checkpoints (SP5,
same ones SP8 reused) are evaluated zero-shot on both new geometries, same
posture as SP8/SP6. green_wave/max_pressure get seeds 42-46 (5 seeds, no
checkpoint needed, cheap); idqn stays at SP8's n=3 (42-44) -- no more
corridor_peak checkpoints exist without new training, same constraint SP8's
own incident-seed-widening follow-up hit.

Deliberately bypasses compare.py, for the exact reason
analysis/irregular_net_compare.py's docstring gives: non-default net_file is
a tag dimension compare.py's glob (`eval_{entity}_{scenario}_seed*.csv`)
doesn't know, and teaching it a fifth/sixth tag risks the same glob-confound
bug class SP6/SP7/SP8 all hit and had to fix. Reading each run's tripinfo XML
directly by its exact path sidesteps that class of bug entirely.

    python -m analysis.irregular_net_compare2
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
SP8_IRREGULAR_CSV = os.path.join(REPO, "analysis", "irregular_net_compare.csv")
OUT_CSV = os.path.join(REPO, "analysis", "irregular_net_compare2.csv")

SCENARIO = "corridor_peak"
LAM = 0.5
STEPS = 100_000
MIN_GREEN = 10

# label -> net file. "irregular" (SP8's 578m/78m) is included via
# SP8_IRREGULAR_CSV, not re-run here.
NETS = {
    "irregular2": "corridor_irregular2.net.xml",   # reverse skew: 78m/578m
    "irregular3": "corridor_irregular3.net.xml",   # moderate, same direction: 278m/78m
}
BASELINE_SEEDS = (42, 43, 44, 45, 46)   # green_wave/max_pressure -- no checkpoint needed
IDQN_SEEDS = (42, 43, 44)               # only seeds with a corridor_peak checkpoint on disk

os.environ.setdefault("TIME_TO_TELEPORT", "300")


def _regular_reference(controller: str, seeds) -> pd.DataFrame:
    """green_wave/max_pressure's already-computed regular-net corridor_peak
    numbers (SP4), filtered to this eval's seeds."""
    df = pd.read_csv(CORRIDOR_SWEEP_CSV)
    df = df[(df["controller"] == controller) & (df["scenario"] == SCENARIO) &
            (df["min_green"] == MIN_GREEN) & (df["seed"].isin(seeds))]
    return df[["seed", "delay_per_trip", "trips"]].sort_values("seed")


def _sp8_irregular_reference() -> pd.DataFrame:
    """SP8's own corridor_irregular (578m/78m) numbers, read back rather than
    re-run -- gives this script's report a 4-way (regular, SP8's irregular,
    irregular2, irregular3) table for free."""
    return pd.read_csv(SP8_IRREGULAR_CSV)


def _run_baseline(controller: str, net_label: str, net_file: str, seed: int,
                  force: bool) -> dict:
    stem = (f"logs/eval_{controller}_{SCENARIO}_seed{seed}_mg{MIN_GREEN}"
            f"_net{net_file.removesuffix('.net.xml').removeprefix('corridor_')}")
    trip = tripinfo_path(stem)
    if force or not os.path.exists(trip):
        cb.run(SCENARIO, controller, seed, min_green=MIN_GREEN, tripinfo=True,
              net_file=net_file)
    row = reduce_tripinfo(trip)
    return {"controller": controller, "net": net_label, "seed": seed,
            "delay_per_trip": row["trip_time_loss_mean"], "trips": row["trips_completed"]}


def _run_idqn(net_label: str, net_file: str, seed: int, force: bool) -> dict:
    """Zero-shot: corridor_peak-trained IDQN checkpoint (SP5) on the new geometry."""
    stem = tcd._eval_out_stem(SCENARIO, SCENARIO, LAM, seed, MIN_GREEN, STEPS,
                              net_file=net_file)
    trip = tripinfo_path(stem)
    if force or not os.path.exists(trip):
        tcd.evaluate(SCENARIO, LAM, seed, MIN_GREEN, STEPS, tripinfo=True,
                    eval_scenario=SCENARIO, net_file=net_file)
    row = reduce_tripinfo(trip)
    return {"controller": "idqn", "net": net_label, "seed": seed,
            "delay_per_trip": row["trip_time_loss_mean"], "trips": row["trips_completed"]}


def run_all(force: bool = False) -> pd.DataFrame:
    rows = []
    for controller in ("green_wave", "max_pressure"):
        ref = _regular_reference(controller, BASELINE_SEEDS)
        for _, r in ref.iterrows():
            rows.append({"controller": controller, "net": "regular", "seed": int(r["seed"]),
                        "delay_per_trip": r["delay_per_trip"], "trips": int(r["trips"])})
        for net_label, net_file in NETS.items():
            for seed in BASELINE_SEEDS:
                rows.append(_run_baseline(controller, net_label, net_file, seed, force))
                r = rows[-1]
                print(f"[{controller}/{net_label}] seed{seed} "
                      f"delay/trip={r['delay_per_trip']:7.2f}s trips={r['trips']:5d}",
                      flush=True)
    for seed in IDQN_SEEDS:
        # regular-net idqn reference: reuse SP5/SP8's tripinfo if present
        stem = tcd._eval_out_stem(SCENARIO, SCENARIO, LAM, seed, MIN_GREEN, STEPS)
        trip = tripinfo_path(stem)
        if force or not os.path.exists(trip):
            tcd.evaluate(SCENARIO, LAM, seed, MIN_GREEN, STEPS, tripinfo=True,
                        eval_scenario=SCENARIO)
        row = reduce_tripinfo(trip)
        rows.append({"controller": "idqn", "net": "regular", "seed": seed,
                    "delay_per_trip": row["trip_time_loss_mean"], "trips": row["trips_completed"]})
        for net_label, net_file in NETS.items():
            rows.append(_run_idqn(net_label, net_file, seed, force))
            r = rows[-1]
            print(f"[idqn/{net_label}] seed{seed} delay/trip={r['delay_per_trip']:7.2f}s "
                  f"trips={r['trips']:5d}", flush=True)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    sp8 = _sp8_irregular_reference()
    sp8_irr = sp8[sp8["net"] == "irregular"].copy()
    sp8_irr["net"] = "irregular_sp8"
    full = pd.concat([df, sp8_irr], ignore_index=True)

    print(f"\n=== {SCENARIO}, mg{MIN_GREEN}: regular vs three asymmetric-spacing variants ===")
    summary = full.groupby(["controller", "net"])["delay_per_trip"].agg(["mean", "std", "count"])
    print(summary.to_string(float_format=lambda x: f"{x:6.2f}"))

    print("\n--- per-controller shift (variant - regular), paired by seed ---")
    for net_label in ("irregular_sp8", "irregular2", "irregular3"):
        print(f"\n  [{net_label}]")
        for controller, g in full[full["net"].isin(["regular", net_label])].groupby("controller"):
            wide = g.pivot(index="seed", columns="net", values="delay_per_trip")
            if "regular" not in wide or net_label not in wide:
                continue
            wide = wide.dropna()
            delta = wide[net_label] - wide["regular"]
            print(f"    {controller:13s}: {delta.mean():+6.2f}s +/- {delta.std():5.2f}s  "
                  f"(regular {wide['regular'].mean():.2f}s -> {net_label} "
                  f"{wide[net_label].mean():.2f}s, n={len(wide)})")

    print("\n--- who wins on each variant? (lower delay/trip is better) ---")
    for net_label in ("regular", "irregular_sp8", "irregular2", "irregular3"):
        irr = full[full["net"] == net_label].groupby("controller")["delay_per_trip"].mean()
        print(f"  [{net_label}]")
        for controller, delay in irr.sort_values().items():
            print(f"    {controller:13s}: {delay:6.2f}s")


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
