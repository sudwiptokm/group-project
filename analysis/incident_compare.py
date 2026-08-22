"""SP7: mid-episode incident/blockage -- each controller's delay under a
15-minute lane closure on C1_C2, compared to its own no-incident corridor_peak
number. See docs/superpowers/specs/2026-08-22-sp7-corridor-incident-design.md.

    python -m analysis.incident_compare
"""
import argparse
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
OUT_CSV = os.path.join(REPO, "analysis", "incident_compare.csv")

SCENARIO = "corridor_peak"
SEEDS = (42, 43, 44)
MIN_GREEN = 10
LAM = 0.5
STEPS = 100_000

os.environ.setdefault("TIME_TO_TELEPORT", "300")

# SP5's in-distribution idqn/corridor_peak/no-incident mean delay -- idqn has
# no row in analysis/corridor_sweep.csv (that file only holds the non-RL
# baselines), so its no-incident reference is this disclosed constant instead
# of a CSV lookup (docs/FINDINGS_2026-08-21-sp5-idqn-vs-corrected-bar.md).
INDIST_IDQN_DELAY = 16.56


def run_baseline(controller: str, seed: int, force: bool = False) -> dict:
    """One incident-eval episode for a non-RL baseline. Resumable."""
    stem = f"logs/eval_{controller}_{SCENARIO}_seed{seed}_mg{MIN_GREEN}_incident"
    trip = tripinfo_path(stem)
    if force or not os.path.exists(trip):
        cb.run(SCENARIO, controller, seed, min_green=MIN_GREEN, tripinfo=True,
              incident=True)
    row = reduce_tripinfo(trip)
    return {"controller": controller, "scenario": SCENARIO, "seed": seed,
            "min_green": MIN_GREEN, "delay_per_trip": row["trip_time_loss_mean"],
            "trips": row["trips_completed"]}


def run_idqn(seed: int, force: bool = False) -> dict:
    """One zero-shot incident-eval episode for IDQN, reusing the corridor_peak
    checkpoint. Resumable."""
    stem = tcd._eval_out_stem(SCENARIO, SCENARIO, LAM, seed, MIN_GREEN, STEPS,
                              incident=True)
    trip = tripinfo_path(stem)
    if force or not os.path.exists(trip):
        tcd.evaluate(SCENARIO, LAM, seed, MIN_GREEN, STEPS, tripinfo=True,
                     incident=True)
    row = reduce_tripinfo(trip)
    return {"controller": "idqn", "scenario": SCENARIO, "seed": seed,
            "min_green": MIN_GREEN, "delay_per_trip": row["trip_time_loss_mean"],
            "trips": row["trips_completed"]}


def incident_sweep(seeds=SEEDS, force: bool = False) -> pd.DataFrame:
    rows = []
    for controller in ("green_wave", "max_pressure"):
        for seed in seeds:
            rows.append(run_baseline(controller, seed, force))
            r = rows[-1]
            print(f"[{len(rows)}/9] {controller:13s} seed{seed} incident "
                  f"delay/trip={r['delay_per_trip']:7.1f}s trips={r['trips']:5d}",
                  flush=True)
    for seed in seeds:
        rows.append(run_idqn(seed, force))
        r = rows[-1]
        print(f"[{len(rows)}/9] idqn          seed{seed} incident "
              f"delay/trip={r['delay_per_trip']:7.1f}s trips={r['trips']:5d}",
              flush=True)
    return pd.DataFrame(rows)


def no_incident_for(controller: str, seed: int) -> float:
    """This controller's own no-incident corridor_peak delay for one seed --
    the number this seed's incident delay is measured against (seed-matched
    pairing, the discipline analysis/idqn_sweep.py:paired_vs and
    analysis/idqn_zeroshot.py:paired_gap already use elsewhere in this
    codebase). Not a cross-seed mean: max_pressure's no-incident distribution
    is confirmed bimodal across seeds (seeds 42/44 ~28-29s, seed 43 ~21.9s
    per corridor_sweep.csv), so a mean would misrepresent any individual
    seed's baseline. idqn isn't in corridor_sweep.csv (that file only holds
    the non-RL baselines), so it uses the disclosed constant
    INDIST_IDQN_DELAY for every seed -- the one case where a scalar shared
    across seeds is correct and intentional."""
    if controller == "idqn":
        return INDIST_IDQN_DELAY
    df = pd.read_csv(CORRIDOR_SWEEP_CSV)
    rows = df[(df["controller"] == controller) & (df["scenario"] == SCENARIO) &
              (df["min_green"] == MIN_GREEN) & (df["seed"] == seed)]
    if rows.empty:
        raise ValueError(
            f"no no-incident row for {controller}/{SCENARIO}/mg{MIN_GREEN}/seed{seed}")
    return float(rows["delay_per_trip"].iloc[0])


def incident_cost(incident_df: pd.DataFrame, controller: str, no_incident: dict) -> dict:
    """Mean/sd incident delay minus each seed's own no-incident number --
    the Δ the SP7 decision rule compares across controllers, not raw delay
    (idqn already starts from a higher no-incident baseline than green_wave,
    per SP5, so raw delay alone would misrank this). no_incident maps
    seed -> that seed's own no-incident delay (build with no_incident_for);
    subtraction is seed-matched, not a controller-wide mean applied to every
    seed."""
    rows = incident_df[incident_df["controller"] == controller]
    baseline = rows["seed"].map(no_incident)
    if baseline.isna().any():
        missing = sorted(set(rows.loc[baseline.isna(), "seed"]))
        raise ValueError(f"no no-incident baseline for {controller} seed(s) {missing}")
    delta = rows["delay_per_trip"] - baseline
    return {
        "controller": controller,
        "incident_mean": float(rows["delay_per_trip"].mean()),
        "no_incident_mean": float(baseline.mean()),
        "cost_mean": float(delta.mean()),
        "cost_sd": float(delta.std(ddof=1)) if len(delta) > 1 else float("nan"),
        "n": int(len(delta)),
    }


def report(incident_df: pd.DataFrame) -> None:
    print(f"\n################ {SCENARIO}, incident "
          f"(C1_C2 lane closed 1800-2700s) ################")
    for controller in ("green_wave", "max_pressure", "idqn"):
        seeds = sorted(incident_df.loc[incident_df["controller"] == controller, "seed"].unique())
        no_incident = {int(s): no_incident_for(controller, int(s)) for s in seeds}
        cost = incident_cost(incident_df, controller, no_incident)
        print(f"  {controller:13s} no-incident={cost['no_incident_mean']:6.2f}s  "
              f"incident={cost['incident_mean']:6.2f}s  "
              f"cost(delta)={cost['cost_mean']:+6.2f} +/- {cost['cost_sd']:.2f}s  "
              f"n={cost['n']}")


def main():
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    df = incident_sweep(args.seeds, args.force)
    if os.path.exists(OUT_CSV):
        prior = pd.read_csv(OUT_CSV)
        df = pd.concat([prior, df]).drop_duplicates(subset=["controller", "seed"], keep="last")
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(df)} rows)")
    report(df)


if __name__ == "__main__":
    main()
