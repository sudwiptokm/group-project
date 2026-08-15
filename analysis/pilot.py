"""Score the pilot retrain against the two controllers that matter.

The peak null this project reported was produced entirely at min_green=10, a
floor analysis/actuated.py shows is unwinnable for ANY controller. This scores
the retrain at the corrected floor against both references, paired per demand
seed:

    static 60 s plan        91.8 +/- 19.9 s  -- the competent fixed plan
    queue-actuated, mg 60   82.5 +/- 10.1 s  -- the bar that actually matters

Beating the static plan is not enough. The actuated controller already does
that with no reward, no training and no sample budget, so an agent that merely
matches it has demonstrated nothing about learning.

Two groupings, because Stage-1 defect 4 was reporting one of them as the other:

  * per TRAINING seed -- is the result one lucky policy, or all of them?
  * per DEMAND seed, paired -- the comparison against the references, on
    matched traffic, which is the only form that survived the audit.

    python analysis/pilot.py --algo dqn
"""
import argparse
import glob
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd

from analysis.tripinfo import count_departures, reduce_tripinfo
from env_common import SCENARIO_ROUTES, tripinfo_path

EVAL_RUN = re.compile(r"seed(\d+)_t(\d+)_mg(\d+)")


def _delay(path: str, departed: int) -> dict:
    return reduce_tripinfo(tripinfo_path(path[: path.index("_conn")]),
                           departed=departed)


def _reference(pattern: str, departed: int, label: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(pattern)):
        seed = int(re.search(r"seed(\d+)", os.path.basename(path)).group(1))
        trips = _delay(path, departed)
        if trips:
            rows.append({"demand_seed": seed, label: trips["trip_time_loss_mean"]})
    return pd.DataFrame(rows).set_index("demand_seed")


def main(algo: str, min_green: int, lam_tag: str, horizon: float) -> None:
    departed = count_departures(SCENARIO_ROUTES["peak"], horizon=horizon)

    rows = []
    pattern = (f"logs/eval_{algo}_peak_lam{lam_tag}_seed*_mg{min_green}"
               "_conn*_ep*.csv")
    for path in sorted(glob.glob(pattern)):
        m = EVAL_RUN.search(os.path.basename(path))
        if not m:
            continue  # no training-seed tag: cannot attribute it to a policy
        trips = _delay(path, departed)
        if not trips:
            continue
        rows.append({
            "demand_seed": int(m.group(1)),
            "train_seed": int(m.group(2)),
            "delay": trips["trip_time_loss_mean"],
            "trips": trips["trips_completed"],
        })
    if not rows:
        raise SystemExit(f"no evaluated runs matching {pattern}")
    df = pd.DataFrame(rows)

    static = _reference(f"analysis/static_logs/g{min_green}_seed*_conn*_ep*.csv",
                        departed, "static")
    actuated = _reference(
        f"analysis/actuated_logs/peak_mg{min_green}_seed*_conn*_ep*.csv",
        departed, "actuated")

    print(f"{algo} at min_green={min_green}: "
          f"{df.delay.mean():.1f} +/- {df.delay.std():.1f} s over "
          f"{len(df)} runs ({df.train_seed.nunique()} policies x "
          f"{df.demand_seed.nunique()} demand seeds), "
          f"{df.trips.mean():.0f} trips\n")

    # Defect 4 was reporting one policy's demand spread as seed variance. Show
    # the policies separately so nobody has to take that on trust again.
    print("per training seed (mean over demand seeds) — one lucky policy, or all?")
    per_policy = df.groupby("train_seed").delay.agg(["mean", "std", "count"])
    print(per_policy.round(2).to_string() + "\n")

    g = df.groupby("demand_seed").delay.mean().to_frame(algo)
    for ref in (static, actuated):
        g = g.join(ref)
    print("per demand seed, paired against each reference:")
    for ref in ("static", "actuated"):
        if ref in g:
            g[f"vs_{ref}"] = g[algo] - g[ref]
    print(g.round(1).to_string() + "\n")

    for ref in ("static", "actuated"):
        col = f"vs_{ref}"
        if col not in g:
            continue
        diff = g[col].dropna()
        verdict = "BEATS" if diff.mean() < 0 else "loses to"
        print(f"vs {ref:9s}: {diff.mean():+.1f} s (sd {diff.std():.1f}), "
              f"wins {(diff < 0).sum()}/{len(diff)} seeds — {verdict} it"
              + ("  <-- the bar that matters" if ref == "actuated" else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="dqn")
    p.add_argument("--min-green", type=int, default=60)
    p.add_argument("--lam-tag", default="05", help="lambda tag in the filename")
    p.add_argument("--episode-seconds", type=float, default=3600.0)
    a = p.parse_args()
    main(a.algo, a.min_green, a.lam_tag, a.episode_seconds)
