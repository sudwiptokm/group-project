"""Compare the adaptive (queue-actuated) controller against the best static plan.

Answers the question the static sweep raises but cannot settle: is there adaptive
headroom at this junction that our RL merely failed to find, or is there none to
find? A non-learning controller needs no reward, no credit assignment and no
sample budget, so it separates the two -- see analysis/actuated.py.

Reads both sweeps' per-run outputs and prints one table, paired per seed so the
comparison cannot be confounded by demand draw (the Stage-1 defect this project
exists to avoid repeating).

    python analysis/headroom.py
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

from analysis.tripinfo import reduce_tripinfo
from env_common import tripinfo_path

STATIC = os.path.join(REPO, "analysis", "static_logs")
ACTUATED = os.path.join(REPO, "analysis", "actuated_logs")
STATIC_RUN = re.compile(r"g(\d+)_seed(\d+)_conn\d+_ep\d+\.csv$")
ACT_RUN = re.compile(r"peak_mg(\d+)_seed(\d+)_conn\d+_ep\d+\.csv$")


def _rows(logs_dir: str, pattern: re.Pattern, key: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(logs_dir, "*_conn*_ep*.csv"))):
        m = pattern.search(os.path.basename(path))
        if not m:
            continue
        trips = reduce_tripinfo(tripinfo_path(path[: path.index("_conn")]))
        if not trips:
            continue
        rows.append({
            key: int(m.group(1)),
            "seed": int(m.group(2)),
            "delay": trips["trip_time_loss_mean"],
            "completed": trips["trips_completed"],
        })
    return pd.DataFrame(rows)


def main(reference_green: int = 60, table: str = "") -> None:
    static = _rows(STATIC, STATIC_RUN, "green")
    act = _rows(ACTUATED, ACT_RUN, "min_green")
    if static.empty or act.empty:
        raise SystemExit("need both sweeps: analysis/static_timing.py and analysis/actuated.py")

    if table:
        # Per-run rows, like analysis/static_sweep.csv: the actuated_logs/ dir is
        # gitignored (~1 MB of tripinfo per run), so this aggregate is what the
        # write-ups cite and what survives in the repo.
        act.sort_values(["min_green", "seed"]).to_csv(table, index=False)

    ref = static[static.green == reference_green].set_index("seed")
    print(f"reference: static {reference_green} s plan — "
          f"{ref.delay.mean():.1f} s delay, {ref.completed.mean():.0f} trips\n")

    out = []
    for mg, grp in act.groupby("min_green"):
        g = grp.set_index("seed")
        common = g.index.intersection(ref.index)
        diff = (g.loc[common, "delay"] - ref.loc[common, "delay"])
        out.append({
            "min_green": mg,
            "n": len(g),
            "delay": g.delay.mean(),
            "delay_sd": g.delay.std(),
            "completed": g.completed.mean(),
            # spread, not just the mean: the mean difference against the static
            # plan sits inside the seed noise, so consistency across demand
            # draws is the part of this comparison that is actually resolvable.
            "trips_min": g.completed.min(),
            "trips_max": g.completed.max(),
            f"vs_static{reference_green}": diff.mean(),
            "diff_sd": diff.std(),
            "seeds_beating_static": int((diff < 0).sum()),
        })
    print(pd.DataFrame(out).round(1).to_string(index=False))

    print(f"\nstatic {reference_green} s, per seed (the row above is measured "
          f"against this):")
    print(ref[["delay", "completed"]].round(1).to_string())
    print(f"delay sd {ref.delay.std():.1f}, trips {ref.completed.min():.0f}"
          f"-{ref.completed.max():.0f} "
          f"(spread {ref.completed.max() - ref.completed.min():.0f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reference-green", type=int, default=60,
                   help="static plan the adaptive controller is measured against")
    p.add_argument("--table", default="analysis/actuated_sweep.csv",
                   help="where to write the per-run actuated sweep table")
    a = p.parse_args()
    main(a.reference_green, a.table)
