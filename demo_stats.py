"""Paired multi-seed comparison for the demo: fixed-time vs actuated vs DQN.

    python demo_stats.py [first_seed] [last_seed]

Metric: delay per COMPLETED trip (SUMO timeLoss), the report's headline metric, plus trips completed.
Same demand seed for every controller (paired). Wilcoxon signed-rank on the per-seed differences.
"""
import sys

import pandas as pd
import itertools

import numpy as np

from analysis.tripinfo import reduce_tripinfo


def wilcoxon_p(d):
    """Exact two-sided Wilcoxon signed-rank p (no scipy dependency; fine for n <= ~20)."""
    d = np.asarray([x for x in d if x != 0])
    n = len(d)
    if n == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    for v in np.unique(np.abs(d)):            # average ranks over ties
        m = np.abs(d) == v
        ranks[m] = ranks[m].mean()
    w = ranks[d > 0].sum()
    tot = ranks.sum()
    stats = [sum(r for r, keep in zip(ranks, signs) if keep)
             for signs in itertools.product([0, 1], repeat=n)]
    dev = abs(w - tot / 2)
    return float(np.mean([abs(x - tot / 2) >= dev - 1e-9 for x in stats]))

args = [x for x in sys.argv[1:] if not x.startswith("--")]
a = int(args[0]) if len(args) > 0 else 42
b = int(args[1]) if len(args) > 1 else 51
# --exclude 51,49 : sensitivity run without those seeds (always report it next to the full run)
excl = {int(x) for f in sys.argv[1:] if f.startswith("--exclude=") for x in f.split("=")[1].split(",")}
PATHS = {
    "fixed": "logs/eval_fixedtime_peak_seed{s}_g60_tripinfo.xml",
    "actuated": "analysis/actuated_logs/peak_mg60_seed{s}_tripinfo.xml",
    "dqn": "logs/eval_dqn_peak_lam00_seed{s}_mg60_tripinfo.xml",
}
rows = []
for s in range(a, b + 1):
    if s in excl:
        continue
    r = {"seed": s}
    for k, p in PATHS.items():
        t = reduce_tripinfo(p.format(s=s))
        r[f"{k}_delay"] = t.get("trip_time_loss_mean", float("nan"))
        r[f"{k}_done"] = t.get("trips_completed", 0)
    rows.append(r)
df = pd.DataFrame(rows).dropna()
if not excl:
    df.to_csv("analysis/demo_stats.csv", index=False)
n = len(df)
print(f"\n=== peak demand, teleport=300, min-green 60 s, n={n} paired seeds ({a}-{b}){' EXCLUDING ' + str(sorted(excl)) if excl else ''} ===")
print(df.round(1).to_string(index=False))
print("\nmean delay per completed trip (s):  " +
      "   ".join(f"{k} {df[f'{k}_delay'].mean():.1f}" for k in PATHS))
print("mean trips completed:               " +
      "   ".join(f"{k} {df[f'{k}_done'].mean():.0f}" for k in PATHS))
print("\npaired difference in delay (first - second; positive = second is faster):")
for x, y in [("fixed", "dqn"), ("actuated", "dqn"), ("fixed", "actuated")]:
    d = df[f"{x}_delay"] - df[f"{y}_delay"]
    p = wilcoxon_p(d)
    print(f"  {x:8s} - {y:8s}: mean {d.mean():+6.2f} s   {int((d > 0).sum())}/{n} seeds favour {y}   Wilcoxon p={p:.3f}")
