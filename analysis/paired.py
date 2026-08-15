"""Paired per-seed comparison: each RL run against the fixed-time run on the
same SUMO seed (same demand realisation), instead of against seed 0.
"""
import glob
import re

import numpy as np
import pandas as pd

SEEDS = [42, 43, 44, 45, 46]
ALGOS = ["dqn", "a2c", "ppo", "qrdqn"]


def onset(df, thr=0.05):
    s, t = df.system_mean_speed.values, df.step.values
    for i in range(len(s)):
        if s[i] < thr and (s[i:] < thr).all():
            return t[i]
    return None


def load(pat):
    f = glob.glob(pat)
    return pd.read_csv(f[0]) if f else None


for sc in ["peak", "offpeak"]:
    base = {}
    for s in SEEDS + [0]:
        d = load(f"logs/eval_fixedtime_{sc}_seed{s}_conn*_ep*.csv")
        if d is not None:
            base[s] = d.system_mean_waiting_time.mean()
    print(f"\n===== {sc} =====")
    print("fixed-time by seed:", {k: round(v, 1) for k, v in sorted(base.items())})
    have = [s for s in SEEDS if s in base]
    if have:
        b = np.array([base[s] for s in have])
        print(f"fixed-time over seeds {have}: mean {b.mean():.1f} ± {b.std(ddof=1):.1f}"
              f"  (reported single-seed value was {base.get(0, float('nan')):.1f})")

    for algo in ALGOS:
        per_seed, paired = [], []
        for s in have:
            d = load(f"logs/eval_{algo}_{sc}_lam05_seed{s}_conn*_ep*.csv")
            if d is None:
                continue
            w = d.system_mean_waiting_time.mean()
            per_seed.append(w)
            paired.append(100.0 * (w - base[s]) / base[s])
        if not per_seed:
            continue
        a = np.array(per_seed)
        p = np.array(paired)
        old = 100.0 * (a.mean() - base[0]) / base[0] if 0 in base else float("nan")
        print(f"  {algo:6s} wait {a.mean():8.1f} ± {a.std(ddof=1):6.1f} | "
              f"paired vs same-seed fixed: {p.mean():+6.1f}% ± {p.std(ddof=1):5.1f} | "
              f"old (vs seed-0 fixed): {old:+6.1f}%")
