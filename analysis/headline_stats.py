"""Paired statistics on this project's headline gaps.

Run once, non-training: reads existing per-seed CSVs under analysis/ (already
committed, produced by the SP5-SP15b sweeps) and reports, for each headline
claim, the paired per-seed differences, their mean and standard deviation,
and -- only where n is large enough for either to carry information -- a 95%
bootstrap CI and a paired Wilcoxon signed-rank test.

Two n-dependent rules, both applied here rather than left to the reader:

1. A two-sided Wilcoxon signed-rank test cannot reach p<0.05 at n<=5
   regardless of effect size (its minimum attainable two-sided p is 2*0.5**n
   -- 0.25 at n=3, 0.0625 at n=5), so no p-value is reported below n=6.
2. A bootstrap resamples the observed differences, so at n=3 the resample
   mean can take only a handful of discrete values and the resulting
   interval systematically understates uncertainty -- it describes the
   spread of three numbers, not the sampling distribution of the effect.
   No CI is reported below n=6 either. Small-n rows report every per-seed
   difference instead, so a reader can see the whole sample directly.

Wilcoxon rows also report the minimum two-sided p attainable at their n
(MIN_P), because at n=10 a reported p=0.0020 IS that floor: it means all ten
signed differences pointed the same way and nothing more precise than that.

Usage: python3 -m analysis.headline_stats
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RNG = np.random.default_rng(0)
N_BOOT = 20000

# Below this n, neither a bootstrap CI nor a Wilcoxon p carries information;
# see the module docstring. Rows below it report per-seed differences instead.
MIN_N_FOR_INTERVAL = 6


def bootstrap_ci(diffs, n_boot=N_BOOT, alpha=0.05):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = RNG.choice(diffs, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def report(label, a, b, seeds_a, seeds_b, note=""):
    """a, b: arrays of the SAME metric for two conditions, paired by seed."""
    common = sorted(set(seeds_a) & set(seeds_b))
    a_map = dict(zip(seeds_a, a))
    b_map = dict(zip(seeds_b, b))
    diffs = np.array([a_map[s] - b_map[s] for s in common])
    n = len(diffs)
    mean = diffs.mean()
    sd = diffs.std(ddof=1) if n > 1 else float("nan")
    per_seed = " ".join(f"{s}:{d:+.3f}" for s, d in zip(common, diffs))
    line = f"{label}: n={n}, mean diff={mean:+.3f}, sd={sd:.3f}"
    within_a = np.std([a_map[s] for s in common], ddof=1) if n > 1 else float("nan")
    within_b = np.std([b_map[s] for s in common], ddof=1) if n > 1 else float("nan")
    row = dict(label=label, n=n, mean=mean, sd=sd, per_seed=per_seed,
               sd_condition_a=within_a, sd_condition_b=within_b,
               ci_lo=float("nan"), ci_hi=float("nan"), p=float("nan"),
               min_p=2 * (0.5 ** n))
    if n >= MIN_N_FOR_INTERVAL:
        lo, hi = bootstrap_ci(diffs)
        row["ci_lo"], row["ci_hi"] = lo, hi
        line += f", 95% bootstrap CI=[{lo:+.3f}, {hi:+.3f}]"
        try:
            stat, p = wilcoxon(diffs)
            row["p"] = p
            line += (f", Wilcoxon signed-rank p={p:.4g}"
                     f" (min attainable at n={n}: {row['min_p']:.4g})")
        except ValueError as e:
            line += f", Wilcoxon: {e}"
    else:
        # No interval and no p-value at this n -- show the whole sample instead.
        line += (f", per-seed diffs [{per_seed}]"
                 f", range [{diffs.min():+.3f}, {diffs.max():+.3f}]"
                 f"; no CI or p reported at n={n} (see module docstring)")
    if note:
        line += f"  [{note}]"
    print(line)
    return row


def main():
    rows = []

    print("=== SP9: idqn vs green_wave, irregular net, n=10 ===")
    df = pd.read_csv("analysis/irregular_net_compare.csv")
    df = df[df.net == "irregular"]
    gw = df[df.controller == "green_wave"].sort_values("seed")
    iq = df[df.controller == "idqn"].sort_values("seed")
    rows.append(report(
        "SP9 irregular: green_wave - idqn (positive = idqn faster)",
        gw.delay_per_trip.values, iq.delay_per_trip.values,
        gw.seed.values, iq.seed.values,
    ))

    print("\n=== SP14b: lambda 0.5 vs 0.25, n=10, paired by seed ===")
    df = pd.read_csv("analysis/lambda_ablation_n10.csv")
    for geom in ["regular", "irregular"]:
        sub = df[df.geometry == geom]
        a = sub[sub.lam == 0.50].sort_values("seed")
        b = sub[sub.lam == 0.25].sort_values("seed")
        rows.append(report(
            f"SP14b {geom}: lam0.50 - lam0.25 (positive = 0.25 faster)",
            a.delay_per_trip.values, b.delay_per_trip.values,
            a.seed.values, b.seed.values,
        ))

    print("\n=== SP14: lambda 0.5 vs 0.25, n=3 (original), paired by seed ===")
    df = pd.read_csv("analysis/lambda_ablation.csv")
    for geom in ["regular", "irregular"]:
        sub = df[df.geometry == geom]
        a = sub[sub.lam == 0.50].sort_values("seed")
        b = sub[sub.lam == 0.25].sort_values("seed")
        rows.append(report(
            f"SP14 {geom} (n=3): lam0.50 - lam0.25",
            a.delay_per_trip.values, b.delay_per_trip.values,
            a.seed.values, b.seed.values,
        ))

    print("\n=== SP13: idqn vs green_wave at representative ratios, n=3 (seeds 42-44 common) ===")
    df = pd.read_csv("analysis/geometry_sweep.csv")
    for r in [0.50, 0.60, 0.75, 0.90]:
        sub = df[np.isclose(df.ratio, r)]
        gw = sub[sub.controller == "green_wave"].sort_values("seed")
        iq = sub[sub.controller == "idqn"].sort_values("seed")
        note = "inside band [0.51,0.80]" if 0.51 <= r <= 0.80 else "outside band"
        rows.append(report(
            f"SP13 r={r}: green_wave - idqn",
            gw.delay_per_trip.values, iq.delay_per_trip.values,
            gw.seed.values, iq.seed.values, note=note,
        ))

    print("\n=== SP13e: idqn vs green_wave at representative low ratios, n=3 (seeds 42-44 common) ===")
    df = pd.read_csv("analysis/geometry_sweep_lowr.csv")
    for r in [0.10, 0.20, 0.45]:
        sub = df[np.isclose(df.ratio, r)]
        gw = sub[sub.controller == "green_wave"].sort_values("seed")
        iq = sub[sub.controller == "idqn"].sort_values("seed")
        note = "inside band [0.103,0.341]" if 0.103 <= r <= 0.341 else "outside band"
        rows.append(report(
            f"SP13e r={r}: green_wave - idqn",
            gw.delay_per_trip.values, iq.delay_per_trip.values,
            gw.seed.values, iq.seed.values, note=note,
        ))

    print("\n=== Single intersection: static-60s vs actuated-min_green-60, n=5 ===")
    st = pd.read_csv("analysis/static_sweep.csv")
    st = st[st.green == 60].sort_values("seed")
    ac = pd.read_csv("analysis/actuated_sweep.csv")
    ac = ac[ac.min_green == 60].sort_values("seed")
    rows.append(report(
        "static60 - actuated_mg60 (positive = actuated faster)",
        st.delay.values, ac.delay.values,
        st.seed.values, ac.seed.values,
    ))

    out = pd.DataFrame(rows)
    out.to_csv("analysis/headline_stats.csv", index=False)
    print("\nWrote analysis/headline_stats.csv")


if __name__ == "__main__":
    main()
