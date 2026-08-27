"""Paired Wilcoxon signed-rank tests and bootstrap CIs on this project's headline gaps.

Run once, non-training: reads existing per-seed CSVs under analysis/ (already
committed, produced by the SP5-SP15b sweeps) and reports, for each headline
claim, the paired mean difference, a 95% bootstrap CI on that mean, and a
paired Wilcoxon signed-rank test where n is large enough for the test to
carry any resolution (n=3 two-sided Wilcoxon cannot reach p<0.05 regardless
of effect size -- min attainable two-sided p at n=3 is 0.25 -- so n=3 rows
report the bootstrap CI only and say so explicitly, rather than a
meaningless p-value).

Usage: python -m analysis.headline_stats
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RNG = np.random.default_rng(0)
N_BOOT = 20000


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
    lo, hi = bootstrap_ci(diffs)
    line = f"{label}: n={n}, mean diff={mean:+.3f}, sd={sd:.3f}, 95% bootstrap CI=[{lo:+.3f}, {hi:+.3f}]"
    if n >= 6:
        try:
            stat, p = wilcoxon(diffs)
            line += f", Wilcoxon signed-rank p={p:.4g}"
        except ValueError as e:
            line += f", Wilcoxon: {e}"
    else:
        line += f", Wilcoxon not meaningful at n={n} (min two-sided p={2 * (0.5 ** n):.3g}; use the CI)"
    if note:
        line += f"  [{note}]"
    print(line)
    return dict(label=label, n=n, mean=mean, sd=sd, ci_lo=lo, ci_hi=hi)


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
