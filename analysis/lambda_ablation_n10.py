"""SP14 follow-up: does the lambda=0.25-vs-0.5 delay gap survive n=3 -> n=10?

SP14 (docs/FINDINGS_2026-08-26-sp14-lambda-ablation.md) found lambda=0.25 beats
the project default lambda=0.5 on delay, on both geometries, at n=3 seeds --
but flagged the irregular-net gap (18.23s vs 18.48s, 0.25s) as small relative
to this project's typical seed-to-seed noise (SP9 found sigma~0.38s at n=10
for a comparably-sized effect), and asked for the same n=3->n=10 widening SP9
did for the geometry flip. This trains the 7 missing seeds (45-51) for ONLY
lambda in {0.25, 0.5} -- not all 5 lambdas, since the other 3 arms (0.0, 0.75,
1.0) aren't the comparison in question and widening them would cost 3x more
training for no bearing on this specific gap -- and reuses
analysis.lambda_ablation's own run_one/NETS/SCENARIO so the eval protocol is
byte-for-byte identical to the n=3 result it's extending.

    LAMBDAS="0.25 0.5" SEEDS="45 46 47 48 49 50 51" JOBS=6 ./run_lambda_ablation.sh
    python -m analysis.lambda_ablation_n10
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd

from analysis.lambda_ablation import NETS, SCENARIO, run_one

OUT_CSV = os.path.join(REPO, "analysis", "lambda_ablation_n10.csv")

LAMBDAS = (0.25, 0.5)
SEEDS = tuple(range(42, 52))  # 42-44 from SP14 (n=3), 45-51 new


def run_all(force: bool = False) -> pd.DataFrame:
    rows = []
    for geometry, net_file in NETS.items():
        for lam in LAMBDAS:
            for seed in SEEDS:
                r = run_one(lam, seed, geometry, net_file, force)
                rows.append(r)
                print(f"[{geometry}/lam{lam:<4}] seed{seed} "
                      f"delay/trip={r['delay_per_trip']:7.2f}s "
                      f"safety_total={r['safety_total']:9.1f} "
                      f"({'cached' if r['cached'] else 'ran'})", flush=True)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    for geometry in NETS:
        g = df[df["geometry"] == geometry]
        print(f"\n=== {SCENARIO}, {geometry}: lambda=0.25 vs 0.5, n=3 vs n=10 ===")
        for n, seeds in (("n=3", (42, 43, 44)), ("n=10", SEEDS)):
            sub = g[g["seed"].isin(seeds)]
            summary = sub.groupby("lam")["delay_per_trip"].agg(["mean", "std", "count"])
            print(f"  {n}:")
            print(summary.to_string(float_format=lambda x: f"{x:7.3f}",
                                    header=False).replace("\n", "\n    "))
        wide10 = g.pivot_table(index="seed", columns="lam", values="delay_per_trip")
        wide10 = wide10.reindex(SEEDS)
        d10 = (wide10[0.5] - wide10[0.25]).dropna()
        d3 = d10.loc[[42, 43, 44]]
        print(f"  0.5-minus-0.25 delay, paired by seed: n=3 {d3.mean():+.3f} +/- "
              f"{d3.std():.3f}   n=10 {d10.mean():+.3f} +/- {d10.std():.3f}   "
              f"(positive = 0.25 still faster)")
        n_agree = (d10 > 0).sum()
        print(f"  sign agrees with n=3 direction on {n_agree}/10 seeds")


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
