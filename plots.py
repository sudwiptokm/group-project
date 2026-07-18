"""Visualise the comparison table produced by compare.py.

Reads logs/comparison.csv (one row per scenario/lambda/algo with <metric>_mean and
<metric>_std columns, plus a fixed-time baseline row per group) and writes figures
to results/:

  bars_<scenario>_lam<LAM>.png   RL algos vs fixed-time, mean waiting time + std
  tradeoff_<scenario>.png        efficiency & safety-proxy vs lambda for the winner
                                 (only when a scenario has >= 2 lambda values)

The eval CSVs log sumo-rl's standard metrics (waiting time, stopped count, speed),
not the raw safety penalty, so the tradeoff plot uses system_total_stopped as a
safety-adjacent proxy alongside the waiting-time efficiency metric.

Run AFTER compare.py has written logs/comparison.csv:

    python compare.py
    python plots.py                 # -> results/*.png
    python plots.py --csv logs/comparison.csv --out results
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless: save files, never open a window
import matplotlib.pyplot as plt
import pandas as pd

EFF = "system_mean_waiting_time"      # efficiency headline (lower = better)
SAFETY_PROXY = "system_total_stopped"  # safety-adjacent proxy (lower = better)
BASELINE = "fixedtime"

# filename lambda tag -> numeric value
LAM_VALUE = {"00": 0.0, "05": 0.5, "10": 1.0}


def _lam_float(tag: str) -> float:
    return LAM_VALUE.get(str(tag), float("nan"))


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"lam": str})
    # zero-pad any lambda tag pandas may have coerced (e.g. 0 -> "0")
    df["lam"] = df["lam"].str.zfill(2)
    return df


def plot_bars(df: pd.DataFrame, out_dir: str) -> list:
    """One bar chart of mean waiting time per (scenario, lambda) group."""
    written = []
    for (scenario, lam), grp in df.groupby(["scenario", "lam"]):
        grp = grp.sort_values(f"{EFF}_mean")
        if grp.empty:
            continue
        labels = grp["algo"].tolist()
        means = grp[f"{EFF}_mean"].tolist()
        errs = grp[f"{EFF}_std"].fillna(0).tolist()
        colors = ["tab:orange" if a == BASELINE else "tab:blue" for a in labels]

        fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(labels)), 4))
        ax.bar(labels, means, yerr=errs, capsize=4, color=colors)
        ax.set_ylabel("mean waiting time (s)  — lower is better")
        ax.set_title(f"{scenario} · λ={_lam_float(lam)} — RL vs fixed-time (orange)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"bars_{scenario}_lam{lam}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def plot_tradeoff(df: pd.DataFrame, out_dir: str) -> list:
    """Efficiency + safety-proxy vs lambda, per scenario (needs >= 2 lambdas)."""
    written = []
    for scenario, sdf in df.groupby("scenario"):
        rl = sdf[sdf["algo"] != BASELINE]
        lams_present = sorted(rl["lam"].unique(), key=_lam_float)
        if len(lams_present) < 2:
            continue  # a curve needs at least two lambda points

        fig, (ax_eff, ax_saf) = plt.subplots(1, 2, figsize=(11, 4))
        for algo, adf in rl.groupby("algo"):
            adf = adf.sort_values("lam", key=lambda s: s.map(_lam_float))
            xs = [_lam_float(l) for l in adf["lam"]]
            ax_eff.errorbar(xs, adf[f"{EFF}_mean"], yerr=adf[f"{EFF}_std"].fillna(0),
                            marker="o", capsize=3, label=algo)
            ax_saf.errorbar(xs, adf[f"{SAFETY_PROXY}_mean"],
                            yerr=adf[f"{SAFETY_PROXY}_std"].fillna(0),
                            marker="o", capsize=3, label=algo)

        # fixed-time reference (lambda-independent) as a horizontal line
        base = sdf[sdf["algo"] == BASELINE]
        if not base.empty:
            ax_eff.axhline(base[f"{EFF}_mean"].iloc[0], ls="--", color="tab:orange",
                           label="fixed-time")
            ax_saf.axhline(base[f"{SAFETY_PROXY}_mean"].iloc[0], ls="--",
                           color="tab:orange", label="fixed-time")

        ax_eff.set(xlabel="safety weight λ", ylabel="mean waiting time (s)",
                   title=f"{scenario}: efficiency vs λ")
        ax_saf.set(xlabel="safety weight λ", ylabel="total stopped (safety proxy)",
                   title=f"{scenario}: safety proxy vs λ")
        for ax in (ax_eff, ax_saf):
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"tradeoff_{scenario}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="logs/comparison.csv")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"{args.csv} not found — run `python compare.py` first")

    df = load(args.csv)
    if df.empty:
        raise SystemExit(f"{args.csv} is empty — no results to plot yet")

    os.makedirs(args.out, exist_ok=True)
    written = plot_bars(df, args.out) + plot_tradeoff(df, args.out)

    if not written:
        print("nothing plotted (need at least one populated group)")
    else:
        print("wrote:")
        for path in written:
            print(f"  {path}")


if __name__ == "__main__":
    main()
