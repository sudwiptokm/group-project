"""Generate the two figures the consolidated report cites: the geometry
dose-response curve (SP13/SP13e) and the lambda efficiency/safety curve
(SP14/SP14b). Reads existing committed CSVs only, no new simulation.

Usage: python -m analysis.make_report_figures
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "docs/figures"


def geometry_curve():
    hi = pd.read_csv("analysis/geometry_sweep.csv")
    lo = pd.read_csv("analysis/geometry_sweep_lowr.csv")
    lo = lo[lo.ratio < 0.50]  # avoid double-counting the shared r=0.50 point
    df = pd.concat([lo, hi], ignore_index=True)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = {"green_wave": "#1b7a3d", "idqn": "#1f5fa8", "max_pressure": "#a8461f"}
    for ctrl, g in df.groupby("controller"):
        agg = g.groupby("ratio")["delay_per_trip"].agg(["mean", "std"]).reset_index()
        agg = agg.sort_values("ratio")
        ax.errorbar(agg.ratio, agg["mean"], yerr=agg["std"].fillna(0), marker="o",
                    label=ctrl, color=colors.get(ctrl), capsize=3, linewidth=1.6)

    for lo_r, hi_r in [(0.51, 0.80), (0.103, 0.341)]:
        ax.axvspan(lo_r, hi_r, color="gray", alpha=0.12)

    ax.set_xlabel("asymmetry ratio r  (nominal C1-C2 length / span, span=400 m)")
    ax.set_ylabel("delay per completed trip (s)")
    ax.set_title("Geometry dose-response, span=400 m (SP13 + SP13e)\nshaded = bounded bands where idqn beats green_wave")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/geometry_dose_response.png", dpi=150)
    plt.close(fig)


def lambda_curve():
    n3 = pd.read_csv("analysis/lambda_ablation.csv")
    n10 = pd.read_csv("analysis/lambda_ablation_n10.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=False)
    for ax, geom in zip(axes, ["regular", "irregular"]):
        sub3 = n3[n3.geometry == geom].groupby("lam")["delay_per_trip"].agg(["mean", "std"]).reset_index()
        ax.errorbar(sub3.lam, sub3["mean"], yerr=sub3["std"], marker="o", color="#555555",
                    label="n=3 (SP14)", capsize=3, linewidth=1.6)

        sub10 = n10[n10.geometry == geom].groupby("lam")["delay_per_trip"].agg(["mean", "std"]).reset_index()
        ax.errorbar(sub10.lam, sub10["mean"], yerr=sub10["std"], marker="s", color="#1f5fa8",
                    label="n=10 (SP14b, lam 0.25/0.5 only)", capsize=3, linewidth=1.8, linestyle="--")

        ax.axvline(0.25, color="green", alpha=0.3, linestyle=":")
        ax.axvline(0.5, color="red", alpha=0.3, linestyle=":")
        ax.set_xlabel("lambda (safety weight)")
        ax.set_title(geom)
    axes[0].set_ylabel("delay per completed trip (s)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Efficiency/safety frontier: delay vs lambda (SP14/SP14b)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/lambda_curve.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    geometry_curve()
    lambda_curve()
    print("Wrote figures to", OUT)
