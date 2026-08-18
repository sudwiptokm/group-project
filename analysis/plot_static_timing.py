"""Plot the static green-duration sweep — the peak result the project now reports.

Reads the per-run CSVs and tripinfo XMLs written by analysis/static_timing.py and
draws delay per completed trip against green duration, with the spread over
seeds. Two panels sharing one x axis (never two y scales on one panel):

    upper   delay per completed trip (s), mean +/- sd over seeds  <- the headline
    lower   trips completed, as a share of the demand the routes ask for

The 10 s point is called out because it is what Stage 1 called "fixed-time" --
the sweep's own worst plan.

    python analysis/plot_static_timing.py            # -> results/static_timing_peak.png
"""
import argparse
import glob
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import matplotlib

matplotlib.use("Agg")  # headless: save files, never open a window
import matplotlib.pyplot as plt
import pandas as pd

from analysis.tripinfo import count_departures, reduce_tripinfo
from env_common import SCENARIO_ROUTES, tripinfo_path

LOGS = os.path.join(REPO, "analysis", "static_logs")

# Greens that are indistinguishable from one another over seeds 42-46: paired
# differences against the 60 s plan run +/-13 s where the seed-to-seed spread is
# ~30 s. Reported as a band precisely so nobody quotes a tuned argmin.
PLATEAU = (45, 90)
RUN = re.compile(r"g(\d+)_seed(\d+)_conn\d+_ep\d+\.csv$")

# one series -> one hue; the callout is the only other colour on the figure
INK = "#1f2933"
MUTED = "#7b8794"
SERIES = "#1f6feb"
CALLOUT = "#b42318"


def collect(logs_dir: str = LOGS, horizon: float = 3600.0) -> pd.DataFrame:
    """One row per (green, seed) run found on disk."""
    departed = count_departures(SCENARIO_ROUTES["peak"], horizon=horizon)
    rows = []
    for path in sorted(glob.glob(os.path.join(logs_dir, "g*_conn*_ep*.csv"))):
        m = RUN.search(os.path.basename(path))
        if not m:
            continue
        green, seed = int(m.group(1)), int(m.group(2))
        stem = path[: path.index("_conn")]
        trips = reduce_tripinfo(tripinfo_path(stem), departed=departed)
        if not trips:
            continue  # run predates tripinfo logging
        rows.append({
            "green": green,
            "seed": seed,
            "delay": trips["trip_time_loss_mean"],
            "completion": trips.get("trip_completion_rate", float("nan")),
            "wait": pd.read_csv(path).system_mean_waiting_time.mean(),
        })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("green").agg(
        n=("seed", "count"),
        delay=("delay", "mean"), delay_sd=("delay", "std"),
        completion=("completion", "mean"),
        wait=("wait", "mean"),
    )
    return g.reset_index().sort_values("green")


def plot(g: pd.DataFrame, out: str) -> str:
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12},
    )

    ax.errorbar(g.green, g.delay, yerr=g.delay_sd.fillna(0), color=SERIES,
                linewidth=2, marker="o", markersize=7, capsize=4,
                ecolor=MUTED, elinewidth=1)

    # Mark the PLATEAU, not the sample argmin. Paired over seeds, everything in
    # PLATEAU differs by less than the seed-to-seed spread, so calling any one of
    # them "the optimum" would be reading seed noise as a policy effect.
    lo, hi = PLATEAU
    band = g[(g.green >= lo) & (g.green <= hi)]
    if not band.empty:
        ax.axvspan(lo, hi, color=SERIES, alpha=0.06, zorder=0)
        ax.annotate(f"flat optimum: {lo}–{hi} s green\n"
                    "(differences < spread across seeds)",
                    xy=((lo + hi) / 2, band.delay.min()), xytext=(0, 52),
                    textcoords="offset points", color=INK, fontsize=10,
                    ha="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1))

    worst10 = g[g.green == 10]
    if not worst10.empty:
        row = worst10.iloc[0]
        ax.plot(row.green, row.delay, marker="o", markersize=9, color=CALLOUT,
                zorder=3)
        # below the marker: the curve rises steeply to the right of 10 s, so a
        # label placed there crosses the line it is annotating
        ax.annotate("what Stage 1 called\n“fixed-time”",
                    xy=(row.green, row.delay), xytext=(4, -42),
                    textcoords="offset points", color=CALLOUT, fontsize=10,
                    arrowprops=dict(arrowstyle="-", color=CALLOUT, linewidth=1,
                                    alpha=0.6))

    ax.set_ylabel("delay per completed trip (s)", color=INK)
    ax.set_title("Static green duration sets performance at peak demand",
                 color=INK, fontsize=13, loc="left", pad=12)

    ax2.plot(g.green, g.completion, color=SERIES, linewidth=2, marker="o",
             markersize=6)
    ax2.set_ylabel("trips completed\n(share of demand)", color=INK, fontsize=9)
    ax2.set_xlabel("green duration held per phase (s)", color=INK)
    ax2.set_ylim(0, 1.05)

    for a in (ax, ax2):
        a.grid(axis="y", color="#e4e7eb", linewidth=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color("#cbd2d9")
        a.tick_params(colors=MUTED, labelsize=9)

    lo, hi = int(g.n.min()), int(g.n.max())
    seeds = f"{hi}" if lo == hi else f"{lo}-{hi}"
    fig.text(0.01, -0.02,
             f"peak demand (1.5x), {seeds} seeds per point, 3600 s episodes, "
             "--time-to-teleport 300. Bars = sd over seeds.",
             color=MUTED, fontsize=8)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/static_timing_peak.png")
    p.add_argument("--table", default="analysis/static_sweep.csv",
                   help="where to write the aggregated sweep table")
    a = p.parse_args()

    df = collect()
    if df.empty:
        raise SystemExit(f"no completed sweep runs in {LOGS}")
    g = summarise(df)
    df.sort_values(["green", "seed"]).to_csv(a.table, index=False)
    print(g.round(2).to_string(index=False))
    print(f"\nwrote {plot(g, a.out)} and {a.table}")
