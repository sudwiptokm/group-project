"""Plot the adaptive-headroom result: a static plan against a queue-actuated one.

The static sweep (analysis/plot_static_timing.py) shows a fixed plan beating
every learned policy this project produced. This figure answers the question
that raises and the sweep cannot: was there an adaptive policy to find at all?
A non-learning queue-actuated controller needs no reward and no sample budget,
so if IT cannot beat the best static plan, the headroom is not there.

Both series share one x axis, and the shared quantity is the SAME constraint
seen from two sides: for the static plan it is the green it holds per phase, for
the actuated controller it is `min_green`, the floor below which a switch
request is ignored. Both are "how long a green is forced to run", which is the
parameter the amber arithmetic points at. They are NOT two y scales -- both
series are delay per completed trip, in seconds, over the same seeds.

Read the left-hand end, not the crossover: at a 10 s floor a controller with
perfect queue information is 5.6x worse than a fixed plan, and that floor is
what every peak training run in this project used.

    python analysis/plot_headroom.py     # -> results/headroom_peak.png
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

from analysis.plot_static_timing import collect as collect_static
from analysis.tripinfo import count_departures, reduce_tripinfo
from env_common import SCENARIO_ROUTES, tripinfo_path

ACTUATED_LOGS = os.path.join(REPO, "analysis", "actuated_logs")
ACT_RUN = re.compile(r"peak_mg(\d+)_seed(\d+)_conn\d+_ep\d+\.csv$")

# Greens indistinguishable from one another over seeds 42-46 -- reported as a
# band so nobody quotes a tuned argmin. Same band as plot_static_timing.py.
PLATEAU = (45, 90)

INK = "#1f2933"
MUTED = "#7b8794"
STATIC = "#1f6feb"    # same hue the static-sweep figure already uses
ACTUATED = "#c2610a"  # blue/orange: CVD dE 30.5, and >=3:1 on the surface
CALLOUT = "#b42318"


def collect_actuated(logs_dir: str = ACTUATED_LOGS,
                     horizon: float = 3600.0) -> pd.DataFrame:
    """One row per (min_green, seed) actuated run found on disk."""
    departed = count_departures(SCENARIO_ROUTES["peak"], horizon=horizon)
    rows = []
    for path in sorted(glob.glob(os.path.join(logs_dir, "*_conn*_ep*.csv"))):
        m = ACT_RUN.search(os.path.basename(path))
        if not m:
            continue
        stem = path[: path.index("_conn")]
        trips = reduce_tripinfo(tripinfo_path(stem), departed=departed)
        if not trips:
            continue  # run was killed before it wrote a usable tripinfo
        rows.append({
            "x": int(m.group(1)),
            "seed": int(m.group(2)),
            "delay": trips["trip_time_loss_mean"],
            "completion": trips.get("trip_completion_rate", float("nan")),
        })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("x").agg(
        n=("seed", "count"),
        delay=("delay", "mean"), delay_sd=("delay", "std"),
        completion=("completion", "mean"),
    )
    return g.reset_index().sort_values("x")


def plot(static: pd.DataFrame, act: pd.DataFrame, out: str) -> str:
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8.6, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.12},
    )

    lo, hi = PLATEAU
    ax.axvspan(lo, hi, color=STATIC, alpha=0.05, zorder=0)

    # Linear y, deliberately. The range is 82-517 s, so a log axis buys nothing
    # and costs the headline: it squeezes every series into the top of the panel
    # and makes a 5.6x gap look like a small one.
    #
    # Direct labels sit where the curves are furthest apart, not at their
    # right-hand ends -- there the two series converge and the labels collide.
    for df, colour, label, anchor, offset in (
            (static, STATIC, "static plan", 20, (12, 18)),
            (act, ACTUATED, "queue-actuated", 30, (12, -24))):
        ax.errorbar(df.x, df.delay, yerr=df.delay_sd.fillna(0), color=colour,
                    linewidth=2, marker="o", markersize=7, capsize=4,
                    ecolor=MUTED, elinewidth=1, label=label, zorder=2)
        at = df[df.x == anchor]
        if not at.empty:
            ax.annotate(label, xy=(anchor, at.iloc[0].delay), xytext=offset,
                        textcoords="offset points", color=colour, fontsize=10,
                        fontweight="bold", zorder=5)

    # low in the panel: the top is where the 10 s cliff and its callout live
    ax.annotate(f"static plateau {lo}–{hi} s", xy=((lo + hi) / 2, 0.04),
                xycoords=("data", "axes fraction"), color=MUTED, fontsize=9,
                ha="center")

    # The headline is the left-hand end, not the crossover: this is the floor
    # every peak training run used, and it is unwinnable for any controller.
    worst = act[act.x == 10]
    if not worst.empty:
        row = worst.iloc[0]
        ax.plot(row.x, row.delay, marker="o", markersize=9, color=CALLOUT,
                zorder=4)
        ax.annotate("the floor every peak run used —\n"
                    "5.6× the fixed plan, with nothing to learn",
                    xy=(row.x, row.delay), xytext=(34, 26),
                    textcoords="offset points", color=CALLOUT, fontsize=10,
                    arrowprops=dict(arrowstyle="-", color=CALLOUT,
                                    linewidth=1, alpha=0.6))

    best = act[act.x == 60]
    if not best.empty:
        row = best.iloc[0]
        # above the point: below it is the panel edge, and the annotation was
        # being clipped into the completion panel
        ax.annotate("60 s: matches the best static plan\n"
                    "(−9.3 s, inside the ±23.9 s paired spread)",
                    xy=(row.x, row.delay), xytext=(0, 44),
                    textcoords="offset points", color=INK, fontsize=9.5,
                    ha="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1,
                                    shrinkB=6))

    ax.set_ylabel("delay per completed trip (s)", color=INK)
    ax.set_title("min_green, not the algorithm, is what limits peak performance",
                 color=INK, fontsize=13, loc="left", pad=12)
    leg = ax.legend(frameon=False, loc="upper right", fontsize=10)
    for text in leg.get_texts():
        text.set_color(INK)

    for df, colour in ((static, STATIC), (act, ACTUATED)):
        ax2.plot(df.x, df.completion, color=colour, linewidth=2, marker="o",
                 markersize=6)
    ax2.set_ylabel("trips completed\n(share of demand)", color=INK, fontsize=9)
    ax2.set_xlabel("seconds of green forced per phase — static: the green it "
                   "holds; actuated: its min_green floor", color=INK,
                   fontsize=10)
    ax2.set_ylim(0, 1.05)

    for a in (ax, ax2):
        a.grid(axis="y", color="#e4e7eb", linewidth=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color("#cbd2d9")
        a.tick_params(colors=MUTED, labelsize=9)

    ns = pd.concat([static.n, act.n])
    lo_n, hi_n = int(ns.min()), int(ns.max())
    seeds = f"{hi_n}" if lo_n == hi_n else f"{lo_n}-{hi_n}"
    fig.text(0.01, -0.04,
             f"peak demand (1.5x), {seeds} seeds per point, 3600 s episodes, "
             "--time-to-teleport 300. Bars = sd over seeds. The actuated "
             "controller serves the largest PCU-weighted queue and learns "
             "nothing.",
             color=MUTED, fontsize=8)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/headroom_peak.png")
    a = p.parse_args()

    static_runs = collect_static()
    act_runs = collect_actuated()
    if static_runs.empty or act_runs.empty:
        raise SystemExit("need both sweeps: analysis/static_timing.py and "
                         "analysis/actuated.py")
    # plot_static_timing.collect names the x column "green"; one name here so
    # the two frames can go through the same summarise/plot path
    static_g = summarise(static_runs.rename(columns={"green": "x"}))
    act_g = summarise(act_runs)

    print("static:\n" + static_g.round(2).to_string(index=False))
    print("\nactuated:\n" + act_g.round(2).to_string(index=False))
    print(f"\nwrote {plot(static_g, act_g, a.out)}")
