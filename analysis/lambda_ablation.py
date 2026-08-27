"""SP14: the safety-weight (lambda) ablation the project's own framing needs.

The corridor reward is efficiency - lambda * safety_penalty
(env_common.make_safety_reward_fn), and the whole project is titled
"safety-aware" signal control -- yet every corridor experiment from SP4 to
SP12 was run at the single value lambda=0.5. The efficiency/safety tradeoff
that framing rests on has never been measured. This measures it.

Design
------
Train (run_lambda_ablation.sh, 12 new runs): IDQN at lambda in
{0.0, 0.25, 0.75, 1.0} x seeds {42, 43, 44}, corridor_peak, min_green=10,
steps=100000. The lambda=0.5 x {42,43,44} arm already exists from SP5 and is
reused unchanged, so the 0.5 column here IS the published corridor result
rather than a re-run of it.

Evaluate (this script): each of the 15 checkpoints, zero-shot, on BOTH

  regular    corridor.net.xml            in-distribution -- the geometry every
                                         checkpoint trained on
  irregular  corridor_irregular.net.xml  SP8's 578m/78m asymmetric net, the one
                                         where the green_wave/idqn ranking flips

train() takes no net_file: training only ever happens on corridor.net.xml, so
"both geometries" necessarily means train-once/zero-shot-evaluate-twice, the
same protocol SP8-SP10 used. No new training engineering.

Metrics, both already logged -- no new instrumentation:

  delay_per_trip   analysis.tripinfo.reduce_tripinfo -> trip_time_loss_mean,
                   SUMO timeLoss per COMPLETED trip (the project's headline;
                   see that module for why not the step-CSV mean wait)
  safety_total     sum over the episode's decision windows of
                   system_safety_total (= brake + exposure) from the eval step
                   CSV, written by SafetyLoggingEnv._get_safety_info. The
                   column is a per-window total, not a running one, so summing
                   the rows is the episode cost.

trips is reported alongside delay because the two can only be compared at
comparable throughput: a policy that strands vehicles flatters its own
delay/trip. safety_total_per_trip is reported for the same reason on the
safety side -- a policy that lets fewer vehicles through has fewer chances to
brake hard.

Deliberately bypasses compare.py, same reason analysis/irregular_net_compare2.py
and analysis/geometry_sweep.py give: neither non-default net_file nor a swept
lambda is a dimension compare.py's glob resolves cleanly.

Resumable: an eval whose tripinfo XML and step CSV are already on disk is
reused rather than re-run (--force overrides), so the six lambda=0.5 cells and
anything from an interrupted run cost nothing.

    ./run_lambda_ablation.sh              # first: the 12 missing checkpoints
    python -m analysis.lambda_ablation    # -> analysis/lambda_ablation.csv
                                          #    results/lambda_ablation.png
"""
import argparse
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import matplotlib

matplotlib.use("Agg")  # headless: save files, never open a window
import matplotlib.pyplot as plt
import pandas as pd

import train_corridor_dqn as tcd
from analysis.tripinfo import reduce_tripinfo
from env_common import tripinfo_path

OUT_CSV = os.path.join(REPO, "analysis", "lambda_ablation.csv")
OUT_PNG = os.path.join(REPO, "results", "lambda_ablation.png")

SCENARIO = "corridor_peak"
MIN_GREEN = 10
STEPS = 100_000
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (42, 43, 44)

# label -> net file. "regular" is what every checkpoint trained on;
# "irregular" is SP8's flagship 578m/78m asymmetric net.
NETS = {
    "regular": "corridor.net.xml",
    "irregular": "corridor_irregular.net.xml",
}

# Same environment SP5 trained and evaluated the lambda=0.5 arm under. With
# SUMO's -1 default, junction deadlock becomes an absorbing state and the arms
# are not comparable -- see run_lambda_sweep.sh's header.
os.environ.setdefault("TIME_TO_TELEPORT", "300")

SAFETY_COLS = ("system_safety_brake", "system_safety_exposure", "system_safety_total")


def _step_csv(stem: str) -> str:
    """The eval step CSV for a run whose stem is `stem`.

    evaluate() returns this path directly when it runs; on the cached path it
    has to be recovered, because the connection label and episode number in the
    suffix are assigned at run time and are not derivable from the stem. The
    `_conn` in the pattern is what keeps a stem from matching its own longer
    siblings (`..._s100000_conn3_ep1.csv` vs `..._s100000_netirregular_...`).
    Newest wins if a stem was ever evaluated more than once.
    """
    hits = glob.glob(f"{stem}_conn*_ep*.csv")
    if not hits:
        raise FileNotFoundError(f"no step CSV for {stem}")
    return max(hits, key=os.path.getmtime)


def _safety(step_csv: str) -> dict:
    """Episode safety cost from one eval step CSV.

    Each row is one decision window and each safety column is that window's
    own total (SafetyLoggingEnv.step resets the accumulator every window), so
    the episode cost is the column sum, not the last value.
    """
    df = pd.read_csv(step_csv)
    missing = [c for c in SAFETY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{step_csv} lacks safety columns {missing}")
    out = {c.replace("system_", ""): float(df[c].sum()) for c in SAFETY_COLS}
    out["safety_windows"] = len(df)
    return out


def run_one(lam: float, seed: int, geometry: str, net_file: str,
            force: bool = False) -> dict:
    """Zero-shot eval of one (lambda, seed) checkpoint on one geometry."""
    stem = tcd._eval_out_stem(SCENARIO, SCENARIO, lam, seed, MIN_GREEN, STEPS,
                              net_file=net_file)
    trip = tripinfo_path(stem)
    cached = os.path.exists(trip) and bool(glob.glob(f"{stem}_conn*_ep*.csv"))
    if force or not cached:
        tcd.evaluate(SCENARIO, lam, seed, MIN_GREEN, STEPS, tripinfo=True,
                     eval_scenario=SCENARIO, net_file=net_file)
    row = reduce_tripinfo(trip)
    rec = {"lam": lam, "seed": seed, "geometry": geometry, "net_file": net_file,
           "delay_per_trip": row["trip_time_loss_mean"],
           "trips": row["trips_completed"], "cached": cached and not force}
    rec.update(_safety(_step_csv(stem)))
    rec["safety_total_per_trip"] = (rec["safety_total"] / rec["trips"]
                                    if rec["trips"] else float("nan"))
    return rec


def missing_checkpoints() -> list:
    """(lam, seed) pairs with no complete checkpoint on disk -- reported up
    front rather than discovered 20 minutes into the eval loop."""
    out = []
    for lam in LAMBDAS:
        for seed in SEEDS:
            paths = [tcd._model_path(a, SCENARIO, lam, seed, MIN_GREEN, STEPS)
                     for a in tcd.CORRIDOR_TS_IDS]
            if not all(os.path.exists(p) for p in paths):
                out.append((lam, seed))
    return out


def run_all(force: bool = False) -> pd.DataFrame:
    rows = []
    for geometry, net_file in NETS.items():
        for lam in LAMBDAS:
            for seed in SEEDS:
                r = run_one(lam, seed, geometry, net_file, force)
                rows.append(r)
                print(f"[{geometry}/lam{lam:<4}] seed{seed} "
                      f"delay/trip={r['delay_per_trip']:7.2f}s "
                      f"trips={r['trips']:5d} "
                      f"safety_total={r['safety_total']:9.1f} "
                      f"({'cached' if r['cached'] else 'ran'})", flush=True)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    for geometry in NETS:
        g = df[df["geometry"] == geometry]
        if g.empty:
            continue
        print(f"\n=== {SCENARIO}, mg{MIN_GREEN}, {geometry} "
              f"({NETS[geometry]}): metrics vs lambda (n={len(SEEDS)} seeds) ===")
        summary = g.groupby("lam").agg(
            delay_mean=("delay_per_trip", "mean"),
            delay_std=("delay_per_trip", "std"),
            trips_mean=("trips", "mean"),
            safety_mean=("safety_total", "mean"),
            safety_std=("safety_total", "std"),
            safety_per_trip=("safety_total_per_trip", "mean"),
            brake_mean=("safety_brake", "mean"),
            exposure_mean=("safety_exposure", "mean"),
        )
        print(summary.to_string(float_format=lambda x: f"{x:9.2f}"))

    print("\n--- change from lambda=0 (efficiency-only), paired by seed ---")
    print("    negative delay = safer weighting also faster; "
          "negative safety = the safety term is doing what it claims")
    for geometry in NETS:
        g = df[df["geometry"] == geometry]
        if g.empty:
            continue
        print(f"  [{geometry}]")
        for metric in ("delay_per_trip", "safety_total"):
            wide = g.pivot_table(index="seed", columns="lam", values=metric)
            if 0.0 not in wide.columns:
                continue
            line = []
            for lam in LAMBDAS:
                if lam == 0.0 or lam not in wide.columns:
                    continue
                d = (wide[lam] - wide[0.0]).dropna()
                if d.empty:
                    continue
                line.append(f"lam{lam}: {d.mean():+8.2f} +/- {d.std():6.2f}")
            print(f"    {metric:<16} " + "   ".join(line))

    print("\n--- is there a tradeoff at all? (Pearson r over the 5 lambda means) ---")
    print("    r > 0 means lower delay comes with lower safety cost -- i.e. the two")
    print("    objectives are aligned here, not traded off")
    for geometry in NETS:
        g = df[df["geometry"] == geometry]
        if g.empty:
            continue
        means = g.groupby("lam")[["delay_per_trip", "safety_total"]].mean()
        if len(means) < 3:
            continue
        r = means["delay_per_trip"].corr(means["safety_total"])
        best_delay = means["delay_per_trip"].idxmin()
        best_safety = means["safety_total"].idxmin()
        print(f"  [{geometry}] r={r:+.3f}   best delay at lam={best_delay}, "
              f"best safety at lam={best_safety}")

    windows = df["safety_windows"].unique()
    if len(windows) > 1:
        print(f"\nNOTE: episodes differ in decision-window count {sorted(windows)} -- "
              "safety_total sums are over unequal horizons, compare "
              "safety_total_per_trip instead")


def plot(df: pd.DataFrame, path: str = OUT_PNG) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    styles = {"regular": ("o-", "tab:blue"), "irregular": ("s--", "tab:orange")}

    for geometry in NETS:
        g = df[df["geometry"] == geometry]
        if g.empty:
            continue
        marker, color = styles.get(geometry, ("o-", None))
        m = g.groupby("lam")["delay_per_trip"].agg(["mean", "std"])
        axes[0].errorbar(m.index, m["mean"], yerr=m["std"], fmt=marker,
                         color=color, capsize=3, label=geometry)
        s = g.groupby("lam")["safety_total"].agg(["mean", "std"])
        axes[1].errorbar(s.index, s["mean"], yerr=s["std"], fmt=marker,
                         color=color, capsize=3, label=geometry)
        # tradeoff plane: each point is one lambda's (safety, delay) mean.
        # A real tradeoff traces a downward-sloping frontier; an aligned pair
        # of objectives traces an upward-sloping cloud.
        axes[2].plot(s["mean"], m["mean"], marker, color=color, label=geometry)
        for lam in m.index:
            axes[2].annotate(f"{lam:g}", (s.loc[lam, "mean"], m.loc[lam, "mean"]),
                             textcoords="offset points", xytext=(5, 4), fontsize=8)

    axes[0].set_xlabel("safety weight $\\lambda$")
    axes[0].set_ylabel("delay per completed trip (s)")
    axes[0].set_title("Efficiency vs $\\lambda$")
    axes[1].set_xlabel("safety weight $\\lambda$")
    axes[1].set_ylabel("episode safety cost (brake + exposure)")
    axes[1].set_title("Safety cost vs $\\lambda$")
    axes[2].set_xlabel("episode safety cost")
    axes[2].set_ylabel("delay per completed trip (s)")
    axes[2].set_title("Tradeoff plane (labelled by $\\lambda$)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(title="eval geometry", fontsize=8)
    fig.suptitle(f"IDQN safety-weight ablation -- {SCENARIO}, min_green={MIN_GREEN}, "
                 f"{STEPS//1000}k steps, seeds {SEEDS[0]}-{SEEDS[-1]} "
                 "(trained on the regular net, evaluated zero-shot on both)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")
    return path


def main():
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="re-run every eval instead of reusing cached tripinfo/CSV")
    p.add_argument("--plot-only", action="store_true",
                   help="re-render the figure from the existing CSV, run nothing")
    args = p.parse_args()

    if args.plot_only:
        df = pd.read_csv(OUT_CSV)
    else:
        missing = missing_checkpoints()
        if missing:
            raise SystemExit(
                "missing checkpoints for (lam, seed): "
                + ", ".join(f"({l}, {s})" for l, s in missing)
                + "\nrun ./run_lambda_ablation.sh first")
        df = run_all(args.force)
        df.to_csv(OUT_CSV, index=False)
        print(f"\nwrote {OUT_CSV} ({len(df)} rows)")
    report(df)
    plot(df)


if __name__ == "__main__":
    main()
