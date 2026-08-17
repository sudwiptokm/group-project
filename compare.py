"""
Aggregate evaluation metrics across algorithms and seeds -> comparison table.

Reads the per-episode CSVs written during evaluation, time-averages each run
over the episode, then reports mean ± std per (scenario, λ, algo) group and
ranks within each group by mean waiting time.

Filename conventions
--------------------
RL algos:    logs/eval_<algo>_<scenario>_lam<LAM>_seed<seed>[_t<n>][_mg<n>]_conn<N>_ep<M>.csv
             where <LAM> is the λ float with the dot removed: 0.0→"00", 0.5→"05", 1.0→"10",
             _t<n> is the training seed and _mg<n> the action-space floor
Fixed-time:  logs/eval_fixedtime_<scenario>_seed<seed>_g<green>_conn<N>_ep<M>.csv
             (NO lam fragment; <green> is the static plan's green duration)

Every optional fragment sits AFTER seed<n> so the globs below stay green- and
floor-agnostic. That is deliberate: the globs find the runs, and a separate
warning refuses to average two greens or two floors into one row, rather than
the glob quietly dropping half of them.

Each run's completed-trip metrics are read from the tripinfo XML beside its
step CSV (same stem, minus the _conn/_ep suffix — see env_common.tripinfo_path).

Ranking key
-----------
Default is trip_time_loss_mean: delay per COMPLETED trip. The old key,
system_mean_waiting_time, averages over vehicles still in the network, so it
rewards stranding traffic and turns into a clock under deadlock — it inverted
the Stage-1 ranking. Runs logged before tripinfo existed have no trip columns;
for those the table falls back to the old key and says so. Lower delay / more
throughput = better.

Workflow:
    # train each algo on several seeds / lambdas
    python train.py --algo dqn --scenario peak --lam 0.5 --seed 0 --steps 100000
    # run fixed-time baseline
    python baseline.py --scenario peak --seed 42
    # build the table
    python compare.py
"""

import argparse
import glob
import os
import re

import pandas as pd

from analysis.tripinfo import count_departures, reduce_tripinfo
from env_common import CORRIDOR_SCENARIOS, SCENARIO_ROUTES, tripinfo_path

# episode-averaged columns we surface (present in every sumo-rl eval CSV)
METRICS = [
    # completed-trip metrics (tripinfo) — the headline; see module docstring
    "trip_time_loss_mean",
    "trips_completed",
    "trip_completion_rate",
    "trip_waiting_time_mean",
    "trip_duration_mean",
    # in-network step metrics (sumo-rl CSV) — retained, survivorship-biased
    "system_mean_waiting_time",
    "system_total_stopped",
    "system_mean_speed",
    "system_total_waiting_time",
    "system_safety_brake",     # vulnerability-weighted emergency braking (SafetyLoggingEnv)
    "system_safety_exposure",  # vulnerability-weighted intersection exposure
    "system_safety_total",     # brake + exposure = raw safety penalty
]
RANK_KEY = "trip_time_loss_mean"              # lower is better
LEGACY_RANK_KEY = "system_mean_waiting_time"  # pre-tripinfo runs

# sumo-rl appends "_conn<N>_ep<M>.csv" to the stem make_env was given
_RUN_SUFFIX = re.compile(r"_conn\d+_ep\d+\.csv$")


def _trip_metrics(csv_path: str, departed: int = None) -> dict:
    """Completed-trip metrics for the run that wrote `csv_path` ({} if none)."""
    stem = _RUN_SUFFIX.sub("", csv_path)
    return reduce_tripinfo(tripinfo_path(stem), departed=departed)


def _run_means(logs_dir: str, entity: str, scenario: str, lam=None,
               departed: int = None) -> pd.DataFrame:
    """One row per eval run for `entity`/`scenario`/`lam`, each metric time-averaged.

    Parameters
    ----------
    logs_dir : str
        Directory containing eval CSVs.
    entity : str
        Algorithm name (e.g. "dqn") or "fixedtime".
    scenario : str
        Scenario name (e.g. "peak", "offpeak").
    lam : str or None
        Lambda tag string (e.g. "00", "05", "10") for RL algos.
        None for the fixed-time baseline (no lam fragment in filename).
    departed : int or None
        Vehicles the scenario's demand inserts, for the completion rate.
    """
    if lam is None:
        # fixed-time: eval_fixedtime_<scenario>_seed*.csv (with _conn*_ep* suffix)
        pattern = os.path.join(logs_dir, f"eval_{entity}_{scenario}_seed*.csv")
    else:
        # RL algo: eval_<algo>_<scenario>_lam<lam>_seed*.csv
        pattern = os.path.join(logs_dir, f"eval_{entity}_{scenario}_lam{lam}_seed*.csv")

    rows = []
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path)
        row = {m: df[m].mean() for m in METRICS if m in df.columns}
        row.update(_trip_metrics(path, departed))
        row["run"] = os.path.basename(path)
        rows.append(row)
    return pd.DataFrame(rows)


def _summarise(df: pd.DataFrame, entity: str, scenario: str, lam: str) -> dict:
    """Return a single summary dict for one (entity, scenario, lam) group."""
    row = {"scenario": scenario, "lam": lam, "algo": entity, "n_runs": len(df)}
    for m in METRICS:
        if m in df.columns:
            row[f"{m}_mean"] = df[m].mean()
            row[f"{m}_std"] = df[m].std()
    return row


_GREEN_TAG = re.compile(r"_g(\d+)_")


def _warn_mixed_greens(base: pd.DataFrame, scenario: str) -> None:
    """Refuse to average two different static plans into one 'fixedtime' row.

    The green duration sits in the filename precisely so it stays visible, but
    the compare glob is green-agnostic, so a logs dir holding both a 10 s and a
    60 s baseline would silently report their mean as "the" baseline. That is
    the same class of confound as evaluating agents and baseline on different
    demand seeds — the one that flipped the Stage-1 headline.
    """
    if base.empty:
        return
    greens = {(_GREEN_TAG.search(r).group(1) if _GREEN_TAG.search(r) else "unset")
              for r in base["run"]}
    if len(greens) > 1:
        print(f"  [!] {scenario}: fixed-time runs mix green durations "
              f"{sorted(greens)} — that row averages different static plans. "
              "Keep one green per logs dir ('unset' = the old 10 s cycler, "
              "from before baseline.py had --green).")


_MIN_GREEN_TAG = re.compile(r"_mg(\d+)[_.]")


def _warn_mixed_min_greens(df: pd.DataFrame, scenario: str, entity: str) -> None:
    """Refuse to average two action spaces into one row.

    `min_green` is not a tuning detail — analysis/actuated.py measures a factor
    of 5.6 between a 10 s floor and a 60 s one, larger than any difference
    between algorithms this project has found. Runs at two floors are two
    experiments, and averaging them is the same class of confound as evaluating
    agents and baseline on different demand seeds.
    """
    if df.empty or "run" not in df:
        return
    floors = {(m.group(1) if (m := _MIN_GREEN_TAG.search(r)) else "unset")
              for r in df["run"]}
    if len(floors) > 1:
        print(f"  [!] {scenario}/{entity}: runs mix min_green {sorted(floors)} "
              "— that row averages different action spaces. Keep one floor per "
              "logs dir ('unset' = 10 s, from before make_env's default moved "
              "to 60).")


def _departed(scenario: str, horizon: float):
    """Demand the scenario asks for; None if its route file is missing."""
    route = SCENARIO_ROUTES.get(scenario)
    if not route or not os.path.exists(route):
        return None
    return count_departures(route, horizon=horizon)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="logs")
    p.add_argument("--rank", default=RANK_KEY,
                   help=f"metric to rank by, lower=better (default {RANK_KEY})")
    # the completion-rate denominator is demand over the episode, so it has to
    # match the EPISODE_SECONDS the runs were produced with
    p.add_argument("--episode-seconds", type=float,
                   default=float(os.environ.get("EPISODE_SECONDS", "3600")),
                   help="episode length the eval runs used (denominator for "
                        "trip_completion_rate)")
    args = p.parse_args()

    scenarios = ["peak", "offpeak"]
    lambdas = ["00", "05", "10"]   # filename lambda tags
    algos = ["dqn", "qrdqn", "ppo", "a2c"]

    rows = []
    for scenario in scenarios:
        departed = _departed(scenario, args.episode_seconds)
        # fixed-time baseline is lambda-independent — load once per scenario
        base = _run_means(args.logs, "fixedtime", scenario, departed=departed)
        _warn_mixed_greens(base, scenario)
        for lam in lambdas:
            for algo in algos:
                df = _run_means(args.logs, algo, scenario, lam=lam, departed=departed)
                if df.empty:
                    continue
                _warn_mixed_min_greens(df, scenario, f"{algo}/lam{lam}")
                rows.append(_summarise(df, algo, scenario, lam))
            # always add fixed-time row to every (scenario, lam) group for direct comparison
            if not base.empty:
                rows.append(_summarise(base, "fixedtime", scenario, lam))

    corridor_scenarios = list(CORRIDOR_SCENARIOS)
    corridor_controllers = ["green_wave", "max_pressure"]
    for scenario in corridor_scenarios:
        for ctrl in corridor_controllers:
            df = _run_means(args.logs, ctrl, scenario)
            if not df.empty:
                rows.append(_summarise(df, ctrl, scenario, lam="na"))

    if not rows:
        print("no eval CSVs found yet")
        return

    out_df = pd.DataFrame(rows)

    # runs logged before tripinfo existed carry no trip columns; ranking on an
    # all-missing key would silently order by nothing, so fall back and say so
    rank_key = args.rank
    rank_col = f"{rank_key}_mean"
    note = ""
    if rank_col not in out_df.columns or out_df[rank_col].isna().all():
        note = (f"  [!] no {rank_key} in these runs — ranked by {LEGACY_RANK_KEY},"
                " which is survivorship-biased; re-run eval to get tripinfo]")
        rank_key = LEGACY_RANK_KEY
        rank_col = f"{rank_key}_mean"
    elif out_df[rank_col].isna().any():
        note = "  [!] some runs predate tripinfo and sort last]"

    out_df = out_df.sort_values(
        ["scenario", "lam", rank_col],
        ascending=[True, True, True],
        na_position="last",
    )

    # pretty print with mean ± std columns
    print("\n=== Algorithm comparison grouped by (scenario, λ) "
          f"— ranked by {rank_key} (lower=better) ==={note}\n")
    display_cols = ["scenario", "lam", "algo", "n_runs"]
    display = out_df[display_cols].copy()
    for m in METRICS:
        mc, sc = f"{m}_mean", f"{m}_std"
        if mc in out_df.columns:
            display[m] = [
                f"{a:.3f} ± {b:.3f}" if pd.notna(b) else f"{a:.3f}"
                for a, b in zip(out_df[mc], out_df[sc])
            ]
    print(display.to_string(index=False))

    os.makedirs(args.logs, exist_ok=True)
    csv_path = os.path.join(args.logs, "comparison.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"\nfull numeric table -> {csv_path}")


if __name__ == "__main__":
    main()
