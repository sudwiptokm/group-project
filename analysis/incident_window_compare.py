"""The number SP7's own scope section asked for and never shipped.

docs/superpowers/specs/2026-08-22-sp7-corridor-incident-design.md's Scope
section asks for "a comparison table reporting each controller's
incident-window and whole-episode delay". Only whole-episode delay made it
into docs/FINDINGS_2026-08-22-sp7-corridor-incident.md -- its own "What this
doesn't answer" section discloses that the reported whole-episode delta is
diluted by roughly the ratio of the incident window (900s) to the episode
(3600s), i.e. ~4x, because only trips departing during the closure are
exposed to it at all.

This computes the missing number directly from tripinfo XMLs SP7's own eval
already wrote to disk for the *_incident condition (no new incident-condition
SUMO runs). The only gap on disk was the no-incident WINDOW baseline for
green_wave -- SP4's sweep only kept the aggregated corridor_sweep.csv row,
not per-trip tripinfo, so three quick no-incident green_wave runs were
regenerated (idqn and max_pressure's no-incident tripinfo already existed).

"In-window" trip = depart time in [1800, 2700) -- corridor_baseline.INCIDENT's
own (start_s, start_s+duration_s). Trips that departed during the closure,
not trips merely still en route when it started or ended.

    python -m analysis.incident_window_compare
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd

import corridor_baseline as cb
from analysis.tripinfo import _parse

OUT_CSV = os.path.join(REPO, "analysis", "incident_window_compare.csv")

SEEDS = (42, 43, 44)
CONTROLLERS = ("green_wave", "max_pressure", "idqn")
WINDOW_START = cb.INCIDENT[2]
WINDOW_END = cb.INCIDENT[2] + cb.INCIDENT[3]

_PATH = {
    "green_wave": {
        False: "logs/eval_green_wave_corridor_peak_seed{seed}_mg10_tripinfo.xml",
        True: "logs/eval_green_wave_corridor_peak_seed{seed}_mg10_incident_tripinfo.xml",
    },
    "max_pressure": {
        False: "logs/eval_max_pressure_corridor_peak_seed{seed}_mg10_tripinfo.xml",
        True: "logs/eval_max_pressure_corridor_peak_seed{seed}_mg10_incident_tripinfo.xml",
    },
    "idqn": {
        False: "logs/eval_idqn_corridor_peak_lam05_seed{seed}_mg10_s100000_tripinfo.xml",
        True: "logs/eval_idqn_corridor_peak_lam05_seed{seed}_mg10_s100000_incident_tripinfo.xml",
    },
}


def _delay_stats(path: str, window_only: bool):
    trips = _parse(path)
    losses = []
    for t in trips:
        depart = float(t.get("depart"))
        if window_only and not (WINDOW_START <= depart < WINDOW_END):
            continue
        loss = t.get("timeLoss")
        if loss is not None:
            losses.append(float(loss))
    mean = sum(losses) / len(losses) if losses else float("nan")
    return mean, len(losses)


def build() -> pd.DataFrame:
    rows = []
    for controller in CONTROLLERS:
        for seed in SEEDS:
            for incident in (False, True):
                path = _PATH[controller][incident].format(seed=seed)
                if not os.path.exists(path):
                    print(f"[!] missing {path}, skipping")
                    continue
                whole_mean, whole_n = _delay_stats(path, window_only=False)
                win_mean, win_n = _delay_stats(path, window_only=True)
                rows.append({
                    "controller": controller, "seed": seed, "incident": incident,
                    "whole_episode_delay": whole_mean, "whole_episode_trips": whole_n,
                    "window_delay": win_mean, "window_trips": win_n,
                })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    print(f"\n=== incident window = [{WINDOW_START:.0f}, {WINDOW_END:.0f})s of a 3600s episode "
          f"({(WINDOW_END - WINDOW_START) / 3600:.0%} of it) ===\n")
    for controller in CONTROLLERS:
        g = df[df["controller"] == controller]
        if g.empty:
            continue
        wide_whole = g.pivot(index="seed", columns="incident", values="whole_episode_delay")
        wide_win = g.pivot(index="seed", columns="incident", values="window_delay")
        if True not in wide_whole or False not in wide_whole:
            print(f"  {controller}: incomplete data, skipping")
            continue
        d_whole = (wide_whole[True] - wide_whole[False])
        d_win = (wide_win[True] - wide_win[False])
        ratio = (d_win.mean() / d_whole.mean()) if d_whole.mean() else float("nan")
        n_win = g[g["incident"]]["window_trips"].mean()
        n_whole = g[g["incident"]]["whole_episode_trips"].mean()
        print(f"  {controller:13s} whole-episode Δ: {d_whole.mean():+6.2f}s +/- {d_whole.std():4.2f}s   "
              f"(reported headline)")
        print(f"  {'':13s} in-window Δ:     {d_win.mean():+6.2f}s +/- {d_win.std():4.2f}s   "
              f"({n_win:.0f}/{n_whole:.0f} trips = {n_win/n_whole:.0%} of trips exposed)")
        print(f"  {'':13s} in-window / whole-episode ratio: {ratio:.2f}x\n")


def main():
    df = build()
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(df)} rows)")
    report(df)


if __name__ == "__main__":
    main()
