"""In-window vs whole-episode incident delay, computed directly from
tripinfo XMLs an eval run already wrote to disk.

Background: SP7's own scope section
(docs/superpowers/specs/2026-08-22-sp7-corridor-incident-design.md) asked
for both an incident-window and a whole-episode delay number; only the
whole-episode number shipped in
docs/FINDINGS_2026-08-22-sp7-corridor-incident.md, which discloses that the
whole-episode delta is diluted by roughly the ratio of the incident window
(900s) to the episode (3600s) -- only trips departing during the closure are
exposed to it at all. A later session (referred to in this project's history
as "SP8") built a script of this name to compute the missing number for
SP7's zero-shot idqn/green_wave/max_pressure comparison; that session's work
was left uncommitted and is not part of this worktree's branch history (this
worktree branched at the SP6+SP7 handoff commit, before that session ran).
This file is a from-scratch recreation of that same measurement, generalized
with a `variant` parameter so it can also compute the in-window number for
an SP12 incident-aware checkpoint's own eval outputs, not only SP7's plain
corridor_peak checkpoints -- see
docs/FINDINGS_<date>-sp12-incident-aware-idqn.md.

"In-window" trip = depart time in [WINDOW_START, WINDOW_END) --
corridor_baseline.INCIDENT's own (start_s, start_s+duration_s). Trips that
departed during the closure, not trips merely still en route when it
started or ended (matches the earlier script's definition, cross-checked
against the numbers it reported: recomputing from
`logs/eval_idqn_corridor_peak_lam05_seed{42,43,44}_mg10_s100000[_incident]_tripinfo.xml`
with this exact filter reproduces its published in-window deltas of
green_wave +4.21s / max_pressure +2.69s / idqn +1.13s to within rounding).

    python -m analysis.incident_window_delta --stem-no-incident PATH --stem-incident PATH
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import corridor_baseline as cb
from analysis.tripinfo import _parse

WINDOW_START = cb.INCIDENT[2]
WINDOW_END = cb.INCIDENT[2] + cb.INCIDENT[3]


def delay_stats(path: str, window_only: bool) -> tuple:
    """Mean timeLoss and trip count from the tripinfo at `path`.

    Restricted to trips whose `depart` time falls in
    [WINDOW_START, WINDOW_END) when `window_only` is True; every completed
    trip in the episode otherwise.
    """
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


def window_delta(no_incident_path: str, incident_path: str) -> dict:
    """Both whole-episode and in-window delay/delta for one (seed,
    checkpoint) pair, from its own pair of tripinfo files -- seed-matched by
    construction, since both paths are for the same seed."""
    whole_no, n_whole_no = delay_stats(no_incident_path, window_only=False)
    whole_inc, n_whole_inc = delay_stats(incident_path, window_only=False)
    win_no, n_win_no = delay_stats(no_incident_path, window_only=True)
    win_inc, n_win_inc = delay_stats(incident_path, window_only=True)
    return {
        "whole_episode_no_incident": whole_no,
        "whole_episode_incident": whole_inc,
        "whole_episode_delta": whole_inc - whole_no,
        "whole_episode_trips_no_incident": n_whole_no,
        "whole_episode_trips_incident": n_whole_inc,
        "window_no_incident": win_no,
        "window_incident": win_inc,
        "window_delta": win_inc - win_no,
        "window_trips_no_incident": n_win_no,
        "window_trips_incident": n_win_inc,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--stem-no-incident", required=True,
                   help="path to the no-incident tripinfo XML")
    p.add_argument("--stem-incident", required=True,
                   help="path to the incident tripinfo XML (same seed/checkpoint)")
    args = p.parse_args()
    result = window_delta(args.stem_no_incident, args.stem_incident)
    for k, v in result.items():
        print(f"{k:34s} {v}")
