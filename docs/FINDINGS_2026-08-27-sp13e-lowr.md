# SP13e Findings: r<0.50 (short block first) has its own bounded failure band, not a mirror of the r>0.50 one

## The question

SP13 (`docs/FINDINGS_2026-08-26-sp13-geometry-dose-response.md`) swept r in
[0.50, 0.90] at span=400 and found a bounded failure band, r≈[0.509, 0.797],
where green_wave loses to idqn — recovering on both sides. Every doc in this
series since (SP13b/c/d) extended the *span* axis but kept r≥0.50 throughout,
leaving r<0.50 (the short block coming FIRST, between the entry signal C1
and C2, rather than second, between C2 and the exit signal C3) untested at
every span. This is not just a relabeling of the same asymmetry: C1 is the
corridor's entry point (from W) and C3 is its exit (to E), so swapping which
block is short is a genuinely different geometry, not a mirror image. This
fills that gap at span=400 (SP13's original axis).

## Method

Same span=400, same node/edge template and netconvert invocation as
`build_geometry_sweep_nets.py`. 7 new ratio points below the existing 0.50:
r = 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.10
(`analysis/build_geometry_sweep_nets_lowr.py`), spaced to mirror the
original sweep's density (fine steps near 0.50, sparser toward the extreme).
r=0.50 reuses `corridor.net.xml` and its cached tripinfo (already on disk
from `geometry_sweep.py`) rather than re-running it. Same `corridor_peak`
demand, `min_green=10`, zero-shot protocol, same seed sets (green_wave/
max_pressure n=5 seeds 42-46, idqn n=3 seeds 42-44, SP5's existing
checkpoints, no new training). `analysis/geometry_sweep_lowr.py`.

## Results

Delay/trip in seconds per completed trip, mean (n=5 baseline, n=3 idqn):

| r | green_wave | idqn | max_pressure | idqn − green_wave |
|---|---:|---:|---:|---:|
| 0.10 | 17.69s | 17.87s | 26.40s | +0.09s (green_wave ahead, ~tied) |
| 0.20 | 23.11s | 17.19s | 22.58s | −6.03s (idqn ahead) |
| 0.25 | 22.13s | 16.93s | 21.32s | −5.30s (idqn ahead) |
| 0.30 | 21.46s | 17.12s | 20.61s | −4.44s (idqn ahead) |
| 0.35 | 15.88s | 16.88s | 23.82s | +0.96s (green_wave ahead) |
| 0.40 | 14.13s | 16.99s | 24.53s | +2.81s (green_wave ahead) |
| 0.45 | 13.59s | 16.97s | 25.19s | +3.32s (green_wave ahead) |
| 0.50 | 13.46s | 16.56s | 25.83s | +3.09s (green_wave ahead) |

Two crossings: **r≈0.103, r≈0.341** — a second bounded band, idqn ahead
inside [0.103, 0.341], green_wave ahead on both sides of it (r<0.103 and
0.341<r<0.50). idqn stays flat (16.56s-17.87s, a 1.3s band, matching every
other sweep in this series) — a fourth independent confirmation that idqn's
zero-shot policy is roughly geometry-invariant.

## Verdict: a real second band, not a mirror of the first

**SP13's original [0.51, 0.80] band and this low-r [0.10, 0.34] band are not
reflections of each other.** If block order didn't matter — if only the
magnitude of the length asymmetry mattered, not which block is near the
entry vs. the exit — reflecting the high-r band about r=0.50 would predict a
low-r band at [1−0.80, 1−0.51] = [0.20, 0.49]. The actual low-r band,
[0.103, 0.341], overlaps that prediction only partially and is shifted
noticeably toward the extreme. Block order is a real variable, not just a
labeling convention: a short C1-C2 block (near the entry) produces a
different — narrower and more skewed — failure region than a short C2-C3
block (near the exit) of the same relative length.

**green_wave's shape is not simply "worse toward one edge."** Its delay
across the low-r range is non-monotonic on its own: 13.46s (r=0.50) rising
to a peak around 23.11s (r=0.20), then dropping back to 17.69s at the most
extreme point tested (r=0.10) — a partial recovery at the edge, echoing
SP13's original "recovers outside the band on both sides" pattern, just with
a different peak location and a less complete recovery (17.69s at r=0.10 is
still 4.2s worse than the r=0.50 baseline, unlike SP13's r=0.90 point which
nearly matched its own baseline).

**max_pressure moves in the opposite direction from green_wave here — worth
flagging given SP13d's finding that the two baselines don't always share a
mechanism.** Across r=0.50 down to 0.10, max_pressure's delay *decreases*
monotonically-ish (25.83s → 26.40s at the very end, but 20.61-23.82s through
the middle, its *lowest* points in this whole sweep) while green_wave is
doing the opposite (peaking mid-range). The two baselines' worst points don't
coincide here either — another instance of the pattern SP13d's addendum
surfaced: aggregate-delay co-movement between green_wave and max_pressure is
not something to assume without checking, and here they don't even move in
the same direction.

**Loose, unconfirmed connection to SP8/SP10's original irregular2 net.**
SP10's `corridor_irregular2.net.xml` sits at r≈0.143 nominal — but on
span=700, not span=400, so it's not directly on this sweep's axis (same
caveat SP13 raised about `irregular`/`irregular3`). Its own r happens to
fall inside this doc's [0.103, 0.341] low-r band; whether that's meaningful
or coincidental can't be said without a matching span=700 (or 450/550)
low-r sweep, which hasn't been run.

## What this doesn't answer

- **r<0.50 is now tested only at span=400.** Spans 450, 550, and 700 —
  where SP13c/SP13d already found their high-r crossing counts don't move
  predictably (1→0→3→3) — still have no low-r data at all. Given how much
  the high-r shape varies by span, there's no reason to expect the low-r
  band found here generalizes to other spans either.
- **Only 7 new points sampled, same interpolation-only crossing-location
  caveat as every prior doc in this series.** r=0.10 was the sparsest,
  extreme point tested; nothing below it (shorter than a 40m C1-C2 block)
  has been tried, though netconvert built and validated this net without
  any warnings even at that length.
- **idqn's checkpoints are still n=3** (SP5's constraint), same caveat as
  every prior doc in this series.
- **Mechanism for the block-order asymmetry is not investigated.** This doc
  establishes that order matters (the low-r and high-r bands don't mirror),
  not why — no offset-schedule or queue-timeseries inspection was done here
  the way SP13c/d did for the span=550/450 anomalies.
