# SP13d Findings: a fourth span point breaks the "more crossings as span grows" story entirely

## The question

SP13 (span=400) found 1 crossing. SP13b (span=700) found 3. SP13c (span=550,
midway between them) also found 3, close to span=700's own locations —
bracketing the 1-to-3-crossing transition to (400m, 550m] and suggesting a
sharp jump rather than a gradual drift. This adds a fourth point, 450m, near
the low end of that bracket, to narrow the transition further: is it right
at the edge of 400m, or closer to 550m?

## Method

Identical construction to `build_geometry_sweep_nets_span550.py`, only
`SPAN=450` (`analysis/build_geometry_sweep_nets_span450.py`), same 8 ratios
(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90), all 8 nets newly built. Same
`corridor_peak` demand, `min_green=10`, zero-shot protocol, same seed sets
(green_wave/max_pressure n=5 seeds 42-46, idqn n=3 seeds 42-44, SP5's
existing checkpoints, no new training). `analysis/geometry_sweep_span450.py`.

## Results

Delay/trip in seconds per completed trip, mean (n=5 baseline, n=3 idqn):

| r | green_wave | idqn | max_pressure | idqn − green_wave |
|---|---:|---:|---:|---:|
| 0.50 | 28.36s | 17.23s | 28.48s | −11.02s (idqn ahead) |
| 0.55 | 27.30s | 17.20s | 27.35s | −9.96s (idqn ahead) |
| 0.60 | 25.25s | 17.40s | 25.52s | −7.75s (idqn ahead) |
| 0.65 | 22.96s | 17.63s | 24.26s | −5.32s (idqn ahead) |
| 0.70 | 20.92s | 17.82s | 21.43s | −3.11s (idqn ahead) |
| 0.75 | 19.72s | 17.66s | 21.85s | −2.12s (idqn ahead) |
| 0.80 | 19.45s | 17.70s | 22.90s | −1.77s (idqn ahead) |
| 0.90 | 18.21s | 18.10s | 27.26s | −0.19s (idqn ahead, ~tied) |

**Zero crossings.** idqn is ahead at every single ratio point, the gap
shrinking smoothly from −11.02s at r=0.50 to −0.19s at r=0.90 with no sign
change anywhere. Not span=400's single band, not span=550/700's three
crossings — a third distinct shape.

## Verdict: the crossing count is not moving toward anything monotonically

**The sequence across span so far is 1 → 0 → 3 → 3** (span 400, 450, 550,
700). Going from 400m to 450m, the crossing count didn't creep toward 550's
three — it dropped to zero first. Whatever SP13c's "regime change already
happened by 550m, not gradual" conclusion was describing, it is not a simple
step function of span either: there are at least two distinct regimes between
400m and 550m (zero crossings and three), not one transition point. A "does
the transition happen before or after X" framing no longer fits this data;
the honest read is that crossing count is not a monotonic or simply-bracketed
function of span at all across this range.

**The r=0.50 spike is not green_wave-specific — max_pressure spikes almost
identically, which changes the SP13c addenda's framing.** SP13c's two
addenda inspected green_wave's offset/quantization schedule and its queue
behavior, both implicitly treating this as a green-wave-coordination problem.
But max_pressure — a purely reactive, per-step local controller with no
offset schedule, no fixed cycle, and no coordination logic of any kind —
shows the *same* spike-and-decay shape as green_wave at span=450 (28.48s
falling to 22.90s at r=0.80, before ticking back up to 27.26s at r=0.90) and
was already elevated at span=550 (23.48s at r=0.50, next to green_wave's
23.15s) relative to span=400 (25.83s, actually its own highest r=0.50 value)
and span=700 (22.05s). Compare the four spans' r=0.50 row:

| span | green_wave | idqn | max_pressure |
|---|---:|---:|---:|
| 400 | 13.46s | 16.56s | 25.83s |
| 450 | 28.36s | 17.23s | 28.48s |
| 550 | 23.15s | 18.36s | 23.48s |
| 700 | 16.72s | 19.56s | 22.05s |

green_wave and max_pressure track each other closely at every span (both
spike together at 450/550, both are lower at 400/700); idqn is the outlier,
staying in a tight 16.56-19.56s band regardless of span. This is strong
evidence the r=0.50 spike is a **capacity/geometry effect that any
non-learning controller struggles with** — not a green-wave-specific
offset-alignment artifact. SP13c's offset-schedule addendum already found
the schedule math couldn't explain span=550's spike (it predicted the
opposite ranking); this result explains why the search there came up empty —
the offset schedule was never going to explain a problem that isn't
green-wave-specific in the first place. SP13c's second addendum (queue
buildup localized to C3, during its own green) likely still holds as a
correct description of what happens to green_wave specifically, but it's a
symptom shared with max_pressure's own struggle, not the root mechanism.

**idqn is the only controller that doesn't see this pattern at all** across
all four spans and all eight ratios sampled so far — consistent with SP13/
SP13b/SP13c's running theme that idqn's per-net training makes it robust to
whatever is producing these baseline-controller effects, but note this is
still an *observation*, not a mechanism: nobody has looked at what idqn's
trained policy is actually doing differently at C3 for span=450/550 that
green_wave and max_pressure aren't.

## What this doesn't answer

- **What's actually happening at C3 (or wherever) that affects both
  green_wave and max_pressure at span=450/550 but not 400/700.** SP13c's
  queue-timeseries method (`analysis/queue_timeseries_span_compare.py`) was
  built for this and could be pointed at max_pressure too, to check whether
  it shows the same C3-localized queue buildup green_wave did — that specific
  comparison hasn't been run.
- **Only 4 span points sampled now (400, 450, 550, 700), still not enough
  to characterize the true crossing-count function.** A 1-0-3-3 sequence
  means there are at least two regime boundaries in (400m, 550m]; more points
  in that range (e.g. 420m, 480m, 500m) would be needed to find them, not
  just one more.
- **r<0.50 still untested at any span** — same gap every prior doc in this
  series left open.
- **idqn's checkpoints are still n=3** (SP5's constraint), same caveat as
  every prior doc in this series.
- **Why idqn is immune is still just an observation.** No inspection of
  idqn's actual learned policy/Q-values at the affected geometries has been
  done to confirm what it's doing differently.

## Addendum: max_pressure's queue localization contradicts the "shared mechanism" reading above

The main results section above inferred a shared capacity/geometry effect
from aggregate delay/trip numbers alone (green_wave and max_pressure both
elevated at r=0.50, spans 450/550). Pointing
`analysis/queue_timeseries_span_compare.py` at max_pressure directly, across
all four spans, does not support that reading — **the two controllers' worst
queues are not co-located the same way:**

| controller | span | C1 mean | C2 mean | C3 mean | C3 max | C3 pct≥5 |
|---|---:|---:|---:|---:|---:|---:|
| green_wave | 400 | 1.04 | 0.11 | 0.21 | 5 | 0.28% |
| green_wave | 450 | 1.06 | 1.34 | **2.71** | 15 | 29.17% |
| green_wave | 550 | 1.04 | 0.40 | **2.50** | 14 | 27.50% |
| green_wave | 700 | 1.01 | 0.06 | 0.10 | 3 | 0.00% |
| max_pressure | 400 | 1.07 | 0.55 | 2.02 | 13 | 18.75% |
| max_pressure | 450 | 1.07 | 0.16 | 1.47 | 11 | 15.83% |
| max_pressure | 550 | 1.10 | **0.85** | **0.12** | 5 | 0.14% |
| max_pressure | 700 | 1.08 | 0.17 | 1.13 | 13 | 6.11% |

green_wave's C3 queue is worst exactly at the two spans (450, 550) where its
aggregate delay spikes, and clean at 400/700 — consistent with a single
localized story. max_pressure's C3 queue does *not* follow the same shape:
it's substantial at 400, 450, *and* 700, and — the sharpest contradiction —
**span=550 is max_pressure's best C3 point (0.12 mean, 0.14% ≥5), not a bad
one**, even though span=550 is one of the two spans where max_pressure's
own aggregate r=0.50 delay was elevated (23.48s). Whatever is driving
max_pressure's delay at span=550 is concentrated somewhere else — its C2
queue is elevated there relative to its own other spans (0.85 vs 0.16-0.55),
a different signal entirely.

**This weakens, not confirms, the "shared capacity/geometry effect"
inference this doc drew from aggregate numbers alone.** The two controllers
do both show elevated r=0.50 delay at spans 450/550 in the trip-level data,
but the per-signal instrumentation shows their actual congestion is not in
the same place: green_wave's problem is consistently C3; max_pressure's
moves around (C3 at 400/450/700, C2 at 550) and is *absent* from C3 at
exactly the span where green_wave's is worst. Two readings are both
consistent with this: either the aggregate similarity was partly coincidence
(two controllers independently struggling with a given geometry for
different local reasons), or there is a shared geometry-driven cause whose
effect on any given signal depends on each controller's own reactive/fixed
behavior (max_pressure's queue-driven phase selection could plausibly
relocate a bottleneck to wherever it currently has the least pressure,
unlike green_wave's fixed schedule). This data doesn't distinguish those two
readings, and the root mechanism remains genuinely open — if anything, more
open than this doc's main section suggested.
