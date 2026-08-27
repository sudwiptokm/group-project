# SP13c Findings: does the crossing count move monotonically with span, or jump?

## The question

SP13 (docs/FINDINGS_2026-08-26-sp13-geometry-dose-response.md) found
green_wave's failure at span=400 is a single bounded band, `r` in [0.51,
0.80], with one crossing on each side. SP13b
(docs/FINDINGS_2026-08-27-sp13b-span700-confound.md) reran the same 8-ratio
sweep at span=700 and found three crossings, not one (r≈0.572, 0.750, 0.842),
plus a shifted, non-recovering shape at the high-r end. With only two span
points sampled, SP13b's own "what this doesn't answer" flagged that the
crossing count/location's dependence on span was unresolved — 1-to-3 could be
a gradual drift (some intermediate span shows 2, or a partial third crossing)
or a sharp regime change. This adds a third span point, 550m (the midpoint of
400 and 700), to distinguish the two.

## Method

Identical construction to `build_geometry_sweep_nets_span700.py`, only
`SPAN=550` (`analysis/build_geometry_sweep_nets_span550.py`), same 8 ratios
(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90), all 8 nets newly built (no
net files are shared across the three span axes). Same `corridor_peak`
demand, `min_green=10`, zero-shot protocol, same seed sets (green_wave/
max_pressure n=5 seeds 42-46, idqn n=3 seeds 42-44 using SP5's existing
checkpoints — no new training). `analysis/geometry_sweep_span550.py`, mirrors
`geometry_sweep_span700.py` structure exactly.

## Results

Delay/trip in seconds per completed trip, mean (n=5 baseline, n=3 idqn):

| r | green_wave | idqn | max_pressure | idqn − green_wave |
|---|---:|---:|---:|---:|
| 0.50 | 23.15s | 18.36s | 23.48s | −4.64s (idqn ahead) |
| 0.55 | 20.57s | 18.42s | 23.74s | −2.04s (idqn ahead) |
| 0.60 | 17.50s | 18.50s | 24.54s | +1.03s (green_wave ahead) |
| 0.65 | 16.31s | 18.00s | 25.55s | +1.73s (green_wave ahead) |
| 0.70 | 16.70s | 18.16s | 25.21s | +1.50s (green_wave ahead) |
| 0.75 | 17.69s | 18.27s | 22.60s | +0.63s (green_wave ahead) |
| 0.80 | 23.72s | 18.21s | 22.86s | −5.66s (idqn ahead) |
| 0.90 | 18.08s | 18.27s | 24.08s | +0.19s (green_wave ahead, ~tied) |

Three interpolated crossings: **r≈0.584, r≈0.755, r≈0.897** — matching
span=700's count (three) and landing close to span=700's own locations
(0.572, 0.750, 0.842), not anywhere near span=400's single crossing pair. idqn
stays flat (18.00s-18.50s, a 0.50s band — the tightest of all three spans yet).

## Verdict: the regime change already happened by span=550 — it's a jump, not a drift

**The 1-crossing-to-3-crossing transition is not gradual.** span=550 sits
exactly midway between span=400 (1 crossing) and span=700 (3 crossings), and
it already shows all three crossings, at locations close to span=700's, not
halfway between span=400's single band edges and span=700's three points.
Whatever mechanism produces the extra two crossings is already fully present
by 550m — the transition happens somewhere in (400, 550], not smoothly across
the whole (400, 700) range. A fourth point inside (400, 550) would be needed
to localize it further; this sweep can't say whether it's at 450m or 549m,
only that it's not at 550m or beyond.

**But the r=0.50 baseline breaks the "span costs green_wave delay
monotonically" story SP13b proposed.** SP13b found span=700's symmetric-net
baseline (16.72s) already ran 3.26s slower than span=400's (13.46s),
suggesting longer corridors cost green_wave delay on their own. span=550's
baseline is 23.15s — *higher than both neighbors*, not intermediate. This is
not a smooth function of span: something about span=550 specifically (not a
monotonic span-length effect) is driving green_wave's r=0.50 delay higher
than the longer span=700 net achieves. idqn's r=0.50 value (18.36s) sits
between span=400's (16.56s) and span=700's (19.56s), following span order —
not itself dramatically non-monotonic like green_wave's — so this looks like
a green_wave-specific effect, not a shared demand/geometry artifact.

**r=0.90 recovery is intermediate, not clean.** span=400's green_wave nearly
returns to its own r=0.50 baseline at r=0.90 (14.10s vs 13.46s, +0.64s,
"recovers"). span=700's gets *worse* at r=0.90 relative to its own baseline
(21.33s vs 16.72s, +4.61s, "does not recover" per SP13b). span=550's r=0.90
(18.08s) is *lower* than its own r=0.50 baseline (23.15s, a drop of 5.07s) —
which looks like recovery by that metric, but only because span=550's r=0.50
baseline is the anomalously high point, not because r=0.90 itself is low in
absolute terms (18.08s is unremarkable, in the same range as most of the
sweep's non-spike points). The "recovers at r=0.90" framing depends on what
you compare against; span=550 doesn't cleanly fit either span=400's or
span=700's story.

**idqn keeps getting flatter, not more variable, as span grows.** 400m's band
width, 700m's (0.82s), and now 550m's (0.50s) — narrower each time relative
to span=700, though span=400 vs span=550 vs span=700 isn't a monotonic
sequence by span value (550 < 700 in span but narrower band). Combined with
SP13b's finding, idqn's insensitivity to geometry asymmetry looks robust
across all three tested spans; only green_wave's failure shape moves.

## What this doesn't answer

- **Where exactly the 1→3 crossing transition happens.** Bracketed to
  (400m, 550m] now, down from (400m, 700m) — a real localization, but a
  fourth point in that narrower window (e.g. 450m or 500m) would pin it down
  further, not just confirm three crossings again.
- **Why span=550's r=0.50 baseline is anomalously high** (23.15s, above both
  400m's 13.46s and 700m's 16.72s). This isn't explained by anything in
  SP13/SP13b's proposed mechanisms (short-block absolute length) since r=0.50
  is the symmetric case with no short block at all. Could be a signal-offset/
  cycle-length interaction specific to 550m's absolute geometry, but that's
  unverified — nobody has inspected green_wave's actual offset schedule at
  this net.
- **r<0.50 still untested at any span** — same gap SP13 and SP13b both left
  open, carried forward again.
- **idqn's checkpoints are still n=3** (SP5's constraint), same caveat as
  SP13/SP13b.

## Addendum: inspecting green_wave's offset schedule rules it out as the cause

Follow-up to this doc's own open question about the r=0.50 anomaly.
`corridor_control.green_wave_offsets` sets each downstream signal's offset to
exactly the free-flow travel time from C1 (`(x - x0) / free_flow_speed`);
`fixed_time_phase` then holds a `phase_seconds`-long phase, cycling every
`num_phases * phase_seconds`. With `min_green=10`, `yellow_time=3`,
`delta_time=5` (`corridor_baseline.py`'s constants), `plan_phase_seconds`
gives 15s, and the corridor's 2-phase signals give a 30s cycle. In continuous
time this offset construction is exact by algebra — a platoon leaving C1 at
the start of its green always arrives at a downstream signal exactly as that
signal's own green starts, for *any* offset value, so geometry can't matter
in the ideal case. The only place span-dependence can enter is the 5-second
decision grid: env actions (and therefore phase requests) are only issued
every `delta_time=5s`, so each signal's true transition lands at the first
grid point at or after the ideal continuous transition time, a quantization
delay of `ceil(offset/5)*5 - offset` (0 to 5s), which then also caps the real
overlap between a signal's actual green window and the arriving platoon's
travel-time-implied window.

Computed for the three spans' r=0.50 (symmetric) nets, C1→C2 and C1→C3:

| span | signal | distance | offset | quantization delay | green/platoon overlap |
|---|---|---:|---:|---:|---:|
| 400 | C2 | 200m | 14.40s | 0.60s | 96.0% |
| 400 | C3 | 400m | 28.80s | 1.20s | 92.0% |
| 550 | C2 | 275m | 19.80s | 0.20s | 98.7% |
| 550 | C3 | 550m | 39.60s | 0.40s | 97.3% |
| 700 | C2 | 350m | 25.20s | 4.80s | 68.0% |
| 700 | C3 | 700m | 50.40s | 4.60s | 69.3% |

**This directly contradicts the observed ranking.** By quantization delay
alone, span=550 should be the *best*-aligned of the three (smallest delay,
highest overlap on both signals) and span=700 the worst — but the actual
r=0.50 delay/trip ranks span=550 worst (23.15s), span=700 middle (16.72s),
span=400 best (13.46s). The offset-schedule math, worked through carefully,
predicts the opposite of what happened.

Two more checks against alternative schedule-based explanations, both also
negative:

- **Edge length vs. node distance.** `_green_wave_inputs` computes offsets
  from node coordinates, but the actual routable edge is ~22m shorter than
  the node-to-node distance at every span (178m/200m, 253m/275m, 328m/350m —
  junction geometry eating a near-constant absolute amount), meaning real
  travel time is always slightly *less* than the offset assumes. This bias
  scales smoothly and monotonically with span; it can't produce a spike
  specific to span=550.
- **The SP7 incident closure isn't a factor here.** `corridor_baseline.run`
  only applies `INCIDENT` when explicitly requested (`incident=True`); these
  sweep runs call `cb.run(..., net_file=net_file)` without it, confirmed by
  filename (no `_incident` suffix on the tripinfo output actually read).
- **The delay is a uniform per-trip effect, not rare gridlock outliers**
  (checked the raw `tripinfo.xml` `timeLoss` distributions for seed 42 at all
  three spans): median tracks mean closely and the whole distribution scales
  together (span=550: mean 22.76s/median 22.17s/p99 52.87s vs span=400: mean
  13.44s/median 13.11s/p99 33.26s) — no heavy tail or bimodal spike that would
  indicate a specific spillback/deadlock event unique to span=550. This is
  consistent with a genuine, pervasive miscoordination affecting most trips
  similarly, just not one the offset/quantization schedule predicts.

**Conclusion: the deterministic control-logic layer is not the cause.** The
offset schedule, its grid quantization, and the edge-length/node-distance gap
are all clean, smooth, and (for the two quantifiable ones) actively point the
wrong direction. Whatever produces span=550's anomalous r=0.50 baseline lives
in emergent SUMO dynamics this layer doesn't model — car-following/queue
discharge behavior, junction internal geometry effects on turning movements,
or an interaction between the fixed demand pattern and this specific
absolute geometry. Confirming any of those needs a different kind of
evidence than a schedule inspection can produce: a space-time trajectory or
per-signal queue-length timeseries from the actual sim run, comparing
span=400/550/700 at r=0.50 directly. Not attempted here.
