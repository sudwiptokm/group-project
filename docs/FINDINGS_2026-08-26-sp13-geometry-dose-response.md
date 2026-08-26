# SP13 Findings: is the idqn-beats-green_wave flip a monotonic dose-response, or a bounded failure band?

## The question

SP8-SP10 established that on irregular signal spacing, `idqn` beats
`green_wave` zero-shot — the one reversal of this project's standing
"consolidate on green_wave" result. SP10's own generalization check
(`docs/FINDINGS_2026-08-22-sp10-irregular-generalization.md`) sampled 3
discrete geometries and concluded the flip "generalizes (3/3)". But two of
those three (`corridor_irregular`/`corridor_irregular2`) used a different
total arterial span (700m, C1@0/C3@700) than the regular net (400m,
C1@0/C3@400) — asymmetry and overall corridor length were confounded, and
3 hand-picked points can't distinguish a monotonic trend from something with
real structure in between. This sweeps the asymmetry ratio continuously,
span held fixed, to find out what shape the effect actually has.

## Method

Ratio `r` = nominal C1-C2 length / 400 (fixed C1@0, C3@400 total span; `r`
= 0.50 is `corridor.net.xml` itself, the regular symmetric net). 8 points
swept: 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90 — denser in
[0.50, 0.75] since SP10's `corridor_irregular3.net.xml` (r=0.75, which
happens to sit on this same span=400 axis and is reused verbatim) already
showed idqn ahead while regular (r=0.50) showed the opposite; the threshold
had to be somewhere in that gap. 6 new nets built
(`analysis/build_geometry_sweep_nets.py`, same node/edge template and
`netconvert` invocation as `corridor_irregular3.nod.xml`); r=0.50 and 0.75
reuse existing nets and cached tripinfo, no re-run needed.

Zero-shot only, no new training: idqn's existing `corridor_peak` checkpoints
(SP5, seeds 42-44 — no others exist) evaluated on each geometry, same
posture as SP8/SP10. `green_wave`/`max_pressure` run 5 seeds each (42-46, no
checkpoint needed). All `corridor_peak` demand, `min_green=10`.
(`analysis/geometry_sweep.py`.)

## Results

Delay/trip in seconds per completed trip, mean (sd omitted from this table,
see CSV — max sd across all rows is 4.24s, on `max_pressure`; `green_wave`
and `idqn` sd stay under 0.65s throughout):

| r | green_wave | idqn | max_pressure | idqn − green_wave |
|---|---:|---:|---:|---:|
| 0.50 (regular) | 13.46s | 16.56s | 25.83s | +3.09s (green_wave ahead) |
| 0.55 | **31.17s** | 16.92s | 25.23s | −14.16s (idqn ahead) |
| 0.60 | 31.02s | 17.01s | 24.40s | −13.98s (idqn ahead) |
| 0.65 | 29.24s | 16.83s | 22.61s | −12.32s (idqn ahead) |
| 0.70 | 25.05s | 17.10s | 20.68s | −8.00s (idqn ahead) |
| 0.75 (irregular3) | 20.62s | 17.26s | 21.33s | −3.41s (idqn ahead) |
| 0.80 | 17.05s | 17.31s | 22.93s | +0.18s (green_wave ahead) |
| 0.90 | 14.10s | 17.84s | 25.01s | +3.69s (green_wave ahead) |

Two interpolated crossings, not one: **r≈0.509** and **r≈0.797**. Linear
interpolation on the mean gap between adjacent sampled points.

idqn is flat across the entire range (16.56s-17.84s, a 1.3s band) — the
zero-shot policy barely notices the geometry change. `green_wave` is the
volatile one: it spikes to 31.17s at r=0.55 (2.3x its regular-net delay),
stays elevated through r=0.70, then falls back toward regular-net
performance by r=0.90 (14.10s, close to r=0.50's 13.46s).

## Verdict: not a monotonic dose-response

**green_wave's failure is a bounded band around moderate asymmetry, not a
function that gets monotonically worse as spacing gets more unequal.** SP10's
"the flip generalizes, 3/3" was correct as far as it went — all 3 of its
sampled points happened to land inside the [0.51, 0.80] band where idqn
wins — but the implied picture (irregularity is bad, more irregularity is
worse) is wrong. On this span=400 axis, `green_wave` recovers outside the
band on *both* sides: near-symmetric (r→0.50) and heavily skewed (r→0.90,
where one block shrinks to 40m and effectively stops being a separate
coordination problem).

Plausible mechanism, not confirmed further here: `green_wave`'s single
per-signal offset is exact for the through-movement travel time from C1.
At moderate asymmetry the two unequal-length blocks each want a materially
different offset, and the shared plan serves neither well — the worst case
isn't the most extreme mismatch, it's the region where both blocks are
still long enough to matter but their required offsets diverge most from
whatever compromise the plan settles on. At extreme skew one block becomes
short enough that its offset requirement stops constraining the plan much,
so the shared offset degrades toward serving the one dominant block well
again, same as the regular net serves its one (symmetric) block well.

This changes how the project's irregular-spacing finding should be
stated. Not: "asymmetric spacing breaks green_wave, idqn is the fix for
irregular corridors." Instead: "green_wave is only bad in the [0.51, 0.80]
band of the asymmetry-ratio space sampled here" — plus a hedge, see below.

## What this doesn't answer

- **Span confound not resolved, only sidestepped.** This sweep fixes total
  span at 400m and only found SP10's `irregular`/`irregular2` variants
  (span=700, r=0.857/0.143 respectively on a differently-scaled axis) don't
  sit on this curve. Whether the [0.51, 0.80] band's boundaries are a
  property of the ratio alone, or interact with absolute span, is untested
  — SP10's irregular2 (r=0.143 nominal on a 700m axis, i.e. the *short*
  block first) isn't directly comparable to this sweep's low end at all,
  since this sweep never tested r<0.50 (short block first). A full
  treatment would sweep both r and total span, and cover r<0.50 for
  direction-symmetry.
- **idqn's checkpoints are n=3, not n=10** like SP9's regular/irregular
  comparison — reused SP5's constraint (no seeds beyond 42-44 exist without
  new training). The idqn curve is flat enough (1.3s spread over 8 points)
  that n=3 seems adequate to see the shape, but the two crossing points
  (r≈0.509, r≈0.797) are interpolated from single-point-pair means, not
  independently confirmed at a second n.
- **Mechanism is a plausible story, not verified.** Did not inspect
  green_wave's actual offset schedule or per-junction queue dynamics to
  confirm the "shared offset serves neither block" explanation; it's
  consistent with the shape of the curve but unverified against the
  controller's internals.
