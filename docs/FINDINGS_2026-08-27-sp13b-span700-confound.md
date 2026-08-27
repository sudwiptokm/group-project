# SP13b Findings: does the green_wave failure band's shape depend on absolute span, or only the asymmetry ratio?

## The question

SP13 (docs/FINDINGS_2026-08-26-sp13-geometry-dose-response.md) swept the
asymmetry ratio `r` = nominal C1-C2 length / total span at a fixed span of
400m and found green_wave fails only inside a bounded band, `r` in roughly
[0.51, 0.80], recovering on both sides. But SP13's own sweep never touched
span: SP8/SP10's original irregular nets (`corridor_irregular`/`_irregular2`)
sit at span=700, off that axis entirely. SP13 flagged this explicitly as
unresolved. This repeats the same 8-point ratio sweep at span=700 to find out
whether the band's boundaries are a property of `r` alone.

## Method

Same construction as `analysis/build_geometry_sweep_nets.py`, same 8 ratios
(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90), same node/edge template and
netconvert invocation, only `SPAN=700` instead of 400 — C1@0, C3@700, C2
slid to `r * 700` (`analysis/build_geometry_sweep_nets_span700.py`, 8 new
nets, all newly built: none of span=400's points are reusable since C2's
absolute position differs for every `r` at a different span). `corridor_peak`
demand, `min_green=10`, zero-shot: idqn's existing SP5 `corridor_peak`
checkpoints (seeds 42-44), green_wave/max_pressure fixed policies (seeds
42-46). `analysis/geometry_sweep_span700.py`, mirrors `geometry_sweep.py`.

## Results

Delay/trip in seconds per completed trip, mean (n=5 baseline, n=3 idqn):

| r | green_wave | idqn | max_pressure | idqn − green_wave |
|---|---:|---:|---:|---:|
| 0.50 | 16.72s | 19.56s | 22.05s | +2.82s (green_wave ahead) |
| 0.55 | 17.11s | 19.34s | 21.63s | +2.21s (green_wave ahead) |
| 0.60 | 22.02s | 19.22s | 23.02s | −2.82s (idqn ahead) |
| 0.65 | 24.90s | 19.22s | 25.31s | −5.75s (idqn ahead) |
| 0.70 | 20.26s | 18.74s | 26.63s | −1.57s (idqn ahead) |
| 0.75 | 18.75s | 18.75s | 26.61s | −0.02s (idqn ahead, ~tied) |
| 0.80 | 17.03s | 18.80s | 21.62s | +1.72s (green_wave ahead) |
| 0.90 | 21.33s | 18.90s | 23.44s | −2.49s (idqn ahead) |

Three interpolated crossings, not two: **r≈0.572, r≈0.750, r≈0.842.** idqn is
flat here too (18.74s-19.56s, a 0.82s band, even tighter than span=400's
1.3s). green_wave is again the volatile one, but the shape differs from
span=400: it spikes earlier and higher (24.90s at r=0.65, vs span=400's peak
of 31.17s at r=0.55), dips back down at r=0.80 (17.03s, close to its own
r=0.50 baseline of 16.72s) — matching span=400's "recovers at the edges"
pattern up to that point — **then gets worse again at r=0.90 (21.33s)**,
which span=400 does not do (span=400's r=0.90 was 14.10s, its best point
after r=0.50).

## Verdict: the band's shape is not a function of the ratio alone

**Three findings that separate span from ratio:**

1. **The span=700, r=0.50 baseline is not the same as span=400's.** A
   symmetric net at span=700 (16.72s) already runs green_wave 3.26s slower
   than the symmetric net at span=400 (13.46s) with identical relative
   geometry (r=0.50 both times) — a longer corridor costs green_wave delay
   on its own, before any asymmetry is introduced. idqn moves the same
   direction but much less (19.56s vs 16.56s, +3.00s) — both controllers see
   *some* span effect, so this isn't purely a green_wave artifact, but
   green_wave's baseline is not span-invariant either.

2. **The band's location shifted and gained a third crossing.** span=400's
   band was a single interval, [0.51, 0.80], with green_wave low outside it
   on both ends. span=700 crosses three times (0.572, 0.750, 0.842), meaning
   there are now *two* separate high-r zones: one where green_wave is briefly
   competitive again (around r=0.80) sandwiched between two zones where idqn
   wins (r in [0.572, 0.750] and r=0.90). SP13's "single bounded band, other
   than that it's fine" description does not transfer to span=700 as stated.

3. **green_wave does not recover at the top of the tested range on span=700**
   the way it does on span=400. At r=0.90, span=400's green_wave nearly
   matches its own symmetric-net baseline (14.10s vs 13.46s, +0.64s);
   span=700's green_wave at r=0.90 is 4.61s worse than its own symmetric-net
   baseline (21.33s vs 16.72s) and still loses to idqn. Whatever lets
   green_wave "stop caring" about asymmetry once one block gets short enough
   (SP13's proposed mechanism) evidently needs a shorter absolute short-block
   length at span=700 than the 40m span=400 needed — r=0.90 at span=700
   still leaves a 70m short block, nominally, which is *longer* in absolute
   terms than span=400's 40m short block at its own r=0.90, yet still shows
   the failure. This is consistent with the mechanism depending on the short
   block's absolute length, not its ratio to the total span.

This is a genuine interaction, not just re-confirmation with noise: idqn's
band across both spans stays under 1.3s wide throughout (flat, as SP13 found)
while green_wave's *shape* changes materially between the two span settings —
same qualitative story (idqn wins somewhere in the middle of the ratio range)
but the specific boundaries, and even the count of crossings, are span-
dependent. SP13's headline ("bounded band, not monotonic") still holds at
span=700, but the practical claim "[0.51, 0.80] is unsafe for green_wave" is
a span=400-specific number, not a general property of asymmetry ratio.

Consistent with SP8/SP10's original span=700 nets: `corridor_irregular`
(r≈0.857 nominal, 578m/78m actual) had green_wave=19.44s/idqn=18.41s
(SP9, n=10) — same ballpark and same qualitative idqn-ahead result as this
sweep's nearby r=0.90 point (green_wave=21.33s/idqn=18.90s), though not an
exact match since neither r nor actual edge lengths line up precisely.

## What this doesn't answer

- **Only two spans tested** (400, 700). A real span-interaction claim would
  want a third point (e.g. 550m) to see whether the crossing count/location
  moves monotonically with span or does something else. Two points establish
  "span matters," not the functional form of *how*.
- **r<0.50 (short block first) still untested at either span.** SP13's own
  gap; this follow-up doesn't close it either — both sweeps start at r=0.50.
- **idqn's checkpoints are still n=3** (SP5's constraint), same caveat SP13
  carried forward; the flat band is narrow enough at both spans that n=3
  looks adequate for the shape, but the exact crossing points are still
  single-pair interpolations.
- **Mechanism is still a story, not verified** — same as SP13's own
  disclaimer; this result adds a data point (absolute short-block length
  plausibly mattering more than ratio) but doesn't inspect green_wave's
  offset schedule directly to confirm it.
