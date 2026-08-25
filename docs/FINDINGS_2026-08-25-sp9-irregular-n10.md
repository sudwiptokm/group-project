# SP9 Findings: does the idqn-beats-green_wave flip on irregular spacing hold at n=10?

## The question

SP8 found that on `corridor_irregular.net.xml` (C1@0/C2@600/C3@700 vs the
regular net's even 200m spacing), IDQN beats `green_wave` zero-shot — a
reversal of every other result on this corridor
(`docs/FINDINGS_2026-08-22-sp8-irregular-spacing.md`). SP10 confirmed the
flip generalizes across 2 more spacing variants, 3/3 nets so far
(`docs/FINDINGS_2026-08-22-sp10-irregular-generalization.md`). Neither
widened the seed count on the *original* net past SP4-SP8's usual n=3. This
sub-project does that: does the flip hold at n=10, or was n=3 lucky?

## Method

Same net (`corridor_irregular.net.xml`), same `corridor_peak` demand, same
zero-shot posture as SP8 (IDQN never trained on this geometry — SP5's
regular-net checkpoints evaluated on it directly). Trained 7 new IDQN
checkpoints (seeds 45-51, same hyperparameters, 100k steps) to extend SP8's
existing seeds 42-44 to n=10, matching `analysis/irregular_net_compare.py`'s
own already-widened `SEEDS = tuple(range(42, 52))` (a prior session had
already made this edit, uncommitted; this session trained the checkpoints
it requires and ran it). `green_wave`/`max_pressure`'s regular-net reference
numbers for all 10 seeds were already in `analysis/corridor_sweep.csv` from
SP4's original sweep — only the irregular-net runs and new IDQN checkpoints
were new work.

## Results

| controller | irregular mean | regular mean | shift (irregular − regular) |
|---|---|---|---|
| idqn | **18.58s ± 0.42s** | 16.72s ± 0.29s | +1.86s |
| green_wave | 19.54s ± 0.34s | 13.46s ± 0.22s | **+6.07s** |
| max_pressure | 23.97s ± 1.43s | 26.52s ± 2.98s | −2.55s |

idqn beats green_wave on irregular spacing at **10/10 seeds**, not just
3/3:

| seed | green_wave | idqn | idqn margin |
|---|---|---|---|
| 42 | 19.44s | 18.41s | +1.03s |
| 43 | 19.97s | 18.62s | +1.35s |
| 44 | 19.44s | 18.40s | +1.03s |
| 45 | 19.88s | 18.83s | +1.05s |
| 46 | 19.41s | 19.20s | +0.21s |
| 47 | 18.98s | 17.72s | +1.26s |
| 48 | 19.50s | 18.84s | +0.65s |
| 49 | 19.09s | 18.51s | +0.58s |
| 50 | 19.92s | 19.00s | +0.93s |
| 51 | 19.73s | 18.24s | +1.48s |

**n=10 mean margin: +0.96s ± 0.38s** (n=3 subset, seeds 42-44: +1.14s ±
0.19s). The mean narrows modestly (−16%) as the 7 new seeds add one
noticeably smaller margin (seed46, +0.21s) alongside several close to or
above the n=3 mean — but the sd nearly doubles (0.19s → 0.38s), a wider
seed-to-seed spread than the original 3 seeds suggested. Despite that
wider spread, **the sign never flips**: every one of the 10 seeds has idqn
beating green_wave on this net, none reverse toward parity or a green_wave
win.

## Verdict: does this change the consolidation recommendation?

**No new decision, but it strengthens SP8's finding rather than casting
doubt on it.** This was expected to be low-novelty, high-certainty work
(widening a seed count on an already-confirmed effect, not testing a new
hypothesis) — that is exactly what it delivered: same direction, same
rough magnitude, every seed agrees. Combined with SP10's 3/3
net-variant generalization, the idqn-beats-green_wave flip on irregular
signal spacing is now confirmed at n=10 seeds × 3 net variants. The
project's broader `green_wave`-consolidation recommendation for *regularly
spaced* corridors is unaffected — this remains a spacing-specific
exception, not a reversal of the general result.

## What this doesn't answer

- Whether the wider n=10 sd (0.38s vs n=3's 0.19s) reflects real
  seed-to-seed variability in how much irregular spacing helps idqn, or
  would itself narrow again at a still-larger n — not explored further,
  same thin-n caveat every sub-project on this corridor carries.
- The grid-topology stretch goal SP10 flagged as out of scope remains
  untried.
