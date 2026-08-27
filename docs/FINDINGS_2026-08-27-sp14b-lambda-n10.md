# SP14b Findings: does the lambda=0.25-vs-0.5 delay gap survive n=3 -> n=10?

## The question

SP14 (docs/FINDINGS_2026-08-26-sp14-lambda-ablation.md) found lambda=0.25
beats the project default lambda=0.5 on delay, on both geometries, at n=3
seeds — but flagged the irregular-net gap (18.23s vs 18.48s, 0.25s) as small
relative to this project's typical seed-to-seed noise (SP9 found sigma~0.38s
at n=10 for a comparably-sized effect), and asked for the same n=3->n=10
widening SP9 did for the geometry flip. This trains the 7 missing seeds
(45-51) for lambda in {0.25, 0.5} only — not all 5 lambdas, since the other 3
arms (0.0, 0.75, 1.0) aren't the comparison in question — and re-evaluates
zero-shot on both geometries, same protocol as SP14
(`analysis/lambda_ablation_n10.py`, reusing `analysis.lambda_ablation`'s own
`run_one`/`NETS`/`SCENARIO` byte-for-byte).

## Results

Delay/trip in seconds per completed trip:

| geometry | metric | n=3 | n=10 |
|---|---|---:|---:|
| regular | lambda=0.25 mean | 15.688s ± 0.110 | 15.723s ± 0.239 |
| regular | lambda=0.50 mean | 16.562s ± 0.359 | 16.720s ± 0.366 |
| regular | 0.5 − 0.25, paired by seed | +0.874s ± 0.332 | **+0.997s ± 0.424** |
| irregular | lambda=0.25 mean | 18.232s ± 0.274 | 17.928s ± 0.315 |
| irregular | lambda=0.50 mean | 18.478s ± 0.119 | 18.521s ± 0.331 |
| irregular | 0.5 − 0.25, paired by seed | +0.245s ± 0.379 | **+0.593s ± 0.452** |

Sign (lambda=0.25 faster) agrees with the original n=3 direction on **10/10**
seeds on the regular net, **9/10** on the irregular net.

## Verdict: the gap survives — and grows, it does not shrink

The exact worry SP14 raised was that the irregular-net gap (0.25s at n=3) was
small enough relative to seed noise elsewhere in this project to be
directional rather than solid. At n=10 that gap did not shrink toward zero
the way a noise-driven artifact would be expected to; **it grew to 0.593s ±
0.452**, more than double the n=3 point estimate, and 9 of the 10 seeds
individually favor lambda=0.25. The regular-net gap, already the more solid
of the two at n=3, also grew slightly (+0.874s -> +0.997s) and held 10/10.

This confirms SP14's headline: **lambda=0.25 beats the project's lambda=0.5
default on delay, on both geometries, and this is not an n=3 sampling
artifact.** The one non-agreeing seed on the irregular net (1/10) is
consistent with ordinary seed-to-seed variance layered on top of a real
effect, not evidence against the effect itself — SP9's own n=10 widening of
the idqn-beats-green_wave flip saw a comparable single "weak" seed
(seed46, +0.21s margin) without that undermining the 10/10 sign-agreement
there.

## What this doesn't answer

- **Still IDQN only, still `corridor_peak`, still `min_green=10`** — same
  scope SP14 had. Whether the lambda=0.25 advantage holds under other
  scenarios/floors is untested.
- **HPs still held fixed across lambda arms** (SP14's own disclosed
  limitation, unchanged) — the lambda=0.25 optimum found here is relative to
  HPs selected at lambda=0.5, not independently retuned per lambda.
- **Only lambda in {0.25, 0.5} widened to n=10.** The 0.0/0.75/1.0 arms
  remain at n=3 (SP14's original constraint) — this was a deliberate scope
  cut (the flagged risk was specifically the 0.25-vs-0.5 gap, not the wider
  knee shape), not an oversight, but it means the full 5-point curve's
  n=10 robustness elsewhere on the lambda axis is still unconfirmed.
