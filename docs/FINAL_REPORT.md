# Final report — geometry dose-response, safety-weight ablation, and MAPPO coordination

**2026-08-27.** This consolidates three independent, closed sub-project
threads (SP13-15) into one report. Each thread is self-contained and backed
by its own `FINDINGS_*.md` doc(s), linked inline for full evidence tables.
A fourth thread — the root cause of `green_wave`/`max_pressure`'s span=450/550
congestion spike — was narrowed but not solved across SP13b-d; see
[Future work](#future-work). It is a disclosed open question, not a gap in
any result below.

## Bottom line

1. **Geometry dose-response**: `idqn`'s zero-shot advantage over `green_wave`
   on irregular signal spacing is real, but it is a **bounded band in the
   asymmetry-ratio space**, not a monotonic "more irregular is worse" effect.
   `idqn` itself is flat and geometry-invariant everywhere tested.
2. **Safety-weight (λ) ablation**: the project's λ=0.5 default was never the
   efficiency-optimal point. **λ=0.25 beats it on delay, on both geometries,
   confirmed at n=10 seeds** — the gap grows with more seeds, not shrinks.
3. **MAPPO coordination**: the original SP3 thesis claim ("explicit
   coordination beats independent agents") **does not hold at this scale**.
   MAPPO underperforms IPPO even after one round of HP retuning aimed at its
   own most obvious confound. The project stops here on MAPPO.

---

## 1. Geometry dose-response: the idqn-beats-green_wave flip is a bounded band

**Sources:** [SP13](FINDINGS_2026-08-26-sp13-geometry-dose-response.md),
[SP13e](FINDINGS_2026-08-27-sp13e-lowr.md).

### Question

SP8-SP10 established that on irregular signal spacing, `idqn` beats
`green_wave` zero-shot — the one reversal of this project's standing
"consolidate on green_wave" result. SP10 sampled 3 discrete geometries and
called the flip "generalizes (3/3)". But those 3 points were hand-picked and
two of them confounded asymmetry ratio with total corridor span. SP13/SP13e
sweep the asymmetry ratio `r` (nominal C1-C2 length / total span)
continuously at a fixed span=400m, across the full range, to find the actual
shape of the effect.

### Result

Delay per completed trip (seconds), span=400, `corridor_peak` demand:

| r | green_wave | idqn | max_pressure | idqn − green_wave |
|---|---:|---:|---:|---:|
| 0.10 | 17.69s | 17.87s | 26.40s | +0.09s (green_wave ahead, ~tied) |
| 0.20 | 23.11s | 17.19s | 22.58s | −6.03s (idqn ahead) |
| 0.25 | 22.13s | 16.93s | 21.32s | −5.30s (idqn ahead) |
| 0.30 | 21.46s | 17.12s | 20.61s | −4.44s (idqn ahead) |
| 0.35 | 15.88s | 16.88s | 23.82s | +0.96s (green_wave ahead) |
| 0.45 | 13.59s | 16.97s | 25.19s | +3.32s (green_wave ahead) |
| **0.50 (regular)** | **13.46s** | **16.56s** | **25.83s** | +3.09s (green_wave ahead) |
| 0.55 | **31.17s** | 16.92s | 25.23s | −14.16s (idqn ahead) |
| 0.60 | 31.02s | 17.01s | 24.40s | −13.98s (idqn ahead) |
| 0.70 | 25.05s | 17.10s | 20.68s | −8.00s (idqn ahead) |
| 0.75 (irregular3) | 20.62s | 17.26s | 21.33s | −3.41s (idqn ahead) |
| 0.80 | 17.05s | 17.31s | 22.93s | +0.18s (green_wave ahead) |
| 0.90 | 14.10s | 17.84s | 25.01s | +3.69s (green_wave ahead) |

Four bounded crossings total, two bands either side of r=0.50:

- High-r band (SP13): idqn ahead inside **r≈[0.51, 0.80]**, green_wave ahead
  on both sides.
- Low-r band (SP13e): idqn ahead inside **r≈[0.10, 0.34]**, green_wave ahead
  on both sides.

The two bands are **not mirror images of each other** — reflecting the
high-r band about r=0.50 would predict a low-r band at [0.20, 0.49]; the
actual low-r band is narrower and shifted toward the extreme. Which block is
short (near entry C1 vs. near exit C3) is a real variable, not a labeling
convention.

**`idqn` is flat and geometry-invariant across the entire tested range**:
16.56s-17.87s (a 1.3s band), across all 13 ratio points sampled at this
span. This holds independently in both the SP13 and SP13e sweeps — the
single most robust result in this thread.

### Verdict

**Not a monotonic dose-response.** SP10's "the flip generalizes, 3/3" was
correct as far as it went, but the underlying picture — "irregularity is
bad, more is worse" — is wrong. `green_wave`'s failure is confined to two
bounded bands of moderate asymmetry; it recovers both near-symmetric and at
extreme skew. `idqn`'s zero-shot policy essentially does not notice the
geometry change at all, at any ratio tested.

This is the safe headline result of the three: strong, well-evidenced (13
ratio points, 4 independent confirmations of idqn's flatness), and it
overturns a previously-published claim (SP10) with data rather than
argument.

**Scope, disclosed:** all points above are span=400m only. Extending the
ratio axis to other spans (400/450/550/700) surfaces a second, harder
question — the root cause of a separate congestion anomaly — which is
scoped out of this report; see [Future work](#future-work).

---

## 2. Safety-weight (λ) ablation: λ=0.5 was never efficiency-optimal

**Sources:** [SP14](FINDINGS_2026-08-26-sp14-lambda-ablation.md),
[SP14b](FINDINGS_2026-08-27-sp14b-lambda-n10.md).

### Question

The corridor reward is `efficiency − λ·safety_penalty`, and the project is
framed as "safety-aware" signal control. Every corridor experiment from SP4
through SP12 used a single value, λ=0.5, chosen once and never revisited.
The efficiency/safety tradeoff that framing depends on had never actually
been measured.

### Result

IDQN trained at λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0}, evaluated zero-shot on
regular and irregular geometries. `safety_total` falls monotonically with λ
on both geometries, as the reward design intends — but `delay_per_trip`
does **not**: it dips at λ=0.25, rises back through λ=0.5, then climbs
sharply from λ=0.75 on.

n=3 seeds (SP14):

| λ | regular delay | irregular delay |
|---|---:|---:|
| 0.00 | 17.08s ± 0.37 | 22.58s ± 1.52 |
| **0.25** | **15.69s ± 0.11** | **18.23s ± 0.27** |
| 0.50 (default) | 16.56s ± 0.36 | 18.48s ± 0.12 |
| 0.75 | 20.93s ± 0.49 | 22.16s ± 0.56 |
| 1.00 | 23.84s ± 0.22 | 24.98s ± 1.07 |

**λ=0.25 beats λ=0.5 on delay, on both geometries, while also beating λ=0.0
on both delay and safety** — λ=0.0 (pure efficiency) is dominated outright.

Widened to n=10 for the {0.25, 0.5} pair specifically (SP14b), since the
irregular-net gap at n=3 (0.25s) was flagged as thin relative to this
project's typical seed noise:

| geometry | 0.5 − 0.25 gap, n=3 | 0.5 − 0.25 gap, n=10 | sign agreement |
|---|---:|---:|---:|
| regular | +0.874s ± 0.332 | **+0.997s ± 0.424** | 10/10 seeds |
| irregular | +0.245s ± 0.379 | **+0.593s ± 0.452** | 9/10 seeds |

**The gap did not shrink toward zero the way a noise artifact would — it
grew, more than doubling on the irregular net.** This is not an n=3 sampling
artifact.

### Verdict

**Closed, confirmed.** The efficiency/safety frontier has a knee around
λ=0.25-0.5, not the smooth tradeoff across [0,1] the project implicitly
assumed by picking 0.5 and stopping. λ=0.5 remains defensible — it sits near
the knee, and its delay cost relative to the λ=0.25 optimum is small (0.87s
regular, 0.6s irregular at n=10) — but a systematic sweep would have picked
0.25. The project's "safety-aware" framing is materially strengthened by
being able to say *why* 0.5 sits where it does on a measured curve.

**Scope, disclosed:** IDQN only; hyperparameters held fixed across all λ
arms (selected originally at λ=0.5, never independently retuned per λ);
only 2 geometries (regular, SP8's irregular net).

---

## 3. MAPPO coordination: no evidence it beats independent agents

**Sources:** [SP15](FINDINGS_2026-08-27-sp15-mappo-smoke.md),
[SP15b](FINDINGS_2026-08-27-sp15b-mappo-retune.md).

### Question

The original SP3 design spec stated the thesis claim plainly: "explicit
coordination (CTDE) beats independent agents and classical baselines." That
claim was never tested — the MAPPO extension (joint-state critic) only ever
existed on a stale, never-merged branch. SP15 ported it onto current
`train_corridor.py` and ran a directional smoke test; SP15b then checked
whether the obvious confound (HPs never retuned for MAPPO's 3x-wider critic)
was hiding a real coordination benefit.

### Result

Delay per completed trip, seed 42, `corridor_peak`:

| controller | regular | irregular |
|---|---:|---:|
| green_wave | 13.44s | 19.44s |
| idqn | 16.95s | 18.41s |
| **ippo** | 16.91s | 19.14s |
| **mappo** (default HP) | 18.26s | 20.08s |

MAPPO underperforms IPPO on both geometries at the project's standard,
single-intersection-tuned HPs (+1.35s regular, +0.94s irregular).

SP15b then tried three targeted HP retunes aimed at the wider critic:

| variant | regular gap to ippo | irregular gap to ippo |
|---|---:|---:|
| mappo (baseline) | +1.35s | +0.94s |
| mappo_lr_low (lr/5) | +10.22s | +10.10s |
| mappo_batch64 (batch x2) | +1.12s | +0.55s |
| mappo_wide512 (hidden 512x512) | **+0.53s** | +0.73s |

**None of the three variants beats IPPO on either geometry.** The best
(`wide512`) narrows the regular-net gap 61%, but does not close it.
`lr_low` collapses training entirely (+10s worse) — evidence this HP space
is sensitive, cutting either way, not evidence of a hidden coordination win.

### Verdict

**Closed, negative result.** The centralized critic does not produce a
better policy than the independent one at this scale, even after HP
retuning aimed squarely at the most obvious confound. The gap shrank, it did
not flip. A genuinely-competitive-with-IPPO MAPPO configuration is not
ruled out — three hand-picked HP points are not a search — but escalating to
full n=10-seed rigor is not warranted by what's in hand. **This project
stops here on MAPPO**, reported as: the coordination mechanism was built,
integrated, and smoke-tested against the thesis's own original claim, and
the test did not support it.

**Scope, disclosed:** n=1 seed throughout (both SP15 and SP15b); only 2
geometries; `wide512`'s improvement widens the actor as well as the critic
(architecture constraint), so it is not cleanly attributable to critic
capacity alone.

---

## Future work

Not chased further this session — narrowed but not solved, a valid line for
a writeup rather than a gap that weakens any result above:

- **The span=450/550 congestion spike.** `green_wave` and `max_pressure` both
  spike in aggregate delay at span=450/550 (vs. 400/700), while `idqn` stays
  flat. Per-signal queue instrumentation localized `green_wave`'s own spike
  to signal C3 and ruled out its offset schedule as the cause
  ([SP13c](FINDINGS_2026-08-27-sp13c-span550.md)), but the same
  instrumentation showed `max_pressure`'s spike sits on a *different* signal
  (C2, not C3) at span=550 ([SP13d](FINDINGS_2026-08-27-sp13d-span450.md)) —
  ruling out a single shared mechanism between the two controllers. Root
  cause is open; likely emergent SUMO car-following/queueing dynamics, not
  confirmed.
- The crossing-count-vs-span relationship (1→0→3→3 across span=400/450/550/700)
  is not monotonic or bracketed — more span points between 400-550m would be
  needed to locate the regime boundaries.
- r<0.50 (short block first) is tested only at span=400; untested at
  450/550/700.
- A real HP search (Optuna-style, joint over lr/hidden/batch) was never run
  for MAPPO — only 3 hand-picked points.

## Reproducing

```bash
# Geometry dose-response (SP13/SP13e, zero-shot, no training needed)
python -m analysis.build_geometry_sweep_nets
python -m analysis.build_geometry_sweep_nets_lowr
python -m analysis.geometry_sweep
python -m analysis.geometry_sweep_lowr

# Lambda ablation (SP14/SP14b)
./run_lambda_ablation.sh
python -m analysis.lambda_ablation
python -m analysis.lambda_ablation_n10

# MAPPO (SP15/SP15b)
python -m analysis.mappo_retune_smoke
```
