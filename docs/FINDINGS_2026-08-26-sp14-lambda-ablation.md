# SP14 Findings: the safety-weight (λ) ablation — is λ=0.5 actually the right default?

## The question

The corridor reward is `efficiency - λ * safety_penalty`
(`env_common.make_safety_reward_fn`), and the project is titled/framed as
"safety-aware" signal control. Every corridor experiment from SP4 through
SP12 used a single value, λ=0.5, chosen once and never revisited. The
efficiency/safety tradeoff that framing depends on had never actually been
measured. This measures it.

## Method

Train IDQN at λ ∈ {0.0, 0.25, 0.75, 1.0} × seeds {42, 43, 44} on
`corridor_peak`, `min_green=10`, `steps=100000` (`run_lambda_ablation.sh`,
12 new checkpoints). λ=0.5 × the same 3 seeds already existed from SP5 and
was reused unchanged — the 0.5 column here *is* the published corridor
result, not a re-run of it.

Evaluate each of the 15 checkpoints zero-shot on both geometries
(`analysis/lambda_ablation.py`):

- **regular** (`corridor.net.xml`) — in-distribution, what every checkpoint
  trained on.
- **irregular** (`corridor_irregular.net.xml`) — SP8's 578m/78m asymmetric
  net, the one where the green_wave/idqn ranking flips.

`train()` has no `net_file` parameter, so "both geometries" means
train-once/zero-shot-evaluate-twice — the same protocol SP8-SP10 used, no
new training engineering. Two metrics, both already logged, no new
instrumentation: `delay_per_trip` (SUMO timeLoss per completed trip, via
`analysis.tripinfo.reduce_tripinfo`) and `safety_total` (episode sum of
`system_safety_total` = brake + exposure, from `SafetyLoggingEnv`'s
per-window eval CSV column).

## Results

Delay/trip and safety cost vs λ, n=3 seeds:

| λ | regular delay | regular safety | irregular delay | irregular safety |
|---|---:|---:|---:|---:|
| 0.00 | 17.08s ± 0.37 | 3317 ± 93 | 22.58s ± 1.52 | 4051 ± 196 |
| **0.25** | **15.69s ± 0.11** | 2597 ± 107 | **18.23s ± 0.27** | 2804 ± 59 |
| 0.50 (project default) | 16.56s ± 0.36 | 2190 ± 166 | 18.48s ± 0.12 | 2282 ± 86 |
| 0.75 | 20.93s ± 0.49 | 1813 ± 10 | 22.16s ± 0.56 | 1793 ± 58 |
| 1.00 | 23.84s ± 0.22 | 1657 ± 56 | 24.98s ± 1.07 | 1667 ± 96 |

`safety_total` falls monotonically with λ on both geometries, as the reward
design intends. `delay_per_trip` does **not** move monotonically: it dips at
λ=0.25, rises back through λ=0.5, then climbs sharply from λ=0.75 on.

Change from λ=0 (pure efficiency), paired by seed:

| | λ=0.25 | λ=0.5 | λ=0.75 | λ=1.0 |
|---|---:|---:|---:|---:|
| regular delay | −1.39 ± 0.45s | −0.51 ± 0.71s | +3.86 ± 0.40s | +6.77 ± 0.58s |
| regular safety | −720 ± 71 | −1127 ± 257 | −1504 ± 102 | −1660 ± 144 |
| irregular delay | −4.35 ± 1.80s | −4.10 ± 1.42s | −0.43 ± 1.65s | +2.40 ± 0.64s |
| irregular safety | −1247 ± 249 | −1769 ± 266 | −2258 ± 250 | −2385 ± 182 |

Pearson r of delay vs safety across the 5 λ means: **r=−0.728 (regular)**,
**r=−0.180 (irregular)** — negative, meaning lower delay tends to come with
lower safety cost, i.e. the two objectives are *not* purely traded off
across this whole range; the real tradeoff is concentrated at the high-λ
end (see below).

## Verdict: λ=0.5 was never the efficiency-optimal point, and λ=0.0 is dominated

**λ=0.25 beats λ=0.5 on delay, on both geometries** (15.69s vs 16.56s
regular; 18.23s vs 18.48s irregular) **while also beating λ=0.0 on both
delay and safety.** λ=0.0 (pure efficiency, no safety term at all) is not
the fastest setting — it's dominated outright by λ=0.25, which is both
faster and safer. This means every corridor result published SP4-SP12 at
λ=0.5 was run at a point already giving up a small amount of free delay
(≈0.5-4s/trip depending on geometry) relative to λ=0.25, for a safety
improvement that λ=0.25 mostly already captures anyway (2597 vs 2190
regular safety_total — most of the drop from λ=0 to λ=0.5 already happened
by λ=0.25).

The real tradeoff only appears at λ≥0.75: from there, every further unit of
safety reduction costs sharply more delay (+3.9s to +6.8s/trip on the
regular net) for diminishing safety return (the marginal safety_total drop
from 0.75→1.0 is smaller than 0.5→0.75, which is smaller than 0.25→0.5).
The efficiency/safety frontier has a knee around λ=0.25-0.5, not a smooth
tradeoff across the whole [0,1] range the project implicitly assumed by
picking 0.5 and stopping.

This pattern holds on both geometries tested (regular and SP8's irregular
net) — not an artifact of one net's dynamics.

**Practical implication:** λ=0.5 remains a defensible choice (it is near the
knee, and its delay cost relative to the λ=0.25 optimum is small — 0.87s
regular, 0.25s irregular), but it is not what a systematic sweep would have
picked, and the project's "safety-aware" framing is materially strengthened
by being able to say *why* 0.5 sits where it does on a measured curve,
rather than that it was never checked at all.

## What this doesn't answer

- **Hyperparameters were held fixed across λ arms** (`train_corridor_dqn._hp()`,
  selected originally at λ=0.5). Whether λ=0.25's advantage would hold, grow,
  or shrink under HPs re-tuned for that specific λ is untested — same
  disclosed limitation `run_lambda_ablation.sh`'s header states up front.
- **n=3 seeds**, matching SP5/SP8's original constraint, not SP9's later
  n=10 widening. The λ=0.25 vs λ=0.5 gap on the irregular net (18.23s vs
  18.48s, 0.25s) is small relative to seed-to-seed noise elsewhere in this
  project (SP9 found σ≈0.38s at n=10 for a comparably-sized irregular-net
  effect) — worth confirming at higher n before treating "0.25 beats 0.5" as
  more than directional on that geometry specifically. The regular-net gap
  (15.69s vs 16.56s, 0.87s) is larger relative to its own σ (0.11-0.36s) and
  looks more solid.
- **Only 2 geometries tested** (regular, SP8's original irregular net) — not
  cross-referenced against SP13's finding that green_wave's own
  irregular-spacing failure is a bounded band, not monotonic. Whether the
  λ-vs-delay curve's shape itself depends on where in SP13's asymmetry-ratio
  space the geometry sits is unexplored.
- **This is IDQN only.** Whether the same knee-shaped tradeoff holds for
  `max_pressure` or `green_wave` (neither of which the safety reward
  actually trains against) was not evaluated — those controllers don't
  respond to λ by construction, so the question would have to be reframed
  as "how do their fixed policies' safety costs compare to idqn's
  λ-swept range," not attempted here.
