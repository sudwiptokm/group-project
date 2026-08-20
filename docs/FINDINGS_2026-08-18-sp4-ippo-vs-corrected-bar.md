# IPPO vs the corrected corridor bar

Written 2026-08-18 through 2026-08-20 (runs spanned that window). Executes
path 1 of `docs/HANDOFF_2026-08-18.md`: train IPPO against the corrected
green_wave/max_pressure calibration and report the result, negative or not.
Also folds in a second question raised mid-run — whether uneven cross-street
demand (`corridor_skew`) gives a fixed plan anything to lose to — and a
budget-sensitivity check once the headline result came back lopsided.

All numbers below are delay per completed trip (seconds) from SUMO tripinfo,
never `system_mean_waiting_time` — see `docs/FINDINGS_2026-08-12.md` §1 for
why. `analysis/ippo_sweep.csv`, `analysis/ippo_sweep_peak_100k_check.csv`, and
`analysis/corridor_sweep.csv` hold the raw per-seed rows behind every table
here.

## Throughput and budget

Measured 22.2 agent-steps/s on this machine (an Apple M1 Max, 10 cores, 32GB)
training the multi-agent corridor env — see Task 6 of
`docs/superpowers/plans/2026-08-18-sp4-ippo-corridor-training.md`. The main
sweep used a 16,000-step budget, sized to fit 20 training runs (2 scenarios
x 10 seeds) in a few hours of wall-clock. That budget turned out to matter a
great deal — see "Budget sensitivity" below.

## ppo_core vs SB3 PPO (single intersection, matched HPs, defect 3)

Two runs, same matched hyperparameters (`algos.ALGOS["ppo"]["defaults"]()`),
different step budgets:

| steps | ppo_core delay/trip | SB3 PPO delay/trip | diff |
|---|---:|---:|---:|
| 20,000 | 28.39s | 28.51s | -0.12s |
| 100,000 (full budget) | 28.4s | 28.5s | +0.1s |

Near-parity at both scales. The hand-rolled PPO core is not the explanation
for anything that follows — this was the top technical risk flagged in
`docs/HANDOFF_2026-08-18.md`, and it's closed with real evidence at the
originally-planned budget, not just the reduced-scale first pass.

## IPPO vs green_wave, paired, min_green=10, seeds 42-51, 16,000 steps

| scenario | ippo (mean ± sd) | green_wave (mean ± sd) | paired ippo - gw | wins |
|---|---:|---:|---:|---:|
| corridor_peak  | 35.92s ± 2.74 | 13.46s ± 0.22 | +22.46s ± 2.69 | 0/10 |
| corridor_tidal | 35.60s ± 3.06 | 13.96s ± 0.34 | +21.64s ± 3.15 | 0/10 |

IPPO loses on every seed, on both scenarios, by a wide margin — worse even
than `max_pressure` (19.25-19.43s at its own best floor), the reactive
controller it was also supposed to beat.

## corridor_skew: does uneven cross-street demand help either controller?

Added mid-run as a second test of the corridor's adaptation hypothesis (see
commit `e0e3656`): arterial demand identical to `corridor_peak` (2100 veh/h,
symmetric, constant), cross-street demand held at the same 900 veh/h total
but redistributed unevenly across nodes (C1=150, C2=600, C3=150 veh/h) to
test whether a fixed plan's single global through/cross split — it can vary
its *offset* per node but never its *split* — costs it against reactive or
learned control that can allocate green locally.

| scenario | green_wave (own best) | max_pressure (own best) | gap |
|---|---:|---:|---:|
| corridor_peak  | 13.46s (mg10) | 19.43s (mg15) | +5.97s |
| corridor_tidal | 13.96s (mg10) | 20.19s (mg15) | +6.23s |
| corridor_skew  | 13.45s (mg10) | 19.25s (mg15) | +5.81s |

Indistinguishable from peak/tidal; green_wave wins 10/10 again. The
mechanism (uniform split can't match uneven per-node demand) is real, but
600 veh/h at C2 never approaches the ~1800 veh/h capacity a roughly-even
split leaves each movement, so the "wrong" split never actually queues
anything. Unlike `corridor_tidal`, which explicitly sized its dominant
direction (1400 veh/h) against the ~1125 veh/h an even split can discharge
before picking a number, `corridor_skew`'s magnitude wasn't checked against
cross-street capacity before being chosen. IPPO was never trained on this
scenario — the reference calibration alone was enough to show it doesn't
change the underlying story, so training on it would have spent compute
without new information.

## Budget sensitivity: is 16,000 steps enough for a 3-agent problem?

The 16,000-step budget was sized for wall-clock convenience (Task 6), not
for what 3 agents pooling into one shared buffer need to converge — and it's
6.25x smaller than the 100,000-step budget the single-intersection
validation above used. Before treating +22s as the final word, `corridor_peak`
was retrained at 100,000 steps on 3 of the 10 seeds (42, 43, 44) as a cheap
confirmatory check (`analysis/ippo_sweep_peak_100k_check.csv`):

| seed | ippo (16k) | ippo (100k) | green_wave | gap (100k) |
|---|---:|---:|---:|---:|
| 42 | 35.25s | 16.91s | 13.44s | +3.47s |
| 43 | 31.84s | 18.18s | 13.52s | +4.66s |
| 44 | 35.99s | 18.46s | 13.45s | +5.01s |
| **mean** | **34.36s** | **17.85s** | **13.47s** | **+4.38s** |

Budget was clearly a major factor: 6.25x more steps roughly halved IPPO's
raw delay and cut the gap to green_wave by 80% (+22.46s -> +4.38s on the
matching seeds). But it did not close it. All 3 confirmatory seeds still
lose to green_wave. Given the shape of that improvement (large gain from
16k -> 100k, but landing well short of parity, not asymptoting toward it),
further budget increases were judged unlikely to flip the result and were
not pursued — the remaining 7 seeds at 100k were not run; see "What this
doesn't answer" below.

## Verdict

**IPPO does not clear the green_wave bar on this corridor, at either budget
tested, on either scenario tried.** At the properly-sized 16,000-step
budget the loss is enormous (+22s, worse than the reactive baseline too); at
a 6.25x larger, validated budget (100,000 steps, matching what the
single-intersection sanity check used) the loss shrinks by 80% but remains
real and consistent (+4.4s, 3/3 seeds) on the one scenario re-checked.

This is the negative result path 1 of `docs/HANDOFF_2026-08-18.md`
anticipated as a legitimate possible outcome, and the evidence behind it is
stronger than a single-budget run would have provided: the reference
controllers are correctly calibrated (two real bugs already fixed before
this run), the hand-rolled PPO implementation is validated against a mature
library at matched hyperparameters and budget, a second demand structure
(`corridor_skew`) was checked and didn't change the picture, and the budget
sensitivity was tested rather than assumed. **"A correctly-timed fixed plan
beats both max-pressure and IPPO on this coordinated 3-signal corridor" is
the finding, and it stands up to more scrutiny than the corridor-adaptation
hypothesis it was built to test.**

## What this doesn't answer

- The 100k-step confirmatory check only covered 3 of 10 seeds and only
  `corridor_peak`, not `corridor_tidal`. A full 10-seed, both-scenario run
  at 100k (or higher) steps would firm up the exact magnitude of the
  remaining gap, though it's unlikely to reverse the direction given the
  trend across 16k -> 100k.
- IPPO's hyperparameters are the reused single-intersection tuning (SP2's
  disclosed limitation), never re-tuned for the corridor's 3-agent, pooled-
  buffer setting. A corridor-specific hyperparameter search is untested and
  could plausibly move the remaining ~4.4s gap, though there's no evidence
  either way yet.
- `corridor_skew`'s specific design (600 veh/h at one node) undersaturates
  the shared-split ceiling it was meant to test. A redesigned skew scenario
  that pushes a node's cross demand near or above its phase-split-
  constrained capacity — the same saturation-math discipline `corridor_tidal`
  used — might still find room for adaptive control to matter; this one
  didn't test that harder version.
- Per `docs/HANDOFF_2026-08-18.md`'s path 2, a mid-episode incident/blockage
  scenario was proposed but never designed or built.

## Recommendation

Per the handoff's open decision: this result, plus the single-intersection
result it parallels (`docs/FINDINGS_2026-08-12.md`), makes a coherent,
twice-replicated finding — "a competently-timed fixed plan beats both
reactive and learned control" — at two different network scales. Consolidating
the thesis around that finding (handoff's option 3) is the best-supported
path forward. If the corridor extension continues anyway, it should pursue a
demand scenario actually checked against capacity math before training
(handoff's option 2, done properly this time) rather than more budget on the
scenarios already tested here — the budget-sensitivity check above suggests
diminishing returns, not a path to parity.
