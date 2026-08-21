# IDQN vs the corrected corridor bar — pilot result

Written 2026-08-21. Executes SP5
(docs/superpowers/specs/2026-08-21-sp5-idqn-corridor-design.md): test whether
a true-independent DQN (matching RESCO's IDQN, not this project's
parameter-shared IPPO) closes the gap to green_wave that IPPO could not.

## Config note

The single-intersection DQN pilot's tuned-for-100k config (params/*.json) is
unrecoverable (gitignored, cloud-only, never committed). This run uses the 3
disclosed values (lr=2.3195e-05, learning_starts=5000,
target_update_interval=5000) plus algos.ALGOS['dqn']['defaults']() for
everything else — a best-effort reconstruction, not the original tuned
config.

## dqn_core vs SB3 DQN (single intersection, matched HPs, correctness gate)

| | delay/trip (s) | wall-clock |
|---|---:|---:|
| dqn_core | 28.4s | 3052s |
| SB3 DQN  | 28.5s | 2877s |

Near-parity: -0.1s, essentially identical to the ppo_core-vs-SB3 result SP4
found at the same matched-hyperparameter setup. The top technical risk — the
hand-rolled DQN math being subtly wrong — did not materialize, so the
corridor numbers below rest on a validated core. As with
analysis/validate_ppo_core.py's identical pattern, the two arms here are
evaluated on different demand seeds (`seed` for SB3, `seed+1000` for
dqn_core) — inherited, not new to this run, and worth a hedging clause since
green_wave's own seed-to-seed spread at this floor is roughly 0.7s. (Run
output recorded in
`.superpowers/sdd/2026-08-21-sp5-idqn-corridor-training/progress.md`, since
`logs/` is gitignored and not re-derivable from the checkout.)

## Throughput

Measured 5.6 agent-steps/s on this machine (Task 6) — notably slower than
the single-intersection dqn_core throughput observed in Task 2 (~22-34
steps/s). That comparison isn't representative of steady-state cost, though:
the Task 6 probe sampled only 2000 steps, still inside the epsilon-random
warmup phase, before `learning_starts=5000` ever let a single gradient step
happen — so it measured startup cost, not the actual per-step overhead of
training 3 independent networks/buffers/optimizers.

The real comparison is the matched-100k-budget pilot itself: idqn's 3 seeds
took 12915.90s of total wall-clock (~3.588h) against ippo(100k)'s 12534.40s
(~3.482h) for the same corridor, same step budget — a ratio of 1.030. IDQN
ran about 3% slower than IPPO per env-step at the same budget, not 3x and
not the 4-6x an earlier (also-probe-based) estimate suggested. Per-seed
throughput is close between the two: idqn ≈ 23.7/22.5/23.5 steps/s
(seeds 42/43/44, avg ~23.2), ippo(100k) ≈ 24.3/23.6/24.0 steps/s (avg
~23.9). True independence — 3 separate networks, buffers, and optimizers
instead of one shared policy — is nearly free here, because SUMO's own
per-step simulation cost dominates wall-clock, not the extra
network/buffer/optimizer bookkeeping. The 3-seed pilot took ~3.6h
wall-clock — faster than the ~14.9h the Task 6 throughput probe
extrapolated, because the 2000-step sample it was based on wasn't
representative of steady-state per-seed cost.

## IDQN vs green_wave and IPPO, paired, corridor_peak, min_green=10, seeds 42-44 (pilot only)

| vs | idqn (mean +/- sd) | bar (mean +/- sd) | paired idqn - bar | wins |
|---|---:|---:|---:|---:|
| green_wave | 16.56 +/- 0.36 s | 13.47 +/- 0.04 s | +3.09 +/- 0.37 s | 0/3 |
| ippo (100k, same 3 seeds) | 16.56 +/- 0.36 s | 17.85 +/- 0.82 s | -1.29 +/- 1.18 s | 2/3 |

Trip counts for idqn/green_wave/ippo(100k) on these 3 seeds are all within
~0.2% of each other per seed (seed 42: idqn 2945, green_wave 2944, ippo
2942, 0.10% spread; seed 43: idqn 2991, green_wave 2997, ippo 2997, 0.20%
spread; seed 44: idqn 2950, green_wave 2949, ippo 2945, 0.17% spread) — a
true range of 2942-2997 across all three controllers and seeds. All spreads
are tiny, so there's no survivorship-bias confound — these delay numbers are
directly comparable.

## Decision

Pilot gap to green_wave: +3.09 +/- 0.37s, idqn wins 0/3 seeds. SP4's IPPO gap
on this same 3-seed subset at 100k steps was +4.38s. IDQN's +3.09s closes
that gap by about 29% — real, but not roughly half or better, and still not
a win against green_wave: 0/3 seeds beat it, and the spread is tight (sd
0.37s across the 3 seeds), which reads as a stable, consistent loss rather
than noise a larger sample would likely overturn. Per the pilot's decision
rule (docs/superpowers/plans/2026-08-21-sp5-idqn-corridor-training.md Task
7), the full 10-seed/2-scenario sweep (the remaining 17 seeds, ~84 more
hours of wall-clock) was **not** run, because a partial, tightening-but-
still-losing gap of this shape did not meet the bar for spending that
compute — a qualitative call made under the plan's own explicit latitude for
one, not a scripted threshold, and consistent with the shape of evidence
that led SP4 to stop short of its own full sweep too.

Separately, and worth its own billing: **IDQN edges IPPO by ~1.3s on average
at matched 100k budget (-1.29 +/- 1.18s), winning 2 of 3 seeds** — seed 42 is
actually a loss for IDQN (idqn 16.95s vs ippo 16.91s), and seeds 43/44 are
wins (-1.69s and -2.22s respectively). This is directionally consistent with
RESCO's own finding that independent learners beat parameter-shared ones
here — exactly the mechanism SP5 was built to test — but at n=3 with a
paired sd (1.18s) close to the size of the mean effect itself (1.29s), and
one of the three seeds landing on the wrong side of zero, this is weak
evidence, not a decisive replication. Neither arm clears the fixed-plan bar
either way.

## Verdict

The pilot result does not change the project's consolidation recommendation
(docs/HANDOFF_2026-08-21.md). This is now the third algorithm-scale
replication of "a competently-timed fixed plan beats learned control here":
single-intersection DQN reached a statistical tie with the fixed plan
(docs/FINDINGS_2026-08-12.md), corridor IPPO lost outright
(docs/FINDINGS_2026-08-18-sp4-ippo-vs-corrected-bar.md), and now corridor
IDQN's pilot loses too — closing part of IPPO's gap, but not enough of it,
and not consistently enough across even 3 seeds to expect a larger sample to
flip the direction. Consolidation stands.

That said, the pilot did show real, measured promise, and it's worth stating
as the open thread for anyone who wants to spend the remaining compute
later: IDQN cut IPPO's gap to green_wave by ~29% and edged IPPO on average
at matched budget (2/3 seeds, -1.29 +/- 1.18s) — a narrow, mixed lean, not a
wide or consistent margin, but directionally the result SP5 set out to
test. Whether the full 17-more-seed, 2-scenario sweep — or a
corridor-specific hyperparameter retune, never attempted here — would close
the remaining ~3s gap to green_wave entirely, or firm up the IPPO edge past
n=3, is untested and genuinely open. The gate stopped short of finding out,
by design, not because the promise wasn't real.
