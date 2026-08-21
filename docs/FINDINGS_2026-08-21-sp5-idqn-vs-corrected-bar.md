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
corridor numbers below rest on a validated core.

## Throughput

Measured 5.6 agent-steps/s on this machine (Task 6) — notably slower than
the single-intersection dqn_core throughput observed in Task 2 (~22-34
steps/s), consistent with the ~3x per-step network/buffer/optimizer overhead
of true independence (3 separate networks/buffers/optimizers, one per
corridor signal) versus a single shared policy. The 3-seed pilot took ~3.6h
wall-clock — faster than the ~14.9h the Task 6 throughput probe
extrapolated, because the 2000-step sample it was based on wasn't
representative of steady-state per-seed cost.

## IDQN vs green_wave and IPPO, paired, corridor_peak, min_green=10, seeds 42-44 (pilot only)

| vs | idqn (mean +/- sd) | bar (mean +/- sd) | paired idqn - bar | wins |
|---|---:|---:|---:|---:|
| green_wave | 16.56 +/- 0.36 s | 13.47 +/- 0.04 s | +3.09 +/- 0.37 s | 0/3 |
| ippo (100k, same 3 seeds) | 16.56 +/- 0.36 s | 34.36 +/- 2.21 s | -17.80 +/- 2.24 s | 3/3 |

Trip counts for idqn/green_wave/ippo on these 3 seeds are all within ~0.2% of
each other (2945-2997 range), so there's no survivorship-bias confound —
these delay numbers are directly comparable.

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

Separately, and worth its own billing: **IDQN beats IPPO by -17.80s (3/3
seeds, sd 2.24s)**. True independence — separate networks, buffers, and
optimizers per agent — clearly outperforms parameter-sharing (IPPO's pooled
single policy) on this corridor task, even though neither one clears the
fixed-plan bar. This is exactly the mechanism SP5 was built to test (RESCO's
own finding that IDQN outperforms IPPO), and it replicated here.

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
later: IDQN cut IPPO's gap to green_wave by ~29% and beat IPPO outright by a
wide, consistent margin (3/3 seeds). Whether the full 17-more-seed,
2-scenario sweep — or a corridor-specific hyperparameter retune, never
attempted here — would close the remaining ~3s gap entirely is untested and
genuinely open. The gate stopped short of finding out, by design, not
because the promise wasn't real.
