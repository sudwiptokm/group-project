# Does a magnitude-diverse training curriculum close IDQN's corridor_offpeak gap?

Written 2026-08-25. Executes SP11, the first of two open follow-ups
`docs/FINDINGS_2026-08-22-sp6-idqn-demand-shift.md` left on the table: SP6
found that an IDQN checkpoint trained only on `corridor_peak`'s demand
magnitude has its zero-shot gap to `green_wave` more than triple on
`corridor_offpeak` (+11.26s vs the in-distribution +3.09s), and diagnosed it
as overfitting to demand *magnitude* specifically (the gap held on the two
structural-shift scenarios, `corridor_tidal`/`corridor_skew`). SP6's own
suggested follow-up was to retrain on a magnitude curriculum spanning
peak-to-offpeak and see whether that closes the gap. This is that experiment.

## Method

`train_corridor_dqn.train_curriculum()` trains 3 independent IDQN agents
(C1/C2/C3, same architecture/hyperparameters/step budget as every other
IDQN run in this project) against a 5-point demand-magnitude curriculum
instead of one fixed scenario: at every episode boundary a new route file is
drawn uniformly at random from `CURRICULUM_ROUTES` — `corridor_offpeak`
(0.5x), two new intermediate files at 0.75x/1.0x/1.25x
(`make_scenarios.py`'s `CORRIDOR_CURRICULUM_FACTORS`, same shape as
`corridor_peak`/`corridor_offpeak`, only magnitude scaled), and
`corridor_peak` (1.5x). Same shape throughout, deliberately — only magnitude
varies, so any generalization difference is attributable to seeing a range
of magnitudes during training, not a structural change.

Trained 3 seeds (42/43/44), 100k steps each, `lam=0.5`, `min_green=10` —
matching SP5/SP6's peak-only checkpoints on every axis except the demand
curriculum itself. Evaluated zero-shot on `corridor_offpeak` and
`corridor_peak` (`analysis/idqn_curriculum.py`), paired against the same
`green_wave`/`max_pressure` baseline rows (seeds 42/43/44) SP6 used.

## Results

Delay/trip in seconds of delay per completed trip; ± is sample sd across the
3 seeds.

| scenario | curriculum idqn | green_wave | max_pressure | curriculum gap | peak-only gap (SP6) | change |
|---|---:|---:|---:|---:|---:|---:|
| corridor_offpeak | 21.15 ± 0.59s | 11.67s | 26.26s | +9.41 ± 0.67s | +11.26 ± 0.42s | -1.85s (-16%), narrowed |
| corridor_peak | 16.90 ± 0.21s | 13.46s | 26.52s | +3.43 ± 0.25s | +3.09 ± 0.37s | +0.34s (+11%), widened |

(`analysis/idqn_curriculum.csv` has the per-seed rows.)

## Verdict

**Narrowed, not closed, and not free.** The curriculum checkpoint's
`corridor_offpeak` gap to `green_wave` shrinks by about 16% (+11.26s ->
+9.41s) — a real, directionally-correct effect, consistent with SP6's
diagnosis that the peak-only checkpoint was specifically overfit to demand
*magnitude*. But it doesn't come close to closing the gap: curriculum IDQN
still loses to `green_wave` on every seed on `corridor_offpeak` (0/3, same
as peak-only), and the residual +9.41s gap is still more than double the
in-distribution-style reference. Seeing offpeak's magnitude during training
helps the policy cope with it, but not nearly enough to compete with a
demand-blind fixed plan on that scenario.

The trade-off SP6's design spec anticipated shows up too: the curriculum
checkpoint's `corridor_peak` gap *widens* slightly, +3.09s -> +3.43s
(+11%). Spending training exposure across 5 magnitudes instead of
concentrating it on `corridor_peak` costs a small amount of peak-specific
performance — the classic curriculum-breadth-vs-specialization trade,
visible here but small relative to the offpeak gain.

Net: this does not change the project's consolidation recommendation
(`docs/HANDOFF_2026-08-21.md`). `green_wave` still wins every seed on both
scenarios tested, with or without the curriculum. The curriculum is a
genuine partial mitigation for SP6's offpeak-specific finding — worth noting
as "magnitude curricula help but don't solve this" rather than either
"solved" or "curriculum training doesn't help at all" — but it does not
produce a scenario where IDQN wins, so it doesn't reopen the consolidation
question.
