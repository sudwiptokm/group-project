# IDQN zero-shot generalization across corridor demand shifts

Written 2026-08-22. Executes SP6
(docs/superpowers/specs/2026-08-22-sp6-idqn-demand-shift-design.md): does the
`corridor_peak`-trained IDQN policy (SP5's 3 agents x 3 seeds) generalize
zero-shot to demand shapes it never trained on, or does it overfit to
`corridor_peak`'s stationary, symmetric demand?

## The question, and why "retune the fixed plan" isn't the comparison

Every corridor result to date (SP4's IPPO, SP5's IDQN) trained and evaluated
on the *same* demand scenario, which only answers "can a learned policy match
a fixed plan on the demand it was tuned for." The more pointed question
adaptive control is actually sold on is whether it holds up when demand
shifts to something it wasn't trained on. That question has no natural
"retrain the fixed plan on the new demand, then test the shift" counterpart
in this codebase, because `green_wave` is demand-blind by construction:
`corridor_control.plan_phase_seconds()` derives phase splits purely from
`min_green`/`yellow_time`/`delta_time`, and `corridor_control.green_wave_offsets()`
derives offsets purely from signal position and free-flow speed — no demand
data enters either calculation. `green_wave` runs the identical plan across
`corridor_peak`, `corridor_offpeak`, `corridor_tidal`, and `corridor_skew`
and wins every one of them without ever being "tuned" to any of them, so
there is no fixed-plan-retuning experiment to run here. The well-posed
question SP6 answers instead is whether the *trained IDQN policy* — the only
component in this comparison that actually learned something from
`corridor_peak`'s demand shape — generalizes zero-shot to a demand regime it
never saw, evaluated against the same demand-blind `green_wave` and reactive
`max_pressure` baselines used everywhere else in this project.

## Results

9 zero-shot eval runs (3 `corridor_peak`-trained checkpoint seeds x 3 shifted
scenarios, `min_green=10`, no retraining) via `analysis/idqn_zeroshot.py`,
paired against the matching `green_wave`/`max_pressure` seeds (42/43/44) in
`analysis/corridor_sweep.csv`. All delay/trip figures are seconds of delay
per completed trip; ± is sample sd.

| scenario | idqn (zero-shot, trained on peak) | green_wave | max_pressure | idqn gap-to-green_wave |
|---|---:|---:|---:|---:|
| corridor_peak (reference, in-distribution, SP5) | 16.56 ± 0.36s | 13.47 ± 0.04s | 26.38 ± 3.94s | +3.09 ± 0.37s |
| corridor_offpeak | 23.00 ± 0.37s | 11.74 ± 0.12s | 26.14 ± 1.22s | +11.26 ± 0.42s |
| corridor_tidal | 16.90 ± 0.40s | 14.03 ± 0.11s | 28.47 ± 0.38s | +2.87 ± 0.31s |
| corridor_skew | 16.46 ± 0.34s | 13.34 ± 0.07s | 28.27 ± 1.40s | +3.12 ± 0.37s |

(green_wave/max_pressure figures above use the same 3 seeds — 42/43/44 — as
the IDQN checkpoints, for a like-for-like paired comparison; the full
10-seed baseline sweep in `analysis/corridor_sweep.csv` agrees closely, e.g.
green_wave/corridor_peak is 13.46 ± 0.22s over all 10 seeds.)

## Per-scenario verdict

**corridor_tidal — gap held.** +2.87 ± 0.31s vs the +3.09 ± 0.37s
in-distribution reference — statistically indistinguishable, if anything
marginally tighter. IDQN's absolute delay (16.90s) and green_wave's (14.03s)
both sit close to their `corridor_peak` values. The policy transfers cleanly
to this structurally different (asymmetric-over-time) demand shape.

**corridor_skew — gap held.** +3.12 ± 0.37s vs +3.09 ± 0.37s — as close to
identical as this measurement can distinguish. Same read as `corridor_tidal`:
clean transfer to a structurally different (asymmetric-across-approaches)
demand shape.

**corridor_offpeak — gap widened materially.** +11.26 ± 0.42s vs +3.09 ±
0.37s — more than 3x the in-distribution reference, and the widening is
consistent (sd 0.42s across the 3 seeds, not noise). But this scenario's
demand shift is *magnitude*, not structure (same shape, ~1/3 of
`corridor_peak`'s total demand, a ~67% reduction — `make_scenarios.py`'s
`CORRIDOR_FACTORS` are 1.5 for peak vs 0.5 for offpeak, and completed-trip
counts confirm it: ~977 trips offpeak vs ~2966 peak, ≈33%), and the "harder
for everyone" check separates
two possible explanations: is `corridor_offpeak` intrinsically harder for
any controller, or is IDQN specifically failing to generalize to it?
`max_pressure`'s own gap to `green_wave` on the same scenario, seed-paired,
all 10 baseline seeds, answers this: **+14.59s on corridor_offpeak, +14.57s
on corridor_tidal, +14.31s on corridor_skew, +13.06s on corridor_peak** — flat
to within about 1.5s across all four scenarios, each well inside its own
seed-to-seed sd (0.46-3.09s). `max_pressure` is not measurably worse off on
`corridor_offpeak` than anywhere else. IDQN is the only controller whose
relative disadvantage moves materially across scenarios, and it moves only on
`corridor_offpeak`, from +3s to +11s. This is the signature of an
IDQN-specific generalization failure, not "this scenario is harder for
everyone" — the policy overfit to `corridor_peak`'s demand *magnitude* in a
way that doesn't show up when only the demand's *shape* changes
(`corridor_tidal`/`corridor_skew`).

A second, independent line of evidence points the same way: on
`corridor_offpeak`, `green_wave`'s own delay actually *improves* relative to
`corridor_peak` (13.47s -> 11.74s, demand dropping to a third makes its job
easier), while IDQN's absolute delay *rises* on that same shifted demand
(16.56s -> 23.00s). The same demand shift that makes the problem easier for
the demand-blind baseline makes it harder for the policy that trained on the
other demand magnitude — a cleaner signal than the `max_pressure` cross-check
alone, since `max_pressure` sits delay-saturated at 26-28s across all four
scenarios and is a low-sensitivity instrument for "is this scenario harder
for everyone."

## Verdict

This does not change the project's consolidation recommendation
(`docs/HANDOFF_2026-08-21.md`). The finding is mixed but resolves cleanly per
scenario, as the design spec's decision rule anticipated: IDQN generalizes
well under structural demand shifts (`corridor_tidal`, `corridor_skew` — gap
held at the in-distribution level) and specifically overfits under a
magnitude shift (`corridor_offpeak` — gap more than tripled, and the
"harder for everyone" check rules out the scenario itself being the cause).
In every case, including the two where IDQN held its relative position,
IDQN still loses outright to `green_wave` (0/3 seed wins everywhere) — so
this result narrows *why* IDQN loses (partly overfitting, not purely a
capability gap) without producing a scenario where it wins. That keeps this
a negative finding for the consolidation question specifically, even on the
two scenarios where generalization held.

Two open threads follow, neither changing the recommendation on its own:

- **If someone wants to close the `corridor_offpeak` gap specifically:** a
  follow-up that retrains IDQN on `corridor_offpeak` directly, or on a
  mixed/randomized demand-magnitude curriculum spanning peak-to-offpeak,
  is the natural next step — not run here (out of scope for SP6, which is a
  pure zero-shot evaluation against existing checkpoints, no retraining).
- **IPPO's equivalent test is blocked, not skipped.** SP4's IPPO
  checkpoints were never preserved (`models/` is gitignored and those files
  are no longer on disk), so there is no `corridor_peak`-trained IPPO
  checkpoint to run this same zero-shot comparison against. Running it
  would require a fresh `corridor_peak` IPPO training run first — a real
  option if the project wants a second learned-control datapoint on
  generalization, but it starts from a training run, not from something
  already on disk the way this task was.

Net: this is a genuine, reportable mixed finding — real generalization on
structural shifts, real overfitting on a magnitude shift — layered on top
of, not overturning, the standing conclusion that a competently-timed fixed
plan beats every learned-control variant tried on this corridor so far.
