# SP6 — IDQN demand-shift generalization (design)

**Date:** 2026-08-22
**Status:** Design approved, pending spec review
**Sub-project:** SP6 of the multi-intersection coordinated-MARL thesis extension
**Depends on:** SP1 (`make_corridor_env`, `green_wave`/`max_pressure` baselines,
`analysis/corridor_sweep.py`), SP5 (`train_corridor_dqn.py`, `dqn_core.py`,
the 9 IDQN checkpoints trained on `corridor_peak`,
`docs/FINDINGS_2026-08-21-sp5-idqn-vs-corrected-bar.md`).

## Context

Every corridor result to date (SP4's IPPO, SP5's IDQN) has trained and
evaluated a controller on the *same* demand scenario. That answers "can a
learned policy match a fixed plan on the demand it was tuned for," but not
the question adaptive control is actually sold on: does it hold up when
demand shifts to something it wasn't trained on. That question turned out to
be more pointed here than it looks at first, because of how this project's
fixed-time baseline actually works: `corridor_control.plan_phase_seconds()`
derives the phase split purely from `min_green`/`yellow_time`/`delta_time`,
and `corridor_control.green_wave_offsets()` derives offsets purely from
signal position and free-flow speed. No demand data enters either
calculation. The `green_wave` baseline is therefore already demand-blind by
construction — it runs the identical plan across `corridor_peak`,
`corridor_offpeak`, `corridor_tidal`, and `corridor_skew`, and still wins
every one of them. There is no "retune the fixed plan, then test the
shift it wasn't tuned for" experiment available in this codebase, because
the fixed plan is never tuned to begin with.

The comparable, well-posed question is the one this spec runs: does the
*trained IDQN policy* generalize when evaluated zero-shot on a demand regime
it never saw during training, or does it overfit to `corridor_peak`'s
stationary, symmetric demand shape — something the demand-blind fixed plan
never has to answer for itself. This is cheap to test right now: SP5 left 9
trained checkpoints on disk (`models/idqn_C{1,2,3}_corridor_peak_lam05_seed{42,43,44}_mg10_s100000.pt`),
one set per agent per seed, all trained on `corridor_peak` only. Nothing has
ever evaluated them on anything else.

## Scope of SP6

**In scope:**
- Decoupling `train_corridor_dqn.py`'s `evaluate()` so the scenario used to
  look up a checkpoint and the scenario used to build the eval env can
  differ (today they are the same `--scenario` argument).
- Zero-shot eval of the 9 existing `corridor_peak`-trained checkpoints (3
  agents × seeds 42/43/44) on `corridor_offpeak`, `corridor_tidal`, and
  `corridor_skew` — 9 new eval runs total, no retraining.
- Filling the missing `green_wave`/`max_pressure` baseline for
  `corridor_offpeak` at `min_green=10`, seeds 42-51 — confirmed absent from
  `analysis/corridor_sweep.csv` (unlike `corridor_peak`/`skew`/`tidal`,
  which already have it), via the existing `analysis/corridor_sweep.py`.
- A comparison table: IDQN zero-shot vs `green_wave` vs `max_pressure`, per
  off-training scenario, alongside SP5's in-distribution `corridor_peak`
  number as the reference point.
- A findings doc reporting the outcome either way — generalizes, degrades,
  or mixed.

**Explicitly out of scope (deferred):**
- Any retraining. This is a pure evaluation experiment against checkpoints
  that already exist.
- IPPO. SP4's IPPO checkpoints were saved during training
  (`models/ippo_{tag}.pt`) but `models/` is gitignored and those files are
  no longer on disk — reused/cleaned between sessions, same fate as the
  single-intersection DQN's tuned hyperparameters (SP5's own disclosed
  limitation). An IPPO version of this test would need a fresh
  `corridor_peak` training run first; that's a natural follow-up, not
  bundled here.
- A full retrain-and-re-evaluate cycle for IDQN itself. If the zero-shot
  result shows a real, interesting generalization gap (in either
  direction), a follow-up that trains IDQN directly on a mixed or
  randomized demand curriculum and compares is a reasonable next step —
  not run here.
- Any new demand scenario. This reuses `corridor_offpeak`/`tidal`/`skew` —
  all already exist, calibrated (`corridor_tidal`/`corridor_skew`'s own
  design rationale is documented in `make_scenarios.py`'s module
  docstring).

**Success criterion for SP6:** all 9 zero-shot eval runs complete and
produce trip-level CSVs through the existing `SafetyLoggingEnv`/`save_csv`
path (so they're directly comparable to every other row in
`analysis/corridor_sweep.csv`/`analysis/idqn_sweep.csv`), the missing
`corridor_offpeak` baseline sweep is filled in, and the comparison table
reports IDQN's zero-shot gap-to-`green_wave` on each shifted scenario next
to SP5's in-distribution +3.09s — whichever direction it points.

## Design

### 1. Decoupling checkpoint scenario from eval scenario

`train_corridor_dqn.py`'s `evaluate()` currently takes one `scenario`
argument used for two purposes: building `_model_path(...)` (which
checkpoint to load) and building the eval env (`make_corridor_env(...,
scenario=scenario, ...)`, which demand to run). Today those are always the
same scenario, so the coupling has never mattered. SP6 needs them to
differ: load a `corridor_peak`-trained checkpoint, run it against
`corridor_offpeak`/`tidal`/`skew` demand.

Change: `evaluate()` gains a second parameter, `eval_scenario`, defaulting
to `scenario` when not given (preserving every existing call site's
behavior unchanged). `_model_path(...)`/`_tag(...)` keep using `scenario`
(checkpoint identity — which training run this is). The env and the output
CSV's scenario fragment use `eval_scenario`. The CLI gains `--eval-scenario`
(default: same as `--scenario`), so `python train_corridor_dqn.py --eval
--scenario corridor_peak --eval-scenario corridor_offpeak --seed 42
--lam 0.5 --min-green 10 --steps 100000` loads the peak-trained seed-42
checkpoints and evaluates them on offpeak demand. Output CSV naming needs
both fragments distinguishable (e.g. `eval_idqn_<train-tag>_on_<eval_scenario>_conn<N>_ep<M>.csv`)
so a zero-shot run never collides with or is silently mistaken for an
in-distribution one — the same discipline `compare.py`'s `_warn_mixed_greens`/
`_warn_mixed_min_greens` already enforce for other run-identity fragments.

### 2. Filling the `corridor_offpeak` baseline gap

`analysis/corridor_sweep.csv` has `green_wave`/`max_pressure` rows for
`corridor_peak`, `corridor_skew`, and `corridor_tidal` at `min_green=10`
(seeds 42-51 for all three), but none for `corridor_offpeak` at any floor —
confirmed by inspection, not an assumption. Run
`analysis/corridor_sweep.py` for `corridor_offpeak`, `min_green=10`,
seeds 42-51, appending to the existing CSV via its existing resumable
convention. This is non-RL (fixed-time/max-pressure control logic only) and
fast relative to any RL training in this project — no staging/pilot gate
needed.

### 3. Zero-shot eval runs

9 runs: `{corridor_offpeak, corridor_tidal, corridor_skew} × {seed 42, 43,
44}`, each loading the matching `corridor_peak`-trained seed's 3 agent
checkpoints and evaluating on the shifted scenario's demand via the new
`--eval-scenario` flag. Output: one eval CSV + tripinfo XML per run, same
schema every other corridor eval run uses.

### 4. Comparison table

A small new script, `analysis/idqn_zeroshot.py` (mirrors
`analysis/idqn_sweep.py`'s CSV-reduction logic but does not train or gate a
sweep — it only reduces the 9 new eval CSVs plus the existing baseline
rows), produces one table:

| scenario | idqn (zero-shot, trained on peak) | green_wave | max_pressure | idqn gap-to-green_wave |
|---|---:|---:|---:|---:|
| corridor_peak (reference, in-distribution, from SP5) | 16.56 ± 0.36s | 13.47 ± 0.04s | — | +3.09 ± 0.37s |
| corridor_offpeak | (new) | (new) | (new) | (new) |
| corridor_tidal | (new) | existing | existing | (new) |
| corridor_skew | (new) | existing | existing | (new) |

### 5. Decision rule

Per shifted scenario, compare the zero-shot gap-to-`green_wave` against
SP5's in-distribution +3.09s reference:

- **Gap holds or shrinks** on a shifted scenario → the policy generalizes
  at least as well as it performed in-distribution on that shift. A real,
  reportable finding, and grounds for a follow-up (retrain-on-shifted or a
  mixed-demand curriculum) if the project wants to keep pushing.
- **Gap widens materially and consistently** (checked against
  `max_pressure`'s own delay on that scenario too, so a widened gap that's
  really "this scenario is harder for everyone" doesn't get misread as
  IDQN-specific overfitting) → overfitting to `corridor_peak`'s stationary
  demand shape. Itself a legitimate, reportable negative finding — no
  further compute needed to establish it.
- **Mixed across the three scenarios** → report per-scenario, don't average
  into one number; `corridor_offpeak` (magnitude-only shift, same
  stationary shape) and `corridor_tidal`/`corridor_skew` (structural
  shifts) are different kinds of distribution shift and a policy could
  plausibly handle one without the other.

No compute gate is needed here (unlike SP4/SP5's pilot-then-full-sweep
staging) — all 9 runs are cheap eval-only runs against checkpoints that
already exist, so there's no large commitment to gate.

## Components & boundaries

| Component | Responsibility | Depends on |
|-----------|----------------|------------|
| `train_corridor_dqn.py` (`evaluate()`, CLI) | Decoupled checkpoint-scenario vs eval-scenario; runs a zero-shot eval, writes CSV+tripinfo | `dqn_core`, `make_corridor_env`, existing SP5 checkpoints |
| `analysis/corridor_sweep.py` | Fills the missing `corridor_offpeak`/`min_green=10` `green_wave`/`max_pressure` baseline | `corridor_control`, `corridor_baseline` |
| `analysis/idqn_zeroshot.py` | Reduces the 9 new eval CSVs + existing baseline/SP5 rows into one comparison table | `analysis/tripinfo`, `analysis/corridor_sweep.csv`, SP5's in-distribution number |

## Risks & mitigations

- **Zero-shot eval env seed reuses the training seed** (`evaluate()`'s
  `seed` argument drives both checkpoint identity and, today, the env's
  demand draw) — this is existing SP5 behavior, not introduced here, and
  out of scope to fix; noting it so a reader doesn't mistake these 3 seeds
  for a larger held-out sample than they are. Comparability holds because
  every other corridor row (`green_wave`, `max_pressure`, in-distribution
  IDQN) uses the same seed convention.
- **Confounding "harder scenario for everyone" with "IDQN-specific
  overfitting"** — mitigated by §5's decision rule checking the gap against
  `max_pressure`'s own delay on the same scenario, not just against
  `green_wave` in isolation.
- **n=3 seeds per scenario** — same small-sample caveat SP5's own IPPO-vs-IDQN
  comparison disclosed (paired sd close to effect size at n=3). This spec
  reports the numbers with their spread, not a pass/fail claim, consistent
  with that precedent.
- **IPPO excluded** — checkpoints don't exist on disk to test zero-shot
  without a fresh training run first (§ scope). Flagged as the natural
  follow-up, not silently dropped.

## Open decisions deferred to later sub-projects

- An IPPO equivalent of this test, once/if `corridor_peak` IPPO checkpoints
  are retrained and kept on disk (or `models/` is exempted from
  `.gitignore` for these instead of relying on the current session's local
  cache).
- Training IDQN on a mixed or randomized demand curriculum instead of
  single-scenario `corridor_peak`, if the zero-shot result here shows a
  real, worth-closing generalization gap.
- The mid-episode incident/blockage scenario (SP7, separate spec) — a
  different, more acute kind of distribution shift than the demand-shift
  question this spec answers.
