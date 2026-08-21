# SP5 — Independent DQN (IDQN) corridor training (design)

**Date:** 2026-08-21
**Status:** Design approved, pending spec review
**Sub-project:** SP5 of the multi-intersection coordinated-MARL thesis extension
**Depends on:** SP1 (`make_corridor_env`, green_wave/max_pressure baselines,
`analysis/corridor_sweep.py` calibration), SP4 (the IPPO corridor result this
compares against, `docs/FINDINGS_2026-08-18-sp4-ippo-vs-corrected-bar.md`).

## Context

SP4 found IPPO loses to `green_wave` on the corridor even at a 100k-step
budget (+4.38s, 3/3 confirmatory seeds). That's consistent with the
single-intersection result (`docs/FINDINGS_2026-08-12.md`): DQN got closest
to parity there (statistical tie, not a win), PPO lost clearly. RESCO's own
comparison (Ault et al./the six-task benchmark this project is being read
against) found the same asymmetry — IDQN (independent DQN) is their best and
fastest-converging controller (~100 episodes), while their IPPO needed
~1,400 episodes and failed to converge on 2 of 6 scenarios, including a
corridor task.

SP5 tests whether the algorithm choice, not just the budget, was the gap:
train a genuinely independent DQN (matching RESCO's IDQN architecture, not
this project's parameter-shared IPPO) on the corridor and score it against
the same `green_wave`/`max_pressure` bar SP4 used.

## Scope of SP5

**In scope:** a pure, SUMO-free DQN core (`dqn_core.py`) built for
independent per-agent training (separate network/buffer/optimizer per
signal); a corridor training/eval driver (`train_corridor_dqn.py`) that runs
3 independent DQN instances on `make_corridor_env`; a correctness gate
(`analysis/validate_dqn_core.py`, dqn_core vs SB3 DQN on the single
intersection); a staged sweep (`analysis/idqn_sweep.py`, mirroring
`analysis/ippo_sweep.py`) gated by a 3-seed pilot before committing to the
full 10-seed/2-scenario run; a findings doc appending IDQN's row to the
existing IPPO/green_wave/max_pressure comparison.

**Explicitly out of scope (deferred):**
- Hyperparameter retuning for the corridor — SP5 reuses what's on record from
  the single-intersection DQN work (disclosed limitation, see §4).
- RESCO's exact network architecture (convolutional layers aggregating lanes
  by incoming road) — this project's observation is already a flat
  phase/density/queue feature vector (`env_common.py`), not RESCO's raw
  per-lane spatial encoding, so SP5 uses a plain MLP (SB3 DQN's default
  net_arch) rather than attempting to replicate RESCO's conv layers on a
  representation that doesn't support them.
- `corridor_skew` — SP4 already found this scenario's demand never approaches
  saturation, so it wasn't informative for IPPO and isn't expected to be for
  IDQN either. Not run here.
- The mid-episode incident/blockage scenario — still undesigned, unrelated to
  this sub-project.

**Success criterion for SP5:** independent DQN trains on the corridor without
divergence, its greedy policy evaluates through the existing eval-CSV path,
`dqn_core` passes its unit tests and the SB3 parity check, and the pilot/full
sweep produces a delay-per-completed-trip number directly comparable to SP4's
IPPO and green_wave rows — win, tie, or loss, reported either way.

## Design

### 1. Architecture — true independent, not parameter-shared

Unlike SP2/SP4's IPPO (one shared network, pooled transitions), SP5 gives
each of the 3 corridor signals (C1/C2/C3) its own:
- Q-network + target network (net_arch `[64, 64]`, from
  `algos.ALGOS['dqn']['defaults']()` — see §4)
- replay buffer (`buffer_size` from the same source)
- optimizer and epsilon-greedy exploration schedule

They interact only through the shared SUMO env: each timestep, all 3 agents
select actions from their own local observation, the env steps jointly, and
each agent stores its own transition in its own buffer and trains
independently once its own `learning_starts` threshold is hit. Agents are
homogeneous (identical obs/action spaces, per SP1), so the 3 networks start
from the same architecture and hyperparameters but diverge as training
proceeds.

This is a deliberate divergence from IPPO's parameter-sharing, matching
RESCO's actual IDQN rather than this project's existing "IPPO" (which is
parameter-shared despite the name). The tradeoff: this changes two variables
at once relative to SP4 (algorithm AND sharing), so a result can't cleanly
attribute a win/loss to "DQN vs PPO" alone. That's accepted — the question
SP5 asks is "does RESCO's actual best-performing setup work here," not
"is off-policy strictly better holding sharing fixed."

### 2. Components / files (flat layout, matching the repo)

- **`dqn_core.py`** — pure, SUMO-free DQN math, unit-tested in isolation:
  - `QNetwork(nn.Module)`: MLP, SB3-default architecture.
  - `ReplayBuffer`: fixed-size, uniform sampling.
  - `dqn_loss(...)`: TD loss against a target network, returning scalar loss
    and a diagnostics dict.
  - `EpsilonSchedule`: linear decay per SB3's `exploration_fraction`/
    `exploration_final_eps` semantics.
  - No SUMO, no env, no file IO — same purity discipline as `ppo_core.py`.

- **`train_corridor_dqn.py`** — SUMO-in-loop driver, mirroring
  `train_corridor.py`'s conventions:
  - 3 independent `(QNetwork, target, buffer, optimizer, epsilon)` tuples
    keyed by `env.ts_ids`.
  - Training loop: step env jointly each timestep → each agent stores its own
    transition → each agent samples its own minibatch and updates once
    `learning_starts` reached → target network sync every
    `target_update_interval` steps (per-agent).
  - Checkpoint: `models/idqn_<agent_id>_<scenario>_lam<λ>_seed<s>_mg<mg>_s<steps>.pt`
    — one file per agent, same `_tag`-style naming SP4 established, with an
    agent-id prefix so 3 files per run don't collide.
  - `evaluate()`: load all 3 agents' greedy policies, run through
    `SafetyLoggingEnv` on held-out seeds, flush an eval CSV via
    `env.save_csv` — identical path to `train_corridor.py`'s `evaluate()`, so
    the output schema matches IPPO's eval CSVs exactly.
  - CLI mirrors `train_corridor.py`: `--scenario`, `--lam`, `--seed`,
    `--steps`, `--min-green`, `--eval <model_prefix>`.

- **`analysis/validate_dqn_core.py`** — mirrors
  `analysis/validate_ppo_core.py` exactly: `dqn_core` vs SB3 `DQN`, matched
  hyperparameters (§4), single-intersection env (`scenario='base'`), not a
  pass/fail gate — reports held-out tripinfo delay and wall-clock for both so
  "is dqn_core's gradient step actually equivalent to SB3's" has evidence
  before any corridor number is trusted. Runs first, before the pilot.

- **`analysis/idqn_sweep.py`** — mirrors `analysis/ippo_sweep.py`: same
  tripinfo reduction, same seed set (42-51) for comparability, same
  resumable "reuse what's on disk" design, joins against
  `analysis/corridor_sweep.csv` (green_wave/max_pressure) and
  `analysis/ippo_sweep.csv` (IPPO) for the combined comparison table.
  `--scenario`, `--min-green 10` (SP4's calibrated floor — see §4), `--seeds`,
  `--lam 0.5`, `--steps 100000`.

- **Tests:**
  - `tests/test_dqn_core.py` — TD-loss known-value check (matches a
    hand-computed target for a small transition batch); replay buffer
    sampling shape/uniformity; `QNetwork` output shapes; epsilon schedule
    boundary values (start, mid-decay, floor).
  - `tests/test_train_corridor_dqn.py` (slow) — train a few hundred steps
    across all 3 agents: loss finite per agent, all 3 model files written,
    `evaluate()` produces a non-empty eval CSV with `system_mean_speed > 0`.

### 3. Data flow

```
make_corridor_env(scenario, lam, seed)
   │  per-agent local obs
   ▼
3x independent QNetwork ──actions dict──► env.step ──(obs,rew,dones,info)──►
   │  (each agent: own epsilon-greedy)                    │
   └── 3x independent replay buffer ◄─────────────────────┘
          │  each agent: sample own minibatch, dqn_loss, target sync
          ▼
   models/idqn_<agent>_<scenario>_lam<λ>_seed<s>_mg<mg>_s<steps>.pt  (x3)
          │  evaluate() greedy, all 3 agents, held-out seeds
          ▼
   logs/eval_idqn_<scenario>_lam<λ>_seed<s>_*.csv
          │
          ▼
   analysis/idqn_sweep.py  ──►  idqn ranked vs green_wave / max_pressure / ippo
```

### 4. Defaults and the reconstructed-config limitation

- **Architecture:** true independent (§1) — 3 separate networks, not shared.
- **Hyperparameters:** the single-intersection DQN work's tuned config
  (`params/*.json`, sized for a 100k-step budget per
  `docs/FINDINGS_2026-08-12.md`'s "Training attempted" section) is
  **unrecoverable** — gitignored, cloud-only, never committed to this repo
  (confirmed via `git log --all` on `params/`). Only 3 of its values survive,
  disclosed in prose in that findings doc:
  - `lr` = 2.3e-5
  - `learning_starts` = 5000
  - `target_update_interval` = 5000

  SP5 uses these 3 values and fills every other DQN hyperparameter
  (`buffer_size`, `batch_size`, `gamma`, `train_freq`, `exploration_fraction`,
  `exploration_final_eps`, net_arch) from `algos.ALGOS['dqn']['defaults']()`
  — this repo's own canonical "SB3 defaults" source (buffer_size 50000,
  batch_size 64, gamma 0.99, train_freq 4, exploration_fraction 0.2,
  exploration_final_eps 0.05, net_arch `[64, 64]`), not SB3's raw library
  internals. This is the same source `validate_ppo_core.py`'s `matched_hp()`
  already reads for PPO (`ALGOS['ppo']['defaults']()`), so both validation
  scripts share one convention for what "matched to SB3" means in this
  project. `tau` and `gradient_steps` are not overridden by either
  `algos.py` or the 3 disclosed values, so both stay at SB3's own internal
  defaults. This is a **best-effort reconstruction, not the original tuned
  config**, and must be disclosed as such in the findings doc — same spirit
  as SP2/SP4's disclosed reuse of single-intersection PPO hyperparameters for
  the corridor.
- **min_green floor:** 10, matching SP4's calibrated corridor floor
  (`docs/FINDINGS_2026-08-18-sp4-ippo-vs-corrected-bar.md` §"IPPO vs
  green_wave, paired, min_green=10") so IDQN's numbers are directly
  comparable to the existing IPPO/green_wave/max_pressure rows without a
  floor confound.
- **Reward weight / scenarios:** λ = 0.5, `corridor_peak` and
  `corridor_tidal` — the two scenarios SP4 fully evaluated IPPO on.
  `corridor_skew` excluded (§ scope).
- **Budget:** 100k steps — matches SP4's confirmatory-check budget, not the
  16k primary-sweep budget, and lands close to RESCO's own ~100-episode
  (~72k-step, at this project's 720 steps/episode) convergence point for
  IDQN.

### 5. Staged execution — pilot gate before the full sweep

100k steps × 2 scenarios × 10 seeds × true-independent (3x the per-step
network/buffer overhead of a shared policy) is a large upfront commitment —
estimated 15-18h wall-clock at JOBS≈5-6 on this machine, extrapolating from
the single-intersection DQN throughput measurement (6.43 steps/s) in
`docs/FINDINGS_2026-08-12.md`. Matching the project's existing bias against
over-investing before a signal exists (the SP4 budget-sensitivity check used
the same staging logic), SP5 gates the full run behind a pilot:

1. **Pilot:** `corridor_peak`, seeds 42/43/44, 100k steps (~2-3h) — the exact
   subset SP4's confirmatory check used, so the gap-to-`green_wave` numbers
   line up row-for-row against SP4's +4.38s.
2. **Decision rule:** if IDQN's pilot gap ties or meaningfully closes further
   past SP4's +4.38s → proceed to the remaining 7 `corridor_peak` seeds + all
   10 `corridor_tidal` seeds. If the gap is comparable to or worse than
   SP4's → stop, write up the pilot result as a third replication of "fixed
   plan beats learned control here," and do not spend the remaining ~85% of
   the compute budget.
3. Either outcome is a valid, reportable result — the gate controls compute
   spend, not what counts as success.

### 6. Correctness guard (mitigates hand-rolled-DQN risk)

Before trusting any corridor IDQN number:
1. Unit-test the DQN core (TD-loss known-value, buffer sampling, shapes,
   epsilon schedule boundaries).
2. Run `analysis/validate_dqn_core.py` on the single intersection and confirm
   `dqn_core` lands in the same ballpark as SB3 `DQN` at matched
   hyperparameters (not a pass/fail gate — reported evidence, same framing as
   `validate_ppo_core.py`).

Only after both pass does the pilot run.

## Components & boundaries

| Component | Responsibility | Depends on |
|-----------|----------------|------------|
| `dqn_core.py` | DQN math + independent Q-network/buffer/epsilon; SUMO-free | torch |
| `train_corridor_dqn.py` | 3x independent rollout/train/eval over the corridor env → 3 model files + eval CSV | `dqn_core`, `make_corridor_env`, `SafetyLoggingEnv` |
| `analysis/validate_dqn_core.py` | dqn_core vs SB3 DQN parity report (single intersection) | `dqn_core`, `algos.py`, SB3 |
| `analysis/idqn_sweep.py` | staged pilot → full sweep driver, resumable | `train_corridor_dqn`, `analysis/corridor_sweep.csv`, `analysis/ippo_sweep.csv` |

The pure/impure split (DQN math vs SUMO wiring) mirrors SP2's PPO split —
testable without SUMO, and keeps the correctness guard cheap to run before
any long training job.

## Risks & mitigations

- **Hand-rolled DQN correctness** — mitigated by §6's correctness guard,
  run before the pilot.
- **Reconstructed hyperparameters may not match the original tuned config** —
  accepted and disclosed (§4); this is a best-effort reuse, not a
  reproduction, and the findings doc must say so plainly.
- **Two variables changed at once vs SP4 (algorithm + sharing)** — accepted
  (§1); SP5 answers "does RESCO's actual IDQN setup work here," not an
  isolated DQN-vs-PPO ablation. A future SP could isolate sharing as its own
  variable if this result is close enough to be interesting.
- **Compute cost** — mitigated by the §5 pilot gate; full sweep only runs if
  the pilot shows a meaningfully closer gap than SP4's +4.38s.
- **CSV comparability** — `evaluate()` reuses the exact `SafetyLoggingEnv` +
  `save_csv` path SP4 used, so IDQN rows are directly comparable to the
  existing IPPO/green_wave/max_pressure rows without a schema adapter.

## Open decisions deferred to later sub-projects

- Isolating "parameter-shared vs independent" as its own controlled variable
  (holding algorithm fixed) — optional, only if SP5's result is ambiguous
  enough to warrant it.
- Corridor-specific DQN hyperparameter tuning — only if SP5's reconstructed
  config shows promise and the project decides to keep pushing on learned
  control here.
- RESCO-style conv-per-incoming-road architecture — would require also
  changing this project's observation representation; out of scope unless a
  future SP revisits the observation design itself.
