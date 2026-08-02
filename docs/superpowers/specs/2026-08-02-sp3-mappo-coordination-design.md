# SP3 — MAPPO coordination (design)

**Date:** 2026-08-02
**Status:** Design approved, pending spec review
**Sub-project:** SP3 of the multi-intersection coordinated-MARL thesis extension —
**the research contribution.**
**Depends on:** SP2 (branch `feature/corridor-ippo`) — `ppo_core.py` (the
pluggable-critic seam), `train_corridor.py` (the shared PPO loop), the corridor
env, and the IPPO baseline.

## Context

The thesis claim is **"explicit coordination (CTDE) beats independent agents and
classical baselines."** SP2 built the independent arm (IPPO) on a deliberate
pluggable-critic design: `ppo_core.ActorCritic` already accepts a `state_dim`
distinct from the actor's `obs_dim`. SP3 flips exactly that one seam to a joint
state, producing MAPPO (centralised critic, decentralised actor). Because IPPO
and MAPPO then differ in **only the critic's input**, any performance gap is
attributable to coordination and nothing else.

Locked decisions from brainstorming:
- **Mechanism:** vanilla MAPPO — centralised critic only. The actor stays
  decentralised (local obs). No actor-side communication, no QMIX.
- **Code structure:** refactor the shared `train_corridor` loop with a
  `centralized` selector, so IPPO and MAPPO are literally the same code path with
  one flag flipped — the strongest realisation of the one-variable comparison.
- **Everything else identical to IPPO:** PPO hyperparameters, gradient clipping,
  per-agent local safety-λ reward, λ = 0.5, both corridor scenarios, seeds.

## Scope of SP3

**In scope:** a joint-state builder; a `centralized` flag threaded through
`collect_rollout` / `update` / `train` in `train_corridor.py` so each transition
stores the actor's local `obs` and the critic's `state` (local for IPPO, joint
for MAPPO); a centralised critic via the existing `state_dim` seam; a `--algo
{ippo,mappo}` CLI and `mappo` entity; `mappo` enrolled in `compare.py`; tests
including an IPPO regression guard.

**Explicitly out of scope (deferred):**
- Actor-side communication / message passing — a different contribution.
- QMIX or any value-mixing CTDE — out of the PPO family.
- Network-wide λ reshaping — **SP4**.
- The full multi-seed sweep, plots, convergence curves, and the actual
  MAPPO-vs-IPPO result numbers — **SP5**.

**Success criterion for SP3:** with `--algo mappo`, training runs on the corridor
using a 57-dim centralised critic without divergence; its greedy policy evaluates
through the existing eval-CSV path; `compare.py` ranks `mappo` alongside `ippo` /
`green_wave` / `max_pressure`; the joint-state builder and a centralised
gradient-flow check pass their unit tests; and **all pre-existing IPPO tests
still pass** (the refactor defaults to IPPO behaviour).

## Design

### 1. The one variable — joint state → centralised critic

- **Joint state** = concatenation of every agent's local observation in fixed
  `env.ts_ids` order (C1, C2, C3): 3 × 19 = **57 dimensions**, identical for all
  agents at a given timestep.
- **Critic**: `ActorCritic(obs_dim=19, act_dim=2, state_dim=57)`. Already
  supported by `ppo_core` — no change to `ppo_core.py` beyond what SP2 shipped.
- **Actor**: byte-for-byte unchanged (local obs 19 → 2 logits). Execution stays
  decentralised: at run time each signal acts on its own observation only.
- IPPO is the same class with `state_dim = obs_dim = 19` and `state = local obs`.
  MAPPO uses `state_dim = 57` and `state = joint`. That kwarg + the state fed to
  the critic is the entire difference between the two algorithms.

### 2. Refactored shared loop (state selector)

A `centralized: bool` is threaded through `train_corridor`'s `collect_rollout`,
`update`, and `train`. Each stored transition carries **both**:
- `obs` — the agent's local observation (actor input), and
- `state` — the critic input: `obs_i` when `centralized=False` (IPPO), or the
  joint state when `centralized=True` (MAPPO, same vector for all agents that
  step).

A small pure helper builds the critic states for a step:
`build_states(obs_dict, ts_ids, centralized) -> {agent: state_vector}`. For IPPO
it returns each agent's own obs; for MAPPO it returns the shared joint state for
every agent. Kept SUMO-free so it is unit-tested directly.

The actor always consumes `obs`; the critic always consumes `state`. The single
`ActorCritic` is still parameter-shared across agents, and `state_dim` is chosen
from the flag at construction (`obs_dim` for IPPO, `n_agents * obs_dim` for
MAPPO).

### 3. Rollout / GAE / update

- `collect_rollout` records per-agent `obs` and `state`, with the value computed
  from `state` (`policy.value(state)`).
- **Per-agent GAE is unchanged**: per-agent rewards, bootstrap from
  `V(trailing state)`. Under MAPPO all agents share the joint `V(s)` at each
  timestep, but per-agent returns differ (different local rewards), giving
  per-agent advantages — standard MAPPO.
- `update` and `ppo_core.ppo_loss` are unchanged; the critic simply receives
  57-dim states. Advantage normalisation, minibatching, grad clipping (max-norm
  0.5) all carry over from SP2.

### 4. Entry point / compare / invariants

- `train_corridor.py` gains `--algo {ippo,mappo}` (default `ippo`), which sets
  `centralized`. The model path and eval-CSV entity become `mappo`
  (`models/mappo_<scenario>_lam<λ>_seed<s>.pt`,
  `logs/eval_mappo_<scenario>_lam<λ>_seed<s>_*.csv`). The `_tag` helper is reused.
- `compare.py` enrolls `mappo` on the corridor scenarios × λ tags, mirroring the
  existing `ippo` block, so the ranked table shows mappo vs ippo vs the baselines.
- **Held identical to IPPO** (the isolation contract): PPO HPs from
  `cloud_params/ppo.json`, gradient clipping, the per-agent local safety-λ reward
  already produced by the env, λ = 0.5, both `corridor_peak` / `corridor_offpeak`,
  and the same train/eval seeds. Only the critic state differs.

### 5. Testing

- **Fast, SUMO-free:**
  - `build_states` — MAPPO joint state for 3 agents of dim 19 has length 57 and
    equals the concatenation in `ts_ids` order; IPPO path returns each agent's own
    obs unchanged.
  - Centralised gradient-flow gate — one `update()` with `centralized=True` and a
    57-dim critic moves the shared policy's parameters (mirrors SP2's
    `test_train_corridor_update`, guarding the MAPPO path against a no-op).
- **IPPO regression:** every pre-existing IPPO test (`test_ppo_core.py`,
  `test_train_corridor*.py`) still passes after the refactor — the `centralized`
  flag defaults to IPPO behaviour, so SP2's results are unchanged.
- **Slow smoke:** a short `--algo mappo` train run saves a model and produces a
  readable eval CSV with `system_mean_speed > 0`.
- The MAPPO-vs-IPPO coordination result itself is an SP5 experiment, not a unit
  test.

## Components & boundaries

| Component | Responsibility | Depends on |
|-----------|----------------|------------|
| `build_states` (in `train_corridor.py`) | per-step critic states: local (IPPO) or joint (MAPPO) | numpy |
| `train_corridor.py` (refactored) | shared rollout/update/train with a `centralized` flag; `--algo` CLI | `ppo_core`, `make_corridor_env` |
| `ppo_core.py` (unchanged) | ActorCritic with `state_dim` seam, GAE, ppo_loss | torch |
| `compare.py` (extended) | rank `mappo` alongside `ippo` + baselines | eval CSVs |

The refactor keeps one training code path for both algorithms; the joint-state
builder is the only genuinely new logic, and it is pure and unit-tested.

## Risks & mitigations

- **Refactor regresses IPPO** — mitigated by the IPPO regression test suite and by
  making `centralized=False` reproduce the exact prior code path (state == obs).
- **Joint-state agent ordering drift** — always build the joint state in
  `env.ts_ids` order and assert it in the unit test; a reordering would silently
  change the critic input.
- **Critic under-fitting the wider input** — the reused HPs were tuned for a
  19-dim critic; the 57-dim critic may want more capacity, but changing HPs would
  break the one-variable isolation. Keep HPs fixed and disclose; HP effects are an
  SP5 discussion, not an SP3 change.
- **Weak micro-budget smoke** — as in SP2, the fast gradient-flow gate (not the
  short SUMO run) is the real guard that MAPPO learning is wired; convergence is
  SP5.

## Open decisions deferred to later sub-projects

- Network-wide λ shaping — **SP4**.
- Full MAPPO-vs-IPPO-vs-baseline experiment sweep + plots + convergence — **SP5**.
- Actor-side communication / QMIX — out of scope for this thesis unless a reviewer
  requests a second coordination mechanism.
