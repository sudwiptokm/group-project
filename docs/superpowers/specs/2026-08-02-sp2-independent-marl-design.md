# SP2 — Independent multi-agent RL (IPPO) on the corridor (design)

**Date:** 2026-08-02
**Status:** Design approved, pending spec review
**Sub-project:** SP2 of the multi-intersection coordinated-MARL thesis extension
**Depends on:** SP1 (branch `feature/corridor-env`) — the corridor network,
`make_corridor_env`, and the non-RL baselines.

## Context

SP1 delivered the corridor substrate: a 3-signal arterial network (C1/C2/C3), a
multi-agent env `make_corridor_env` (sumo-rl `single_agent=False`, reusing the
PCU observation and the safety-λ reward), and two non-RL coordinated baselines
(green-wave, max-pressure) that feed `compare.py`.

The thesis contribution is **"explicit coordination (CTDE) beats independent
agents and classical baselines."** That comparison is only clean if the
independent learner and the coordinated learner differ in exactly one thing —
the critic. SP2 builds the **independent** arm (IPPO) as the scaffold SP3's
**MAPPO** extends by swapping the critic alone.

Locked decisions from brainstorming:
- **Framework:** custom, on the existing torch stack (no RLlib/Ray/EPyMARL).
  Rationale: control + viva-defensibility, and the project has a history of
  dependency-pinning pain.
- **Algorithm family:** PPO only — IPPO (this SP) → MAPPO (SP3).
- **Substrate:** a custom shared-policy PPO loop (NOT SB3), so IPPO and MAPPO
  share one training loop and actor and differ only in the critic. This isolates
  the coordination variable — an SB3-IPPO + custom-MAPPO split would confound
  coordination with implementation differences.

## Scope of SP2

**In scope:** a pure, SUMO-free PPO core (shared actor-critic module, GAE,
clipped surrogate loss, entropy bonus); a corridor training/eval driver that
runs parameter-shared IPPO on `make_corridor_env` and writes eval CSVs in the
existing format; a correctness gate (unit tests + a single-intersection sanity
run vs the validated SB3 PPO result); IPPO rows appearing in `compare.py`.

**Explicitly out of scope (deferred):**
- MAPPO / centralised critic — **SP3** (reuses this loop, swaps the critic).
- Network-wide λ reshaping — **SP4** (SP2 uses the per-agent local safety-λ
  reward already in the env).
- The full multi-seed experiment sweep + plots + write-up — **SP5**.
- Corridor-specific hyperparameter tuning — SP2 reuses the single-intersection
  tuned PPO HPs (disclosed limitation).

**Success criterion for SP2:** parameter-shared IPPO trains on the corridor
without divergence, its greedy policy evaluates through the existing eval-CSV
path, `compare.py` ranks `ippo` against `green_wave`/`max_pressure`, and the PPO
core passes its unit tests plus the single-intersection sanity check.

## Design

### 1. Architecture — the pluggable-critic invariant

One shared actor-critic `nn.Module`, used by all three signals (agents are
homogeneous — identical obs/action spaces, guaranteed by SP1). "Independent"
means:
- each agent selects its action from **its own local observation**, and
- the **critic is decentralised** — it values each agent from that agent's own
  local observation.

The value function is a **pluggable component** from the start:
`value = critic(state)`, where SP2 passes the agent's local obs as `state`. SP3's
MAPPO passes a global joint state instead, changing nothing else in the actor or
the training loop. This single seam is what makes the SP2-vs-SP3 comparison a
one-variable experiment.

Parameter sharing: a single network's weights are shared across all agents; each
agent's own observation is the input. All agents' transitions are pooled into one
buffer and used to update the one shared policy (standard parameter-shared IPPO).

### 2. Components / files (flat layout, matching the repo)

- **`ppo_core.py`** — pure, SUMO-free PPO math, unit-tested in isolation:
  - `ActorCritic(nn.Module)`: shared trunk → policy logits head + value head.
    The value head takes a `state` argument distinct from the actor's `obs`
    input, so SP3 can feed it a wider (joint) state without touching the actor.
  - `compute_gae(rewards, values, dones, gamma, lam)`: generalised advantage
    estimation.
  - `ppo_loss(...)`: clipped surrogate policy loss + value loss + entropy bonus,
    returning the scalar loss and a diagnostics dict.
  - No SUMO, no env, no file IO.

- **`train_corridor.py`** — SUMO-in-loop driver:
  - Rollout collector over `make_corridor_env(scenario, lam, seed)` that speaks
    the real sumo-rl multi-agent API (`reset()`→obs dict, `step(actions)`→
    `(obs, rewards, dones, info)` 4-tuple, episode ends on `dones["__all__"]`,
    agents via `env.ts_ids`, per-agent action space via `env.action_spaces(id)`),
    pooling every agent's transition into one shared buffer.
  - Training loop: collect rollouts → `compute_gae` → minibatch `ppo_loss`
    updates → save `models/ippo_<scenario>_lam<λ>_seed<s>.pt` (torch state dict;
    `.pt`, not SB3's `.zip`, since this is a custom torch policy).
  - `evaluate()`: load the shared policy, run it greedily on held-out seeds
    through `SafetyLoggingEnv`, and flush an eval CSV via
    `env.save_csv(env.out_csv_name, env.episode)` — mirroring `corridor_baseline.py`
    — named `logs/eval_ippo_<scenario>_lam<λ>_seed<s>_conn<label>_ep<ep>.csv`.
  - CLI: `--scenario {corridor_peak,corridor_offpeak}`, `--lam`, `--seed`,
    `--steps`, `--eval <model_path>`, honouring `EPISODE_SECONDS`/MODE knobs like
    the single-intersection `train.py`.

- **Tests:**
  - `tests/test_ppo_core.py` — GAE against a hand-computed known value; clipped
    loss sign/shape behaviour; ActorCritic output shapes; critic accepts a state
    dimension distinct from the actor obs dimension (guards the SP3 seam).
  - `tests/test_train_corridor.py` (slow) — train a few hundred steps: loss is
    finite, the shared policy steps the env, a model file is written, and
    `evaluate()` produces a non-empty eval CSV with `system_mean_speed > 0`.

### 3. Data flow

```
make_corridor_env(scenario, lam, seed)
   │  per-agent local obs
   ▼
shared ActorCritic ──actions dict──► env.step ──(obs,rew,dones,info)──►
   │                                                     │
   └──────────── shared rollout buffer ◄─────────────────┘
                     │  compute_gae + ppo_loss (minibatch updates)
                     ▼
        models/ippo_<scenario>_lam<λ>_seed<s>.pt
                     │  evaluate() greedy on held-out seeds
                     ▼
        logs/eval_ippo_<scenario>_lam<λ>_seed<s>_*.csv
                     │
                     ▼
        compare.py  ──►  ippo ranked vs green_wave / max_pressure (/ MAPPO in SP3)
```

### 4. Defaults

- **Parameter sharing:** one shared network (a fully-independent 3-network
  variant is a possible later ablation, not built now).
- **Hyperparameters:** reuse the single-intersection tuned PPO HPs
  (`cloud_params/ppo.json`) as the starting point. Cheap and defensible (same
  algorithm, observation, reward); **disclosed** as a limitation. No corridor
  re-tune in SP2.
- **Reward weight / scenarios:** λ = 0.5 on both `corridor_peak` and
  `corridor_offpeak`, consistent with the single-intersection headline. SP2 uses
  the per-agent local safety-λ reward already produced by the env; network-wide λ
  reshaping is SP4.
- **Seeds:** mirror the single-intersection protocol — 5 train + 5 held-out eval
  (full), 3 + 3 (overnight) — via the same MODE knobs.

### 5. Correctness guard (mitigates hand-rolled-PPO risk)

Hand-written PPO can hide subtle bugs. Before trusting any corridor number:
1. Unit-test the PPO core (GAE known-value, clip-loss behaviour, shapes).
2. Sanity-run the same shared PPO loop on the **single intersection** (one agent)
   and confirm it converges near the validated SB3 PPO result from the
   single-intersection study.
Only after both pass do corridor results count.

## Components & boundaries

| Component | Responsibility | Depends on |
|-----------|----------------|------------|
| `ppo_core.py` | PPO math + shared actor-critic; SUMO-free | torch |
| `train_corridor.py` | rollout/train/eval over the corridor env → model + eval CSV | `ppo_core`, `make_corridor_env`, `SafetyLoggingEnv` |
| `compare.py` (existing) | rank `ippo` alongside baselines | eval CSVs |

The pure/impure split (PPO math vs SUMO wiring) keeps the algorithm testable
without SUMO and gives SP3 a clean, isolated seam (the critic) to extend.

## Risks & mitigations

- **Hand-rolled PPO correctness** — mitigated by the §5 correctness guard.
- **Reused HPs may under-serve the corridor** — accepted and disclosed; SP2's job
  is a valid independent baseline, not a tuned optimum. Re-tuning is a later
  option if a reviewer challenges it.
- **sumo-rl multi-agent API drift** — the real API is pinned in SP1's notes and
  the corridor env test; the rollout collector is validated by the slow smoke
  test before any long run.
- **CSV comparability** — `evaluate()` reuses the exact `SafetyLoggingEnv` +
  `save_csv` path so IPPO rows are directly comparable to the SP1 baselines.

## Open decisions deferred to later sub-projects

- Centralised-critic design + MAPPO training details — **SP3**.
- Whether to also build a fully-independent (non-shared) IPPO ablation — optional,
  post-SP5.
- Network-wide λ shaping — **SP4**.
