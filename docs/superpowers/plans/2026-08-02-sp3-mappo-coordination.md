# SP3 MAPPO Coordination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the IPPO trainer into MAPPO by feeding the shared critic a joint state (concat of all agents' local obs) via a `centralized` flag, so IPPO and MAPPO are one code path differing only in the critic input.

**Architecture:** A pure `build_states` helper returns each agent's critic state — its own local obs (IPPO) or the joint concatenation (MAPPO). A `centralized` flag threads through `collect_rollout` / `update` / `train` / `evaluate` in `train_corridor.py`; each transition stores the actor's local `obs` and the critic's `state`. The actor (local obs → logits) is unchanged; only the critic's `state_dim` and input change. `compare.py` enrolls `mappo` alongside `ippo`.

**Tech Stack:** Python 3.11 (venv), PyTorch, the SP1 corridor env + SP2 `ppo_core`/`train_corridor`, `pytest`. No new dependencies. `ppo_core.py` is NOT modified — its `ActorCritic(state_dim=…)` seam already supports this.

**Conventions:**
- Run in the venv: `source venv/bin/activate` first.
- Branch `feature/corridor-mappo` off `feature/corridor-ippo` (created in Task 1).
- Corridor facts: 3 agents `env.ts_ids == ["C1","C2","C3"]`, local obs `Box(19,)`, action `Discrete(2)`; joint state dim = 3×19 = 57.
- The refactor MUST keep IPPO behaviour when `centralized=False` (state == local obs): all SP2 IPPO tests must still pass.
- `@pytest.mark.slow` marker already registered (pytest.ini). Slow tests need `SUMO_HOME` (set by venv activate).

---

## File Structure

| File | Modify/Create | Responsibility |
|------|---------------|----------------|
| `train_corridor.py` | Modify | add `build_states`; thread `centralized` through collect_rollout/update/train/evaluate; `--algo {ippo,mappo}` CLI |
| `compare.py` | Modify | enroll `mappo` on corridor scenarios |
| `tests/test_build_states.py` | Create | unit tests for the joint/local state builder |
| `tests/test_train_corridor_update.py` | Modify | add `state` to the synthetic buffer + a centralized gradient-flow variant |
| `tests/test_train_corridor.py` | Modify | add a slow MAPPO train+eval smoke |
| `README.md` | Modify | MAPPO (SP3) subsection |

---

## Task 1: Joint-state builder

**Files:**
- Modify: `train_corridor.py` (add `build_states`)
- Test: `tests/test_build_states.py`

- [ ] **Step 1: Create the branch**

```bash
cd /Users/sudwipto/Desktop/group_project
git checkout feature/corridor-ippo
git checkout -b feature/corridor-mappo
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_build_states.py`:
```python
"""Unit tests for the critic state builder (SUMO-free)."""
import numpy as np

import train_corridor as tc


def test_ippo_state_is_local_obs():
    obs = {"C1": np.array([1.0, 2.0]), "C2": np.array([3.0, 4.0])}
    states = tc.build_states(obs, ["C1", "C2"], centralized=False)
    assert np.array_equal(states["C1"], np.array([1.0, 2.0]))
    assert np.array_equal(states["C2"], np.array([3.0, 4.0]))


def test_mappo_state_is_joint_concat_in_order():
    obs = {"C1": np.array([1.0, 2.0]), "C2": np.array([3.0, 4.0]), "C3": np.array([5.0, 6.0])}
    states = tc.build_states(obs, ["C1", "C2", "C3"], centralized=True)
    joint = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    for agent in ("C1", "C2", "C3"):
        assert np.array_equal(states[agent], joint)  # same joint state for all agents


def test_mappo_state_respects_ts_ids_order():
    obs = {"C1": np.array([1.0]), "C2": np.array([2.0]), "C3": np.array([3.0])}
    # a different id order must reorder the concatenation
    states = tc.build_states(obs, ["C3", "C1", "C2"], centralized=True)
    assert np.array_equal(states["C1"], np.array([3.0, 1.0, 2.0]))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/test_build_states.py -v`
Expected: FAIL — `AttributeError: module 'train_corridor' has no attribute 'build_states'`.

- [ ] **Step 4: Add `build_states` to train_corridor.py**

Insert after the `_tag(...)` function in `train_corridor.py`:
```python
def build_states(obs, ts_ids, centralized):
    """Per-agent critic state. IPPO (centralized=False): each agent's own local
    observation. MAPPO (centralized=True): the joint state — all agents' local
    observations concatenated in ts_ids order — shared by every agent.
    """
    if not centralized:
        return {i: obs[i] for i in ts_ids}
    joint = np.concatenate([obs[i] for i in ts_ids])
    return {i: joint for i in ts_ids}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_build_states.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add train_corridor.py tests/test_build_states.py
git commit -m "feat: joint-state builder for MAPPO centralised critic"
```
(No Co-Authored-By / Claude / Anthropic trailer on any commit in this plan.)

---

## Task 2: Thread `centralized` through the training loop

**Files:**
- Modify: `train_corridor.py` (collect_rollout, update, train, evaluate, `__main__`)
- Modify: `tests/test_train_corridor_update.py` (buffer gains `state`; add centralized variant)

This is the core refactor. IPPO behaviour is preserved when `centralized=False`
(state == local obs), guarded by the existing IPPO tests.

- [ ] **Step 1: Update the gradient-flow test to the new buffer shape + add a centralized variant**

Replace the whole body of `tests/test_train_corridor_update.py` with:
```python
"""Fast, SUMO-free guard that a PPO update MOVES the shared policy — for both the
IPPO (local-obs critic) and MAPPO (joint-state critic) paths. Catches a broken
gradient flow / no-op optimizer without needing SUMO.
"""
import copy

import torch

import ppo_core as pc
import train_corridor as tc

OBS_DIM = 19
N_AGENTS = 2  # two synthetic agents


def _buffer(state_dim, n=32):
    """Per-agent buffer with both `obs` (actor input, OBS_DIM) and `state` (critic
    input, state_dim). Reward correlates with action so there is a gradient signal."""
    per = {}
    for agent in ("C1", "C2"):
        acts = [torch.tensor(i % 2) for i in range(n)]
        per[agent] = {
            "obs": [torch.randn(OBS_DIM) for _ in range(n)],
            "state": [torch.randn(state_dim) for _ in range(n)],
            "act": acts,
            "logp": [torch.tensor(-0.6931) for _ in range(n)],
            "val": [0.0 for _ in range(n)],
            "rew": [1.0 if int(acts[i]) == 1 else 0.0 for i in range(n)],
            "done": [0.0] * (n - 1) + [1.0],
        }
    return per


def _run_update(state_dim):
    torch.manual_seed(0)
    policy = pc.ActorCritic(OBS_DIM, 2, state_dim=state_dim)
    optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
    hp = {"gamma": 0.99, "gae_lambda": 0.95, "n_epochs": 4,
          "batch_size": 16, "clip_range": 0.2, "ent_coef": 0.0}
    per = _buffer(state_dim)
    last_states = {a: torch.randn(state_dim) for a in per}
    before = copy.deepcopy(policy.state_dict())
    tc.update(policy, optim, per, hp, last_states)
    after = policy.state_dict()
    return any(not torch.equal(before[k], after[k]) for k in before)


def test_update_moves_policy_ippo_local_state():
    assert _run_update(state_dim=OBS_DIM), "IPPO update moved no parameter"


def test_update_moves_policy_mappo_joint_state():
    assert _run_update(state_dim=OBS_DIM * N_AGENTS), "MAPPO update moved no parameter"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `source venv/bin/activate && pytest tests/test_train_corridor_update.py -v`
Expected: FAIL — `update()` still reads no `state`, or the current signature mismatches (the test now expects `update` to consume `per[agent]["state"]`).

- [ ] **Step 3: Refactor `collect_rollout` to store per-agent critic state**

In `train_corridor.py`, replace the whole `collect_rollout` function with:
```python
def collect_rollout(env, policy, obs, n_steps, centralized):
    """Step the env n_steps, storing a SEPARATE temporal buffer per agent. Each
    transition keeps the actor's local `obs` and the critic's `state` (local obs
    for IPPO, joint state for MAPPO). Returns (per_agent_buffers, trailing_obs)."""
    ids = env.ts_ids
    per = {i: {"obs": [], "state": [], "act": [], "logp": [], "rew": [],
               "val": [], "done": []} for i in ids}
    for _ in range(n_steps):
        obs_t = torch.as_tensor(np.stack([obs[i] for i in ids]), dtype=torch.float32)
        states = build_states(obs, ids, centralized)
        state_t = torch.as_tensor(np.stack([states[i] for i in ids]), dtype=torch.float32)
        with torch.no_grad():
            actions_t, logp_t = policy.act(obs_t)   # actor: local obs
            vals_t = policy.value(state_t)          # critic: state (local or joint)
        actions = {i: int(a) for i, a in zip(ids, actions_t)}
        nobs, rewards, dones, _ = env.step(actions)
        done_all = float(dones["__all__"])
        for k, i in enumerate(ids):
            per[i]["obs"].append(obs_t[k])
            per[i]["state"].append(state_t[k])
            per[i]["act"].append(actions_t[k])
            per[i]["logp"].append(logp_t[k])
            per[i]["val"].append(float(vals_t[k]))
            per[i]["rew"].append(float(rewards[i]))
            per[i]["done"].append(done_all)
        obs = nobs
        if done_all:
            obs = env.reset()
    return per, obs
```

- [ ] **Step 4: Refactor `update` to use the stored critic state + state bootstrap**

Replace the whole `update` function with:
```python
def update(policy, optim, per, hp, last_states):
    """One PPO update: per-agent GAE (bootstrapping V from each agent's trailing
    critic state), then concatenated minibatch updates. The actor is trained on
    local `obs`; the critic on `state` (local for IPPO, joint for MAPPO)."""
    all_obs, all_state, all_act, all_logp, all_adv, all_ret = [], [], [], [], [], []
    for agent, b in per.items():
        with torch.no_grad():
            lv = float(policy.value(
                torch.as_tensor(last_states[agent], dtype=torch.float32)))
        adv, ret = pc.compute_gae(b["rew"], b["val"], b["done"],
                                  hp["gamma"], hp["gae_lambda"], last_value=lv)
        all_obs += b["obs"]
        all_state += b["state"]
        all_act += b["act"]
        all_logp += b["logp"]
        all_adv += adv
        all_ret += ret

    obs = torch.stack(all_obs)
    state = torch.stack(all_state)
    act = torch.stack(all_act)
    old_logp = torch.stack(all_logp).detach()
    adv_t = torch.as_tensor(all_adv, dtype=torch.float32)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t = torch.as_tensor(all_ret, dtype=torch.float32)

    n = obs.shape[0]
    idx = np.arange(n)
    last_info = {}
    for _ in range(hp["n_epochs"]):
        np.random.shuffle(idx)
        for start in range(0, n, hp["batch_size"]):
            bi = idx[start:start + hp["batch_size"]]
            dist = policy.policy(obs[bi])          # actor: local obs
            vals = policy.value(state[bi])         # critic: state
            loss, info = pc.ppo_loss(dist, act[bi], old_logp[bi], adv_t[bi], vals,
                                     ret_t[bi], clip=hp["clip_range"],
                                     ent_coef=hp["ent_coef"])
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optim.step()
            last_info = info
    return last_info
```

- [ ] **Step 5: Refactor `train` to build a critic with the right `state_dim` and pass `centralized`**

Replace the whole `train` function with:
```python
def train(scenario: str, lam: float, seed: int, steps: int,
          centralized: bool = False) -> str:
    hp = _hp()
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = make_corridor_env(seed=seed, scenario=scenario, lam=lam)
    obs_dim, act_dim = _obs_act_dims(env)
    n_agents = len(env.ts_ids)
    state_dim = obs_dim * n_agents if centralized else obs_dim
    policy = pc.ActorCritic(obs_dim, act_dim, state_dim=state_dim, hidden=hp["hidden"])
    optim = torch.optim.Adam(policy.parameters(), lr=hp["lr"])

    obs = env.reset()
    collected = 0  # ENV-steps (matches SB3 total_timesteps semantics)
    while collected < steps:
        per, obs = collect_rollout(env, policy, obs, hp["n_steps"], centralized)
        last_states = build_states(obs, env.ts_ids, centralized)
        update(policy, optim, per, hp, last_states)
        collected += hp["n_steps"]
    env.close()

    algo = "mappo" if centralized else "ippo"
    os.makedirs("models", exist_ok=True)
    path = f"models/{algo}_{_tag(scenario, lam, seed)}.pt"
    torch.save({"state_dict": policy.state_dict(), "hidden": hp["hidden"],
                "centralized": centralized, "state_dim": state_dim}, path)
    print(f"{algo} model saved: {path}")
    return path
```

- [ ] **Step 6: Refactor `evaluate` to rebuild the exact critic + derive the entity**

Replace the whole `evaluate` function with:
```python
def evaluate(model_path: str, scenario: str, lam: float, seed: int) -> str:
    """Greedy eval of a saved policy on a held-out seed. The entity (ippo/mappo)
    and the critic width are read from the checkpoint so the net is rebuilt exactly.
    The actor uses local obs, so eval is identical in shape for IPPO and MAPPO."""
    ckpt = torch.load(model_path, weights_only=True)
    is_dict = isinstance(ckpt, dict) and "state_dict" in ckpt
    centralized = bool(ckpt.get("centralized", False)) if is_dict else False
    algo = "mappo" if centralized else "ippo"

    os.makedirs("logs", exist_ok=True)
    tag = _tag(scenario, lam, seed)
    out_csv = f"logs/eval_{algo}_{tag}"
    env = make_corridor_env(seed=seed, scenario=scenario, lam=lam, out_csv=out_csv)
    obs_dim, act_dim = _obs_act_dims(env)
    if is_dict:
        hidden = tuple(ckpt["hidden"])
        state_dim = ckpt.get("state_dim", obs_dim)
        state = ckpt["state_dict"]
    else:
        hidden, state_dim, state = _hp()["hidden"], obs_dim, ckpt
    policy = pc.ActorCritic(obs_dim, act_dim, state_dim=state_dim, hidden=hidden)
    policy.load_state_dict(state)
    policy.eval()

    obs = env.reset()
    done = False
    while not done:
        ids = env.ts_ids
        obs_t = torch.as_tensor(np.stack([obs[i] for i in ids]), dtype=torch.float32)
        with torch.no_grad():
            logits = policy.actor(obs_t)          # greedy, decentralised (local obs)
        actions = {i: int(a) for i, a in zip(ids, logits.argmax(dim=-1))}
        obs, _, dones, _ = env.step(actions)
        done = dones["__all__"]
    env.save_csv(env.out_csv_name, env.episode)
    env.close()
    out = f"logs/eval_{algo}_{tag}_conn{env.label}_ep{env.episode}.csv"
    print(f"{algo} eval written: {out}")
    return out
```

- [ ] **Step 7: Add `--algo` to the CLI**

Replace the `__main__` block with:
```python
if __name__ == "__main__":
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="corridor_offpeak",
                   choices=["corridor_peak", "corridor_offpeak"])
    p.add_argument("--algo", default="ippo", choices=["ippo", "mappo"])
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--eval", type=str, default=None,
                   help="path to a saved model to evaluate instead of training")
    args = p.parse_args()
    if args.eval:
        evaluate(args.eval, args.scenario, args.lam, args.seed)
    else:
        train(args.scenario, args.lam, args.seed, args.steps,
              centralized=(args.algo == "mappo"))
```

- [ ] **Step 8: Run the fast update tests + the full fast suite (IPPO regression)**

Run:
```bash
source venv/bin/activate
pytest tests/test_train_corridor_update.py -v
pytest -q -m "not slow"
```
Expected: both update tests PASS (IPPO + MAPPO gradient flow); the full fast suite (build_states, ppo_core, corridor, compare, safety) all PASS — the `centralized=False` default preserves IPPO.

- [ ] **Step 9: IPPO slow regression (the refactor must not change IPPO behaviour)**

Run: `EPISODE_SECONDS=200 pytest tests/test_train_corridor.py -q -m slow`
Expected: the existing IPPO slow smoke + learning-check PASS. (A few minutes.)

- [ ] **Step 10: Commit**

```bash
git add train_corridor.py tests/test_train_corridor_update.py
git commit -m "feat: thread centralized flag through the corridor PPO loop (IPPO/MAPPO)"
```

---

## Task 3: MAPPO smoke + compare enrollment + docs

**Files:**
- Modify: `compare.py` (enroll `mappo`), `tests/test_train_corridor.py` (slow MAPPO smoke), `README.md`

- [ ] **Step 1: Write the failing slow MAPPO smoke test**

Append to `tests/test_train_corridor.py`:
```python
@pytest.mark.skipif(not os.environ.get("SUMO_HOME"), reason="SUMO_HOME not set")
def test_mappo_trains_and_evaluates(monkeypatch):
    monkeypatch.setenv("EPISODE_SECONDS", "200")
    model = tc.train("corridor_offpeak", lam=0.5, seed=0, steps=600, centralized=True)
    assert os.path.exists(model)
    assert "mappo" in os.path.basename(model)

    csv = tc.evaluate(model, "corridor_offpeak", lam=0.5, seed=42)
    assert os.path.exists(csv)
    assert "eval_mappo_" in os.path.basename(csv)
    import pandas as pd
    assert pd.read_csv(csv)["system_mean_speed"].mean() > 0
```

- [ ] **Step 2: Run it to verify it passes**

Run: `EPISODE_SECONDS=200 pytest tests/test_train_corridor.py::test_mappo_trains_and_evaluates -v -m slow`
Expected: PASS — a `models/mappo_*.pt` is written (centralised 57-dim critic), the
eval CSV is `eval_mappo_*`, mean speed > 0. (A few minutes.) If it fails on a real
bug, report BLOCKED with the error.

- [ ] **Step 3: Enroll `mappo` in compare.py**

In `compare.py` `main()`, find the `ippo` corridor block added in SP2:
```python
    corridor_lambdas = ["00", "05", "10"]
    for scenario in corridor_scenarios:
        for lam in corridor_lambdas:
            df = _run_means(args.logs, "ippo", scenario, lam=lam)
            if not df.empty:
                rows.append(_summarise(df, "ippo", scenario, lam))
```
Generalise it to both RL entities by replacing that block with:
```python
    corridor_lambdas = ["00", "05", "10"]
    corridor_rl = ["ippo", "mappo"]
    for scenario in corridor_scenarios:
        for entity in corridor_rl:
            for lam in corridor_lambdas:
                df = _run_means(args.logs, entity, scenario, lam=lam)
                if not df.empty:
                    rows.append(_summarise(df, entity, scenario, lam))
```

- [ ] **Step 4: Build the table (mappo eval CSV from Step 2 should still be in logs/)**

Run:
```bash
source venv/bin/activate
python compare.py
```
Expected: `logs/comparison.csv` and the printed table include a `mappo` row under
`corridor_offpeak` (lam 05) alongside `ippo` / `green_wave` / `max_pressure`. Do
NOT commit anything under `logs/` (gitignored).

- [ ] **Step 5: Add a README subsection**

In `README.md`, under the existing "Corridor RL — IPPO (SP2)" section, add a
"Corridor RL — MAPPO (SP3)" subsection documenting: MAPPO = the same
`train_corridor` loop with a centralised critic (joint state = concat of all
signals' local obs, 57-dim), actor unchanged (decentralised execution), selected
with `--algo mappo`; that IPPO and MAPPO differ only in the critic input so the
comparison isolates coordination; and the commands:
```bash
python train_corridor.py --algo mappo --scenario corridor_peak --lam 0.5 --seed 0 --steps 100000
python train_corridor.py --algo mappo --scenario corridor_peak --lam 0.5 --seed 42 \
    --eval models/mappo_corridor_peak_lam05_seed0.pt
python compare.py    # ranks mappo vs ippo vs green_wave / max_pressure
```
Note the full MAPPO-vs-IPPO result is an SP5 experiment.

- [ ] **Step 6: Full fast suite + IPPO regression**

Run:
```bash
source venv/bin/activate
pytest -q -m "not slow"
pytest tests/test_safety_reward.py -q
```
Expected: all fast tests PASS; single-intersection regression PASS.

- [ ] **Step 7: Commit**

```bash
git add compare.py tests/test_train_corridor.py README.md
git commit -m "feat: MAPPO smoke + corridor ranking + docs"
```

---

## Self-Review Notes

- **Spec coverage:** joint-state builder (T1), `centralized` threaded through
  collect_rollout/update/train/evaluate + CLI (T2), centralised critic via the
  existing `state_dim` seam (T2 train), MAPPO smoke + `mappo` enrolled in compare
  + docs (T3). IPPO regression guarded by the unchanged SP2 fast+slow tests re-run
  in T2 Steps 8–9. All SP3 spec sections mapped.
- **`ppo_core.py` unchanged:** confirmed — only `state_dim` (already present) is
  used. No task modifies it.
- **IPPO isolation preserved:** `train(..., centralized=False)` builds
  `state_dim=obs_dim` and `build_states` returns local obs, so the IPPO code path
  and results are byte-identical to SP2 (positional `train(scenario,lam,seed,steps)`
  calls still resolve, defaulting `centralized=False`).
- **Checkpoint back-compat:** `evaluate` reads `centralized`/`state_dim` from the
  checkpoint and falls back to IPPO/`obs_dim` for a bare state_dict, so SP2 `.pt`
  files still evaluate.
- **Naming consistency:** `build_states(obs, ts_ids, centralized)`,
  `update(..., last_states)`, `train(..., centralized=False)`, entities
  `ippo`/`mappo`, `models/<algo>_<tag>.pt`, `logs/eval_<algo>_<scenario>_lam<λ>_seed*` —
  matching `compare.py`'s glob and used identically across tasks.
```
