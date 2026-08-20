# SP4 IPPO-vs-Corrected-Bar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `feature/corridor-ippo` up to the corrected corridor calibration, close its four disclosed defects, then train and evaluate IPPO on `corridor_peak` and `corridor_tidal` at an explicit floor, paired against the already-measured `green_wave` bar, and write down whether it clears it.

**Architecture:** `feature/corridor-ippo` already has a working parameter-shared PPO (`ppo_core.py` + `train_corridor.py`) built on top of the multi-agent corridor env. It predates two calibration fixes on `feature/corridor-integrate` (`f3259ed5` real max-pressure, `10d9fce7` real green-wave) and carries four known defects (`docs/HANDOFF_2026-08-18.md`): wrong ranking metric, single-seed evaluation, an unvalidated hand-rolled PPO core, and an implicit `min_green`. This plan merges the calibration in, fixes the four defects, then runs the corrected experiment using the project's existing tripinfo-based paired-seed methodology (`analysis/corridor_sweep.py`), which has already produced the green_wave/max_pressure numbers this plan compares against — no baseline re-run needed.

**Tech Stack:** Python 3.11 (venv), PyTorch 2.8.0, sumo-rl, pandas, pytest. No new dependencies.

**Spec:** `docs/superpowers/plans/2026-08-02-sp2-independent-marl.md` (original IPPO plan; superseded by this one where they conflict) and `docs/HANDOFF_2026-08-18.md` (the four defects and the decision this plan executes — "path 1").

## Global Constraints

- Run everything in the venv: `source venv/bin/activate` first; `SUMO_HOME` comes from the venv's activate script.
- Work on branch `feature/corridor-ippo` (already pushed to `origin`). Do not touch `feature/corridor-mappo` in this plan.
- `min_green` must be passed explicitly at every call site touched by this plan — never rely on `resolve_min_green`'s fallback to `DEFAULT_MIN_GREEN`/`$MIN_GREEN`. This is defect 4 from the handoff and the exact class of bug the whole calibration effort exists to catch.
- Ranking metric is delay per completed trip from tripinfo (`analysis/tripinfo.reduce_tripinfo`'s `trip_time_loss_mean`), never `system_mean_waiting_time` — see `docs/FINDINGS_2026-08-12.md` §1. This is defect 1.
- Evaluate on ≥10 seeds, paired per seed against the same-seed baseline — this is defect 2, and is exactly what `analysis/corridor_sweep.csv` already contains for `green_wave`/`max_pressure` on seeds 42-51.
- No `Co-Authored-By` / Claude / Anthropic attribution in any commit message in this plan.
- Do not commit anything under `logs/`, `models/`, or `params/` — all three are gitignored.

---

## File Structure

| File | Create/Modify | Responsibility |
|------|---------------|-----------------|
| `train_corridor.py` | Modify | Remove dead `cloud_params/ppo.json` read; make `min_green` explicit end-to-end; scenario CLI gains `corridor_tidal`; eval gains a `tripinfo` path |
| `tests/test_train_corridor.py` | Modify | Cover the inlined HP constant and explicit-`min_green` filenames |
| `analysis/ippo_sweep.py` | Create | corridor_sweep.py-style driver: train/eval IPPO per seed at one floor, reduce via tripinfo, write `analysis/ippo_sweep.csv`, and pair against `analysis/corridor_sweep.csv`'s green_wave rows |
| `tests/test_ippo_sweep.py` | Create | Unit tests for the pairing/verdict logic in `analysis/ippo_sweep.py`, on synthetic data (no SUMO) |
| `analysis/validate_ppo_core.py` | Create | Single-agent head-to-head: hand-rolled `ppo_core` vs SB3 `PPO`, matched hyperparameters/budget, on the single-intersection env |
| `docs/FINDINGS_2026-08-18-sp4-ippo-vs-corrected-bar.md` | Create | The written verdict: does IPPO clear the green_wave bar, paired, on both scenarios |

---

## Task 1: Bring `feature/corridor-ippo` up to the corrected calibration

**Files:**
- Merge only (no source changes)

- [ ] **Step 1: Fetch and check out the branch**

```bash
cd /Users/sudwipto/Desktop/group-project
source venv/bin/activate
git fetch origin
git checkout feature/corridor-ippo
git pull origin feature/corridor-ippo
```

- [ ] **Step 2: Merge the calibration fixes in**

```bash
git merge origin/feature/corridor-integrate -m "merge: bring corridor-ippo up to the corrected green_wave/max_pressure calibration"
```

Expected: `corridor_control.py`, `env_common.py`, `analysis/corridor_sweep.py`, `docs/HANDOFF_2026-08-18.md` and the various `analysis/*_superseded.csv` files come in from `corridor-integrate`. `ppo_core.py`, `train_corridor.py`, `tests/test_ppo_core.py`, `tests/test_train_corridor.py`, `tests/test_train_corridor_update.py` are ippo-only and should not conflict (corridor-integrate never touched them). If any of `corridor_control.py` / `env_common.py` conflict, take `corridor-integrate`'s side — that branch is the corrected reference.

- [ ] **Step 3: Verify the fast suite still passes post-merge**

Run: `pytest -q -m "not slow"`
Expected: all pass, including the pre-existing `test_ppo_core.py` and `test_train_corridor_update.py`.

- [ ] **Step 4: Confirm the calibration commits are present**

Run: `git log --oneline -1 f3259ed5 -- corridor_control.py` and `git log --oneline --all | grep -E "^10d9fce|^f3259ed"` from within the merged branch, or simpler: `git merge-base --is-ancestor f3259ed5 HEAD && echo ok` and same for `10d9fce7`.
Expected: both print `ok`.

- [ ] **Step 5: Push**

```bash
git push origin feature/corridor-ippo
```

---

## Task 2: Remove the dead `cloud_params/ppo.json` dependency

**Files:**
- Modify: `train_corridor.py`
- Test: `tests/test_train_corridor.py`

`train_corridor._hp()` reads `cloud_params/ppo.json`. That path is populated only by `scp`-ing `params/` down from the now-retired AWS box (`docs/AWS_CLOUD_GUIDE.md:299`) and `cloud_*/` is gitignored — on the new local-only machine this file does not exist and `_hp()` raises `FileNotFoundError` on the very first call. The values themselves are already disclosed in `docs/superpowers/plans/2026-08-02-sp2-independent-marl.md`'s header; inline them as a constant so training has no cloud dependency at all.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_corridor.py` (new file — none exists on this branch yet):
```python
"""Unit tests for train_corridor.py's pure-logic pieces (no SUMO)."""
import train_corridor as tc


def test_hp_has_no_file_dependency(tmp_path, monkeypatch):
    # cwd has no cloud_params/ dir at all -- _hp() must not need one
    monkeypatch.chdir(tmp_path)
    hp = tc._hp()
    assert hp["lr"] == 2.3195e-05
    assert hp["n_steps"] == 128
    assert hp["batch_size"] == 32
    assert hp["n_epochs"] == 10
    assert hp["gamma"] == 0.95
    assert hp["gae_lambda"] == 0.9525
    assert hp["clip_range"] == 0.1
    assert hp["ent_coef"] == 0.0081
    assert hp["hidden"] == (256, 256)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_corridor.py -v`
Expected: FAIL — `FileNotFoundError: [Errno 2] No such file or directory: 'cloud_params/ppo.json'`.

- [ ] **Step 3: Inline the constant, delete the file read**

In `train_corridor.py`, replace:
```python
PARAMS_FILE = "cloud_params/ppo.json"


def _hp() -> dict:
    """Reused single-intersection PPO hyperparameters (disclosed limitation)."""
    with open(PARAMS_FILE) as fh:
        p = json.load(fh)
    return {
        "lr": p["learning_rate"],
        "n_steps": p["n_steps"],
        "batch_size": p["batch_size"],
        "n_epochs": p["n_epochs"],
        "gamma": p["gamma"],
        "gae_lambda": p["gae_lambda"],
        "clip_range": p["clip_range"],
        "ent_coef": p["ent_coef"],
        "hidden": tuple(p["net_arch"]),
    }
```
with:
```python
# Reused single-intersection PPO hyperparameters (disclosed limitation, see
# docs/superpowers/plans/2026-08-02-sp2-independent-marl.md header). These came
# from a cloud tuning run whose params/ directory was scp'd to cloud_params/
# (docs/AWS_CLOUD_GUIDE.md); that directory is gitignored and does not exist on
# a local-only checkout, so the exact values are inlined here rather than read
# from a file that would silently vanish on a fresh clone.
_HP = {
    "lr": 2.3195e-05,
    "n_steps": 128,
    "batch_size": 32,
    "n_epochs": 10,
    "gamma": 0.95,
    "gae_lambda": 0.9525,
    "clip_range": 0.1,
    "ent_coef": 0.0081,
    "hidden": (256, 256),
}


def _hp() -> dict:
    return dict(_HP)
```
Also remove the now-unused `import json` if nothing else in the file uses it (check with `grep -n "json\." train_corridor.py` after the edit).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_corridor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train_corridor.py tests/test_train_corridor.py
git commit -m "fix: stop train_corridor from depending on the retired cloud_params dir"
```

---

## Task 3: Make `min_green` explicit end-to-end

**Files:**
- Modify: `train_corridor.py`
- Test: `tests/test_train_corridor.py`

`train()` and `evaluate()` currently call `make_corridor_env(seed=..., scenario=..., lam=...)` without `min_green`, so both silently inherit `resolve_min_green`'s fallback (`$MIN_GREEN` or `DEFAULT_MIN_GREEN=60`) — the single-intersection floor, not a corridor-calibrated one, and a value that can change under the caller from an env var they didn't set. Thread `min_green` through explicitly, require it on the CLI, and fold it into the filename tag so two floors never collide (matching `env_common.eval_csv_stem`'s / `model_path`'s own `_mg{min_green}` convention).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_train_corridor.py`:
```python
def test_tag_includes_min_green():
    assert tc._tag("corridor_peak", 0.5, seed=0, min_green=10) == \
        "corridor_peak_lam05_seed0_mg10"


def test_train_and_evaluate_require_min_green_kwarg():
    import inspect
    assert "min_green" in inspect.signature(tc.train).parameters
    assert "min_green" in inspect.signature(tc.evaluate).parameters
    # no default -- caller must always pass it
    assert inspect.signature(tc.train).parameters["min_green"].default is \
        inspect.Parameter.empty
    assert inspect.signature(tc.evaluate).parameters["min_green"].default is \
        inspect.Parameter.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_corridor.py -v`
Expected: FAIL — `_tag()` takes 3 positional args, no `min_green`.

- [ ] **Step 3: Thread `min_green` through**

In `train_corridor.py`, replace `_tag`:
```python
def _tag(scenario: str, lam: float, seed: int) -> str:
    """Filename tag shared by train (model) and evaluate (eval CSV) so the two
    never diverge; must match compare.py's `eval_ippo_<scenario>_lam<lam>_seed*` glob."""
    return f"{scenario}_lam{str(lam).replace('.', '')}_seed{seed}"
```
with:
```python
def _tag(scenario: str, lam: float, seed: int, min_green: int) -> str:
    """Filename tag shared by train (model) and evaluate (eval CSV) so the two
    never diverge. min_green is folded in (env_common's own eval_csv_stem/
    model_path convention) because a checkpoint or eval CSV is FOR one floor;
    two floors trained on the same scenario/lam/seed must not collide."""
    return f"{scenario}_lam{str(lam).replace('.', '')}_seed{seed}_mg{min_green}"
```

Update `train()`'s signature and body:
```python
def train(scenario: str, lam: float, seed: int, steps: int, min_green: int) -> str:
    hp = _hp()
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = make_corridor_env(seed=seed, scenario=scenario, lam=lam, min_green=min_green)
    obs_dim, act_dim = _obs_act_dims(env)
    policy = pc.ActorCritic(obs_dim, act_dim, hidden=hp["hidden"])
    optim = torch.optim.Adam(policy.parameters(), lr=hp["lr"])

    obs = env.reset()
    collected = 0
    while collected < steps:
        per, obs = collect_rollout(env, policy, obs, hp["n_steps"])
        update(policy, optim, per, hp, last_obs=obs)
        collected += hp["n_steps"]
    env.close()

    os.makedirs("models", exist_ok=True)
    path = f"models/ippo_{_tag(scenario, lam, seed, min_green)}.pt"
    torch.save({"state_dict": policy.state_dict(), "hidden": hp["hidden"]}, path)
    print(f"ippo model saved: {path}")
    return path
```

Update `evaluate()`'s signature and body (also add the optional `tripinfo` pass-through Task 4 needs, so this step is not repeated):
```python
def evaluate(model_path: str, scenario: str, lam: float, seed: int, min_green: int,
             tripinfo: bool = False) -> str:
    """Run the trained shared policy greedily on a held-out seed, writing an eval
    CSV in the SafetyLoggingEnv format so compare.py reads it as `ippo`. With
    tripinfo=True also writes the per-trip XML analysis/ippo_sweep.py reduces."""
    os.makedirs("logs", exist_ok=True)
    tag = _tag(scenario, lam, seed, min_green)
    out_csv = f"logs/eval_ippo_{tag}"
    env = make_corridor_env(seed=seed, scenario=scenario, lam=lam,
                            min_green=min_green, out_csv=out_csv, tripinfo=tripinfo)
    obs_dim, act_dim = _obs_act_dims(env)
    ckpt = torch.load(model_path, weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        hidden, state = tuple(ckpt["hidden"]), ckpt["state_dict"]
    else:
        hidden, state = _hp()["hidden"], ckpt
    policy = pc.ActorCritic(obs_dim, act_dim, hidden=hidden)
    policy.load_state_dict(state)
    policy.eval()

    obs = env.reset()
    done = False
    while not done:
        ids = env.ts_ids
        obs_t = torch.as_tensor(np.stack([obs[i] for i in ids]), dtype=torch.float32)
        with torch.no_grad():
            logits = policy.actor(obs_t)
        actions = {i: int(a) for i, a in zip(ids, logits.argmax(dim=-1))}
        obs, _, dones, _ = env.step(actions)
        done = dones["__all__"]
    env.save_csv(env.out_csv_name, env.episode)
    env.close()
    out = f"logs/eval_ippo_{tag}_conn{env.label}_ep{env.episode}.csv"
    print(f"ippo eval written: {out}")
    return out
```

Update the CLI block: `--scenario` gains `corridor_tidal`, and `--min-green` is required (no default):
```python
if __name__ == "__main__":
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="corridor_offpeak",
                   choices=["corridor_peak", "corridor_offpeak", "corridor_tidal"])
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--min-green", type=int, required=True,
                   help="explicit -- this script never falls back to $MIN_GREEN/DEFAULT_MIN_GREEN")
    p.add_argument("--eval", type=str, default=None,
                   help="path to a saved model to evaluate instead of training")
    p.add_argument("--tripinfo", action="store_true",
                   help="also write the per-trip XML (only meaningful with --eval)")
    args = p.parse_args()
    if args.eval:
        evaluate(args.eval, args.scenario, args.lam, args.seed, args.min_green,
                 tripinfo=args.tripinfo)
    else:
        train(args.scenario, args.lam, args.seed, args.steps, args.min_green)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_corridor.py -v`
Expected: PASS (both new tests, plus Task 2's).

- [ ] **Step 5: Fast suite regression**

Run: `pytest -q -m "not slow"`
Expected: all pass. (`tests/test_train_corridor_update.py` calls `tc.update` directly, not `train`/`evaluate`, so it is unaffected by the signature change.)

- [ ] **Step 6: Commit**

```bash
git add train_corridor.py tests/test_train_corridor.py
git commit -m "fix: require min_green explicitly in train_corridor, never inherit a default"
```

---

## Task 4: `analysis/ippo_sweep.py` — tripinfo-based IPPO evaluation, paired against green_wave

**Files:**
- Create: `analysis/ippo_sweep.py`
- Test: `tests/test_ippo_sweep.py`

Mirrors `analysis/corridor_sweep.py`'s pattern (train/eval per seed, resumable, tripinfo-reduced) but for one entity (`ippo`) at one floor, then pairs against the `green_wave` rows already sitting in `analysis/corridor_sweep.csv` — no baseline re-run.

- [ ] **Step 1: Write the failing tests (pure logic, no SUMO)**

Create `tests/test_ippo_sweep.py`:
```python
"""Unit tests for analysis/ippo_sweep.py's pairing/verdict logic (no SUMO)."""
import pandas as pd
import pytest

ip = pytest.importorskip("analysis.ippo_sweep")


def _baseline_rows(scenario, min_green, delays):
    return pd.DataFrame([
        {"controller": "green_wave", "scenario": scenario, "seed": 42 + i,
         "min_green": min_green, "delay_per_trip": d, "trips": 2900, "wall_s": 1.0}
        for i, d in enumerate(delays)
    ])


def _ippo_rows(scenario, min_green, delays):
    return pd.DataFrame([
        {"controller": "ippo", "scenario": scenario, "seed": 42 + i,
         "min_green": min_green, "delay_per_trip": d, "trips": 2900, "wall_s": 1.0}
        for i, d in enumerate(delays)
    ])


def test_paired_vs_green_wave_seed_alignment():
    baseline = _baseline_rows("corridor_peak", 10, [13.0, 14.0, 13.5])
    ippo = _ippo_rows("corridor_peak", 10, [12.0, 15.0, 13.0])
    d = ip.paired_vs_green_wave(ippo, baseline)
    # ippo - green_wave per seed: [-1.0, 1.0, -0.5]
    assert d["n"] == 3
    assert d["wins"] == 2          # negative diff = ippo wins (lower delay)
    assert abs(d["mean"] - (-1.0 / 6)) < 1e-9


def test_paired_vs_green_wave_requires_matching_seeds():
    baseline = _baseline_rows("corridor_peak", 10, [13.0, 14.0])
    ippo = _ippo_rows("corridor_peak", 10, [12.0])  # only seed 42 present
    d = ip.paired_vs_green_wave(ippo, baseline)
    assert d["n"] == 1             # only the overlapping seed is paired


def test_paired_vs_green_wave_wrong_scenario_raises():
    baseline = _baseline_rows("corridor_tidal", 10, [14.0])
    ippo = _ippo_rows("corridor_peak", 10, [12.0])
    with pytest.raises(ValueError):
        ip.paired_vs_green_wave(ippo, baseline)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ippo_sweep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analysis.ippo_sweep'`.

- [ ] **Step 3: Implement `analysis/ippo_sweep.py`**

Create `analysis/ippo_sweep.py`:
```python
"""IPPO training/eval at one explicit floor, reduced to delay-per-completed-trip
and paired against analysis/corridor_sweep.csv's green_wave rows.

This is corridor_sweep.py's methodology applied to a learned controller instead
of a reference: same tripinfo reduction (docs/FINDINGS_2026-08-12.md section 1),
same seed set (42-51, so the pairing lines up with the rows corridor_sweep.py
already produced), same "resumable, reuse what's on disk" design so an
interrupted local run picks back up.

    python -m analysis.ippo_sweep --scenario corridor_peak --min-green 10 \
        --seeds 42 43 44 45 46 47 48 49 50 51 --lam 0.5 --steps 100000
"""
import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd

import train_corridor as tc
from analysis.tripinfo import reduce_tripinfo
from env_common import tripinfo_path

OUT_CSV = os.path.join(REPO, "analysis", "ippo_sweep.csv")
CORRIDOR_SWEEP_CSV = os.path.join(REPO, "analysis", "corridor_sweep.csv")

os.environ.setdefault("TIME_TO_TELEPORT", "300")


def run_one(scenario: str, seed: int, min_green: int, lam: float, steps: int,
           force: bool = False) -> dict:
    """Train (if no checkpoint exists) + eval one seed, reduced to the ranking
    metric. Resumable: an existing model/tripinfo file is reused."""
    model_path = f"models/ippo_{tc._tag(scenario, lam, seed, min_green)}.pt"
    if force or not os.path.exists(model_path):
        t0 = time.monotonic()
        tc.train(scenario, lam, seed, steps, min_green)
        took = time.monotonic() - t0
    else:
        took = float("nan")
    tc.evaluate(model_path, scenario, lam, seed, min_green, tripinfo=True)
    trip = tripinfo_path(f"logs/eval_ippo_{tc._tag(scenario, lam, seed, min_green)}")
    row = reduce_tripinfo(trip)
    return {
        "controller": "ippo", "scenario": scenario, "seed": seed,
        "min_green": min_green, "delay_per_trip": row["trip_time_loss_mean"],
        "trips": row["trips_completed"], "wall_s": took,
    }


def sweep(scenario: str, seeds, min_green: int, lam: float, steps: int,
         force: bool = False) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        rows.append(run_one(scenario, seed, min_green, lam, steps, force))
        r = rows[-1]
        took = "reused" if pd.isna(r["wall_s"]) else f"{r['wall_s']:.0f}s"
        print(f"[{len(rows)}/{len(seeds)}] ippo seed{seed} "
              f"delay/trip={r['delay_per_trip']:7.1f}s trips={r['trips']:5d}  ({took})",
              flush=True)
    return pd.DataFrame(rows)


def paired_vs_green_wave(ippo_df: pd.DataFrame, baseline_df: pd.DataFrame) -> dict:
    """ippo - green_wave per seed, paired. Both dataframes must be one
    (scenario, min_green) already -- raises if they disagree, the same
    cross-scenario-pairing guard corridor_sweep.paired_diffs relies on."""
    i_scen = set(ippo_df["scenario"])
    b_scen = set(baseline_df["scenario"])
    if i_scen != b_scen or len(i_scen) != 1:
        raise ValueError(f"scenario mismatch: ippo={i_scen} baseline={b_scen}")
    wide = pd.merge(
        ippo_df[["seed", "delay_per_trip"]].rename(columns={"delay_per_trip": "ippo"}),
        baseline_df[["seed", "delay_per_trip"]].rename(columns={"delay_per_trip": "green_wave"}),
        on="seed", how="inner")
    d = wide["ippo"] - wide["green_wave"]
    return {
        "scenario": ippo_df["scenario"].iloc[0],
        "mean": float(d.mean()),
        "sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
        "wins": int((d < 0).sum()),
        "n": int(len(d)),
    }


def load_green_wave_bar(scenario: str, min_green: int) -> pd.DataFrame:
    """green_wave rows already in analysis/corridor_sweep.csv for this
    (scenario, min_green) -- the bar this sweep pairs IPPO against."""
    df = pd.read_csv(CORRIDOR_SWEEP_CSV)
    return df[(df["controller"] == "green_wave") & (df["scenario"] == scenario) &
              (df["min_green"] == min_green)]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True,
                   choices=["corridor_peak", "corridor_offpeak", "corridor_tidal"])
    p.add_argument("--min-green", type=int, required=True)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")

    df = sweep(args.scenario, args.seeds, args.min_green, args.lam, args.steps, args.force)
    if os.path.exists(OUT_CSV):
        prior = pd.read_csv(OUT_CSV)
        df = pd.concat([prior, df]).drop_duplicates(
            subset=["scenario", "seed", "min_green"], keep="last")
    df.to_csv(OUT_CSV, index=False)

    bar = load_green_wave_bar(args.scenario, args.min_green)
    if bar.empty:
        print(f"no green_wave rows in {CORRIDOR_SWEEP_CSV} for "
              f"{args.scenario}/mg{args.min_green} -- cannot pair")
    else:
        this_run = df[(df["scenario"] == args.scenario) &
                      (df["min_green"] == args.min_green)]
        result = paired_vs_green_wave(this_run, bar)
        print(f"\nippo - green_wave, {args.scenario} mg{args.min_green}: "
              f"{result['mean']:+.2f} +/- {result['sd']:.2f} s, "
              f"ippo wins {result['wins']}/{result['n']}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ippo_sweep.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Fast suite regression**

Run: `pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add analysis/ippo_sweep.py tests/test_ippo_sweep.py
git commit -m "feat: tripinfo-based IPPO sweep, paired against the existing green_wave bar"
```

---

## Task 5: Validate `ppo_core` against SB3 PPO on the single intersection (defect 3 — the top technical risk)

**Files:**
- Create: `analysis/validate_ppo_core.py`
- Test: `tests/test_validate_ppo_core.py`

The handoff calls this out explicitly: "Validate `ppo_core.py`'s hand-rolled PPO against SB3 PPO on the single-intersection env at matched hyperparameters/budget first — that's the top technical risk in the stack, and it's untested." This produces the head-to-head evidence, not a pass/fail gate — a from-scratch reimplementation is not expected to exactly match a mature library, but it must not be wildly worse under identical conditions.

- [ ] **Step 1: Write the failing unit test (pure logic, no SUMO)**

Create `tests/test_validate_ppo_core.py`:
```python
"""Unit test for the single-agent rollout adapter validate_ppo_core.py adds --
the piece that lets ppo_core (built for the multi-agent corridor) run on the
single-agent intersection env's (obs, reward, terminated, truncated) API."""
import pytest

vpc = pytest.importorskip("analysis.validate_ppo_core")


def test_matched_hyperparameters_come_from_algos_ppo_defaults():
    # must be byte-for-byte what SB3 PPO trains with on this task, or the
    # comparison is not apples-to-apples
    from algos import ALGOS
    sb3_defaults = ALGOS["ppo"]["defaults"]()
    hp = vpc.matched_hp()
    assert hp["lr"] == sb3_defaults["learning_rate"]
    assert hp["n_steps"] == sb3_defaults["n_steps"]
    assert hp["batch_size"] == sb3_defaults["batch_size"]
    assert hp["n_epochs"] == sb3_defaults["n_epochs"]
    assert hp["gamma"] == sb3_defaults["gamma"]
    assert hp["gae_lambda"] == sb3_defaults["gae_lambda"]
    assert hp["clip_range"] == sb3_defaults["clip_range"]
    assert hp["ent_coef"] == sb3_defaults["ent_coef"]
    assert hp["hidden"] == tuple(sb3_defaults["policy_kwargs"]["net_arch"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validate_ppo_core.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analysis.validate_ppo_core'`.

- [ ] **Step 3: Implement `analysis/validate_ppo_core.py`**

Create `analysis/validate_ppo_core.py`:
```python
"""Head-to-head: hand-rolled ppo_core vs SB3 PPO, matched hyperparameters and
step budget, on the single-agent intersection env (scenario 'base').

ppo_core/train_corridor were built for the multi-agent corridor's dict-based
API (obs/reward/dones keyed by agent id). The single-intersection env is
single-agent sumo-rl (Gymnasium-style: obs, reward, terminated, truncated), so
this file adapts ppo_core's ActorCritic/compute_gae/ppo_loss to that API rather
than reusing train_corridor.collect_rollout/update directly.

Not a pass/fail gate: a from-scratch reimplementation is not expected to exactly
match a mature library's sample efficiency. It reports both learning curves and
held-out tripinfo delay so the risk this comparison exists to surface --
"is ppo_core's gradient step actually equivalent to SB3's" -- has evidence
either way before any corridor number is trusted.

    python -m analysis.validate_ppo_core --steps 100000 --seed 0
"""
import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor

import ppo_core as pc
from algos import ALGOS, build
from analysis.tripinfo import reduce_tripinfo
from env_common import make_env, tripinfo_path

MIN_GREEN = 60  # env_common.DEFAULT_MIN_GREEN -- explicit here for the same
                # reason it must be explicit everywhere else in this project


def matched_hp() -> dict:
    """ppo_core's hyperparameter names <- algos.ALGOS['ppo']['defaults']()."""
    d = ALGOS["ppo"]["defaults"]()
    return {
        "lr": d["learning_rate"], "n_steps": d["n_steps"],
        "batch_size": d["batch_size"], "n_epochs": d["n_epochs"],
        "gamma": d["gamma"], "gae_lambda": d["gae_lambda"],
        "clip_range": d["clip_range"], "ent_coef": d["ent_coef"],
        "hidden": tuple(d["policy_kwargs"]["net_arch"]),
    }


def train_ppo_core(seed: int, steps: int) -> str:
    """Single-agent PPO training loop using ppo_core, matched to SB3 PPO's
    hyperparameters. Mirrors train_corridor.collect_rollout/update but against
    the single-agent (obs, reward, terminated, truncated) API."""
    hp = matched_hp()
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = make_env(seed=seed, scenario="base", min_green=MIN_GREEN)
    policy = pc.ActorCritic(env.observation_space.shape[0], env.action_space.n,
                            hidden=hp["hidden"])
    optim = torch.optim.Adam(policy.parameters(), lr=hp["lr"])

    obs, _ = env.reset()
    collected = 0
    while collected < steps:
        buf = {k: [] for k in ("obs", "act", "logp", "rew", "val", "done")}
        for _ in range(hp["n_steps"]):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_t, logp_t = policy.act(obs_t)
                val_t = policy.value(obs_t)
            nobs, reward, terminated, truncated, _ = env.step(int(action_t[0]))
            done = terminated or truncated
            buf["obs"].append(obs_t[0]); buf["act"].append(action_t[0])
            buf["logp"].append(logp_t[0]); buf["val"].append(float(val_t[0]))
            buf["rew"].append(float(reward)); buf["done"].append(float(done))
            obs = nobs
            if done:
                obs, _ = env.reset()

        with torch.no_grad():
            last_val = float(policy.value(
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0])
        adv, ret = pc.compute_gae(buf["rew"], buf["val"], buf["done"],
                                  hp["gamma"], hp["gae_lambda"], last_value=last_val)
        obs_b = torch.stack(buf["obs"]); act_b = torch.stack(buf["act"])
        old_logp = torch.stack(buf["logp"]).detach()
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)

        n = obs_b.shape[0]
        idx = np.arange(n)
        for _ in range(hp["n_epochs"]):
            np.random.shuffle(idx)
            for start in range(0, n, hp["batch_size"]):
                b = idx[start:start + hp["batch_size"]]
                dist = policy.policy(obs_b[b]); vals = policy.value(obs_b[b])
                loss, _ = pc.ppo_loss(dist, act_b[b], old_logp[b], adv_t[b], vals,
                                      ret_t[b], clip=hp["clip_range"],
                                      ent_coef=hp["ent_coef"])
                optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optim.step()
        collected += hp["n_steps"]
    env.close()

    os.makedirs("models", exist_ok=True)
    path = f"models/validate_ppo_core_seed{seed}.pt"
    torch.save({"state_dict": policy.state_dict(), "hidden": hp["hidden"]}, path)
    return path


def eval_ppo_core(model_path: str, seed: int) -> float:
    out_csv = f"logs/eval_validate_ppo_core_seed{seed}"
    env = make_env(seed=seed, scenario="base", min_green=MIN_GREEN,
                   out_csv=out_csv, tripinfo=True)
    ckpt = torch.load(model_path, weights_only=True)
    policy = pc.ActorCritic(env.observation_space.shape[0], env.action_space.n,
                            hidden=tuple(ckpt["hidden"]))
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    obs, _ = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            logits = policy.actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        obs, _, terminated, truncated, _ = env.step(int(logits.argmax(dim=-1)[0]))
        done = terminated or truncated
    env.save_csv(env.out_csv_name, env.episode)
    env.close()
    return reduce_tripinfo(tripinfo_path(out_csv))["trip_time_loss_mean"]


def train_eval_sb3(seed: int, steps: int) -> float:
    params = ALGOS["ppo"]["defaults"]()
    env = Monitor(make_env(seed=seed, scenario="base", min_green=MIN_GREEN))
    model = build("ppo", env, params, seed=seed, tb_log="logs/tb")
    model.learn(total_timesteps=steps)
    env.close()

    out_csv = f"logs/eval_validate_sb3_ppo_seed{seed}"
    eval_env = make_env(seed=seed, scenario="base", min_green=MIN_GREEN,
                        out_csv=out_csv, tripinfo=True)
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
    eval_env.save_csv(eval_env.out_csv_name, eval_env.episode)
    eval_env.close()
    return reduce_tripinfo(tripinfo_path(out_csv))["trip_time_loss_mean"]


if __name__ == "__main__":
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.monotonic()
    ppo_core_model = train_ppo_core(args.seed, args.steps)
    ppo_core_delay = eval_ppo_core(ppo_core_model, seed=args.seed + 1000)
    t1 = time.monotonic()
    sb3_delay = train_eval_sb3(args.seed, args.steps)
    t2 = time.monotonic()

    print(f"\nppo_core:  delay/trip={ppo_core_delay:.1f}s  wall={t1 - t0:.0f}s")
    print(f"sb3 PPO:   delay/trip={sb3_delay:.1f}s  wall={t2 - t1:.0f}s")
    print(f"ppo_core - sb3: {ppo_core_delay - sb3_delay:+.1f}s")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validate_ppo_core.py -v`
Expected: PASS.

- [ ] **Step 5: Fast suite regression**

Run: `pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 6: Run the real validation (slow, SUMO)**

Run:
```bash
python -m analysis.validate_ppo_core --steps 100000 --seed 0
```
Expected: prints both delay/trip numbers and the difference. Record the two numbers — they go into the findings doc in Task 7. This is evidence-gathering, not a scripted pass/fail: if `ppo_core` is dramatically worse (e.g. >2x delay, or it never brings the policy off a random baseline), stop and debug `ppo_core`/`update()` before spending any further compute on corridor training — that would mean the top technical risk materialised.

- [ ] **Step 7: Commit**

```bash
git add analysis/validate_ppo_core.py tests/test_validate_ppo_core.py
git commit -m "feat: validate ppo_core against SB3 PPO, matched HPs, single intersection"
```

---

## Task 6: Measure throughput, size the step budget

**Files:**
- None (measurement only)

Per the handoff: "size the run to what the machine can actually do — measure `agent-steps/s` for one short IPPO run before committing to a step budget." Do this before Task 7's real training runs.

- [ ] **Step 1: Timed short run**

Run:
```bash
source venv/bin/activate
python -c "
import time
import train_corridor as tc
t0 = time.monotonic()
tc.train('corridor_peak', lam=0.5, seed=0, steps=2000, min_green=10)
dt = time.monotonic() - t0
print(f'{2000/dt:.1f} agent-steps/s, {dt:.1f}s for 2000 steps')
"
rm -f models/ippo_corridor_peak_lam05_seed0_mg10.pt
```

- [ ] **Step 2: Pick the step budget**

Using the measured agent-steps/s, choose a `--steps` value for Task 7 so that 10 seeds x 2 scenarios completes in an acceptable wall-clock window on this machine. Write the chosen number and the throughput measurement down at the top of `docs/FINDINGS_2026-08-18-sp4-ippo-vs-corrected-bar.md` (created in Task 7) before running anything else — this is the same "measure before trusting" discipline the corridor calibration itself used, and the number belongs in the writeup regardless of what it turns out to be.

No commit for this task — it produces a number that feeds the next task, not a code change.

---

## Task 7: Train + evaluate IPPO on both scenarios, paired against the bar, write the verdict

**Files:**
- Create: `docs/FINDINGS_2026-08-18-sp4-ippo-vs-corrected-bar.md`

Floor choice: `min_green=10`. This matches `green_wave`'s own-best floor (the bar being cleared) and is the finer of the two references' best floors, so IPPO is not handicapped with a coarser action grid than the controller it must beat. Seeds: 42-51, matching `analysis/corridor_sweep.csv`'s existing `green_wave` rows exactly, so `ippo_sweep.paired_vs_green_wave` pairs cleanly with no re-run of the baseline.

- [ ] **Step 1: Run the sweep on `corridor_peak`**

Run (replace `<STEPS>` with Task 6's chosen budget):
```bash
source venv/bin/activate
python -m analysis.ippo_sweep --scenario corridor_peak --min-green 10 --lam 0.5 \
    --steps <STEPS> --seeds 42 43 44 45 46 47 48 49 50 51
```
Expected: 10 lines of per-seed output, then a paired-vs-green_wave summary line. This can be interrupted and re-run — `run_one` reuses an existing model/tripinfo file, same as `corridor_sweep.py`.

- [ ] **Step 2: Run the sweep on `corridor_tidal`**

Run:
```bash
python -m analysis.ippo_sweep --scenario corridor_tidal --min-green 10 --lam 0.5 \
    --steps <STEPS> --seeds 42 43 44 45 46 47 48 49 50 51
```

- [ ] **Step 3: Write the findings doc**

Create `docs/FINDINGS_2026-08-18-sp4-ippo-vs-corrected-bar.md` with (fill in every `<...>` with the actual numbers from Steps 1-2 and Task 5/6 — no placeholders left in the committed version):

```markdown
# IPPO vs the corrected corridor bar

Written 2026-08-18. Executes path 1 of docs/HANDOFF_2026-08-18.md: train IPPO
against the corrected green_wave/max_pressure calibration and report the
result, negative or not.

## Throughput and budget

Measured <AGENT_STEPS_PER_S> agent-steps/s on this machine (analysis in
Task 6 of docs/superpowers/plans/2026-08-18-sp4-ippo-corridor-training.md).
Trained at <STEPS> steps per seed on that basis.

## ppo_core vs SB3 PPO (single intersection, matched HPs, defect 3)

| | delay/trip (s) |
|---|---:|
| ppo_core | <PPO_CORE_DELAY> |
| SB3 PPO  | <SB3_DELAY> |

<one or two sentences: is ppo_core competitive, and is that enough confidence
to trust the corridor numbers below>

## IPPO vs green_wave, paired, min_green=10, seeds 42-51

| scenario | ippo (mean +/- sd) | green_wave (mean +/- sd, from analysis/corridor_sweep.csv) | paired ippo - gw | wins |
|---|---:|---:|---:|---:|
| corridor_peak  | <IPPO_PEAK_MEAN> +/- <IPPO_PEAK_SD> s | 13.46 +/- 0.22 s | <DIFF_PEAK> +/- <DIFF_PEAK_SD> s | <WINS_PEAK>/10 |
| corridor_tidal | <IPPO_TIDAL_MEAN> +/- <IPPO_TIDAL_SD> s | 13.96 +/- 0.34 s | <DIFF_TIDAL> +/- <DIFF_TIDAL_SD> s | <WINS_TIDAL>/10 |

## Verdict

<Does IPPO clear the bar on either scenario? State the result plainly, in
either direction. If it does not clear the bar (paired mean >= 0, wins < 5/10)
on both scenarios: state that explicitly as the negative result path 1 of the
handoff anticipated, and that this is the disciplined outcome to report, not a
failure of the exercise. If it does clear the bar on one or both: say which,
by how much, and that SP3 (MAPPO) is the natural next step per the handoff's
open decision.>
```

- [ ] **Step 4: Commit**

```bash
git add analysis/ippo_sweep.csv docs/FINDINGS_2026-08-18-sp4-ippo-vs-corrected-bar.md
git commit -m "docs: IPPO vs the corrected corridor bar -- the path-1 result"
```

Do NOT commit anything under `logs/` or `models/` (gitignored); `analysis/ippo_sweep.csv` is the only run output this plan tracks, matching how `analysis/corridor_sweep.csv` is already tracked.

- [ ] **Step 5: Push**

```bash
git push origin feature/corridor-ippo
```

---

## Self-Review Notes

- **Spec coverage:** all four handoff defects addressed — Task 4 fixes the metric (defect 1), Task 7's seed set fixes the single-seed problem (defect 2), Task 5 fixes the unvalidated PPO core (defect 3), Task 3 fixes the implicit `min_green` (defect 4). Task 1 merges in the calibration itself. Task 2 fixes a blocker found during planning (dead `cloud_params/ppo.json` path) that would otherwise stop Task 1's very first training run.
- **Placeholder scan:** the only bracketed placeholders are in the findings-doc template in Task 7 Step 3, explicitly called out as "fill in from the actual run" with real numbers required before commit — not a deferred TODO.
- **Type/name consistency:** `_tag(scenario, lam, seed, min_green)`, `train(scenario, lam, seed, steps, min_green)`, `evaluate(model_path, scenario, lam, seed, min_green, tripinfo=False)` used identically across Tasks 2-4 and 7's CLI invocations. `ippo_sweep.run_one`/`sweep`/`paired_vs_green_wave`/`load_green_wave_bar` names match between implementation (Task 4 Step 3) and test (Task 4 Step 1).
- **Deferred correctly:** MAPPO (SP3, separate branch, out of scope here); a fixed-plan-can't-serve scenario design (path 2 of the handoff — not chosen); re-tuning IPPO's hyperparameters for the corridor (the plan explicitly reuses the disclosed single-intersection HPs, consistent with the original SP2 plan's own disclosed limitation).
```
