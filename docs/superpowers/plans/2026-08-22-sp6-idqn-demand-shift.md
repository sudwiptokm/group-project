# SP6 IDQN Demand-Shift Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the 9 existing `corridor_peak`-trained IDQN checkpoints (SP5) zero-shot on `corridor_offpeak`/`corridor_tidal`/`corridor_skew` — demand scenarios they never trained on — and compare each zero-shot gap-to-`green_wave` against SP5's in-distribution +3.09s reference, to find out whether the trained policy generalizes or overfits to `corridor_peak`'s stationary demand shape.

**Architecture:** `train_corridor_dqn.py`'s `evaluate()` gets a new `eval_scenario` parameter (default: `scenario`) that decouples which checkpoint is loaded from which demand scenario the eval env runs, via a new pure helper `_eval_out_stem()`. A new script, `analysis/idqn_zeroshot.py`, drives the 9 zero-shot eval runs, reduces them to delay-per-completed-trip, and pairs each shifted scenario's IDQN row against the existing `green_wave`/`max_pressure` rows in `analysis/corridor_sweep.csv` (mirroring `analysis/idqn_sweep.py`'s `paired_vs` contract). No training — this is a pure evaluation experiment against artifacts SP5 already produced.

**Tech Stack:** Python 3.11 (venv), PyTorch 2.8.0, sumo-rl, pandas, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-22-sp6-idqn-demand-shift-design.md`.

## Global Constraints

- Run everything in the venv: `source venv/bin/activate` first; `SUMO_HOME` comes from the venv's activate script.
- Work directly on `main` — this project has no per-SP feature-branch model since SP4 (`docs/HANDOFF_2026-08-21.md`).
- `min_green` is always `10` in this plan — SP4's calibrated corridor floor, and the floor SP5's checkpoints were trained at. Never omit it or rely on a default.
- Ranking metric is delay per completed trip from tripinfo (`analysis/tripinfo.reduce_tripinfo`'s `trip_time_loss_mean`), never `system_mean_waiting_time` — see `docs/FINDINGS_2026-08-12.md` §1.
- This plan trains nothing. Every eval run in Task 4 loads an existing checkpoint (`models/idqn_C{1,2,3}_corridor_peak_lam05_seed{42,43,44}_mg10_s100000.pt`) — if any of the 9 are missing, stop and report it rather than retraining (retraining is explicitly out of scope, spec §Scope).
- No `Co-Authored-By` / Claude / Anthropic attribution in any commit message in this plan.
- Do not commit anything under `logs/` or `models/` (both gitignored). `analysis/corridor_sweep.csv` and `analysis/idqn_zeroshot.csv` are the two run-output files this plan tracks (both already-tracked CSV aggregates, matching `analysis/idqn_sweep.csv`'s precedent).

---

## File Structure

| File | Create/Modify | Responsibility |
|------|---------------|-----------------|
| `train_corridor_dqn.py` | Modify | `evaluate()` gains `eval_scenario` param + `_eval_out_stem()` helper; CLI gains `--eval-scenario` |
| `tests/test_idqn_hp.py` | Modify | Unit tests for `_eval_out_stem()` and the new CLI flag |
| `tests/test_train_corridor_dqn.py` | Modify | Slow smoke test: zero-shot eval actually runs the checkpoint against a different scenario's demand |
| `analysis/corridor_sweep.csv` | Modify (data) | Filled with `corridor_offpeak` `green_wave`/`max_pressure` rows at `min_green=10`, seeds 42-51 |
| `analysis/idqn_zeroshot.py` | Create | Runs the 9 zero-shot evals, reduces to delay-per-trip, pairs against `green_wave`/`max_pressure` |
| `tests/test_idqn_zeroshot.py` | Create | Unit tests for the pairing/loading logic, on synthetic data (no SUMO) |
| `analysis/idqn_zeroshot.csv` | Create (data) | Per-seed zero-shot eval results |
| `docs/FINDINGS_2026-08-22-sp6-idqn-demand-shift.md` | Create | The written verdict |

---

## Task 1: Decouple checkpoint scenario from eval scenario in `train_corridor_dqn.py`

**Files:**
- Modify: `train_corridor_dqn.py` (the `evaluate()` function and its CLI block)
- Test: `tests/test_idqn_hp.py`
- Test: `tests/test_train_corridor_dqn.py`

**Interfaces:**
- Consumes: `train_corridor_dqn._tag(scenario, lam, seed, min_green, steps) -> str` (already exists).
- Produces: `_eval_out_stem(scenario, eval_scenario, lam, seed, min_green, steps) -> str`; `evaluate(scenario, lam, seed, min_green, steps, tripinfo=False, eval_scenario=None) -> str` (extended signature — `eval_scenario` is new, defaults to `None` meaning "same as `scenario`"). Both consumed by Task 4's `analysis/idqn_zeroshot.py`.

- [ ] **Step 1: Write the failing fast tests (pure logic, no SUMO)**

Add to `tests/test_idqn_hp.py`:
```python
def test_eval_out_stem_in_distribution_unchanged():
    assert tcd._eval_out_stem("corridor_peak", "corridor_peak", 0.5, 42, 10, 100000) == \
        "logs/eval_idqn_corridor_peak_lam05_seed42_mg10_s100000"


def test_eval_out_stem_zero_shot_appends_on_scenario():
    assert tcd._eval_out_stem("corridor_peak", "corridor_offpeak", 0.5, 42, 10, 100000) == \
        "logs/eval_idqn_corridor_peak_lam05_seed42_mg10_s100000_on_corridor_offpeak"


def test_evaluate_eval_scenario_defaults_to_none():
    import inspect
    sig = inspect.signature(tcd.evaluate)
    assert "eval_scenario" in sig.parameters
    assert sig.parameters["eval_scenario"].default is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_idqn_hp.py -v`
Expected: FAIL — `AttributeError: module 'train_corridor_dqn' has no attribute '_eval_out_stem'` (and the third test fails on the missing parameter).

- [ ] **Step 3: Implement the decoupling**

In `train_corridor_dqn.py`, add this function right above `evaluate()`:
```python
def _eval_out_stem(scenario: str, eval_scenario: str, lam: float, seed: int,
                   min_green: int, steps: int) -> str:
    """Eval CSV path stem. When eval_scenario differs from the checkpoint's
    training scenario (SP6 zero-shot generalization eval), '_on_<eval_scenario>'
    is appended so a zero-shot run's output can never collide with or be
    mistaken for an in-distribution one -- same discipline compare.py's
    _warn_mixed_greens/_warn_mixed_min_greens enforce for other run-identity
    fragments (env_common.py's docstring convention: every optional fragment
    sits after seed<n>)."""
    tag = _tag(scenario, lam, seed, min_green, steps)
    if eval_scenario != scenario:
        return f"logs/eval_idqn_{tag}_on_{eval_scenario}"
    return f"logs/eval_idqn_{tag}"
```

Replace the existing `evaluate()` function with:
```python
def evaluate(scenario: str, lam: float, seed: int, min_green: int, steps: int,
            tripinfo: bool = False, eval_scenario: str = None) -> str:
    """Run all 3 agents' greedy policies for one episode, writing one eval
    CSV in the SafetyLoggingEnv format so compare.py reads it as `idqn`. With
    tripinfo=True also writes the per-trip XML analysis/idqn_sweep.py reduces.

    Loads each agent's checkpoint by reconstructing its path from _model_path
    -- callers never pass paths directly, so train() and evaluate() can never
    disagree about where a checkpoint lives.

    eval_scenario, if given, evaluates the checkpoint's greedy policy against
    a DIFFERENT demand scenario than it was trained on -- SP6's zero-shot
    generalization eval (docs/superpowers/specs/2026-08-22-sp6-idqn-demand-shift-design.md).
    Defaults to `scenario` (today's in-distribution behaviour, unchanged): the
    checkpoint is always looked up under `scenario`, but the env runs whichever
    scenario `eval_scenario` names."""
    os.makedirs("logs", exist_ok=True)
    eval_scenario = eval_scenario or scenario
    out_csv = _eval_out_stem(scenario, eval_scenario, lam, seed, min_green, steps)
    env = make_corridor_env(seed=seed, scenario=eval_scenario, lam=lam,
                            min_green=min_green, out_csv=out_csv, tripinfo=tripinfo)
    ids = env.ts_ids
    obs_dim, act_dim = _obs_act_dims(env)
    policies = {}
    for i in ids:
        ckpt = torch.load(_model_path(i, scenario, lam, seed, min_green, steps),
                          weights_only=True)
        q_net = dc.QNetwork(obs_dim, act_dim, hidden=tuple(ckpt["hidden"]))
        q_net.load_state_dict(ckpt["state_dict"])
        q_net.eval()
        policies[i] = q_net

    obs = env.reset()
    done = False
    while not done:
        actions = {}
        for i in ids:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                actions[i] = int(policies[i](obs_t).argmax(dim=-1).item())
        obs, _, dones, _ = env.step(actions)
        done = dones["__all__"]
    env.save_csv(env.out_csv_name, env.episode)
    env.close()
    out = f"{out_csv}_conn{env.label}_ep{env.episode}.csv"
    print(f"idqn eval written: {out}")
    return out
```

In the `if __name__ == "__main__":` CLI block, add the flag right after `--seed`:
```python
    p.add_argument("--eval-scenario", default=None, choices=list(CORRIDOR_SCENARIOS),
                   help="demand scenario to evaluate on, if different from "
                        "--scenario (zero-shot; defaults to --scenario)")
```
And change the `--eval` branch's call to:
```python
    if args.eval:
        evaluate(args.scenario, args.lam, args.seed, args.min_green, args.steps,
                 tripinfo=args.tripinfo, eval_scenario=args.eval_scenario)
```

- [ ] **Step 4: Run fast tests to verify they pass**

Run: `pytest tests/test_idqn_hp.py -v`
Expected: all PASS (existing tests + 3 new ones).

- [ ] **Step 5: Write the slow smoke test**

Add to `tests/test_train_corridor_dqn.py`:
```python
@pytest.mark.skipif(not os.environ.get("SUMO_HOME"), reason="SUMO_HOME not set")
def test_idqn_zero_shot_eval_runs_on_different_scenario(monkeypatch):
    monkeypatch.setenv("EPISODE_SECONDS", "200")
    tcd.train("corridor_peak", lam=0.5, seed=1, steps=600, min_green=10)
    csv = tcd.evaluate("corridor_peak", lam=0.5, seed=1, min_green=10, steps=600,
                       eval_scenario="corridor_offpeak")
    assert "_on_corridor_offpeak" in csv
    assert os.path.exists(csv)
    import pandas as pd
    df = pd.read_csv(csv)
    assert df["system_mean_speed"].mean() > 0
```

- [ ] **Step 6: Run the slow test (SUMO required)**

Run: `pytest tests/test_train_corridor_dqn.py -v -m slow`
Expected: PASS (skipped if `SUMO_HOME` is unset).

- [ ] **Step 7: Fast suite regression**

Run: `pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add train_corridor_dqn.py tests/test_idqn_hp.py tests/test_train_corridor_dqn.py
git commit -m "feat: decouple IDQN eval scenario from checkpoint scenario for zero-shot eval"
```

---

## Task 2: Fill the missing `corridor_offpeak` baseline

**Files:**
- None (existing script, data-only task)

`analysis/corridor_sweep.csv` has `green_wave`/`max_pressure` rows for `corridor_peak`, `corridor_skew`, and `corridor_tidal` at `min_green=10` (seeds 42-51), but none for `corridor_offpeak` at any floor — confirmed by inspection during design. `analysis/corridor_sweep.py` already handles this scenario and floor; it's just never been run for it.

- [ ] **Step 1: Run the baseline sweep for `corridor_offpeak`**

Run:
```bash
source venv/bin/activate
python analysis/corridor_sweep.py --scenario corridor_offpeak --min-greens 10 \
    --seeds 42 43 44 45 46 47 48 49 50 51
```
Expected: prints 20 lines (`green_wave` + `max_pressure`, 10 seeds each), then the per-scenario report block for `corridor_offpeak`. This is fast (non-RL, single-episode-per-row).

- [ ] **Step 2: Verify the CSV was appended, not overwritten**

Run: `python -c "import pandas as pd; df = pd.read_csv('analysis/corridor_sweep.csv'); print(df['scenario'].value_counts())"`
Expected: `corridor_peak`, `corridor_skew`, `corridor_tidal` counts unchanged from before this task, plus a new `corridor_offpeak` count of 20 (2 controllers × 10 seeds).

- [ ] **Step 3: Commit**

```bash
git add analysis/corridor_sweep.csv
git commit -m "data: fill missing corridor_offpeak green_wave/max_pressure baseline"
```

---

## Task 3: `analysis/idqn_zeroshot.py` — zero-shot eval driver and comparison

**Files:**
- Create: `analysis/idqn_zeroshot.py`
- Test: `tests/test_idqn_zeroshot.py`

**Interfaces:**
- Consumes: `train_corridor_dqn._tag`, `train_corridor_dqn._eval_out_stem`, `train_corridor_dqn.evaluate` (Task 1); `analysis.tripinfo.reduce_tripinfo`; `env_common.tripinfo_path`, `env_common.CORRIDOR_SCENARIOS`.
- Produces: `run_one(eval_scenario, seed, force=False) -> dict`; `zeroshot_sweep(scenarios, seeds, force=False) -> pd.DataFrame`; `load_baseline(controller, scenario, min_green=10) -> pd.DataFrame`; `paired_gap(idqn_df, bar_df) -> dict`; `report(zeroshot_df) -> None`. Consumed directly (as a script) by Task 4's real run.

- [ ] **Step 1: Write the failing tests (pure logic, no SUMO)**

Create `tests/test_idqn_zeroshot.py`:
```python
"""Unit tests for analysis/idqn_zeroshot.py's pairing/loading logic (no SUMO)."""
import pandas as pd
import pytest

iz = pytest.importorskip("analysis.idqn_zeroshot")


def _rows(scenario, delays, seeds=(42, 43, 44)):
    return pd.DataFrame([
        {"scenario": scenario, "seed": s, "min_green": 10, "delay_per_trip": d,
         "trips": 2900}
        for s, d in zip(seeds, delays)
    ])


def test_paired_gap_seed_alignment():
    idqn = _rows("corridor_offpeak", [12.0, 15.0, 13.0])
    gw = _rows("corridor_offpeak", [13.0, 14.0, 13.5])
    gap = iz.paired_gap(idqn, gw)
    # idqn - green_wave per seed: [-1.0, 1.0, -0.5]
    assert gap["n"] == 3
    assert gap["wins"] == 2
    assert abs(gap["mean"] - (-1.0 / 6)) < 1e-9
    assert gap["scenario"] == "corridor_offpeak"


def test_paired_gap_only_pairs_overlapping_seeds():
    idqn = _rows("corridor_tidal", [12.0], seeds=(42,))
    gw = _rows("corridor_tidal", [13.0, 14.0], seeds=(42, 43))
    gap = iz.paired_gap(idqn, gw)
    assert gap["n"] == 1


def test_paired_gap_wrong_scenario_raises():
    idqn = _rows("corridor_peak", [12.0])
    gw = _rows("corridor_tidal", [14.0])
    with pytest.raises(ValueError):
        iz.paired_gap(idqn, gw)


def test_load_baseline_filters_controller_scenario_min_green(tmp_path, monkeypatch):
    csv = tmp_path / "corridor_sweep.csv"
    pd.DataFrame([
        {"controller": "green_wave", "scenario": "corridor_offpeak", "seed": 42,
         "min_green": 10, "delay_per_trip": 9.0, "trips": 2900, "wall_s": 1.0},
        {"controller": "max_pressure", "scenario": "corridor_offpeak", "seed": 42,
         "min_green": 10, "delay_per_trip": 9.5, "trips": 2900, "wall_s": 1.0},
        {"controller": "green_wave", "scenario": "corridor_offpeak", "seed": 42,
         "min_green": 20, "delay_per_trip": 20.0, "trips": 2900, "wall_s": 1.0},
    ]).to_csv(csv, index=False)
    monkeypatch.setattr(iz, "CORRIDOR_SWEEP_CSV", str(csv))
    df = iz.load_baseline("green_wave", "corridor_offpeak", min_green=10)
    assert len(df) == 1
    assert df.iloc[0]["delay_per_trip"] == 9.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_idqn_zeroshot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analysis.idqn_zeroshot'`.

- [ ] **Step 3: Implement `analysis/idqn_zeroshot.py`**

Create `analysis/idqn_zeroshot.py`:
```python
"""SP6: IDQN zero-shot demand-shift generalization.

Evaluates the 9 existing corridor_peak-trained IDQN checkpoints (SP5) on
demand scenarios they never trained on -- corridor_offpeak, corridor_tidal,
corridor_skew -- and pairs each shifted scenario's result against the
existing green_wave/max_pressure rows in analysis/corridor_sweep.csv.

No training happens here. See
docs/superpowers/specs/2026-08-22-sp6-idqn-demand-shift-design.md.

    python -m analysis.idqn_zeroshot
    python -m analysis.idqn_zeroshot --scenarios corridor_tidal --seeds 42
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd

import train_corridor_dqn as tcd
from analysis.tripinfo import reduce_tripinfo
from env_common import CORRIDOR_SCENARIOS, tripinfo_path

CORRIDOR_SWEEP_CSV = os.path.join(REPO, "analysis", "corridor_sweep.csv")
OUT_CSV = os.path.join(REPO, "analysis", "idqn_zeroshot.csv")

TRAIN_SCENARIO = "corridor_peak"
SHIFTED_SCENARIOS = ("corridor_offpeak", "corridor_tidal", "corridor_skew")
SEEDS = (42, 43, 44)
LAM = 0.5
STEPS = 100_000
MIN_GREEN = 10

os.environ.setdefault("TIME_TO_TELEPORT", "300")

# SP5's in-distribution reference: idqn - green_wave on corridor_peak, same
# seeds, same checkpoints (docs/FINDINGS_2026-08-21-sp5-idqn-vs-corrected-bar.md).
INDIST_GAP = {"scenario": TRAIN_SCENARIO, "mean": 3.09, "sd": 0.37, "wins": 0, "n": 3}


def run_one(eval_scenario: str, seed: int, force: bool = False) -> dict:
    """Zero-shot eval: load the corridor_peak-trained seed's checkpoint, run
    it on eval_scenario's demand. Resumable -- an existing tripinfo file for
    this exact (eval_scenario, seed) is reused."""
    stem = tcd._eval_out_stem(TRAIN_SCENARIO, eval_scenario, LAM, seed, MIN_GREEN, STEPS)
    trip = tripinfo_path(stem)
    if force or not os.path.exists(trip):
        tcd.evaluate(TRAIN_SCENARIO, LAM, seed, MIN_GREEN, STEPS, tripinfo=True,
                     eval_scenario=eval_scenario)
    row = reduce_tripinfo(trip)
    return {
        "scenario": eval_scenario, "seed": seed, "min_green": MIN_GREEN,
        "delay_per_trip": row["trip_time_loss_mean"], "trips": row["trips_completed"],
    }


def zeroshot_sweep(scenarios=SHIFTED_SCENARIOS, seeds=SEEDS, force: bool = False) -> pd.DataFrame:
    rows = []
    total = len(scenarios) * len(seeds)
    for scenario in scenarios:
        for seed in seeds:
            rows.append(run_one(scenario, seed, force))
            r = rows[-1]
            print(f"[{len(rows)}/{total}] idqn zero-shot {scenario} seed{seed} "
                  f"delay/trip={r['delay_per_trip']:7.1f}s trips={r['trips']:5d}",
                  flush=True)
    return pd.DataFrame(rows)


def load_baseline(controller: str, scenario: str, min_green: int = MIN_GREEN) -> pd.DataFrame:
    """green_wave/max_pressure rows already in analysis/corridor_sweep.csv for
    this (controller, scenario, min_green)."""
    df = pd.read_csv(CORRIDOR_SWEEP_CSV)
    return df[(df["controller"] == controller) & (df["scenario"] == scenario) &
              (df["min_green"] == min_green)]


def paired_gap(idqn_df: pd.DataFrame, bar_df: pd.DataFrame) -> dict:
    """idqn - bar_df per seed, paired. Both dataframes must be one scenario --
    raises if they disagree, same cross-scenario guard
    analysis.idqn_sweep.paired_vs enforces."""
    i_scen = set(idqn_df["scenario"])
    b_scen = set(bar_df["scenario"])
    if i_scen != b_scen or len(i_scen) != 1:
        raise ValueError(f"scenario mismatch: idqn={i_scen} bar={b_scen}")
    wide = pd.merge(
        idqn_df[["seed", "delay_per_trip"]].rename(columns={"delay_per_trip": "idqn"}),
        bar_df[["seed", "delay_per_trip"]].rename(columns={"delay_per_trip": "bar"}),
        on="seed", how="inner")
    d = wide["idqn"] - wide["bar"]
    return {
        "scenario": idqn_df["scenario"].iloc[0],
        "mean": float(d.mean()),
        "sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
        "wins": int((d < 0).sum()), "n": int(len(d)),
    }


def report(zeroshot_df: pd.DataFrame) -> None:
    print(f"\n=== reference: idqn in-distribution, {INDIST_GAP['scenario']} (SP5) ===")
    print(f"  gap vs green_wave: {INDIST_GAP['mean']:+.2f} +/- {INDIST_GAP['sd']:.2f} s, "
          f"idqn wins {INDIST_GAP['wins']}/{INDIST_GAP['n']}")

    for scenario, g in zeroshot_df.groupby("scenario"):
        print(f"\n################ {scenario} (zero-shot) ################")
        gw = load_baseline("green_wave", scenario)
        mp = load_baseline("max_pressure", scenario)
        if gw.empty:
            print(f"  [!] no green_wave baseline for {scenario}/mg{MIN_GREEN} -- cannot pair")
            continue
        gap = paired_gap(g, gw)
        print(f"  idqn - green_wave: {gap['mean']:+.2f} +/- {gap['sd']:.2f} s, "
              f"idqn wins {gap['wins']}/{gap['n']}  "
              f"(in-distribution reference: {INDIST_GAP['mean']:+.2f}s)")
        if not mp.empty:
            mp_wide = pd.merge(
                mp[["seed", "delay_per_trip"]].rename(columns={"delay_per_trip": "mp"}),
                gw[["seed", "delay_per_trip"]].rename(columns={"delay_per_trip": "gw"}),
                on="seed", how="inner")
            mp_delta = float((mp_wide["mp"] - mp_wide["gw"]).mean())
            print(f"  max_pressure - green_wave (same scenario, for the "
                  f"'harder for everyone' check): {mp_delta:+.2f}s")
        else:
            print(f"  [!] no max_pressure baseline for {scenario}/mg{MIN_GREEN}")


def main():
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", default=list(SHIFTED_SCENARIOS),
                   choices=list(CORRIDOR_SCENARIOS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    df = zeroshot_sweep(args.scenarios, args.seeds, args.force)
    if os.path.exists(OUT_CSV):
        prior = pd.read_csv(OUT_CSV)
        df = pd.concat([prior, df]).drop_duplicates(subset=["scenario", "seed"], keep="last")
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(df)} rows)")
    report(df)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_idqn_zeroshot.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Fast suite regression**

Run: `pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add analysis/idqn_zeroshot.py tests/test_idqn_zeroshot.py
git commit -m "feat: analysis/idqn_zeroshot, zero-shot IDQN eval paired against green_wave/max_pressure"
```

---

## Task 4: Run the 9 zero-shot evals and build the comparison table

**Files:**
- None (run task, produces `analysis/idqn_zeroshot.csv`)

Before running: verify the 9 checkpoints this task depends on actually exist.

- [ ] **Step 1: Verify the SP5 checkpoints are on disk**

Run:
```bash
for a in C1 C2 C3; do for s in 42 43 44; do
  f="models/idqn_${a}_corridor_peak_lam05_seed${s}_mg10_s100000.pt"
  [ -f "$f" ] && echo "ok: $f" || echo "MISSING: $f"
done; done
```
Expected: 9 `ok:` lines. If any are `MISSING`, stop — retraining is out of scope for this plan (spec §Scope); report which are missing and why (per SP5's handoff, `models/` is gitignored and has been known to get cleaned between sessions).

- [ ] **Step 2: Run the zero-shot sweep**

Run:
```bash
source venv/bin/activate
python -m analysis.idqn_zeroshot
```
Expected: 9 eval runs (3 scenarios × 3 seeds), then the printed report comparing each shifted scenario's `idqn - green_wave` gap against SP5's +3.09 ± 0.37s in-distribution reference, plus `max_pressure - green_wave` on the same scenario for the "harder for everyone" check. Record every printed number — it goes into the findings doc in Task 5.

- [ ] **Step 3: Verify the output CSV**

Run: `python -c "import pandas as pd; df = pd.read_csv('analysis/idqn_zeroshot.csv'); print(df)"`
Expected: 9 rows, one per (scenario, seed).

- [ ] **Step 4: Commit**

```bash
git add analysis/idqn_zeroshot.csv
git commit -m "data: SP6 IDQN zero-shot eval results, corridor_peak checkpoints on offpeak/tidal/skew"
```

---

## Task 5: Findings doc

**Files:**
- Create: `docs/FINDINGS_2026-08-22-sp6-idqn-demand-shift.md`

- [ ] **Step 1: Write the findings doc**

Using the numbers recorded in Task 4 Step 2, write `docs/FINDINGS_2026-08-22-sp6-idqn-demand-shift.md` covering:
- One paragraph restating the question (does the corridor_peak-trained IDQN policy generalize to demand shapes it never trained on) and why the fixed-plan-retuning framing doesn't apply here (green_wave is demand-blind by construction — see the design spec's Context section).
- A table: scenario | idqn zero-shot delay/trip (mean ± sd) | green_wave delay/trip | max_pressure delay/trip | idqn gap-to-green_wave — one row per shifted scenario plus the `corridor_peak` in-distribution reference row from SP5.
- Per scenario, state plainly whether the gap held, shrank, or widened relative to the +3.09 ± 0.37s in-distribution reference, and whether `max_pressure`'s own delay moved similarly (the "harder for everyone" check from the spec's decision rule).
- A verdict paragraph: does this change the project's consolidation recommendation (`docs/HANDOFF_2026-08-21.md`)? State the open thread either way — if the policy generalized, the natural follow-up is IPPO's equivalent test (blocked on retraining, since no IPPO checkpoints survive on disk) or a full retrain-and-re-evaluate on the shifted scenario; if it overfit, that's a standalone negative finding needing no further compute.

- [ ] **Step 2: Commit**

```bash
git add docs/FINDINGS_2026-08-22-sp6-idqn-demand-shift.md
git commit -m "docs: SP6 findings -- IDQN zero-shot generalization across corridor demand shifts"
```
