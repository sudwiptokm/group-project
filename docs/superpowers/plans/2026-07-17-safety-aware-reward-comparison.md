# Safety-Aware Reward & Two-Stage Comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a vulnerability-weighted composite safety-aware reward (λ-parameterised) to the shared SUMO env, plus peak/off-peak scenarios and a fixed-time baseline in the comparison frame, then run a two-stage algorithm comparison.

**Architecture:** The reward stays the env-level invariant in `env_common.py`; a factory `make_safety_reward_fn(lam)` produces the sumo-rl `reward_fn(ts)` callable. `make_env` gains `scenario` and `lam` args. `train.py`/`tune.py` expose them as CLI flags; `run_experiment.sh` loops SCENARIO × LAMBDA. A fixed-time baseline is emitted into the same eval-CSV frame and folded into `compare.py`.

**Tech Stack:** Python 3.9, sumo-rl 1.4.5, Stable-Baselines3 + sb3-contrib, Optuna, SUMO (eclipse-sumo via pip), pytest (new).

**Spec:** `docs/superpowers/specs/2026-07-17-safety-aware-reward-comparison-design.md`

**Preconditions:** venv active, `SUMO_HOME` set, on branch `feature/safety-aware-reward-comparison`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `env_common.py` (modify) | Vulnerability table, `_vehicle_vuln`, `_safety_penalty`, `_efficiency`, `make_safety_reward_fn`, `SAFETY_SCALE`, `B_THRESH`, `SCENARIO_ROUTES`; `make_env(seed, scenario, lam, gui, out_csv)`. |
| `tests/test_safety_reward.py` (create) | Unit tests for the safety math using a mock traffic signal (no SUMO). |
| `make_scenarios.py` (create) | Generate `traffic_peak.rou.xml` / `traffic_offpeak.rou.xml` by scaling `vehsPerHour`. |
| `traffic_peak.rou.xml`, `traffic_offpeak.rou.xml` (generated) | Peak/off-peak demand variants. |
| `baseline.py` (create) | Run one fixed-time (no-learning) episode per scenario → `logs/eval_fixedtime_<scenario>_seed<s>_*.csv`. |
| `train.py` (modify) | `--scenario`, `--lam` flags; scenario+λ encoded in model + CSV names. |
| `tune.py` (modify) | `--scenario`, `--lam` flags passed into the tuning env. |
| `compare.py` (modify) | Parse (scenario, λ) from CSV names; add fixed-time row; rank per group; safety metrics. |
| `run_experiment.sh` (modify) | `SCENARIO` and `LAMBDA` env-var axes looped through tune→train→eval→compare. |
| `requirements.txt` (modify) | Add `pytest`. |
| `README.md`, `WALKTHROUGH.md`, `PROGRESS_CHECKLIST.md` (modify) | Document reward, scenarios, λ, calibration, protocol. |

**Naming convention (used everywhere):**
- model: `models/<algo>_<scenario>_lam<LAM>_seed<seed>.zip`
- train CSV prefix: `logs/<algo>_<scenario>_lam<LAM>_seed<seed>`
- eval CSV prefix: `logs/eval_<algo>_<scenario>_lam<LAM>_seed<seed>`
- baseline CSV prefix: `logs/eval_fixedtime_<scenario>_seed<seed>`

`<LAM>` is the λ float with the dot removed (e.g. `0.5` → `05`, `1.0` → `10`, `0` → `00`) so filenames stay glob-safe.

---

## Task 1: Test infra + vulnerability weighting

**Files:**
- Modify: `requirements.txt`
- Create: `tests/test_safety_reward.py`
- Modify: `env_common.py`

- [ ] **Step 1: Add pytest to requirements**

Append this line to `requirements.txt`:

```
pytest>=8.0
```

Then install:

Run: `pip install pytest`
Expected: pytest installs successfully.

- [ ] **Step 2: Write the failing test for `_vehicle_vuln`**

Create `tests/test_safety_reward.py`:

```python
"""Unit tests for the safety-aware reward math (no SUMO needed)."""
from types import SimpleNamespace

import pytest

import env_common as ec


def test_vehicle_vuln_exact_types():
    assert ec._vehicle_vuln("moto") == 1.0
    assert ec._vehicle_vuln("auto") == 0.6
    assert ec._vehicle_vuln("car") == 0.3


def test_vehicle_vuln_distribution_suffix():
    # vType ids may carry a distribution suffix like "moto@0"
    assert ec._vehicle_vuln("moto@0") == 1.0
    assert ec._vehicle_vuln("auto@3") == 0.6


def test_vehicle_vuln_unknown_defaults_to_car():
    assert ec._vehicle_vuln("bus") == ec.DEFAULT_VULN == 0.3
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_safety_reward.py::test_vehicle_vuln_exact_types -v`
Expected: FAIL — `AttributeError: module 'env_common' has no attribute '_vehicle_vuln'`.

- [ ] **Step 4: Implement the vulnerability table and lookup**

In `env_common.py`, directly after the PCU block (after line defining `DEFAULT_PCU` / `_vehicle_pcu`), add:

```python
# ----------------------------------------------------------------------------
# Safety-aware reward — vulnerability weights (inverse of crash protection)
# Mirrors the PCU idea: the more exposed the rider, the higher the weight.
# ----------------------------------------------------------------------------
VULNERABILITY = {"moto": 1.0, "auto": 0.6, "car": 0.3}
DEFAULT_VULN = 0.3  # unknown type -> treat as a car (least vulnerable)

B_THRESH = 4.5      # m/s^2 : |deceleration| above this counts as an emergency brake
SAFETY_SCALE = 1.0  # calibration constant; set in Task 9 (see spec section 4)


def _vehicle_vuln(type_id: str) -> float:
    # prefix-match so a distribution suffix (e.g. "moto@0") still resolves
    for name, w in VULNERABILITY.items():
        if type_id == name or type_id.startswith(name):
            return w
    return DEFAULT_VULN
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_safety_reward.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/test_safety_reward.py env_common.py
git commit -m "feat: add vulnerability weighting for safety reward"
```

---

## Task 2: Safety penalty (brake + exposure terms)

**Files:**
- Modify: `env_common.py`
- Modify: `tests/test_safety_reward.py`

- [ ] **Step 1: Write the failing test for `_safety_penalty`**

Add to `tests/test_safety_reward.py`:

```python
def _fake_ts(*, lane_vehicles, accel, types, is_yellow, internal_lanes, internal_vehicles):
    """Build a duck-typed traffic-signal stand-in for _safety_penalty.

    lane_vehicles      : {lane_id: [veh_id, ...]}   approach lanes
    accel              : {veh_id: acceleration}     (negative = braking)
    types              : {veh_id: type_id}
    internal_lanes     : [lane_id, ...]             junction internal lanes
    internal_vehicles  : {lane_id: [veh_id, ...]}
    """
    def get_last_step_vehicle_ids(lane):
        return lane_vehicles.get(lane, internal_vehicles.get(lane, []))

    # getControlledLinks -> [[(in_lane, out_lane, via_lane)], ...]
    links = [[("in", "out", via)] for via in internal_lanes]

    sumo = SimpleNamespace(
        lane=SimpleNamespace(getLastStepVehicleIDs=get_last_step_vehicle_ids),
        vehicle=SimpleNamespace(
            getAcceleration=lambda v: accel[v],
            getTypeID=lambda v: types[v],
        ),
        trafficlight=SimpleNamespace(getControlledLinks=lambda _id: links),
    )
    return SimpleNamespace(sumo=sumo, id="C", lanes=list(lane_vehicles.keys()),
                           is_yellow=is_yellow)


def test_safety_penalty_counts_hard_braking_weighted():
    # one moto braking hard (accel -5 < -4.5) -> weight 1.0 ; one car cruising -> 0
    ts = _fake_ts(
        lane_vehicles={"L1": ["m1", "c1"]},
        accel={"m1": -5.0, "c1": 0.0},
        types={"m1": "moto", "c1": "car"},
        is_yellow=False,
        internal_lanes=[":C_0"],
        internal_vehicles={},
    )
    assert ec._safety_penalty(ts) == pytest.approx(1.0)


def test_safety_penalty_ignores_soft_braking():
    # accel -3 is above -B_THRESH (-4.5) -> not an emergency brake
    ts = _fake_ts(
        lane_vehicles={"L1": ["m1"]},
        accel={"m1": -3.0},
        types={"m1": "moto"},
        is_yellow=False,
        internal_lanes=[":C_0"],
        internal_vehicles={},
    )
    assert ec._safety_penalty(ts) == pytest.approx(0.0)


def test_safety_penalty_exposure_only_when_yellow():
    # an auto sitting on an internal lane; counts only while is_yellow
    common = dict(
        lane_vehicles={"L1": []},
        accel={},
        types={"a1": "auto"},
        internal_lanes=[":C_0"],
        internal_vehicles={":C_0": ["a1"]},
    )
    assert ec._safety_penalty(_fake_ts(is_yellow=False, **common)) == pytest.approx(0.0)
    assert ec._safety_penalty(_fake_ts(is_yellow=True, **common)) == pytest.approx(0.6)


def test_safety_penalty_sums_brake_and_exposure():
    ts = _fake_ts(
        lane_vehicles={"L1": ["m1"]},
        accel={"m1": -6.0},
        types={"m1": "moto", "a1": "auto"},
        is_yellow=True,
        internal_lanes=[":C_0"],
        internal_vehicles={":C_0": ["a1"]},
    )
    # brake: moto 1.0 ; exposure: auto 0.6
    assert ec._safety_penalty(ts) == pytest.approx(1.6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_safety_reward.py -k safety_penalty -v`
Expected: FAIL — `AttributeError: module 'env_common' has no attribute '_safety_penalty'`.

- [ ] **Step 3: Implement `_internal_lanes` and `_safety_penalty`**

In `env_common.py`, after `_vehicle_vuln`, add:

```python
def _internal_lanes(ts) -> list:
    # via (internal junction) lane is the 3rd element of each controlled link
    links = ts.sumo.trafficlight.getControlledLinks(ts.id)
    return list({lk[0][2] for lk in links if lk and lk[0][2]})


def _safety_penalty(ts) -> float:
    """Composite, vulnerability-weighted safety penalty for the current step.

    brake_term    : sum of vulnerability over vehicles braking harder than B_THRESH
    exposure_term : sum of vulnerability over vehicles on internal junction lanes
                    while the phase is yellow / clearing
    """
    sumo = ts.sumo

    brake_term = 0.0
    for lane in ts.lanes:
        for vid in sumo.lane.getLastStepVehicleIDs(lane):
            if sumo.vehicle.getAcceleration(vid) < -B_THRESH:
                brake_term += _vehicle_vuln(sumo.vehicle.getTypeID(vid))

    exposure_term = 0.0
    if ts.is_yellow:
        for lane in _internal_lanes(ts):
            for vid in sumo.lane.getLastStepVehicleIDs(lane):
                exposure_term += _vehicle_vuln(sumo.vehicle.getTypeID(vid))

    return brake_term + exposure_term
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_safety_reward.py -v`
Expected: all passed (7 total).

- [ ] **Step 5: Commit**

```bash
git add env_common.py tests/test_safety_reward.py
git commit -m "feat: composite vulnerability-weighted safety penalty"
```

---

## Task 3: Reward factory `make_safety_reward_fn`

**Files:**
- Modify: `env_common.py`
- Modify: `tests/test_safety_reward.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_safety_reward.py`:

```python
def test_reward_lambda_zero_is_pure_efficiency(monkeypatch):
    monkeypatch.setattr(ec, "_efficiency", lambda ts: 7.0)
    # safety must be ignored entirely at lam=0
    monkeypatch.setattr(ec, "_safety_penalty", lambda ts: 999.0)
    fn = ec.make_safety_reward_fn(0.0)
    assert fn(object()) == pytest.approx(7.0)


def test_reward_subtracts_scaled_safety(monkeypatch):
    monkeypatch.setattr(ec, "_efficiency", lambda ts: 7.0)
    monkeypatch.setattr(ec, "_safety_penalty", lambda ts: 4.0)
    fn = ec.make_safety_reward_fn(0.5, scale=2.0)
    # 7.0 - 0.5 * (4.0 / 2.0) = 6.0
    assert fn(object()) == pytest.approx(6.0)


def test_reward_fn_has_unique_name():
    assert ec.make_safety_reward_fn(1.0).__name__ != ec.make_safety_reward_fn(0.0).__name__
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_safety_reward.py -k reward -v`
Expected: FAIL — `AttributeError: module 'env_common' has no attribute 'make_safety_reward_fn'`.

- [ ] **Step 3: Implement `_efficiency` and `make_safety_reward_fn`**

In `env_common.py`, add after `_safety_penalty`. `TrafficSignal` is already imported (line 17), so no new import is needed:

```python
def _efficiency(ts) -> float:
    # sumo-rl's built-in diff-waiting-time reward; manages ts.last_measure state
    return TrafficSignal._diff_waiting_time_reward(ts)


def make_safety_reward_fn(lam: float, scale: float = None):
    """Return a sumo-rl reward_fn(ts) = diff_waiting_time - lam * safety/scale.

    At lam == 0 the safety term is not even computed (pure efficiency, and an
    exact ablation baseline). `scale` defaults to the calibrated SAFETY_SCALE.
    """
    s = SAFETY_SCALE if scale is None else scale

    def reward_fn(ts):
        eff = _efficiency(ts)
        if lam == 0.0:
            return eff
        return eff - lam * (_safety_penalty(ts) / s)

    reward_fn.__name__ = f"safety_reward_lam{lam}"
    return reward_fn
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_safety_reward.py -v`
Expected: all passed (10 total).

- [ ] **Step 5: Commit**

```bash
git add env_common.py tests/test_safety_reward.py
git commit -m "feat: lambda-parameterised safety-aware reward factory"
```

---

## Task 4: Wire reward + scenario into `make_env`

**Files:**
- Modify: `env_common.py:85-106`

- [ ] **Step 1: Add scenario route map above `make_env`**

In `env_common.py`, immediately above `def make_env(...)`, add:

```python
# route file per demand scenario (generated by make_scenarios.py)
SCENARIO_ROUTES = {
    "base": "traffic.rou.xml",
    "peak": "traffic_peak.rou.xml",
    "offpeak": "traffic_offpeak.rou.xml",
}
```

- [ ] **Step 2: Replace the `make_env` signature and body**

Replace the whole `make_env` function (currently `env_common.py:85-106`) with:

```python
def make_env(seed: int, scenario: str = "base", lam: float = 0.0,
             gui: bool = False, out_csv: str = None) -> SumoEnvironment:
    # vtypes + sublane always; gui-settings only under sumo-gui (plain sumo rejects it)
    extra = "--additional-files vtypes.add.xml --lateral-resolution 0.5"
    if gui:
        extra += " --gui-settings-file gui-settings.xml --start --quit-on-end"
    return SumoEnvironment(
        net_file="intersection.net.xml",
        route_file=SCENARIO_ROUTES[scenario],
        observation_class=PCUObservationFunction,
        use_gui=gui,
        num_seconds=3600,
        delta_time=5,          # seconds between agent decisions
        yellow_time=3,
        min_green=10,
        max_green=60,
        reward_fn=make_safety_reward_fn(lam),   # SAME reward for all agents at a given lam
        single_agent=True,              # one TL -> SB3 single-agent API
        sumo_seed=seed,
        out_csv_name=out_csv,
        sumo_warnings=False,
        additional_sumo_cmd=extra,
    )
```

- [ ] **Step 3: Verify the module still imports and unit tests pass**

Run: `python -c "import env_common; print(env_common.SCENARIO_ROUTES['peak'])" && pytest tests/test_safety_reward.py -q`
Expected: prints `traffic_peak.rou.xml`; all safety tests pass.

- [ ] **Step 4: Commit**

```bash
git add env_common.py
git commit -m "feat: make_env takes scenario + lambda, wires safety reward"
```

---

## Task 5: Scenario route file generator

**Files:**
- Create: `make_scenarios.py`
- Generated: `traffic_peak.rou.xml`, `traffic_offpeak.rou.xml`

- [ ] **Step 1: Write the generator**

Create `make_scenarios.py`:

```python
"""Generate peak / off-peak demand variants from traffic.rou.xml.

Only the vehsPerHour flow rates are scaled; routes, vTypeDistribution and edge
ids are left untouched (edge ids must keep matching intersection.net.xml).
"""
import re

SRC = "traffic.rou.xml"
FACTORS = {"traffic_peak.rou.xml": 1.5, "traffic_offpeak.rou.xml": 0.5}

_FLOW = re.compile(r'vehsPerHour="([0-9.]+)"')


def scale_file(src: str, dst: str, factor: float) -> None:
    with open(src) as fh:
        text = fh.read()

    def repl(m):
        return f'vehsPerHour="{max(1, round(float(m.group(1)) * factor))}"'

    with open(dst, "w") as fh:
        fh.write(_FLOW.sub(repl, text))
    print(f"wrote {dst} (x{factor})")


if __name__ == "__main__":
    for dst, factor in FACTORS.items():
        scale_file(SRC, dst, factor)
```

- [ ] **Step 2: Generate the files**

Run: `python make_scenarios.py`
Expected: prints `wrote traffic_peak.rou.xml (x1.5)` and `wrote traffic_offpeak.rou.xml (x0.5)`.

- [ ] **Step 3: Sanity-check each scenario loads in SUMO**

Run:
```bash
sumo -n intersection.net.xml -r traffic_peak.rou.xml \
  -a vtypes.add.xml --lateral-resolution 0.5 --no-step-log --end 200 2>&1 | tail -5
sumo -n intersection.net.xml -r traffic_offpeak.rou.xml \
  -a vtypes.add.xml --lateral-resolution 0.5 --no-step-log --end 200 2>&1 | tail -5
```
Expected: both run to the 200 s cutoff with no "edge is not known" / route errors.

- [ ] **Step 4: Commit**

```bash
git add make_scenarios.py traffic_peak.rou.xml traffic_offpeak.rou.xml
git commit -m "feat: peak/off-peak demand scenario generator + files"
```

---

## Task 6: Thread `--scenario` / `--lam` through `train.py`

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Read the current arg + naming code**

Run: `grep -n "make_env\|add_argument\|models/\|logs/\|def train\|def evaluate\|args\." train.py`
Expected: shows the two `make_env` calls (lines ~63, ~77), the argparse block, and the model/CSV path strings.

- [ ] **Step 2: Add a filename tag helper near the top of `train.py`**

After the imports in `train.py`, add:

```python
def _tag(scenario: str, lam: float) -> str:
    # lam 0.5 -> "05", 1.0 -> "10", 0 -> "00" ; glob-safe filename fragment
    return f"{scenario}_lam{str(lam).replace('.', '')}"
```

- [ ] **Step 3: Update `train(...)` to take + use scenario/lam**

Change the `def train(...)` signature to include `scenario` and `lam`, and update the `make_env` call and the model/CSV names. The train call currently reads:

```python
    env = make_env(seed=seed, gui=False, out_csv=f"logs/{algo}_seed{seed}")
```

Replace with:

```python
    tag = _tag(scenario, lam)
    env = make_env(seed=seed, scenario=scenario, lam=lam, gui=False,
                   out_csv=f"logs/{algo}_{tag}_seed{seed}")
```

And where the model is saved (the `models/{algo}_seed{seed}.zip` string), replace with:

```python
    model.save(f"models/{algo}_{tag}_seed{seed}")
```

- [ ] **Step 4: Update `evaluate(...)` to take + use scenario/lam**

The evaluate call currently reads:

```python
    env = make_env(seed=seed, gui=gui, out_csv=f"logs/eval_{algo}_seed{seed}")
```

Replace with:

```python
    tag = _tag(scenario, lam)
    env = make_env(seed=seed, scenario=scenario, lam=lam, gui=gui,
                   out_csv=f"logs/eval_{algo}_{tag}_seed{seed}")
```

Add `scenario` and `lam` to the `def evaluate(...)` signature.

- [ ] **Step 5: Add the CLI flags and pass them through**

In the argparse block add:

```python
    p.add_argument("--scenario", default="base", choices=["base", "peak", "offpeak"])
    p.add_argument("--lam", type=float, default=0.0, help="safety-reward weight")
```

Update the two dispatch calls at the bottom to pass `scenario=args.scenario, lam=args.lam` into `train(...)` and `evaluate(...)`.

- [ ] **Step 6: Smoke-test a tiny train run**

Run: `python train.py --algo dqn --scenario offpeak --lam 0.5 --seed 0 --steps 500`
Expected: completes; creates `models/dqn_offpeak_lam05_seed0.zip` and a `logs/dqn_offpeak_lam05_seed0*.csv`.

- [ ] **Step 7: Commit**

```bash
git add train.py
git commit -m "feat: train.py --scenario/--lam flags, scenario+lam in output names"
```

---

## Task 7: Thread `--scenario` / `--lam` through `tune.py`

**Files:**
- Modify: `tune.py`

- [ ] **Step 1: Read the current tune env calls + args**

Run: `grep -n "make_env\|add_argument\|args\.\|def make_objective\|def main" tune.py`
Expected: shows the two `make_env` calls (~44, ~62) and the argparse block.

- [ ] **Step 2: Add scenario/lam args and thread into the objective**

In the argparse block of `tune.py` add:

```python
    p.add_argument("--scenario", default="peak", choices=["base", "peak", "offpeak"])
    p.add_argument("--lam", type=float, default=0.5, help="safety-reward weight for tuning")
```

(Defaults match the Stage-1 reference point: tune on `peak` at λ=0.5.)

- [ ] **Step 3: Pass scenario/lam into both `make_env` calls**

Update `make_objective(...)` (and `_eval_reward` if it builds its own env) so both `make_env` calls pass `scenario=scenario, lam=lam`. The training env call:

```python
        env = make_env(seed=train_seed, scenario=scenario, lam=lam, gui=False, out_csv=None)
```

The eval env call:

```python
    env = make_env(seed=seed, scenario=scenario, lam=lam, gui=False, out_csv=None)
```

Add `scenario` and `lam` parameters to `make_objective(...)` (and any eval helper) and pass `args.scenario, args.lam` from `main()`.

- [ ] **Step 4: Smoke-test a tiny tune run**

Run: `python tune.py --algo dqn --trials 2 --steps 500 --scenario peak --lam 0.5`
Expected: completes 2 trials, writes `params/dqn.json`.

- [ ] **Step 5: Commit**

```bash
git add tune.py
git commit -m "feat: tune.py --scenario/--lam flags (default peak, lam 0.5)"
```

---

## Task 8: Fixed-time baseline runner

**Files:**
- Create: `baseline.py`

- [ ] **Step 1: Write the baseline runner**

Create `baseline.py`:

```python
"""Fixed-time (no-learning) baseline through the SAME env + eval CSV frame.

Steps the shared env with a cyclic phase policy so the fixed-time baseline is
recorded in exactly the metric format compare.py consumes, per scenario.
"""
import argparse
import os

from env_common import make_env


def run_baseline(scenario: str, seed: int) -> str:
    os.makedirs("logs", exist_ok=True)
    csv = f"logs/eval_fixedtime_{scenario}_seed{seed}"
    # lam=0 -> reward term irrelevant here; we never learn, just cycle phases
    env = make_env(seed=seed, scenario=scenario, lam=0.0, gui=False, out_csv=csv)

    obs, _ = env.reset()
    n_actions = env.action_space.n
    action, done = 0, False
    while not done:
        obs, _, terminated, truncated, _ = env.step(action)
        action = (action + 1) % n_actions   # round-robin green phases = fixed-time
        done = terminated or truncated
    env.unwrapped.save_csv(csv, 0) if hasattr(env, "unwrapped") else env.save_csv(csv, 0)
    env.close()
    print(f"baseline written: {csv}")
    return csv


if __name__ == "__main__":
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="base", choices=["base", "peak", "offpeak"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run_baseline(args.scenario, args.seed)
```

- [ ] **Step 2: Verify the env exposes `action_space.n` and `save_csv`**

Run: `grep -n "def save_csv\|action_space" venv/lib/python3.9/site-packages/sumo_rl/environment/env.py | head`
Expected: shows a `save_csv` method and an `action_space` attribute. If `save_csv` lives on the unwrapped env only, the `hasattr` branch in Step 1 already handles it.

- [ ] **Step 3: Run the baseline for one scenario**

Run: `python baseline.py --scenario offpeak --seed 0`
Expected: prints `baseline written: logs/eval_fixedtime_offpeak_seed0`; a matching CSV exists in `logs/`.

- [ ] **Step 4: Commit**

```bash
git add baseline.py
git commit -m "feat: fixed-time baseline runner into eval CSV frame"
```

---

## Task 9: Calibrate `SAFETY_SCALE`

**Files:**
- Modify: `env_common.py`
- Modify: `docs/superpowers/specs/2026-07-17-safety-aware-reward-comparison-design.md` (record the number)

- [ ] **Step 1: Measure the two reward-component magnitudes at λ=0**

Create a throwaway probe and run it:

```bash
python - <<'PY'
import numpy as np, env_common as ec
from env_common import make_env
env = make_env(seed=0, scenario="peak", lam=0.0)
env.reset()
ts = env.unwrapped.traffic_signals[list(env.unwrapped.traffic_signals)[0]]
effs, safs = [], []
done = False
a, n = 0, env.action_space.n
while not done:
    _, _, term, trunc, _ = env.step(a); a = (a + 1) % n
    effs.append(abs(ec._efficiency(ts))); safs.append(ec._safety_penalty(ts))
    done = term or trunc
env.close()
me, ms = np.mean(effs), np.mean(safs)
print(f"mean|eff|={me:.3f} mean_safety={ms:.3f} suggested SAFETY_SCALE={ms/me if me else 0:.3f}")
PY
```
Expected: prints mean magnitudes and a suggested scale = mean_safety / mean|eff|.

> Note: calling `_efficiency(ts)` inside the loop advances `last_measure`; that is
> acceptable here because we only want rough magnitudes for calibration, not the
> exact training signal. The reported ratio is the calibration target.

- [ ] **Step 2: Set `SAFETY_SCALE` to the suggested value**

Edit `env_common.py` and replace `SAFETY_SCALE = 1.0` with the suggested value (round to 2 significant figures, e.g. `SAFETY_SCALE = 0.42`). This makes λ≈1 an "equal-emphasis" point.

- [ ] **Step 3: Record the calibration in the spec**

In the spec file section 4, replace the calibration paragraph's `SAFETY_SCALE` mention with the concrete measured value and the two magnitudes observed (e.g. "measured mean|eff|=X, mean_safety=Y on peak seed 0 → SAFETY_SCALE=Y/X").

- [ ] **Step 4: Re-run unit tests (they pass regardless of the constant)**

Run: `pytest tests/test_safety_reward.py -q`
Expected: all passed (the reward tests inject their own `scale`).

- [ ] **Step 5: Commit**

```bash
git add env_common.py docs/superpowers/specs/2026-07-17-safety-aware-reward-comparison-design.md
git commit -m "chore: calibrate and record SAFETY_SCALE"
```

---

## Task 10: `compare.py` — fixed-time row + group by (scenario, λ)

**Files:**
- Modify: `compare.py`

- [ ] **Step 1: Read the current aggregation code**

Run: `grep -n "def _run_means\|glob\|eval_\|def main\|RANK_KEY\|METRICS\|to_csv\|print" compare.py`
Expected: shows the eval-CSV glob pattern, the metric list, and the print/rank logic.

- [ ] **Step 2: Generalise the glob to parse (algo, scenario, λ)**

`compare.py` currently globs `logs/eval_<algo>_seed*.csv`. Update `_run_means` so it accepts a filename prefix and treats every entity (each real algo AND `fixedtime`) uniformly. Change the glob in `_run_means` to:

```python
    files = glob.glob(os.path.join(logs_dir, f"eval_{entity}_{scenario}_*_seed*.csv"))
```

where `entity` is the algo name or the literal `"fixedtime"`, and `scenario` is passed in. For `fixedtime` the CSV names have no `lam` fragment, so also accept:

```python
    files += glob.glob(os.path.join(logs_dir, f"eval_{entity}_{scenario}_seed*.csv"))
```

- [ ] **Step 3: Build the per-(scenario, λ) comparison in `main`**

Rewrite `main()` to iterate scenarios and λ groups, always including the fixed-time baseline row for that scenario:

```python
def main():
    scenarios = ["peak", "offpeak"]
    lambdas = ["00", "05", "10"]   # filename tags
    rows = []
    for scenario in scenarios:
        # fixed-time baseline (scenario-level, lam-independent)
        base = _run_means("logs", "fixedtime", scenario)
        for lam in lambdas:
            for algo in ["dqn", "qrdqn", "ppo", "a2c"]:
                df = _run_means("logs", algo, scenario, lam=lam)
                if df.empty:
                    continue
                rows.append(_summarise(df, algo=algo, scenario=scenario, lam=lam))
            if not base.empty:
                rows.append(_summarise(base, algo="fixedtime", scenario=scenario, lam=lam))
    out = pd.DataFrame(rows).sort_values(["scenario", "lam", RANK_KEY])
    out.to_csv("logs/comparison.csv", index=False)
    print(out.to_string(index=False))
```

Add a `lam` keyword (default `None`) to `_run_means` that, when set, narrows the glob to `..._lam{lam}_seed*.csv`, and add a small `_summarise(df, algo, scenario, lam)` helper that returns a dict of `{scenario, lam, algo, mean±std per METRIC}`. Keep `RANK_KEY = "system_mean_waiting_time"` (lower is better).

- [ ] **Step 4: Run compare on whatever CSVs exist so far**

Run: `python compare.py`
Expected: prints a table grouped by scenario/λ (may be sparse until the full run); writes `logs/comparison.csv` without error.

- [ ] **Step 5: Commit**

```bash
git add compare.py
git commit -m "feat: compare.py groups by scenario/lambda, adds fixed-time row"
```

---

## Task 11: `run_experiment.sh` — SCENARIO × LAMBDA axes

**Files:**
- Modify: `run_experiment.sh`

- [ ] **Step 1: Read the current driver structure**

Run: `grep -n "ALGOS\|SEEDS\|STEPS\|for \|train.py\|tune.py\|compare.py\|run(" run_experiment.sh`
Expected: shows the existing env-var defaults, the algo/seed loops, and the `run()` wrapper.

- [ ] **Step 2: Add SCENARIO / LAMBDA defaults**

Near the other env-var defaults at the top of `run_experiment.sh`, add:

```bash
SCENARIOS="${SCENARIOS:-peak offpeak}"
LAMBDAS="${LAMBDAS:-0.5}"          # Stage 1 reference; set to "0 0.5 1.0" for Stage 2
TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2 3 4}"
EVAL_SEEDS="${EVAL_SEEDS:-42 43 44 45 46}"
```

- [ ] **Step 3: Wrap train/eval loops in scenario × lambda**

Wrap the existing train and eval loops so each runs per scenario and per λ, and pass the flags through. Training loop body becomes:

```bash
for scenario in $SCENARIOS; do
  for lam in $LAMBDAS; do
    tag="${scenario}_lam${lam//./}"
    for algo in $ALGOS; do
      for s in $TRAIN_SEEDS; do
        [ -f "models/${algo}_${tag}_seed${s}.zip" ] && continue   # resumable
        run python train.py --algo "$algo" --scenario "$scenario" --lam "$lam" \
          --seed "$s" --steps "$STEPS"
      done
    done
  done
done
```

Eval loop body becomes:

```bash
for scenario in $SCENARIOS; do
  python baseline.py --scenario "$scenario" --seed 0 || true   # fixed-time row
  for lam in $LAMBDAS; do
    tag="${scenario}_lam${lam//./}"
    for algo in $ALGOS; do
      ref="models/${algo}_${tag}_seed$(echo $TRAIN_SEEDS | awk '{print $1}').zip"
      for s in $EVAL_SEEDS; do
        [ -f "logs/eval_${algo}_${tag}_seed${s}"*.csv ] 2>/dev/null && continue
        run python train.py --algo "$algo" --eval "$ref" \
          --scenario "$scenario" --lam "$lam" --seed "$s"
      done
    done
  done
done
```

Keep the tuning step per algo unchanged except add `--scenario peak --lam 0.5` (Stage-1 reference) to each `tune.py` invocation.

- [ ] **Step 4: Dry-run a tiny end-to-end pass**

Run:
```bash
STEPS=500 TRAIN_SEEDS="0" EVAL_SEEDS="42" SCENARIOS="offpeak" LAMBDAS="0.5" \
  ALGOS="dqn" ./run_experiment.sh --skip-tune
```
Expected: trains one tiny model, runs the offpeak baseline, evaluates once, runs `compare.py`, writes `logs/comparison.csv`. No hard errors.

- [ ] **Step 5: Commit**

```bash
git add run_experiment.sh
git commit -m "feat: run_experiment.sh scenario x lambda axes + fixed-time baseline"
```

---

## Task 12: Ablation sanity test (safety term actually works)

**Files:**
- Create: `tests/test_ablation_manual.md` (a recorded manual check, not automated — needs SUMO)

- [ ] **Step 1: Run a short λ=0 vs λ=1 comparison on one algo/scenario**

```bash
python train.py --algo dqn --scenario peak --lam 0.0 --seed 0 --steps 20000
python train.py --algo dqn --scenario peak --lam 1.0 --seed 0 --steps 20000
python train.py --algo dqn --eval models/dqn_peak_lam00_seed0.zip --scenario peak --lam 0.0 --seed 42
python train.py --algo dqn --eval models/dqn_peak_lam10_seed0.zip --scenario peak --lam 1.0 --seed 42
```
Expected: four runs complete; two eval CSVs written.

- [ ] **Step 2: Record the directional result**

Create `tests/test_ablation_manual.md` noting the mean weighted emergency-braking (or `system_total_stopped` as a proxy if brakes aren't in the CSV) for λ=0 vs λ=1, and confirm λ=1 is not worse on safety. If it is worse, flag for reward-weight review before the full run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ablation_manual.md
git commit -m "test: record safety-reward ablation directional check"
```

---

## Task 13: Documentation refresh

**Files:**
- Modify: `README.md`, `WALKTHROUGH.md`, `PROGRESS_CHECKLIST.md`

- [ ] **Step 1: Update WALKTHROUGH.md reward section**

In `WALKTHROUGH.md` section 4.1, replace the `reward_fn="diff-waiting-time"` description with the safety-aware reward: composite vulnerability-weighted penalty (moto 1.0 / auto 0.6 / car 0.3), `reward = diff_waiting_time − λ·safety/SAFETY_SCALE`, and note λ is the invariant per comparison stage. Add `make_scenarios.py` and `baseline.py` to the file-by-file list.

- [ ] **Step 2: Update README run instructions**

In `README.md`, add the two-stage protocol commands: `python make_scenarios.py` first; Stage 1 `LAMBDAS="0.5" ./run_experiment.sh`; Stage 2 (winner algo) `ALGOS="<winner>" LAMBDAS="0 0.5 1.0" ./run_experiment.sh --skip-tune`; note `caffeinate` for overnight.

- [ ] **Step 3: Tick completed items in PROGRESS_CHECKLIST.md**

Mark the prerequisite code-change items done; leave the run/analysis items open.

- [ ] **Step 4: Commit**

```bash
git add README.md WALKTHROUGH.md PROGRESS_CHECKLIST.md
git commit -m "docs: safety reward, scenarios, two-stage protocol"
```

---

## Execution note (post-implementation — the actual experiment)

After Tasks 1–13 land and the tiny smoke runs pass, the real compute run is operational, not code:

```bash
python make_scenarios.py                                   # once
# Stage 1 — pick best algo (tunes + trains all algos at lam 0.5)
caffeinate -i ./run_experiment.sh
python compare.py                                          # read the winner
# Stage 2 — sweep lambda on the winner only
caffeinate -i env ALGOS="<winner>" LAMBDAS="0 0.5 1.0" ./run_experiment.sh --skip-tune
python compare.py                                          # tradeoff table -> logs/comparison.csv
```

Reduce `TRAIN_SEEDS`/`EVAL_SEEDS` to 3 each if wall-clock overruns.
