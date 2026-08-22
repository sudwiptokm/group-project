# SP7 Corridor Mid-Episode Incident Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic mid-episode lane-closure mechanism to the corridor env, apply it identically to `green_wave`, `max_pressure`, and zero-shot IDQN, and measure each controller's incident cost (delay under the incident minus its own no-incident `corridor_peak` number) to test whether reactive/learned control earns something specifically under disruption that a demand-blind fixed plan cannot.

**Architecture:** A new pure pair, `corridor_control.incident_lane_id`/`incident_action`, decides *when* to apply/revert a lane closure (mirrors `corridor_control.py`'s existing pure-math discipline — `plan_phase_seconds`, `fixed_time_phase`). `env_common.SafetyLoggingEnv` gains an `incident` constructor param and calls those pure functions from its already-overridden `_sumo_step()`, issuing the actual TraCI calls (`self.sumo.lane.setDisallowed`/`setAllowed`). `make_corridor_env` passes `incident` straight through. `corridor_baseline.py` and `train_corridor_dqn.py` (SP6's already-decoupled `evaluate()`) each get an `--incident` flag wired to the same `INCIDENT` constant, so all three controllers face the identical event with no duplicated logic. `analysis/incident_compare.py` runs the 9 incident eval runs and reports each controller's cost (Δ) against its own no-incident number.

**Tech Stack:** Python 3.11 (venv), PyTorch 2.8.0, sumo-rl, TraCI, pandas, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-22-sp7-corridor-incident-design.md`.

## Global Constraints

- Run everything in the venv: `source venv/bin/activate` first; `SUMO_HOME` comes from the venv's activate script.
- Work directly on `main`.
- **This plan assumes SP6's Task 1 has already landed on `main`** (`train_corridor_dqn.py`'s `_eval_out_stem()`/`evaluate(..., eval_scenario=...)` decoupling) — Task 4 below extends that same function further. If SP6 hasn't been merged yet, merge it first; do not duplicate `_eval_out_stem` here.
- `min_green` is always `10`, `corridor_peak` is the only demand scenario used — matching SP5/SP6's checkpoints and calibrated floor. Never stack the incident on `corridor_tidal`/`corridor_skew` (spec §Scope: one variable under test at a time).
- The incident is one fixed, deterministic event for the whole plan: lane `C1_C2_0` (1 of 2 lanes on the `C1_C2` arterial edge), closed from t=1800s to t=2700s (15 minutes) within the 3600s episode. This constant lives once in `corridor_baseline.py` (`INCIDENT`) and is imported everywhere else it's needed — never redefined.
- Ranking metric is delay per completed trip from tripinfo (`analysis/tripinfo.reduce_tripinfo`'s `trip_time_loss_mean`), never `system_mean_waiting_time`.
- This plan trains nothing. IDQN's incident eval reuses the existing `corridor_peak` checkpoints (SP5/SP6) zero-shot — if any of the 9 `models/idqn_*` files are missing, stop rather than retrain (spec §Scope).
- No `Co-Authored-By` / Claude / Anthropic attribution in any commit message in this plan.
- Do not commit anything under `logs/` or `models/` (both gitignored). `analysis/corridor_sweep.csv` and `analysis/incident_compare.csv` are the two run-output files this plan tracks.

---

## File Structure

| File | Create/Modify | Responsibility |
|------|---------------|-----------------|
| `corridor_control.py` | Modify | Pure `incident_lane_id`/`incident_action` — when to open/close the lane, SUMO-free |
| `tests/test_corridor_control.py` | Modify | Unit tests for the two new pure functions |
| `env_common.py` | Modify | `SafetyLoggingEnv` gains `incident` param + apply/revert logic in `_sumo_step()`; `make_corridor_env` gains passthrough `incident` param |
| `tests/test_corridor_env.py` | Modify | Fast constructor test + slow SUMO smoke test verifying the lane actually closes/reopens |
| `corridor_baseline.py` | Modify | `INCIDENT` constant, `--incident` CLI flag, `run()` gains `incident` param |
| `tests/test_corridor_control.py` (or a new baseline test file) | Modify | Fast test for the CLI/`run()` wiring |
| `train_corridor_dqn.py` | Modify | `evaluate()`/`_eval_out_stem()`/CLI gain `incident` param, importing `INCIDENT` from `corridor_baseline` |
| `tests/test_idqn_hp.py` | Modify | Fast tests for the extended `_eval_out_stem()`/CLI |
| `analysis/incident_compare.py` | Create | Runs the 9 incident eval runs, reduces to delay-per-trip, reports each controller's incident cost (Δ) |
| `tests/test_incident_compare.py` | Create | Unit tests for the cost-computation logic, on synthetic data (no SUMO) |
| `analysis/incident_compare.csv` | Create (data) | Per-controller-per-seed incident eval results |
| `docs/FINDINGS_2026-08-22-sp7-corridor-incident.md` | Create | The written verdict |

---

## Task 1: Pure incident timing logic in `corridor_control.py`

**Files:**
- Modify: `corridor_control.py`
- Test: `tests/test_corridor_control.py`

**Interfaces:**
- Produces: `incident_lane_id(edge_id: str, lane_index: int) -> str`; `incident_action(t: float, start_s: float, duration_s: float, applied: bool) -> Optional[str]` (returns `"apply"`, `"revert"`, or `None`). Both consumed by Task 2's `SafetyLoggingEnv._sumo_step()`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_corridor_control.py`:
```python
def test_incident_lane_id_format():
    assert cc.incident_lane_id("C1_C2", 0) == "C1_C2_0"
    assert cc.incident_lane_id("C1_C2", 1) == "C1_C2_1"


def test_incident_action_applies_at_start():
    assert cc.incident_action(t=1799.0, start_s=1800.0, duration_s=900.0, applied=False) is None
    assert cc.incident_action(t=1800.0, start_s=1800.0, duration_s=900.0, applied=False) == "apply"
    assert cc.incident_action(t=2000.0, start_s=1800.0, duration_s=900.0, applied=False) == "apply"


def test_incident_action_reverts_after_duration():
    assert cc.incident_action(t=2699.0, start_s=1800.0, duration_s=900.0, applied=True) is None
    assert cc.incident_action(t=2700.0, start_s=1800.0, duration_s=900.0, applied=True) == "revert"
    assert cc.incident_action(t=3000.0, start_s=1800.0, duration_s=900.0, applied=True) == "revert"


def test_incident_action_noop_once_settled():
    # already applied and still inside the window -> nothing to do
    assert cc.incident_action(t=2000.0, start_s=1800.0, duration_s=900.0, applied=True) is None
    # already reverted and past the window -> nothing to do
    assert cc.incident_action(t=3000.0, start_s=1800.0, duration_s=900.0, applied=False) is None
```

(`import corridor_control as cc` should already be at the top of `tests/test_corridor_control.py`; add it if not present.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_corridor_control.py -v`
Expected: FAIL — `AttributeError: module 'corridor_control' has no attribute 'incident_lane_id'`.

- [ ] **Step 3: Implement the pure functions**

Add to `corridor_control.py` (after `max_pressure_phase`):
```python
from typing import Optional


def incident_lane_id(edge_id: str, lane_index: int) -> str:
    """SUMO's lane-id convention: '<edge_id>_<lane_index>'."""
    return f"{edge_id}_{lane_index}"


def incident_action(t: float, start_s: float, duration_s: float,
                    applied: bool) -> Optional[str]:
    """What to do to the incident lane at simulation time `t`, given whether
    the closure is already applied.

    Returns 'apply' the first time t reaches start_s (closure not yet
    applied), 'revert' the first time t reaches start_s + duration_s (closure
    still applied), or None otherwise -- including every step after the
    caller has already acted on the returned instruction, so the caller can
    poll this every simulation second without double-applying or
    double-reverting."""
    if not applied and t >= start_s:
        return "apply"
    if applied and t >= start_s + duration_s:
        return "revert"
    return None
```

(Add the `Optional` import to the existing `from typing import Dict, List` line at the top instead, if that's cleaner: `from typing import Dict, List, Optional`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_corridor_control.py -v`
Expected: all PASS (existing tests + 4 new ones).

- [ ] **Step 5: Fast suite regression**

Run: `pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add corridor_control.py tests/test_corridor_control.py
git commit -m "feat: pure incident apply/revert timing logic in corridor_control"
```

---

## Task 2: Wire the incident into `SafetyLoggingEnv`/`make_corridor_env`

**Files:**
- Modify: `env_common.py`
- Test: `tests/test_corridor_env.py`

**Interfaces:**
- Consumes: `corridor_control.incident_lane_id`, `corridor_control.incident_action` (Task 1).
- Produces: `SafetyLoggingEnv(..., incident=None)` (new kwarg: `Optional[Tuple[str, int, float, float]]` = `(edge_id, lane_index, start_s, duration_s)`); `make_corridor_env(..., incident=None)` (same type, passthrough). Consumed by Task 3 (`corridor_baseline.py`) and Task 4 (`train_corridor_dqn.py`).

- [ ] **Step 1: Write the failing fast test (constructor wiring, no SUMO)**

Add to `tests/test_corridor_env.py`:
```python
def test_safety_logging_env_incident_defaults_to_none():
    import inspect
    sig = inspect.signature(SafetyLoggingEnv.__init__)
    assert "incident" in sig.parameters
    assert sig.parameters["incident"].default is None


def test_make_corridor_env_incident_defaults_to_none():
    import inspect
    sig = inspect.signature(make_corridor_env)
    assert "incident" in sig.parameters
    assert sig.parameters["incident"].default is None
```

(Adjust the import at the top of the file to include `SafetyLoggingEnv` and `make_corridor_env` from `env_common` if not already imported.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_corridor_env.py -v -k incident`
Expected: FAIL — `AssertionError` (parameter doesn't exist yet).

- [ ] **Step 3: Implement the wiring**

In `env_common.py`, add `import corridor_control as cc` near the top (with the other project-local imports).

Modify `SafetyLoggingEnv`:
```python
class SafetyLoggingEnv(SumoEnvironment):
    """... (existing docstring, then append:)

    Optionally applies one deterministic mid-episode lane closure (SP7):
    `incident=(edge_id, lane_index, start_s, duration_s)` closes that lane to
    passenger traffic at start_s and reopens it at start_s + duration_s.
    None (default) means no incident -- every existing call site is
    unaffected. Scoped to a single episode per env instance: this plan's
    incident eval runs are all single-episode (baseline.py/train_corridor_dqn
    evaluate()), so the applied/reverted flag is never reset mid-run; a
    training loop that reset() across many episodes with an incident set
    would need that handled too, but that's out of scope here (spec
    docs/superpowers/specs/2026-08-22-sp7-corridor-incident-design.md
    explicitly defers incident-aware training)."""

    def __init__(self, *args, incident=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._safety_window = _SafetyWindow()
        self._incident = incident
        self._incident_applied = False

    def step(self, action):
        # a new decision window begins; the totals read after super().step()
        # cover exactly the seconds this action was in force
        self._safety_window.reset()
        return super().step(action)

    def _sumo_step(self):
        super()._sumo_step()
        # traffic_signals do not exist yet during the reset that starts SUMO
        for ts in getattr(self, "traffic_signals", {}).values():
            self._safety_window.accumulate(ts)
        if self._incident is not None:
            edge_id, lane_index, start_s, duration_s = self._incident
            t = self.sumo.simulation.getTime()
            action = cc.incident_action(t, start_s, duration_s, self._incident_applied)
            if action is not None:
                lane_id = cc.incident_lane_id(edge_id, lane_index)
                if action == "apply":
                    self.sumo.lane.setDisallowed(lane_id, ["passenger"])
                    self._incident_applied = True
                else:  # "revert"
                    self.sumo.lane.setAllowed(lane_id, [])
                    self._incident_applied = False

    # ... (_get_safety_info, _compute_info unchanged)
```

Modify `make_corridor_env`'s signature and body:
```python
def make_corridor_env(seed: int, scenario: str = "corridor_offpeak",
                      lam: float = 0.0, gui: bool = False,
                      out_csv: Optional[str] = None,
                      teleport: int = None, tripinfo: bool = False,
                      min_green: int = None,
                      incident: Optional[tuple] = None) -> "SafetyLoggingEnv":
    """... (existing docstring, then append:)

    `incident`, if given, is forwarded straight to SafetyLoggingEnv -- see
    that class's docstring for the format and scope."""
    # ... (unchanged body up to the return)
    return SafetyLoggingEnv(
        net_file="corridor.net.xml",
        route_file=SCENARIO_ROUTES[scenario],
        observation_class=PCUObservationFunction,
        use_gui=gui,
        num_seconds=int(os.environ.get("EPISODE_SECONDS", "3600")),
        delta_time=CORRIDOR_DELTA_TIME,
        yellow_time=CORRIDOR_YELLOW_TIME,
        min_green=min_green,
        max_green=60,
        reward_fn=make_safety_reward_fn(lam),
        single_agent=False,
        sumo_seed=seed,
        time_to_teleport=teleport,
        out_csv_name=out_csv,
        sumo_warnings=False,
        additional_sumo_cmd=extra,
        incident=incident,
    )
```

- [ ] **Step 4: Run fast test to verify it passes**

Run: `pytest tests/test_corridor_env.py -v -k incident`
Expected: 2 PASS.

- [ ] **Step 5: Write the slow smoke test (SUMO required)**

Add to `tests/test_corridor_env.py`:
```python
@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("SUMO_HOME"), reason="SUMO_HOME not set")
def test_incident_closes_and_reopens_lane(monkeypatch):
    monkeypatch.setenv("EPISODE_SECONDS", "20")
    env = make_corridor_env(seed=0, scenario="corridor_offpeak", min_green=10,
                            incident=("C1_C2", 0, 5.0, 10.0))
    env.reset()
    lane_id = "C1_C2_0"
    seen_closed = False
    done = False
    while not done:
        t = env.sumo.simulation.getTime()
        if 5.0 <= t < 15.0:
            assert "passenger" in env.sumo.lane.getDisallowed(lane_id)
            seen_closed = True
        actions = {i: 0 for i in env.ts_ids}
        _, _, dones, _ = env.step(actions)
        done = dones["__all__"]
    assert seen_closed, "incident window was never reached in a 20s episode"
    assert list(env.sumo.lane.getDisallowed(lane_id)) == []
    env.close()
```

(`os` and `pytest` should already be imported at the top of `tests/test_corridor_env.py`; add `pytestmark`/imports as needed to match the file's existing slow-test convention, e.g. `tests/test_train_corridor_dqn.py`'s `pytestmark = pytest.mark.slow` pattern if the file doesn't already mark slow tests individually.)

- [ ] **Step 6: Run the slow test (SUMO required)**

Run: `pytest tests/test_corridor_env.py -v -m slow -k incident`
Expected: PASS (skipped if `SUMO_HOME` is unset).

- [ ] **Step 7: Fast suite regression**

Run: `pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add env_common.py tests/test_corridor_env.py
git commit -m "feat: wire mid-episode lane-closure incident into SafetyLoggingEnv/make_corridor_env"
```

---

## Task 3: `--incident` flag on `corridor_baseline.py`

**Files:**
- Modify: `corridor_baseline.py`
- Test: `tests/test_corridor_control.py` (or wherever `corridor_baseline` is already covered — check for an existing test file first; if none, add a small new one, `tests/test_corridor_baseline.py`)

**Interfaces:**
- Produces: `INCIDENT = ("C1_C2", 0, 1800.0, 900.0)` (module-level constant, imported by Task 4); `run(scenario, controller, seed, min_green=None, tripinfo=True, incident=False)` (extended signature — `incident` is new).

- [ ] **Step 1: Write the failing fast test**

Create (or add to an existing corridor-baseline test file) `tests/test_corridor_baseline.py`:
```python
"""Unit tests for corridor_baseline.py's incident wiring (no SUMO)."""
import inspect

import corridor_baseline as cb


def test_incident_constant_matches_spec():
    assert cb.INCIDENT == ("C1_C2", 0, 1800.0, 900.0)


def test_run_incident_param_defaults_false():
    sig = inspect.signature(cb.run)
    assert "incident" in sig.parameters
    assert sig.parameters["incident"].default is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_corridor_baseline.py -v`
Expected: FAIL — `AttributeError: module 'corridor_baseline' has no attribute 'INCIDENT'`.

- [ ] **Step 3: Implement the wiring**

In `corridor_baseline.py`, add near the top (after `FREE_FLOW_SPEED`):
```python
# SP7's one fixed incident: close 1 of 2 lanes on the C1_C2 arterial edge for
# 15 minutes starting mid-episode. See
# docs/superpowers/specs/2026-08-22-sp7-corridor-incident-design.md for why
# this edge/timing/duration. Defined once here; train_corridor_dqn.py imports
# it rather than redefining it, so every controller in a comparison faces the
# identical event.
INCIDENT = ("C1_C2", 0, 1800.0, 900.0)
```

Modify `run()`:
```python
def run(scenario: str, controller: str, seed: int, min_green: int = None,
        tripinfo: bool = True, incident: bool = False) -> str:
    """... (existing docstring, then append:)

    incident=True applies INCIDENT (this module's one fixed lane closure) to
    the run; the output CSV name carries an '_incident' fragment so it can
    never be averaged together with a no-incident run of the same
    (controller, scenario, seed, min_green) -- same discipline compare.py's
    _warn_mixed_* guards enforce elsewhere."""
    os.makedirs("logs", exist_ok=True)
    min_green = resolve_min_green(min_green)
    csv = f"logs/eval_{controller}_{scenario}_seed{seed}_mg{min_green}"
    if incident:
        csv += "_incident"
    env = make_corridor_env(seed=seed, scenario=scenario, lam=0.0, out_csv=csv,
                            min_green=min_green, tripinfo=tripinfo,
                            incident=INCIDENT if incident else None)
    env.reset()
    done = False
    while not done:
        if controller == "green_wave":
            actions = green_wave_actions(env)
        else:
            actions = _max_pressure_actions(env)
        _, _, dones, _ = env.step(actions)
        done = dones["__all__"]
    env.save_csv(env.out_csv_name, env.episode)
    env.close()
    out = f"{csv}_conn{env.label}_ep{env.episode}.csv"
    print(f"corridor baseline written: {out}")
    return out
```

In the `if __name__ == "__main__":` CLI block, add:
```python
    p.add_argument("--incident", action="store_true",
                   help=f"apply the SP7 lane-closure incident ({INCIDENT})")
```
And update the final call:
```python
    run(args.scenario, args.controller, args.seed, min_green=args.min_green,
        tripinfo=not args.no_tripinfo, incident=args.incident)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_corridor_baseline.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Fast suite regression**

Run: `pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add corridor_baseline.py tests/test_corridor_baseline.py
git commit -m "feat: --incident flag on corridor_baseline, applies the SP7 lane closure"
```

---

## Task 4: `--incident` flag on `train_corridor_dqn.py`'s eval path

**Files:**
- Modify: `train_corridor_dqn.py`
- Test: `tests/test_idqn_hp.py`

**Interfaces:**
- Consumes: `corridor_baseline.INCIDENT` (Task 3); SP6's `_eval_out_stem(scenario, eval_scenario, lam, seed, min_green, steps)` and `evaluate(..., eval_scenario=None)`.
- Produces: `_eval_out_stem(scenario, eval_scenario, lam, seed, min_green, steps, incident=False)` (extended signature — `incident` is new); `evaluate(..., incident=False)` (extended signature).

This task assumes SP6's Task 1 already landed (Global Constraints) — `_eval_out_stem` and `evaluate`'s `eval_scenario` parameter already exist; this task adds `incident` alongside them.

- [ ] **Step 1: Write the failing fast tests**

Add to `tests/test_idqn_hp.py`:
```python
def test_eval_out_stem_incident_appends_suffix():
    assert tcd._eval_out_stem("corridor_peak", "corridor_peak", 0.5, 42, 10, 100000,
                              incident=True) == \
        "logs/eval_idqn_corridor_peak_lam05_seed42_mg10_s100000_incident"


def test_eval_out_stem_incident_and_zero_shot_combine():
    assert tcd._eval_out_stem("corridor_peak", "corridor_offpeak", 0.5, 42, 10, 100000,
                              incident=True) == \
        "logs/eval_idqn_corridor_peak_lam05_seed42_mg10_s100000_on_corridor_offpeak_incident"


def test_evaluate_incident_defaults_to_false():
    import inspect
    sig = inspect.signature(tcd.evaluate)
    assert "incident" in sig.parameters
    assert sig.parameters["incident"].default is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_idqn_hp.py -v -k incident`
Expected: FAIL — `TypeError: _eval_out_stem() got an unexpected keyword argument 'incident'`.

- [ ] **Step 3: Implement the extension**

In `train_corridor_dqn.py`, add `import corridor_baseline as cb` near the top (with the other project-local imports — `corridor_baseline` doesn't import `train_corridor_dqn`, so this is not circular).

Modify `_eval_out_stem`:
```python
def _eval_out_stem(scenario: str, eval_scenario: str, lam: float, seed: int,
                   min_green: int, steps: int, incident: bool = False) -> str:
    """Eval CSV path stem. '_on_<eval_scenario>' is appended when
    eval_scenario differs from the checkpoint's training scenario (SP6
    zero-shot). '_incident' is appended when the SP7 lane closure was applied
    (docs/superpowers/specs/2026-08-22-sp7-corridor-incident-design.md) --
    both fragments can combine, since SP7's incident eval is itself zero-shot
    against the corridor_peak checkpoints."""
    tag = _tag(scenario, lam, seed, min_green, steps)
    stem = f"logs/eval_idqn_{tag}"
    if eval_scenario != scenario:
        stem += f"_on_{eval_scenario}"
    if incident:
        stem += "_incident"
    return stem
```

Modify `evaluate()`'s signature and body:
```python
def evaluate(scenario: str, lam: float, seed: int, min_green: int, steps: int,
            tripinfo: bool = False, eval_scenario: str = None,
            incident: bool = False) -> str:
    """... (existing docstring, then append:)

    incident=True applies corridor_baseline.INCIDENT to the eval env -- the
    same fixed lane closure every controller in the SP7 comparison faces."""
    os.makedirs("logs", exist_ok=True)
    eval_scenario = eval_scenario or scenario
    out_csv = _eval_out_stem(scenario, eval_scenario, lam, seed, min_green, steps,
                             incident=incident)
    env = make_corridor_env(seed=seed, scenario=eval_scenario, lam=lam,
                            min_green=min_green, out_csv=out_csv, tripinfo=tripinfo,
                            incident=cb.INCIDENT if incident else None)
    # ... (rest of the function body unchanged: load checkpoints, run episode,
    # save_csv, close, return f"{out_csv}_conn{env.label}_ep{env.episode}.csv")
```

In the CLI block, add:
```python
    p.add_argument("--incident", action="store_true",
                   help=f"apply the SP7 lane-closure incident ({cb.INCIDENT})")
```
And update the `--eval` branch's call:
```python
    if args.eval:
        evaluate(args.scenario, args.lam, args.seed, args.min_green, args.steps,
                 tripinfo=args.tripinfo, eval_scenario=args.eval_scenario,
                 incident=args.incident)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_idqn_hp.py -v`
Expected: all PASS.

- [ ] **Step 5: Fast suite regression**

Run: `pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add train_corridor_dqn.py tests/test_idqn_hp.py
git commit -m "feat: --incident flag on train_corridor_dqn's eval path"
```

---

## Task 5: `analysis/incident_compare.py`

**Files:**
- Create: `analysis/incident_compare.py`
- Test: `tests/test_incident_compare.py`

**Interfaces:**
- Consumes: `corridor_baseline.run`, `corridor_baseline.INCIDENT` (Task 3); `train_corridor_dqn._tag`, `train_corridor_dqn.evaluate` (Task 4); `analysis.tripinfo.reduce_tripinfo`; `env_common.tripinfo_path`.
- Produces: `run_baseline(controller, seed, force=False) -> dict`; `run_idqn(seed, force=False) -> dict`; `incident_sweep(seeds, force=False) -> pd.DataFrame`; `no_incident_mean(controller) -> float`; `incident_cost(incident_df, controller, no_incident) -> dict`; `report(incident_df) -> None`.

- [ ] **Step 1: Write the failing tests (pure logic, no SUMO)**

Create `tests/test_incident_compare.py`:
```python
"""Unit tests for analysis/incident_compare.py's cost-computation logic (no SUMO)."""
import pandas as pd
import pytest

ic = pytest.importorskip("analysis.incident_compare")


def _rows(controller, delays, seeds=(42, 43, 44)):
    return pd.DataFrame([
        {"controller": controller, "scenario": "corridor_peak", "seed": s,
         "min_green": 10, "delay_per_trip": d, "trips": 2900}
        for s, d in zip(seeds, delays)
    ])


def test_incident_cost_mean_and_sd():
    df = _rows("green_wave", [15.0, 16.0, 17.0])
    cost = ic.incident_cost(df, "green_wave", no_incident=13.47)
    assert cost["controller"] == "green_wave"
    assert abs(cost["incident_mean"] - 16.0) < 1e-9
    assert abs(cost["cost_mean"] - (16.0 - 13.47)) < 1e-9
    assert cost["n"] == 3


def test_incident_cost_filters_by_controller():
    df = pd.concat([_rows("green_wave", [15.0, 16.0, 17.0]),
                    _rows("max_pressure", [14.0, 14.5, 15.0])])
    cost = ic.incident_cost(df, "max_pressure", no_incident=13.0)
    assert cost["n"] == 3
    assert abs(cost["incident_mean"] - 14.5) < 1e-9


def test_no_incident_mean_reads_corridor_sweep_csv(tmp_path, monkeypatch):
    csv = tmp_path / "corridor_sweep.csv"
    pd.DataFrame([
        {"controller": "green_wave", "scenario": "corridor_peak", "seed": 42,
         "min_green": 10, "delay_per_trip": 13.0, "trips": 2900, "wall_s": 1.0},
        {"controller": "green_wave", "scenario": "corridor_peak", "seed": 43,
         "min_green": 10, "delay_per_trip": 14.0, "trips": 2900, "wall_s": 1.0},
    ]).to_csv(csv, index=False)
    monkeypatch.setattr(ic, "CORRIDOR_SWEEP_CSV", str(csv))
    assert abs(ic.no_incident_mean("green_wave") - 13.5) < 1e-9


def test_no_incident_mean_raises_if_missing(tmp_path, monkeypatch):
    csv = tmp_path / "corridor_sweep.csv"
    pd.DataFrame([{"controller": "green_wave", "scenario": "corridor_tidal", "seed": 42,
                   "min_green": 10, "delay_per_trip": 13.0, "trips": 2900, "wall_s": 1.0}]
                ).to_csv(csv, index=False)
    monkeypatch.setattr(ic, "CORRIDOR_SWEEP_CSV", str(csv))
    with pytest.raises(ValueError):
        ic.no_incident_mean("green_wave")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_incident_compare.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analysis.incident_compare'`.

- [ ] **Step 3: Implement `analysis/incident_compare.py`**

Create `analysis/incident_compare.py`:
```python
"""SP7: mid-episode incident/blockage -- each controller's delay under a
15-minute lane closure on C1_C2, compared to its own no-incident corridor_peak
number. See docs/superpowers/specs/2026-08-22-sp7-corridor-incident-design.md.

    python -m analysis.incident_compare
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd

import corridor_baseline as cb
import train_corridor_dqn as tcd
from analysis.tripinfo import reduce_tripinfo
from env_common import tripinfo_path

CORRIDOR_SWEEP_CSV = os.path.join(REPO, "analysis", "corridor_sweep.csv")
OUT_CSV = os.path.join(REPO, "analysis", "incident_compare.csv")

SCENARIO = "corridor_peak"
SEEDS = (42, 43, 44)
MIN_GREEN = 10
LAM = 0.5
STEPS = 100_000

os.environ.setdefault("TIME_TO_TELEPORT", "300")

# SP5's in-distribution idqn/corridor_peak/no-incident mean delay -- idqn has
# no row in analysis/corridor_sweep.csv (that file only holds the non-RL
# baselines), so its no-incident reference is this disclosed constant instead
# of a CSV lookup (docs/FINDINGS_2026-08-21-sp5-idqn-vs-corrected-bar.md).
INDIST_IDQN_DELAY = 16.56


def run_baseline(controller: str, seed: int, force: bool = False) -> dict:
    """One incident-eval episode for a non-RL baseline. Resumable."""
    stem = f"logs/eval_{controller}_{SCENARIO}_seed{seed}_mg{MIN_GREEN}_incident"
    trip = tripinfo_path(stem)
    if force or not os.path.exists(trip):
        cb.run(SCENARIO, controller, seed, min_green=MIN_GREEN, tripinfo=True,
              incident=True)
    row = reduce_tripinfo(trip)
    return {"controller": controller, "scenario": SCENARIO, "seed": seed,
            "min_green": MIN_GREEN, "delay_per_trip": row["trip_time_loss_mean"],
            "trips": row["trips_completed"]}


def run_idqn(seed: int, force: bool = False) -> dict:
    """One zero-shot incident-eval episode for IDQN, reusing the corridor_peak
    checkpoint. Resumable."""
    stem = tcd._eval_out_stem(SCENARIO, SCENARIO, LAM, seed, MIN_GREEN, STEPS,
                              incident=True)
    trip = tripinfo_path(stem)
    if force or not os.path.exists(trip):
        tcd.evaluate(SCENARIO, LAM, seed, MIN_GREEN, STEPS, tripinfo=True,
                     incident=True)
    row = reduce_tripinfo(trip)
    return {"controller": "idqn", "scenario": SCENARIO, "seed": seed,
            "min_green": MIN_GREEN, "delay_per_trip": row["trip_time_loss_mean"],
            "trips": row["trips_completed"]}


def incident_sweep(seeds=SEEDS, force: bool = False) -> pd.DataFrame:
    rows = []
    for controller in ("green_wave", "max_pressure"):
        for seed in seeds:
            rows.append(run_baseline(controller, seed, force))
            r = rows[-1]
            print(f"[{len(rows)}/9] {controller:13s} seed{seed} incident "
                  f"delay/trip={r['delay_per_trip']:7.1f}s trips={r['trips']:5d}",
                  flush=True)
    for seed in seeds:
        rows.append(run_idqn(seed, force))
        r = rows[-1]
        print(f"[{len(rows)}/9] idqn          seed{seed} incident "
              f"delay/trip={r['delay_per_trip']:7.1f}s trips={r['trips']:5d}",
              flush=True)
    return pd.DataFrame(rows)


def no_incident_mean(controller: str) -> float:
    """Each controller's own no-incident corridor_peak delay -- the number the
    incident cost is measured against. idqn isn't in corridor_sweep.csv (that
    file only holds the non-RL baselines), so it uses INDIST_IDQN_DELAY."""
    if controller == "idqn":
        return INDIST_IDQN_DELAY
    df = pd.read_csv(CORRIDOR_SWEEP_CSV)
    rows = df[(df["controller"] == controller) & (df["scenario"] == SCENARIO) &
              (df["min_green"] == MIN_GREEN)]
    if rows.empty:
        raise ValueError(f"no no-incident rows for {controller}/{SCENARIO}/mg{MIN_GREEN}")
    return float(rows["delay_per_trip"].mean())


def incident_cost(incident_df: pd.DataFrame, controller: str, no_incident: float) -> dict:
    """Mean/sd incident delay minus the controller's own no-incident number --
    the Δ the SP7 decision rule compares across controllers, not raw delay
    (idqn already starts from a higher no-incident baseline than green_wave,
    per SP5, so raw delay alone would misrank this)."""
    rows = incident_df[incident_df["controller"] == controller]
    delta = rows["delay_per_trip"] - no_incident
    return {
        "controller": controller,
        "incident_mean": float(rows["delay_per_trip"].mean()),
        "no_incident": no_incident,
        "cost_mean": float(delta.mean()),
        "cost_sd": float(delta.std(ddof=1)) if len(delta) > 1 else float("nan"),
        "n": int(len(delta)),
    }


def report(incident_df: pd.DataFrame) -> None:
    print(f"\n################ {SCENARIO}, incident "
          f"(C1_C2 lane closed 1800-2700s) ################")
    for controller in ("green_wave", "max_pressure", "idqn"):
        cost = incident_cost(incident_df, controller, no_incident_mean(controller))
        print(f"  {controller:13s} no-incident={cost['no_incident']:6.2f}s  "
              f"incident={cost['incident_mean']:6.2f}s  "
              f"cost(delta)={cost['cost_mean']:+6.2f} +/- {cost['cost_sd']:.2f}s  "
              f"n={cost['n']}")


def main():
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    df = incident_sweep(args.seeds, args.force)
    if os.path.exists(OUT_CSV):
        prior = pd.read_csv(OUT_CSV)
        df = pd.concat([prior, df]).drop_duplicates(subset=["controller", "seed"], keep="last")
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(df)} rows)")
    report(df)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_incident_compare.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Fast suite regression**

Run: `pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add analysis/incident_compare.py tests/test_incident_compare.py
git commit -m "feat: analysis/incident_compare, per-controller incident cost vs no-incident baseline"
```

---

## Task 6: Run the 9 incident evals and build the comparison table

**Files:**
- None (run task, produces `analysis/incident_compare.csv`)

- [ ] **Step 1: Verify the SP5 checkpoints are on disk**

Run:
```bash
for a in C1 C2 C3; do for s in 42 43 44; do
  f="models/idqn_${a}_corridor_peak_lam05_seed${s}_mg10_s100000.pt"
  [ -f "$f" ] && echo "ok: $f" || echo "MISSING: $f"
done; done
```
Expected: 9 `ok:` lines. If any are missing, stop (retraining is out of scope, spec §Scope).

- [ ] **Step 2: Verify the no-incident reference rows exist**

Run: `python -c "
import pandas as pd
df = pd.read_csv('analysis/corridor_sweep.csv')
for c in ('green_wave', 'max_pressure'):
    rows = df[(df.controller==c) & (df.scenario=='corridor_peak') & (df.min_green==10)]
    print(c, len(rows), 'rows')
"`
Expected: both controllers report >0 rows (these already exist from SP4's original corridor calibration — no new sweep needed, unlike SP6's `corridor_offpeak` gap).

- [ ] **Step 3: Run the incident sweep**

Run:
```bash
source venv/bin/activate
python -m analysis.incident_compare
```
Expected: 9 eval runs (`green_wave` × 3 seeds, `max_pressure` × 3 seeds, `idqn` × 3 seeds), then the printed per-controller report: no-incident delay, incident delay, and cost (Δ) with spread. Record every printed number — it goes into the findings doc in Task 7.

- [ ] **Step 4: Verify the output CSV**

Run: `python -c "import pandas as pd; df = pd.read_csv('analysis/incident_compare.csv'); print(df)"`
Expected: 9 rows (3 controllers × 3 seeds).

- [ ] **Step 5: Commit**

```bash
git add analysis/incident_compare.csv
git commit -m "data: SP7 incident eval results, green_wave/max_pressure/idqn under C1_C2 closure"
```

---

## Task 7: Findings doc

**Files:**
- Create: `docs/FINDINGS_2026-08-22-sp7-corridor-incident.md`

- [ ] **Step 1: Write the findings doc**

Using the numbers recorded in Task 6 Step 3, write `docs/FINDINGS_2026-08-22-sp7-corridor-incident.md` covering:
- One paragraph restating the question (does reactive/learned control earn something specifically under an unplanned mid-episode disruption that a demand-blind fixed plan cannot) and the confound this design was built to avoid (`max_pressure` — already reactive, non-learned — stays in the comparison so a positive result can be attributed to *learning* rather than to *reacting in general*).
- A table: controller | no-incident delay/trip | incident delay/trip | incident cost (Δ, mean ± sd) — one row each for `green_wave`, `max_pressure`, `idqn`.
- State plainly which controller's Δ is smallest, and specifically whether IDQN's Δ beats both `green_wave`'s *and* `max_pressure`'s (the interesting claim — learning adds something reacting alone doesn't) or only beats `green_wave`'s while running comparable to or worse than `max_pressure`'s (the expected, non-novel "any reactive controller beats a blind plan under disruption" result) — per the spec's §5 decision rule.
- A verdict paragraph: does this change the project's consolidation recommendation (`docs/HANDOFF_2026-08-21.md`)? If IDQN's Δ is smaller than both baselines', that's real evidence worth a follow-up (incident-aware retraining, spec's deferred open decision). If not, this is a third disconfirming replication of the same shape SP4/SP5/SP6 found, on the one scenario type designed to give learned control its best shot.

- [ ] **Step 2: Commit**

```bash
git add docs/FINDINGS_2026-08-22-sp7-corridor-incident.md
git commit -m "docs: SP7 findings -- corridor mid-episode incident, green_wave vs max_pressure vs idqn"
```
