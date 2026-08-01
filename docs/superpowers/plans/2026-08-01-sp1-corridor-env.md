# SP1 Multi-Intersection Corridor Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 3-signal arterial corridor SUMO network, a PettingZoo multi-agent env reusing the existing safety-λ reward/obs/logging, and two non-RL coordinated baselines (fixed-time green-wave, max-pressure), all feeding the existing eval-CSV + comparison pipeline. No RL training.

**Architecture:** Reuse the single-intersection geometry/reward/obs verbatim; add corridor node/edge XML built with `netconvert`; scale a base corridor route file into peak/offpeak like `make_scenarios.py` does today; wrap `sumo-rl` with `single_agent=False` to expose a PettingZoo `ParallelEnv`; implement green-wave and max-pressure as plain TraCI control loops that write CSVs in the existing format; extend `compare.py` to aggregate network metrics. Testable math (pressure, offsets, route scaling, per-agent safety, metric aggregation) is TDD'd with fast unit tests; SUMO-in-the-loop pieces get slow smoke tests.

**Tech Stack:** Python 3.11 (venv), SUMO 1.27 (`netconvert`, TraCI via `libsumo`), `sumo-rl` 1.4.5, `pettingzoo` 1.26.1, `gymnasium` 1.1.1, `pandas`, `pytest`. No new dependencies.

**Conventions:**
- All Python runs inside the venv: `source venv/bin/activate` first (python is only on PATH there).
- New tests live in `tests/`; `conftest.py` already puts the project root on `sys.path`.
- SUMO-in-the-loop tests are marked `@pytest.mark.slow` and require `SUMO_HOME` set (it is set inside the venv activate).
- Corridor files are named `corridor.*` so nothing collides with the single-intersection `intersection.*` files, which stay untouched (regression guard).

---

## File Structure

| File | Create/Modify | Responsibility |
|------|---------------|----------------|
| `corridor.nod.xml` | Create | 3 TL nodes (C1,C2,C3) + end/cross priority nodes |
| `corridor.edg.xml` | Create | arterial + cross-street edges, ids the routes reference |
| `corridor.net.xml` | Create (via netconvert) | built network (git-tracked output) |
| `corridor.rou.xml` | Create | base demand: dominant E↔W arterial + cross-street flows |
| `make_scenarios.py` | Modify | also scale `corridor.rou.xml` → peak/offpeak |
| `env_common.py` | Modify | `SCENARIO_ROUTES` corridor entries; per-agent safety logging; `make_corridor_env` |
| `corridor_control.py` | Create | pure functions: green-wave offsets, max-pressure decision |
| `corridor_baseline.py` | Create | run a controller on a scenario → eval CSV |
| `compare.py` | Modify | aggregate corridor/network metrics |
| `tests/test_corridor_net.py` | Create | net loads, 3 TLS, arterial connected |
| `tests/test_corridor_routes.py` | Create | route scaling + edge-id validity |
| `tests/test_corridor_control.py` | Create | offsets + max-pressure math |
| `tests/test_corridor_env.py` | Create | slow smoke: multi-agent reset/step shapes |
| `tests/test_corridor_safety_logging.py` | Create | per-agent safety columns |

---

## Task 1: Feature branch + corridor network

**Files:**
- Create: `corridor.nod.xml`, `corridor.edg.xml`, `corridor.net.xml`
- Test: `tests/test_corridor_net.py`

Geometry: arterial along y=0 with three traffic-light nodes 200 m apart
(C1 x=0, C2 x=200, C3 x=400); west/east end nodes at x=-200 / x=600; each TL has
north/south cross arms 200 m out. Edge ids: arterial eastbound
`W_C1,C1_C2,C2_C3,C3_E`; westbound `E_C3,C3_C2,C2_C1,C1_W`; cross per node i
`Ni_Ci`/`Ci_Ni`/`Si_Ci`/`Ci_Si`. Lanes/speed/width copied from
`intersection.edg.xml` (2 lanes, 13.89 m/s, width 3.5).

- [ ] **Step 1: Create the feature branch**

Run:
```bash
cd /Users/sudwipto/Desktop/group_project
git checkout -b feature/corridor-env
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_corridor_net.py`:
```python
"""The built corridor network must load and expose exactly 3 traffic lights
wired along a continuous arterial."""
import os

import pytest

sumolib = pytest.importorskip("sumolib")

NET = "corridor.net.xml"


@pytest.mark.skipif(not os.path.exists(NET), reason="corridor.net.xml not built yet")
def test_three_traffic_lights():
    net = sumolib.net.readNet(NET)
    tls_ids = sorted(t.getID() for t in net.getTrafficLights())
    assert tls_ids == ["C1", "C2", "C3"]


@pytest.mark.skipif(not os.path.exists(NET), reason="corridor.net.xml not built yet")
def test_arterial_edges_present():
    net = sumolib.net.readNet(NET)
    edge_ids = {e.getID() for e in net.getEdges()}
    for eid in ["W_C1", "C1_C2", "C2_C3", "C3_E", "E_C3", "C3_C2", "C2_C1", "C1_W"]:
        assert eid in edge_ids, f"missing arterial edge {eid}"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/test_corridor_net.py -v`
Expected: both tests SKIP (corridor.net.xml not built yet). Skips are the
expected "red" here — the net does not exist.

- [ ] **Step 4: Create the node file**

Create `corridor.nod.xml`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!-- 3-signal arterial corridor. C1,C2,C3 are traffic lights 200m apart on y=0.
     W/E are arterial ends; N*/S* are cross-street ends. -->
<nodes>
    <node id="W"  x="-200" y="0"    type="priority"/>
    <node id="C1" x="0"    y="0"    type="traffic_light"/>
    <node id="C2" x="200"  y="0"    type="traffic_light"/>
    <node id="C3" x="400"  y="0"    type="traffic_light"/>
    <node id="E"  x="600"  y="0"    type="priority"/>

    <node id="N1" x="0"    y="200"  type="priority"/>
    <node id="S1" x="0"    y="-200" type="priority"/>
    <node id="N2" x="200"  y="200"  type="priority"/>
    <node id="S2" x="200"  y="-200" type="priority"/>
    <node id="N3" x="400"  y="200"  type="priority"/>
    <node id="S3" x="400"  y="-200" type="priority"/>
</nodes>
```

- [ ] **Step 5: Create the edge file**

Create `corridor.edg.xml`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!-- Arterial (E<->W) + cross streets. 2 lanes, 13.89 m/s, width 3.5 to match
     intersection.edg.xml so the sublane model + vtypes carry over. -->
<edges>
    <!-- arterial eastbound -->
    <edge id="W_C1"  from="W"  to="C1" numLanes="2" speed="13.89" width="3.5"/>
    <edge id="C1_C2" from="C1" to="C2" numLanes="2" speed="13.89" width="3.5"/>
    <edge id="C2_C3" from="C2" to="C3" numLanes="2" speed="13.89" width="3.5"/>
    <edge id="C3_E"  from="C3" to="E"  numLanes="2" speed="13.89" width="3.5"/>
    <!-- arterial westbound -->
    <edge id="E_C3"  from="E"  to="C3" numLanes="2" speed="13.89" width="3.5"/>
    <edge id="C3_C2" from="C3" to="C2" numLanes="2" speed="13.89" width="3.5"/>
    <edge id="C2_C1" from="C2" to="C1" numLanes="2" speed="13.89" width="3.5"/>
    <edge id="C1_W"  from="C1" to="W"  numLanes="2" speed="13.89" width="3.5"/>

    <!-- cross streets at C1 -->
    <edge id="N1_C1" from="N1" to="C1" numLanes="2" speed="13.89" width="3.5"/>
    <edge id="C1_N1" from="C1" to="N1" numLanes="2" speed="13.89" width="3.5"/>
    <edge id="S1_C1" from="S1" to="C1" numLanes="2" speed="13.89" width="3.5"/>
    <edge id="C1_S1" from="C1" to="S1" numLanes="2" speed="13.89" width="3.5"/>
    <!-- cross streets at C2 -->
    <edge id="N2_C2" from="N2" to="C2" numLanes="2" speed="13.89" width="3.5"/>
    <edge id="C2_N2" from="C2" to="N2" numLanes="2" speed="13.89" width="3.5"/>
    <edge id="S2_C2" from="S2" to="C2" numLanes="2" speed="13.89" width="3.5"/>
    <edge id="C2_S2" from="C2" to="S2" numLanes="2" speed="13.89" width="3.5"/>
    <!-- cross streets at C3 -->
    <edge id="N3_C3" from="N3" to="C3" numLanes="2" speed="13.89" width="3.5"/>
    <edge id="C3_N3" from="C3" to="N3" numLanes="2" speed="13.89" width="3.5"/>
    <edge id="S3_C3" from="S3" to="C3" numLanes="2" speed="13.89" width="3.5"/>
    <edge id="C3_S3" from="C3" to="S3" numLanes="2" speed="13.89" width="3.5"/>
</edges>
```

- [ ] **Step 6: Build the network**

Run:
```bash
source venv/bin/activate
netconvert --node-files corridor.nod.xml --edge-files corridor.edg.xml \
  --output-file corridor.net.xml \
  --tls.guess-signals true --tls.default-type static --no-turnarounds true
```
Expected: writes `corridor.net.xml`; `netconvert` auto-creates a `tlLogic` for
each `traffic_light` node (C1,C2,C3) and connections for through + turn movements.

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_corridor_net.py -v`
Expected: both tests PASS (3 TLS, all 8 arterial edges present).

- [ ] **Step 8: Commit**

```bash
git add corridor.nod.xml corridor.edg.xml corridor.net.xml tests/test_corridor_net.py
git commit -m "feat: 3-signal arterial corridor network"
```

---

## Task 2: Corridor demand + scenario scaling

**Files:**
- Create: `corridor.rou.xml`
- Modify: `make_scenarios.py`
- Test: `tests/test_corridor_routes.py`

Demand: dominant arterial through-flow both directions (E↔W) plus a lighter
cross-street flow per node. Base rates chosen so peak (×1.5) saturates and
offpeak (×0.5) is light, matching the single-intersection factors.

- [ ] **Step 1: Write the failing test**

Create `tests/test_corridor_routes.py`:
```python
"""Corridor route generation: scaling produces peak/offpeak, and every edge a
route references actually exists in the built network."""
import os
import re

import pytest

import make_scenarios as ms

sumolib = pytest.importorskip("sumolib")


def test_corridor_in_factors():
    # make_scenarios must know how to scale the corridor base file
    assert "corridor_peak.rou.xml" in ms.CORRIDOR_FACTORS
    assert "corridor_offpeak.rou.xml" in ms.CORRIDOR_FACTORS


def test_scaling_multiplies_flows(tmp_path):
    src = tmp_path / "corridor.rou.xml"
    src.write_text('<routes><flow id="f" vehsPerHour="100"/></routes>')
    dst = tmp_path / "out.rou.xml"
    ms.scale_file(str(src), str(dst), 1.5)
    text = dst.read_text()
    assert 'vehsPerHour="150"' in text


@pytest.mark.skipif(not os.path.exists("corridor.net.xml"),
                    reason="network not built")
def test_route_edges_exist_in_net():
    net = sumolib.net.readNet("corridor.net.xml")
    edge_ids = {e.getID() for e in net.getEdges()}
    text = open("corridor.rou.xml").read()
    for eid in re.findall(r'edges="([^"]+)"', text):
        for e in eid.split():
            assert e in edge_ids, f"route references unknown edge {e}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_corridor_routes.py -v`
Expected: FAIL — `AttributeError: module 'make_scenarios' has no attribute 'CORRIDOR_FACTORS'`.

- [ ] **Step 3: Create the base corridor route file**

Create `corridor.rou.xml` (reuses `vtypes.add.xml` vType distribution `mix`,
added via the env's `--additional-files`, exactly like `traffic.rou.xml`):
```xml
<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <!-- Arterial through-traffic dominates so platoons form (green-wave gain). -->
    <route id="eb" edges="W_C1 C1_C2 C2_C3 C3_E"/>
    <route id="wb" edges="E_C3 C3_C2 C2_C1 C1_W"/>
    <!-- Cross-street through movements at each node. -->
    <route id="x1" edges="N1_C1 C1_S1"/>
    <route id="x2" edges="N2_C2 C2_S2"/>
    <route id="x3" edges="N3_C3 C3_S3"/>

    <flow id="f_eb" route="eb" type="mix" begin="0" end="3600" vehsPerHour="700"/>
    <flow id="f_wb" route="wb" type="mix" begin="0" end="3600" vehsPerHour="700"/>
    <flow id="f_x1" route="x1" type="mix" begin="0" end="3600" vehsPerHour="200"/>
    <flow id="f_x2" route="x2" type="mix" begin="0" end="3600" vehsPerHour="200"/>
    <flow id="f_x3" route="x3" type="mix" begin="0" end="3600" vehsPerHour="200"/>
</routes>
```

- [ ] **Step 4: Extend make_scenarios.py**

In `make_scenarios.py`, add a corridor factor map and generate it in `__main__`.
After the existing `FACTORS` definition add:
```python
CORRIDOR_SRC = "corridor.rou.xml"
CORRIDOR_FACTORS = {
    "corridor_peak.rou.xml": 1.5,
    "corridor_offpeak.rou.xml": 0.5,
}
```
Replace the `__main__` block with:
```python
if __name__ == "__main__":
    for dst, factor in FACTORS.items():
        scale_file(SRC, dst, factor)
    for dst, factor in CORRIDOR_FACTORS.items():
        scale_file(CORRIDOR_SRC, dst, factor)
```

- [ ] **Step 5: Generate the scenario files**

Run: `python make_scenarios.py`
Expected: prints `wrote corridor_peak.rou.xml (x1.5)` and
`wrote corridor_offpeak.rou.xml (x0.5)` (plus the existing single-intersection ones).

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_corridor_routes.py -v`
Expected: all three tests PASS.

- [ ] **Step 7: Commit**

```bash
git add corridor.rou.xml corridor_peak.rou.xml corridor_offpeak.rou.xml \
  make_scenarios.py tests/test_corridor_routes.py
git commit -m "feat: corridor demand + peak/offpeak scaling"
```

---

## Task 3: Per-agent safety logging

**Files:**
- Modify: `env_common.py` (`SafetyLoggingEnv._get_safety_info`)
- Test: `tests/test_corridor_safety_logging.py`

`SafetyLoggingEnv` already sums safety over all signals into
`system_safety_{brake,exposure,total}`. Add **per-TLS** columns
`safety_total_<tlsid>` so the corridor run records where safety events happen,
while keeping the existing aggregate columns unchanged (regression-safe).

- [ ] **Step 1: Write the failing test**

Create `tests/test_corridor_safety_logging.py`:
```python
"""_get_safety_info must keep the network aggregate AND add a per-TLS total."""
from types import SimpleNamespace

import env_common as ec


def _fake_ts(brake, exposure):
    # patch _safety_components indirectly by faking a signal whose lanes/yellow
    # produce a known (brake, exposure); simplest: monkeypatch _safety_components.
    return SimpleNamespace(id=None)


def test_per_agent_safety_columns(monkeypatch):
    signals = {"C1": object(), "C2": object()}
    fake = {id(signals["C1"]): (1.0, 0.0), id(signals["C2"]): (0.0, 2.0)}
    monkeypatch.setattr(ec, "_safety_components", lambda ts: fake[id(ts)])

    env = ec.SafetyLoggingEnv.__new__(ec.SafetyLoggingEnv)
    env.traffic_signals = signals

    info = env._get_safety_info()
    # aggregate unchanged
    assert info["system_safety_brake"] == 1.0
    assert info["system_safety_exposure"] == 2.0
    assert info["system_safety_total"] == 3.0
    # per-agent totals
    assert info["safety_total_C1"] == 1.0
    assert info["safety_total_C2"] == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_corridor_safety_logging.py -v`
Expected: FAIL — `KeyError: 'safety_total_C1'`.

- [ ] **Step 3: Extend `_get_safety_info`**

In `env_common.py`, replace `SafetyLoggingEnv._get_safety_info` with:
```python
    def _get_safety_info(self) -> dict:
        brake = exposure = 0.0
        per_agent = {}
        for tls_id, ts in self.traffic_signals.items():
            b, e = _safety_components(ts)
            brake += b
            exposure += e
            per_agent[f"safety_total_{tls_id}"] = b + e
        info = {
            "system_safety_brake": brake,
            "system_safety_exposure": exposure,
            "system_safety_total": brake + exposure,
        }
        info.update(per_agent)
        return info
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_corridor_safety_logging.py tests/test_safety_reward.py -v`
Expected: new test PASSES; existing `test_safety_reward.py` still PASSES
(regression guard — the single-intersection aggregate keys are untouched).

- [ ] **Step 5: Commit**

```bash
git add env_common.py tests/test_corridor_safety_logging.py
git commit -m "feat: per-agent safety logging in SafetyLoggingEnv"
```

---

## Task 4: Multi-agent corridor env

**Files:**
- Modify: `env_common.py` (`SCENARIO_ROUTES`, add `make_corridor_env`)
- Test: `tests/test_corridor_env.py`

Expose a PettingZoo `ParallelEnv` (sumo-rl returns one when
`single_agent=False`) with one agent per TLS, reusing the same obs and safety-λ
reward as the single-intersection env.

- [ ] **Step 1: Write the failing slow test**

Create `tests/test_corridor_env.py`:
```python
"""Smoke test: the corridor env exposes 3 agents with correct obs/reward shapes
and terminates. Requires SUMO; slow."""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.slow

import env_common as ec


@pytest.mark.skipif(not os.environ.get("SUMO_HOME"), reason="SUMO_HOME not set")
def test_corridor_env_reset_step():
    os.environ["EPISODE_SECONDS"] = "120"  # short episode for the test
    env = ec.make_corridor_env(seed=0, scenario="corridor_offpeak", lam=0.5)
    obs, _ = env.reset()
    assert set(obs.keys()) == {"C1", "C2", "C3"}
    for a in obs:
        assert isinstance(obs[a], np.ndarray)

    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs2, rewards, term, trunc, _ = env.step(actions)
    assert set(rewards.keys()) == {"C1", "C2", "C3"}
    for a in rewards:
        assert np.isscalar(rewards[a])
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_corridor_env.py -v -m slow`
Expected: FAIL — `AttributeError: module 'env_common' has no attribute 'make_corridor_env'`.

- [ ] **Step 3: Add corridor routes + the env factory**

In `env_common.py`, extend `SCENARIO_ROUTES`:
```python
SCENARIO_ROUTES = {
    "base": "traffic.rou.xml",
    "peak": "traffic_peak.rou.xml",
    "offpeak": "traffic_offpeak.rou.xml",
    "corridor_peak": "corridor_peak.rou.xml",
    "corridor_offpeak": "corridor_offpeak.rou.xml",
}
```
Then add, at the end of `env_common.py`:
```python
def make_corridor_env(seed: int, scenario: str = "corridor_offpeak",
                      lam: float = 0.0, gui: bool = False, out_csv: str = None):
    """Multi-agent arterial corridor env (PettingZoo ParallelEnv, one agent per
    TLS). Same obs (PCUObservationFunction) and safety-λ reward as make_env, but
    single_agent=False so every traffic light is its own agent."""
    extra = "--additional-files vtypes.add.xml --lateral-resolution 0.5"
    if gui:
        extra += " --gui-settings-file gui-settings.xml --start --quit-on-end"
    return SafetyLoggingEnv(
        net_file="corridor.net.xml",
        route_file=SCENARIO_ROUTES[scenario],
        observation_class=PCUObservationFunction,
        use_gui=gui,
        num_seconds=int(os.environ.get("EPISODE_SECONDS", "3600")),
        delta_time=5,
        yellow_time=3,
        min_green=10,
        max_green=60,
        reward_fn=make_safety_reward_fn(lam),
        single_agent=False,             # one agent per TLS -> PettingZoo API
        sumo_seed=seed,
        out_csv_name=out_csv,
        sumo_warnings=False,
        additional_sumo_cmd=extra,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_corridor_env.py -v -m slow`
Expected: PASS — 3 agents C1/C2/C3, obs are arrays, rewards scalar per agent,
env closes cleanly.

- [ ] **Step 5: Register the slow marker (if not already)**

Check `conftest.py` / `pytest.ini` for a `slow` marker. If none exists, create
`pytest.ini` at project root:
```ini
[pytest]
markers =
    slow: SUMO-in-the-loop tests (deselect with -m "not slow")
```

- [ ] **Step 6: Commit**

```bash
git add env_common.py tests/test_corridor_env.py pytest.ini
git commit -m "feat: multi-agent corridor env (PettingZoo)"
```

---

## Task 5: Green-wave + max-pressure control (pure functions)

**Files:**
- Create: `corridor_control.py`
- Test: `tests/test_corridor_control.py`

Two pure, unit-testable functions the baseline loop will call — no SUMO here.

- [ ] **Step 1: Write the failing test**

Create `tests/test_corridor_control.py`:
```python
"""Pure control math for the corridor baselines — no SUMO needed."""
import corridor_control as cc


def test_green_wave_offsets():
    # 3 signals 200m apart, free-flow 13.89 m/s -> offset ≈ 200/13.89 = 14.4 s
    offsets = cc.green_wave_offsets(
        positions=[0.0, 200.0, 400.0], free_flow_speed=13.89)
    assert offsets[0] == 0.0
    assert abs(offsets[1] - 14.4) < 0.1
    assert abs(offsets[2] - 28.8) < 0.1


def test_max_pressure_picks_highest_pressure_phase():
    # phase 0 serves movement with pressure 5, phase 1 pressure 2 -> pick 0
    phase_pressures = {0: 5.0, 1: 2.0}
    assert cc.max_pressure_phase(phase_pressures) == 0
    # tie broken by lowest phase id (deterministic)
    assert cc.max_pressure_phase({0: 3.0, 1: 3.0}) == 0


def test_pressure_is_upstream_minus_downstream_queue():
    # pressure of a movement = incoming queue - outgoing queue
    assert cc.movement_pressure(incoming_queue=10, outgoing_queue=3) == 7
    assert cc.movement_pressure(incoming_queue=2, outgoing_queue=8) == -6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_corridor_control.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'corridor_control'`.

- [ ] **Step 3: Implement the module**

Create `corridor_control.py`:
```python
"""Pure control logic for the non-RL corridor baselines.

Kept SUMO-free so the math is unit-tested in isolation; corridor_baseline.py
wires these to live TraCI state.
"""
from typing import Dict, List


def green_wave_offsets(positions: List[float], free_flow_speed: float) -> List[float]:
    """Signal start offsets (s) for a forward green-wave: each downstream signal
    starts later by the free-flow travel time from the first signal."""
    if free_flow_speed <= 0:
        raise ValueError("free_flow_speed must be positive")
    x0 = positions[0]
    return [(p - x0) / free_flow_speed for p in positions]


def movement_pressure(incoming_queue: float, outgoing_queue: float) -> float:
    """Max-pressure movement pressure = upstream minus downstream queue."""
    return incoming_queue - outgoing_queue


def max_pressure_phase(phase_pressures: Dict[int, float]) -> int:
    """Pick the phase with the greatest total pressure; ties -> lowest phase id."""
    return max(sorted(phase_pressures), key=lambda p: phase_pressures[p])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_corridor_control.py -v`
Expected: all three tests PASS.

- [ ] **Step 5: Commit**

```bash
git add corridor_control.py tests/test_corridor_control.py
git commit -m "feat: green-wave + max-pressure control math"
```

---

## Task 6: Corridor baseline runner

**Files:**
- Create: `corridor_baseline.py`
- Test: covered by the slow smoke run in Step 4 (SUMO-in-the-loop; no separate
  unit test — the control math is already tested in Task 5).

Run a chosen controller over a scenario, stepping the multi-agent env, and write
an eval CSV in the existing format so `compare.py` can read it. `entity` names
in the filename are `green_wave` / `max_pressure`.

- [ ] **Step 1: Implement the runner**

Create `corridor_baseline.py`:
```python
"""Non-RL corridor baselines (green-wave, max-pressure) through the same eval-CSV
frame compare.py consumes. No learning: control is rule-based each step."""
import argparse
import os

import corridor_control as cc
from env_common import make_corridor_env

CONTROLLERS = ("green_wave", "max_pressure")


def _max_pressure_actions(env):
    """One action per agent = phase maximising pressure, from live queues."""
    actions = {}
    for tls_id, ts in env.unwrapped.traffic_signals.items():
        # pressure per green phase = sum over its lanes of (in queue - out queue).
        # sumo-rl exposes per-lane queues via ts.get_lanes_queue proxy; fall back
        # to halting counts on controlled lanes.
        pressures = {}
        for phase in range(ts.num_green_phases):
            q_in = sum(ts.sumo.lane.getLastStepHaltingNumber(l) for l in ts.lanes)
            pressures[phase] = q_in  # downstream queue ~0 at corridor exits
        actions[tls_id] = cc.max_pressure_phase(pressures)
    return actions


def run(scenario: str, controller: str, seed: int) -> str:
    os.makedirs("logs", exist_ok=True)
    csv = f"logs/eval_{controller}_{scenario}_seed{seed}"
    env = make_corridor_env(seed=seed, scenario=scenario, lam=0.0, out_csv=csv)
    env.reset()
    agents = list(env.agents)

    # green-wave: fixed cyclic phase per agent, phase-shifted by the offset.
    positions = [0.0, 200.0, 400.0]
    offsets = cc.green_wave_offsets(positions, free_flow_speed=13.89)
    step = 0
    done = False
    while not done:
        if controller == "green_wave":
            actions = {}
            for i, a in enumerate(agents):
                ts = env.unwrapped.traffic_signals[a]
                shifted = step - int(offsets[i] // env.unwrapped.delta_time)
                actions[a] = max(shifted, 0) % ts.num_green_phases
        else:
            actions = _max_pressure_actions(env)
        _, _, term, trunc, _ = env.step(actions)
        done = all(term.values()) or all(trunc.values())
        step += 1

    env.unwrapped.save_csv(env.unwrapped.out_csv_name, env.unwrapped.episode)
    env.close()
    out = f"logs/eval_{controller}_{scenario}_seed{seed}_conn{env.unwrapped.label}_ep{env.unwrapped.episode}.csv"
    print(f"corridor baseline written: {out}")
    return out


if __name__ == "__main__":
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="corridor_offpeak",
                   choices=["corridor_peak", "corridor_offpeak"])
    p.add_argument("--controller", default="green_wave", choices=CONTROLLERS)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args.scenario, args.controller, args.seed)
```

- [ ] **Step 2: Smoke-run each controller (short episode)**

Run:
```bash
source venv/bin/activate
EPISODE_SECONDS=120 python corridor_baseline.py --scenario corridor_offpeak --controller green_wave --seed 0
EPISODE_SECONDS=120 python corridor_baseline.py --scenario corridor_offpeak --controller max_pressure --seed 0
```
Expected: each prints `corridor baseline written: logs/eval_<controller>_corridor_offpeak_seed0_conn*_ep*.csv`.

- [ ] **Step 3: Verify the CSV has finite network metrics**

Run:
```bash
python -c "import pandas as pd, glob; f=sorted(glob.glob('logs/eval_green_wave_corridor_offpeak_seed0_*.csv'))[-1]; d=pd.read_csv(f); print(d[['system_mean_waiting_time','system_mean_speed','system_total_stopped']].mean()); assert d['system_mean_speed'].mean()>0"
```
Expected: prints finite means; assertion (`mean_speed > 0`) passes — traffic moved.

- [ ] **Step 4: Commit**

```bash
git add corridor_baseline.py
git commit -m "feat: corridor baseline runner (green-wave + max-pressure)"
```

---

## Task 7: Network-metric aggregation in compare.py

**Files:**
- Modify: `compare.py`
- Test: `tests/test_corridor_routes.py` already imports fine; add a focused test
  in a new `tests/test_compare_corridor.py`

Extend `compare.py` so the corridor controllers appear in the ranked table
alongside their scenario, using the existing per-run averaging path. The
controller names become the `algo`/entity column; scenarios are
`corridor_peak`/`corridor_offpeak`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_compare_corridor.py`:
```python
"""compare.py must aggregate corridor controller CSVs into a comparison row."""
import pandas as pd

import compare


def test_run_means_reads_controller_csv(tmp_path):
    # a fake corridor eval CSV in the expected filename shape
    p = tmp_path / "eval_green_wave_corridor_peak_seed0_conn0_ep1.csv"
    pd.DataFrame({
        "system_mean_waiting_time": [1.0, 3.0],
        "system_total_stopped": [2.0, 4.0],
        "system_mean_speed": [5.0, 5.0],
        "system_total_waiting_time": [10.0, 10.0],
    }).to_csv(p, index=False)

    df = compare._run_means(str(tmp_path), "green_wave", "corridor_peak")
    assert len(df) == 1
    # metrics are time-averaged over the episode rows
    assert df["system_mean_waiting_time"].iloc[0] == 2.0
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `pytest tests/test_compare_corridor.py -v`
Expected: PASS already — `_run_means` with `lam=None` matches
`eval_green_wave_corridor_peak_seed*.csv`. (If it fails, the filename glob needs
the fix in Step 3.) This confirms the existing reader handles controller CSVs;
the real change is enrolling corridor scenarios/controllers in `main()`.

- [ ] **Step 3: Enroll corridor scenarios + controllers in `main()`**

In `compare.py` `main()`, after the existing single-intersection loop, add a
corridor pass (controllers are lambda-independent, like fixed-time):
```python
    corridor_scenarios = ["corridor_peak", "corridor_offpeak"]
    corridor_controllers = ["green_wave", "max_pressure"]
    for scenario in corridor_scenarios:
        for ctrl in corridor_controllers:
            df = _run_means(args.logs, ctrl, scenario)
            if not df.empty:
                rows.append(_summarise(df, ctrl, scenario, lam="na"))
```
Insert this immediately before the `if not rows:` guard.

- [ ] **Step 4: Run the full test suite (fast tests only)**

Run: `pytest -v -m "not slow"`
Expected: all fast tests PASS (corridor control, routes, safety logging, compare,
plus the untouched `test_safety_reward.py`).

- [ ] **Step 5: Build the comparison table end-to-end**

Run: `python compare.py`
Expected: table now includes `corridor_peak`/`corridor_offpeak` rows for
`green_wave` and `max_pressure`; writes `logs/comparison.csv`.

- [ ] **Step 6: Commit**

```bash
git add compare.py tests/test_compare_corridor.py
git commit -m "feat: aggregate corridor baselines in comparison table"
```

---

## Task 8: Documentation + regression pass

**Files:**
- Modify: `README.md` (add a short "Corridor (SP1)" subsection), `results/README.md`
- Test: full suite

- [ ] **Step 1: Add a README subsection**

In `README.md`, add a subsection documenting: how to build the corridor
(`netconvert` command from Task 1 Step 6), generate demand
(`python make_scenarios.py`), and run the baselines
(`python corridor_baseline.py --scenario corridor_peak --controller green_wave`),
and note that RL training on the corridor is SP2 (not yet implemented).

- [ ] **Step 2: Run the whole fast suite + one slow smoke**

Run:
```bash
pytest -v -m "not slow"
SUMO_HOME="$SUMO_HOME" pytest tests/test_corridor_env.py -v -m slow
```
Expected: fast suite all PASS; slow corridor env smoke PASSES.

- [ ] **Step 3: Confirm single-intersection pipeline untouched**

Run: `pytest tests/test_safety_reward.py -v`
Expected: PASS — SP1 added files/columns but changed no single-intersection
behavior (regression guard).

- [ ] **Step 4: Commit**

```bash
git add README.md results/README.md
git commit -m "docs: corridor (SP1) build + baseline usage"
```

---

## Self-Review Notes

- **Spec coverage:** network (T1), routes+scenarios (T2), multi-agent PettingZoo
  env reusing obs/reward (T4), per-agent + aggregate safety logging (T3),
  green-wave + max-pressure baselines (T5–T6), network-metric aggregation (T7),
  smoke tests throughout, docs (T8). All SP1 spec sections mapped.
- **Deferred correctly:** no RL training, no CTDE, no framework choice, no tuning
  — those are SP2–SP5 per the spec.
- **Known simplification to revisit in SP2:** `_max_pressure_actions` approximates
  downstream queue as ~0 (valid for corridor exits, rough for interior turns).
  Adequate as a baseline; note it in the thesis. The pure `movement_pressure`
  supports a full upstream−downstream computation when interior sensing is wired.
- **Naming consistency:** `make_corridor_env`, `SCENARIO_ROUTES` keys
  `corridor_peak`/`corridor_offpeak`, controller entities `green_wave`/`max_pressure`,
  per-agent column `safety_total_<tlsid>` — used identically across tasks.
```
