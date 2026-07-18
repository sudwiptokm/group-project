# Design — Safety-Aware Reward & Two-Stage Algorithm Comparison

Date: 2026-07-17
Project: Smart Traffic Signal Optimization for Heterogeneous Traffic (Group 7)
Branch: `feature/safety-aware-reward-comparison`

## 1. Purpose

Extend the existing fair-comparison RL ladder (DQN / QR-DQN / PPO / A2C on a shared
SUMO intersection env) with:

1. A **safety-aware reward** term motivated by the literature (Samalla & Chunchu 2025:
   risky two-wheeler maneuvers in weak-lane traffic), composite and vulnerability-weighted.
2. A **two-stage experiment protocol** that picks the best algorithm, then traces the
   efficiency-vs-safety tradeoff on the winner across a swept safety weight λ.
3. **Peak / off-peak** demand scenarios.
4. A **fixed-time baseline** integrated into the same comparison metric frame (the
   project's headline claim: RL vs fixed-time).

The reward, observation, and environment remain the invariant across algorithms so the
comparison stays apples-to-apples (per the Noaeen et al. 2022 reproducibility warning).

## 2. Scope

In scope:
- New safety reward function parameterised by λ (env-level invariant).
- Vulnerability-weighted composite safety penalty (emergency braking + intersection exposure).
- Peak/off-peak route file variants + scenario selection in `make_env` and the driver.
- Fixed-time baseline row in `compare.py`; results grouped by (scenario, λ).
- Two-stage run protocol wired through `run_experiment.sh` (SCENARIO, LAMBDA axes).

Out of scope (YAGNI):
- Per-λ re-tuning — Stage-1 tuned hyperparameters are reused across λ.
- Full 3-λ × 4-algo grid — only the winning algo is swept over λ.
- Safety term in the **observation** — reward only; observation stays as-is.
- Continuous-action algorithms (SAC) — action space is discrete phase selection.

## 3. Safety Penalty

Computed each agent decision step, over the lanes of the controlled intersection
(incoming approach lanes + internal junction lanes).

### Vulnerability weights
Inverse of crash protection, mirroring the existing PCU idea:

| type | vClass | vulnerability weight `v` |
|------|--------|--------------------------|
| moto | motorcycle | 1.0 |
| auto | moped (3-wheeler) | 0.6 |
| car  | passenger | 0.3 |
| unknown | — | 0.3 (treat as car, conservative) |

Prefix-match the type id (as `_vehicle_pcu` already does) so distribution suffixes
(e.g. `moto@...`) resolve.

### Sub-terms
```
brake_term    = Σ v(type)  over vehicles with acceleration < −B_THRESH
                B_THRESH = 4.5 m/s²   (hard/emergency deceleration)

exposure_term = Σ v(type)  over vehicles on internal junction lanes
                while the current phase is yellow / clearing

safety(step)  = brake_term + exposure_term      (equal weight, no extra knob)
```

- `brake_term` read via traci `vehicle.getAcceleration` for vehicles on the
  controlled lanes; negative accel below threshold counts as an emergency brake.
- `exposure_term` counts vehicles currently on the junction's internal lanes during a
  yellow/clearing phase — vehicles caught mid-intersection at a phase switch.
- Both sub-terms are vulnerability-weighted and summed with equal weight to avoid
  introducing another free parameter.

## 4. Reward Combination

```
reward(step) = diff_waiting_time  −  λ · (safety / SAFETY_SCALE)
```

- `diff_waiting_time` — sumo-rl's existing built-in reward, unchanged, so the
  efficiency component stays comparable to earlier runs.
- `SAFETY_SCALE` — a one-time calibration constant. Procedure: run one episode at
  λ=0, record mean |diff_waiting_time| and mean raw safety magnitude, set `SAFETY_SCALE`
  so the two are of comparable magnitude. Then **lock and document** the value. This makes
  λ ≈ 1 a meaningful "equal emphasis" point rather than an arbitrary multiplier.
  **Calibrated value (peak scenario, seed 0, via `calibrate_probe.py`):**
  `mean|diff_waiting_time| = 8.44`, `mean_safety = 0.206` →
  **`SAFETY_SCALE = 0.206 / 8.44 ≈ 0.024`** (locked in `env_common.py`). At λ=1 the
  scaled safety term `safety / 0.024` then has mean magnitude ≈ 8.44, matching efficiency.
- **λ is the invariant per stage.** At a given λ, every algorithm sees the identical
  reward function; fairness preserved. λ becomes an experiment axis only in Stage 2.

### Component / interface
`env_common.py` gains a reward-function factory:

```
make_safety_reward_fn(lam: float, safety_scale: float) -> Callable[[TrafficSignal], float]
```

Returns a closure with signature `reward_fn(traffic_signal) -> float` (the sumo-rl
contract). `make_env(seed, scenario, lam, gui, out_csv)` passes the resulting callable as
`reward_fn`. All PCU-observation behaviour is untouched.

## 5. Demand Scenarios

- Two route files derived from the current `traffic.rou.xml`:
  - `traffic_peak.rou.xml` — heavier flows (scale through/turn `vehsPerHour` up).
  - `traffic_offpeak.rou.xml` — lighter flows (scale down).
- Same routes/vTypeDistribution/edge ids — only flow rates differ. Edge ids must still
  match `intersection.net.xml` exactly.
- `make_env(scenario="peak"|"offpeak")` selects the route file. Default keeps current
  behaviour if scenario unspecified.

## 6. Two-Stage Experiment Protocol

### Stage 1 — pick the best algorithm (reference λ = 0.5)
For each algo in {dqn, qrdqn, ppo, a2c}:
1. Tune once (`tune.py --algo X`) at λ=0.5 → `params/X.json`.
2. Train 5 seeds × 2 scenarios (peak, off-peak) at 100k steps with tuned params.
3. Evaluate on 5 held-out seeds × 2 scenarios.
`compare.py` ranks all algos + fixed-time baseline by mean waiting time → **winner**.
≈ 40 training runs (4 × 5 × 2).

### Stage 2 — safety tradeoff curve (winner only)
- λ ∈ {0, 0.5, 1.0}, 5 seeds × 2 scenarios, reusing the winner's Stage-1 tuned params
  (no re-tuning per λ).
- λ=0 doubles as the **ablation** (safety reward off).
- λ=0.5 runs already exist from Stage 1 → only λ∈{0,1.0} add new runs (≈ +20).
- Output: efficiency (mean waiting time) vs safety (mean weighted brakes/exposure) curve.

### Deliverables
1. Ranked algorithm table incl. fixed-time baseline, per scenario.
2. Chosen best (algorithm, hyperparameters).
3. Efficiency-vs-safety tradeoff curve for the winner across λ.

## 7. Prerequisite Code Changes

| File | Change |
|------|--------|
| `env_common.py` | Add vulnerability table, `make_safety_reward_fn(lam, scale)`, `SAFETY_SCALE` constant; extend `make_env` with `scenario` + `lam` args wiring the reward + route file. |
| `traffic_peak.rou.xml`, `traffic_offpeak.rou.xml` | New route variants (scaled flows), edge ids matching the net. |
| `run_experiment.sh` | Add `SCENARIO` and `LAMBDA` env-var axes; loop tune→train→eval→compare over them; resumable/fault-tolerant behaviour preserved. |
| `compare.py` | Add fixed-time baseline row (one no-learning baseline episode per scenario through the eval CSV path); group/rank by (scenario, λ); surface safety metrics alongside waiting time. |

## 8. Testing

- **Reward unit check:** at λ=0, `make_safety_reward_fn` output must equal the plain
  `diff-waiting-time` reward step-for-step (safety term contributes 0).
- **Calibration check:** confirm `SAFETY_SCALE` yields comparable mean magnitudes for the
  two reward components at λ=0 (record the numbers in the spec/README).
- **Scenario smoke test:** each route variant loads in SUMO without edge-id errors
  (`sumo -c` sanity run) and produces a non-empty tripinfo.
- **Ablation sanity:** λ=1.0 should reduce mean weighted emergency-braking vs λ=0 for the
  winning algo (directional check the safety term actually does something).
- **Full system test (Sprint 6):** `run_experiment.sh` end-to-end on a reduced budget
  completes and `compare.py` emits a populated `logs/comparison.csv`.

## 9. Risks / Open Items

- **Compute:** ~60 training runs × 100k steps × 1h SUMO episodes + tuning. Overnight
  unattended on laptop; use `caffeinate` to prevent sleep. If wall-clock overruns,
  fall back to 3 seeds.
- **`SAFETY_SCALE` calibration** is a judgement call — must be recorded and justified.
- **exposure_term implementation** depends on how sumo-rl exposes the yellow/clearing
  phase and internal-lane vehicle lists via traci; verify the accessor during build.
- **B_THRESH = 4.5 m/s²** and vulnerability weights (1.0/0.6/0.3) are defensible defaults;
  cite/justify in the report.
