# Presentation Content — RL Traffic Signal Control (Group 7)

> ## ⚠ SUPERSEDED — do not cite the peak results
>
> The peak numbers below (DQN/A2C −24% vs fixed-time) do not hold. Six defects
> are documented with measurements in `docs/FINDINGS_2026-08-12.md` (2026-08-12).
> Corrected, the result reverses: **no learned policy beats a competently timed
> static plan** — best static 11.5 s ± 0.68 against 20–33 s for the learned
> policies. The "fixed-time" baseline referred to throughout is a 10 s-green
> cycler (`baseline.py:23`), not a fixed-time plan.
>
> **Off-peak results are unaffected and still valid.**


> Single source of content for the slide deck. Each `##` = one slide.
> Replace `[Member A/B/C/D]` with real names before presenting.

---

## Slide 1 — Title

**Reinforcement Learning for Traffic Signal Control**
Heterogeneous urban traffic · SUMO + Stable-Baselines3

**Group 7**

A fair, tuned comparison — DQN · QR-DQN · PPO · A2C — against a fixed-time
baseline, with a safety-aware reward.

---

## Slide 2 — The problem

- Urban intersections are congested; **fixed-time signals** ignore live demand.
- Traffic here is **heterogeneous**: motorcycles, auto-rickshaws, cars — very
  different footprints and behaviour.
- Raw vehicle counts mislead a controller: 3 motorbikes ≠ 3 cars in road-space.
- **Question:** can an RL controller beat the fixed-time baseline, and *which*
  algorithm is best?

---

## Slide 3 — Objectives (what we set out to do)

1. Build a realistic single-intersection SUMO model with **mixed vehicle types**.
2. Design an RL controller with a **PCU-weighted** observation and a
   **safety-aware** reward.
3. Run a **fair comparison** of four RL algorithms — same environment, reward,
   observation; only the algorithm changes.
4. Benchmark all four against the **fixed-time baseline**, across **peak** and
   **off-peak** demand.

---

## Slide 4 — What's done & how (the environment)

**Done: a complete, working RL traffic-control pipeline.**

- **Network:** 4-arm intersection built in SUMO (netedit).
- **Vehicles:** motorcycles / auto-rickshaws / cars (60 / 25 / 15 %), sublane
  model so 2-wheelers filter realistically.
- **Observation — PCU-weighted:** the agent sees *passenger-car equivalents*, not
  raw counts (moto 0.3, auto 0.5, car 1.0), plus phase one-hot, min-green flag,
  density, and queue.
- **Action:** discrete **phase selection** (which green phase runs next).
- **Simulator:** `sumo-rl` `SumoEnvironment` over TraCI; agents from
  Stable-Baselines3 / sb3-contrib.

---

## Slide 5 — What's done & how (the safety-aware reward)

```
reward = Δwaiting_time − λ · (safety_penalty / SAFETY_SCALE)
```

- **Efficiency term:** reduction in total waiting time (throughput).
- **Safety term:** composite, **vulnerability-weighted** penalty — emergency
  braking + intersection exposure during yellow — each vehicle weighted by its
  fragility (moto 1.0 / auto 0.6 / car 0.3).
- **λ** is held **identical across all four algorithms** → apples-to-apples.
- λ = 0 → pure efficiency (exact ablation); λ > 0 → adds the safety cost.
- Reported results use the reference **λ = 0.5** — with the **braking component
  only**. A sampling defect (since fixed) meant the exposure component never
  fired in these runs; efficiency results are unaffected. See Limitations.

---

## Slide 6 — What's done & how (the method / experiment)

| | |
|---|---|
| **Algorithms** | DQN (baseline), QR-DQN, PPO, A2C |
| **Shared** | same env, reward, observation, action space, seeds, eval protocol |
| **Tuning** | Optuna hyperparameter search **per algorithm** |
| **Training** | 5 seeds each |
| **Evaluation** | 5 held-out seeds → mean ± std |
| **Scenarios** | peak (oversaturated) + off-peak (light) |
| **Baseline** | fixed-time controller through the *same* eval pipeline |

*Not SAC — it is continuous-action and doesn't fit discrete phase selection.
QR-DQN (distributional DQN) is the fourth rung instead.*

---

## Slide 7 — Results: peak demand (oversaturated)

| Algorithm | Mean wait (s) | ± std | vs fixed-time |
|---|--:|--:|--:|
| **DQN** | 1002.8 | 402 | **−24.0%** |
| **A2C** | 1003.3 | 86 | **−24.0%** |
| fixed-time | 1319.2 | — | baseline |
| PPO | 1356.5 | 45 | +2.8% |
| QR-DQN | 1400.7 | 5 | +6.2% |

- Fixed-time backs up to ~1319 s mean wait.
- **DQN and A2C each cut mean waiting ~24%.**
- A2C nearly ties DQN's mean with far tighter seed spread (±86 vs ±402).
- *Plots:* `results/bars_peak_lam05.png` (absolute waiting) and
  `results/improvement_peak_lam05.png` (% reduction vs fixed-time — DQN/A2C ≈ −24%)

---

## Slide 8 — Results: off-peak demand (light traffic)

| Algorithm | Mean wait (s) | ± std | Speed (m/s) |
|---|--:|--:|--:|
| fixed-time | 0.39 | — | 10.40 |
| DQN | 0.48 | 0.01 | 10.50 |
| PPO | 1.76 | 0.07 | 10.01 |
| QR-DQN | 1.99 | 0.23 | 9.91 |
| A2C | 36.0 | 1.5 | 4.75 |

- Fixed-time is **already near-optimal** (0.39 s) — no RL agent beats it.
- The real result: **all four stay mobile** (no gridlock).
- DQN is within a hair; A2C is weakest but valid.
- *Plots:* `results/bars_offpeak_lam05_logy.png` (waiting, log scale — readable)
  and `results/speed_offpeak_lam05.png` (mobility: a2c collapses to 4.75 m/s, the
  rest stay ~10 m/s → all mobile, a2c weakest)

---

## Slide 9 — Reading the results honestly

- **Peak:** RL clearly helps — DQN/A2C cut waiting ~24%. **DQN is the overall
  winner** (best mean, biggest congestion relief).
- **Off-peak:** the baseline is near-optimal; RL *cannot* beat it — an honest
  **ceiling**, not a failure. All agents stay mobile.
- **One disclosed asymmetry:** off-peak A2C's hyperparameters were selected on
  *waiting time* rather than the shaped reward. At light demand the `−λ·safety`
  term dominates, so the reward-optimal policy is "never switch" = gridlock;
  tuning on waiting time rejects that collapse. Training reward, env, and eval
  stay identical across all algorithms.

---

## Slide 10 — Task distribution (who did what)

> Everyone contributed to the algorithm / model-building core; each member also
> owned a supporting area.

| Member | Model-building role | Supporting area |
|---|---|---|
| **Sudwipto Kumar Mondal** | **DQN + QR-DQN** — algorithm registry & Optuna search spaces (`algos.py`), tuning + training these agents | RL environment & safety-aware reward (`env_common.py`), experiment driver + cloud/AWS runs (`run_experiment.sh`, `run_parallel.sh`) |
| **Swatej Parmar** | **PPO** — tuning, training, convergence checks; shared training/eval loop (`train.py`) | SUMO model & scenarios (`intersection.*.xml`, `vtypes.add.xml`, `make_scenarios.py`), fixed-time baseline (`baseline.py`) |
| **Aleana Biju** | **A2C** — tuning, training, and the off-peak gridlock-collapse fix (waiting-time objective + entropy floor) | PCU observation design, comparison + plots (`compare.py`, `plots.py`), results write-up |

*All three: literature review (11 papers), fair-comparison design (shared
env/reward/obs), report, and this presentation.*

---

## Slide 11 — Issues faced & how we fixed them

| Issue | Fix |
|---|---|
| **Off-peak "gridlock collapse"** — A2C/PPO/QR-DQN produced byte-identical results (agent learned to *never switch phase*) | Per-scenario hyperparameter tuning; for A2C, tune on **waiting time** + entropy floor + eval cap so gridlock scores worst and is rejected |
| **SUMO install** — Homebrew cask route failed / needed XQuartz | Install SUMO **via pip** (`eclipse-sumo`, prebuilt binaries); set `SUMO_HOME` from the venv |
| **`sumo-gui` blank window + `BadShmSeg` spam** on macOS | Run XQuartz on `:1` with **MIT-SHM disabled** (`Xquartz :1 -extension MIT-SHM`) |
| **Cloud runs crashed** — Ubuntu 24.04 Python 3.12 broke the `torch==2.8.0` pin | Use a **Python 3.11** venv on the cloud box |
| **Cloud: SUMO "Could not connect in 1 tries"** | Install the X11 libs pip-SUMO needs (`libXrender`, etc.) on minimal Ubuntu |
| **Route/network edge-id mismatch** — sim wouldn't start | Rename edge ids in routes to match `intersection.net.xml` exactly |
| **`sumo-rl` API** — `SumoEnvironment` has no `additional_files` kwarg | Load vtypes via `additional_sumo_cmd="--additional-files vtypes.add.xml"` |
| **SUMO XML parser broke on comments** containing `--` | Removed `--` from XML comments |

---

## Slide 12 — Areas of improvement & tentative time

| Improvement | Why | Tentative effort |
|---|---|---|
| **Full-budget re-run** (100k steps, 3600 s episodes, 5/5 seeds) on a server | Tighten numbers to publication strength | ~1–2 days compute (server) |
| **Complete the λ safety-tradeoff curve** (λ = 0.0 / 0.5 / 1.0 for RL, not just fixed-time) | Quantify what safety costs in efficiency | ~1 day |
| **Remove the A2C tuning asymmetry** — re-tune all four off-peak on the same waiting-time objective | Fully symmetric comparison | ~0.5 day |
| **Multi-intersection arterial corridor + coordinated MARL** (IPPO / MAPPO) | Scale from one junction to a corridor — thesis-level lift | ~2–3 weeks (env prototype already built) |
| **Real-world demand calibration** (measured flows instead of synthetic) | External validity | ~1 week (data-dependent) |

---

## Slide 13 — Live demo

Play `presentation/demo_dqn_peak.mp4` — the **trained DQN agent** controlling the
peak-demand intersection, with a live metrics HUD (avg wait, queue, speed,
emergency brakes, collisions).

- Watch: PCU-weighted queues build and clear; the agent switches phase under mixed
  moto/auto/car traffic. Colour by type — orange moto, blue auto, grey car.
- **This is a *qualitative* demo — the agent in action.** The headline **−24%**
  result is an *episode-mean* over a full 3600 s run; read it from the results
  slides (`bars_peak_lam05.png`, `improvement_peak_lam05.png`), not the clip's
  instantaneous HUD.
- Do **not** show a live fixed-time-vs-RL race: a short window can invert the
  aggregate result (early transient + high DQN seed variance).

---

## Slide 14 — Conclusions

- An RL controller with a **PCU-weighted observation** and a **safety-aware
  reward** beats a fixed-time signal **where it matters — congested peak demand
  (~24% less waiting).**
- Under a fair, tuned, multi-seed comparison, **DQN is the most reliable
  winner**; A2C matches it at peak with tighter variance.
- Off-peak exposes an honest ceiling: a good fixed plan is hard to beat in light
  traffic.
- **Next:** multi-intersection corridor with coordinated MARL (prototype built).

---

## Slide 15 — Thank you / Q&A

Group 7 · RL Traffic Signal Control · SUMO + Stable-Baselines3
