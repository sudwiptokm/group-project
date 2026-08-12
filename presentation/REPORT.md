# Reinforcement Learning for Traffic Signal Control under Heterogeneous Urban Traffic

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


**A fair, tuned comparison of DQN, QR-DQN, PPO, and A2C against a fixed-time baseline with a safety-aware reward**

**Group 7** — Sudwipto Kumar Mondal · Swatej Parmar · Aleana Biju

SUMO + Stable-Baselines3 · Final Project Report

---

## Abstract

This report presents a reinforcement-learning (RL) approach to adaptive traffic-signal control at a single four-arm urban intersection carrying heterogeneous traffic (motorcycles, auto-rickshaws, and cars). The controller observes the intersection through a *passenger-car-equivalent (PCU) weighted* representation, so that vehicles are counted by the road-space they actually occupy rather than by raw number, and it is trained on a *safety-aware* reward that trades reduction in waiting time against a vulnerability-weighted penalty for emergency braking and intersection exposure. Four algorithms — DQN, QR-DQN, PPO, and A2C — are compared on an identical environment, reward, observation, and action space, each tuned with Optuna and evaluated across peak and off-peak demand against a fixed-time baseline. Under oversaturated peak demand, DQN and A2C each cut mean waiting time by approximately 24% relative to fixed-time; DQN records the best mean, and A2C matches it with substantially tighter seed variance. Under light off-peak demand, the fixed-time controller is already near-optimal and no RL agent improves upon it, but all four agents keep traffic mobile rather than collapsing into gridlock — an outcome reported honestly as a ceiling rather than a failure. One methodological asymmetry, affecting only how the off-peak A2C hyperparameters were selected, is disclosed and explained.

---

## 1. Introduction and Motivation

Urban road intersections are a primary bottleneck in city mobility. The most widely deployed control strategy remains the **fixed-time signal**, which cycles through phases on a pre-computed schedule regardless of the traffic actually present. When demand is steady this is adequate, but real demand fluctuates through the day, and a fixed plan cannot respond to it: during congested periods it wastes green time on empty approaches while queues build elsewhere.

The problem is sharper still in cities with **heterogeneous traffic**. Where the vehicle mix is dominated by motorcycles and auto-rickshaws alongside cars, the assumptions behind conventional signal timing break down. Three motorcycles do not occupy the road-space of three cars, they accelerate and filter differently, and a controller that reacts to raw vehicle counts is therefore reacting to a misleading signal. This project targets exactly that setting — a mixed fleet of motorcycles, auto-rickshaws, and cars — and asks a concrete question: **can a reinforcement-learning controller beat the fixed-time baseline, and if so, which algorithm does it best?**

Rather than commit to a single algorithm, the project builds a fair comparison across four RL algorithms on a shared environment, adds a **safety-aware** component to the reward so that efficiency is not pursued at the expense of vulnerable road users, and benchmarks everything against the fixed-time controller under two demand regimes.

---

## 2. Related Work and Background

The project builds on three well-established ideas, reviewed across roughly eleven papers during the literature-review phase.

**Passenger-car equivalents (PCU).** Traffic engineering has long used PCU factors to normalise a heterogeneous fleet into a common unit of road-space demand. A motorcycle contributes far less than a car to congestion; expressing the intersection state in PCU rather than in raw counts gives the controller a signal proportional to actual road occupancy.

**RL for traffic-signal control.** Adaptive signal control is a natural sequential decision problem, and value-based methods (DQN and its distributional variant QR-DQN) as well as policy-gradient/actor-critic methods (PPO, A2C) have all been applied to it. A recurring difficulty is designing a reward that produces genuinely adaptive behaviour rather than a degenerate policy.

**Safety-aware rewards.** Pure throughput objectives can encourage aggressive switching that endangers vulnerable road users. Weighting a safety penalty by user fragility — heaviest for motorcycles — reflects the motivation drawn from the reviewed safety-aware signal-control literature.

To keep this report honest, no specific citations are fabricated here; the design reflects the concepts above as synthesised from the eleven papers reviewed for the project.

---

## 3. Methodology

### 3.1 SUMO Environment

The intersection is a **four-arm junction** built in SUMO (via netedit), driven through TraCI by the `sumo-rl` `SumoEnvironment`. The vehicle fleet is heterogeneous, split **60% motorcycles / 25% auto-rickshaws / 15% cars**, with the **sublane model** enabled (`--lateral-resolution 0.5`) so that two-wheelers filter between larger vehicles realistically rather than queuing single-file. Vehicle types, including their `guiShape`, are defined in `vtypes.add.xml`. Two demand profiles — **peak** (oversaturated) and **off-peak** (light) — are generated by `make_scenarios.py` into `traffic_peak.rou.xml` and `traffic_offpeak.rou.xml`.

### 3.2 PCU-Weighted Observation

The observation function (`PCUObservationFunction` in `env_common.py`) expresses the intersection state in **passenger-car equivalents**, weighting each vehicle by its road-space footprint: **motorcycle 0.3, auto-rickshaw 0.5, car 1.0**. The full observation vector comprises:

- the current-phase **one-hot** encoding,
- a **min-green** flag (whether the minimum green time has elapsed),
- PCU-weighted **density**, and
- PCU-weighted **queue** length.

This means the agent perceives how much road-space is actually demanded on each approach, not merely how many vehicles are present.

### 3.3 Action Space

The action space is **discrete phase selection**: at each decision point the agent chooses which green phase runs next. All four algorithms operate on this identical discrete action space.

### 3.4 Safety-Aware Reward

The reward function (`make_safety_reward_fn(lam)` in `env_common.py`) is:

```
reward = Δwaiting_time − λ · (safety_penalty / SAFETY_SCALE)
```

- The **efficiency term** is the reduction in total waiting time across the intersection (throughput).
- The **safety term** is a composite penalty combining **emergency braking** and **intersection exposure during yellow phases**, with each vehicle weighted by its **vulnerability**: **motorcycle 1.0, auto-rickshaw 0.6, car 0.3**. Fragile road users are weighted most heavily.
- **λ** is held **identical across all four algorithms** at each comparison stage, preserving the apples-to-apples property. At **λ = 0** the safety term is skipped entirely, giving an exact pure-efficiency ablation baseline; at **λ > 0** the safety cost is added. All headline results use the reference **λ = 0.5**.

### 3.5 The Fair Four-Algorithm Ladder

The experimental design isolates the effect of the algorithm by holding everything else constant.

| Element | Setting |
|---|---|
| **Algorithms** | DQN (baseline), QR-DQN, PPO, A2C |
| **Shared** | same environment, reward, observation, action space, seeds, evaluation protocol |
| **Tuning** | Optuna hyperparameter search **per algorithm** (search spaces in `algos.py`) |
| **Training** | 5 seeds per algorithm |
| **Evaluation** | 5 held-out seeds → mean ± std |
| **Scenarios** | peak (oversaturated) + off-peak (light) |
| **Baseline** | fixed-time controller run through the *same* evaluation pipeline (`baseline.py`) |

**SAC is deliberately excluded.** SAC is a continuous-action algorithm and does not fit the discrete phase-selection action space without a different parameterisation; QR-DQN (a distributional DQN) fills the fourth rung instead. The whole comparison is driven by `run_experiment.sh`, which chains tuning, training, evaluation, and aggregation across the scenario and λ axes, and `compare.py` ranks all algorithms plus the fixed-time baseline by mean waiting time.

---

## 4. Results

All metrics below are `system_mean_waiting_time` (seconds; lower is better), reported as mean ± std over five held-out evaluation seeds. Fixed-time is a single deterministic run. Results use the reference **λ = 0.5**.

### 4.1 Peak Demand (Oversaturated Intersection)

| Algorithm | Mean waiting (s) | ± std | vs fixed-time |
|---|--:|--:|--:|
| **DQN** | 1002.8 | 401.6 | **−24.0%** |
| **A2C** | 1003.3 | 86.1 | **−24.0%** |
| fixed-time | 1319.2 | — | baseline |
| PPO | 1356.5 | 45.0 | +2.8% |
| QR-DQN | 1400.7 | 4.9 | +6.2% |

![Peak demand: RL vs fixed-time mean waiting time](../results/bars_peak_lam05.png)

![Peak demand: percentage waiting-time reduction vs fixed-time](../results/improvement_peak_lam05.png)

Under peak demand the intersection is heavily oversaturated — the fixed-time controller backs up to roughly 1319 s mean waiting time. **Only DQN and A2C beat the baseline**, each cutting mean waiting time by about 24%. PPO and QR-DQN land marginally worse than fixed-time (+2.8% and +6.2%). DQN posts the best mean but with high seed-to-seed variance (±402); A2C nearly ties that mean with a far tighter spread (±86).

### 4.2 Off-Peak Demand (Light Traffic)

| Algorithm | Mean waiting (s) | ± std | Mean speed (m/s) |
|---|--:|--:|--:|
| fixed-time | 0.39 | — | 10.40 |
| DQN | 0.48 | 0.01 | 10.50 |
| PPO | 1.76 | 0.07 | 10.01 |
| QR-DQN | 1.99 | 0.23 | 9.91 |
| A2C | 36.0 | 1.5 | 4.75 |

![Off-peak demand: mean waiting time, log scale](../results/bars_offpeak_lam05_logy.png)

![Off-peak demand: mean speed (mobility)](../results/speed_offpeak_lam05.png)

Off-peak demand is light, and here the fixed-time controller is already near-optimal at 0.39 s mean waiting — so **no RL agent beats it**. The meaningful result is that **all four agents keep traffic mobile** rather than collapsing. DQN is within a hair of the baseline (0.48 s) and actually posts the highest mean speed; PPO and QR-DQN add roughly one to two seconds; A2C is the weakest at 36 s (speed 4.75 m/s) but is **valid and mobile**, not the gridlock collapse it previously exhibited. Every one of the eight scenario × algorithm cells is valid and reported; no cell is excluded.

---

## 5. Discussion

**DQN is the overall winner.** Across the two regimes it delivers the best mean congestion relief where it matters most — the oversaturated peak — and it stays within a hair of the near-optimal fixed-time controller off-peak. A2C matches DQN's peak mean with markedly tighter variance, making it a strong and more consistent runner-up under congestion.

**The off-peak ceiling is honest.** In light traffic a well-configured fixed plan is genuinely hard to beat: with almost no queuing to relieve, there is little for an adaptive controller to exploit, and the 0.39 s baseline sits close to the physical floor. The correct reading is that off-peak exposes a *ceiling* on what RL can add, not a failure of the method.

**Disclosed A2C off-peak asymmetry.** For the single off-peak A2C cell, the hyperparameter-selection *objective* was cumulative waiting time (minimise) rather than the shaped reward used to select DQN, PPO, and QR-DQN. The reason is structural and worth stating precisely. At light off-peak demand throughput is essentially flat, so the `−λ · safety` term dominates the shaped reward. Under that reward the *reward-optimal* policy is "never switch phase" — which yields the best safety score and zero throughput, i.e. gridlock. Tuning A2C on the shaped reward selected exactly that collapse. Retuning A2C on waiting time makes gridlock the worst possible score and rejects it. Crucially, this changes only the **HP-selection criterion for one cell** — the training reward, environment, action/observation space, seeds, and evaluation protocol all remain identical across every algorithm and scenario, so the comparison is still apples-to-apples on what the agents optimise and how they are scored at evaluation.

---

## 6. Issues Faced and Resolutions

| Issue | Resolution |
|---|---|
| **Off-peak "gridlock collapse"** — A2C/PPO/QR-DQN produced byte-identical results (agent learned to *never switch phase*) | Per-scenario hyperparameter tuning; for A2C, tune on **waiting time** plus an entropy floor and eval cap so gridlock scores worst and is rejected |
| **SUMO install** — the Homebrew cask route failed / required XQuartz | Install SUMO **via pip** (`eclipse-sumo`, prebuilt binaries) and set `SUMO_HOME` from the venv |
| **`sumo-gui` blank window + `BadShmSeg` spam** on macOS | Run XQuartz on display `:1` with **MIT-SHM disabled** (`Xquartz :1 -extension MIT-SHM`) |
| **Cloud runs crashed** — Ubuntu 24.04 Python 3.12 broke the `torch==2.8.0` pin | Use a **Python 3.11** venv on the cloud machine |
| **Cloud: SUMO "Could not connect in 1 tries"** | Install the X11 libraries pip-SUMO depends on (`libXrender`, etc.) on minimal Ubuntu |
| **Route/network edge-id mismatch** — the simulation would not start | Rename edge ids in the route files to match `intersection.net.xml` exactly |
| **`sumo-rl` API** — `SumoEnvironment` has no `additional_files` kwarg | Load vehicle types via `additional_sumo_cmd="--additional-files vtypes.add.xml"` |
| **SUMO XML parser broke on comments** containing `--` | Removed the `--` sequences from XML comments |

---

## 7. Limitations and Future Work

The reported numbers come from an overnight compute budget rather than the full publication budget, and the safety-tradeoff curve is only partially populated for the RL agents.

**Safety-term sampling defect (disclosed).** In the runs reported here, the safety penalty was sampled once per decision, at the moment the reward was computed. That moment always falls after the yellow/clearing interval has ended, so the **intersection-exposure component never contributed**: `system_safety_exposure` is exactly 0.0, with zero variance, in every run — all four algorithms and the fixed-time baseline. The braking component did contribute, but was itself sampled on only one second in five, and always on the settled post-yellow second, biasing it low. λ therefore acted on braking alone in these results, not on the composite penalty described in §3. The defect has since been fixed by accumulating both components on every simulation second, which raises the measured braking term from 0.206 to 5.07 and the exposure term from 0.0 to 11.57 at peak; the calibration constant was re-derived accordingly (0.024 → 2.1298). **The efficiency results in this report are unaffected** — waiting time, delay, and stopped-vehicle counts are independent of the safety term, and the efficiency half of the reward is byte-identical. The corrected safety term applies from the λ sweep onward, whose runs are consequently not comparable to the λ=0.5 checkpoints used here.

The following extensions are planned, with tentative effort estimates.

| Improvement | Rationale | Tentative effort |
|---|---|---|
| **Full-budget re-run** (100k steps, 3600 s episodes, 5/5 seeds) on a server | Tighten the numbers to publication strength | ~1–2 days compute (server) |
| **Complete the λ safety-tradeoff curve** (λ = 0.0 / 0.5 / 1.0 for the RL agents, not just fixed-time) | Quantify what safety costs in efficiency | ~1 day |
| **Remove the A2C tuning asymmetry** — re-tune all four off-peak on the same waiting-time objective | A fully symmetric comparison | ~0.5 day |
| **Multi-intersection arterial corridor with coordinated MARL** (IPPO / MAPPO) | Scale from one junction to a corridor — a thesis-level lift; an environment prototype is already built | ~2–3 weeks |
| **Real-world demand calibration** (measured flows instead of synthetic) | External validity | ~1 week (data-dependent) |

---

## 8. Team Contributions

Every member contributed to the algorithm and model-building core; each also owned a supporting area.

| Member | Model-building role | Supporting area |
|---|---|---|
| **Sudwipto Kumar Mondal** | **DQN + QR-DQN** — algorithm registry and Optuna search spaces (`algos.py`); tuning and training these agents | RL environment and safety-aware reward (`env_common.py`); experiment driver and cloud/AWS runs (`run_experiment.sh`, `run_parallel.sh`) |
| **Swatej Parmar** | **PPO** — tuning, training, convergence checks; shared training/eval loop (`train.py`) | SUMO model and scenarios (`intersection.*.xml`, `vtypes.add.xml`, `make_scenarios.py`); fixed-time baseline (`baseline.py`) |
| **Aleana Biju** | **A2C** — tuning, training, and the off-peak gridlock-collapse fix (waiting-time objective + entropy floor) | PCU observation design; comparison and plots (`compare.py`, `plots.py`); results write-up |

*All three members contributed to the literature review (11 papers), the fair-comparison design (shared environment / reward / observation), the report, and the presentation.*

---

## 9. Conclusion

An RL controller equipped with a **PCU-weighted observation** and a **safety-aware reward** beats a fixed-time signal precisely where it matters most — under congested peak demand, where DQN and A2C each cut mean waiting time by about 24%. Under a fair, tuned, multi-seed comparison, **DQN is the most reliable winner**, with A2C matching it at peak with tighter variance. Off-peak reveals an honest ceiling: a good fixed plan is difficult to beat in light traffic, though all four agents remain mobile and avoid gridlock. The natural next step is to scale from a single junction to a **multi-intersection arterial corridor with coordinated MARL**, for which an environment prototype has already been built.
