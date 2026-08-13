# Reinforcement Learning for Traffic Signal Control under Heterogeneous Urban Traffic

> ## Revision note — 2026-08-12
>
> The peak results in this report were rewritten after an audit found six
> defects in how they were produced (§4.1). The earlier claim that DQN and A2C
> cut mean waiting time ~24% under peak demand **is withdrawn**; corrected, no
> learned policy in this project beats a competently timed static plan.
> Measurements and reproduction commands: `docs/FINDINGS_2026-08-12.md`.
>
> **The off-peak results are unaffected and stand as originally reported.**


**A fair, tuned comparison of DQN, QR-DQN, PPO, and A2C against a fixed-time baseline with a safety-aware reward**

**Group 7** — Sudwipto Kumar Mondal · Swatej Parmar · Aleana Biju

SUMO + Stable-Baselines3 · Final Project Report

---

## Abstract

This report presents a reinforcement-learning (RL) approach to adaptive traffic-signal control at a single four-arm urban intersection carrying heterogeneous traffic (motorcycles, auto-rickshaws, and cars). The controller observes the intersection through a *passenger-car-equivalent (PCU) weighted* representation, so that vehicles are counted by the road-space they actually occupy rather than by raw number, and it is trained on a *safety-aware* reward that trades reduction in waiting time against a vulnerability-weighted penalty for emergency braking and intersection exposure. Four algorithms — DQN, QR-DQN, PPO, and A2C — are compared on an identical environment, reward, observation, and action space, each tuned with Optuna and evaluated across peak and off-peak demand against a fixed-time baseline. Under light off-peak demand, the fixed-time controller is already near-optimal and no RL agent improves upon it, but all four agents keep traffic mobile rather than collapsing into gridlock — an outcome reported honestly as a ceiling rather than a failure. Under oversaturated peak demand the project reports a negative result: an audit of the peak pipeline found six defects, five of them independent and any one sufficient to void the headline, and correcting them reverses the earlier conclusion. Sweeping static green durations shows that the best fixed plan at this intersection outperforms every learned policy produced here by a factor of two to three, and the cause is identified as structural — with a 3 s amber and a 10 s minimum green, a controller that switches often loses up to 23% of its capacity to clearance time, a cost `diff_waiting_time` surfaces too late for credit assignment. The reportable contributions are therefore the negative result and its mechanism, the measurement defects (three of which concern common `sumo-rl` usage rather than this codebase), and the off-peak ceiling. One methodological asymmetry, affecting only how the off-peak A2C hyperparameters were selected, is disclosed and explained.

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

Two metrics appear below and they are **not** interchangeable. `system_mean_waiting_time` (seconds) is `sumo-rl`'s per-step metric: `getWaitingTime()` averaged over the vehicles *still in the network*, then averaged over the episode. It is survivorship-biased — a controller that strands traffic is graded on the vehicles it did not strand — and under deadlock it degenerates into a clock that advances one second per simulated second. The off-peak results in §4.2 predate the switch and are quoted in it; §4.1 additionally reports **delay per completed trip** (`timeLoss` from SUMO's `tripinfo`, one row per trip that actually finished) together with throughput and completion rate, which is the metric the project now ranks on. Both are quoted with the episode length and teleport setting they were measured under, because neither is comparable across those settings. Results use the reference **λ = 0.5**; the λ ablation has never been run.

### 4.1 Peak Demand (Oversaturated Intersection)

#### 4.1.1 The withdrawn result

The following was reported in the previous revision and is withdrawn in full.

| Algorithm | Mean waiting (s) | ± std | vs fixed-time |
|---|--:|--:|--:|
| DQN | 1002.8 | 401.6 | −24.0% |
| A2C | 1003.3 | 86.1 | −24.0% |
| fixed-time | 1319.2 | — | baseline |
| PPO | 1356.5 | 45.0 | +2.8% |
| QR-DQN | 1400.7 | 4.9 | +6.2% |

Six defects were found, five of them independent of one another and any one sufficient to invalidate the number.

1. **The metric was a gridlock clock.** Peak deadlocked at t = 780 s (629 vehicles, mean speed ~1e-5 m/s); the 1319 s figure is the area under that ramp, not a delay. It also inverted the ranking: A2C deadlocked on 5 of 5 seeds and scored *best*, while QR-DQN kept traffic moving and scored *worst*.
2. **The baseline ran on different traffic than the agents.** The demand realisation is seed-dependent; the baseline was evaluated on seed 0 and the agents on seeds 42–46. Fixed-time spans 242–1319 s across those six seeds — a 5.4× spread, with seed 0 the worst draw. Paired against the baseline on its *own* seed, every algorithm reverses sign: DQN −23.9% → **+56.9%**, A2C → **+67.6%**, PPO → **+118.4%**, QR-DQN → **+123.3%**.
3. **Every evaluated model predates the safety fix** described in §7 and was never retrained; the 2026-08-06 pass re-evaluated old checkpoints under corrected logging.
4. **The reported ± std was not seed variance.** The driver evaluated only the reference training seed's checkpoint, so four of the five models trained per cell were never evaluated; ±402 is one policy's spread across five demand realisations.
5. **The gridlock was a library default.** `sumo-rl` sets `time_to_teleport = -1` where SUMO's own default is 300 s. With teleporting disabled and permissive left turns, junction deadlock is an *absorbing* state, and a demand sweep from 0.9× to 1.4× is bimodal rather than graded. At `--time-to-teleport 300` peak demand is stable and measurable; the demand level itself needed no change.
6. **The "fixed-time" baseline was not a fixed-time plan.** It advanced the phase on every decision step, which the environment clamps to the minimum green — a 10 s-green cycler, not the 42 s program in the network file, and the *worst* point of the sweep in §4.1.2.

#### 4.1.2 What replaces it: the best static plan

Sweeping the green duration of a static plan over the same environment (`analysis/static_timing.py`; peak 1.5×, seeds 42–46, 3600 s episodes, `--time-to-teleport 300`; figure in `results/static_timing_peak.png`):

| Green (s) | Delay per completed trip (s) | Trips completed | vs 60 s, paired | In-network wait (s) |
|---|---:|---:|---:|---:|
| 10 | 298.6 ± 75.7 | 76.2% | +206.9, loses 5/5 | 29.6 |
| 20 | 384.7 ± 39.4 | 73.2% | +292.9, loses 5/5 | 33.6 |
| 30 | 257.4 ± 133.6 | 87.2% | +165.6, loses 4/5 | 22.6 |
| **45** | **104.9 ± 38.2** | 95.7% | +13.1, wins 3/5 | 13.6 |
| **60** | **91.8 ± 19.9** | 94.3% | — | 15.0 |
| **75** | **91.3 ± 6.6** | 95.1% | −0.4, wins 1/5 | 16.0 |
| **90** | **103.4 ± 18.1** | 94.5% | +11.6, wins 1/5 | 19.4 |
| 120 | 110.8 ± 6.0 | 93.8% | +19.0, wins 1/5 | 24.5 |

Bold rows form the plateau. The fourth column is the mean per-seed difference against the 60 s plan on the *same* demand seed, together with the number of seeds on which that green wins — a paired comparison, since §4.1.1 defect 2 shows how misleading unpaired means are here.

Two features of the curve carry the result. First, **the 10 s and 20 s plans do not clear the demand**: 76% and 73% of trips complete, against roughly 95% everywhere on the plateau. The Stage-1 baseline was not merely slow, it left a quarter of the traffic unserved — precisely the population that the in-network waiting-time metric then declined to count. Second, **the interior is flat**.

**That flatness is a plateau, not a tuned optimum, and the distinction matters.** Within 45–90 s the paired differences are +13.1 s, −0.4 s and +11.6 s while the spread across seeds at any single green is roughly 30 s. The sample minimum is 75 s, but it loses to 60 s on four of five seeds and its entire advantage comes from one outlier draw — treating it as the optimum would repeat defect 2 on a smaller scale. Performance is therefore flat across roughly 45–90 s of green, and the results below quote 60 s as a mid-plateau round number.

The flatness makes the comparison harder to dismiss rather than easier: the static plan that outperforms every learned policy required no tuning skill to find, so it cannot be characterised as an unfairly optimised baseline.

The sweep deliberately runs past 60 s: `sumo-rl`'s `TrafficSignal` stores `max_green` and never reads it, so the `max_green = 60` configured in `env_common.make_env` constrains nothing — longer greens are reachable by a static plan and by a learned policy alike.

At a 60 s green the fixed-time plan performs as follows over seeds 42–46 (`baseline.py --scenario peak --green 60`, aggregated by `compare.py`):

| Metric | Value |
|---|--:|
| Delay per completed trip | **91.8 ± 19.9 s** |
| Trips completed | 4076 ± 137 |
| Completion rate vs demand | 0.943 ± 0.032 |
| In-network mean waiting time | 14.97 ± 3.55 s |

The last two rows illustrate the metric problem directly: the same runs report 14.97 s of in-network mean waiting and 66.4 s of waiting per completed trip, a factor of four apart.

**There is no comparable RL row, and this report does not manufacture one.** Every peak checkpoint predates the safety fix, and two retraining attempts at a 20k-step budget produced no learning (§7). What can be stated is that the Stage-1 policies scored 20–33 s of in-network mean waiting on 1200 s episodes where a static 60 s plan scores 11.5 s — a factor of two to three in the wrong direction.

#### 4.1.3 Why: lost time to amber

With `yellow_time` = 3 s, every phase switch spends three seconds serving nobody:

| Green duration | Share of cycle lost to clearance |
|---|--:|
| 10 s | 3/13 = **23%** |
| 20 s | 3/23 = 13% |
| 60 s | 3/63 = **4.8%** |

The agent decides every 5 s with a 10 s minimum green, so it operates precisely in the region where switching is cheap to attempt and expensive to pay for, and `diff_waiting_time` rewards the immediate queue drop on the approach just served while the clearance cost lands several decisions later. One mechanism accounts for the Stage-1 result, the pilot, and the 20k-step null together, which is why the finding is reported as structural rather than as a training-budget artefact.

A competing explanation was tested and rejected: that teleports corrupt the reward by erasing a jammed vehicle's accumulated waiting time. Reward on teleport steps is mostly *more* negative (−19.1, −18.5, −13.3), because teleports occur when the jam is already severe.

#### 4.1.4 Was there adaptive headroom to find?

§4.1.3 claims the limitation is structural. That claim is testable without training anything, and leaving it untested would be the same error the rest of §4.1 documents — asserting a conclusion the measurement does not yet support.

A static plan outperforming every learned policy admits two readings: the project's reinforcement learning failed to find an adaptive policy that exists, or no such policy exists at this junction. Further training cannot separate them, because a second null result is consistent with both. A non-learning demand-responsive controller can, since it requires no reward specification, no credit assignment and no sample budget; if it cannot beat the best static plan, the headroom is absent rather than merely unfound.

`analysis/actuated.py` implements the classic queue-actuated policy: at each decision step it serves whichever green phase has the largest PCU-weighted queue on the lanes it discharges, subject to the environment's `min_green`. It is deliberately not max-pressure control, which requires downstream occupancy and would measure a different thing on approaches this short. The minimum green is swept because §4.1.3's amber arithmetic identifies it as the parameter that should matter.

Peak 1.5×, seeds 42–46, 3600 s episodes, `--time-to-teleport 300`, paired against the static 60 s plan on the same demand seeds (`analysis/headroom.py`; per-run rows in `analysis/actuated_sweep.csv`):

| `min_green` (s) | Delay per completed trip (s) | Trips completed | vs static 60 s, paired | Seeds beaten |
|---|---:|---:|---:|---:|
| **10** | 517.5 ± 208.4 | 2925 | +425.7 ± 217.5 | 0/5 |
| 20 | 337.0 ± 220.5 | 3455 | +245.3 ± 228.9 | 1/5 |
| 30 | 186.1 ± 108.0 | 3876 | +94.3 ± 115.3 | 1/5 |
| 45 | 161.9 ± 64.2 | 4022 | +70.2 ± 66.3 | 1/5 |
| **60** | **82.5 ± 10.1** | **4156** | **−9.3 ± 23.9** | **3/5** |
| 75 | 92.2 ± 0.9 | 4119 | +0.4 ± 20.8 | 1/5 |
| 90 | 118.7 ± 23.7 | 4038 | +26.9 ± 24.5 | 0/5 |

![Static plan against the queue-actuated controller at peak demand](../results/headroom_peak.png)

The figure (`analysis/plot_headroom.py`) places both controllers on a single axis. The shared x is one constraint seen from two sides: for the static plan it is the green held per phase, for the actuated controller the floor below which a switch request is ignored. Both series are delay per completed trip over the same seeds, so this is one scale rather than two, and the lower panel carries completion for both.

**The minimum green is the binding constraint, not the choice of algorithm.** At the 10 s floor this project trained and evaluated on, a controller with perfect queue information and nothing to learn performs 5.6 times worse than a fixed plan, and leaves a quarter of the demand unserved — 2925 completed trips against the static plan's 4076, and 2008 on the worst seed. It issues 125–168 phase switches per episode against 38–60 at a 75–90 s floor, each costing three seconds of clearance. The sweep is U-shaped and has turned by 90 s, so 60 s is an interior optimum rather than a monotone preference for longer greens.

**The 60 s row requires the same discipline as §4.1.2.** A −9.3 s advantage over the static plan sits inside the noise: the paired difference has a standard deviation of 23.9 s across five seeds. The mean is not the result. What is resolvable is consistency:

| Controller | Delay sd | Trips completed |
|---|--:|--:|
| Static 60 s plan | 19.9 s | 3834–4162 (spread 328) |
| Actuated, `min_green` = 60 | 10.1 s | 4142–4177 (spread **35**) |

The static plan's weak draw is seed 43 (126.3 s, 3834 trips); the actuated controller takes that seed at 83.1 s and 4146 trips. The adaptive advantage is not a lower average — it is the absence of a bad seed.

Neither reading is therefore correct on its own. At `min_green` = 10 there was nothing for a learned controller to find, and that is precisely where the entire training budget was spent, which makes the peak null over-determined rather than informative about reinforcement learning. At `min_green` = 60 there is something to find, but it amounts to roughly a ten per cent variance reduction that a controller requiring no training already collects. **The appropriate reference for future work is consequently the actuated controller at a matched minimum green (82.5 ± 10.1 s), not the static plan**, and a learned policy must beat it by a margin large enough to clear a 24 s paired standard deviation.

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

These off-peak figures are quoted as originally measured: `system_mean_waiting_time`, against the 10 s-cycler baseline, without completed-trip metrics. The conclusion survives the audit regardless. At 0.39 s there is no headroom for any controller, nothing is stranded for the survivorship bias to conceal, and the ranking does not depend on the reward the models were trained against; a better-timed static baseline would only widen the gap in fixed-time's favour.

Off-peak demand is light, and here the fixed-time controller is already near-optimal at 0.39 s mean waiting — so **no RL agent beats it**. The meaningful result is that **all four agents keep traffic mobile** rather than collapsing. DQN is within a hair of the baseline (0.48 s) and actually posts the highest mean speed; PPO and QR-DQN add roughly one to two seconds; A2C is the weakest at 36 s (speed 4.75 m/s) but is **valid and mobile**, not the gridlock collapse it previously exhibited. Every one of the eight scenario × algorithm cells is valid and reported; no cell is excluded.

---

## 5. Discussion

**No algorithm can be declared a winner, and the project no longer claims one.** The peak comparison that ranked DQN first was produced by the pipeline audited in §4.1, and nothing survives it. Off-peak, DQN is the closest to the baseline but does not beat it. A ranking of four algorithms is only meaningful once at least one of them beats a competent static plan, and none does.

**The ceiling is the finding, and it is the same ceiling in both regimes.** Off-peak it is obvious: with almost no queuing to relieve, the 0.39 s baseline sits near the physical floor. At peak it is less obvious but better evidenced — the static sweep in §4.1.2 shows a well-chosen green already captures most of the achievable performance, while the amber-loss arithmetic in §4.1.3 shows that the action space on offer (two phases, 5 s decisions, 10 s minimum green) charges a controller up to 23% of capacity for the switching it is being asked to learn. A learned controller is competing for a margin the environment has largely spent in advance.

**This makes the negative result informative rather than inconclusive.** It identifies a mechanism, predicts where RL *would* have room, and — in §4.1.4 — tests that prediction instead of resting on it. The prediction held in its first and most important part: raising the minimum green from 10 s to 60 s is worth a factor of six to a controller that learns nothing at all, which locates the limitation in the action space rather than in the learning. It also failed in a second, more instructive part: even with the floor corrected, the adaptive margin over a good static plan is roughly ten per cent and lies inside the seed noise on the mean. **The honest summary is that this project's peak training budget was spent in a region of the action space where no controller can win, and that correcting it would still leave a small prize.** That is why the remaining structural extensions — a protected left-turn phase, or coordination across several junctions, neither of which a single static plan can imitate — matter more than any further work on this junction.

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
| **Raise `min_green` to 60 s, then retrain** | Measured, not guessed: §4.1.4 shows the 10 s floor costs even a perfect-information controller a factor of 5.6, so every peak training run to date was conducted where nothing can win. Judge the result against the actuated controller (82.5 ± 10.1 s), not the static plan | ~0.5 day + retrain |
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

This project set out to establish which of four RL algorithms best beats a fixed-time signal at a heterogeneous-traffic intersection, and it ends with a different and better-supported answer: **at an isolated two-phase junction, a competently timed static plan is hard to beat, and we can say why.** With a 3 s amber and a 10 s minimum green, frequent switching costs up to 23% of capacity in clearance time, and the shaped reward surfaces that cost too late for credit assignment at any budget this project could run. Off-peak reaches the same ceiling from the other side, at 0.39 s mean waiting, with all four agents mobile and none ahead of the baseline.

Getting there required withdrawing the earlier peak headline and rebuilding the measurement stack around it: completed-trip delay instead of an in-network average that a stranded queue inflates, a swept static plan instead of a 10 s cycler mislabelled as fixed-time, and baselines paired to the agents' own demand seeds. Three of the six defects concern how `sumo-rl` is commonly used rather than anything specific to this codebase, which makes them worth reporting in their own right.

The finding also points at where an adaptive controller *would* have room, and §4.1.4 tests that pointer rather than asserting it. A non-learning queue-actuated controller — no reward, no training, perfect queue information — is 5.6 times worse than the fixed plan at the 10 s minimum green this project ran on, and matches it at 60 s. The action space, not the algorithm, was the binding constraint, and the entire peak training budget was spent inside it. Correcting the floor is therefore the first thing any continuation should do, and the reference to beat afterwards is that actuated controller rather than the static plan.

Even corrected, the margin at this junction is about ten per cent and sits inside the seed noise, which is the clearest possible argument for changing the problem rather than the optimiser. Coordination across junctions is the one thing a static plan cannot imitate. Scaling to a **multi-intersection arterial corridor with coordinated MARL** — for which an environment prototype has already been built — is therefore not merely the next increment but the first setting in which the question this project asked can be answered in RL's favour.
