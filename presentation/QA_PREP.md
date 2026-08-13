# Group 7 · RL Traffic Signal Control · Progress-review prep

Terminology, technical specifications, team roles, and probable examiner
questions with answers.

> **This file supersedes `presentation/QA_PREP.pdf` (7 Aug 2026).** That PDF is
> **withdrawn**: its Part D.5 and Part E are built on the peak headline that the
> audit of 2026-08-12 disproved, and its Part A still calls an actuated baseline
> "not implemented here", which is no longer true. Do not take it into a viva.
> The PDF has no source in the repo and `presentation/*.pdf` is gitignored, so
> this Markdown is now the source of record. Regenerate a PDF from it if one is
> wanted.
>
> Every figure quoted here was read from the code or the results files, not
> recalled. Where a number was withdrawn, it is shown **as withdrawn** rather
> than deleted, because "what changed and why" is now the most likely question
> in the room.

**One dependency, stated honestly.** A pilot retrain at the corrected
`min_green` (2 algos × 3 seeds × 30k steps, evaluated on seeds 42–46) was
running when this was written. Until it lands there is **no valid RL row at
peak** — say exactly that. Do not present the pilot's numbers before they exist,
and when they do, remember that a null at 30k steps is weak evidence while a
positive is strong (see D.5).

## Contents

- Part A — Terminology in full
- Part B — Technical specifications: what each part does, and how
- Part C — Who built what, and how they did it
- Part D — Probable questions and answers
- Part E — Numbers to memorise, and the traps

---

## Part A · Terminology in full

Anything on a slide that is an abbreviation, a symbol, or a term of art. Read the
middle column aloud if asked; the right column is why it is in this project.

### A.1 Tools and frameworks

| Term | Full meaning | Role here |
|---|---|---|
| SUMO | Simulation of Urban MObility — open-source microscopic traffic simulator | Simulates every vehicle individually, second by second. The world our controller acts on. |
| TraCI | Traffic Control Interface — SUMO's remote-control socket API | How Python reads vehicle states and writes signal phases while the simulation runs. |
| sumo-rl | A library wrapping SUMO as a Gymnasium RL environment | Supplies `SumoEnvironment` and `TrafficSignal`; we subclass both rather than rewrite them. **Three of our six audit findings are defects in how this library is commonly used** — see D.8. |
| SB3 | Stable-Baselines3 — reference PyTorch implementations of RL algorithms | Supplies DQN, PPO, A2C. One library for all arms removes implementation bias. |
| sb3-contrib | Community extension package to Stable-Baselines3 | Supplies QR-DQN, which is not in core SB3. |
| Optuna | Hyperparameter optimisation using Bayesian search (TPE) | Tunes each algorithm separately so no arm is hand-favoured. |
| netedit | SUMO's graphical network editor | Used to draw the four-arm junction, its lanes and its connections. |
| Gymnasium | The maintained successor to OpenAI Gym | The `reset()`/`step()` contract that lets any SB3 algorithm drive our environment. |

### A.2 Traffic-engineering terms

| Term | Full meaning | Role here |
|---|---|---|
| PCU | Passenger Car Unit — road space one vehicle occupies relative to one car | moto 0.3, auto 0.5, car 1.0. The observation counts PCU, not vehicles. |
| Sublane model | Lanes divided into lateral strips, so vehicles hold a continuous lateral position | Lets motorcycles filter through gaps beside cars instead of queueing behind them. |
| Phase | One combination of green/yellow/red across all approaches | Our junction has 2 green phases (north–south, east–west) plus their yellows. |
| Yellow / clearing interval | The amber period between two greens | 3 s here. **This is the mechanism behind the whole peak result** — every switch spends 3 s serving nobody. |
| **Minimum green** | Shortest time a green must be held before it may be ended | **60 s** (`DEFAULT_MIN_GREEN`, or `$MIN_GREEN`). It was hard-wired to **10 s** for every Stage-1 result, and that turned out to be the binding constraint on the whole project — see D.5. sumo-rl enforces it by *silently ignoring* early switch requests. |
| **Maximum green** | Longest a green may be held | Configured at 60 s and **inert**: sumo-rl's `TrafficSignal` stores `max_green` and never reads it (`traffic_signal.py:77` is its only occurrence). It constrains nothing, for a static plan or a learned policy. |
| Cycle | One full rotation through all phases | The static plan in the `.net.xml` is 42+3+42+3 = 90 s. |
| Oversaturated | Demand exceeds capacity, so queues grow rather than clear within a cycle | The peak scenario. Note this is why episode length matters: the backlog deepens across the hour. |
| Fixed-time control | A signal plan with pre-set durations that ignores live traffic | Our baseline — now **a swept static plan** (`baseline.py --green`, default 60 s), not the 10 s cycler Stage 1 used under this name. |
| **Actuated control** | Fixed plan extended or cut short in response to vehicles | **Implemented** (`analysis/actuated.py`), and it produced the decisive result of the project. The 7 Aug pack said "not implemented here — a fair criticism"; that criticism has been answered. See D.4. |

### A.3 Reinforcement-learning terms

| Term | Full meaning | Role here |
|---|---|---|
| Agent / policy | The controller; the policy maps an observation to an action | One agent controlling one junction. |
| Observation | What the agent sees each decision | Phase one-hot, min-green flag, per-lane PCU density and PCU queue. |
| Action space | The set of choices available | `Discrete(2)` — which green phase to run next. **The action space, not the algorithm, is what limited this project.** |
| Reward | The scalar the agent maximises | Change in waiting time, minus λ times a scaled safety penalty. |
| Episode | One complete run from reset to termination | 3600 simulation seconds = one hour of traffic. |
| Seed | The number initialising the random generator | Controls vehicle arrivals. 5 training seeds, 5 held-out evaluation seeds. **Comparing across different seeds is what broke the Stage-1 headline** — see D.8. |
| On-policy | Learns only from data generated by the current policy | PPO, A2C. Sample-hungry but stable. |
| Off-policy | Learns from stored past experience via a replay buffer | DQN, QR-DQN. Sample-efficient, but the buffer can hold stale transitions. |
| Replay buffer | Memory of past transitions | 20k–100k transitions in our search space. |
| Value-based | Learns the value of actions, acts greedily on it | DQN family. No explicit policy network. |
| Policy-gradient | Directly adjusts policy parameters up the reward gradient | PPO, A2C. |
| Actor-critic | A policy (actor) trained using a learned value function (critic) | A2C and PPO both are; A2C is the simpler one. |
| Distributional RL | Learns the whole distribution of returns, not just the mean | QR-DQN's premise. |
| Quantile regression | Fitting a set of quantiles of a distribution | How QR-DQN represents that return distribution. |
| γ (gamma) | Discount factor | Searched over 0.95 / 0.99 / 0.995. |
| Entropy coefficient | Weight on a bonus for keeping the policy random | A2C's collapse fix: floored at 1e-3 so it cannot go deterministic. |
| GAE (λ_GAE) | Generalised Advantage Estimation | Searched 0.9–1.0 for PPO and A2C. Unrelated to our safety λ. |
| MlpPolicy | Plain fully-connected network | Our observation is a flat vector, so no convolutions are needed. |
| ε-greedy | Act randomly with probability ε, otherwise greedily | DQN and QR-DQN. |
| **Queue-actuated control** | Non-learning policy: serve whichever phase has the largest queue | Our reference controller (`analysis/actuated.py`). PCU-weighted, subject to `min_green`. It needs no reward, no credit assignment and no sample budget, which is exactly why it can answer a question training cannot. |

### A.4 Algorithms, spelled out

| Short | Full name | One-line mechanism |
|---|---|---|
| DQN | Deep Q-Network | Learns Q(s,a) with a replay buffer and a slowly-updated target network; acts greedily on the maximum. |
| QR-DQN | Quantile Regression Deep Q-Network | Same loop as DQN, but each action's return is a set of quantiles rather than one mean. |
| PPO | Proximal Policy Optimization | Policy gradient that clips each update so the new policy cannot move far from the old one. |
| A2C | Advantage Actor-Critic | Synchronous actor-critic. No replay, no clipping — cheapest and twitchiest of the four. |

### A.5 Symbols and constants specific to this project

| Symbol | Meaning | Value / note |
|---|---|---|
| λ (lambda) | Weight on the safety penalty in the reward | 0.5 for all Stage-1 results. **The λ ablation has never been run** — only λ=0.5 exists, so "safety-aware" is in the title and not in the results. Volunteer this. |
| `SAFETY_SCALE` | Puts the safety penalty on the same numeric scale as efficiency | 2.1298 (was 0.024 before the fix) = 17.97 ÷ 8.44. Derived at a **10 s** floor, which is why `calibrate_probe.py` pins that floor rather than inheriting the new default. |
| `B_THRESH` | Deceleration above which braking counts as an emergency brake | 4.5 m/s². |
| σ (sigma) | Standard deviation across evaluation seeds | Used throughout. ~~DQN peak σ = 401.5 s~~ — **withdrawn**: that was one policy's spread across five *demand* seeds, not seed variance (defect 4). |
| `delta_time` | Simulation seconds between agent decisions | 5 s. |
| `yellow_time` | Duration of the amber interval | 3 s. That `delta_time > yellow_time` is exactly what made the exposure term unreachable. |
| `min_green` | Action-space floor | **60 s** now; **10 s** for every Stage-1 result. The single most consequential number in the project. |
| `time_to_teleport` | How long a stuck vehicle waits before SUMO removes it | **300 s** (SUMO's own default). sumo-rl ships −1, which makes junction deadlock an absorbing state — defect 5. |
| `trip_time_loss_mean` | Delay per **completed** trip, from SUMO `tripinfo` | The ranking metric now. |
| `safety_brake` | Vulnerability-weighted count of emergency-braking events | 0.206 before the fix, 5.07 after (peak, seed 0). |
| `safety_exposure` | Vulnerability-weighted count of vehicles inside the junction during amber | 0.0 before the fix (all runs), 11.57 after. |
| Vulnerability weight | How exposed a road user is — the inverse idea to PCU | moto 1.0, auto 0.6, car 0.3. |

---

## Part B · Technical specifications

### B.1 The network — `intersection.net.xml`

A single four-arm junction (north, south, east, west), two lanes per approach,
built in netedit. The signal is one controller, id `C`, with 16 controlled links.

- Two green phases: north–south through-and-right, then east–west
  through-and-right. Each is followed by a 3 s all-amber. The static plan baked
  into the file is 42 + 3 + 42 + 3, a 90 s cycle.
- Every entry-to-exit movement crosses a short internal lane inside the junction
  box, named with a leading colon, e.g. `:C_4_0`. These are what the exposure
  term counts vehicles on.
- The sublane model is enabled with `--lateral-resolution 0.5`.

**Why the sublane model matters.** Without it a 0.8 m motorcycle would queue
nose-to-tail behind a 1.8 m car and occupy a whole lane. With it, two-wheelers
filter into lateral gaps. This is the single modelling choice that makes the
heterogeneous-traffic framing meaningful rather than decorative.

**Worth knowing:** the two green phases have **permissive** left turns
(`GGGgrrrr`). Combined with sumo-rl's `time_to_teleport = -1` default, that made
deadlock an absorbing state — the junction locks and nothing ever clears it.
See D.8, defect 5.

### B.2 Vehicle types — `vtypes.add.xml`

| Type | Length × width | accel / decel | Max speed | PCU | Vulnerability | Share |
|---|---|---|---|---|---|---|
| moto | 2.0 × 0.8 m | 3.5 / 6.0 m/s² | 22.0 m/s | 0.3 | 1.0 | 60 % |
| auto | 2.8 × 1.3 m | 2.0 / 5.0 m/s² | 14.0 m/s | 0.5 | 0.6 | 25 % |
| car | 4.5 × 1.8 m | 2.6 / 4.5 m/s² | 25.0 m/s | 1.0 | 0.3 | 15 % |

Sampled per spawned vehicle from a `vTypeDistribution` with probabilities
0.60 / 0.25 / 0.15. Motorcycles also get `latAlignment="arbitrary"` and higher
lateral speed and sublane lane-changing eagerness, which is what lets them
filter.

The two weight tables are deliberate inversions. PCU says the motorcycle costs
the least road space; vulnerability says it carries the most risk.

### B.3 Demand scenarios — `make_scenarios.py`

- Base `traffic.rou.xml` defines 12 flows totalling 2,880 veh/h across the four
  arms, heaviest north–south (500 veh/h each way) and east–west through (400).
- `make_scenarios.py` rewrites only `vehsPerHour` — routes, type distribution
  and edges untouched.
- Peak = ×1.5 → 4,320 veh/h. Off-peak = ×0.5 → 1,440 veh/h.
- All flows are constant-rate over the full 3600 s, so the demand is stationary
  — but peak is *oversaturated*, so the queue deepens across the hour. That is
  why training and evaluation both use full-hour episodes.

**Peak demand needed no change.** A sweep from 0.9× to 1.4× was bimodal and
non-monotonic (free flow ~0.5 s or deadlock ~1200 s, nothing between) — a
stochastic tipping process, not a demand level. At `--time-to-teleport 300`,
peak 1.5× is stable and measurable.

### B.4 The observation — `PCUObservationFunction` in `env_common.py`

A flat float32 vector, normalised to roughly [0, 1], of length
`num_green_phases + 1 + 2 × num_lanes`:

| Component | Size | What it encodes |
|---|---|---|
| Phase one-hot | 2 | Which green phase is currently active. |
| Min-green flag | 1 | 1 if the current green has run at least `min_green + yellow_time`, so a switch is legal. |
| PCU density per lane | one per lane | Sum of PCU on the lane ÷ lane capacity in PCU. |
| PCU queue per lane | one per lane | Same, counting only halting vehicles. |

Lane capacity in PCU ≈ lane length ÷ (car length + minimum gap).

**Why PCU rather than vehicle counts.** Three motorcycles and three cars are the
same number but not the same demand on green time.

### B.5 The action space and decision timing

- `Discrete(2)` — choose which green phase should run next.
- The agent is asked every 5 simulation seconds (`delta_time`).
- Selecting the running phase continues it; selecting the other inserts the 3 s
  amber automatically, then switches.
- A switch is refused until `yellow_time + min_green` has elapsed. **At the
  Stage-1 floor that was 13 s; at the corrected floor it is 63 s.** sumo-rl
  refuses *silently* — the request is discarded with no signal to the agent.
- `max_green` bounds nothing (A.2).
- So the agent does not set durations directly. It repeatedly answers "hold or
  switch?", and durations emerge.

### B.6 The reward — `make_safety_reward_fn` in `env_common.py`

```
reward = diff_waiting_time − λ · (safety_penalty / SAFETY_SCALE)
```

**Efficiency half.** `diff_waiting_time` is sumo-rl's built-in reward: the change
in total accumulated waiting time at this junction since the last decision.
Unmodified, so this half stays comparable with other sumo-rl work.

**Safety half.** Two vulnerability-weighted sub-terms, summed with equal weight:

- `brake_term` — for each vehicle on a controlled lane decelerating below
  −4.5 m/s², add its vulnerability weight.
- `exposure_term` — while amber, for each vehicle on an internal junction lane,
  add its vulnerability weight. "Caught in the box at the switch."

Both are now accumulated on **every simulation second** inside the decision
window by `_SafetyWindow`, and read at the action step. Previously sampled once,
at the action step itself — see D.6.

**Why λ needs a scale constant.** The two halves have unrelated units.
`SAFETY_SCALE` is measured once, at λ=0, as mean safety ÷ mean |efficiency|,
then locked: 17.97 ÷ 8.44 = 2.1298.

**Known weakness, and it matters for the peak result.** `diff_waiting_time`
rewards the immediate queue drop on the approach just served, and only registers
the lost-time cost of switching several decisions later. At a 10 s floor the
agent sits exactly where switching is cheap to attempt and ruinous to pay for.

### B.7 The algorithms and their search spaces — `algos.py`

One registry maps each name to an SB3 class, a default config, and an Optuna
sampling function. All four use `MlpPolicy`.

| Arm | Class | Searched over |
|---|---|---|
| dqn | `stable_baselines3.DQN` | lr 1e-5..1e-3 (log), buffer 20k/50k/100k, `learning_starts`, batch, γ, `train_freq`, `target_update_interval`, exploration fraction and final ε, net arch |
| qrdqn | `sb3_contrib.QRDQN` | Identical space to DQN — deliberately, so the only difference is the distributional head |
| ppo | `stable_baselines3.PPO` | lr, `n_steps` 128/256/512, batch constrained to divide `n_steps`, `n_epochs`, γ, GAE λ, clip range, entropy coefficient, net arch |
| a2c | `stable_baselines3.A2C` | lr capped at 3e-4, `n_steps` 16/32/64, γ, GAE λ, entropy coefficient floored at 1e-3, vf coefficient, `max_grad_norm`, net arch |

The A2C space is narrower, and that is disclosed rather than hidden.

**Note for any retrain:** `params/*.json` were tuned for a 100k-step budget
(lr ~2.3e-5, `learning_starts` 5000). Against a 20–30k budget they spend a
quarter of it on random actions. They were also selected at a **10 s** floor, so
they are stale for the corrected action space — hyperparameters are selected
*for* an action space.

### B.8 Training and evaluation protocol

| Stage | What happens | Budget (full mode) |
|---|---|---|
| Tune | Optuna searches each algorithm separately at the reference point (peak, λ=0.5). Best config → `params/<algo>.json`. | 30 trials × 20,000 steps |
| Train | Each algorithm on 5 seeds per scenario. Checkpoints to `models/`. | 100,000 agent steps per run |
| Evaluate | Checkpoints rolled out deterministically on 5 held-out seeds (42–46). | 5 episodes per arm per scenario |
| Compare | `compare.py` time-averages each run, then aggregates across seeds. | Ranked by `trip_time_loss_mean`, lower better |

**Two protocol defects found in the audit, both now fixed:**

1. The driver evaluated only `models/<algo>_<tag>_seed$REF_SEED.zip` — the first
   training seed — across all eval seeds. **Four of the five models trained per
   cell were never evaluated at all**, so the reported ±std was one policy's
   spread across demand realisations. Fixed by `train.py --train-seed`, which
   tags the eval CSV `_t<n>`.
2. Filenames now also carry `_mg<n>`, the action-space floor, and `compare.py`
   warns rather than averaging two floors — or two static greens — into one row.

**What is held identical across arms.** Environment, reward, observation, action
space, episode length, training seeds, evaluation seeds, evaluation protocol.
Only the algorithm changes.

### B.9 The fixed-time baseline — `baseline.py`

Runs through the identical environment, so its metrics land in the format
`compare.py` consumes. It never learns: it holds each green for `--green`
seconds, then switches.

**What it used to be, and why that was defect 6.** `baseline.py` advanced the
action *every decision step*, which with `delta_time=5` and `min_green=10` is a
**10 s-green cycler** — not a fixed-time plan in any traffic-engineering sense,
and the *worst* point of the static sweep. Every "vs fixed-time" number in Stage
1 compares against that controller. The 7 Aug pack flagged the ~13–15 s green as
a caveat; the audit promoted it to a defect.

`--green` now defaults to 60 s (mid-plateau), and the green goes in the CSV name
after the seed so two plans cannot be averaged into one "fixed-time" row.

### B.10 The queue-actuated reference — `analysis/actuated.py`

The non-learning controller that settles what training cannot. Each decision
step it serves whichever green phase has the largest **PCU-weighted queue** on
the lanes it discharges, subject to `min_green`.

- Deliberately **not** max-pressure, which needs downstream occupancy and would
  measure something else on approaches this short.
- Perfect queue information, no reward to misspecify, no credit assignment, no
  sample budget. **If it cannot beat the best static plan, the headroom is not
  there to be found.**
- `analysis/headroom.py` pairs it against the static plan per seed;
  `analysis/plot_headroom.py` draws both on one axis.

### B.11 Metrics — what each one measures

| Metric | Meaning | Watch out for |
|---|---|---|
| **`trip_time_loss_mean`** | Delay per **completed** trip, vs a free-flow run of the same route | **The ranking metric.** From SUMO `tripinfo`, one row per finished trip. |
| `trips_completed` / `trip_completion_rate` | Throughput, and share of offered demand that finished | The number that exposed the 10 s plan: 76 % completion vs ~95 % on the plateau. |
| `system_mean_waiting_time` | Average accumulated wait per vehicle **currently in the network** | **Survivorship-biased.** Vehicles that clear stop contributing; stuck ones contribute forever. Under deadlock it degenerates into a clock at 1 s per second. |
| `system_total_waiting_time` | Waiting summed over vehicles in the network | Scales with occupancy. |
| `system_total_stopped` | Vehicles halted at an instant | A congestion-extent measure. |
| `system_mean_speed` | Mean speed across vehicles in the network | Mobility; catches gridlock. |
| `system_safety_brake` / `_exposure` / `_total` | The raw safety sub-terms, logged unscaled | The quantities the reward actually penalises. |

**The 4× gap, memorise it.** On the same peak runs: 14.97 s in-network mean wait
against 66.4 s of waiting per completed trip. That gap *is* the survivorship
bias, measured rather than argued.

---

## Part C · Who built what, and how

Everyone contributed to the algorithm and model-building core. Each also owned a
supporting area.

### Sudwipto Kumar Mondal

**Model-building — DQN and QR-DQN**

- Built the algorithm registry in `algos.py`, so `tune.py` and `train.py` stay
  algorithm-agnostic.
- Wrote a single shared search space for both off-policy arms, so DQN and QR-DQN
  differ only in the distributional head.
- Ran the Optuna study and multi-seed training for both arms.

**Supporting — environment and safety-aware reward (`env_common.py`)**

- Implemented `PCUObservationFunction` and the safety-aware reward: vulnerability
  weights, the 4.5 m/s² threshold, the amber exposure term, and the
  `SAFETY_SCALE` calibration procedure.
- Found, root-caused and fixed the sampling defect: traced exposure being
  identically zero to sumo-rl evaluating the reward only after the decision
  window; wrote 12 failing tests first, then the `_SafetyWindow` accumulator,
  then re-derived the calibration constant.

**Supporting — the audit and the measurement stack**

- Audited the Stage-1 peak pipeline and withdrew its headline: six defects, five
  independent, any one sufficient (D.8).
- Rebuilt the metric stack on completed-trip delay (`analysis/tripinfo.py`), the
  swept static baseline (`baseline.py --green`), and paired per-seed comparison.
- Wrote the queue-actuated reference controller and the `min_green` sweep that
  identified the action space as the binding constraint.

**Supporting — experiment driver and cloud runs**

- `run_experiment.sh`: resumable tune → train → evaluate → compare pipeline with
  overnight and full presets.
- `run_parallel.sh`: fans the job list across cores, interrupt-safe.

### Swatej Parmar

**Model-building — PPO**

- Tuned and trained the PPO arm, including convergence checking.
- Constrained the PPO search so batch size always divides `n_steps` — otherwise
  PPO silently drops a partial batch each rollout and the comparison is subtly
  unfair.

**Supporting — SUMO model and scenarios**

- Built the four-arm network in netedit: lanes, connections, and the
  two-green-phase signal program.
- Authored `vtypes.add.xml` — three vehicle types with physical dimensions,
  acceleration and braking limits, and the lateral parameters that make sublane
  filtering behave.
- Wrote `make_scenarios.py`, deriving peak and off-peak by scaling only flow
  rates.

**Supporting — shared training loop and fixed-time baseline**

- `train.py`: one train-and-evaluate path used by every algorithm.
- `baseline.py`: the non-learning control, cycling phases through the identical
  environment.

### Aleana Biju

**Model-building — A2C, and the off-peak collapse fix**

- Tuned and trained the A2C arm.
- Diagnosed the off-peak failure: at light demand the −λ·safety term dominates,
  so the reward-optimal policy is to never switch — gridlock scoring well.
- Fixed it with three changes, each traceable to a symptom: entropy coefficient
  floored at 1e-3, learning rate capped at 3e-4 with lengthened rollouts, and
  `max_grad_norm` searched.
- This is the one methodological asymmetry in the project, and it is disclosed.

**Supporting — PCU observation design**

- Worked out the PCU weighting and capacity normalisation that turns raw vehicle
  lists into a bounded observation vector.

**Supporting — comparison, plots and write-up**

- `compare.py`: globs the per-run evaluation CSVs, time-averages each run, then
  aggregates across seeds into one ranked table.
- `plots.py` and the report narrative.

Shared by all three: the literature review (11 papers), the fair-comparison
design, the report, and this presentation.

---

## Part D · Probable questions and answers

Grouped by theme. Answer in the first paragraph; the *If pressed* line is what to
add only if pushed. Where an answer is a hypothesis rather than a verified
result, it says so — do not upgrade it under pressure.

### D.1 Framing and motivation

**Q — Why reinforcement learning at all? Adaptive signal control is a solved
industrial problem.**

It is solved for homogeneous traffic. SCATS and SCOOT tune green splits against
detector counts, and counts are the wrong quantity when 60 % of the fleet is
motorcycles that filter laterally. Our contribution is not "RL beats fixed-time"
— it is a controlled comparison on a PCU-weighted, safety-aware formulation.

*If pressed:* We are not claiming deployment readiness. And note the project's
actual finding is a negative one, which we report as a finding rather than
bury.

**Q — What is the actual research question?**

Given an identical environment, reward, observation and evaluation protocol,
which family of RL algorithm controls a heterogeneous-traffic intersection best,
and does that answer change with demand? Everything in the design exists to keep
"only the algorithm changes" true.

*If pressed:* We can now answer a prior question the original one assumed away —
**whether there is anything at this junction for an adaptive controller to win.**
Mostly there is not, and that is the result.

**Q — Why a single intersection?**

It isolates the algorithm comparison from the coordination problem. At one
junction there is no credit assignment across agents, so any difference between
arms is attributable to the learner.

*If pressed:* It also turned out to bound what was winnable. At a 2-phase
isolated junction with permissive lefts, near-optimal control is close to "hold
a long green and alternate" — which is why the corridor is the next step rather
than more training here.

**Q — Is this novel enough?**

The individual pieces are not novel. What we contribute is a controlled
comparison under a single invariant environment, plus a methodological audit
whose findings generalise past this project — three of the six are defects in how
`sumo-rl` is commonly used. The honest framing is "a careful benchmark and a
cautionary result", not "a new algorithm".

**Q — Why should anyone trust results from a simulator?**

They should not, as deployment evidence. SUMO gives a controlled, repeatable
testbed where the counterfactual — the same hour of traffic under a different
controller — is actually available. The limitation is external validity, and the
demand is synthetic rather than calibrated to measured flows.

### D.2 Environment and modelling

**Q — Where do the PCU weights 0.3 / 0.5 / 1.0 come from?**

The conventional passenger-car-equivalent ordering for two-wheelers,
three-wheelers and cars, relative to a car at 1.0. Defensible defaults rather
than values estimated from local data.

*If pressed:* The right robustness check is to re-run with perturbed weights and
see whether the ranking moves. We have not done that.

**Q — Why is the vehicle mix 60 % motorcycles?**

It reflects the modal split in the South Asian urban context the study is aimed
at. It is set in a `vTypeDistribution` and is a single-line change to vary.

**Q — What does the sublane model actually change?**

Without it, SUMO puts every vehicle at its lane centre and a motorcycle occupies
a full lane. With `--lateral-resolution 0.5`, lanes are 0.5 m strips and vehicles
hold a continuous lateral position, so a 0.8 m motorcycle can filter past a 1.8 m
car. Without this, the whole heterogeneous-traffic premise is cosmetic.

**Q — Your observation has no downstream or neighbouring-junction information.
Is it not partially observable?**

Yes, it is. The agent sees only its own approaches. At an isolated junction the
missing information is mainly arrival prediction. We accept the partial
observability rather than paper over it; a recurrent policy or an arrivals
feature would be the fix.

**Q — Why is the episode one hour?**

3600 simulation seconds matches the flow definitions, which are specified in
vehicles per hour, and is long enough for queues to build and clear several
times. It matters more than it looks: peak is oversaturated, so the backlog
deepens across the hour, and a shorter episode would never show a controller the
states it is evaluated on.

**Q — Why only two green phases? Real junctions have protected turns.**

The network gives each approach a shared through-and-right movement with
permitted lefts, so two green phases cover it. Adding protected left phases would
enlarge the action space.

*If pressed:* This is now one of our two recommended structural extensions,
precisely because a 2-phase junction leaves an adaptive controller so little to
win. It is not orthogonal to the result — it *is* the result.

### D.3 Reward and safety

**Q — SUMO's car-following model prevents collisions. How can you claim anything
about safety?**

We do not measure crashes and we should not claim to. We measure two surrogates:
emergency braking below −4.5 m/s², and vehicles caught inside the junction during
amber. The correct phrasing is "reduces conflict surrogates", never "reduces
accidents".

**Q — Why 4.5 m/s² for the braking threshold?**

It is the conventional boundary between comfortable and emergency deceleration,
and sits below our vehicle types' `emergencyDecel` (8–9 m/s²) so it triggers
before the physical limit. A defensible default, not derived from our data.

**Q — Why are vulnerability weights the inverse of PCU?**

Because road space and crash risk run opposite ways. A motorcycle takes the least
space (PCU 0.3) and offers its rider the least protection (vulnerability 1.0).
One table for both would make the controller treat the most exposed road user as
the least important.

**Q — Why add safety to the reward instead of constraining it?**

Scalarisation with a single λ was chosen for simplicity and because it makes λ an
interpretable experiment axis. A constrained formulation — Lagrangian or CMDP —
would be more principled: it targets a safety budget rather than a trade-off
weight, and avoids the reward-hacking we observed off-peak. Fair criticism, and a
good Stage-3 direction.

**Q — At λ=0.5 the safety term was worth about 40 % of the reward magnitude. Was
it doing anything?**

Before the fix, less than we claimed — it acted on a braking signal sampled one
second in five, always on the calmest second of the cycle. After the fix, with
the scale re-derived, the safety term averages 3.91 against efficiency 5.64 at
λ=0.5, and both sub-terms are live.

**Q — Post-fix, exposure is more than double braking. Is that the weighting you
want?**

That is a decision for you. Exposure counts every vehicle on an internal lane on
every amber second, so it scales with junction occupancy — meaning λ now weights
"clear the junction before amber" more heavily than "do not brake hard".
Defensible, because being caught in the box is the more serious conflict, but it
should be a stated choice before the sweep is interpreted.

**Q — Could an agent game this reward?**

It did, off-peak. With light demand there is little waiting time to recover, so
the safety penalty dominates and never switching scores well while producing
gridlock. A2C found that policy. We rejected it by tuning off-peak A2C on waiting
time and flooring the entropy coefficient.

### D.4 Algorithms, baselines and fairness

**Q — Why these four algorithms?**

They span the two axes that matter. Off-policy vs on-policy: DQN and QR-DQN learn
from a replay buffer, PPO and A2C from fresh rollouts. Within each pair, a simple
member and a more sophisticated one. So the comparison isolates what each
mechanism buys.

**Q — Why not compare against an actuated baseline rather than fixed-time?**

**We did, and it became the most important result in the project.** The 7 Aug
version of this pack said we had not implemented one and called that a fair
criticism. `analysis/actuated.py` now implements the standard queue-actuated
policy: serve whichever phase has the largest PCU-weighted queue, subject to
`min_green`.

It was built to settle a question the static sweep raises but cannot answer. A
fixed plan beating every learned policy is consistent with two readings — our RL
failed to find an adaptive policy that exists, or there is none to find here —
and more training separates neither, because a second null fits both. A
controller that needs no reward, no credit assignment and no sample budget does
separate them.

Sweeping its `min_green` at peak, paired against the static 60 s plan on the same
seeds:

| `min_green` (s) | delay / completed trip (s) | trips | vs static 60 s, paired | seeds beaten |
|---|---:|---:|---:|---:|
| **10** | 517.5 ± 208.4 | 2925 | +425.7 ± 217.5 | 0/5 |
| 20 | 337.0 ± 220.5 | 3455 | +245.3 ± 228.9 | 1/5 |
| 30 | 186.1 ± 108.0 | 3876 | +94.3 ± 115.3 | 1/5 |
| 45 | 161.9 ± 64.2 | 4022 | +70.2 ± 66.3 | 1/5 |
| **60** | **82.5 ± 10.1** | **4156** | **−9.3 ± 23.9** | **3/5** |
| 75 | 92.2 ± 0.9 | 4119 | +0.4 ± 20.8 | 1/5 |
| 90 | 118.7 ± 23.7 | 4038 | +26.9 ± 24.5 | 0/5 |

*If pressed on "why not max-pressure?":* it needs downstream occupancy and would
measure something different on approaches this short. Queue-actuated is the
standard non-learning reference for a 2-phase junction.

**Q — Is the comparison actually fair, given A2C was tuned differently
off-peak?**

Every arm shares the environment, reward, observation, action space, seeds and
evaluation protocol. The single difference is the objective used to select A2C's
off-peak hyperparameters — waiting time rather than the shaped reward — and it
affects selection only, not training. The alternative was to report a gridlocked
A2C that scored well on the reward.

*If pressed:* The clean fix is to re-tune all four off-peak on the same
waiting-time objective. About half a day, and on the improvement list.

**Q — Did you tune per scenario?**

Tuning ran once, at peak with λ=0.5, and the configuration was reused across
scenarios and λ values. A deliberate compute trade-off and a real limitation.

*If pressed:* There is now a sharper version of this problem. Hyperparameters are
selected **for an action space**, and ours were selected at a 10 s floor. Any
retrain at 60 s should re-tune, which is why `tune.py` now takes `--min-green`
and `run_experiment.sh` sets one floor for a whole grid.

**Q — Is 30 Optuna trials enough?**

Modest for a 9–10 dimensional space. Enough to avoid pathological corners, which
is what fairness requires, but not enough to claim any arm was tuned to its
ceiling. The comparison is "equally and modestly tuned".

**Q — ~~QR-DQN is meant to improve on DQN. Why is it your worst peak arm?~~**

**The premise is withdrawn.** That ranking came from the invalidated peak
comparison. There is currently no valid peak ranking of any kind — see D.5. If
asked about QR-DQN specifically, the only defensible statement is that it was
given exactly the same search space and budget as DQN, so any future difference
will not be an unequal-tuning artefact.

### D.5 Results and statistics

**Q — What are your peak results?**

**There is no valid RL result at peak, and we do not manufacture one.** Every
peak checkpoint predates the safety fix and none was retrained; two 20k-step
retraining attempts produced no learning. What we have instead is a
well-measured statement about the problem:

| Controller | delay per completed trip |
|---|---|
| static plan, 10 s green (what Stage 1 called "fixed-time") | 298.6 ± 75.7 s, 76 % completion |
| static plan, 60 s green (mid-plateau) | 91.8 ± 19.9 s, 94 % completion |
| queue-actuated, `min_green` 60 | 82.5 ± 10.1 s, 96 % completion |

*If pressed:* Stage-1 policies sat at 20–33 s in-network wait where a 60 s static
plan sits at 11.5 s, on the same 1200 s episodes — a factor of 2–3 the wrong way.
That is comparable within itself but not to the 3600 s figures above.

**Q — What is the headline finding, then?**

**`min_green` was the binding constraint, not the algorithm.** At the 10 s floor
every peak run used, a controller with perfect queue information and nothing to
learn is 5.6× worse than a fixed plan and strands a quarter of the traffic. So
the peak null is **over-determined**: the entire training budget was spent in a
region of the action space where no controller can win. It says nothing about RL.

The mechanism is arithmetic. With a 3 s amber, a 10 s green loses 3/13 = **23 %**
of the cycle to clearance, against 4.8 % at 60 s. The actuated controller
requests 125–168 switches per episode at a 10 s floor, against 38–60 at 75–90 s.

**Q — So fix the floor and RL wins?**

Probably not, and we should not promise it. At a 60 s floor the actuated
controller beats the best static plan by 9.3 s — but the paired difference has an
sd of **23.9 s** across five seeds, so that is inside the noise. **The mean is
not the result.** What is resolvable is consistency:

| | delay sd | trips completed |
|---|---:|---:|
| static 60 s plan | 19.9 s | 3834–4162 (spread 328) |
| actuated, `min_green` 60 | 10.1 s | 4142–4177 (spread **35**) |

The static plan's bad draw is seed 43 (126.3 s, 3834 trips); the actuated
controller takes that same seed at 83.1 s and 4146 trips. **The adaptive gain is
not a lower mean — it is not having a bad seed.**

**Q — Then what would a valid RL result have to beat?**

The **actuated controller at a matched floor** — 82.5 ± 10.1 s — not the static
plan. Matching the static plan proves nothing, because a policy that needs no
training already does that. And it has to beat it by enough to clear a 24 s
paired sd.

**Q — Did you run a significance test?**

No, and we should not pretend otherwise. Everything above is reported as paired
per-seed differences with the spread, and the counts of seeds won, precisely
because with five seeds a point estimate is not the honest object. Non-parametric
tests — Mann-Whitney U, or bootstrap CIs on the paired difference — are the
concrete next step if a claim needs to survive scrutiny.

**Q — Off-peak, fixed-time beats every agent. Is that not a failure?**

It is the expected result, and reporting it as such is the honest move. Off-peak
demand is 1,440 veh/h against a peak of 4,320 that the junction only just serves
— queues barely form, so a sensible fixed plan is already near-optimal at 0.39 s
and there is nothing for an adaptive controller to recover. The informative part is that three of four agents
stay within about 1.6 s of the baseline and none gridlocks.

*If pressed:* This half **survives the audit**. At 0.39 s nothing is stranded for
the survivorship bias to hide, and the ranking does not depend on the reward the
models were trained against. Two caveats: these runs have no completed-trip
metrics, and their baseline is still the 10 s cycler — a better static plan would
only widen the gap in fixed-time's favour.

**Q — Which algorithm is the winner?**

**None, and the question is premature.** Ranking four algorithms only means
something once one of them beats a competent static plan, and none does.
Off-peak, DQN is closest to the baseline (0.48 s vs 0.39 s) but does not beat it.
At peak there is no valid row at all.

### D.6 The safety-sampling defect and the fix

**Q — How did you find it?**

The tell was uniformity. `safety_exposure` was not small or noisy — it was
exactly 0.0 with zero variance in every row, across four algorithms, two
scenarios, five seeds and the fixed-time baseline. A quantity that varies by
nature and never varies at all is not a finding about traffic; it is a finding
about the measurement.

**Q — What was the actual cause?**

sumo-rl computes rewards and info only at action steps — after `_run_steps` has
advanced the simulation by `delta_time`. The amber interval is raised at the
start of that window and cleared `yellow_time` seconds in. Since sumo-rl asserts
`delta_time > yellow_time`, and we run 5 against 3, the amber is always over
before anything is measured. The accessors were correct; the sampling point was
wrong.

**Q — Why did your tests not catch it?**

Because they tested the function, not the timing. The unit tests fed a synthetic
traffic-signal stub with `is_yellow` set True and correctly asserted exposure was
counted. What no test covered was whether `is_yellow` is *ever* True at the moment
the environment calls that function. That is an integration property, and we had
no integration-level assertion.

**Q — How do you know the fix is correct and has not broken something else?**

Three checks. Twelve tests written failing-first. An end-to-end SUMO run showing
exposure fires in 38 of 120 decision windows. And a control: mean |efficiency|
measures 8.44 before and after, confirming the efficiency half of the reward is
untouched.

**Q — What else might be wrong that you have not found?**

That is the right question, and the answer is emphatically not "nothing" — the
audit that followed found six more things (D.8). The class of bug is "correct
function, wrong sampling point", and the general defence is an integration-level
assertion for every quantity that enters the reward or the results table.

### D.7 Next steps and cost

**Q — What should happen next, in order?**

1. **Raise `min_green` to 60 s before retraining anything.** Measured, not
   guessed. Wired: `env_common.DEFAULT_MIN_GREEN`, `$MIN_GREEN`, `--min-green` on
   `train.py`/`tune.py`.
2. **Re-tune at that floor**, since hyperparameters are selected for an action
   space.
3. **Score against the actuated controller**, not the static plan.
4. **Expect ≤10 %, mostly variance.** That expectation is the strongest argument
   for changing the problem rather than the optimiser.
5. Only then the λ ablation, which has never been run.

**Q — Is the λ ablation still worth running?**

Yes, but not first, and not on the current action space. `run_lambda_sweep.sh`
exists and is resumable. Running it at a 10 s floor would produce a
safety/efficiency trade-off curve for a controller that cannot control.

**Q — What is the path to a thesis-level contribution?**

The multi-intersection arterial corridor with coordinated multi-agent RL — IPPO
or MAPPO. The environment prototype is already built. Coordination across
junctions is the one thing a static plan cannot imitate, so it is the first
setting where this project's question can be answered in RL's favour.

*If pressed:* Note the corridor design doc's open decision "carry the winner
(DQN) forward" is void — that ranking is withdrawn, so the corridor work must
either re-run its own baselines or choose on other grounds.

### D.8 The audit — why the headline changed

**Q — Your earlier report said DQN and A2C cut waiting time 24 %. What happened?**

That claim is withdrawn. I audited the pipeline that produced it and found six
defects, five independent and any one sufficient to void the number. Corrected,
the result reverses: no learned policy we produced beats a competently timed
static plan.

1. **The metric was a gridlock clock.** `system_mean_waiting_time` averages over
   vehicles *still in the network*, so a deadlocked junction accumulates 1 s of
   "waiting" per second while everything that escapes stops counting. Peak locked
   at t = 780 s; the 1319 s figure is the area under that ramp. It also inverted
   the ranking — A2C deadlocked on 5 of 5 seeds and therefore scored *best*.
2. **The baseline ran on different traffic than the agents** — seed 0 vs seeds
   42–46. Fixed-time alone spans 242–1319 s across those seeds, and seed 0 was
   the worst draw. Paired per seed, every arm flips sign.
3. **Every evaluated model predates the safety fix** and was never retrained.
4. **The "±std over 5 seeds" was not seed variance** — only the reference seed's
   checkpoint was ever evaluated.
5. **The gridlock was a library default**, not the demand: sumo-rl ships
   `time_to_teleport = -1` where SUMO's own default is 300.
6. **The "fixed-time" baseline was a 10 s-green cycler**, not a fixed-time plan —
   and the worst point of the static sweep.

**Q — Withdrawn numbers, for reference only**

| Algorithm | Mean waiting (s) | ± std | vs fixed-time | Paired per seed |
|---|---:|---:|---:|---:|
| dqn | 1002.76 | 401.55 | −24.0 % | **+56.9 %** |
| a2c | 1003.33 | 86.14 | −24.0 % | **+67.6 %** |
| fixed-time | 1319.17 | — | baseline | — |
| ppo | 1356.53 | 45.03 | +2.8 % | **+118.4 %** |
| qrdqn | 1400.69 | 4.92 | +6.2 % | **+123.3 %** |

Quote these **only** as the numbers being withdrawn.

**Q — Isn't this just admitting the project failed?**

No. The measurement stack was wrong and is now right, and the corrected result is
a finding rather than an absence: at a 2-phase isolated junction there is little
for a learned controller to win, and we can say *why* — amber lost time — and
*how much* — a factor of 5.6 attributable to one parameter. Three of the six
defects concern how `sumo-rl` is commonly used rather than anything specific to
this codebase, which makes them worth reporting in their own right.

*If pressed:* The alternative was to present a 24 % improvement that a paired
comparison turns into a 57 % regression. Finding that ourselves, before the viva,
is the better outcome.

---

## Part E · Numbers and traps

### E.1 Memorise these

| Quantity | Value |
|---|---|
| **Peak, static plan (60 s, mid-plateau)** | **91.8 ± 19.9 s** delay/completed trip · 4076 trips · **94.3 %** completion · 14.97 s in-network |
| **Peak, queue-actuated (`min_green` 60)** | **82.5 ± 10.1 s** · 4156 trips · −9.3 ± 23.9 s paired · wins 3/5 seeds |
| **Peak, queue-actuated (`min_green` 10)** | **517.5 ± 208.4 s** · 2925 trips · **5.6× the fixed plan** · wins 0/5 |
| Static plateau | **45–90 s** green; paired differences ±13 s against a ~30 s seed spread |
| The 10 s plan | 298.6 s, **76 %** completion (20 s: 73 %) vs ~95 % on the plateau |
| Robustness, actuated vs static | trips 4142–4177 (spread **35**) vs 3834–4162 (spread **328**); delay sd 10.1 vs 19.9 |
| Amber arithmetic | 3 s yellow → **23 %** of cycle lost at 10 s green, **4.8 %** at 60 s |
| Switch requests, actuated | 125–168/episode at a 10 s floor vs 38–60 at 75–90 s |
| The 4× metric gap | 14.97 s in-network vs 66.4 s waiting per completed trip, same runs |
| Off-peak mean wait | fixedtime **0.391 s** · dqn 0.48 · ppo 1.76 · qrdqn 1.99 · a2c 36.0 (mobile, 4.75 m/s) |
| Demand | base 2,880 veh/h · peak ×1.5 = **4,320** · off-peak ×0.5 = 1,440 |
| Safety terms, peak seed 0 | brake 0.206 → **5.07** · exposure 0.0 → **11.57** · mean\|eff\| **8.44 both sides** |
| `SAFETY_SCALE` | 0.024 → **2.1298** (= 17.97 / 8.44) |
| Timing | decision 5 s · amber 3 s · **min green 60 s (was 10)** · max green 60 s **but inert** · episode 3600 s · teleport **300 s** |
| Tests | **53 green** |
| **Withdrawn** | ~~peak DQN/A2C −24 % (1319.2 → 1002.8 / 1003.3 s)~~ · paired: +56.9 / +67.6 / +118 / +123 % |

### E.2 The traps

Each is a place where the obvious phrasing is wrong.

- **Do not name an algorithm winner.** There is no valid peak ranking. Off-peak
  DQN is closest but does not beat the baseline. The old "DQN best mean, A2C best
  reliability" line came from the withdrawn table.
- **Do not say the −9.3 s actuated gain is an improvement.** It is inside the
  noise (paired sd 23.9 s). The defensible claim is *consistency*: 35 trips of
  spread against 328.
- **Do not quote 75 s as "the optimal green".** It is the sample minimum and it
  loses to 60 s on four of five seeds. Say **plateau, 45–90 s**. Quoting the
  argmin repeats defect 2 at smaller scale, in front of the panel.
- **Do not say the safety term reduces accidents.** SUMO does not model crashes.
  It reduces two conflict surrogates.
- **Do not call the Stage-1 baseline an optimised fixed-time plan.** It was a 10 s
  cycler — the worst point of the sweep — and that is defect 6, not a caveat.
- **Do not say "our RL failed".** Say the training budget was spent at a floor
  where no controller can win, so the null is over-determined. The distinction is
  the whole contribution.
- **Do not treat a null from the pilot retrain as evidence of no headroom.** At
  ~42 episodes a positive result is informative and a null is weak — it cannot
  separate "no signal" from "too few episodes".

### E.3 Questions to put to your supervisor

- **Scope.** Is the corrected negative result plus the methodological audit an
  acceptable deliverable, or is a positive RL result required? That decides
  whether to spend the remaining time on the corridor or on retraining here.
- **Safety weighting.** Post-fix, exposure outweighs braking, so λ now means
  "clear the junction" more than "brake gently". Is that the intended emphasis?
  It should be stated before the λ sweep is interpreted.
- **Seed budget.** Five seeds leaves a ~24 s paired sd on the comparison that
  matters. Buy more seeds and run a non-parametric test, or proceed and report
  the spread?
