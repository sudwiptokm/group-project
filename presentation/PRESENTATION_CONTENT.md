# Presentation Content — RL Traffic Signal Control (Group 7)

> ## Revision note — 2026-08-12
>
> The peak slides were rewritten after an audit found six defects in how those
> numbers were produced (slide 7). The DQN/A2C −24% claim **is withdrawn**;
> corrected, no learned policy here beats a competently timed static plan.
> Measurements and reproduction: `docs/FINDINGS_2026-08-12.md`.
>
> **The off-peak slides are unaffected and stand as originally written.**


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

## Slide 7 — Peak: the result we reported, and why it is withdrawn

| Algorithm | Mean wait (s) | ± std | vs fixed-time | |
|---|--:|--:|--:|---|
| DQN | 1002.8 | 402 | −24.0% | **withdrawn** |
| A2C | 1003.3 | 86 | −24.0% | **withdrawn** |
| fixed-time | 1319.2 | — | baseline | **not a fixed-time plan** |
| PPO | 1356.5 | 45 | +2.8% | **withdrawn** |
| QR-DQN | 1400.7 | 5 | +6.2% | **withdrawn** |

**Six defects. Five independent. Any one voids the number.**

1. **The metric was a gridlock clock** — `system_mean_waiting_time` averages over
   vehicles *still in the network*, so under deadlock it counts 1 s per second.
   Peak locked at t = 780 s. A2C deadlocked 5/5 seeds and scored **best**.
2. **Baseline ran on different traffic** — baseline seed 0, agents seeds 42–46.
   Fixed-time spans 242–1319 s across seeds (5.4×); seed 0 was the worst draw.
   Paired per seed: DQN −23.9% → **+56.9%**, A2C → **+67.6%**, PPO → **+118%**,
   QR-DQN → **+123%**.
3. **Every model predates the safety fix** and was never retrained.
4. **The ±std was not seed variance** — only the reference seed's checkpoint was
   ever evaluated; ±402 is one policy's spread across five demand draws.
5. **The gridlock was a library default** — `sumo-rl` ships
   `time_to_teleport = -1` (SUMO's own default is 300), which makes deadlock
   absorbing. The demand level needed no change.
6. **The "fixed-time" baseline was a 10 s-green cycler** — the worst point of the
   sweep on the next slide.

- *Do not show* `results/bars_peak_lam05.png` or `improvement_peak_lam05.png` —
  both plot the withdrawn numbers.
- Measurements + reproduction: `docs/FINDINGS_2026-08-12.md`.

---

## Slide 7b — Peak: what actually controls performance

| green (s) | delay per completed trip | trips completed | vs 60 s, paired |
|---|--:|--:|--:|
| 10 | 298.6 ± 75.7 | **76%** | +207, loses 5/5 |
| 20 | 384.7 ± 39.4 | **73%** | +293, loses 5/5 |
| 30 | 257.4 ± 133.6 | 87% | +166, loses 4/5 |
| **45** | **104.9 ± 38.2** | 96% | +13, wins 3/5 |
| **60** | **91.8 ± 19.9** | 94% | — |
| **75** | **91.3 ± 6.6** | 95% | −0.4, wins 1/5 |
| **90** | **103.4 ± 18.1** | 95% | +12, wins 1/5 |
| 120 | 110.8 ± 6.0 | 94% | +19, wins 1/5 |

Peak 1.5×, seeds 42–46, 3600 s episodes, `--time-to-teleport 300`
(`analysis/static_timing.py`). *Plot:* `results/static_timing_peak.png`.

- **The 10 s plan Stage 1 called "fixed-time" leaves a quarter of the traffic
  unserved** — 76% of trips complete against ~95% on the plateau. Those are
  exactly the vehicles the in-network metric stops counting.

- **A long green wins, and it does not need tuning to.** Performance is flat
  across **45–90 s** — paired on the same seeds those greens differ by ±13 s
  against a ~30 s seed-to-seed spread — and every plan in that band beats every
  learned policy this project produced by 2–3×. Say "plateau", not "optimum":
  nobody can dismiss this as an unfairly hand-tuned baseline.
- **Mechanism — lost time to amber.** `yellow_time` = 3 s, so each switch serves
  nobody for 3 s: **23%** of the cycle at a 10 s green, **4.8%** at 60 s. The
  agent decides every 5 s with `min_green` = 10 s — exactly where switching is
  cheap to try and ruinous to pay for — and `diff_waiting_time` only surfaces
  that cost several decisions later.
- **No valid RL row exists at peak**, and we do not invent one: the checkpoints
  predate the safety fix and two 20k-step retraining attempts produced no
  learning. Stage-1 policies sat at 20–33 s where a 60 s static plan sits at
  11.5 s — same 1200 s episodes, same in-network metric, so those two are
  comparable to each other but not to the 3600 s numbers above.
- **Also found:** `sumo-rl` stores `max_green` and never reads it — our
  `max_green = 60` constrains nothing, which is why the sweep runs past 60 s.

---

## Slide 7c — Did our RL fail, or was there nothing to find?

The static result fits both readings, and **more training separates neither** — a
second null is consistent with each. So we tested it with a controller that has
nothing to learn: **queue-actuated**, perfect queue information, no reward, no
credit assignment, no sample budget. If *it* cannot beat the best static plan,
the headroom is not there.

| `min_green` (s) | delay / completed trip | trips done | vs static 60 s, paired |
|---|--:|--:|--:|
| **10** | 517.5 ± 208.4 | 2925 | +426, wins 0/5 |
| 20 | 337.0 ± 220.5 | 3455 | +245, wins 1/5 |
| 30 | 186.1 ± 108.0 | 3876 | +94, wins 1/5 |
| 45 | 161.9 ± 64.2 | 4022 | +70, wins 1/5 |
| **60** | **82.5 ± 10.1** | **4156** | **−9.3, wins 3/5** |
| 75 | 92.2 ± 0.9 | 4119 | +0.4, wins 1/5 |
| 90 | 118.7 ± 23.7 | 4038 | +27, wins 0/5 |

Peak 1.5×, seeds 42–46, 3600 s episodes, `--time-to-teleport 300`
(`analysis/actuated.py` + `analysis/headroom.py`; rows in
`analysis/actuated_sweep.csv`).

- **`min_green` was the binding constraint — not the algorithm.** At the 10 s
  floor this project trained on, a controller that *cannot* be accused of
  under-training is **5.6× worse** than a fixed plan and strands a quarter of the
  traffic (2925 trips vs 4076; 2008 on the worst seed). It requests **125–168
  switches per episode** against 38–60 at a 75–90 s floor, each costing 3 s of
  amber. **The entire peak training budget was spent where no controller can
  win** — which is why the peak null is over-determined.
- **The curve is U-shaped and has turned by 90 s**, so 60 s is an interior
  optimum, not "longer is always better".
- **Read the 60 s row honestly.** −9.3 s against the static plan is *inside the
  noise* — the paired difference has sd 23.9 s. The mean is not the finding.
  What is resolvable is consistency: delay sd **10.1 s vs 19.9 s**, and trips
  completed **4142–4177 (spread 35) vs 3834–4162 (spread 328)**. Static's bad
  draw is seed 43 (126.3 s, 3834 trips); actuated takes that seed at 83.1 s and
  4146 trips. **The adaptive gain is not a lower mean — it is not having a bad
  seed.**
- **So: neither "we failed" nor "nothing exists".** At `min_green` = 10 there was
  nothing to find; at 60 there is, but it is a ~10% variance reduction that a
  *non-learning* controller already collects. **The bar RL has to clear is the
  actuated controller (82.5 ± 10.1 s), not the static plan** — and by enough to
  beat a 24 s paired sd.

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
- **This half survives the audit.** At 0.39 s there is no headroom, so nothing is
  stranded for the metric defect to hide, and the ranking does not depend on the
  reward the models were trained against. Caveat to state if asked: the baseline
  here is still the 10 s cycler, and a better static plan would only widen the
  gap against RL.
- *Plots:* `results/bars_offpeak_lam05_logy.png` (waiting, log scale — readable)
  and `results/speed_offpeak_lam05.png` (mobility: a2c collapses to 4.75 m/s, the
  rest stay ~10 m/s → all mobile, a2c weakest)

---

## Slide 9 — Reading the results honestly

- **Peak:** no valid RL result. The standard to beat is **any static plan in the
  45–90 s green band**, and the Stage-1 policies were **2–3× worse** than it. The cause is structural —
  amber lost time at a 2-phase junction — not a training budget we can buy.
- **`min_green` = 10 was the binding constraint, and we measured it.** A
  controller that learns nothing is 5.6× worse than the fixed plan at that floor
  and matches it at 60 s, so the peak null is **over-determined**. Fix the floor
  before retraining anything; afterwards the bar is the **actuated controller**
  (82.5 ± 10.1 s), not the static plan — and even then the prize is ~10%.
- **Off-peak:** the baseline is near-optimal; RL *cannot* beat it — an honest
  **ceiling**, not a failure. All agents stay mobile.
- **λ ablation:** never run. Only λ = 0.5 exists, so "safety-aware" is in the
  title and not yet in the results. Say this before anyone asks.
- **No algorithm winner is claimed.** Ranking four algorithms only means
  something once one of them beats a competent static plan; none does.
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
- **This is a *qualitative* demo — the agent in action.** The clip is a Stage-1
  checkpoint whose peak performance claim is withdrawn: it shows the mechanism,
  not a result. Do not quote a number off it.
- Do **not** show a live fixed-time-vs-RL race: the RL side has no valid peak
  measurement behind it, and a short window proves nothing either way.

---

## Slide 14 — Conclusions

- **At an isolated 2-phase intersection, a competently timed static plan is hard
  to beat — and we can say why.** Amber lost time dominates at the switching
  frequency the action space allows, so there is little left for a policy to win.
- That is a **result about the problem**, not a training failure: one mechanism
  explains Stage 1, the pilot, and the 20k-step null together.
- **Off-peak confirms the same ceiling** from the other side — 0.39 s, all four
  agents mobile, none ahead.
- The **methodology findings travel further than the controller did** — three of
  the six defects concern how `sumo-rl` is commonly used, not this codebase.
- **Next:** multi-intersection corridor with coordinated MARL (prototype built).
  Coordination across junctions is the one thing a static plan cannot do — which
  is exactly what the single-intersection finding exposes.

---

## Slide 15 — Thank you / Q&A

Group 7 · RL Traffic Signal Control · SUMO + Stable-Baselines3
