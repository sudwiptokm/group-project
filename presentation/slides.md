---
marp: true
theme: default
paginate: true
size: 16:9
header: 'RL Traffic Signal Control — Group 7'
style: |
  section { font-size: 24px; }
  h1 { color: #1a3a5c; }
  h2 { color: #1a3a5c; }
  table { font-size: 20px; }
  section.lead h1 { font-size: 46px; }
  .small { font-size: 18px; }
  .green { color: #1a7f37; font-weight: bold; }
  .red { color: #b42318; font-weight: bold; }
---

<!-- _class: lead -->

# Reinforcement Learning for Traffic Signal Control

### Heterogeneous urban traffic · SUMO + Stable-Baselines3

**Group 7**

A fair, tuned comparison — DQN · QR-DQN · PPO · A2C — against a fixed-time baseline, with a safety-aware reward.

---

## The problem

- Urban intersections are congested; **fixed-time signals** ignore live demand.
- Traffic here is **heterogeneous**: motorcycles, auto-rickshaws, and cars share the road and behave very differently.
- Raw vehicle counts mislead a controller — 3 motorbikes ≠ 3 cars in road-space.
- **Question:** can a reinforcement-learning controller beat the fixed-time baseline, and *which* algorithm is best?

---

## Objectives

1. Build a realistic single-intersection SUMO model with **mixed vehicle types**.
2. Design an RL controller with a **PCU-weighted** observation and a **safety-aware** reward.
3. Run a **fair comparison** of four RL algorithms — same environment, reward, and observation; only the algorithm changes.
4. Benchmark all of them against the **fixed-time baseline**, across **peak** and **off-peak** demand.

---

## The environment

- **Network:** 4-arm intersection, built in SUMO (netedit).
- **Vehicles:** motorcycles / auto-rickshaws / cars (60 / 25 / 15 %), sublane model so 2-wheelers filter realistically.
- **Observation — PCU-weighted:** the agent sees *passenger-car equivalents*, not raw counts (moto 0.3, auto 0.5, car 1.0), plus phase one-hot, min-green flag, density and queue.
- **Action:** discrete **phase selection** (which green phase to run next).
- **Simulator:** `sumo-rl` `SumoEnvironment` over TraCI; agents from Stable-Baselines3 / sb3-contrib.

---

## The safety-aware reward

$$ \text{reward} = \Delta\,\text{waiting\_time} \; - \; \lambda \cdot \frac{\text{safety\_penalty}}{\text{SAFETY\_SCALE}} $$

- **Efficiency term:** reduction in total waiting time (throughput).
- **Safety term:** composite, **vulnerability-weighted** penalty — emergency braking + intersection exposure during yellow — each vehicle weighted by its fragility (moto 1.0 / auto 0.6 / car 0.3).
- **λ** is held **identical across all four algorithms** in a comparison → apples-to-apples.
- λ = 0 → pure efficiency (exact ablation baseline); λ > 0 → adds the safety cost.

<span class="small">Reported results use the reference **λ = 0.5** — with the **braking component only**. A sampling defect (since fixed) meant the exposure component never fired in these runs; efficiency results are unaffected. See Limitations.</span>

---

## Method — a fair 4-algorithm ladder

| | |
|---|---|
| **Algorithms** | DQN (baseline), QR-DQN, PPO, A2C |
| **Shared** | same env, reward, observation, action space, seeds, eval protocol |
| **Tuning** | Optuna hyperparameter search **per algorithm** |
| **Training** | 5 seeds each |
| **Evaluation** | 5 held-out seeds → mean ± std |
| **Scenarios** | peak (oversaturated) + off-peak (light) |
| **Baseline** | fixed-time controller through the *same* eval pipeline |

<span class="small">Not SAC — it is continuous-action and doesn't fit discrete phase selection. QR-DQN (distributional DQN) is the fourth rung instead.</span>

---

## Peak — the result we reported, and why it is withdrawn

| Algorithm | Mean wait (s) | vs fixed-time | |
|---|--:|--:|---|
| DQN | 1002.8 ± 402 | −24.0% | <span class="red">withdrawn</span> |
| A2C | 1003.3 ± 86 | −24.0% | <span class="red">withdrawn</span> |
| fixed-time | 1319.2 | baseline | <span class="red">not a fixed-time plan</span> |
| PPO | 1356.5 ± 45 | +2.8% | <span class="red">withdrawn</span> |
| QR-DQN | 1400.7 ± 5 | +6.2% | <span class="red">withdrawn</span> |

**Six defects. Five independent. Any one voids the number.**

1. The metric was a **gridlock clock** — `system_mean_waiting_time` averages over vehicles *still in the network*, so it counts 1 s/s under deadlock. A2C deadlocked 5/5 seeds and scored **best**.
2. **Baseline ran on different traffic** — baseline on seed 0, agents on 42–46. Fixed-time spans 242–1319 s across seeds (5.4×); seed 0 was the worst draw.
3. Every model **predates the safety fix** and was never retrained.
4. The **"±std over 5 seeds" was one policy's demand spread** — 4 of 5 models per cell were never evaluated.
5. The gridlock was a **library default** (`time_to_teleport = -1`), not the demand.
6. The **"fixed-time" baseline was a 10 s-green cycler** — the worst point of the sweep on the next slide.

<span class="small">Measurements and reproduction: `docs/FINDINGS_2026-08-12.md`. Paired per seed, the direction reverses: DQN −23.9% → **+56.9%**, A2C → **+67.6%**, PPO → **+118%**, QR-DQN → **+123%**.</span>

---

## Peak — what actually controls performance

<div class="columns" style="display:grid; grid-template-columns: 1.05fr 1fr; gap:18px; align-items:center;">
<div>

| green (s) | delay / completed trip | trips done | vs 60 s |
|---|--:|--:|--:|
| 10 | 298.6 ± 75.7 | <span class="red">76%</span> | +207, 0/5 |
| 20 | 384.7 ± 39.4 | <span class="red">73%</span> | +293, 0/5 |
| 30 | 257.4 ± 133.6 | 87% | +166, 1/5 |
| **45** | **104.9 ± 38.2** | 96% | +13, 3/5 |
| **60** | **91.8 ± 19.9** | 94% | — |
| **75** | **91.3 ± 6.6** | 95% | −0.4, 1/5 |
| **90** | **103.4 ± 18.1** | 95% | +12, 1/5 |
| 120 | 110.8 ± 6.0 | 94% | +19, 1/5 |

</div>
<div>

![w:520](../results/static_timing_peak.png)

</div>
</div>

**The 10 s plan Stage 1 called "fixed-time" leaves a quarter of the traffic unserved** (76% of trips complete vs ~95% on the plateau) — exactly the vehicles the in-network metric then stops counting.

<span class="small">Peak 1.5×, seeds 42–46, 3600 s episodes, `--time-to-teleport 300`. Reproduce: `analysis/static_timing.py`; figure `results/static_timing_peak.png`.</span>

**A plateau, not a tuned optimum.** Paired on the same seeds, greens from **45 s to 90 s** differ by ±13 s against a seed-to-seed spread of ~30 s. The static plan that beats every learned policy needed **no tuning skill to find** — which is why this is not an unfairly optimised baseline.

**Mechanism — lost time to amber.** `yellow_time` = 3 s, so every switch spends 3 s serving nobody: **23%** of the cycle at a 10 s green, **4.8%** at 60 s. The agent decides every 5 s with `min_green` = 10 s — exactly where switching is cheap to attempt and ruinous to pay for — and `diff_waiting_time` only surfaces that cost several decisions later.

**There is no valid RL row at peak.** The models predate the safety fix, and two 20k-step retraining attempts produced no learning. Stage-1 policies sat at 20–33 s where a static 60 s plan sits at 11.5 s — a factor of 2–3 the wrong way (same 1200 s episodes and in-network metric; not comparable to the 3600 s figures).

---

## Peak — "our RL failed" or "nothing to find"?

The static result fits both, and **more training separates neither** — a second null is consistent with each. So we tested it with a controller that has *nothing to learn*: queue-actuated, perfect queue information, no reward, no sample budget. If **it** can't beat the best static plan, the headroom isn't there.

<div class="columns" style="display:grid; grid-template-columns: 1fr 1fr; gap:18px; align-items:start;">
<div>

| `min_green` | delay / trip | trips done | vs static 60 s |
|---|--:|--:|--:|
| <span class="red">**10**</span> | <span class="red">517.5 ± 208.4</span> | <span class="red">2925</span> | <span class="red">+426, 0/5</span> |
| 20 | 337.0 ± 220.5 | 3455 | +245, 1/5 |
| 30 | 186.1 ± 108.0 | 3876 | +94, 1/5 |
| 45 | 161.9 ± 64.2 | 4022 | +70, 1/5 |
| **60** | **82.5 ± 10.1** | **4156** | **−9.3, 3/5** |
| 75 | 92.2 ± 0.9 | 4119 | +0.4, 1/5 |
| 90 | 118.7 ± 23.7 | 4038 | +27, 0/5 |

</div>
<div>

**`min_green` was the binding constraint — not the algorithm.**

At the **10 s floor this project ran on**, a controller that *cannot* be under-trained is <span class="red">**5.6× worse**</span> than a fixed plan and strands a quarter of the traffic. 125–168 switches/episode × 3 s amber.

**The whole peak training budget was spent where nothing can win.**

</div>
</div>

**Read the 60 s row honestly:** −9.3 s is **inside the noise** (paired sd 23.9 s). The mean is not the finding — <span class="green">consistency</span> is:

| | delay sd | trips completed |
|---|--:|--:|
| static 60 s | 19.9 s | 3834–4162 (spread **328**) |
| actuated, `min_green` 60 | 10.1 s | 4142–4177 (spread **35**) |

Static's bad draw is seed 43 (126.3 s, 3834 trips); actuated takes that seed at 83.1 s, 4146 trips. **The adaptive gain is not a lower mean — it's not having a bad seed.**

<span class="small">So: neither "we failed" nor "nothing exists". At `min_green` = 10 there was nothing to find; at 60 there is, but it's a ~10% variance reduction a *non-learning* controller already collects. **The bar RL must clear is the actuated controller (82.5 ± 10.1 s), not the static plan.** Reproduce: `analysis/actuated.py`, `analysis/headroom.py`; rows in `analysis/actuated_sweep.csv`.</span>

---

## Results — off-peak demand (light traffic)

<div class="columns" style="display:grid; grid-template-columns: 1fr 1fr; gap:20px; align-items:center;">
<div>

| Algorithm | Mean wait (s) | Speed (m/s) |
|---|--:|--:|
| fixed-time | 0.39 | 10.40 |
| DQN | 0.48 ± 0.01 | 10.50 |
| PPO | 1.76 ± 0.07 | 10.01 |
| QR-DQN | 1.99 ± 0.23 | 9.91 |
| A2C | 36.0 ± 1.5 | 4.75 |

Fixed-time is **already near-optimal** (0.39 s) — no RL agent can improve on it. The real result: **all four stay mobile**. DQN is within a hair; A2C is weakest but valid (no gridlock).

<span class="small">**This half survives the audit.** At 0.39 s there is no headroom, so nothing is stranded for the metric defect to hide, and the ranking does not depend on the reward the models were trained against. The baseline here is still the 10 s cycler — a better static plan would only widen the gap.</span>

</div>
<div>

![w:420](../results/bars_offpeak_lam05_logy.png)
![w:420](../results/speed_offpeak_lam05.png)

</div>
</div>

---

## Reading the results honestly

- **Peak:** no valid RL result. The standard to beat is **any static plan in the 45–90 s green band**, and Stage-1 policies were **2–3× worse** than it. The cause is structural — amber lost time at a 2-phase junction — not a training budget we can buy our way out of.
- **`min_green` = 10 was the binding constraint**, and we measured it: a controller that learns nothing is **5.6× worse** than the fixed plan at that floor, and matches it at 60 s. The peak null is **over-determined** — the budget was spent where no controller can win. Fix the floor first; the bar afterwards is the **actuated controller**, not the static plan.
- **Off-peak:** the baseline is near-optimal; RL *cannot* beat it — the honest finding is a **ceiling**, not a failure. All agents stay mobile.
- **λ ablation:** never run. Only λ = 0.5 exists — "safety-aware" is in the title and not yet in the results.
- **One disclosed asymmetry:** off-peak A2C's hyperparameters were selected on *waiting time* rather than the shaped reward. At light demand the `−λ·safety` term dominates, so the reward-optimal policy is "never switch" = gridlock. Tuning on waiting time rejects that collapse. Training reward, env, and eval stay identical across all algorithms.

---

## Work update — safety reward: defect found and fixed

**Flagged last review:** `safety_exposure` = 0 in every run; λ appeared to act through braking alone.

**Root cause — structural, not a data quirk.** sumo-rl evaluates the reward only *after* the decision window has elapsed. `set_next_phase()` raises `is_yellow`; `TrafficSignal.update()` clears it after `yellow_time`; sumo-rl asserts `delta_time > yellow_time` (we run **5 > 3**). So `is_yellow` was **always False** whenever the penalty was measured:

- **exposure** — unreachable by construction: `0.0` with **zero variance** in every Stage-1 row, all four algorithms *and* the fixed-time baseline
- **braking** — sampled 1 second in 5, always the settled post-yellow second: <span class="red">biased low</span>, not merely sparse

**Fix:** per-simulation-second accumulator; reward and logged metrics now read identical totals. 12 new tests written failing-first, suite 24/24 green.

| term (peak, seed 0) | before | after |
|---|---|---|
| `safety_brake` | 0.206 | 5.07 |
| `safety_exposure` | 0.0 | 11.57 |
| `mean\|efficiency\|` | 8.44 | <span class="green">8.44 — unchanged</span> |

<span class="small">`SAFETY_SCALE` re-derived 0.024 → 2.1298. At λ=0.5 the safety term is 3.91 against |efficiency| 5.64 — intended weighting preserved.</span>

---

## Work update — what the audit changed

**The measurement stack is now honest, and it cost us the peak headline.**

| | was | now |
|---|---|---|
| Ranking metric | in-network mean wait (survivorship-biased) | **delay per completed trip** + throughput + completion rate, from SUMO `tripinfo` |
| "Fixed-time" | 10 s-green cycler | **a swept static plan** (`baseline.py --green`), reported mid-plateau at 60 s |
| Baseline seeds | seed 0 only | the **agents' own seeds**, paired |
| Deadlock | absorbing (`time_to_teleport = -1`) | SUMO default 300 → peak is measurable, ~9 teleports/episode |

<span class="small">The two metrics disagree by **4×** on the same peak runs — 14.97 s in-network vs 66.43 s waiting per completed trip. That gap *is* the survivorship bias, measured.</span>

**Also found:** sumo-rl stores `max_green` and never reads it, so the `max_green = 60` in our env constrains nothing — for the static sweep or for a learned policy.

**Also found:** a non-learning queue-actuated controller settles the "did our RL fail, or is there nothing to find?" question — see the headroom slide.

**Next, in order:** (1) **raise `min_green` to 60 s** — measured, not guessed — and retrain there; (2) score it against the **actuated controller** (82.5 ± 10.1 s), not the static plan, since tying a policy that needs no training proves nothing; (3) *then* run the λ ablation, which has never been run. More training on the **current** action space is not the missing ingredient — and even on a corrected one the prize is ~10%, which is the argument for the corridor.

---

## Live demo — trained DQN agent (peak)

*(play `demo_dqn_peak.mp4` — DQN controlling peak traffic, live metrics HUD)*

- Watch: PCU-weighted queues build and clear as the agent switches phase under
  mixed moto/auto/car traffic.
- Colour by type — <span style="color:#e8820c">orange moto</span>, <span style="color:#2f6fd0">blue auto</span>, grey car.

<span class="small">Qualitative only — the agent in action. This clip is a Stage-1 checkpoint, and its peak performance claim is withdrawn; it shows the mechanism, not a result.</span>

---

## Conclusions

- **At an isolated 2-phase intersection, a competently timed static plan is hard to beat — and we can now say why.** Amber lost time dominates at the switching frequency the action space allows, so the policy space contains little the plan does not already have.
- That is a **result about the problem**, not a training failure: it explains Stage 1, the pilot, and the 20k-step null with one mechanism.
- **Off-peak confirms the same ceiling** from the other side: at 0.39 s there is no headroom, and all four agents stay mobile.
- The **methodology findings travel further than the controller did** — three of the six defects are in how `sumo-rl` is commonly used, not in this codebase.

**Future work — the direction the finding points at:** a **multi-intersection arterial corridor** with coordinated multi-agent RL. Coordination across junctions is the one thing a static plan cannot do, which is exactly what the single-intersection result exposes. Env prototype already built.

---

<!-- _class: lead -->

# Thank you

### Questions?

<span class="small">Group 7 · RL Traffic Signal Control · SUMO + Stable-Baselines3</span>
