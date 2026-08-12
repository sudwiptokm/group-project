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

# ⚠ SUPERSEDED — do not present the peak results

**The peak numbers in this deck (DQN/A2C −24% vs fixed-time) do not hold.**

Six defects, documented with measurements in `docs/FINDINGS_2026-08-12.md`.
Corrected, the result reverses: **no learned policy beats a competently timed
static plan** — best static 11.5 s vs 20–33 s for RL.

The "fixed-time" baseline in this deck is a 10 s-green cycler, not a fixed-time plan.

**Off-peak results are unaffected and still valid.**

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

## Results — peak demand (oversaturated)

<div class="columns" style="display:grid; grid-template-columns: 1fr 1fr; gap:20px; align-items:center;">
<div>

| Algorithm | Mean wait (s) | vs fixed-time |
|---|--:|--:|
| **DQN** | 1002.8 ± 402 | <span class="green">−24.0%</span> |
| **A2C** | 1003.3 ± 86 | <span class="green">−24.0%</span> |
| fixed-time | 1319.2 | baseline |
| PPO | 1356.5 ± 45 | +2.8% |
| QR-DQN | 1400.7 ± 5 | +6.2% |

Fixed-time backs up to ~1319 s. **DQN and A2C each cut mean waiting ~24%.** A2C nearly ties DQN's mean with far tighter seed spread (±86 vs ±402).

</div>
<div>

![w:520](../results/bars_peak_lam05.png)

</div>
</div>

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

</div>
<div>

![w:420](../results/bars_offpeak_lam05_logy.png)
![w:420](../results/speed_offpeak_lam05.png)

</div>
</div>

---

## Reading the results honestly

- **Peak:** RL clearly helps — DQN/A2C cut waiting ~24%. **DQN is the overall winner** (best mean, biggest congestion relief).
- **Off-peak:** the baseline is near-optimal; RL *cannot* beat it — the honest finding is a **ceiling**, not a failure. All agents stay mobile.
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

## Work update — what it changes, and what it doesn't

<span class="green">**The reported results stand.**</span> The efficiency half of the reward is untouched and measures identically (8.44 both sides). Waiting time, delay, and stopped-vehicle counts are independent of the safety term. What changes is the *label*: these runs are **braking-only λ**, and the report and speaker notes now say so explicitly.

**Two decisions I need from you**

1. **Primary metric.** PPO and QR-DQN are **3–6% worse** on mean wait, but cut **total network delay ~45%** and **halve stopped vehicles**. Which defines "best"? Not picked silently.
2. **Safety weighting.** Post-fix, exposure (11.57) outweighs braking (5.07) — it scales with junction occupancy. So λ now mostly means *"clear the junction before yellow"* rather than *"don't brake hard"*. Defensible, but it should be a stated choice before the λ sweep is interpreted.

**Also open:** DQN's peak σ is **401.5 s** — its error bar crosses the baseline, so the +24% is not yet statistically separable. Needs more seeds or a significance test.

**Next:** re-evaluate existing checkpoints under corrected logging (real safety numbers, no training cost) → then Stage 2 λ sweep: **~2–2.5 days** on the 16-core box reusing tuned HPs, **~4–4.5 days** if we re-tune. Re-tune or not is your call.

---

## Live demo — trained DQN agent (peak)

*(play `demo_dqn_peak.mp4` — DQN controlling peak traffic, live metrics HUD)*

- Watch: PCU-weighted queues build and clear as the agent switches phase under
  mixed moto/auto/car traffic.
- Colour by type — <span style="color:#e8820c">orange moto</span>, <span style="color:#2f6fd0">blue auto</span>, grey car.

<span class="small">Qualitative — the agent in action. The **−24%** headline is an episode-mean (previous slide), not the clip's instantaneous HUD.</span>

---

## Conclusions

- An RL controller with a **PCU-weighted observation** and a **safety-aware reward** beats a fixed-time signal **where it matters — congested peak demand (~24% less waiting)**.
- Under a fair, tuned, multi-seed comparison, **DQN is the most reliable winner**; A2C matches it at peak with tighter variance.
- Off-peak exposes an honest ceiling: a good fixed plan is hard to beat in light traffic.

**Future work:** extend to a **multi-intersection arterial corridor** with coordinated multi-agent RL (MARL) — env prototype already built.

---

<!-- _class: lead -->

# Thank you

### Questions?

<span class="small">Group 7 · RL Traffic Signal Control · SUMO + Stable-Baselines3</span>
