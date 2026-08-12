# Speaker Notes + Viva Q&A Prep — Group 7

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


**RL Traffic Signal Control · SUMO + Stable-Baselines3**
Presenters: **Sudwipto Kumar Mondal**, **Swatej Parmar**, **Aleana Biju**

Suggested split (flexible):
- **Sudwipto** — opens (problem / objectives / method), results, conclusion.
- **Swatej** — environment + SUMO model.
- **Aleana** — reward, results-honesty, live demo.

Target: **~10–12 min** talk. Time budgets below sum to ~11 min. Leave slack for Q&A.

---

# PART A — Speaker Notes (slide-by-slide)

## Slide 1 — Title *(~30s · Sudwipto)*
- Greet, introduce Group 7 and the three of us by name.
- One-line hook: "We taught a traffic light to think — and we tested whether it actually beats a normal fixed timer."
- Frame the deck: a **fair, tuned comparison** of four RL algorithms against a fixed-time baseline, with a **safety-aware** reward, on **heterogeneous** urban traffic.
- Don't dwell — this is a landing slide.

## Slide 2 — The problem *(~1 min · Sudwipto)*
- Fixed-time signals run on a clock; they ignore what's actually queued.
- Our traffic is **mixed**: motorcycles, auto-rickshaws, cars — very different footprints and filtering behaviour.
- Key insight to land: **raw counts mislead** — 3 motorbikes take far less road-space than 3 cars. A naive counter over-reacts to two-wheelers.
- Pose the two questions we answer: (1) can RL beat fixed-time, and (2) **which** algorithm is best?

## Slide 3 — Objectives *(~45s · Sudwipto)*
- Four concrete goals: build a realistic mixed-traffic SUMO intersection; design a PCU-weighted, safety-aware RL controller; run a **fair** four-algorithm comparison; benchmark all four vs fixed-time across peak and off-peak.
- Stress the word **fair** — same environment, reward, observation; only the algorithm changes. That's the backbone of the whole project.
- **[HANDOFF]** "I'll hand over to Swatej to walk through the environment we built."

## Slide 4 — Environment (what's done + how) *(~1.5 min · Swatej)*
- Lead with the headline: **a complete, working end-to-end RL traffic-control pipeline** — not a toy.
- Network: 4-arm intersection built in SUMO's netedit.
- Vehicles: moto / auto / car at 60/25/15%, with the **sublane model** so two-wheelers filter through gaps realistically — this is what makes the traffic genuinely heterogeneous.
- Observation is **PCU-weighted**: the agent sees passenger-car equivalents (moto 0.3, auto 0.5, car 1.0), plus phase one-hot, min-green flag, density and queue.
- Action: **discrete phase selection** — the agent picks which green phase runs next.
- Stack: `sumo-rl`'s SumoEnvironment over TraCI, agents from Stable-Baselines3 / sb3-contrib.

## Slide 5 — Safety-aware reward *(~1.5 min · Swatej → Aleana can take reward if preferred)*
- Show the equation: `reward = Δwaiting_time − λ·(safety_penalty / SAFETY_SCALE)`.
- **Efficiency term:** reduction in total waiting time — the throughput driver.
- **Safety term:** composite, **vulnerability-weighted** — emergency braking + intersection exposure during yellow. Each vehicle weighted by fragility: **moto 1.0 / auto 0.6 / car 0.3** (note: inverse of the PCU weights — a motorbike is small in road-space but the most vulnerable).
- **λ is identical across all four algorithms** — that's what keeps the comparison honest.
- λ = 0 gives a pure-efficiency ablation; we report the reference **λ = 0.5**.
- **[HANDOFF]** "Back to Sudwipto for the experimental method."

## Slide 6 — Method / experiment *(~1 min · Sudwipto)*
- Algorithm ladder: **DQN** (baseline), **QR-DQN**, **PPO**, **A2C**.
- Everything shared: env, reward, observation, action space, seeds, eval protocol.
- **Optuna** hyperparameter tuning **per algorithm** — nobody's hand-picked to win.
- Train 5 seeds, evaluate on 5 held-out seeds → mean ± std. Fixed-time baseline runs through the **same** eval pipeline.
- Pre-empt the SAC question: **not SAC** — it's continuous-action, doesn't fit discrete phase selection; **QR-DQN** (distributional DQN) is our fourth rung.
- **[HANDOFF]** "Now the results — starting with peak demand."

## Slide 7 — Results: peak demand *(~1.5 min · Sudwipto)*
- Set the scene: peak is **oversaturated** — fixed-time backs up to **~1319 s** mean wait.
- Headline: **DQN and A2C each cut mean waiting ~24%** (both ~1003 s).
- Nuance worth saying aloud: **A2C nearly ties DQN's mean but with far tighter seed spread (±86 vs ±402)** — a real stability-vs-peak-performance tradeoff.
- PPO and QR-DQN land marginally *worse* than fixed-time here (+2.8%, +6.2%) — be honest about that, don't hide it.
- Point at the plot: `results/bars_peak_lam05.png`.

## Slide 8 — Results: off-peak demand *(~1.5 min · Aleana)*
- Flip the story: off-peak is light, fixed-time is **already near-optimal at 0.39 s** — no RL agent beats it.
- The real result: **all four stay mobile — no gridlock.** DQN is within a hair (0.48 s).
- A2C is the weakest at 36 s but **valid and mobile** (speed 4.75 m/s, tight across seeds) — reported, not excluded.
- This sets up the honesty slide — don't spin off-peak as a win.
- Plot: `results/bars_offpeak_lam05.png`.

## Slide 9 — Reading the results honestly *(~1.5 min · Aleana)*
- Peak: RL clearly helps; **DQN is the overall winner** — best mean *and* biggest congestion relief.
- Off-peak: an honest **ceiling, not a failure** — a good fixed plan is genuinely hard to beat in light traffic. All agents stay mobile.
- Disclose the **one asymmetry** proactively (owning it beats getting caught): off-peak A2C's hyperparameters were selected on **waiting time**, not the shaped reward. Why: at light demand the `−λ·safety` term dominates, so the reward-optimal policy is "never switch" = gridlock; tuning on waiting time rejects that collapse.
- Nail the line: **training reward, environment, and evaluation stay identical across all algorithms — only how A2C's HPs were picked differs, and we're disclosing it.**
- **[HANDOFF]** "Aleana / Sudwipto — over to the team split."

## Work update — safety reward: defect found and fixed *(~2 min · Sudwipto)*
*(Two slides. Supervisor-update pair — inserted after "Reading the results honestly". Drop both if presenting to a general audience.)*

- **Open with what was flagged, not with the fix.** "Last review I flagged that `safety_exposure` was zero in every run. I've since traced it, and it's worse than a logging gap — the term could never have fired."
- **Root cause in one breath:** sumo-rl computes the reward only after the decision window has elapsed. Yellow is raised at the start of the window and cleared `yellow_time` seconds in — and sumo-rl *asserts* `delta_time > yellow_time`, so yellow is always over by the time we measure. No configuration could have made it fire.
- **The braking half was worse than sparse:** sampled 1 second in 5, and always the settled post-yellow second — i.e. systematically the calmest one. Biased low, not just undersampled. Say "biased", it's the honest word.
- **Fix:** accumulate both components every simulation second; reward and metrics read the same totals. 12 tests, written failing-first. Mention TDD only if asked — don't sell process.
- **The number that matters is the one that *didn't* move:** `mean|efficiency|` = 8.44 before and after. That's what licenses "the reported results stand". Lead with it if challenged.
- **Do not oversell the fix.** It changes what λ measures, not what the efficiency comparison found. If asked "so are your results wrong?" → "The efficiency results are unaffected and I can show why. What was wrong is the label: these are braking-only λ runs, and the report now says that."
- **Then hand them the two decisions** — primary metric, and whether exposure-outweighing-braking is the weighting they want. Ask, don't propose-and-defend; both are genuinely theirs to rule on.
- **Have ready if pressed:** `SAFETY_SCALE` moved 0.024 → 2.1298 because the old value was calibrated against the broken signal; λ=0.5 safety term is now 3.91 vs |efficiency| 5.64, so the intended weighting survived.
- **Do not claim** re-evaluated safety numbers for the existing checkpoints unless that run has finished and you have looked at the table. If asked: "running now, I'll send it."

## Slide 10 — Task distribution *(~45s · whoever holds the floor)*
- Everyone did the core algorithm/model work; each also owned a supporting area.
- Sudwipto: DQN + QR-DQN, algorithm registry & Optuna spaces, env + safety reward, experiment driver + cloud runs.
- Swatej: PPO, shared train/eval loop, SUMO model & scenarios, fixed-time baseline.
- Aleana: A2C + the off-peak gridlock fix, PCU observation design, comparison + plots + write-up.
- Shared by all three: literature review (11 papers), fair-comparison design, report, this presentation.
- Keep it brief — don't read the table.

## Slide 11 — Issues faced & fixes *(~1 min · Swatej)*
- Don't read every row — pick 2–3 that show real engineering.
- Highlight: the **off-peak gridlock collapse** — A2C/PPO/QR-DQN gave byte-identical results because agents learned to never switch phase; fixed by per-scenario tuning + waiting-time objective + entropy floor.
- One infra example: SUMO via pip (Homebrew route failed), plus the cloud Python 3.12 vs torch pin fix.
- Framing: "these are the honest scars of a real pipeline, and each has a concrete fix."

## Slide 12 — Areas of improvement *(~45s · Sudwipto)*
- Full-budget re-run (100k steps, 3600 s episodes, 5/5 seeds) to tighten numbers.
- Complete the λ safety-tradeoff curve for the RL agents, not just fixed-time.
- Remove the A2C asymmetry by re-tuning all four off-peak on the same objective.
- The big one: **multi-intersection corridor with coordinated MARL** (IPPO/MAPPO) — thesis-level lift, **and the env prototype is already built.**
- Real-world demand calibration for external validity.

## Slide 13 — Live demo *(~1 min · Aleana)*
- Trained agent controlling the 4-arm intersection in sumo-gui.
- Point out what to watch: PCU-weighted queues, phase switching under mixed traffic; colour by type — orange moto, blue auto, grey car.
- **Play the recorded clip** — say plainly that sumo-gui over macOS X11 uses software OpenGL and is laggy live, so a recording keeps it smooth. (Not a bug, a rendering limitation.)

## Slide 14 — Conclusions *(~45s · Sudwipto)*
- RL with PCU-weighted observation + safety-aware reward **beats fixed-time where it matters — congested peak demand, ~24% less waiting.**
- Under a fair, tuned, multi-seed comparison, **DQN is the most reliable winner**; A2C matches it at peak with tighter variance.
- Off-peak is an honest ceiling — good fixed plans are hard to beat in light traffic.
- Next step: multi-intersection corridor with coordinated MARL (prototype built).

## Slide 15 — Thank you / Q&A *(~15s · all)*
- Thank the panel, invite questions. Have the Q&A prep below ready.

---

# PART B — Q&A / Viva Prep

**Why not SAC?**
SAC is a continuous-action algorithm; our action space is discrete phase selection (pick which green phase runs next), which SAC can't address without a different parameterisation. So it's out of scope. We use QR-DQN — distributional DQN — as the fourth rung instead, which fits the discrete space cleanly.

**Why does no RL agent beat fixed-time off-peak?**
Because off-peak the baseline is already near-optimal — 0.39 s mean wait. When there's barely any traffic, a sensible fixed cycle leaves almost nothing to optimise. This is an honest performance ceiling, not a failure of the agents. The meaningful result is that all four stay mobile — no gridlock — with DQN within a hair at 0.48 s.

**Defend the A2C off-peak asymmetry.**
For off-peak A2C only, we selected hyperparameters on cumulative waiting time rather than the shaped reward. The reason is principled: at light demand throughput is flat, so the `−λ·safety` term dominates the shaped reward, and the reward-optimal policy becomes "never switch phase" — maximum safety, zero throughput, i.e. gridlock. Tuning on the shaped reward literally selects that collapse; tuning on waiting time makes gridlock score worst and rejects it. Critically, this changes only the HP-selection criterion for one cell — the training reward, environment, and evaluation protocol are identical across every algorithm and scenario, and we disclose it openly.

**Why PCU weighting?**
Because the traffic is heterogeneous and raw counts mislead a controller — three motorbikes occupy far less road-space than three cars. PCU (passenger-car equivalent) weighting lets the agent see road-space demand, not vehicle count: moto 0.3, auto 0.5, car 1.0. Without it the controller over-reacts to two-wheelers.

**How is the safety term defined?**
It's a composite, vulnerability-weighted penalty combining emergency braking and intersection exposure during yellow phases. Each vehicle is weighted by its fragility — moto 1.0, auto 0.6, car 0.3 — the inverse of the PCU weights, because a motorbike is small in road-space but the most vulnerable road user. It enters the reward as `−λ·(safety_penalty/SAFETY_SCALE)`.

**Say this too, unprompted — do not wait to be asked.** In the runs on these slides, only the braking half was live. The penalty was sampled once per decision, and that sample always lands after the yellow interval has ended, so the exposure component never fired — `system_safety_exposure` is 0.0 with zero variance in every row, baseline included. We found it, traced it to the sampling point rather than the accessors, and fixed it by accumulating every simulation second; exposure now measures 11.57 at peak against braking's 5.07, and the calibration constant moved 0.024 → 2.1298. The efficiency results are untouched by this — the waiting-time half of the reward is identical, and mean|efficiency| measures 8.44 before and after. If pressed on why the numbers stand: λ was doing less than we claimed, so these are results for a braking-only penalty, and we now label them that way.

**Is the comparison fair?**
Yes. Every algorithm shares the same environment, reward function, observation, action space, seeds, and evaluation protocol, and each was Optuna-tuned independently so none is hand-favoured. Only the algorithm changes. The single disclosed exception is the off-peak A2C HP-selection objective, which we flag explicitly.

**Why is DQN the winner if A2C ties it at peak?**
DQN has the best mean at peak (1002.8 vs 1003.3 s) and the biggest congestion relief, and it's competitive off-peak (0.48 s, within a hair of the baseline). A2C ties DQN's peak mean with tighter variance but is the weakest off-peak at 36 s. DQN is the most reliable across both regimes, so it's the overall winner; A2C's tighter peak variance is a genuine tradeoff we note rather than hide.

**Why the high DQN seed variance at peak (±402)?**
The peak scenario is heavily oversaturated, and oversaturated regimes are intrinsically sensitive — small policy differences cascade into large queue differences, so seed-to-seed spread widens. A2C is more stable there (±86), which is the tradeoff: DQN wins on mean, A2C wins on consistency. A full-budget re-run with more seeds would tighten DQN's interval.

**What would you improve with more time?**
A full-budget re-run (100k steps, 3600 s episodes, 5/5 seeds) to tighten the numbers; completing the λ safety-tradeoff curve for the RL agents; removing the A2C off-peak asymmetry by re-tuning all four on the same objective; scaling to a multi-intersection corridor with coordinated MARL (IPPO/MAPPO — prototype already built); and real-world demand calibration with measured flows.

**Real-world deployment concerns?**
Three main ones: the sim-to-real gap — SUMO dynamics only approximate real driver behaviour; the need to calibrate on measured demand flows rather than synthetic ones for external validity; and safety certification, since a learned controller must be verifiably safe before it touches live signals. Our safety-aware reward is a step toward the third but not a substitute for formal certification.

**Why only a single intersection?**
Scope — we deliberately bounded the project to one junction to run a rigorous, fully controlled comparison first. The natural next step is a multi-intersection arterial corridor with coordinated MARL, and we've already built the environment prototype for it, so the path forward is concrete rather than speculative.

---

## Fast-reference numbers (memorise)
- Peak fixed-time: **1319 s**. DQN **1003 s (±402)**, A2C **1003 s (±86)** — both **−24%**. PPO +2.8%, QR-DQN +6.2%.
- Off-peak fixed-time: **0.39 s**. DQN **0.48 s**, PPO 1.76 s, QR-DQN 1.99 s, A2C 36 s (mobile, 4.75 m/s).
- PCU weights: moto **0.3** / auto **0.5** / car **1.0**. Safety weights: moto **1.0** / auto **0.6** / car **0.3**.
- Vehicle mix: 60/25/15% moto/auto/car. Reference **λ = 0.5**. Reward = `Δwaiting − λ·(safety/SCALE)`.
- Winner: **DQN** (best mean + biggest relief). A2C = tighter peak variance, weakest off-peak.
