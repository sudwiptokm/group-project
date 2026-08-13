# Speaker Notes + Viva Q&A Prep — Group 7

> ## Revision note — 2026-08-12
>
> The peak notes were rewritten after an audit found six defects behind those
> numbers. The DQN/A2C −24% claim **is withdrawn** — quote it only as the claim
> being withdrawn. Corrected, no learned policy here beats a competently timed
> static plan. Measurements: `docs/FINDINGS_2026-08-12.md`.
>
> **The off-peak notes are unaffected and stand as originally written.**
>
> **Do not take `presentation/QA_PREP.pdf` (7 Aug) into a viva** — its Part D.5
> and Part E run on the withdrawn numbers, and it still calls an actuated
> baseline "not implemented here". Use [`QA_PREP.md`](QA_PREP.md), which
> supersedes it and is the source of record.


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

## Slide 7 — Peak: the withdrawn result *(~2 min · Sudwipto)*
- **Open by withdrawing it, in your own voice, before showing the table.** "We reported a 24% cut at peak. That result does not hold, and I'll show you exactly why — six defects, five of them independent, any one enough on its own."
- Do not apologise your way through this slide. Finding it is the work; the room will read confidence in the diagnosis as competence, and hedging as doubt about the rest.
- Take the six in order but **land hard on two**: the metric was a gridlock clock (A2C deadlocked 5/5 seeds and therefore scored *best* — that inversion sells the whole point), and the baseline ran on seed 0 while the agents ran on 42–46, where fixed-time alone spans 242–1319 s.
- Have the paired numbers ready: per seed, DQN −23.9% becomes **+56.9%**, A2C **+67.6%**, PPO **+118%**, QR-DQN **+123%**.
- Say plainly that the "fixed-time" baseline was a 10 s-green cycler — that sets up slide 7b, where 10 s is the *worst* plan in the sweep.
- **Do not show** `bars_peak_lam05.png` / `improvement_peak_lam05.png` — they plot the withdrawn numbers.
- If asked "so is any of this real?" → "Off-peak is unaffected and I'll show why in two slides. At peak the honest answer is that we have no valid RL measurement, and I'd rather say that than defend a number I've disproved."

## Slide 7b — Peak: what actually controls performance *(~2 min · Sudwipto)*
- This is the slide the project is now built on. Deliver it as a finding, not as damage control.
- The sweep: static green duration against delay per completed trip, seeds 42–46. **Every static plan in the 45–90 s band beats every learned policy we produced by 2–3×.**
- **The strongest single number on this slide is a completion rate, not a delay.** The 10 s plan Stage 1 called "fixed-time" completes **76%** of trips (20 s: 73%) against ~95% on the plateau. It wasn't slow, it was leaving a quarter of the traffic unserved — and those are exactly the vehicles the old metric stopped counting. Defect 1 and defect 6 are the same story told twice.
- **Say "plateau", never "optimum".** Paired on the same seeds those greens differ by ±13 s against a ~30 s seed spread. This is the pre-emptive answer to "you hand-tuned the baseline": the plan that beats our agents took no tuning skill to find. If you quote the sample minimum (75 s) as *the* best, you have made defect 2 again in front of the panel — it loses to 60 s on four of five seeds.
- **Mechanism in one breath:** 3 s of amber per switch — 23% of the cycle at a 10 s green, 4.8% at 60 s. The agent decides every 5 s with a 10 s minimum green, so it lives exactly where switching is cheap to try and ruinous to pay for, and `diff_waiting_time` bills it several decisions too late for credit assignment.
- The payoff line: **one mechanism explains Stage 1, the pilot, and the 20k-step null together.** That is why we call it structural rather than a training-budget problem.
- Say there is **no valid RL row** and we are not inventing one — the checkpoints predate the safety fix and two retraining attempts produced no learning. Stage-1 policies sat at 20–33 s where a 60 s static plan sits at 11.5 s.
- Worth a sentence if you have time: `sumo-rl` stores `max_green` and never reads it, so our `max_green = 60` constrained nothing — which is why the sweep runs past 60 s instead of stopping there.

## Slide 7c — Did our RL fail, or was there nothing to find? *(~2 min · Sudwipto)*
- **Open with the question, not the table.** "A static plan beating our agents fits two stories: we failed to find an adaptive policy, or there isn't one here. More training can't tell them apart — a second null fits both. So we tested it with a controller that has nothing to learn."
- Describe the control in one breath: **queue-actuated** — serve whichever phase has the biggest PCU-weighted queue, subject to `min_green`. Perfect queue information, no reward to misspecify, no credit assignment, no sample budget. **If it can't beat the best static plan, the headroom isn't there.**
- **The headline is the 10 s row, and it's about us.** At the floor we actually trained on, a controller that *cannot* be accused of under-training is **5.6× worse** than the fixed plan and strands a quarter of the traffic. 125–168 switches an episode, 3 s of amber each. Land the consequence: **our entire peak training budget was spent in a region where no controller can win.** The peak null was over-determined.
- Note the curve **turns by 90 s** — 60 s is an interior optimum, not "longer is always better". Pre-empts "so why not just make the green enormous?"
- **Then do to your own result what you did to Stage 1's.** −9.3 s at 60 s looks like a win; the paired sd is 23.9 s, so it is *inside the noise*. Say it before the panel does. **The mean is not the finding.**
- **What is resolvable is consistency:** delay sd 10.1 vs 19.9, and trips completed 4142–4177 (spread **35**) against static's 3834–4162 (spread **328**). Static's bad draw is seed 43 — 126.3 s, 3834 trips; actuated takes that same seed at 83.1 s and 4146 trips. One line to remember: **the adaptive gain is not a lower mean, it's not having a bad seed.**
- **Closing line:** neither "we failed" nor "nothing exists". At `min_green` = 10 there was nothing to find; at 60 there is, but it's a ~10% variance reduction a controller that learns nothing already collects. **So the bar for RL is the actuated controller, not the static plan.**
- If asked "why not max-pressure?" → it needs downstream occupancy and would measure something different on approaches this short; queue-actuated is the standard non-learning reference for a 2-phase junction.
- If asked "doesn't this kill the project?" → it does the opposite: it converts our main recommendation from a guess into a measurement. We were choosing between three fixes; now we know which one and by how much.
- **Reading the plot** (`results/headroom_peak.png`): both controllers on one x axis — for the static plan it is the green it holds, for the actuated one its `min_green` floor. Same y, delay per completed trip, same seeds. Point at the **left-hand end, not the crossover**: that is where the argument is. The lower panel is completion, and it shows both controllers stranding traffic below ~45 s.

## Slide 8 — Results: off-peak demand *(~1.5 min · Aleana)*
- Flip the story: off-peak is light, fixed-time is **already near-optimal at 0.39 s** — no RL agent beats it.
- The real result: **all four stay mobile — no gridlock.** DQN is within a hair (0.48 s).
- A2C is the weakest at 36 s but **valid and mobile** (speed 4.75 m/s, tight across seeds) — reported, not excluded.
- This sets up the honesty slide — don't spin off-peak as a win.
- Plot: `results/bars_offpeak_lam05.png`.

## Slide 9 — Reading the results honestly *(~1.5 min · Aleana)*
- Peak: **no valid RL result**, and the standard to beat is any static plan in the 45–90 s band. Do not soften this into "mixed results".
- Add the headroom finding in one sentence: **`min_green` = 10 was the binding constraint, not the algorithm** — a controller that learns nothing is 5.6× worse than the fixed plan at that floor and matches it at 60 s. That makes the peak null over-determined and tells us exactly what to fix first.
- Off-peak: an honest **ceiling, not a failure** — a good fixed plan is genuinely hard to beat in light traffic. All agents stay mobile.
- Say the two together: **the ceiling is the same ceiling in both regimes**, and at peak we can now name the mechanism behind it.
- Volunteer the λ gap before anyone finds it: only λ = 0.5 was ever run, so "safety-aware" is in the title and not yet in the results. The sweep driver is written and ready.
- If asked which algorithm won: **none.** Ranking four algorithms only means something once one of them beats a competent static plan.
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
- Land the finding as a finding: **at an isolated 2-phase junction a competently timed static plan is hard to beat, and we can say why** — amber lost time at the switching frequency the action space allows.
- **Off-peak reaches the same ceiling from the other side** — 0.39 s, all four mobile, none ahead.
- The methodology findings travel further than the controller did: three of the six defects are about how `sumo-rl` is commonly used, not about our code.
- Next step: multi-intersection corridor with coordinated MARL (prototype built) — **coordination is the one thing a static plan cannot imitate**, which is exactly what this result points at. Make that the last sentence; it turns a negative result into a direction.

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

**Which algorithm won?**
None of them, and we no longer claim one. The peak ranking that put DQN first came out of the pipeline we audited, and none of it survives — the metric inverted the order, and the baseline ran on different traffic than the agents. Off-peak, DQN is the closest to the baseline but does not beat it. Ranking four algorithms only becomes meaningful once at least one of them beats a competently timed static plan, and none does.

**Isn't a negative result just a failed project?**
No, because it comes with a mechanism and a prediction. We can point at exactly what costs the controller its margin — 3 s of amber per switch, up to 23% of capacity at the switching frequency the action space allows — and that one mechanism explains the Stage-1 result, the pilot, and the 20k-step null together. It also predicts where RL *would* have room: longer minimum greens, a protected left-turn phase, or coordination across junctions. A result that tells you where to look next is worth more than a headline number we would have had to withdraw later.

**Why not just train longer?**
Because we tested that and it is not the binding constraint. Two 20k-step attempts produced no learning, and an 8k single-seed pilot that looked promising failed to replicate over three seeds. More importantly, the static sweep sets the bar: a plan with no state, no learning, and one parameter beats every policy we trained. The gap is in the policy space, not the sample budget.

**How do you know the metric was the problem and not the agents?**
Two independent checks. First, the ranking inverted under it: A2C deadlocked on 5 of 5 seeds and scored *best*, because a locked network accumulates one second of "waiting" per second while the vehicles that escape stop contributing. Second, on the same runs the in-network average reads 14.97 s while delay per completed trip reads 66.4 s — a factor of four apart. We now rank on completed-trip delay and throughput, which cannot be improved by holding vehicles back.

**What would you improve with more time?**
A full-budget re-run (100k steps, 3600 s episodes, 5/5 seeds) to tighten the numbers; completing the λ safety-tradeoff curve for the RL agents; removing the A2C off-peak asymmetry by re-tuning all four on the same objective; scaling to a multi-intersection corridor with coordinated MARL (IPPO/MAPPO — prototype already built); and real-world demand calibration with measured flows.

**Real-world deployment concerns?**
Three main ones: the sim-to-real gap — SUMO dynamics only approximate real driver behaviour; the need to calibrate on measured demand flows rather than synthetic ones for external validity; and safety certification, since a learned controller must be verifiably safe before it touches live signals. Our safety-aware reward is a step toward the third but not a substitute for formal certification.

**Why only a single intersection?**
Scope — we deliberately bounded the project to one junction to run a rigorous, fully controlled comparison first. The natural next step is a multi-intersection arterial corridor with coordinated MARL, and we've already built the environment prototype for it, so the path forward is concrete rather than speculative.

---

## Fast-reference numbers (memorise)
- **Peak, current:** static plan at 60 s (mid-plateau) = **91.8 ± 19.9 s delay per completed trip**, 4076 trips, **94.3%** of demand completed, 14.97 s in-network wait. No valid RL row. Plateau = **45–90 s**; paired differences ±13 s against a ~30 s seed spread.
- **Peak, withdrawn:** fixed-time 1319 s, DQN/A2C 1003 s (−24%), PPO +2.8%, QR-DQN +6.2%. Quote these **only** as the numbers being withdrawn. Paired per seed they become +56.9 / +67.6 / +118 / +123%.
- **Actuated headroom probe:** queue-actuated, non-learning. `min_green` 10 = **517.5 ± 208.4 s**, 2925 trips (**5.6×** the static plan, +426 paired, wins 0/5); `min_green` **60 = 82.5 ± 10.1 s**, 4156 trips, −9.3 paired, wins 3/5; 75 = 92.2, 90 = 118.7 (curve turns). Robustness is the real gain: trips **4142–4177 (spread 35)** vs static **3834–4162 (spread 328)**; delay sd 10.1 vs 19.9. Seed 43: static 126.3 s / 3834 trips, actuated 83.1 s / 4146 trips. **−9.3 s is inside the noise — paired sd 23.9 s.**
- **Amber arithmetic:** 3 s yellow → 23% of the cycle lost at a 10 s green, 4.8% at 60 s. Decisions every 5 s, `min_green` 10 s. Switch requests: 125–168/episode at a 10 s floor vs 38–60 at 75–90 s.
- **The 4× gap:** same peak runs, 14.97 s in-network vs 66.4 s waiting per completed trip — survivorship bias, measured.
- Off-peak fixed-time: **0.39 s**. DQN **0.48 s**, PPO 1.76 s, QR-DQN 1.99 s, A2C 36 s (mobile, 4.75 m/s). Unaffected by the audit.
- PCU weights: moto **0.3** / auto **0.5** / car **1.0**. Safety weights: moto **1.0** / auto **0.6** / car **0.3**.
- Vehicle mix: 60/25/15% moto/auto/car. Reference **λ = 0.5**. Reward = `Δwaiting − λ·(safety/SCALE)`.
- **No winner is claimed.** The old "winner: DQN, best mean + biggest relief" line came from the withdrawn peak table — do not use it. Off-peak DQN is closest to the baseline (0.48 s vs 0.39 s) but does not beat it, and at peak there is no valid RL row at all. Ranking four algorithms only means something once one of them beats a competent static plan, and none does.
