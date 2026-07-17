# Progress Checklist — Smart Traffic Signal Optimization (Group 7)

Status snapshot: **17 Jul 2026** · Target: **7 Aug 2026** (testing complete, report + viva ready).
Legend: ✅ done · 🟡 partial · ❌ not started · ➕ added beyond original plan

## Where we are (one line)

Calendar says mid–**Sprint 5** (Experiments & Tuning). Reality: Sprints 1–4 done **only for DQN, 1 seed, 6 episodes**. The headline deliverable — **comparative results (DQN vs fixed-time)** — is not produced yet. Runway to 7 Aug is fine **if** the experiment ladder runs this week.

## Scope change vs original plan (note for report/viva)

- ➕ Plan = **DQN vs fixed-time** only. Code = **4-algo ladder**: DQN, QR-DQN, PPO, A2C (fair-comparison design, shared env/reward/obs). More impressive, more compute.
- 🟡 Reward: plan wanted **waiting + queue + safety-aware** term. Code uses **`diff-waiting-time` only**. No safety reward term implemented (Samalla & Chunchu motivation unused).
- Action space: plan said "multi-discrete extend/switch"; code uses **discrete phase selection**. Fine, but wording in report must match code.
- ⚠️ Baseline gap: `compare.py` ranks the **RL algos against each other** — there is **no fixed-time row** in it. Plan's core claim is *DQN vs fixed-time*. Fixed-time baseline exists as `tripinfo.xml` but is **not** in the same metric frame as the eval CSVs. Must bridge this.

---

## Sprint 1 — Requirements & Setup (M1, 14 Jun) ✅

- [x] Kickoff & scope
- [x] Literature review (11 papers, PDF)
- [x] Select junction & traffic data
- [x] Git + SUMO/TraCI env (venv, `requirements.txt`)
- [x] Simulation network + experiment plan designed

## Sprint 2 — Baseline SUMO Model (M2, 21 Jun) 🟡

- [x] Build SUMO 4-arm network (`intersection.net.xml` + nod/edg)
- [x] Heterogeneous vehicles + sublane model (`vtypes.add.xml`, 60/25/15 moto/auto/car, `--lateral-resolution 0.5`)
- [x] Calibrate demand (flows in `traffic.rou.xml`) — **single profile only**; peak/off-peak split not done
- [x] Fixed-time baseline controller (`intersection.sumocfg`, `tripinfo.xml` written)
- [ ] 🟡 Validate baseline & **record baseline metrics in comparable form** — `tripinfo.xml` exists but not reduced to avg-delay/queue/throughput row usable against RL

## Sprint 3 — RL Controller Design (M3, 5 Jul) ✅

- [x] State representation (`PCUObservationFunction`: phase one-hot + min-green flag + PCU density + PCU queue)
- [x] Action space (discrete phase selection)
- [x] Reward (`diff-waiting-time`) — 🟡 no queue/safety term (see scope note)
- [x] DQN implemented (via SB3 / sb3-contrib) — ➕ plus QR-DQN, PPO, A2C in `algos.py`
- [x] TraCI ↔ agent integration (via `sumo-rl` `SumoEnvironment`)

## Sprint 4 — Integration & Training (M4, 12 Jul) 🟡

- [x] Training loop + logging (`train.py`, CSV + TensorBoard `logs/tb/`)
- [x] Initial training run — **DQN seed0 only** (6 episodes, `models/dqn_seed0.zip`)
- [ ] 🟡 Convergence sanity-check & fixes — no documented convergence check; only 1 short run
- [ ] ❌ Same for QR-DQN / PPO / A2C — not trained at all

## Sprint 5 — Experiments & Tuning (M5, 26 Jul) ❌ ← **YOU ARE HERE, behind**

- [ ] ❌ Hyperparameter tuning **for every model** (confirmed in-scope) — `tune.py` exists, **`params/` is empty** (never run). Must run per algo: dqn, qrdqn, ppo, a2c
- [ ] ❌ Train across demand scenarios (peak / off-peak) — only one demand profile, one seed
- [ ] ❌ Comparative experiments: **DQN vs fixed-time** — not run (and fixed-time not in compare frame)
- [ ] ❌ Collect metrics (avg delay, queue, throughput) — no `logs/eval_*.csv`, no `logs/comparison.csv`

## Sprint 6 — Analysis, Testing & Report (M6, 7 Aug) 🟡

- [ ] ❌ Analyse & visualise results (no plots)
- [x] 🟡 Report material started — strong `README.md` + `WALKTHROUGH.md` already written
- [ ] ❌ Robustness & full system testing
- [ ] ❌ Final review / buffer
- [ ] ❌ Viva materials

---

## Goal (confirmed): tune every model, compare all, pick best

Deliverable = **best (algorithm, hyperparameters)** chosen from a fair, tuned comparison.
Full protocol per algorithm in {dqn, qrdqn, ppo, a2c}:
1. **Tune** (`tune.py --algo X`) → `params/X.json`
2. **Train** on N seeds with tuned params (`train.py --algo X --seed s`)
3. **Eval** on held-out seeds → `logs/eval_*.csv`
4. **Rank** all + fixed-time baseline (`compare.py`) → pick winner by mean waiting time
Infra for all 4 steps already exists; `run_experiment.sh` chains them. This is a **run** job.

## Critical path to hit 7 Aug (do in order)

1. **Bridge fixed-time into the metric frame.** Add a fixed-time row to `compare.py` (parse `tripinfo.xml` or run a no-learning baseline episode through the same eval CSV path). Without this the winner isn't measured against the baseline. **Highest priority — it's the thesis.**
2. **Tune every algo.** `tune.py --algo {dqn,qrdqn,ppo,a2c}` → fills `params/`. Reduced-budget trials (e.g. `--trials 30 --steps 20000`). Heaviest compute item.
3. **Train all × seeds with tuned params.** 4 algos × 3 seeds × 100k steps.
4. **Eval on held-out seeds → `compare.py`.** Produces ranked table + winner. Clears Sprint 5.
   → Steps 2–4 = one command: `./run_experiment.sh`.
5. **Demand scenarios.** Peak + off-peak `traffic.rou.xml` variants, re-run compare per scenario.
6. **Analyse + plots + report + full system testing + viva** — Sprint 6.

## Decisions (LOCKED 17 Jul)

- **Compute:** full budget — 30 trials/algo tuning, 100k steps train, laptop overnight unattended.
- **Seeds:** 5 train + 5 held-out eval per algo (stronger mean ± std).
- **Scenarios:** peak + off-peak (two demand profiles; doubles run count).
- **Reward:** implement safety-aware term (queue/conflict penalty on top of diff-waiting-time).

## Prerequisite code changes (before any run)

1. **Safety-aware reward** in `env_common.py` — custom reward = diff-waiting-time − λ·(safety penalty). Must be held identical across all algos.
2. **Peak / off-peak route files** — two `traffic.rou.xml` variants; `make_env`/`run_experiment.sh` parameterised by scenario.
3. **Fixed-time baseline into `compare.py`** — baseline row in same metric frame as RL eval CSVs.

## Run scale (locked choices)

- Tune: 4 algos × 30 trials × 20k steps × 2 scenarios
- Train: 4 algos × 5 seeds × 100k steps × 2 scenarios
- Eval: 4 algos × 5 held-out seeds × 2 scenarios + fixed-time baseline
- → heavy. Overnight unattended via `run_experiment.sh`. Confirm laptop won't sleep (`caffeinate`).
