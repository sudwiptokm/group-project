# Progress Checklist — Smart Traffic Signal Optimization (Group 7)

Status snapshot: **17 Jul 2026** · Target: **7 Aug 2026** (testing complete, report + viva ready).
Legend: ✅ done · 🟡 partial · ❌ not started · ➕ added beyond original plan

## Where we are (one line)

Calendar says mid–**Sprint 5** (Experiments & Tuning). Reality: Sprints 1–4 done **only for DQN, 1 seed, 6 episodes**. The headline deliverable — **comparative results (DQN vs fixed-time)** — is not produced yet. Runway to 7 Aug is fine **if** the experiment ladder runs this week.

## Scope change vs original plan (note for report/viva)

- ➕ Plan = **DQN vs fixed-time** only. Code = **4-algo ladder**: DQN, QR-DQN, PPO, A2C (fair-comparison design, shared env/reward/obs). More impressive, more compute.
- ✅ Reward: plan wanted **waiting + queue + safety-aware** term. Code now implements `diff_waiting_time − λ·(safety_penalty / SAFETY_SCALE)` via `make_safety_reward_fn(lam)` in `env_common.py`. λ = 0 gives the pure diff-waiting-time ablation baseline; λ > 0 adds the vulnerability-weighted safety penalty (emergency braking + intersection exposure). Samalla & Chunchu motivation is now directly reflected.
- Action space: plan said "multi-discrete extend/switch"; code uses **discrete phase selection**. Fine, but wording in report must match code.
- ✅ Baseline gap resolved: `baseline.py` runs the fixed-time controller through the same eval-CSV path, and `compare.py` now includes a **fixed-time row** in the ranked table. The plan's core claim (*RL vs fixed-time*) is directly measurable.

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
- [x] Reward — safety-aware λ-weighted reward (`make_safety_reward_fn`) implemented in `env_common.py`; λ = 0 = pure diff-waiting-time ablation
- [x] DQN implemented (via SB3 / sb3-contrib) — ➕ plus QR-DQN, PPO, A2C in `algos.py`
- [x] TraCI ↔ agent integration (via `sumo-rl` `SumoEnvironment`)

## Sprint 4 — Integration & Training (M4, 12 Jul) 🟡

- [x] Training loop + logging (`train.py`, CSV + TensorBoard `logs/tb/`)
- [x] Initial training run — **DQN seed0 only** (6 episodes, `models/dqn_seed0.zip`)
- [ ] 🟡 Convergence sanity-check & fixes — no documented convergence check; only 1 short run
- [ ] ❌ Same for QR-DQN / PPO / A2C — not trained at all

## Sprint 5 — Experiments & Tuning (M5, 26 Jul) ❌ ← **YOU ARE HERE, behind**

- [ ] ❌ Hyperparameter tuning **for every model** (confirmed in-scope) — `tune.py` exists, **`params/` is empty** (never run). Must run per algo: dqn, qrdqn, ppo, a2c
- [ ] ❌ Train across demand scenarios (peak / off-peak) — infra done (`make_scenarios.py` + scenario axis in `run_experiment.sh`); **run not executed yet**
- [ ] ❌ Comparative experiments: **RL algos vs fixed-time** — infra done (`baseline.py` + fixed-time row in `compare.py`); **run not executed yet**
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

1. ~~**Bridge fixed-time into the metric frame.**~~ ✅ Done — `baseline.py` + `compare.py` fixed-time row implemented.
2. ~~**Peak / off-peak scenario files.**~~ ✅ Done — `make_scenarios.py` implemented; `run_experiment.sh` scenario axis added.
3. ~~**Safety-aware reward.**~~ ✅ Done — `make_safety_reward_fn(lam)` in `env_common.py`.
4. **Run Stage 1** — generate scenarios, then `caffeinate -i ./run_experiment.sh` (all algos, λ=0.5, peak+offpeak). Read winner from `compare.py`. **Heaviest compute item.**
5. **Run Stage 2** — `caffeinate -i env ALGOS="<winner>" LAMBDAS="0.0 0.5 1.0" ./run_experiment.sh --skip-tune` → tradeoff table in `logs/comparison.csv`.
6. **Analyse + plots + report + full system testing + viva** — Sprint 6.

## Decisions (LOCKED 17 Jul)

- **Compute:** full budget — 30 trials/algo tuning, 100k steps train, laptop overnight unattended.
- **Seeds:** 5 train + 5 held-out eval per algo (stronger mean ± std).
- **Scenarios:** peak + off-peak (two demand profiles; doubles run count).
- **Reward:** implement safety-aware term (queue/conflict penalty on top of diff-waiting-time).

## Prerequisite code changes (before any run)

- [x] **Safety-aware reward** in `env_common.py` — `make_safety_reward_fn(lam)` produces `diff_waiting_time − λ·(safety_penalty / SAFETY_SCALE)`; vulnerability-weighted composite penalty (emergency braking + intersection exposure); held identical across all algos at a given λ.
- [x] **Peak / off-peak route files** — `make_scenarios.py` generates `traffic_peak.rou.xml` + `traffic_offpeak.rou.xml`; `make_env`, `train.py`, `tune.py`, and `run_experiment.sh` parameterised by `--scenario`.
- [x] **Fixed-time baseline into `compare.py`** — `baseline.py` runs fixed-time controller through eval-CSV path; `compare.py` includes fixed-time row in ranked table.
- [x] **`run_experiment.sh` scenario × λ axes** — `LAMBDAS` env-var sweeps λ values; `--skip-tune` stage-2 flag; scenario loop across peak/offpeak.

> **Code/infra for the two-stage tuned comparison is complete and committed on
> branch `feature/safety-aware-reward-comparison`.** The actual compute run
> (tuning, training, eval, analysis, plots, report) has NOT been executed yet —
> those remain open items in the sprint checklists below.

## Run scale — MODE toggle (overnight ⇄ full)

Reality check: full budget on a laptop ≈ **270 h** (one 3600 s SUMO episode ≈ 97 s
wall-clock). So `run_experiment.sh` has a `MODE` toggle — run `overnight` first,
`full` later (ideally on a server). Explicit env vars override either preset.

| | `MODE=overnight` (~12–18 h) | `MODE=full` (~11 days) |
|---|---|---|
| Episode (`EPISODE_SECONDS`) | 1200 s | 3600 s |
| Train steps | 30k | 100k |
| Tune trials × steps | 12 × 10k | 30 × 20k |
| Train / eval seeds | 3 / 3 | 5 / 5 |
| Scenarios | peak + offpeak | peak + offpeak |

Run: `caffeinate -i ./run_experiment.sh` (overnight, default) → `python compare.py`.
Then `MODE=full` later. See README "Running the experiment — A-Z".
