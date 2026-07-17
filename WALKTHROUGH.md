# Project Walkthrough — RL Traffic Signal Control for Heterogeneous Traffic

A complete, step-by-step tour of what this project does, how the pieces fit
together, and what every file and function is responsible for.

---

## 1. What are we actually building?

We are training a **reinforcement-learning (RL) traffic-signal controller** for a
single 4-arm intersection. Instead of a fixed-time signal (green for N seconds,
then switch), an RL agent watches the traffic in real time and decides which
green phase to show next, aiming to **minimise vehicle waiting time**.

Two things make this project distinctive:

1. **Heterogeneous traffic.** The road carries a South/South-East-Asia-style mix:
   ~60% motorcycles, ~25% auto-rickshaws (3-wheelers), ~15% cars. These vehicles
   have different sizes, speeds, and — crucially — different road-space footprints.
   The simulation uses SUMO's **sublane model** so small vehicles *filter and weave*
   between larger ones instead of queueing single-file (real lane-free behaviour).

2. **PCU-weighted observation.** The agent does not see raw vehicle counts. It sees
   **Passenger-Car-Unit (PCU) equivalents**: motorcycle = 0.3, auto = 0.5, car = 1.0.
   So "10 motorcycles" registers as far less road demand than "10 cars", which is
   physically correct and gives the controller a better signal.

### The research contribution

The point of the project is an **apples-to-apples algorithm comparison**. We hold
the environment, reward, and observation *identical* and swap only the RL algorithm:

| Algorithm | Family | Role |
|-----------|--------|------|
| **DQN** | value-based, off-policy | baseline |
| **QR-DQN** | distributional DQN, off-policy | from `sb3-contrib` |
| **PPO** | policy-gradient, on-policy | |
| **A2C** | actor-critic, on-policy | |

All four support the **discrete** action space (pick the next green phase).

> **Why not SAC?** SAC is a continuous-action algorithm. Our action space is
> discrete phase selection, so SAC would need a different parameterisation and is
> out of scope. QR-DQN is the fourth rung instead.

---

## 2. High-level pipeline

```
 SUMO world files (XML)
        │
        ▼
 env_common.make_env()  ──►  SumoEnvironment with PCU observation
        │
        │   algos.py supplies hyperparameters + the algorithm class
        ▼
 tune.py  ──►  params/<algo>.json   (best hyperparameters, optional)
        │
        ▼
 train.py  ──►  models/<algo>_seed<s>.zip   +   logs/*.csv   +   logs/tb/
        │
        ▼  (train.py --eval on held-out seeds)
 logs/eval_<algo>_seed<s>_*.csv
        │
        ▼
 compare.py  ──►  ranked table  +  logs/comparison.csv
```

`run_experiment.sh` wires the whole chain together for an unattended run.

---

## 3. The simulation world (SUMO XML files)

These files define the physical scene. They are built once and then reused by
every training and evaluation run.

### 3.1 `intersection.nod.xml` — nodes
Defines 5 nodes:
- `C` at `(0,0)` — the **centre**, typed `traffic_light` (this is what the agent controls).
- `N`, `S`, `E`, `W` — the four arm endpoints, 200 m out, typed `priority`.

### 3.2 `intersection.edg.xml` — edges (roads)
8 edges = one **in** and one **out** per arm:
- `*_in` = arm → centre (approach)
- `*_out` = centre → arm (exit)

Each has **2 lanes**, 50 km/h (13.89 m/s), width 3.5 m. Wider lanes give the
sublane model room for 2/3-wheelers to filter at runtime.

### 3.3 `intersection.net.xml` — compiled network
The **compiled** network produced by `netconvert`/`netedit` from the nodes and
edges. This is the file the simulator actually loads. It also contains the
default traffic-light program.

### 3.4 `vtypes.add.xml` — vehicle type definitions
Defines the three heterogeneous vehicle types:

| id | vClass | guiShape | length | notes |
|------|-----------|-----------------|--------|-------|
| `moto` | motorcycle | motorcycle | 2.0 m | narrow, agile, filters aggressively |
| `auto` | moped | moped | 2.8 m | 3-wheeler (no native 3-wheeler vClass in SUMO) |
| `car` | passenger | passenger/sedan | 4.5 m | holds lane centre; the PCU = 1.0 reference |

`latAlignment="arbitrary"` + a small `minGapLat` on the small vehicles is what lets
them weave laterally instead of lining up single-file.

> **Note:** the PCU weights are **not** stored here — they live in the observation
> function (`env_common.py`). Keep them in sync: moto 0.3, auto 0.5, car 1.0.

### 3.5 `traffic.rou.xml` — routes and demand
- `vTypeDistribution id="mixed"` — samples each spawned vehicle's type at
  **60% moto / 25% auto / 15% car**.
- **Routes** — through + left + right movement for every approach.
- **Flows** — `vehsPerHour` per movement (through movements heavy at 400–500/h,
  turns lighter at 120–150/h) over a **1-hour (3600 s)** episode.

> **Critical:** edge ids in this file must match the ids in `intersection.net.xml`
> exactly, or SUMO errors out.

### 3.6 `intersection.sumocfg` — standalone run config
Config for running the sim **without RL** (fixed-time baseline). Loads the net +
routes + vtypes, sets a 3600 s episode, enables the sublane model
(`lateral-resolution 0.5`), disables teleporting so jams stay visible, and writes
`tripinfo.xml`. Run it with `sumo -c intersection.sumocfg`.

### 3.7 `gui-settings.xml` — sumo-gui appearance
View scheme for the GUI: real vehicle silhouettes, colour-by-type (orange moto,
blue auto, grey car), size exaggeration for small vehicles, dark background, and a
playback delay. Purely cosmetic — used only under `sumo-gui`.

---

## 4. The Python code, file by file

### 4.1 `env_common.py` — the shared environment (the invariant)

This is the heart of the "fair comparison" guarantee. **Every** algorithm trains
and evaluates on the environment defined here — same reward, same observation.

**`PCU` / `DEFAULT_PCU`**
Dictionary of PCU weights (`moto=0.3, auto=0.5, car=1.0`); unknown types default
to 1.0 (treat as a car — the conservative choice).

**`_vehicle_pcu(type_id) -> float`**
Maps a vehicle's type id to its PCU weight. Prefix-matches so a type carrying a
distribution suffix (e.g. `moto@...`) still resolves correctly.

**`PCUObservationFunction(ObservationFunction)`**
The custom observation. For the traffic signal it builds a vector:

```
[ phase one-hot | min-green-elapsed flag | per-lane PCU density | per-lane PCU queue ]
```

- `__call__()` — reads live traffic via the traci connection:
  - **phase one-hot** — which green phase is currently active.
  - **min-green flag** — 1 if the current phase has been green long enough to legally switch.
  - **PCU density** per lane = (sum of PCU of all vehicles on the lane) / (lane capacity in PCU), clipped to `[0, 1]`.
  - **PCU queue** per lane = same, but only counting *halting* vehicles (speed < 0.1 m/s).
  - Lane capacity ≈ `lane_length / CAR_FOOTPRINT` where `CAR_FOOTPRINT = 4.5 + 2.0`.
- `observation_space()` — a `Box(0, 1)` of size `num_green_phases + 1 + 2 * num_lanes`.

**`make_env(seed, gui=False, out_csv=None) -> SumoEnvironment`**
The single environment factory. It pins every knob so nothing varies between
algorithms:
- `net_file`, `route_file`, `observation_class=PCUObservationFunction`
- `num_seconds=3600` (1-hour episode)
- `delta_time=5` — seconds between agent decisions
- `yellow_time=3`, `min_green=10`, `max_green=60`
- `reward_fn="diff-waiting-time"` — **the same reward for all agents**
- `single_agent=True` — one traffic light → Stable-Baselines3 single-agent API
- `additional_sumo_cmd` — loads `vtypes.add.xml` and enables the sublane model
  (`--lateral-resolution 0.5`). Under GUI it also loads `gui-settings.xml`.

> The vtypes are loaded via `additional_sumo_cmd` because sumo-rl 1.4.5's
> `SumoEnvironment` has no `additional_files` keyword argument.

### 4.2 `algos.py` — the algorithm registry

Provides, for each algorithm: the SB3 class, a `defaults()` function (RL-Zoo-style
"no tuning" hyperparameters), and a `sample(trial)` function (the Optuna search
space). All return plain kwargs passed straight to the algorithm constructor.
`policy` is always `"MlpPolicy"` (Box observation, discrete action).

**`_NET_ARCHS`** — three MLP presets: `small [64,64]`, `medium [256,256]`, `large [400,300]`.

**Off-policy (DQN + QR-DQN share a shape):**
- `_off_policy_defaults()` — learning rate, replay buffer size, batch size, gamma,
  train frequency, target-update interval, exploration schedule, net-arch.
- `_off_policy_sample(trial)` — the Optuna search space over those same knobs.

**On-policy PPO:**
- `_ppo_defaults()` — lr, n_steps, batch_size, n_epochs, gamma, GAE lambda, clip range, entropy coef, net-arch.
- `_ppo_sample(trial)` — search space. Guards that `batch_size` divides `n_steps`
  so PPO doesn't silently drop a partial batch.

**On-policy A2C:**
- `_a2c_defaults()` — lr, n_steps, gamma, GAE lambda, entropy/value coefficients, net-arch.
- `_a2c_sample(trial)` — search space.

**`ALGOS`** — the registry dict mapping `"dqn" / "qrdqn" / "ppo" / "a2c"` to
`{cls, defaults, sample}`.

**`build(algo, env, params, seed, tb_log=None)`** — instantiates the chosen
algorithm class on the env with the given kwargs.

### 4.3 `train.py` — train or evaluate one algorithm

The main entry point for a single run. Selected with `--algo`.

**`load_params(algo, use_defaults) -> dict`**
Returns tuned hyperparameters from `params/<algo>.json` if that file exists and
`--defaults` was not passed; otherwise returns the algorithm's defaults.

**`_materialise(saved) -> dict`**
Converts a saved param dict (which stores `net_arch` as a plain list) back into
constructor kwargs by rebuilding `policy_kwargs`.

**`train(algo, steps, seed, use_defaults)`**
1. Ensure `models/` and `logs/` exist.
2. `make_env(...)` with a per-episode CSV path, then wrap in SB3 `Monitor`.
3. Load hyperparameters, `build(...)` the model.
4. `model.learn(total_timesteps=steps)` with a progress bar.
5. Save to `models/<algo>_seed<seed>.zip`; close the env.

**`evaluate(algo, model_path, seed, gui)`**
1. `make_env(...)` (optionally with GUI), `load` the saved model.
2. Run **one deterministic episode**, summing reward.
3. **Explicitly `save_csv(...)`** — sumo-rl only flushes its metrics CSV on the
   *next* `reset()`, which a single eval episode never triggers; so we force the
   write before closing.
4. Print total reward and the metrics CSV path.

**`__main__`**
- Exits early if `SUMO_HOME` is not set.
- Argparse flags: `--algo`, `--steps`, `--seed`, `--eval <model>`, `--gui`, `--defaults`.
- With `--eval` → run `evaluate`; otherwise → run `train`.

### 4.4 `tune.py` — Optuna hyperparameter search

Searches per algorithm and writes the best hyperparameters to
`params/<algo>.json`, which `train.py` then picks up automatically.

**`_serialisable(params) -> dict`**
Flattens constructor kwargs into a JSON-safe dict (stores `net_arch` as a list,
drops the nested `policy_kwargs`).

**`_eval_reward(model, seed) -> float`**
Runs one deterministic held-out episode and returns the total reward.

**`make_objective(algo, steps, train_seed, eval_seeds)`**
Returns the Optuna `objective(trial)`:
1. `sample(trial)` the hyperparameters (search space from `algos.py`).
2. Train on `train_seed` for the reduced `steps` budget.
3. If a bad hyperparameter combo throws, mark the trial **pruned** so the whole
   study survives.
4. Evaluate on each held-out seed, return the **mean** episode reward.

**`main()`**
- Exits if `SUMO_HOME` unset.
- Args: `--algo`, `--trials`, `--steps`, `--train-seed`, `--eval-seeds`.
- Creates a `TPESampler` study with `direction="maximize"`, optimises, then writes
  the best trial's hyperparameters to `params/<algo>.json`.

### 4.5 `compare.py` — aggregate and rank

Turns the evaluation CSVs into a ranked comparison table.

**`METRICS` / `RANK_KEY`**
The surfaced metrics (`system_mean_waiting_time`, `system_total_stopped`,
`system_mean_speed`, `system_total_waiting_time`); the ranking key is
`system_mean_waiting_time` (**lower is better**).

**`_run_means(logs_dir, algo) -> DataFrame`**
Globs every eval CSV for the algorithm, and for each run time-averages each metric
over the episode → one row per run.

**`main()`**
1. For each algorithm, gather its runs and compute **mean ± std** across seeds.
2. Sort by mean waiting time, print a formatted `mean ± std` table.
3. Announce the winner and write the full table to `logs/comparison.csv`.

### 4.6 `run_experiment.sh` — the one-shot driver

Runs the whole ladder unattended: **tune → train (all seeds) → eval (held-out) →
compare**.

- **Resumable** — existing `params/`, `models/`, and eval CSVs are skipped, so a
  re-run continues where it left off. `--force` ignores them and redoes everything.
- **Fault-tolerant** — the `run()` wrapper logs a failed step and continues rather
  than aborting the batch.
- **Configurable** via env vars: `ALGOS`, `TRAIN_SEEDS`, `EVAL_SEEDS`, `STEPS`,
  `TUNE_TRIALS`, `TUNE_STEPS`, `TUNE_EVAL_SEEDS`.
- Evaluation uses each algorithm's **first-seed** model as the reference checkpoint.
- Flags: `--skip-tune`, `--skip-train`, `--skip-eval`, `--force`.
- Writes a timestamped log to `logs/experiment_<timestamp>.log`.

---

## 5. Step-by-step: running the whole thing yourself

### Step 0 — Setup (once)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# SUMO ships via pip (eclipse-sumo). Point SUMO_HOME at it:
echo "export SUMO_HOME=$(python -c 'import sumo; print(sumo.SUMO_HOME)')" >> venv/bin/activate
source venv/bin/activate
```

### Step 1 — (Optional) sanity-check the world
```bash
sumo -c intersection.sumocfg      # runs the fixed-time baseline, writes tripinfo.xml
```

### Step 2 — (Optional) tune hyperparameters
```bash
python tune.py --algo dqn   --trials 30 --steps 20000
python tune.py --algo qrdqn --trials 30 --steps 20000
python tune.py --algo ppo   --trials 30 --steps 20000
python tune.py --algo a2c   --trials 30 --steps 20000
# → writes params/<algo>.json, auto-loaded by train.py
```

### Step 3 — Train each algorithm on several seeds
```bash
for a in dqn qrdqn ppo a2c; do
  for s in 0 1 2; do
    python train.py --algo $a --seed $s --steps 100000
  done
done
# → models/<algo>_seed<s>.zip, per-episode CSVs, TensorBoard scalars in logs/tb/
```

### Step 4 — Evaluate each model on held-out seeds
```bash
for a in dqn qrdqn ppo a2c; do
  for s in 42 43 44; do
    python train.py --algo $a --eval models/${a}_seed0.zip --seed $s
  done
done
# → logs/eval_<algo>_seed<s>_*.csv
```

### Step 5 — Build the comparison table
```bash
python compare.py
# → prints mean ± std per algorithm, ranks by waiting time, writes logs/comparison.csv
```

### Or: do Steps 2–5 in one command
```bash
./run_experiment.sh                 # full ladder, default budgets
./run_experiment.sh --skip-tune     # reuse current params/ (or defaults)
STEPS=50000 TRAIN_SEEDS="0 1" ./run_experiment.sh   # smaller, faster pass
```

### Monitor training
```bash
tensorboard --logdir logs/tb
```

---

## 6. Where outputs land

| Path | Contents |
|------|----------|
| `models/` | Saved model checkpoints, e.g. `dqn_seed0.zip` |
| `params/` | Tuned hyperparameters per algorithm (written by `tune.py`) |
| `logs/` | Per-episode CSVs, eval CSVs, `logs/comparison.csv`, run logs |
| `logs/tb/` | TensorBoard scalars |
| `tripinfo.xml` | Per-trip metrics from the standalone baseline sim |

---

## 7. Gotchas (learned the hard way)

- **sumo-rl 1.4.5** has no `additional_files` kwarg — vtypes load via
  `additional_sumo_cmd="--additional-files vtypes.add.xml ..."`.
- **XML comments must not contain `--`** — it breaks the SUMO XML parser.
- **`vTypeDistribution`** references existing types via `vTypes="..."
  probabilities="..."` attributes, not `<vType refId=.../>` children.
- **Edge ids** in `traffic.rou.xml` must match `intersection.net.xml` exactly.
- **Eval CSV flush** — sumo-rl only writes its CSV on the next `reset()`, so
  `evaluate()` calls `save_csv(...)` explicitly for the single eval episode.
- **The sublane model** (`lateral-resolution 0.5`) is the knob that enables
  lane-free weaving; without it the heterogeneous traffic queues single-file and
  the whole premise collapses.
