# RL Traffic Signal Control — Heterogeneous Traffic (SUMO + Stable-Baselines3)

Reinforcement-learning traffic signal control for a single intersection with
heterogeneous traffic (motorcycles, auto-rickshaws, cars). The observation is
**PCU-weighted** (moto 0.3, auto 0.5, car 1.0) so the controller sees passenger-car
equivalents instead of raw vehicle counts.

The project compares an algorithm ladder — **DQN** (baseline), **QR-DQN**, **PPO**,
**A2C** — on the **same environment, same reward, same observation**; only the
algorithm changes. All four support the discrete phase-selection action space.

> **Not SAC.** SAC is a continuous-action algorithm and cannot be applied to this
> discrete action space without a different parameterisation, so it is out of scope.
> QR-DQN (distributional DQN) is the fourth rung instead.

## Project files

| File | Purpose |
|------|---------|
| `env_common.py` | Shared env: PCU observation + `make_env` (identical for every algo) |
| `algos.py` | Algorithm registry: DQN / QR-DQN / PPO / A2C — defaults + Optuna search spaces |
| `train.py` | Train / evaluate one algorithm (`--algo`) |
| `tune.py` | Optuna hyperparameter search per algorithm → `params/<algo>.json` |
| `compare.py` | Aggregate held-out eval metrics across algos + seeds → ranked table |
| `intersection.nod.xml`, `intersection.edg.xml` | Network source files (nodes, edges) |
| `intersection.net.xml` | Compiled SUMO network (from netconvert/netedit) |
| `traffic.rou.xml` | Routes + heterogeneous traffic demand |
| `vtypes.add.xml` | Vehicle type definitions (moto / auto / car, with `guiShape`) |
| `gui-settings.xml` | sumo-gui view scheme (shapes, colour-by-type, playback delay) |
| `intersection.sumocfg` | SUMO config for running the sim standalone |
| `params/` | Tuned hyperparameters per algorithm (written by `tune.py`) |
| `logs/` | Per-episode CSV metrics + TensorBoard logs (`logs/tb/`) |
| `models/` | Saved model checkpoints (e.g. `dqn_seed0.zip`, `ppo_seed0.zip`) |

## Setup

Requires Python 3.9+. SUMO itself is installed **via pip** (`eclipse-sumo` ships
prebuilt binaries: `sumo`, `sumo-gui`, `netedit`, `netconvert`, `duarouter`) — no
Homebrew or separate SUMO install needed.

```bash
git clone https://github.com/sudwiptokm/group-project.git
cd group-project

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`SUMO_HOME` must point at the pip-installed SUMO. Easiest is to append it to the
venv activate script once:

```bash
echo "export SUMO_HOME=$(python -c 'import sumo; print(sumo.SUMO_HOME)')" >> venv/bin/activate
source venv/bin/activate   # re-source to pick it up
```

(`train.py` exits early if `SUMO_HOME` is not set.)

## Training

```bash
python train.py --algo dqn --steps 100000 --seed 0     # baseline
python train.py --algo qrdqn --steps 100000 --seed 0
python train.py --algo ppo --steps 100000 --seed 0
python train.py --algo a2c --steps 100000 --seed 0
```

- `--algo` selects the agent (`dqn` / `qrdqn` / `ppo` / `a2c`); default `dqn`.
- If `params/<algo>.json` exists (from `tune.py`) it is loaded automatically;
  pass `--defaults` to force the RL-Zoo-style defaults instead.
- Runs headless (no GUI needed).
- ~19 fps on an Apple Silicon MacBook → 100k steps ≈ 1.5 h.
- Outputs: `models/<algo>_seed<seed>.zip`, per-episode CSVs in `logs/`,
  TensorBoard scalars in `logs/tb/`.

For the full protocol, run each algorithm on seeds 0–2 (equal step budget), then
report mean ± std across seeds.

## Hyperparameter tuning (Optuna)

Search per algorithm; the best trial is written to `params/<algo>.json` and picked
up automatically by `train.py`:

```bash
python tune.py --algo dqn   --trials 30 --steps 20000
python tune.py --algo qrdqn --trials 30 --steps 20000
python tune.py --algo ppo   --trials 30 --steps 20000
python tune.py --algo a2c   --trials 30 --steps 20000
```

Each trial trains on the reduced budget, then maximises mean episode reward on
held-out eval seeds (`--eval-seeds`, default `42 43`). Search spaces live in
`algos.py`.

## Comparing algorithms — pick the best

After training + evaluating each algorithm on held-out seeds, aggregate:

```bash
# evaluate each trained model on held-out seeds
for a in dqn qrdqn ppo a2c; do
  for s in 42 43 44; do
    python train.py --algo $a --eval models/${a}_seed0.zip --seed $s
  done
done

python compare.py
```

`compare.py` reads the eval CSVs, time-averages each run, reports mean ± std per
algorithm, ranks by mean waiting time (lower = better), and writes
`logs/comparison.csv`.

## One-shot driver (unattended)

`run_experiment.sh` runs the whole ladder — tune → train (all seeds) → eval
(held-out) → compare — in one go. Resumable: existing `params/`, `models/` and
eval CSVs are skipped, so re-running continues where it left off. A single failed
run is logged and skipped, not fatal. Progress + a timestamped log in `logs/`.

```bash
./run_experiment.sh                      # full ladder, default budgets
./run_experiment.sh --skip-tune          # reuse current params/ (or defaults)
./run_experiment.sh --force              # ignore existing artifacts, redo all
ALGOS="dqn ppo" ./run_experiment.sh      # subset of algorithms
STEPS=50000 TRAIN_SEEDS="0 1" TUNE_TRIALS=15 ./run_experiment.sh   # smaller run
```

Tunable via env vars: `ALGOS`, `TRAIN_SEEDS`, `EVAL_SEEDS`, `STEPS`, `TUNE_TRIALS`,
`TUNE_STEPS`, `TUNE_EVAL_SEEDS`. Eval uses each algo's first-seed model as the
reference checkpoint.

> Full defaults ≈ 4 algos × (tuning + 3 training runs × ~1.5 h) — an overnight job.
> Trim `TRAIN_SEEDS` / `TUNE_TRIALS` / `STEPS` for a quick pass first.

Monitor training:

```bash
tensorboard --logdir logs/tb
```

## Evaluation

Evaluate a saved model on a held-out seed (pass the matching `--algo`):

```bash
python train.py --algo dqn --eval models/dqn_seed0.zip --seed 42
```

Add `--gui` to watch the evaluation in sumo-gui (see GUI note below).
Metrics land in `logs/eval_seed<seed>` and `tripinfo.xml`.

## Running the simulation standalone (no RL)

```bash
sumo -c intersection.sumocfg
```

## GUI on macOS (sumo-gui / netedit)

`netedit` opens as a native window — just run `netedit` with the venv active.

`sumo-gui` is X11-based and needs [XQuartz](https://www.xquartz.org/).

**Do not just `open -a XQuartz` and run on `:0`** — sumo-gui then spams
`X Error ... BadShmSeg` and the window renders **blank** (MIT-SHM shared-memory
pixmaps fail over XQuartz). Run the X server with shared memory disabled instead:

```bash
# start X on :1 with MIT-SHM off, plus a window manager
/opt/X11/bin/Xquartz :1 -extension MIT-SHM &
DISPLAY=:1 /opt/X11/bin/quartz-wm &            # after /tmp/.X11-unix/X1 appears

source venv/bin/activate
export SUMO_HOME=$(python -c 'import sumo; print(sumo.SUMO_HOME)')
DISPLAY=:1 sumo-gui -c intersection.sumocfg --window-size 1400,900
```

`Fontconfig error` on launch is harmless. `BadShmSeg` should be **gone** with the
above — if you still see it, you're on `:0`. Live playback is laggy (software GL
over X11); for a demo, record/screenshot rather than watch live.

**Visual quality.** `intersection.sumocfg` loads `gui-settings.xml`, which turns on
real vehicle silhouettes (`vehicleQuality=2` + `guiShape` on each vType: motorcycle
/ moped / sedan), colours vehicles **by type** (orange moto, blue auto, grey car),
scales small 2/3-wheelers up so they stay visible, and adds a dark clean background
+ playback `<delay>`. Tune the look by editing `gui-settings.xml` (raise `<delay>`
for slower motion, `vehicleSize.exaggeration` for bigger vehicles).

Note: sumo-gui over XQuartz uses software OpenGL, so live playback is inherently
laggy on macOS — that's the X11 renderer, not the scene. For a smooth demo, record
frames via `<snapshot>` / the GUI's video export instead of watching live.

## Gotchas (learned the hard way)

- sumo-rl 1.4.5 `SumoEnvironment` has no `additional_files` kwarg — vtypes are
  loaded via `additional_sumo_cmd="--additional-files vtypes.add.xml ..."` in
  `train.py`.
- XML comments must not contain `--` — it breaks the SUMO XML parser.
- `vTypeDistribution` references existing types via `vTypes="..." probabilities="..."`
  attributes, not `<vType refId=.../>` children.
- Edge ids in `traffic.rou.xml` must match the ids in `intersection.net.xml`
  exactly — rename one side to match before running.
