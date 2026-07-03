# RL Traffic Signal Control — Heterogeneous Traffic (SUMO + Stable-Baselines3)

Reinforcement-learning traffic signal control for a single intersection with
heterogeneous traffic (motorcycles, auto-rickshaws, cars). The observation is
**PCU-weighted** (moto 0.3, auto 0.5, car 1.0) so the controller sees passenger-car
equivalents instead of raw vehicle counts.

The project compares an algorithm ladder — DQN (baseline), then QR-DQN / PPO / SAC —
on the **same environment, same reward, same observation**; only the algorithm changes.

## Project files

| File | Purpose |
|------|---------|
| `train.py` | Training / evaluation entry point (DQN baseline) |
| `intersection.nod.xml`, `intersection.edg.xml` | Network source files (nodes, edges) |
| `intersection.net.xml` | Compiled SUMO network (from netconvert/netedit) |
| `traffic.rou.xml` | Routes + heterogeneous traffic demand |
| `vtypes.add.xml` | Vehicle type definitions (moto / auto / car) |
| `intersection.sumocfg` | SUMO config for running the sim standalone |
| `logs/` | Per-episode CSV metrics + TensorBoard logs (`logs/tb/`) |
| `models/` | Saved model checkpoints (e.g. `dqn_seed0.zip`) |

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
python train.py --steps 100000 --seed 0
```

- Runs headless (no GUI needed).
- ~19 fps on an Apple Silicon MacBook → 100k steps ≈ 1.5 h.
- Outputs: `models/dqn_seed<seed>.zip`, per-episode CSVs in `logs/`,
  TensorBoard scalars in `logs/tb/`.

For the full protocol, repeat with seeds 1–4 (equal step budget per seed), then
report mean ± std across seeds.

Monitor training:

```bash
tensorboard --logdir logs/tb
```

## Evaluation

Evaluate a saved model on a held-out seed:

```bash
python train.py --eval models/dqn_seed0.zip --seed 42
```

Add `--gui` to watch the evaluation in sumo-gui (see GUI note below).
Metrics land in `logs/eval_seed<seed>` and `tripinfo.xml`.

## Running the simulation standalone (no RL)

```bash
sumo -c intersection.sumocfg
```

## GUI on macOS (sumo-gui / netedit)

`netedit` opens as a native window — just run `netedit` with the venv active.

`sumo-gui` is X11-based and needs [XQuartz](https://www.xquartz.org/):

```bash
open -a XQuartz
source venv/bin/activate
export DISPLAY=:0
export XAUTHORITY=$(ls -t ~/.serverauth.* | head -1)
sumo-gui -c intersection.sumocfg
```

`Fontconfig error` and `BadShmSeg` warnings on launch are harmless.

## Gotchas (learned the hard way)

- sumo-rl 1.4.5 `SumoEnvironment` has no `additional_files` kwarg — vtypes are
  loaded via `additional_sumo_cmd="--additional-files vtypes.add.xml ..."` in
  `train.py`.
- XML comments must not contain `--` — it breaks the SUMO XML parser.
- `vTypeDistribution` references existing types via `vTypes="..." probabilities="..."`
  attributes, not `<vType refId=.../>` children.
- Edge ids in `traffic.rou.xml` must match the ids in `intersection.net.xml`
  exactly — rename one side to match before running.
