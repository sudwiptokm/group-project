"""
Train / evaluate one algorithm on the shared intersection env.

The env, reward and observation are fixed in env_common.py; only the algorithm
changes (see algos.py). Keep the protocol identical across agents:
    equal training-step budget, 3-5 seeds each, evaluate on held-out seeds,
    report mean +/- std (compare.py aggregates).

Hyperparameters:
    - default: RL-Zoo-style defaults from algos.py
    - tuned:   if params/<algo>.json exists (written by tune.py), it is loaded
               automatically unless --defaults is passed.

Prereqs:
    SUMO_HOME set; pip install -r requirements.txt; intersection.net.xml built.

Run:
    python train.py --algo dqn   --steps 100000 --seed 0
    python train.py --algo qrdqn --steps 100000 --seed 0
    python train.py --algo dqn --eval models/dqn_seed0.zip --seed 42   # held-out
"""

import argparse
import json
import os
import sys

from stable_baselines3.common.monitor import Monitor

from algos import ALGOS, build
from env_common import make_env

PARAMS_DIR = "params"


def load_params(algo: str, use_defaults: bool) -> dict:
    """Tuned params/<algo>.json if present (and not overridden), else defaults."""
    path = os.path.join(PARAMS_DIR, f"{algo}.json")
    if not use_defaults and os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        print(f"[{algo}] using tuned hyperparameters from {path}")
        # net_arch is stored as a name key by tune.py -> rebuild policy_kwargs
        return _materialise(saved)
    print(f"[{algo}] using default hyperparameters")
    return ALGOS[algo]["defaults"]()


def _materialise(saved: dict) -> dict:
    """Convert a saved param dict (net_arch as list) back into cls kwargs."""
    params = dict(saved)
    net_arch = params.pop("net_arch", None)
    if net_arch is not None:
        params["policy_kwargs"] = dict(net_arch=net_arch)
    return params


# ----------------------------------------------------------------------------
def train(algo: str, steps: int, seed: int, use_defaults: bool):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = make_env(seed=seed, gui=False, out_csv=f"logs/{algo}_seed{seed}")
    env = Monitor(env)

    params = load_params(algo, use_defaults)
    model = build(algo, env, params, seed=seed, tb_log="logs/tb")
    model.learn(total_timesteps=steps, progress_bar=True)

    path = f"models/{algo}_seed{seed}.zip"
    model.save(path)
    env.close()
    print(f"saved {path}")


def evaluate(algo: str, model_path: str, seed: int, gui: bool):
    env = make_env(seed=seed, gui=gui, out_csv=f"logs/eval_{algo}_seed{seed}")
    model = ALGOS[algo]["cls"].load(model_path)
    obs, _ = env.reset()
    done = False
    total_r = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_r += reward
        done = terminated or truncated
    # sumo-rl only flushes the CSV on the NEXT reset(); a single eval episode
    # never gets one, so save it explicitly before closing the connection.
    env.save_csv(env.out_csv_name, env.episode)
    env.close()
    csv = f"logs/eval_{algo}_seed{seed}_conn{env.label}_ep{env.episode}.csv"
    print(f"eval {algo} seed={seed} total_reward={total_r:.1f}  (metrics -> {csv})")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    if "SUMO_HOME" not in os.environ:
        sys.exit("SUMO_HOME not set — see project setup (Phase 1).")

    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=list(ALGOS), default="dqn")
    p.add_argument("--steps", type=int, default=100_000, help="training timesteps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval", type=str, default=None, help="path to a saved model to evaluate")
    p.add_argument("--gui", action="store_true", help="show sumo-gui during eval")
    p.add_argument("--defaults", action="store_true",
                   help="ignore params/<algo>.json, force default hyperparameters")
    args = p.parse_args()

    if args.eval:
        evaluate(args.algo, args.eval, seed=args.seed, gui=args.gui)
    else:
        train(args.algo, steps=args.steps, seed=args.seed, use_defaults=args.defaults)
