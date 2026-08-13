"""
Train / evaluate one algorithm on the shared intersection env.

The env, reward and observation are fixed in env_common.py; only the algorithm
changes (see algos.py). Keep the protocol identical across agents:
    equal training-step budget, 3-5 seeds each, evaluate on held-out seeds,
    report mean +/- std (compare.py aggregates).

Hyperparameters:
    - default: RL-Zoo-style defaults from algos.py
    - tuned:   if params/<algo>_<scenario>.json exists (written by tune.py) it is
               loaded automatically unless --defaults is passed; falls back to the
               legacy params/<algo>.json if the scenario-specific file is absent.

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
from env_common import (DEFAULT_MIN_GREEN, eval_csv_stem, make_env,
                        resolve_min_green)

PARAMS_DIR = "params"


def _tag(scenario: str, lam: float) -> str:
    # lam 0.5 -> "05", 1.0 -> "10", 0 -> "00" ; glob-safe filename fragment
    return f"{scenario}_lam{str(lam).replace('.', '')}"


def load_params(algo: str, use_defaults: bool, scenario: str = None) -> dict:
    """Tuned params if present (and not overridden), else defaults.

    Prefers the scenario-specific file params/<algo>_<scenario>.json (HPs tuned on
    one demand regime don't transfer — see tune.py), falling back to the legacy
    scenario-agnostic params/<algo>.json for backward compatibility.
    """
    if not use_defaults:
        candidates = []
        if scenario is not None:
            candidates.append(f"{algo}_{scenario}.json")
        candidates.append(f"{algo}.json")            # legacy fallback
        for name in candidates:
            path = os.path.join(PARAMS_DIR, name)
            if os.path.exists(path):
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
def train(algo: str, steps: int, seed: int, use_defaults: bool,
          scenario: str = "base", lam: float = 0.0, min_green: int = None):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    tag = _tag(scenario, lam)
    env = make_env(seed=seed, scenario=scenario, lam=lam, gui=False,
                   out_csv=f"logs/{algo}_{tag}_seed{seed}",
                   min_green=min_green)
    env = Monitor(env)

    params = load_params(algo, use_defaults, scenario=scenario)
    model = build(algo, env, params, seed=seed, tb_log="logs/tb")
    model.learn(total_timesteps=steps, progress_bar=True)

    path = f"models/{algo}_{tag}_seed{seed}.zip"
    model.save(path)
    env.close()
    print(f"saved {path}")


def evaluate(algo: str, model_path: str, seed: int, gui: bool,
             scenario: str = "base", lam: float = 0.0, train_seed: int = None,
             min_green: int = None):
    tag = _tag(scenario, lam)
    # `seed` is the DEMAND seed; train_seed names the checkpoint; min_green names
    # the action space. All three go in the filename, after "seed<n>" so
    # compare.py's eval_<algo>_<tag>_seed*.csv glob still hits -- see
    # env_common.eval_csv_stem for why each one has to be there.
    min_green = resolve_min_green(min_green)
    stem = eval_csv_stem(algo, tag, seed, train_seed=train_seed,
                         min_green=min_green)
    # tripinfo on: evaluation is the one place completed-trip delay/throughput
    # is needed, and it is a single episode so nothing overwrites it
    env = make_env(seed=seed, scenario=scenario, lam=lam, gui=gui, out_csv=stem,
                   tripinfo=True, min_green=min_green)
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
    csv = f"{stem}_conn{env.label}_ep{env.episode}.csv"
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
    p.add_argument("--scenario", default="base", choices=["base", "peak", "offpeak"])
    p.add_argument("--lam", type=float, default=0.0, help="safety-reward weight")
    p.add_argument("--train-seed", type=int, default=None,
                   help="seed the evaluated checkpoint was trained with; tags the "
                        "eval CSV so several checkpoints can share a demand seed")
    p.add_argument("--min-green", type=int, default=None,
                   help=f"shortest green before a switch is honoured, in seconds "
                        f"(default {DEFAULT_MIN_GREEN}, or $MIN_GREEN). This is "
                        "the action space, not a tuning detail: at 10 s even a "
                        "non-learning controller is 5.6x worse than a fixed plan "
                        "— see docs/FINDINGS_2026-08-12.md")
    args = p.parse_args()

    if args.eval:
        evaluate(args.algo, args.eval, seed=args.seed, gui=args.gui,
                 scenario=args.scenario, lam=args.lam, train_seed=args.train_seed,
                 min_green=args.min_green)
    else:
        train(args.algo, steps=args.steps, seed=args.seed, use_defaults=args.defaults,
              scenario=args.scenario, lam=args.lam, min_green=args.min_green)
