"""
Optuna hyperparameter search for one algorithm.

For each trial: sample hyperparameters (search space in algos.py), train for a
reduced budget on the train seed, then evaluate on held-out seeds and return the
mean episode reward. Optuna maximises that. The best trial's hyperparameters are
written to params/<algo>.json, which train.py then loads automatically for the
full-budget, multi-seed runs used in the comparison.

Keep the search cheap enough to run per algorithm:
    python tune.py --algo dqn   --trials 30 --steps 20000
    python tune.py --algo qrdqn --trials 30 --steps 20000
    python tune.py --algo ppo   --trials 30 --steps 20000
    python tune.py --algo a2c   --trials 30 --steps 20000

Then train with the tuned params (picked up automatically):
    python train.py --algo dqn --steps 100000 --seed 0
"""

import argparse
import json
import os
import sys

import numpy as np
import optuna

from algos import ALGOS, build
from env_common import make_env

PARAMS_DIR = "params"


def _serialisable(params: dict) -> dict:
    """Flatten cls kwargs into a JSON-safe dict (net_arch as a list, no policy_kwargs)."""
    out = dict(params)
    pk = out.pop("policy_kwargs", None)
    if pk and "net_arch" in pk:
        out["net_arch"] = pk["net_arch"]
    return out


def _eval_reward(model, seed: int, scenario: str, lam: float) -> float:
    env = make_env(seed=seed, scenario=scenario, lam=lam, gui=False, out_csv=None)
    try:
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        return total
    finally:
        env.close()


def make_objective(algo: str, steps: int, train_seed: int, eval_seeds, scenario: str, lam: float):
    def objective(trial: optuna.Trial) -> float:
        params = ALGOS[algo]["sample"](trial)
        env = make_env(seed=train_seed, scenario=scenario, lam=lam, gui=False, out_csv=None)
        try:
            model = build(algo, env, params, seed=train_seed, tb_log=None)
            model.learn(total_timesteps=steps, progress_bar=False)
        except Exception as e:
            # a bad hyperparameter combo shouldn't kill the whole study
            print(f"trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
        finally:
            env.close()

        rewards = [_eval_reward(model, s, scenario=scenario, lam=lam) for s in eval_seeds]
        mean_r = float(np.mean(rewards))
        trial.set_user_attr("eval_rewards", rewards)
        return mean_r

    return objective


def main():
    if "SUMO_HOME" not in os.environ:
        sys.exit("SUMO_HOME not set — see project setup (Phase 1).")

    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=list(ALGOS), required=True)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--steps", type=int, default=20_000, help="per-trial training budget")
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--scenario", default="peak", choices=["base", "peak", "offpeak"])
    p.add_argument("--lam", type=float, default=0.5, help="safety-reward weight for tuning")
    args = p.parse_args()

    os.makedirs(PARAMS_DIR, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{args.algo}_tuning",
        sampler=optuna.samplers.TPESampler(seed=args.train_seed),
    )
    objective = make_objective(args.algo, args.steps, args.train_seed, args.eval_seeds, args.scenario, args.lam)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = ALGOS[args.algo]["sample"](
        optuna.trial.FixedTrial(study.best_params)
    )
    out_path = os.path.join(PARAMS_DIR, f"{args.algo}.json")
    with open(out_path, "w") as f:
        json.dump(_serialisable(best), f, indent=2)

    print(f"\nbest mean eval reward: {study.best_value:.1f}")
    print(f"best params written to {out_path}")


if __name__ == "__main__":
    main()
