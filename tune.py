"""
Optuna hyperparameter search for one algorithm.

For each trial: sample hyperparameters (search space in algos.py), train for a
reduced budget on the train seed, then evaluate on held-out seeds and return the
mean episode reward. Optuna maximises that. The best trial's hyperparameters are
written to params/<algo>_<scenario>.json (HPs must be tuned per scenario — see the
out_path note below), which train.py then loads automatically by scenario for the
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
import time

import numpy as np
import optuna

# Per-eval-episode wall-clock cap (seconds). A divergent policy can gridlock the
# junction; with --time-to-teleport -1 the jam never clears and each SUMO step
# slows to a crawl, so one bad trial can run for hours. If an eval episode
# exceeds this, we abort it and prune the trial (a policy that can't finish an
# episode in time is not one we want to select anyway). Override via env var.
EVAL_WALL_CAP = float(os.environ.get("EVAL_WALL_CAP", "120"))

from algos import ALGOS, build
from env_common import DEFAULT_MIN_GREEN, make_env

PARAMS_DIR = "params"


def _serialisable(params: dict) -> dict:
    """Flatten cls kwargs into a JSON-safe dict (net_arch as a list, no policy_kwargs)."""
    out = dict(params)
    pk = out.pop("policy_kwargs", None)
    if pk and "net_arch" in pk:
        out["net_arch"] = pk["net_arch"]
    return out


def _eval_waiting(model, seed: int, scenario: str, lam: float,
                  min_green: int = None) -> float:
    """Cumulative system waiting time over one eval episode (lower = better).

    We tune on the *reported* metric (waiting time), NOT the shaped reward.
    Reason: reward = diff_waiting - lam*safety. At offpeak the traffic is light
    so the efficiency term is tiny and near-flat, leaving the safety term to
    dominate. The reward-optimal policy is then "never switch phase" — no phase
    changes means no conflicting movements means near-zero safety penalty — i.e.
    a do-nothing gridlock (best safety, ~zero throughput). A2C, with a near-zero
    entropy coefficient, converges straight to that degenerate optimum (the
    ~1122s constant-action collapse). Optimising waiting time directly makes the
    gridlock the *worst* possible score, so Optuna rejects it.

    Wall-clock guarded: a gridlocked policy makes SUMO crawl, so we cap the
    episode and raise TimeoutError (caught upstream -> prune) if it overruns.
    """
    env = make_env(seed=seed, scenario=scenario, lam=lam, gui=False, out_csv=None,
                   min_green=min_green)
    try:
        obs, _ = env.reset()
        done = False
        total_wait = 0.0
        t0 = time.monotonic()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_wait += float(info.get("system_total_waiting_time", 0.0))
            if time.monotonic() - t0 > EVAL_WALL_CAP:
                raise TimeoutError(
                    f"eval seed {seed} exceeded {EVAL_WALL_CAP}s (policy likely gridlocked)"
                )
            done = terminated or truncated
        return total_wait
    finally:
        env.close()


def make_objective(algo: str, steps: int, train_seed: int, eval_seeds, scenario: str,
                   lam: float, min_green: int = None):
    def objective(trial: optuna.Trial) -> float:
        params = ALGOS[algo]["sample"](trial)
        # Hyperparameters are selected FOR an action space: a policy tuned
        # against a 10 s floor is not the right policy for a 60 s one, so the
        # floor has to be identical here and in training.
        env = make_env(seed=train_seed, scenario=scenario, lam=lam, gui=False,
                       out_csv=None, min_green=min_green)
        try:
            model = build(algo, env, params, seed=train_seed, tb_log=None)
            model.learn(total_timesteps=steps, progress_bar=False)
        except Exception as e:
            # a bad hyperparameter combo shouldn't kill the whole study
            print(f"trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
        finally:
            env.close()

        # Objective = cumulative waiting time (lower = better); Optuna maximises,
        # so return the negative. A TimeoutError from a gridlocked/crawling eval
        # prunes the trial instead of letting it run for hours.
        try:
            waits = [_eval_waiting(model, s, scenario=scenario, lam=lam,
                                   min_green=min_green) for s in eval_seeds]
        except TimeoutError as e:
            print(f"trial {trial.number} pruned: {e}")
            raise optuna.TrialPruned()
        mean_wait = float(np.mean(waits))
        trial.set_user_attr("eval_waiting", waits)
        return -mean_wait

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
    p.add_argument("--min-green", type=int, default=None,
                   help=f"action-space floor to tune against, in seconds "
                        f"(default {DEFAULT_MIN_GREEN}, or $MIN_GREEN). Must match "
                        "the floor training will use — params selected at one "
                        "floor do not transfer to another")
    args = p.parse_args()

    os.makedirs(PARAMS_DIR, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{args.algo}_tuning",
        sampler=optuna.samplers.TPESampler(seed=args.train_seed),
    )
    objective = make_objective(args.algo, args.steps, args.train_seed, args.eval_seeds,
                               args.scenario, args.lam, min_green=args.min_green)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = ALGOS[args.algo]["sample"](
        optuna.trial.FixedTrial(study.best_params)
    )
    # Per-scenario file: HPs tuned on one demand regime (e.g. peak) collapse when
    # reused on another (offpeak's near-zero reward starves a peak-tuned tiny lr →
    # constant-action gridlock). train.py loads params/<algo>_<scenario>.json.
    out_path = os.path.join(PARAMS_DIR, f"{args.algo}_{args.scenario}.json")
    with open(out_path, "w") as f:
        json.dump(_serialisable(best), f, indent=2)

    # best_value is -mean_waiting (we maximise the negative); report as waiting
    print(f"\nbest mean eval waiting time: {-study.best_value:.1f}")
    print(f"best params written to {out_path}")


if __name__ == "__main__":
    main()
