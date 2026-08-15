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

The study is kept in a SQLite file (--storage, one per algo/scenario/floor), so
a search is RESUMABLE and PARALLEL: --trials is a target for the study as a
whole, not a per-process count. A killed run tops the study back up to the
target instead of starting from zero, and several workers can share one study
by pointing at the same storage with different --sampler-seed values. At peak,
one trial is ~30 minutes of wall clock, so a 30-trial search is a multi-hour
job that will be interrupted at least once -- see run_tune_mg60.sh.
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
from env_common import DEFAULT_MIN_GREEN, make_env, resolve_min_green

PARAMS_DIR = "params"


def _serialisable(params: dict) -> dict:
    """Flatten cls kwargs into a JSON-safe dict (net_arch as a list, no policy_kwargs)."""
    out = dict(params)
    pk = out.pop("policy_kwargs", None)
    if pk and "net_arch" in pk:
        out["net_arch"] = pk["net_arch"]
    return out


def study_id(algo: str, scenario: str, min_green: int) -> str:
    """Name of the study for this search.

    Floor and scenario are part of a study's identity: two floors are two
    different search problems and must never share a trial history.
    """
    return f"{algo}_{scenario}_mg{min_green}"


def storage_url(study_name: str) -> str:
    return f"sqlite:///{os.path.join(PARAMS_DIR, study_name + '.db')}"


def make_storage(url: str):
    """Storage with a heartbeat, so a killed worker doesn't strand its trial.

    Runs here are killed mid-trial routinely. Without a heartbeat the trial it
    was running stays RUNNING in the database forever: it never completes, never
    counts towards the target, and the sampler keeps treating it as in flight.
    With one, a worker that stops writing for `grace_period` is declared FAILED
    and its slot is reusable.
    """
    return optuna.storages.RDBStorage(
        url=url, heartbeat_interval=60, grace_period=180,
    )


def _completed(study) -> int:
    return sum(1 for t in study.trials
               if t.state == optuna.trial.TrialState.COMPLETE)


def _stop_at_target(target: int):
    """Stop this worker once the STUDY (not this process) has `target` trials.

    With several workers on one storage, each would otherwise run `target`
    trials of its own. It also makes a resumed search top up to the target
    rather than repeat it.
    """
    def callback(study, trial):
        if _completed(study) >= target:
            study.stop()
    return callback


def _write_params(path: str, params: dict, provenance: dict) -> None:
    """Write the tuned params plus the settings they were selected under.

    The provenance keys are underscore-prefixed and stripped by train.py before
    the dict reaches the algorithm constructor. They exist because
    hyperparameters are selected FOR an action space and a training budget:
    params tuned at a 10 s floor are not the right params for a 60 s one, and
    the file used to record neither -- so a floor could silently change under a
    params file that looked unchanged.
    """
    payload = dict(provenance)
    payload.update(params)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)          # atomic: parallel workers can't tear the file


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
    p.add_argument("--storage", default=None,
                   help="Optuna storage URL for the study (default: a SQLite "
                        "file under params/, one per algo/scenario/floor). The "
                        "study resumes from it, so --trials is a target for the "
                        "study rather than a count for this process")
    p.add_argument("--sampler-seed", type=int, default=None,
                   help="TPE sampler seed (default: --train-seed). Give parallel "
                        "workers on one storage different values, or their "
                        "startup trials are identical and the search wastes them")
    p.add_argument("--init-only", action="store_true",
                   help="create the study and exit, running no trials. Workers "
                        "starting together on a fresh SQLite file race each "
                        "other's schema creation, so a launcher calls this once "
                        "before forking them")
    args = p.parse_args()

    os.makedirs(PARAMS_DIR, exist_ok=True)
    min_green = resolve_min_green(args.min_green)

    study_name = study_id(args.algo, args.scenario, min_green)
    storage = make_storage(args.storage or storage_url(study_name))

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=args.train_seed if args.sampler_seed is None else args.sampler_seed),
    )
    if args.init_only:
        print(f"study {study_name} ready at {storage.url}")
        return

    done = _completed(study)
    if done:
        print(f"resuming study {study_name}: {done}/{args.trials} trials complete")
    if done >= args.trials:
        print("target already met — nothing to run")
    else:
        objective = make_objective(args.algo, args.steps, args.train_seed, args.eval_seeds,
                                   args.scenario, args.lam, min_green=min_green)
        # n_trials bounds this worker; the callback stops it as soon as the
        # SHARED study reaches the target, so N workers still run ~N trials
        # total rather than N x target.
        study.optimize(objective, n_trials=args.trials - done,
                       callbacks=[_stop_at_target(args.trials)],
                       show_progress_bar=False)

    if not _completed(study):
        sys.exit("no trial completed — nothing to write (check the log for prunes)")

    best = ALGOS[args.algo]["sample"](
        optuna.trial.FixedTrial(study.best_params)
    )
    # Per-scenario file: HPs tuned on one demand regime (e.g. peak) collapse when
    # reused on another (offpeak's near-zero reward starves a peak-tuned tiny lr →
    # constant-action gridlock). train.py loads params/<algo>_<scenario>.json.
    out_path = os.path.join(PARAMS_DIR, f"{args.algo}_{args.scenario}.json")
    _write_params(out_path, _serialisable(best), {
        "_min_green": min_green,
        "_scenario": args.scenario,
        "_lam": args.lam,
        "_tune_steps": args.steps,
        "_trials": _completed(study),
    })

    # best_value is -mean_waiting (we maximise the negative); report as waiting
    print(f"\nbest mean eval waiting time: {-study.best_value:.1f}")
    print(f"best params written to {out_path} (min_green={min_green}, "
          f"{_completed(study)} trials)")


if __name__ == "__main__":
    main()
