"""Fixed-time (no-learning) baseline through the SAME env + eval CSV frame.

Steps the shared env with a cyclic phase policy so the fixed-time baseline is
recorded in exactly the metric format compare.py consumes, per scenario.
"""
import argparse
import os

from env_common import make_env


def run_baseline(scenario: str, seed: int) -> str:
    os.makedirs("logs", exist_ok=True)
    csv = f"logs/eval_fixedtime_{scenario}_seed{seed}"
    # lam=0 -> reward term irrelevant here; we never learn, just cycle phases
    env = make_env(seed=seed, scenario=scenario, lam=0.0, gui=False, out_csv=csv)

    obs, _ = env.reset()
    n_actions = env.action_space.n
    action, done = 0, False
    while not done:
        obs, _, terminated, truncated, _ = env.step(action)
        action = (action + 1) % n_actions   # round-robin green phases = fixed-time
        done = terminated or truncated
    # sumo-rl only flushes the CSV on the NEXT reset(); a single eval episode
    # never gets one, so save it explicitly.
    # Mirrors train.py evaluate() exactly: env.save_csv(env.out_csv_name, env.episode)
    env.save_csv(env.out_csv_name, env.episode)
    env.close()
    csv_out = f"logs/eval_fixedtime_{scenario}_seed{seed}_conn{env.label}_ep{env.episode}.csv"
    print(f"baseline written: {csv_out}")
    return csv_out


if __name__ == "__main__":
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="base", choices=["base", "peak", "offpeak"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run_baseline(args.scenario, args.seed)
