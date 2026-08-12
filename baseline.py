"""Fixed-time (no-learning) baseline through the SAME env + eval CSV frame.

Steps the shared env with a cyclic phase policy so the fixed-time baseline is
recorded in exactly the metric format compare.py consumes, per scenario.

Green duration
--------------
`--green` sets how long each green phase is held, in simulation seconds. The
original behaviour switched on every decision step, i.e. a 10 s-green cycler
(delta_time=5 plus yellow_time=3 rounds up to min_green=10) -- not the 42 s
program in the net file, and not a competently timed plan. The static sweep in
analysis/static_timing.py shows the intersection is strongly green-duration
sensitive at peak (10 s: 26.7 s mean wait, 60 s: 11.5 s), so the green the
baseline runs at decides whether the comparison against RL is fair at all.

The green appears in the CSV name AFTER the seed fragment
(`..._seed<seed>_g<green>_conn<N>_ep<M>.csv`) so compare.py's
`eval_fixedtime_<scenario>_seed*` glob still matches. That also means CSVs for
two different greens in the same logs dir merge into one "fixedtime" group --
keep one green per logs dir, or pass compare.py a different --logs.
"""
import argparse
import os

from env_common import make_env

DEFAULT_GREEN = 60   # best static plan at peak (analysis/static_timing.py)


def run_baseline(scenario: str, seed: int, green: int = DEFAULT_GREEN,
                 teleport: int = None) -> str:
    os.makedirs("logs", exist_ok=True)
    csv = f"logs/eval_fixedtime_{scenario}_seed{seed}_g{green}"
    # lam=0 -> reward term irrelevant here; we never learn, just cycle phases
    env = make_env(seed=seed, scenario=scenario, lam=0.0, gui=False, out_csv=csv,
                   teleport=teleport)

    # decision steps to hold one green; the env clamps to min_green/max_green
    hold = max(1, green // env.delta_time)
    obs, _ = env.reset()
    n_actions = env.action_space.n
    action, done, i = 0, False, 0
    while not done:
        obs, _, terminated, truncated, _ = env.step(action)
        i += 1
        if i % hold == 0:
            action = (action + 1) % n_actions   # round-robin greens = fixed-time
        done = terminated or truncated
    # sumo-rl only flushes the CSV on the NEXT reset(); a single eval episode
    # never gets one, so save it explicitly.
    # Mirrors train.py evaluate() exactly: env.save_csv(env.out_csv_name, env.episode)
    env.save_csv(env.out_csv_name, env.episode)
    env.close()
    csv_out = f"{csv}_conn{env.label}_ep{env.episode}.csv"
    print(f"baseline written: {csv_out}")
    return csv_out


if __name__ == "__main__":
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="base", choices=["base", "peak", "offpeak"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--green", type=int, default=DEFAULT_GREEN,
                   help="seconds each green phase is held (default: best static plan)")
    p.add_argument("--teleport", type=int, default=None,
                   help="SUMO --time-to-teleport; default from TIME_TO_TELEPORT env var")
    args = p.parse_args()
    run_baseline(args.scenario, args.seed, args.green, args.teleport)
