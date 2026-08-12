"""How much headroom is there for RL at this intersection?

Sweeps static green durations to find the best achievable fixed plan. If the
best static plan is close to what the learned policies score, the action space
(2 green phases, min_green 10, decisions every 5 s) simply does not leave room
for RL to win, and the null result is structural rather than a training-budget
artefact.

The sweep is NOT capped at make_env's max_green=60: sumo-rl's TrafficSignal
stores max_green and never reads it (traffic_signal.py:77 is its only use), so
nothing enforces an upper bound on green. Longer greens are reachable both here
and by a learned policy, and the sweep has to cover them to claim an optimum.

Also records teleports and the per-step reward so the reward-corruption
hypothesis can be checked: a teleport erases a jammed vehicle's accumulated
waiting time, which shows up as a large positive diff-waiting-time reward the
policy did nothing to earn.

    python static_timing.py --green 30 --seed 42
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import pandas as pd

from analysis.tripinfo import reduce_tripinfo
from env_common import make_env, tripinfo_path

OUT = os.path.join(REPO, "analysis", "static_logs")


def run(green: int, seed: int, teleport: int = 300) -> dict:
    """Hold each green phase for `green` seconds, then switch."""
    os.makedirs(OUT, exist_ok=True)
    csv = os.path.join(OUT, f"g{green}_seed{seed}")
    env = make_env(seed=seed, scenario="peak", lam=0.0, gui=False, out_csv=csv,
                   teleport=teleport, tripinfo=True)

    hold = max(1, green // 5)          # decision steps per green (delta_time=5)
    obs, _ = env.reset()
    action, done, i = 0, False, 0
    teleports = 0
    rewards, tp_step = [], []
    while not done:
        obs, r, terminated, truncated, _ = env.step(action)
        n_tp = env.sumo.simulation.getStartingTeleportNumber()
        teleports += n_tp
        rewards.append(r)
        tp_step.append(n_tp)
        i += 1
        if i % hold == 0:
            action = (action + 1) % env.action_space.n
        done = terminated or truncated
    env.save_csv(env.out_csv_name, env.episode)
    env.close()

    df = pd.read_csv(f"{csv}_conn{env.label}_ep{env.episode}.csv")
    r = np.array(rewards)
    tp = np.array(tp_step)
    # does reward spike on teleport steps relative to the rest?
    r_tp = r[tp > 0].mean() if (tp > 0).any() else float("nan")
    r_no = r[tp == 0].mean() if (tp == 0).any() else float("nan")

    # completed-trip metrics: `wait` below is the survivorship-biased in-network
    # average, kept for continuity with the earlier sweep. `delay` is the metric
    # the results now rank on. See analysis/tripinfo.py.
    trips = reduce_tripinfo(tripinfo_path(csv))

    row = {
        "green": green,
        "seed": seed,
        "delay": trips.get("trip_time_loss_mean", float("nan")),
        "completed": trips.get("trips_completed", 0),
        "wait": df.system_mean_waiting_time.mean(),
        "speed": df.system_mean_speed.mean(),
        "stopped": df.system_total_stopped.mean(),
        "teleports": teleports,
        "rew_mean": r.mean(),
        "rew_on_teleport": r_tp,
        "rew_off_teleport": r_no,
    }
    print("STATIC " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--green", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--teleport", type=int, default=300)
    a = p.parse_args()
    run(a.green, a.seed, a.teleport)
