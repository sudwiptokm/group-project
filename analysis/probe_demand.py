"""Fixed-time probe at a candidate demand scale.

Writes a scaled copy of traffic.rou.xml into the scratchpad, registers it as a
throwaway scenario, and runs the round-robin fixed-time policy over it. Nothing
in the repo is modified: the real traffic_*.rou.xml files are left alone.

    python probe_demand.py --factor 0.9 --seed 42
"""
import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd

import env_common
from env_common import make_env

OUT = os.path.join(REPO, "analysis", "probe_logs")
_FLOW = re.compile(r'vehsPerHour="([0-9.]+)"')


def scaled_route(factor: float) -> str:
    """Write traffic.rou.xml with every flow rate scaled by `factor`."""
    dst = os.path.join(OUT, f"probe_f{factor:.2f}.rou.xml")
    text = open("traffic.rou.xml").read()
    text = _FLOW.sub(lambda m: f'vehsPerHour="{max(1, round(float(m.group(1)) * factor))}"', text)
    with open(dst, "w") as fh:
        fh.write(text)
    return dst


def probe(factor: float, seed: int, teleport: int = -1) -> dict:
    os.makedirs(OUT, exist_ok=True)
    env_common.SCENARIO_ROUTES["probe"] = scaled_route(factor)
    csv = os.path.join(OUT, f"f{factor:.2f}_tt{teleport}_seed{seed}")
    env = make_env(seed=seed, scenario="probe", lam=0.0, gui=False, out_csv=csv,
                   teleport=teleport)

    obs, _ = env.reset()
    action, done = 0, False
    # all three are per-step counts, so they only mean anything accumulated
    teleports = departed = arrived = 0
    while not done:
        obs, _, terminated, truncated, _ = env.step(action)
        # teleports are the escape hatch out of deadlock; count them so a
        # "measurable" run that only measures teleporting is visible as such
        teleports += env.sumo.simulation.getStartingTeleportNumber()
        departed += env.sumo.simulation.getDepartedNumber()
        arrived += env.sumo.simulation.getArrivedNumber()
        action = (action + 1) % env.action_space.n
        done = terminated or truncated
    env.save_csv(env.out_csv_name, env.episode)
    env.close()

    df = pd.read_csv(f"{csv}_conn{env.label}_ep{env.episode}.csv")
    # sustained gridlock = mean speed stays under 0.05 m/s to the end of the run
    s, t = df.system_mean_speed.values, df.step.values
    onset = next((t[i] for i in range(len(s)) if s[i] < 0.05 and (s[i:] < 0.05).all()), None)
    row = {
        "factor": factor,
        "seed": seed,
        "tt": teleport,
        "teleports": teleports,
        "departed": departed,
        "arrived": arrived,
        "wait": df.system_mean_waiting_time.mean(),
        "speed": df.system_mean_speed.mean(),
        "stopped": df.system_total_stopped.mean(),
        "final_speed": df.system_mean_speed.iloc[-1],
        "gridlock_onset": onset,
    }
    print("PROBE " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--factor", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--teleport", type=int, default=-1)
    a = p.parse_args()
    probe(a.factor, a.seed, a.teleport)
