"""Is there anything for an ADAPTIVE controller to win here? Measure, don't argue.

The static sweep shows a fixed plan beating every learned policy this project
produced. Two very different explanations fit that:

  A. our RL failed to find the adaptive policy that exists, or
  B. at a 2-phase isolated junction there is no adaptive policy worth finding.

Training more agents cannot separate A from B -- a null result is consistent
with both. A non-learning demand-responsive controller can: it needs no reward,
no credit assignment and no sample budget, so if IT cannot beat the best static
plan, the headroom is not there to be found (B). If it can, the gap is ours (A)
and the action space / reward is where to look.

The controller here is queue-actuated: every decision step it serves whichever
green phase has the largest PCU-weighted queue on the lanes it discharges,
subject to the environment's `min_green` (sumo-rl silently ignores a switch
request made earlier -- TrafficSignal.set_next_phase). That is the classic
actuated-control policy and the natural non-learning upper reference for a
2-phase junction; it is deliberately NOT max-pressure, which needs downstream
occupancy and would measure something else on approaches this short.

`--min-green` is the knob the finding points at: with a 3 s amber, a controller
free to switch every 10 s loses 23% of the cycle to clearance. Sweeping it says
whether a longer floor is what unlocks adaptive control.

    python analysis/actuated.py --min-green 10 --seed 42
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd

from analysis.tripinfo import reduce_tripinfo
from env_common import _vehicle_pcu, make_env, tripinfo_path

OUT = os.path.join(REPO, "analysis", "actuated_logs")


def _served_lanes(ts) -> list:
    """For each green phase, the incoming lanes it discharges.

    Read off the phase state strings rather than hard-coded: link i of
    getControlledLinks corresponds to character i of the state, so a 'G'/'g'
    there means this phase serves that link's incoming lane.
    """
    links = ts.sumo.trafficlight.getControlledLinks(ts.id)
    served = []
    for phase in ts.green_phases:
        lanes = set()
        for i, link in enumerate(links):
            if i < len(phase.state) and phase.state[i] in "Gg":
                for conn in link:
                    if conn:
                        lanes.add(conn[0])   # incoming lane
        served.append(sorted(lanes))
    return served


def _queue_pcu(ts, lanes) -> float:
    """PCU-weighted count of halting vehicles waiting on `lanes`.

    PCU-weighted for the same reason the observation is: three motorcycles are
    not three cars of demand, and an unweighted queue would over-serve the
    two-wheeler approaches.
    """
    total = 0.0
    for lane in lanes:
        for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
            if ts.sumo.vehicle.getSpeed(vid) < 0.1:
                total += _vehicle_pcu(ts.sumo.vehicle.getTypeID(vid))
    return total


def run(seed: int, min_green: int = 10, scenario: str = "peak",
        teleport: int = 300) -> dict:
    os.makedirs(OUT, exist_ok=True)
    csv = os.path.join(OUT, f"{scenario}_mg{min_green}_seed{seed}")
    env = make_env(seed=seed, scenario=scenario, lam=0.0, gui=False, out_csv=csv,
                   teleport=teleport, tripinfo=True, min_green=min_green)

    obs, _ = env.reset()
    ts = env.traffic_signals[env.ts_ids[0]]
    served = _served_lanes(ts)

    action, done, switches = 0, False, 0
    while not done:
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            break
        queues = [_queue_pcu(ts, lanes) for lanes in served]
        best = max(range(len(queues)), key=queues.__getitem__)
        if best != action:
            switches += 1
        action = best
    env.save_csv(env.out_csv_name, env.episode)
    env.close()

    df = pd.read_csv(f"{csv}_conn{env.label}_ep{env.episode}.csv")
    trips = reduce_tripinfo(tripinfo_path(csv))
    row = {
        "min_green": min_green,
        "seed": seed,
        "delay": trips.get("trip_time_loss_mean", float("nan")),
        "completed": trips.get("trips_completed", 0),
        "wait": df.system_mean_waiting_time.mean(),
        "switch_requests": switches,
    }
    print("ACTUATED " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-green", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scenario", default="peak", choices=["base", "peak", "offpeak"])
    p.add_argument("--teleport", type=int, default=300)
    a = p.parse_args()
    run(a.seed, a.min_green, a.scenario, a.teleport)
