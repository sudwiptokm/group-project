"""Non-RL corridor baselines (green-wave, max-pressure) through the same eval-CSV
frame compare.py consumes. No learning: control is rule-based each step."""
import argparse
import os

import corridor_control as cc
from env_common import make_corridor_env

CONTROLLERS = ("green_wave", "max_pressure")

# green-wave inputs, sourced from the network: signal x-positions (m) of C1,C2,C3
# from corridor.nod.xml; free-flow speed from the arterial edges in corridor.edg.xml.
# Keep in sync if the corridor geometry/speed changes.
SIGNAL_POSITIONS = [0.0, 200.0, 400.0]
FREE_FLOW_SPEED = 13.89


def _max_pressure_actions(env):
    """Return per-agent action dict using max-pressure rule.

    For each traffic signal, compute the pressure for each green phase:
    pressure = total halting vehicles on the incoming lanes that phase gives
    green to (downstream approximated as 0). Pick the max-pressure phase.

    Phase state is read from ts.green_phases[p].state (sumolib.net.Phase).
    A link index i is served by a phase if state[i] in 'Gg'.
    links = ts.sumo.trafficlight.getControlledLinks(ts.id): links[i] is a list
    of (in_lane, out_lane, via) tuples; in_lane = links[i][0][0].
    """
    actions = {}
    for ts_id in env.ts_ids:
        ts = env.traffic_signals[ts_id]
        links = ts.sumo.trafficlight.getControlledLinks(ts.id)
        pressures = {}
        for p in range(ts.num_green_phases):
            state = ts.green_phases[p].state
            served_lanes = {
                links[i][0][0]
                for i in range(len(state))
                if i < len(links) and links[i] and state[i] in "Gg"
            }
            incoming = sum(
                ts.sumo.lane.getLastStepHaltingNumber(lane)
                for lane in served_lanes
            )
            # downstream queue approximated as 0 (corridor exits drain freely);
            # use the shared pure helper so the pressure definition lives in one place
            pressures[p] = cc.movement_pressure(incoming, outgoing_queue=0.0)
        actions[ts_id] = cc.max_pressure_phase(pressures)
    return actions


def run(scenario: str, controller: str, seed: int) -> str:
    """Run one controller over one corridor scenario for a full episode and write
    the eval CSV. Returns the CSV path
    (logs/eval_<controller>_<scenario>_seed<seed>_conn<label>_ep<episode>.csv)."""
    os.makedirs("logs", exist_ok=True)
    csv = f"logs/eval_{controller}_{scenario}_seed{seed}"
    env = make_corridor_env(seed=seed, scenario=scenario, lam=0.0, out_csv=csv)
    env.reset()
    offsets = cc.green_wave_offsets(SIGNAL_POSITIONS, free_flow_speed=FREE_FLOW_SPEED)
    step = 0
    done = False
    while not done:
        if controller == "green_wave":
            # green-wave is inline because it needs `step` (the loop counter) as
            # state; max-pressure is stateless per step so it lives in a helper.
            actions = {}
            for i, ts_id in enumerate(env.ts_ids):
                ngp = env.traffic_signals[ts_id].num_green_phases
                # offsets are quantised to whole decision steps: at delta_time=5s a
                # 14.4s offset becomes 2 steps (10s). This coarsens the green wave —
                # a known limitation of the discrete decision interval, not a bug.
                shifted = step - int(offsets[i] // env.delta_time)
                actions[ts_id] = max(shifted, 0) % ngp
        else:
            actions = _max_pressure_actions(env)
        _, _, dones, _ = env.step(actions)
        done = dones["__all__"]
        step += 1
    # sumo-rl only flushes the CSV on the NEXT reset(); a single eval episode
    # never gets one, so save it explicitly (mirrors baseline.py exactly).
    env.save_csv(env.out_csv_name, env.episode)
    env.close()
    out = (
        f"logs/eval_{controller}_{scenario}_seed{seed}"
        f"_conn{env.label}_ep{env.episode}.csv"
    )
    print(f"corridor baseline written: {out}")
    return out


if __name__ == "__main__":
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scenario",
        default="corridor_offpeak",
        choices=["corridor_peak", "corridor_offpeak"],
    )
    p.add_argument("--controller", default="green_wave", choices=CONTROLLERS)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args.scenario, args.controller, args.seed)
