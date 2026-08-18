"""Non-RL corridor baselines (green-wave, max-pressure) through the same eval-CSV
frame compare.py consumes. No learning: control is rule-based each step."""
import argparse
import os

import corridor_control as cc
from env_common import (CORRIDOR_SCENARIOS, DEFAULT_MIN_GREEN,
                        make_corridor_env, resolve_min_green)

CONTROLLERS = ("green_wave", "max_pressure")

# green-wave inputs, sourced from the network: signal x-positions (m) of C1,C2,C3
# from corridor.nod.xml; free-flow speed from the arterial edges in corridor.edg.xml.
# Keep in sync if the corridor geometry/speed changes.
SIGNAL_POSITIONS = [0.0, 200.0, 400.0]
FREE_FLOW_SPEED = 13.89


def _phase_movements(ts, phase_index) -> set:
    """The (incoming_lane, outgoing_lane) pairs a green phase discharges.

    Phase state is read from ts.green_phases[p].state (sumolib.net.Phase).
    A link index i is served by a phase if state[i] in 'Gg'.
    links = ts.sumo.trafficlight.getControlledLinks(ts.id): links[i] is a list
    of (in_lane, out_lane, via) tuples.
    """
    links = ts.sumo.trafficlight.getControlledLinks(ts.id)
    state = ts.green_phases[phase_index].state
    movements = set()
    for i, link in enumerate(links):
        if i < len(state) and state[i] in "Gg":
            for conn in link:
                if conn:
                    movements.add((conn[0], conn[1]))   # (in_lane, out_lane)
    return movements


def green_wave_actions(env) -> dict:
    """Fixed-time coordinated plan, evaluated at the current simulation TIME.

    Each signal holds a phase for `plan_phase_seconds(min_green, ...)` and its
    whole cycle is shifted by the free-flow travel time from the first signal,
    so a platoon released upstream meets the same phase downstream.

    The predecessor asked for `(step - offset_steps) % num_green_phases`, i.e. it
    alternated its request every decision step and let sumo-rl's min_green
    blocking decide the switching times. That produced a controller with no green
    duration of its own: the realised cycle came out 15, 15, 25, 25, 35, 35 s for
    min_green 5, 10, 15, 20, 25, 30 -- adjacent floors collapsing onto identical
    plans -- and the offsets were re-timed by the blocking, which is the
    progression a green wave exists to create. A fixed-time baseline that is not
    a fixed-time plan is the defect in docs/FINDINGS_2026-08-12.md section 6.

    Switching still lands on the decision grid, so an offset of 14.4 s is
    realised as 15 s. That quantisation is a property of delta_time=5 and is
    disclosed; it is not the same thing as having no offset at all.
    """
    offsets = cc.green_wave_offsets(SIGNAL_POSITIONS, free_flow_speed=FREE_FLOW_SPEED)
    actions = {}
    for i, ts_id in enumerate(env.ts_ids):
        ts = env.traffic_signals[ts_id]
        phase_seconds = cc.plan_phase_seconds(ts.min_green, ts.yellow_time,
                                              env.delta_time)
        actions[ts_id] = cc.fixed_time_phase(env.sim_step, offsets[i],
                                             ts.num_green_phases, phase_seconds)
    return actions


def _max_pressure_actions(env):
    """Return per-agent action dict using the max-pressure rule.

    This is real max-pressure: every movement contributes its upstream queue
    MINUS the queue on the lane it discharges into. The first version of this
    function passed outgoing_queue=0.0, which is not max-pressure at all -- it
    reduces to "serve the longest incoming queue", a purely local rule with no
    knowledge of whether the receiving lane can accept traffic.

    That mattered here for two reasons. It is the reference an RL controller
    has to beat, and beating a weakened reference is the defect this project
    already withdrew a headline over (docs/FINDINGS_2026-08-12.md section 6, the
    "fixed-time" baseline that was not a fixed-time plan). And on a corridor the
    downstream term is precisely the coordination signal: without it the
    controller cheerfully discharges into a lane that is already backed up,
    which is the spillback case a multi-intersection controller exists to
    handle.

    Queues are read once per lane per decision step and cached: the same lane
    appears in several movements, and each read is a TraCI round trip.
    """
    actions = {}
    for ts_id in env.ts_ids:
        ts = env.traffic_signals[ts_id]
        cache = {}

        def queue_of(lane, _ts=ts, _cache=cache):
            if lane not in _cache:
                _cache[lane] = _ts.sumo.lane.getLastStepHaltingNumber(lane)
            return _cache[lane]

        pressures = {
            p: cc.phase_pressure(_phase_movements(ts, p), queue_of)
            for p in range(ts.num_green_phases)
        }
        actions[ts_id] = cc.max_pressure_phase(pressures)
    return actions


def run(scenario: str, controller: str, seed: int, min_green: int = None,
        tripinfo: bool = True) -> str:
    """Run one controller over one corridor scenario for a full episode and write
    the eval CSV. Returns the CSV path
    (logs/eval_<controller>_<scenario>_seed<seed>_mg<min_green>_conn<label>_ep<episode>.csv).

    `min_green` is not a detail here -- it defines both baselines:

      * green_wave holds each phase for plan_phase_seconds(min_green), so the
        floor sets the plan's phase duration and therefore its cycle. A run at
        min_green=10 is a 15 s-phase plan, near the regime the single
        intersection measured a perfect-information controller to be 5.6x worse
        than a fixed plan in. The floor is this baseline's design parameter and
        has to be swept before the baseline means anything.
      * max_pressure is reactive, so the floor is a genuine constraint on it --
        the same knob analysis/actuated.py swept at the single junction.

    Neither had ever been calibrated; that is Phase 1. The floor is recorded in
    the CSV name so two floors cannot be averaged into one row.
    """
    os.makedirs("logs", exist_ok=True)
    min_green = resolve_min_green(min_green)
    csv = f"logs/eval_{controller}_{scenario}_seed{seed}_mg{min_green}"
    env = make_corridor_env(seed=seed, scenario=scenario, lam=0.0, out_csv=csv,
                            min_green=min_green, tripinfo=tripinfo)
    env.reset()
    done = False
    while not done:
        if controller == "green_wave":
            actions = green_wave_actions(env)
        else:
            actions = _max_pressure_actions(env)
        _, _, dones, _ = env.step(actions)
        done = dones["__all__"]
    # sumo-rl only flushes the CSV on the NEXT reset(); a single eval episode
    # never gets one, so save it explicitly (mirrors baseline.py exactly).
    env.save_csv(env.out_csv_name, env.episode)
    env.close()
    out = (
        f"logs/eval_{controller}_{scenario}_seed{seed}_mg{min_green}"
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
        choices=list(CORRIDOR_SCENARIOS),
    )
    p.add_argument("--controller", default="green_wave", choices=CONTROLLERS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-green", type=int, default=None,
                   help=f"action-space floor in seconds (default "
                        f"{DEFAULT_MIN_GREEN}, or $MIN_GREEN). For green_wave "
                        "this IS the green duration -- see run()")
    p.add_argument("--no-tripinfo", action="store_true",
                   help="skip per-trip output (the ranking metric comes from it)")
    args = p.parse_args()
    run(args.scenario, args.controller, args.seed, min_green=args.min_green,
        tripinfo=not args.no_tripinfo)
