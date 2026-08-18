"""Pure control logic for the non-RL corridor baselines.

Kept SUMO-free so the math is unit-tested in isolation; corridor_baseline.py
wires these to live TraCI state.
"""
from typing import Dict, List


def green_wave_offsets(positions: List[float], free_flow_speed: float) -> List[float]:
    """Signal start offsets (s) for a forward green-wave: each downstream signal
    starts later by the free-flow travel time from the first signal."""
    if free_flow_speed <= 0:
        raise ValueError("free_flow_speed must be positive")
    x0 = positions[0]
    return [(p - x0) / free_flow_speed for p in positions]


def plan_phase_seconds(min_green: int, yellow_time: int, delta_time: int) -> int:
    """How long a fixed-time plan holds each phase, given the swept floor.

    Two constraints. The signal may not act until `min_green + yellow_time` has
    elapsed, so a shorter phase would be silently refused. And the controller can
    only ask at decision steps, so the duration has to land on the `delta_time`
    grid or the request arrives at a moment the signal cannot honour it. Round
    the floor up to satisfy both.

    Green is what is left after yellow, and stays at or above the floor.
    """
    need = min_green + yellow_time
    return int(-(-need // delta_time) * delta_time)      # ceil to the grid


def fixed_time_phase(t: float, offset: float, num_phases: int,
                     phase_seconds: int) -> int:
    """The phase a fixed-time plan holds at time `t`, shifted by `offset`.

    A fixed-time plan is a function of TIME, not of how many decisions have been
    taken: phase 0 for phase_seconds, then phase 1, and so on, repeating every
    num_phases * phase_seconds. `offset` shifts the whole cycle later, which is
    what makes a green wave a wave -- a signal offset by the travel time from its
    upstream neighbour shows an arriving platoon the phase that neighbour showed
    when it released them.

    The predecessor asked for `(step - offset_steps) % num_phases`, alternating
    its request every decision step and leaving the actual switching times to
    min_green blocking. That is not a plan: the green duration was whatever the
    blocking produced, distinct floors collapsed onto identical cycles, and the
    offsets were re-timed out of existence. See tests/test_green_wave_plan.py.
    """
    cycle = num_phases * phase_seconds
    return int(((t - offset) % cycle) // phase_seconds)


def movement_pressure(incoming_queue: float, outgoing_queue: float) -> float:
    """Max-pressure movement pressure = upstream minus downstream queue."""
    return incoming_queue - outgoing_queue


def phase_pressure(movements, queue_of) -> float:
    """Total pressure of a phase = sum of its movements' pressures.

    `movements` is an iterable of (incoming_lane, outgoing_lane) pairs the phase
    gives green to; `queue_of(lane)` returns that lane's queue.

    The downstream term is what makes this max-pressure rather than
    "serve the longest queue". Dropping it (treating every outgoing queue as 0)
    turns the controller into a purely local rule that will happily discharge
    into a lane that is already full, which on a corridor is exactly the
    spillback case coordination is supposed to handle. Movements are
    de-duplicated by (in, out) pair, so a lane feeding several phases is not
    counted twice within one phase.
    """
    return sum(queue_of(i) - queue_of(o) for i, o in set(movements))


def max_pressure_phase(phase_pressures: Dict[int, float]) -> int:
    """Pick the phase with the greatest total pressure; ties -> lowest phase id."""
    return max(sorted(phase_pressures), key=lambda p: phase_pressures[p])
