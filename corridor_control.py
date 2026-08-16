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
