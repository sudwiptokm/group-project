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


def max_pressure_phase(phase_pressures: Dict[int, float]) -> int:
    """Pick the phase with the greatest total pressure; ties -> lowest phase id."""
    return max(sorted(phase_pressures), key=lambda p: phase_pressures[p])
