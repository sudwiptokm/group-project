"""green_wave must be an actual fixed-time coordinated plan.

The first implementation asked for `(step - offset_steps) % num_phases` at every
decision step -- it alternated its request every step and let sumo-rl's min_green
blocking decide when a change was allowed. Two consequences, both measured:

  * sumo_rl.TrafficSignal.set_next_phase treats a request naming the CURRENT
    phase as a no-op, so a blocked request was always followed by a same-phase
    request and acceptance could only land on every SECOND decision step. The
    effective cycle came out 15, 15, 25, 25, 35, 35 s for min_green 5, 10, 15,
    20, 25, 30 -- distinct floors collapsing onto identical plans.
  * the green duration was never set by anything; it fell out of the blocking.
    So the swept "floor" was not the plan's green, and the progression offsets
    were applied as integer step shifts to a pattern that the blocking then
    re-timed, which destroys the progression a green wave exists to create.

A fixed-time plan is a function of TIME: each signal holds phase p for
phase_seconds, cycling, shifted by its offset. That is what these tests pin.
"""
import pytest

import corridor_control as cc


def test_each_phase_is_held_for_its_full_duration():
    p = [cc.fixed_time_phase(t, offset=0.0, num_phases=2, phase_seconds=20)
         for t in range(0, 40, 5)]
    assert p == [0, 0, 0, 0, 1, 1, 1, 1]


def test_the_plan_cycles():
    cycle = 2 * 20
    for t in (0, 7, 19, 20, 33):
        assert cc.fixed_time_phase(t, 0.0, 2, 20) == \
               cc.fixed_time_phase(t + cycle, 0.0, 2, 20)


def test_offset_delays_a_downstream_signal_by_the_offset():
    """A signal offset by o shows at t+o what the upstream signal shows at t.

    This is the progression itself: a platoon leaving the upstream signal at t
    arrives downstream at t+o and must find the same phase green.
    """
    for t in range(0, 80, 5):
        assert cc.fixed_time_phase(t + 15.0, offset=15.0, num_phases=2,
                                   phase_seconds=20) == \
               cc.fixed_time_phase(t, offset=0.0, num_phases=2, phase_seconds=20)


def test_negative_time_minus_offset_still_cycles():
    # t < offset happens for every downstream signal at the start of an episode
    assert cc.fixed_time_phase(0, offset=15.0, num_phases=2, phase_seconds=20) in (0, 1)
    assert cc.fixed_time_phase(0, offset=15.0, num_phases=2, phase_seconds=20) == \
           cc.fixed_time_phase(40, offset=15.0, num_phases=2, phase_seconds=20)


def test_phase_seconds_respects_the_floor_and_the_decision_grid():
    """Per-phase duration must clear min_green + yellow AND land on a decision
    step, or the request arrives at a moment the signal is not allowed to act."""
    assert cc.plan_phase_seconds(min_green=5, yellow_time=3, delta_time=5) == 10
    assert cc.plan_phase_seconds(min_green=10, yellow_time=3, delta_time=5) == 15
    assert cc.plan_phase_seconds(min_green=15, yellow_time=3, delta_time=5) == 20
    assert cc.plan_phase_seconds(min_green=60, yellow_time=3, delta_time=5) == 65


def test_every_swept_floor_maps_to_a_distinct_plan():
    """The aliasing regression: mg5 and mg10 were the same 15 s cycle, as were
    mg15/mg20 and mg25/mg30, so the floor curve had fewer points than labels."""
    floors = [5, 10, 15, 20, 25, 30, 45, 60, 75, 90]
    seconds = [cc.plan_phase_seconds(mg, 3, 5) for mg in floors]
    assert len(set(seconds)) == len(floors), dict(zip(floors, seconds))


def test_green_is_what_is_left_after_yellow():
    # the plan holds the phase for phase_seconds, of which yellow eats the first
    # yellow_time; green never drops below the floor being swept
    for mg in (5, 10, 15, 20, 25, 30, 45, 60, 75, 90):
        p = cc.plan_phase_seconds(mg, yellow_time=3, delta_time=5)
        assert p - 3 >= mg, f"mg{mg}: green {p - 3}s is below the floor"
