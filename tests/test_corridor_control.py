"""Pure control math for the corridor baselines — no SUMO needed."""
import corridor_control as cc


def test_green_wave_offsets():
    # 3 signals 200m apart, free-flow 13.89 m/s -> offset ≈ 200/13.89 = 14.4 s
    offsets = cc.green_wave_offsets(
        positions=[0.0, 200.0, 400.0], free_flow_speed=13.89)
    assert offsets[0] == 0.0
    assert abs(offsets[1] - 14.4) < 0.1
    assert abs(offsets[2] - 28.8) < 0.1


def test_max_pressure_picks_highest_pressure_phase():
    # phase 0 serves movement with pressure 5, phase 1 pressure 2 -> pick 0
    phase_pressures = {0: 5.0, 1: 2.0}
    assert cc.max_pressure_phase(phase_pressures) == 0
    # tie broken by lowest phase id (deterministic)
    assert cc.max_pressure_phase({0: 3.0, 1: 3.0}) == 0


def test_pressure_is_upstream_minus_downstream_queue():
    # pressure of a movement = incoming queue - outgoing queue
    assert cc.movement_pressure(incoming_queue=10, outgoing_queue=3) == 7
    assert cc.movement_pressure(incoming_queue=2, outgoing_queue=8) == -6
