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


class TestPhasePressure:
    """Sums a phase's movements, each upstream minus downstream.

    The downstream term is the difference between max-pressure and "serve the
    longest queue". It is tested explicitly because the first corridor baseline
    passed outgoing_queue=0.0 everywhere, which silently made the reference a
    local rule -- and an RL controller that beat THAT would have been beating a
    weakened baseline, the defect this project already withdrew a headline over.
    """

    @staticmethod
    def _queues(mapping):
        return lambda lane: mapping[lane]

    def test_sums_movement_pressures(self):
        q = self._queues({"in_a": 10, "out_a": 3, "in_b": 5, "out_b": 1})

        assert cc.phase_pressure([("in_a", "out_a"), ("in_b", "out_b")], q) == 11

    def test_a_blocked_downstream_lane_makes_pressure_negative(self):
        """The case the old version could not see: a long incoming queue whose
        receiving lane is already full is NOT worth serving."""
        q = self._queues({"in": 8, "out": 20})

        assert cc.phase_pressure([("in", "out")], q) == -12

    def test_a_full_downstream_lane_loses_to_a_clear_one(self):
        """Two phases with identical incoming queues must be separated by their
        downstream state; ignoring it would make these tie."""
        q = self._queues({"in_1": 10, "out_1": 9, "in_2": 10, "out_2": 0})

        blocked = cc.phase_pressure([("in_1", "out_1")], q)
        clear = cc.phase_pressure([("in_2", "out_2")], q)

        assert clear > blocked
        assert cc.max_pressure_phase({0: blocked, 1: clear}) == 1

    def test_duplicate_movements_are_not_counted_twice(self):
        """getControlledLinks can list the same (in, out) pair more than once;
        counting it twice would inflate that phase's pressure."""
        q = self._queues({"in": 6, "out": 1})

        once = cc.phase_pressure([("in", "out")], q)
        twice = cc.phase_pressure([("in", "out"), ("in", "out")], q)

        assert once == twice == 5

    def test_no_movements_is_zero_pressure(self):
        assert cc.phase_pressure([], self._queues({})) == 0
