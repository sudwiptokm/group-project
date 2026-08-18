"""_get_safety_info must keep the network aggregate AND add a per-TLS total.

Rewritten when the corridor branch was integrated with main. The original test
stubbed `_safety_components` and asserted that `_get_safety_info` called it once
per signal at the action step. That is precisely the sampling defect main fixed:
at an action step `is_yellow` is always back down (sumo-rl asserts
delta_time > yellow_time), so exposure can never fire, and braking is only
sampled in the final second of the window. The per-agent breakdown must
therefore be read from the per-second accumulator, not recomputed -- so these
tests drive the window and assert the columns come out of it.
"""
import pytest

import env_common as ec


class _FakeTS:
    """Stands in for a TrafficSignal: only its id is used by the window."""

    def __init__(self, ts_id):
        self.id = ts_id


@pytest.fixture
def env_with_window(monkeypatch):
    """A SafetyLoggingEnv with a live window and two signals, no SUMO."""
    signals = {"C1": _FakeTS("C1"), "C2": _FakeTS("C2")}
    env = ec.SafetyLoggingEnv.__new__(ec.SafetyLoggingEnv)
    env.traffic_signals = signals
    env._safety_window = ec._SafetyWindow()
    # the window caches internal lanes per signal; keep SUMO out of it
    monkeypatch.setattr(ec, "_internal_lanes", lambda ts: [])
    return env


def _accumulate(env, monkeypatch, per_second):
    """Feed the window `per_second` samples, as _sumo_step would each second."""
    for sample in per_second:
        monkeypatch.setattr(ec, "_safety_components",
                            lambda ts, internal_lanes=None, _s=sample: _s[ts.id])
        for ts in env.traffic_signals.values():
            env._safety_window.accumulate(ts)


def test_per_agent_and_aggregate_come_from_the_window(env_with_window, monkeypatch):
    _accumulate(env_with_window, monkeypatch,
                [{"C1": (1.0, 0.0), "C2": (0.0, 2.0)}])

    info = env_with_window._get_safety_info()

    assert info["system_safety_brake"] == 1.0
    assert info["system_safety_exposure"] == 2.0
    assert info["system_safety_total"] == 3.0
    assert info["safety_total_C1"] == 1.0
    assert info["safety_total_C2"] == 2.0


def test_per_agent_totals_accumulate_over_the_whole_window(env_with_window, monkeypatch):
    """The regression the window exists for: three simulation seconds of braking
    must all count, not just the last one. Reading the instantaneous sample at
    the action step would report 1.0 here instead of 3.0."""
    _accumulate(env_with_window, monkeypatch, [
        {"C1": (1.0, 0.0), "C2": (0.0, 0.0)},
        {"C1": (1.0, 0.0), "C2": (0.0, 1.0)},
        {"C1": (1.0, 0.0), "C2": (0.0, 1.0)},
    ])

    info = env_with_window._get_safety_info()

    assert info["safety_total_C1"] == 3.0
    assert info["safety_total_C2"] == 2.0
    assert info["system_safety_total"] == 5.0


def test_a_signal_with_no_samples_still_reports_zero(env_with_window, monkeypatch):
    """compare.py reads these columns positionally across runs; a missing key
    for a quiet signal would shift every downstream column."""
    _accumulate(env_with_window, monkeypatch,
                [{"C1": (2.0, 0.0), "C2": (0.0, 0.0)}])

    info = env_with_window._get_safety_info()

    assert info["safety_total_C2"] == 0.0


def test_window_reset_starts_the_next_decision_window_clean(env_with_window, monkeypatch):
    _accumulate(env_with_window, monkeypatch,
                [{"C1": (5.0, 0.0), "C2": (0.0, 5.0)}])
    env_with_window._safety_window.reset()
    _accumulate(env_with_window, monkeypatch,
                [{"C1": (1.0, 0.0), "C2": (0.0, 0.0)}])

    info = env_with_window._get_safety_info()

    assert info["safety_total_C1"] == 1.0
    assert info["safety_total_C2"] == 0.0
