"""_get_safety_info must keep the network aggregate AND add a per-TLS total."""
import pytest

import env_common as ec


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_per_agent_safety_columns(monkeypatch):
    signals = {"C1": object(), "C2": object()}
    fake = {id(signals["C1"]): (1.0, 0.0), id(signals["C2"]): (0.0, 2.0)}
    monkeypatch.setattr(ec, "_safety_components", lambda ts: fake[id(ts)])

    env = ec.SafetyLoggingEnv.__new__(ec.SafetyLoggingEnv)
    env.traffic_signals = signals

    info = env._get_safety_info()
    # aggregate unchanged
    assert info["system_safety_brake"] == 1.0
    assert info["system_safety_exposure"] == 2.0
    assert info["system_safety_total"] == 3.0
    # per-agent totals
    assert info["safety_total_C1"] == 1.0
    assert info["safety_total_C2"] == 2.0
