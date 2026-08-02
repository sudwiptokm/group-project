"""Slow smoke: short IPPO train + eval must write a readable eval CSV. Requires SUMO."""
import os

import pytest

pytestmark = pytest.mark.slow

import train_corridor as tc


@pytest.mark.skipif(not os.environ.get("SUMO_HOME"), reason="SUMO_HOME not set")
def test_ippo_trains_and_evaluates(monkeypatch):
    monkeypatch.setenv("EPISODE_SECONDS", "200")
    model = tc.train("corridor_offpeak", lam=0.5, seed=0, steps=600)
    assert os.path.exists(model)

    csv = tc.evaluate(model, "corridor_offpeak", lam=0.5, seed=42)
    assert os.path.exists(csv)
    import pandas as pd
    df = pd.read_csv(csv)
    # policy is mobile (not gridlock-collapsed) and metrics finite
    assert df["system_mean_speed"].mean() > 0
