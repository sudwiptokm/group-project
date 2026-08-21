"""Slow smoke: short IDQN train + eval must write a readable eval CSV. Requires SUMO."""
import os

import pytest

pytestmark = pytest.mark.slow

import train_corridor_dqn as tcd


@pytest.mark.skipif(not os.environ.get("SUMO_HOME"), reason="SUMO_HOME not set")
def test_idqn_trains_and_evaluates(monkeypatch):
    monkeypatch.setenv("EPISODE_SECONDS", "200")
    paths = tcd.train("corridor_offpeak", lam=0.5, seed=0, steps=600, min_green=10)
    assert set(paths.keys()) == set(tcd.CORRIDOR_TS_IDS)
    for p in paths.values():
        assert os.path.exists(p)

    # evaluate() has no model_path param (unlike train_corridor.evaluate) -- it
    # reconstructs each checkpoint's path via _model_path(..., seed, ...), so
    # seed must match the seed train() was called with (same convention
    # analysis/idqn_sweep.run_one uses: one seed for both calls).
    csv = tcd.evaluate("corridor_offpeak", lam=0.5, seed=0, min_green=10, steps=600)
    assert os.path.exists(csv)
    import pandas as pd
    df = pd.read_csv(csv)
    assert df["system_mean_speed"].mean() > 0
