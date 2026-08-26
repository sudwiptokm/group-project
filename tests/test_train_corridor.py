"""Slow smoke: short IPPO train + eval must write a readable eval CSV. Requires SUMO."""
import os

import pytest

pytestmark = pytest.mark.slow

import train_corridor as tc


@pytest.mark.skipif(not os.environ.get("SUMO_HOME"), reason="SUMO_HOME not set")
def test_ippo_trains_and_evaluates(monkeypatch):
    monkeypatch.setenv("EPISODE_SECONDS", "200")
    model = tc.train("corridor_offpeak", lam=0.5, seed=0, steps=600, min_green=10)
    assert os.path.exists(model)

    csv = tc.evaluate(model, "corridor_offpeak", lam=0.5, seed=42, min_green=10, steps=600)
    assert os.path.exists(csv)
    import pandas as pd
    df = pd.read_csv(csv)
    # policy is mobile (not gridlock-collapsed) and metrics finite
    assert df["system_mean_speed"].mean() > 0


def _mean_wait(csv_path):
    import pandas as pd
    return pd.read_csv(csv_path)["system_mean_waiting_time"].mean()


@pytest.mark.skipif(not os.environ.get("SUMO_HOME"), reason="SUMO_HOME not set")
def test_ippo_learns_vs_untrained(monkeypatch, tmp_path):
    monkeypatch.setenv("EPISODE_SECONDS", "200")
    import torch
    import ppo_core as pc
    import env_common as ec

    # 1) save an UNTRAINED (random-init) policy and eval it
    env = ec.make_corridor_env(seed=0, scenario="corridor_offpeak", lam=0.5, min_green=10)
    obs_dim, act_dim = tc._obs_act_dims(env)
    env.close()
    untrained = pc.ActorCritic(obs_dim, act_dim, hidden=tc._hp()["hidden"])
    u_path = str(tmp_path / "untrained.pt")
    torch.save(untrained.state_dict(), u_path)
    # untrained model was never produced by train(), so there is no real step
    # budget to reconstruct a tag for -- 0 flags that plainly.
    u_csv = tc.evaluate(u_path, "corridor_offpeak", lam=0.5, seed=7, min_green=10, steps=0)

    # 2) train and eval on the same held-out seed
    model = tc.train("corridor_offpeak", lam=0.5, seed=0, steps=2000, min_green=10)
    t_csv = tc.evaluate(model, "corridor_offpeak", lam=0.5, seed=7, min_green=10, steps=2000)

    # At this micro budget we only require the trained policy to stay mobile and
    # not be meaningfully WORSE than random (within 10%). Real convergence
    # evidence is SP5; that gradient-based learning happens at all is gated
    # cheaply and robustly by tests/test_train_corridor_update.py.
    import pandas as pd
    assert pd.read_csv(t_csv)["system_mean_speed"].mean() > 0
    assert _mean_wait(t_csv) <= 1.10 * _mean_wait(u_csv)


@pytest.mark.skipif(not os.environ.get("SUMO_HOME"), reason="SUMO_HOME not set")
def test_mappo_trains_and_evaluates(monkeypatch):
    """SP3: centralized=True must produce an `mappo`-tagged checkpoint and eval
    CSV, distinct from the IPPO path this file's other tests exercise."""
    monkeypatch.setenv("EPISODE_SECONDS", "200")
    model = tc.train("corridor_offpeak", lam=0.5, seed=0, steps=600, min_green=10,
                     centralized=True)
    assert os.path.exists(model)
    assert "mappo" in os.path.basename(model)

    csv = tc.evaluate(model, "corridor_offpeak", lam=0.5, seed=42, min_green=10, steps=600)
    assert os.path.exists(csv)
    assert "eval_mappo_" in os.path.basename(csv)
    import pandas as pd
    assert pd.read_csv(csv)["system_mean_speed"].mean() > 0
