"""Slow smoke: short IDQN train + eval must write a readable eval CSV. Requires SUMO."""
import os

import pytest

pytestmark = pytest.mark.slow

import train_corridor_dqn as tcd


@pytest.mark.skipif(not os.environ.get("SUMO_HOME"), reason="SUMO_HOME not set")
def test_idqn_trains_and_evaluates(monkeypatch):
    monkeypatch.setenv("EPISODE_SECONDS", "200")
    # steps must clear _DISCLOSED["learning_starts"] (5000) and
    # _DISCLOSED["target_update_interval"] (5000) so train()'s own
    # update-triggering logic (dc.dqn_loss + optimizer.step() + target sync)
    # is actually exercised, not just the checkpoint-saving path -- at a
    # shorter budget this test would pass identically even if that wiring
    # were broken, since t >= learning_starts and t % target_update_interval
    # == 0 would never be true.
    steps = 5200
    seed = 0
    paths = tcd.train("corridor_offpeak", lam=0.5, seed=seed, steps=steps, min_green=10)
    assert set(paths.keys()) == set(tcd.CORRIDOR_TS_IDS)
    for p in paths.values():
        assert os.path.exists(p)

    # Prove train()'s own loop actually reached and executed dc.dqn_loss +
    # optimizer.step() for at least one agent -- tests/test_dqn_core.py's
    # test_dqn_loss_gradient_updates_qnetwork_parameters already proves
    # dqn_loss + an optimizer step move a QNetwork's weights when called
    # directly; this proves train_corridor_dqn.train() actually reaches that
    # code path, not just that dqn_core works in isolation.
    #
    # tcd.CORRIDOR_TS_IDS[0] is the first id train() builds a QNetwork for
    # (immediately after its torch.manual_seed(seed) call), so that agent's
    # q_net is the very first QNetwork train() constructs. Assert the
    # assumption explicitly rather than hardcoding "C1", in case
    # CORRIDOR_TS_IDS's order ever changes.
    assert tcd.CORRIDOR_TS_IDS[0] == "C1"
    first_id = tcd.CORRIDOR_TS_IDS[0]

    # Nothing between manual_seed(seed) and that QNetwork() call touches
    # torch's global RNG (make_corridor_env doesn't import torch;
    # np.random.default_rng(seed) is an independent numpy Generator, not
    # torch's RNG). So a freshly-seeded QNetwork built the same way has
    # IDENTICAL initial weights to first_id's pre-training q_net -- any
    # difference after loading the checkpoint proves a real gradient update
    # happened.
    import torch
    import dqn_core as dc
    import env_common as ec

    probe_env = ec.make_corridor_env(seed=seed, scenario="corridor_offpeak", lam=0.5,
                                     min_green=10)
    obs_dim, act_dim = tcd._obs_act_dims(probe_env)
    probe_env.close()

    torch.manual_seed(seed)
    fresh = dc.QNetwork(obs_dim, act_dim, hidden=tcd._hp()["hidden"])

    ckpt = torch.load(paths[first_id], weights_only=True)
    trained = dc.QNetwork(obs_dim, act_dim, hidden=tuple(ckpt["hidden"]))
    trained.load_state_dict(ckpt["state_dict"])

    changed = any(
        not torch.equal(a, b)
        for a, b in zip(fresh.state_dict().values(), trained.state_dict().values())
    )
    assert changed, (f"{first_id}'s Q-network weights are unchanged from init -- "
                     "train() never reached dqn_loss/optimizer.step()")

    # evaluate() has no model_path param (unlike train_corridor.evaluate) -- it
    # reconstructs each checkpoint's path via _model_path(..., seed, ...), so
    # seed must match the seed train() was called with (same convention
    # analysis/idqn_sweep.run_one uses: one seed for both calls).
    csv = tcd.evaluate("corridor_offpeak", lam=0.5, seed=seed, min_green=10, steps=steps)
    assert os.path.exists(csv)
    import pandas as pd
    df = pd.read_csv(csv)
    assert df["system_mean_speed"].mean() > 0


@pytest.mark.skipif(not os.environ.get("SUMO_HOME"), reason="SUMO_HOME not set")
def test_idqn_incident_aware_training_writes_distinct_checkpoints_and_evaluates(monkeypatch):
    """SP12 smoke test: train() with incident_prob>0 must (a) not crash, (b)
    actually exercise the incident code path at least once across a short
    run, and (c) save/load checkpoints under `variant` without colliding
    with a plain (incident_prob=0) run at the same scenario/lam/seed.

    EPISODE_SECONDS is shrunk to 60s so the test is fast; the incident tuple
    below is scaled down to fit inside that window (start=20s, duration=20s)
    instead of SP7's real (1800s, 900s) -- this test is only pinning that
    the training-time incident plumbing (env.set_incident + _sumo_step's
    existing apply/revert logic) runs correctly at some timing, not
    reproducing SP7's exact scenario (that's covered by
    test_incident_closes_and_reopens_lane and the real SP12 training run).
    """
    monkeypatch.setenv("EPISODE_SECONDS", "60")
    steps = 5200
    seed = 0
    scaled_incident = ("C1_C2", 0, 20.0, 20.0)

    paths = tcd.train("corridor_offpeak", lam=0.5, seed=seed, steps=steps,
                      min_green=10, incident=scaled_incident, incident_prob=1.0,
                      variant="incaware")
    for p in paths.values():
        assert os.path.exists(p)
        assert "incaware" in p

    # must not collide with the plain (no incident) checkpoint path for the
    # same scenario/lam/seed/min_green/steps
    plain_path = tcd._model_path(tcd.CORRIDOR_TS_IDS[0], "corridor_offpeak", 0.5,
                                 seed, 10, steps)
    assert plain_path not in paths.values()

    csv = tcd.evaluate("corridor_offpeak", lam=0.5, seed=seed, min_green=10,
                       steps=steps, variant="incaware")
    assert "incaware" in csv
    assert os.path.exists(csv)
    import pandas as pd
    df = pd.read_csv(csv)
    assert df["system_mean_speed"].mean() > 0


@pytest.mark.skipif(not os.environ.get("SUMO_HOME"), reason="SUMO_HOME not set")
def test_idqn_zero_shot_eval_runs_on_different_scenario(monkeypatch):
    monkeypatch.setenv("EPISODE_SECONDS", "200")
    tcd.train("corridor_peak", lam=0.5, seed=1, steps=600, min_green=10)
    csv = tcd.evaluate("corridor_peak", lam=0.5, seed=1, min_green=10, steps=600,
                       eval_scenario="corridor_offpeak")
    assert "_on_corridor_offpeak" in csv
    assert os.path.exists(csv)
    import pandas as pd
    df = pd.read_csv(csv)
    assert df["system_mean_speed"].mean() > 0
