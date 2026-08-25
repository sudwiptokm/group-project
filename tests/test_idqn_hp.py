"""Unit tests for train_corridor_dqn.py's pure-logic pieces (no SUMO)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

import dqn_core as dc
import train_corridor_dqn as tcd
from algos import ALGOS


def test_corridor_ts_ids_constant():
    assert tcd.CORRIDOR_TS_IDS == ("C1", "C2", "C3")


def test_tag_includes_min_green_and_steps():
    assert tcd._tag("corridor_peak", 0.5, seed=0, min_green=10, steps=100000) == \
        "corridor_peak_lam05_seed0_mg10_s100000"


def test_model_path_includes_agent_id():
    p = tcd._model_path("C1", "corridor_peak", 0.5, 0, 10, 100000)
    assert p == "models/idqn_C1_corridor_peak_lam05_seed0_mg10_s100000.pt"


def test_model_path_agents_do_not_collide():
    paths = {a: tcd._model_path(a, "corridor_peak", 0.5, 0, 10, 100000)
             for a in tcd.CORRIDOR_TS_IDS}
    assert len(set(paths.values())) == 3


def test_hp_uses_disclosed_values_and_algos_defaults():
    d = ALGOS["dqn"]["defaults"]()
    hp = tcd._hp()
    assert hp["lr"] == 2.3195e-05
    assert hp["learning_starts"] == 5000
    assert hp["target_update_interval"] == 5000
    assert hp["buffer_size"] == d["buffer_size"]
    assert hp["batch_size"] == d["batch_size"]
    assert hp["gamma"] == d["gamma"]
    assert hp["train_freq"] == d["train_freq"]
    assert hp["exploration_fraction"] == d["exploration_fraction"]
    assert hp["exploration_final_eps"] == d["exploration_final_eps"]
    assert hp["hidden"] == tuple(d["policy_kwargs"]["net_arch"])


def test_train_and_evaluate_require_min_green_kwarg():
    import inspect
    assert "min_green" in inspect.signature(tcd.train).parameters
    assert "min_green" in inspect.signature(tcd.evaluate).parameters
    assert inspect.signature(tcd.train).parameters["min_green"].default is \
        inspect.Parameter.empty
    assert inspect.signature(tcd.evaluate).parameters["min_green"].default is \
        inspect.Parameter.empty


def test_eval_out_stem_in_distribution_unchanged():
    assert tcd._eval_out_stem("corridor_peak", "corridor_peak", 0.5, 42, 10, 100000) == \
        "logs/eval_idqn_corridor_peak_lam05_seed42_mg10_s100000"


def test_eval_out_stem_zero_shot_appends_on_scenario():
    assert tcd._eval_out_stem("corridor_peak", "corridor_offpeak", 0.5, 42, 10, 100000) == \
        "logs/eval_idqn_corridor_peak_lam05_seed42_mg10_s100000_on_corridor_offpeak"


def test_evaluate_eval_scenario_defaults_to_none():
    import inspect
    sig = inspect.signature(tcd.evaluate)
    assert "eval_scenario" in sig.parameters
    assert sig.parameters["eval_scenario"].default is None


def test_eval_out_stem_incident_appends_suffix():
    assert tcd._eval_out_stem("corridor_peak", "corridor_peak", 0.5, 42, 10, 100000,
                              incident=True) == \
        "logs/eval_idqn_corridor_peak_lam05_seed42_mg10_s100000_incident"


def test_eval_out_stem_incident_and_zero_shot_combine():
    assert tcd._eval_out_stem("corridor_peak", "corridor_offpeak", 0.5, 42, 10, 100000,
                              incident=True) == \
        "logs/eval_idqn_corridor_peak_lam05_seed42_mg10_s100000_on_corridor_offpeak_incident"


def test_evaluate_incident_defaults_to_false():
    import inspect
    sig = inspect.signature(tcd.evaluate)
    assert "incident" in sig.parameters
    assert sig.parameters["incident"].default is False


def test_tag_variant_appends_fragment():
    assert tcd._tag("corridor_peak", 0.5, seed=42, min_green=10, steps=100000,
                    variant="incaware") == \
        "corridor_peak_lam05_seed42_mg10_s100000_incaware"


def test_tag_variant_defaults_to_empty_and_unchanged():
    # same string as test_tag_includes_min_green_and_steps -- pins that
    # omitting variant reproduces the pre-SP12 tag exactly, not just "close".
    assert tcd._tag("corridor_peak", 0.5, seed=0, min_green=10, steps=100000) == \
        "corridor_peak_lam05_seed0_mg10_s100000"


def test_model_path_variant_avoids_collision_with_plain_checkpoint():
    plain = tcd._model_path("C1", "corridor_peak", 0.5, 42, 10, 100000)
    incaware = tcd._model_path("C1", "corridor_peak", 0.5, 42, 10, 100000,
                               variant="incaware")
    assert plain != incaware
    assert plain == "models/idqn_C1_corridor_peak_lam05_seed42_mg10_s100000.pt"
    assert incaware == \
        "models/idqn_C1_corridor_peak_lam05_seed42_mg10_s100000_incaware.pt"


def test_eval_out_stem_variant_appends_before_on_and_incident_fragments():
    stem = tcd._eval_out_stem("corridor_peak", "corridor_peak", 0.5, 42, 10, 100000,
                              incident=True, variant="incaware")
    assert stem == \
        "logs/eval_idqn_corridor_peak_lam05_seed42_mg10_s100000_incaware_incident"


def test_train_incident_prob_and_variant_default_to_pre_sp12_behaviour():
    import inspect
    sig = inspect.signature(tcd.train)
    assert sig.parameters["incident"].default is None
    assert sig.parameters["incident_prob"].default == 0.0
    assert sig.parameters["variant"].default == ""


def test_evaluate_variant_defaults_to_empty():
    import inspect
    sig = inspect.signature(tcd.evaluate)
    assert sig.parameters["variant"].default == ""


def test_set_incident_updates_config_and_resets_applied_flag():
    """Pure-logic check of SafetyLoggingEnv.set_incident (SP12) against a
    bare stand-in object, no SUMO/real env construction needed -- the method
    body only touches self._incident/_incident_applied."""
    import env_common as ec

    class _Dummy:
        pass

    d = _Dummy()
    d._incident = ("C1_C2", 0, 1800.0, 900.0)
    d._incident_applied = True

    ec.SafetyLoggingEnv.set_incident(d, None)
    assert d._incident is None
    assert d._incident_applied is False

    ec.SafetyLoggingEnv.set_incident(d, ("C1_C2", 0, 1800.0, 900.0))
    assert d._incident == ("C1_C2", 0, 1800.0, 900.0)
    assert d._incident_applied is False


def test_evaluate_routes_eval_scenario_to_env_and_checkpoint_scenario_to_load(monkeypatch):
    """Pins that eval_scenario actually reaches make_corridor_env (the env the
    episode runs in), while the checkpoint scenario -- NOT eval_scenario --
    still governs which checkpoint file is loaded. The existing zero-shot
    smoke test (tests/test_train_corridor_dqn.py) only checks the output
    filename; it would pass even if eval_scenario were threaded into the
    filename but not the env, silently still running corridor_peak demand
    under a corridor_offpeak-looking name. No SUMO required: make_corridor_env
    and torch.load are both mocked, and the mocked episode ends after exactly
    one env.step() call."""
    obs_dim, act_dim, hidden = 3, 2, (4,)
    ts_ids = list(tcd.CORRIDOR_TS_IDS)

    def _obs():
        return {i: np.zeros(obs_dim, dtype=np.float32) for i in ts_ids}

    fake_env = SimpleNamespace(
        ts_ids=ts_ids,
        out_csv_name="fake_stem",
        episode=1,
        label=0,
        observation_spaces=lambda tid: SimpleNamespace(shape=(obs_dim,)),
        action_spaces=lambda tid: SimpleNamespace(n=act_dim),
        reset=lambda: _obs(),
        # dones["__all__"]=True immediately -- the evaluate() loop exits after
        # this one env.step() call, so the RL loop itself is never exercised.
        step=lambda actions: (_obs(), {i: 0.0 for i in ts_ids}, {"__all__": True}, {}),
        save_csv=lambda name, episode: None,
        close=lambda: None,
    )
    make_env_mock = MagicMock(return_value=fake_env)
    monkeypatch.setattr(tcd, "make_corridor_env", make_env_mock)

    # a real, small QNetwork's state_dict so q_net.load_state_dict succeeds
    real_net = dc.QNetwork(obs_dim, act_dim, hidden=hidden)
    ckpt = {"state_dict": real_net.state_dict(), "hidden": hidden}
    torch_load_mock = MagicMock(return_value=ckpt)
    monkeypatch.setattr(tcd.torch, "load", torch_load_mock)

    tcd.evaluate("corridor_peak", lam=0.5, seed=42, min_green=10, steps=100000,
                 eval_scenario="corridor_offpeak")

    # the episode must run on the EVAL scenario, not the checkpoint's training
    # scenario
    assert make_env_mock.call_args.kwargs["scenario"] == "corridor_offpeak"

    # but each checkpoint must still be loaded from its TRAINING scenario
    assert torch_load_mock.call_count == len(ts_ids)
    for call in torch_load_mock.call_args_list:
        ckpt_path = call.args[0]
        assert "corridor_peak" in ckpt_path
        assert "corridor_offpeak" not in ckpt_path
