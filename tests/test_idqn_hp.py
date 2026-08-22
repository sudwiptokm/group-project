"""Unit tests for train_corridor_dqn.py's pure-logic pieces (no SUMO)."""
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
