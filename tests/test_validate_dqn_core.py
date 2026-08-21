"""Unit test for validate_dqn_core.py's matched-hyperparameter contract --
the same "must be byte-for-byte SB3's defaults" guard validate_ppo_core.py
has for PPO."""
import pytest

vdc = pytest.importorskip("analysis.validate_dqn_core")


def test_matched_hyperparameters_come_from_algos_dqn_defaults():
    from algos import ALGOS
    sb3_defaults = ALGOS["dqn"]["defaults"]()
    hp = vdc.matched_hp()
    assert hp["lr"] == sb3_defaults["learning_rate"]
    assert hp["buffer_size"] == sb3_defaults["buffer_size"]
    assert hp["learning_starts"] == sb3_defaults["learning_starts"]
    assert hp["batch_size"] == sb3_defaults["batch_size"]
    assert hp["gamma"] == sb3_defaults["gamma"]
    assert hp["train_freq"] == sb3_defaults["train_freq"]
    assert hp["target_update_interval"] == sb3_defaults["target_update_interval"]
    assert hp["exploration_fraction"] == sb3_defaults["exploration_fraction"]
    assert hp["exploration_final_eps"] == sb3_defaults["exploration_final_eps"]
    assert hp["hidden"] == tuple(sb3_defaults["policy_kwargs"]["net_arch"])
