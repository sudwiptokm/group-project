"""Unit test for the single-agent rollout adapter validate_ppo_core.py adds --
the piece that lets ppo_core (built for the multi-agent corridor) run on the
single-agent intersection env's (obs, reward, terminated, truncated) API."""
import pytest

vpc = pytest.importorskip("analysis.validate_ppo_core")


def test_matched_hyperparameters_come_from_algos_ppo_defaults():
    # must be byte-for-byte what SB3 PPO trains with on this task, or the
    # comparison is not apples-to-apples
    from algos import ALGOS
    sb3_defaults = ALGOS["ppo"]["defaults"]()
    hp = vpc.matched_hp()
    assert hp["lr"] == sb3_defaults["learning_rate"]
    assert hp["n_steps"] == sb3_defaults["n_steps"]
    assert hp["batch_size"] == sb3_defaults["batch_size"]
    assert hp["n_epochs"] == sb3_defaults["n_epochs"]
    assert hp["gamma"] == sb3_defaults["gamma"]
    assert hp["gae_lambda"] == sb3_defaults["gae_lambda"]
    assert hp["clip_range"] == sb3_defaults["clip_range"]
    assert hp["ent_coef"] == sb3_defaults["ent_coef"]
    assert hp["hidden"] == tuple(sb3_defaults["policy_kwargs"]["net_arch"])
