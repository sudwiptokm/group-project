"""Unit tests for train_corridor.py's pure-logic pieces (no SUMO)."""
import train_corridor as tc


def test_hp_has_no_file_dependency(tmp_path, monkeypatch):
    # cwd has no cloud_params/ dir at all -- _hp() must not need one
    monkeypatch.chdir(tmp_path)
    hp = tc._hp()
    assert hp["lr"] == 2.3195e-05
    assert hp["n_steps"] == 128
    assert hp["batch_size"] == 32
    assert hp["n_epochs"] == 10
    assert hp["gamma"] == 0.95
    assert hp["gae_lambda"] == 0.9525
    assert hp["clip_range"] == 0.1
    assert hp["ent_coef"] == 0.0081
    assert hp["hidden"] == (256, 256)
