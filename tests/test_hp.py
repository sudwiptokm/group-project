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


def test_tag_includes_min_green():
    assert tc._tag("corridor_peak", 0.5, seed=0, min_green=10) == \
        "corridor_peak_lam05_seed0_mg10"


def test_train_and_evaluate_require_min_green_kwarg():
    import inspect
    assert "min_green" in inspect.signature(tc.train).parameters
    assert "min_green" in inspect.signature(tc.evaluate).parameters
    # no default -- caller must always pass it
    assert inspect.signature(tc.train).parameters["min_green"].default is \
        inspect.Parameter.empty
    assert inspect.signature(tc.evaluate).parameters["min_green"].default is \
        inspect.Parameter.empty
