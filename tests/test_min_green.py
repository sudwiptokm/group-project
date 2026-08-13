"""`min_green` is a knob on the ACTION SPACE, and it decided the peak result.

The actuated probe (analysis/actuated.py, docs/FINDINGS_2026-08-12.md) showed a
non-learning controller is 5.6x worse than a fixed plan at a 10 s floor and
matches it at 60 s. Every peak training run this project did used 10 s, because
that was make_env's default and no driver could override it. These tests pin
the two things that stops recurring: the default, and the fact that the floor
now appears in the run's filename so two floors cannot be averaged into one row.
"""
import glob
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_common import eval_csv_stem, resolve_min_green  # noqa: E402


class TestResolveMinGreen:
    """Same resolution pattern as TIME_TO_TELEPORT/EPISODE_SECONDS: explicit
    argument wins, else the env var, else the measured default."""

    def test_defaults_to_the_measured_sixty(self, monkeypatch):
        monkeypatch.delenv("MIN_GREEN", raising=False)
        assert resolve_min_green(None) == 60

    def test_env_var_overrides_the_default(self, monkeypatch):
        monkeypatch.setenv("MIN_GREEN", "45")
        assert resolve_min_green(None) == 45

    def test_explicit_argument_beats_the_env_var(self, monkeypatch):
        # static_timing.py/baseline.py pass 10 explicitly to reproduce the
        # published sweep; a stray MIN_GREEN in the shell must not rewrite it.
        monkeypatch.setenv("MIN_GREEN", "60")
        assert resolve_min_green(10) == 10

    def test_explicit_zero_is_honoured_not_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("MIN_GREEN", "60")
        assert resolve_min_green(0) == 0


class TestMakeEnvUsesTheResolvedFloor:
    """The trap this change exists to remove: make_env hard-defaulted to 10, no
    driver could override it, so every training run silently used the floor the
    actuated probe later showed is unwinnable."""

    @pytest.fixture
    def recorded(self, monkeypatch):
        """Capture make_env's kwargs without starting SUMO."""
        import env_common
        seen = {}

        def fake_env(**kwargs):
            seen.update(kwargs)
            return object()

        monkeypatch.setattr(env_common, "SafetyLoggingEnv", fake_env)
        monkeypatch.delenv("MIN_GREEN", raising=False)
        return seen

    def test_defaults_to_sixty(self, recorded, monkeypatch):
        import env_common
        env_common.make_env(seed=42, scenario="peak")
        assert recorded["min_green"] == 60

    def test_explicit_floor_is_passed_through(self, recorded):
        import env_common
        # static_timing.py sweeps greens below 60 and must not be clamped
        env_common.make_env(seed=42, scenario="peak", min_green=10)
        assert recorded["min_green"] == 10

    def test_env_var_reaches_the_environment(self, recorded, monkeypatch):
        import env_common
        monkeypatch.setenv("MIN_GREEN", "45")
        env_common.make_env(seed=42, scenario="peak")
        assert recorded["min_green"] == 45


class TestEvalCsvStem:
    def test_carries_the_min_green_so_two_floors_stay_distinguishable(self):
        stem = eval_csv_stem("dqn", "peak_lam05", seed=42, min_green=60)
        assert stem == "logs/eval_dqn_peak_lam05_seed42_mg60"

    def test_keeps_the_train_seed_tag(self):
        stem = eval_csv_stem("dqn", "peak_lam05", seed=42, train_seed=3,
                             min_green=60)
        assert stem == "logs/eval_dqn_peak_lam05_seed42_t3_mg60"

    def test_omits_the_tag_when_min_green_is_unspecified(self):
        # pre-existing runs have no mg fragment; naming must not invent one
        assert eval_csv_stem("dqn", "peak_lam05", seed=42) == \
            "logs/eval_dqn_peak_lam05_seed42"

    @pytest.mark.parametrize("train_seed", [None, 3])
    def test_still_matches_compare_globs(self, tmp_path, train_seed):
        """The regression that matters.

        compare.py finds runs with eval_<algo>_<scenario>_lam<lam>_seed*.csv.
        Any fragment added to the stem has to sit AFTER the seed or the glob
        stops matching and the algorithm silently vanishes from the table.
        """
        stem = eval_csv_stem("dqn", "peak_lam05", seed=42,
                             train_seed=train_seed, min_green=60)
        run = tmp_path / (os.path.basename(stem) + "_conn0_ep1.csv")
        run.write_text("step,system_mean_waiting_time\n0,1.0\n")

        hits = glob.glob(str(tmp_path / "eval_dqn_peak_lam05_seed*.csv"))

        assert hits == [str(run)]


class TestCompareWarnsOnMixedMinGreens:
    """A logs dir holding a 10 s-floor and a 60 s-floor run of the same algo
    would average two different action spaces into one row -- the same class of
    confound as the seed mismatch that flipped the Stage-1 headline."""

    def _runs(self, *names):
        return pd.DataFrame({"run": list(names)})

    def test_warns_when_a_group_mixes_floors(self, capsys):
        from compare import _warn_mixed_min_greens

        _warn_mixed_min_greens(self._runs(
            "eval_dqn_peak_lam05_seed42_mg10_conn0_ep1.csv",
            "eval_dqn_peak_lam05_seed43_mg60_conn0_ep1.csv",
        ), "peak", "dqn")

        out = capsys.readouterr().out
        assert "min_green" in out
        assert "10" in out and "60" in out

    def test_silent_when_every_run_shares_a_floor(self, capsys):
        from compare import _warn_mixed_min_greens

        _warn_mixed_min_greens(self._runs(
            "eval_dqn_peak_lam05_seed42_mg60_conn0_ep1.csv",
            "eval_dqn_peak_lam05_seed43_mg60_conn0_ep1.csv",
        ), "peak", "dqn")

        assert capsys.readouterr().out == ""

    def test_untagged_runs_count_as_their_own_floor(self, capsys):
        """A run from before this change carries no mg fragment. Mixing one with
        a tagged run is exactly the case worth shouting about, because the
        untagged one is a 10 s floor and looks like nothing at all."""
        from compare import _warn_mixed_min_greens

        _warn_mixed_min_greens(self._runs(
            "eval_dqn_peak_lam05_seed42_conn0_ep1.csv",
            "eval_dqn_peak_lam05_seed43_mg60_conn0_ep1.csv",
        ), "peak", "dqn")

        assert "min_green" in capsys.readouterr().out

    def test_empty_group_does_not_warn(self, capsys):
        from compare import _warn_mixed_min_greens

        _warn_mixed_min_greens(pd.DataFrame(), "peak", "dqn")

        assert capsys.readouterr().out == ""
