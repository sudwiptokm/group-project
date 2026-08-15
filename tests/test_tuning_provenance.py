"""A params file has to say which action space it was selected for.

Hyperparameters are chosen FOR a floor and a budget. params/*.json used to
record neither, so `train.py --min-green 60` loading a file tuned at 10 s looked
identical to loading the right one — the confound the peak retrain exists to
remove (docs/FINDINGS_2026-08-12.md, item 5). tune.py now writes underscore-
prefixed provenance keys; these tests pin that train.py strips them before they
reach the algorithm constructor, and warns when they disagree with the run.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train  # noqa: E402
from tune import _serialisable, _write_params  # noqa: E402


class TestMaterialiseDropsProvenance:
    """The keys are metadata, not constructor arguments: DQN(_min_green=60)
    raises, so a params file with provenance would break every training run."""

    def test_underscore_keys_are_stripped(self):
        params = train._materialise({
            "_min_green": 60, "_tune_steps": 20000, "learning_rate": 1e-4,
        })

        assert params == {"learning_rate": 1e-4}

    def test_net_arch_still_becomes_policy_kwargs(self):
        params = train._materialise({"_min_green": 60, "net_arch": [64, 64]})

        assert params == {"policy_kwargs": {"net_arch": [64, 64]}}

    def test_files_without_provenance_are_unchanged(self):
        # every params/*.json on disk predates this and must still load
        params = train._materialise({"learning_rate": 1e-4, "gamma": 0.99})

        assert params == {"learning_rate": 1e-4, "gamma": 0.99}


class TestFloorMismatchWarning:
    def test_warns_when_the_floors_differ(self, capsys):
        train._warn_floor_mismatch("dqn", "params/dqn_peak.json",
                                   {"_min_green": 10}, min_green=60)

        out = capsys.readouterr().out
        assert "WARNING" in out and "10" in out and "60" in out

    def test_silent_when_the_floors_match(self, capsys):
        train._warn_floor_mismatch("dqn", "params/dqn_peak.json",
                                   {"_min_green": 60}, min_green=60)

        assert capsys.readouterr().out == ""

    def test_untagged_file_warns_because_it_is_a_ten_second_floor(self, capsys):
        """Legacy params carry no floor. They were all selected at 10 s — the
        floor the actuated probe showed no controller can win at — so silence
        here would be the exact failure this check exists for."""
        train._warn_floor_mismatch("dqn", "params/dqn.json", {}, min_green=60)

        assert "WARNING" in capsys.readouterr().out

    def test_no_floor_to_check_against_is_not_a_warning(self, capsys):
        train._warn_floor_mismatch("dqn", "params/dqn.json",
                                   {"_min_green": 10}, min_green=None)

        assert capsys.readouterr().out == ""


class TestWriteParams:
    def test_provenance_and_params_land_in_one_file(self, tmp_path):
        path = str(tmp_path / "dqn_peak.json")

        _write_params(path, {"learning_rate": 1e-4},
                      {"_min_green": 60, "_tune_steps": 20000})

        saved = json.loads(open(path).read())
        assert saved["_min_green"] == 60
        assert saved["learning_rate"] == 1e-4

    def test_written_file_round_trips_through_materialise(self, tmp_path):
        """The join that matters: what tune.py writes has to be loadable by
        train.py as constructor kwargs, with nothing extra left in it."""
        path = str(tmp_path / "dqn_peak.json")

        _write_params(path, _serialisable({
            "learning_rate": 1e-4,
            "policy_kwargs": {"net_arch": [64, 64]},
        }), {"_min_green": 60})

        params = train._materialise(json.loads(open(path).read()))

        assert params == {"learning_rate": 1e-4,
                          "policy_kwargs": {"net_arch": [64, 64]}}

    def test_no_partial_file_is_left_behind(self, tmp_path):
        """Parallel workers share a study and each writes the current best, so
        the write is atomic — a torn file would poison the next training run."""
        path = str(tmp_path / "dqn_peak.json")

        _write_params(path, {"learning_rate": 1e-4}, {"_min_green": 60})

        assert os.listdir(tmp_path) == ["dqn_peak.json"]


class TestStopAtTarget:
    """--trials is a target for the STUDY, not a count for the process: the
    search runs across several workers and survives a kill, so both a resumed
    run and a second worker must top the study up rather than repeat it."""

    class _FakeTrial:
        def __init__(self, complete):
            import optuna
            self.state = (optuna.trial.TrialState.COMPLETE if complete
                          else optuna.trial.TrialState.PRUNED)

    class _FakeStudy:
        def __init__(self, complete, pruned=0):
            self.trials = ([TestStopAtTarget._FakeTrial(True)] * complete +
                           [TestStopAtTarget._FakeTrial(False)] * pruned)
            self.stopped = False

        def stop(self):
            self.stopped = True

    def test_stops_once_the_study_reaches_the_target(self):
        from tune import _stop_at_target

        study = self._FakeStudy(complete=30)
        _stop_at_target(30)(study, None)

        assert study.stopped

    def test_keeps_going_below_the_target(self):
        from tune import _stop_at_target

        study = self._FakeStudy(complete=29)
        _stop_at_target(30)(study, None)

        assert not study.stopped

    def test_pruned_trials_do_not_count_towards_the_target(self):
        """A pruned trial is a gridlocked policy that timed out, not a sample of
        the search space. Counting them would end the search early with fewer
        real trials than asked for."""
        from tune import _completed

        assert _completed(self._FakeStudy(complete=5, pruned=10)) == 5
