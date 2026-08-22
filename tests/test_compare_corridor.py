"""compare.py must aggregate corridor controller CSVs into a comparison row."""
import pandas as pd

import compare


def test_run_means_reads_controller_csv(tmp_path):
    # a fake corridor eval CSV in the expected filename shape
    p = tmp_path / "eval_green_wave_corridor_peak_seed0_conn0_ep1.csv"
    pd.DataFrame({
        "system_mean_waiting_time": [1.0, 3.0],
        "system_total_stopped": [2.0, 4.0],
        "system_mean_speed": [5.0, 5.0],
        "system_total_waiting_time": [10.0, 10.0],
    }).to_csv(p, index=False)

    df = compare._run_means(str(tmp_path), "green_wave", "corridor_peak")
    assert len(df) == 1
    # metrics are time-averaged over the episode rows
    assert df["system_mean_waiting_time"].iloc[0] == 2.0


class TestCompareWarnsOnMixedEvalScenarios:
    """A corridor_peak checkpoint zero-shot-evaluated on corridor_offpeak still
    matches the corridor_peak glob (_run_means keys on the CHECKPOINT scenario
    embedded early in the filename, not the demand actually run) -- averaging
    it into the corridor_peak row silently mixes in a different demand. Same
    class of confound as TestCompareWarnsOnMixedMinGreens in
    tests/test_min_green.py; mirrors that test's shape."""

    def _runs(self, *names):
        return pd.DataFrame({"run": list(names)})

    def test_warns_when_a_group_carries_a_zero_shot_run(self, capsys):
        from compare import _warn_mixed_eval_scenarios

        _warn_mixed_eval_scenarios(self._runs(
            "eval_idqn_corridor_peak_lam05_seed42_mg10_s100000_conn0_ep1.csv",
            "eval_idqn_corridor_peak_lam05_seed43_mg10_s100000_on_corridor_offpeak_conn1_ep1.csv",
        ), "corridor_peak", "idqn/lam05")

        out = capsys.readouterr().out
        assert "_on_" in out
        assert "corridor_offpeak" in out

    def test_silent_when_every_run_is_in_distribution(self, capsys):
        from compare import _warn_mixed_eval_scenarios

        _warn_mixed_eval_scenarios(self._runs(
            "eval_idqn_corridor_peak_lam05_seed42_mg10_s100000_conn0_ep1.csv",
            "eval_idqn_corridor_peak_lam05_seed43_mg10_s100000_conn1_ep1.csv",
        ), "corridor_peak", "idqn/lam05")

        assert capsys.readouterr().out == ""

    def test_empty_group_does_not_warn(self, capsys):
        from compare import _warn_mixed_eval_scenarios

        _warn_mixed_eval_scenarios(pd.DataFrame(), "corridor_peak", "idqn/lam05")

        assert capsys.readouterr().out == ""


class TestCompareWarnsOnMixedIncident:
    """SP7 tags incident-eval CSVs with '_incident' after seed<n>, same
    convention as every other optional fragment `_run_means`'s glob is
    agnostic to -- averaging one into a plain baseline/algo row silently
    mixes a mid-episode lane-closure run into a no-incident mean. Same
    class of confound as TestCompareWarnsOnMixedEvalScenarios above;
    mirrors that test's shape."""

    def _runs(self, *names):
        return pd.DataFrame({"run": list(names)})

    def test_warns_when_a_group_carries_an_incident_run(self, capsys):
        from compare import _warn_mixed_incident

        _warn_mixed_incident(self._runs(
            "eval_green_wave_corridor_peak_seed42_mg10_conn0_ep1.csv",
            "eval_green_wave_corridor_peak_seed43_mg10_incident_conn1_ep1.csv",
        ), "corridor_peak", "green_wave")

        out = capsys.readouterr().out
        assert "_incident" in out
        assert "eval_green_wave_corridor_peak_seed43_mg10_incident_conn1_ep1.csv" in out

    def test_silent_when_no_run_is_incident_tagged(self, capsys):
        from compare import _warn_mixed_incident

        _warn_mixed_incident(self._runs(
            "eval_green_wave_corridor_peak_seed42_mg10_conn0_ep1.csv",
            "eval_green_wave_corridor_peak_seed43_mg10_conn1_ep1.csv",
        ), "corridor_peak", "green_wave")

        assert capsys.readouterr().out == ""

    def test_empty_group_does_not_warn(self, capsys):
        from compare import _warn_mixed_incident

        _warn_mixed_incident(pd.DataFrame(), "corridor_peak", "green_wave")

        assert capsys.readouterr().out == ""
