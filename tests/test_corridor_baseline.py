"""Unit tests for corridor_baseline.py's incident wiring (no SUMO)."""
import inspect
from unittest.mock import MagicMock, patch

import corridor_baseline as cb


def test_incident_constant_matches_spec():
    assert cb.INCIDENT == ("C1_C2", 0, 1800.0, 900.0)


def test_run_incident_param_defaults_false():
    sig = inspect.signature(cb.run)
    assert "incident" in sig.parameters
    assert sig.parameters["incident"].default is False


def test_run_output_path_includes_incident_fragment():
    """Verify that run() returns a path with _incident fragment when incident=True,
    and without it when incident=False. This catches the bug where the returned
    path doesn't match the file written to disk."""
    # Mock make_corridor_env to avoid requiring SUMO
    mock_env = MagicMock()
    mock_env.label = "someconn"
    mock_env.episode = 1
    mock_env.out_csv_name = "logs/eval_green_wave_corridor_offpeak_seed0_mg15_incident.csv"
    # Make the env.step() exit the loop on first call
    mock_env.step.return_value = (None, None, {"__all__": True}, None)
    # Mock ts_ids to avoid green_wave_actions errors
    mock_env.ts_ids = []

    with patch("corridor_baseline.make_corridor_env", return_value=mock_env):
        # Test without incident
        result_no_incident = cb.run(
            scenario="corridor_offpeak",
            controller="green_wave",
            seed=0,
            min_green=15,
            tripinfo=True,
            incident=False,
        )
        assert "_incident" not in result_no_incident
        assert result_no_incident.endswith("_conn" + mock_env.label + "_ep1.csv")

        # Test with incident
        result_with_incident = cb.run(
            scenario="corridor_offpeak",
            controller="green_wave",
            seed=0,
            min_green=15,
            tripinfo=True,
            incident=True,
        )
        assert "_incident" in result_with_incident
        assert result_with_incident.endswith("_incident_conn" + mock_env.label + "_ep1.csv")
