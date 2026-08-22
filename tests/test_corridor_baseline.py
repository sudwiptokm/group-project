"""Unit tests for corridor_baseline.py's incident wiring (no SUMO)."""
import inspect

import corridor_baseline as cb


def test_incident_constant_matches_spec():
    assert cb.INCIDENT == ("C1_C2", 0, 1800.0, 900.0)


def test_run_incident_param_defaults_false():
    sig = inspect.signature(cb.run)
    assert "incident" in sig.parameters
    assert sig.parameters["incident"].default is False
