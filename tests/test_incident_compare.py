"""Unit tests for analysis/incident_compare.py's cost-computation logic (no SUMO)."""
import pandas as pd
import pytest

ic = pytest.importorskip("analysis.incident_compare")


def _rows(controller, delays, seeds=(42, 43, 44)):
    return pd.DataFrame([
        {"controller": controller, "scenario": "corridor_peak", "seed": s,
         "min_green": 10, "delay_per_trip": d, "trips": 2900}
        for s, d in zip(seeds, delays)
    ])


def test_incident_cost_mean_and_sd():
    df = _rows("green_wave", [15.0, 16.0, 17.0])
    cost = ic.incident_cost(df, "green_wave", no_incident=13.47)
    assert cost["controller"] == "green_wave"
    assert abs(cost["incident_mean"] - 16.0) < 1e-9
    assert abs(cost["cost_mean"] - (16.0 - 13.47)) < 1e-9
    assert cost["n"] == 3


def test_incident_cost_filters_by_controller():
    df = pd.concat([_rows("green_wave", [15.0, 16.0, 17.0]),
                    _rows("max_pressure", [14.0, 14.5, 15.0])])
    cost = ic.incident_cost(df, "max_pressure", no_incident=13.0)
    assert cost["n"] == 3
    assert abs(cost["incident_mean"] - 14.5) < 1e-9


def test_no_incident_mean_reads_corridor_sweep_csv(tmp_path, monkeypatch):
    csv = tmp_path / "corridor_sweep.csv"
    pd.DataFrame([
        {"controller": "green_wave", "scenario": "corridor_peak", "seed": 42,
         "min_green": 10, "delay_per_trip": 13.0, "trips": 2900, "wall_s": 1.0},
        {"controller": "green_wave", "scenario": "corridor_peak", "seed": 43,
         "min_green": 10, "delay_per_trip": 14.0, "trips": 2900, "wall_s": 1.0},
    ]).to_csv(csv, index=False)
    monkeypatch.setattr(ic, "CORRIDOR_SWEEP_CSV", str(csv))
    assert abs(ic.no_incident_mean("green_wave") - 13.5) < 1e-9


def test_no_incident_mean_raises_if_missing(tmp_path, monkeypatch):
    csv = tmp_path / "corridor_sweep.csv"
    pd.DataFrame([{"controller": "green_wave", "scenario": "corridor_tidal", "seed": 42,
                   "min_green": 10, "delay_per_trip": 13.0, "trips": 2900, "wall_s": 1.0}]
                ).to_csv(csv, index=False)
    monkeypatch.setattr(ic, "CORRIDOR_SWEEP_CSV", str(csv))
    with pytest.raises(ValueError):
        ic.no_incident_mean("green_wave")
