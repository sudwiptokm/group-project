"""Unit tests for analysis/idqn_zeroshot.py's pairing/loading logic (no SUMO)."""
import pandas as pd
import pytest

iz = pytest.importorskip("analysis.idqn_zeroshot")


def _rows(scenario, delays, seeds=(42, 43, 44)):
    return pd.DataFrame([
        {"scenario": scenario, "seed": s, "min_green": 10, "delay_per_trip": d,
         "trips": 2900}
        for s, d in zip(seeds, delays)
    ])


def test_paired_gap_seed_alignment():
    idqn = _rows("corridor_offpeak", [12.0, 15.0, 13.0])
    gw = _rows("corridor_offpeak", [13.0, 14.0, 13.5])
    gap = iz.paired_gap(idqn, gw)
    # idqn - green_wave per seed: [-1.0, 1.0, -0.5]
    assert gap["n"] == 3
    assert gap["wins"] == 2
    assert abs(gap["mean"] - (-1.0 / 6)) < 1e-9
    assert gap["scenario"] == "corridor_offpeak"


def test_paired_gap_only_pairs_overlapping_seeds():
    idqn = _rows("corridor_tidal", [12.0], seeds=(42,))
    gw = _rows("corridor_tidal", [13.0, 14.0], seeds=(42, 43))
    gap = iz.paired_gap(idqn, gw)
    assert gap["n"] == 1


def test_paired_gap_wrong_scenario_raises():
    idqn = _rows("corridor_peak", [12.0])
    gw = _rows("corridor_tidal", [14.0])
    with pytest.raises(ValueError):
        iz.paired_gap(idqn, gw)


def test_load_baseline_filters_controller_scenario_min_green(tmp_path, monkeypatch):
    csv = tmp_path / "corridor_sweep.csv"
    pd.DataFrame([
        {"controller": "green_wave", "scenario": "corridor_offpeak", "seed": 42,
         "min_green": 10, "delay_per_trip": 9.0, "trips": 2900, "wall_s": 1.0},
        {"controller": "max_pressure", "scenario": "corridor_offpeak", "seed": 42,
         "min_green": 10, "delay_per_trip": 9.5, "trips": 2900, "wall_s": 1.0},
        {"controller": "green_wave", "scenario": "corridor_offpeak", "seed": 42,
         "min_green": 20, "delay_per_trip": 20.0, "trips": 2900, "wall_s": 1.0},
    ]).to_csv(csv, index=False)
    monkeypatch.setattr(iz, "CORRIDOR_SWEEP_CSV", str(csv))
    df = iz.load_baseline("green_wave", "corridor_offpeak", min_green=10)
    assert len(df) == 1
    assert df.iloc[0]["delay_per_trip"] == 9.0
