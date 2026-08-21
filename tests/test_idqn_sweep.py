"""Unit tests for analysis/idqn_sweep.py's pairing logic (no SUMO)."""
import pandas as pd
import pytest

idq = pytest.importorskip("analysis.idqn_sweep")


def _rows(controller, scenario, min_green, delays):
    return pd.DataFrame([
        {"controller": controller, "scenario": scenario, "seed": 42 + i,
         "min_green": min_green, "delay_per_trip": d, "trips": 2900, "wall_s": 1.0}
        for i, d in enumerate(delays)
    ])


def test_paired_vs_seed_alignment():
    baseline = _rows("green_wave", "corridor_peak", 10, [13.0, 14.0, 13.5])
    idqn = _rows("idqn", "corridor_peak", 10, [12.0, 15.0, 13.0])
    d = idq.paired_vs(idqn, baseline, "green_wave")
    # idqn - green_wave per seed: [-1.0, 1.0, -0.5]
    assert d["n"] == 3
    assert d["wins"] == 2          # negative diff = idqn wins (lower delay)
    assert abs(d["mean"] - (-1.0 / 6)) < 1e-9
    assert d["vs"] == "green_wave"


def test_paired_vs_requires_matching_seeds():
    baseline = _rows("ippo", "corridor_peak", 10, [13.0, 14.0])
    idqn = _rows("idqn", "corridor_peak", 10, [12.0])  # only seed 42 present
    d = idq.paired_vs(idqn, baseline, "ippo")
    assert d["n"] == 1             # only the overlapping seed is paired


def test_paired_vs_wrong_scenario_raises():
    baseline = _rows("green_wave", "corridor_tidal", 10, [14.0])
    idqn = _rows("idqn", "corridor_peak", 10, [12.0])
    with pytest.raises(ValueError):
        idq.paired_vs(idqn, baseline, "green_wave")
