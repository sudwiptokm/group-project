"""Unit tests for analysis/ippo_sweep.py's pairing/verdict logic (no SUMO)."""
import pandas as pd
import pytest

ip = pytest.importorskip("analysis.ippo_sweep")


def _baseline_rows(scenario, min_green, delays):
    return pd.DataFrame([
        {"controller": "green_wave", "scenario": scenario, "seed": 42 + i,
         "min_green": min_green, "delay_per_trip": d, "trips": 2900, "wall_s": 1.0}
        for i, d in enumerate(delays)
    ])


def _ippo_rows(scenario, min_green, delays):
    return pd.DataFrame([
        {"controller": "ippo", "scenario": scenario, "seed": 42 + i,
         "min_green": min_green, "delay_per_trip": d, "trips": 2900, "wall_s": 1.0}
        for i, d in enumerate(delays)
    ])


def test_paired_vs_green_wave_seed_alignment():
    baseline = _baseline_rows("corridor_peak", 10, [13.0, 14.0, 13.5])
    ippo = _ippo_rows("corridor_peak", 10, [12.0, 15.0, 13.0])
    d = ip.paired_vs_green_wave(ippo, baseline)
    # ippo - green_wave per seed: [-1.0, 1.0, -0.5]
    assert d["n"] == 3
    assert d["wins"] == 2          # negative diff = ippo wins (lower delay)
    assert abs(d["mean"] - (-1.0 / 6)) < 1e-9


def test_paired_vs_green_wave_requires_matching_seeds():
    baseline = _baseline_rows("corridor_peak", 10, [13.0, 14.0])
    ippo = _ippo_rows("corridor_peak", 10, [12.0])  # only seed 42 present
    d = ip.paired_vs_green_wave(ippo, baseline)
    assert d["n"] == 1             # only the overlapping seed is paired


def test_paired_vs_green_wave_wrong_scenario_raises():
    baseline = _baseline_rows("corridor_tidal", 10, [14.0])
    ippo = _ippo_rows("corridor_peak", 10, [12.0])
    with pytest.raises(ValueError):
        ip.paired_vs_green_wave(ippo, baseline)
