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


def test_completion_gap_none_within_tolerance():
    # green_wave completes 2900, ippo completes 2890 -- well under 2% spread
    baseline = _baseline_rows("corridor_peak", 10, [13.0, 14.0, 13.5])
    ippo = pd.DataFrame([
        {"controller": "ippo", "scenario": "corridor_peak", "seed": 42 + i,
         "min_green": 10, "delay_per_trip": d, "trips": 2890, "wall_s": 1.0}
        for i, d in enumerate([12.0, 15.0, 13.0])
    ])
    assert ip.completion_gap(ippo, baseline) is None


def test_completion_gap_flags_over_tolerance():
    # green_wave completes 2900, ippo strands a chunk of them at 2400 --
    # (2900 - 2400) / 2900 ~= 17% spread, well over the 2% tolerance
    baseline = _baseline_rows("corridor_peak", 10, [13.0, 14.0, 13.5])
    ippo = pd.DataFrame([
        {"controller": "ippo", "scenario": "corridor_peak", "seed": 42 + i,
         "min_green": 10, "delay_per_trip": d, "trips": 2400, "wall_s": 1.0}
        for i, d in enumerate([12.0, 15.0, 13.0])
    ])
    gap = ip.completion_gap(ippo, baseline)
    assert gap is not None
    assert gap["scenario"] == "corridor_peak"
    assert gap["spread"] > 0.02
    assert gap["trips"]["ippo"] == 2400
    assert gap["trips"]["green_wave"] == 2900


def test_completion_gap_wrong_scenario_raises():
    baseline = _baseline_rows("corridor_tidal", 10, [14.0])
    ippo = _ippo_rows("corridor_peak", 10, [12.0])
    with pytest.raises(ValueError):
        ip.completion_gap(ippo, baseline)
