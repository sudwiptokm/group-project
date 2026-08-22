"""Unit tests for analysis/idqn_sweep.py's pairing logic (no SUMO)."""
import pandas as pd
import pytest

idq = pytest.importorskip("analysis.idqn_sweep")

import train_corridor_dqn as tcd


def _rows(controller, scenario, min_green, delays):
    return pd.DataFrame([
        {"controller": controller, "scenario": scenario, "seed": 42 + i,
         "min_green": min_green, "delay_per_trip": d, "trips": 2900, "wall_s": 1.0}
        for i, d in enumerate(delays)
    ])


def test_run_one_trip_path_uses_eval_out_stem(monkeypatch):
    """run_one's tripinfo lookup must route through
    train_corridor_dqn._eval_out_stem -- the single source of truth for the
    eval-CSV filename convention (added for SP6's zero-shot eval) -- rather
    than hand-building the stem inline. idqn_sweep only ever evaluates
    in-distribution, so this passes `scenario` as both the checkpoint scenario
    and the eval scenario; that must produce the identical stem the old
    inline `f"logs/eval_idqn_{tcd._tag(...)}"` construction did (pure
    refactor, no behavior change)."""
    scenario, seed, min_green, lam, steps = "corridor_peak", 42, 10, 0.5, 100000

    # checkpoints "exist" so run_one skips training entirely
    monkeypatch.setattr(idq.os.path, "exists", lambda p: True)
    monkeypatch.setattr(tcd, "evaluate", lambda *a, **k: "unused.csv")

    seen = {}

    def fake_tripinfo_path(stem):
        seen["stem"] = stem
        return "fake_tripinfo.xml"

    monkeypatch.setattr(idq, "tripinfo_path", fake_tripinfo_path)
    monkeypatch.setattr(idq, "reduce_tripinfo", lambda path: {
        "trip_time_loss_mean": 1.0, "trips_completed": 100,
    })

    idq.run_one(scenario, seed, min_green, lam, steps)

    expected = tcd._eval_out_stem(scenario, scenario, lam, seed, min_green, steps)
    assert seen["stem"] == expected
    # sanity: still matches the old hand-built convention exactly
    assert seen["stem"] == f"logs/eval_idqn_{tcd._tag(scenario, lam, seed, min_green, steps)}"


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


def test_completion_gap_none_within_tolerance():
    # green_wave completes 2900, idqn completes 2890 -- well under 2% spread
    baseline = _rows("green_wave", "corridor_peak", 10, [13.0, 14.0, 13.5])
    idqn = pd.DataFrame([
        {"controller": "idqn", "scenario": "corridor_peak", "seed": 42 + i,
         "min_green": 10, "delay_per_trip": d, "trips": 2890, "wall_s": 1.0}
        for i, d in enumerate([12.0, 15.0, 13.0])
    ])
    assert idq.completion_gap(idqn, baseline, "green_wave") is None


def test_completion_gap_flags_over_tolerance():
    # green_wave completes 2900, idqn strands a chunk of them at 2400 --
    # (2900 - 2400) / 2900 ~= 17% spread, well over the 2% tolerance
    baseline = _rows("green_wave", "corridor_peak", 10, [13.0, 14.0, 13.5])
    idqn = pd.DataFrame([
        {"controller": "idqn", "scenario": "corridor_peak", "seed": 42 + i,
         "min_green": 10, "delay_per_trip": d, "trips": 2400, "wall_s": 1.0}
        for i, d in enumerate([12.0, 15.0, 13.0])
    ])
    gap = idq.completion_gap(idqn, baseline, "green_wave")
    assert gap is not None
    assert gap["scenario"] == "corridor_peak"
    assert gap["vs"] == "green_wave"
    assert gap["spread"] > 0.02
    assert gap["trips"]["idqn"] == 2400
    assert gap["trips"]["green_wave"] == 2900


def test_completion_gap_wrong_scenario_raises():
    baseline = _rows("green_wave", "corridor_tidal", 10, [14.0])
    idqn = _rows("idqn", "corridor_peak", 10, [12.0])
    with pytest.raises(ValueError):
        idq.completion_gap(idqn, baseline, "green_wave")


def _wall_rows(controller, scenario, min_green, wall_s_values):
    return pd.DataFrame([
        {"controller": controller, "scenario": scenario, "seed": 42 + i,
         "min_green": min_green, "delay_per_trip": 13.0, "trips": 2900,
         "wall_s": w}
        for i, w in enumerate(wall_s_values)
    ])


def test_budget_gap_none_within_tolerance():
    idqn = _wall_rows("idqn", "corridor_peak", 10, [4220.0, 4439.0, 4255.0])
    ippo = _wall_rows("ippo", "corridor_peak", 10, [4120.0, 4244.0, 4168.0])
    assert idq.budget_gap(idqn, ippo, "ippo") is None


def test_budget_gap_flags_mismatched_budgets():
    # idqn at ~4200s/seed (100k steps) vs ippo at ~900s/seed (16k steps):
    # this is exactly SP5's original bug -- a >4x wall_s gap.
    idqn = _wall_rows("idqn", "corridor_peak", 10, [4220.0, 4439.0, 4255.0])
    ippo = _wall_rows("ippo", "corridor_peak", 10, [692.0, 944.0, 1128.0])
    gap = idq.budget_gap(idqn, ippo, "ippo")
    assert gap is not None
    assert gap["vs"] == "ippo"
    assert gap["ratio"] > 2.0


def test_budget_gap_ignores_reused_nan_wall_s():
    # a fully-reused sweep (all checkpoints already on disk) has wall_s=NaN
    # for every row -- budget_gap must not blow up or false-positive on that
    idqn = _wall_rows("idqn", "corridor_peak", 10, [float("nan")] * 3)
    ippo = _wall_rows("ippo", "corridor_peak", 10, [4120.0, 4244.0, 4168.0])
    assert idq.budget_gap(idqn, ippo, "ippo") is None


def test_load_ippo_bar_defaults_to_100k_csv():
    assert idq.load_ippo_bar.__defaults__[0] == idq.IPPO_SWEEP_100K_CSV
