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
    no_incident = {42: 13.0, 43: 13.47, 44: 14.0}
    cost = ic.incident_cost(df, "green_wave", no_incident)
    assert cost["controller"] == "green_wave"
    assert abs(cost["incident_mean"] - 16.0) < 1e-9
    expected_deltas = [15.0 - 13.0, 16.0 - 13.47, 17.0 - 14.0]
    assert abs(cost["cost_mean"] - (sum(expected_deltas) / 3)) < 1e-9
    assert cost["n"] == 3


def test_incident_cost_filters_by_controller():
    df = pd.concat([_rows("green_wave", [15.0, 16.0, 17.0]),
                    _rows("max_pressure", [14.0, 14.5, 15.0])])
    no_incident = {42: 13.0, 43: 13.0, 44: 13.0}
    cost = ic.incident_cost(df, "max_pressure", no_incident)
    assert cost["n"] == 3
    assert abs(cost["incident_mean"] - 14.5) < 1e-9


def test_incident_cost_uses_seed_matched_baseline_not_mean():
    # seed 43's no-incident baseline (10.0) is far from seeds 42/44's (30.0),
    # mirroring max_pressure's confirmed bimodal no-incident distribution
    # (corridor_sweep.csv: seeds 42/44 ~28-29s, seed 43 ~21.9s). A mean-based
    # baseline (here: 23.33 for all three seeds) would produce cost deltas of
    # [17.67, 18.67, 19.67] -- tight, misleadingly low variance. The correct
    # seed-matched deltas are [11.0, 32.0, 13.0], which have a much larger sd;
    # this is exactly the signal a mean-based baseline would hide.
    df = _rows("max_pressure", [41.0, 42.0, 43.0])
    no_incident = {42: 30.0, 43: 10.0, 44: 30.0}
    cost = ic.incident_cost(df, "max_pressure", no_incident)
    assert abs(cost["cost_sd"] - 11.590226) < 1e-5
    assert cost["cost_sd"] > 5.0  # would be ~1.0 under mean-based subtraction


def test_no_incident_for_reads_corridor_sweep_csv(tmp_path, monkeypatch):
    csv = tmp_path / "corridor_sweep.csv"
    pd.DataFrame([
        {"controller": "green_wave", "scenario": "corridor_peak", "seed": 42,
         "min_green": 10, "delay_per_trip": 13.0, "trips": 2900, "wall_s": 1.0},
        {"controller": "green_wave", "scenario": "corridor_peak", "seed": 43,
         "min_green": 10, "delay_per_trip": 14.0, "trips": 2900, "wall_s": 1.0},
    ]).to_csv(csv, index=False)
    monkeypatch.setattr(ic, "CORRIDOR_SWEEP_CSV", str(csv))
    # exact per-seed values, not the 13.5 mean
    assert abs(ic.no_incident_for("green_wave", 42) - 13.0) < 1e-9
    assert abs(ic.no_incident_for("green_wave", 43) - 14.0) < 1e-9


def test_no_incident_for_raises_if_missing(tmp_path, monkeypatch):
    csv = tmp_path / "corridor_sweep.csv"
    pd.DataFrame([{"controller": "green_wave", "scenario": "corridor_tidal", "seed": 42,
                   "min_green": 10, "delay_per_trip": 13.0, "trips": 2900, "wall_s": 1.0}]
                ).to_csv(csv, index=False)
    monkeypatch.setattr(ic, "CORRIDOR_SWEEP_CSV", str(csv))
    with pytest.raises(ValueError):
        ic.no_incident_for("green_wave", 42)


def test_no_incident_for_raises_if_seed_missing(tmp_path, monkeypatch):
    # row exists for the (controller, scenario, min_green) combo but not for
    # this specific seed -- must still raise, not silently fall back to a
    # mean or another seed's value.
    csv = tmp_path / "corridor_sweep.csv"
    pd.DataFrame([{"controller": "green_wave", "scenario": "corridor_peak", "seed": 42,
                   "min_green": 10, "delay_per_trip": 13.0, "trips": 2900, "wall_s": 1.0}]
                ).to_csv(csv, index=False)
    monkeypatch.setattr(ic, "CORRIDOR_SWEEP_CSV", str(csv))
    with pytest.raises(ValueError):
        ic.no_incident_for("green_wave", 43)


def test_no_incident_for_idqn_reads_its_own_per_seed_tripinfo(monkeypatch):
    # idqn has no per-seed row in corridor_sweep.csv by design -- unlike a
    # disclosed scalar constant shared across seeds, its no-incident baseline
    # must now be read per-seed from its own no-incident tripinfo XML (SP5
    # checkpoints), same seed-matching discipline as green_wave/max_pressure.
    seen_stems = []

    def fake_reduce_tripinfo(path, departed=None):
        seen_stems.append(path)
        # fabricate a distinct delay per seed so a mean-collapsed stub
        # couldn't accidentally pass this test
        delay = {42: 16.952, 43: 16.490, 44: 16.244}[int(path.split("seed")[1].split("_")[0])]
        return {"trip_time_loss_mean": delay}

    monkeypatch.setattr(ic, "reduce_tripinfo", fake_reduce_tripinfo)

    assert abs(ic.no_incident_for("idqn", 42) - 16.952) < 1e-9
    assert abs(ic.no_incident_for("idqn", 43) - 16.490) < 1e-9
    assert abs(ic.no_incident_for("idqn", 44) - 16.244) < 1e-9
    # each seed reads its own tripinfo stem, not one shared path
    assert len(set(seen_stems)) == 3
    for seed, path in zip((42, 43, 44), seen_stems):
        assert f"seed{seed}" in path
        assert path.endswith("_tripinfo.xml")
