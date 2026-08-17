"""The corridor sweep's report must separate scenarios.

analysis/corridor_sweep.csv accumulates every scenario ever swept. report()
originally pivoted on min_green alone, so the moment a second scenario was
added it averaged corridor_peak and corridor_tidal into one cell -- 20 "seeds"
per floor, a mean over two different demands, and a "best floor" that belongs
to neither. That is the same class of defect as averaging two min_green values
into one row, which the CSV filename convention exists to prevent.
"""
import re

import pandas as pd
import pytest

cs = pytest.importorskip("analysis.corridor_sweep")


def _rows(scenario, controller, per_floor):
    """per_floor: {min_green: [delay per seed]}"""
    return [
        {"controller": controller, "scenario": scenario, "seed": 42 + i,
         "min_green": mg, "delay_per_trip": d, "trips": 2000, "wall_s": 1.0}
        for mg, delays in per_floor.items()
        for i, d in enumerate(delays)
    ]


@pytest.fixture
def two_scenarios():
    # pooled, mg5 looks best for both controllers (mean 20 vs 25 at mg15);
    # per scenario, tidal's best floor is mg15. A report that pools cannot see
    # this, and mg15 is exactly where the peak sweep found max_pressure's win.
    rows = []
    rows += _rows("corridor_peak", "green_wave", {5: [10.0, 10.0], 15: [40.0, 40.0]})
    rows += _rows("corridor_tidal", "green_wave", {5: [30.0, 30.0], 15: [10.0, 10.0]})
    rows += _rows("corridor_peak", "max_pressure", {5: [9.0, 9.0], 15: [41.0, 41.0]})
    rows += _rows("corridor_tidal", "max_pressure", {5: [31.0, 31.0], 15: [8.0, 8.0]})
    return pd.DataFrame(rows)


def test_best_floor_is_found_per_scenario(two_scenarios):
    best = cs.best_floors(two_scenarios)
    assert best[("corridor_peak", "green_wave")][0] == 5
    assert best[("corridor_tidal", "green_wave")][0] == 15
    assert best[("corridor_tidal", "max_pressure")] == (15, 8.0)


def test_paired_diffs_never_pair_across_scenarios(two_scenarios):
    diffs = {(d["scenario"], d["min_green"]): d
             for d in cs.paired_diffs(two_scenarios)}
    # two seeds per (scenario, floor), never four
    assert all(d["n"] == 2 for d in diffs.values()), diffs
    # tidal mg15: max_pressure 8.0 vs green_wave 10.0 -> -2.0, wins both seeds
    tidal = diffs[("corridor_tidal", 15)]
    assert tidal["mean"] == pytest.approx(-2.0)
    assert tidal["wins"] == 2
    # peak mg15: max_pressure is worse there; pooling would have cancelled these
    assert diffs[("corridor_peak", 15)]["mean"] == pytest.approx(+1.0)
    assert diffs[("corridor_peak", 15)]["wins"] == 0


def test_report_runs_over_every_scenario(two_scenarios, capsys):
    cs.report(two_scenarios)
    out = capsys.readouterr().out
    assert "corridor_peak" in out and "corridor_tidal" in out


def _completion_frame(gw_trips, mp_trips):
    rows = []
    for controller, trips in (("green_wave", gw_trips), ("max_pressure", mp_trips)):
        for mg, n in trips.items():
            for seed in (42, 43):
                rows.append({"controller": controller, "scenario": "corridor_tidal",
                             "seed": seed, "min_green": mg,
                             "delay_per_trip": 20.0, "trips": n, "wall_s": 1.0})
    return pd.DataFrame(rows)


def test_diverging_completion_counts_are_flagged():
    """Delay per COMPLETED trip is survivorship-biased across controllers.

    If one controller jams an approach so badly that 500 vehicles never finish,
    its delay per completed trip is computed over the survivors and can look
    BETTER than a controller that cleared everyone. That is the defect behind
    the withdrawn -24% headline, and corridor_tidal is deliberately loaded past
    what a fixed round-robin can discharge, so it is expected here rather than
    hypothetical.
    """
    df = _completion_frame({5: 3000, 15: 2400}, {5: 2990, 15: 2990})
    flagged = {g["min_green"] for g in cs.completion_gaps(df)}
    assert flagged == {15}, cs.completion_gaps(df)


def test_completion_gap_reports_both_sides():
    df = _completion_frame({5: 3000, 15: 2400}, {5: 2990, 15: 2990})
    gap = next(g for g in cs.completion_gaps(df) if g["min_green"] == 15)
    assert gap["scenario"] == "corridor_tidal"
    assert gap["trips"]["green_wave"] == pytest.approx(2400)
    assert gap["trips"]["max_pressure"] == pytest.approx(2990)


def test_report_warns_when_completion_diverges(capsys):
    cs.report(_completion_frame({5: 3000, 15: 2400}, {5: 2990, 15: 2990}))
    out = capsys.readouterr().out
    assert "survivorship" in out.lower()
    assert re.search(r"mg\s+15:", out), out
