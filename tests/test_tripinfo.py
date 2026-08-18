"""Reducer for SUMO tripinfo output -> completed-trip metrics."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.tripinfo import count_departures, reduce_tripinfo  # noqa: E402

ROUTES = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <route id="N_S" edges="n_c c_s"/>
    <vehicle id="v0" route="N_S" depart="0.00"/>
    <flow id="f_a" route="N_S" begin="0" end="3600" vehsPerHour="720"/>
    <flow id="f_b" route="N_S" begin="0" end="1800" number="100"/>
</routes>
"""

TRIPINFO = """<?xml version="1.0" encoding="UTF-8"?>
<tripinfos>
    <tripinfo id="v0" depart="0.00" departDelay="0.00" arrival="60.00"
              duration="60.00" routeLength="600.00" waitingTime="10.00"
              waitingCount="1" timeLoss="20.00" vType="car" devices="tripinfo_v0"/>
    <tripinfo id="v1" depart="10.00" departDelay="2.00" arrival="90.00"
              duration="80.00" routeLength="600.00" waitingTime="30.00"
              waitingCount="3" timeLoss="40.00" vType="moto@1" devices="tripinfo_v1"/>
    <tripinfo id="v2" depart="20.00" departDelay="0.00" arrival="140.00"
              duration="120.00" routeLength="600.00" waitingTime="0.00"
              waitingCount="0" timeLoss="30.00" vType="auto" devices="tripinfo_v2"/>
</tripinfos>
"""


@pytest.fixture
def tripinfo_file(tmp_path):
    p = tmp_path / "run_tripinfo.xml"
    p.write_text(TRIPINFO)
    return str(p)


def test_completed_trips_are_counted(tripinfo_file):
    """Throughput is the number of trips that finished, not vehicles inserted."""
    assert reduce_tripinfo(tripinfo_file)["trips_completed"] == 3


def test_delay_and_wait_are_per_completed_trip_means(tripinfo_file):
    m = reduce_tripinfo(tripinfo_file)
    assert m["trip_time_loss_mean"] == pytest.approx((20 + 40 + 30) / 3)
    assert m["trip_waiting_time_mean"] == pytest.approx((10 + 30 + 0) / 3)
    assert m["trip_duration_mean"] == pytest.approx((60 + 80 + 120) / 3)
    assert m["trip_depart_delay_mean"] == pytest.approx(2 / 3)


def test_time_loss_is_also_reported_per_vehicle_class(tripinfo_file):
    """Heterogeneous traffic is the point of the study: a plan that clears cars
    by stranding motorcycles must not look identical to one that does not.
    vType suffixes (distribution ids like "moto@1") resolve to the base class."""
    m = reduce_tripinfo(tripinfo_file)
    assert m["trip_time_loss_mean_moto"] == pytest.approx(40.0)
    assert m["trip_time_loss_mean_car"] == pytest.approx(20.0)
    assert m["trip_time_loss_mean_auto"] == pytest.approx(30.0)
    assert m["trips_completed_moto"] == 1


def test_departed_total_gives_a_completion_rate(tripinfo_file):
    """Only completed trips appear in tripinfo, so throughput alone cannot tell
    'cleared everything' from 'inserted almost nothing'. With the demand known,
    the completion rate can."""
    m = reduce_tripinfo(tripinfo_file, departed=6)
    assert m["trip_completion_rate"] == pytest.approx(0.5)


def test_completion_rate_absent_when_demand_unknown(tripinfo_file):
    assert "trip_completion_rate" not in reduce_tripinfo(tripinfo_file)


@pytest.fixture
def route_file(tmp_path):
    p = tmp_path / "demand.rou.xml"
    p.write_text(ROUTES)
    return str(p)


def test_demand_counts_flows_and_explicit_vehicles(route_file):
    """make_scenarios.py writes vehsPerHour flows, not <vehicle> elements, so a
    counter that only sees <vehicle>/number reports a demand of ~zero."""
    assert count_departures(route_file) == pytest.approx(1 + 720 + 100)


def test_demand_is_clipped_to_the_episode_horizon(route_file):
    """A 1200 s episode does not experience 3600 s of demand. Without the clip
    the completion rate of a short episode looks like a third of the truth."""
    # 1200 s: vehicle v0, 720/h * 1200 s = 240, and 100 spread over f_b's 1800 s
    assert count_departures(route_file, horizon=1200) == pytest.approx(
        1 + 240 + 100 * (1200 / 1800)
    )


def test_missing_file_is_an_empty_result_not_a_crash(tmp_path):
    """compare.py aggregates over runs predating tripinfo logging."""
    assert reduce_tripinfo(str(tmp_path / "nope.xml")) == {}


def test_empty_tripinfo_reports_zero_throughput(tmp_path):
    """A gridlocked episode completes no trips. That is a real, reportable
    result -- 0 throughput -- not a missing measurement."""
    p = tmp_path / "empty_tripinfo.xml"
    p.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<tripinfos/>\n')
    m = reduce_tripinfo(str(p))
    assert m["trips_completed"] == 0
    assert m["trip_time_loss_mean"] != m["trip_time_loss_mean"]  # NaN


def test_truncated_file_still_reduces(tmp_path):
    """SUMO writes tripinfo incrementally and only closes </tripinfos> on a
    clean shutdown. A killed run leaves the tag open; the trips already written
    are still valid data."""
    p = tmp_path / "cut_tripinfo.xml"
    p.write_text(TRIPINFO.replace("</tripinfos>", ""))
    assert reduce_tripinfo(str(p))["trips_completed"] == 3
