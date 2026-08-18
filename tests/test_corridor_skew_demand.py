"""Corridor skew route generation: cross-street demand redistributes unevenly
across C1/C2/C3 while the arterial stays exactly at corridor_peak's rate.

Parses corridor_skew.rou.xml directly (no SUMO) -- structural checks belong
here, not in a slow SUMO-in-the-loop test."""
import pathlib
import re

import pytest

import make_scenarios as ms

ROOT = pathlib.Path(__file__).parent.parent
ROU = ROOT / "corridor.rou.xml"
PEAK_ROU = ROOT / "corridor_peak.rou.xml"


def _flows(text: str) -> list:
    """Every <flow .../> in a route file, as attribute dicts."""
    return [dict(re.findall(r'(\w+)="([^"]*)"', attrs))
            for attrs in re.findall(r"<flow\b([^>]*?)/>", text)]


def _rate(flow: dict) -> float:
    """Arrival rate (veh/s) out of a period="exp(lambda)" attribute."""
    m = re.match(r"exp\(([0-9.]+)\)$", flow["period"])
    assert m, f"flow {flow.get('id')} is not exponential: {flow.get('period')}"
    return float(m.group(1))


def _veh_per_hour(flow: dict) -> float:
    return _rate(flow) * 3600.0


@pytest.fixture
def skew(tmp_path):
    dst = tmp_path / "corridor_skew.rou.xml"
    ms.write_skew(str(ROU), str(dst))
    return _flows(dst.read_text())


def test_skew_in_scenario_outputs():
    # make_scenarios must emit the skew file, not only the scaled/tidal variants
    assert ms.CORRIDOR_SKEW_DST == "corridor_skew.rou.xml"


def test_skew_cross_street_demand_matches_spec(skew):
    """C1=150, C2=600, C3=150 veh/h -- uneven per node, 900 veh/h total."""
    by_route = {f["route"]: _veh_per_hour(f) for f in skew
                if f["route"].startswith("x")}
    assert by_route["x1"] == pytest.approx(150.0, abs=0.01)
    assert by_route["x2"] == pytest.approx(600.0, abs=0.01)
    assert by_route["x3"] == pytest.approx(150.0, abs=0.01)


def test_skew_cross_street_total_is_unchanged_from_peak(skew):
    # same 900 veh/h total as corridor_peak's 3x300 -- redistributed, not added
    total = sum(_veh_per_hour(f) for f in skew if f["route"].startswith("x"))
    assert total == pytest.approx(900.0, abs=0.05)


def test_skew_arterial_matches_corridor_peak_exactly(skew, tmp_path):
    """eb/wb flows are byte-identical to corridor_peak.rou.xml's."""
    peak_text = PEAK_ROU.read_text()
    skew_text = (tmp_path / "corridor_skew.rou.xml").read_text()

    peak_arterial = [ln for ln in peak_text.splitlines()
                     if 'id="f_eb"' in ln or 'id="f_wb"' in ln]
    skew_arterial = [ln for ln in skew_text.splitlines()
                     if 'id="f_eb"' in ln or 'id="f_wb"' in ln]
    assert peak_arterial and skew_arterial
    assert peak_arterial == skew_arterial


def test_skew_arrivals_are_stochastic(skew, tmp_path):
    # deterministic vehsPerHour would make every "held-out demand seed" the
    # same arrival pattern -- the defect make_scenarios' docstring documents
    text = (tmp_path / "corridor_skew.rou.xml").read_text()
    assert "vehsPerHour" not in text
    for f in skew:
        _rate(f)


def test_skew_rejects_unknown_flow_id(tmp_path):
    # a silent default would let a new flow in corridor.rou.xml be scaled by
    # the wrong (or no) rate without anyone noticing
    src = tmp_path / "corridor.rou.xml"
    src.write_text('<routes><flow id="f_mystery" route="eb" vehsPerHour="100"/></routes>')
    with pytest.raises(KeyError):
        ms.write_skew(str(src), str(tmp_path / "out.rou.xml"))


def test_skew_scenario_is_registered():
    ec = pytest.importorskip("env_common")
    assert ec.SCENARIO_ROUTES["corridor_skew"] == ms.CORRIDOR_SKEW_DST


def test_corridor_scenarios_include_skew():
    ec = pytest.importorskip("env_common")
    assert "corridor_skew" in ec.CORRIDOR_SCENARIOS
