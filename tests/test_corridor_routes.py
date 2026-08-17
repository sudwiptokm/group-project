"""Corridor route generation: scaling produces peak/offpeak, the tidal profile
reverses the dominant direction at half time, and every edge a route references
actually exists in the built network."""
import pathlib
import re

import pytest

import make_scenarios as ms

sumolib = pytest.importorskip("sumolib")

# anchor to project root so paths work regardless of pytest's invocation dir
ROOT = pathlib.Path(__file__).parent.parent
NET = ROOT / "corridor.net.xml"
ROU = ROOT / "corridor.rou.xml"


def test_corridor_in_factors():
    # make_scenarios must know how to scale the corridor base file
    assert "corridor_peak.rou.xml" in ms.CORRIDOR_FACTORS
    assert "corridor_offpeak.rou.xml" in ms.CORRIDOR_FACTORS


def test_scaling_multiplies_flows(tmp_path):
    src = tmp_path / "corridor.rou.xml"
    src.write_text('<routes><flow id="f" vehsPerHour="100"/></routes>')
    dst = tmp_path / "out.rou.xml"
    ms.scale_file(str(src), str(dst), 1.5)
    text = dst.read_text()
    assert 'vehsPerHour="150"' in text


def _flows(text: str) -> list:
    """Every <flow .../> in a route file, as attribute dicts."""
    return [dict(re.findall(r'(\w+)="([^"]*)"', attrs))
            for attrs in re.findall(r"<flow\b([^>]*?)/>", text)]


def _rate(flow: dict) -> float:
    """Arrival rate (veh/s) out of a period="exp(lambda)" attribute."""
    m = re.match(r"exp\(([0-9.]+)\)$", flow["period"])
    assert m, f"flow {flow.get('id')} is not exponential: {flow.get('period')}"
    return float(m.group(1))


@pytest.fixture
def tidal(tmp_path):
    dst = tmp_path / "corridor_tidal.rou.xml"
    ms.write_tidal(str(ROU), str(dst))
    return _flows(dst.read_text())


@pytest.fixture
def peak(tmp_path):
    dst = tmp_path / "corridor_peak.rou.xml"
    ms.scale_file(str(ROU), str(dst), 1.5, stochastic=True)
    return _flows(dst.read_text())


def test_tidal_in_scenario_outputs():
    # make_scenarios must emit the tidal file, not only the scaled variants
    assert ms.CORRIDOR_TIDAL_DST == "corridor_tidal.rou.xml"


def test_tidal_reverses_dominant_direction_at_half_time(tidal):
    eb = sorted((f for f in tidal if f["route"] == "eb"),
                key=lambda f: float(f["begin"]))
    wb = sorted((f for f in tidal if f["route"] == "wb"),
                key=lambda f: float(f["begin"]))
    assert len(eb) == 2 and len(wb) == 2, "each arterial flow splits in two"
    assert _rate(eb[0]) > _rate(eb[1]), "eastbound dominates the first half"
    assert _rate(wb[1]) > _rate(wb[0]), "westbound dominates the second half"
    # the tide is symmetric: the same load, pointed the other way
    assert _rate(eb[0]) == pytest.approx(_rate(wb[1]))
    assert _rate(eb[1]) == pytest.approx(_rate(wb[0]))


def test_tidal_windows_tile_the_episode(tidal):
    for route in ("eb", "wb"):
        windows = sorted((float(f["begin"]), float(f["end"]))
                         for f in tidal if f["route"] == route)
        assert windows == [(0.0, ms.TIDAL_SWITCH_S),
                           (ms.TIDAL_SWITCH_S, ms.EPISODE_SECONDS)], route


def test_tidal_flows_are_sorted_by_departure_time(tidal):
    """SUMO reads a route file in order and SILENTLY DROPS any flow that begins
    earlier than one already read.

    The first version of write_tidal emitted eb_h1, eb_h2, wb_h1, wb_h2, x1..x3
    in source order. SUMO inserted eb_h1, eb_h2 and wb_h2 and discarded wb_h1
    and all three cross flows -- 1219 vehicles instead of 2300, with no error,
    which is a scenario measuring something other than what it says it does.
    """
    begins = [float(f["begin"]) for f in tidal]
    assert begins == sorted(begins), [(f["id"], f["begin"]) for f in tidal]


def test_tidal_keeps_every_flow(tidal):
    ids = {f["id"].rsplit("_h", 1)[0] for f in tidal}
    assert ids == set(ms.TIDAL_PROFILE)


def test_tidal_cross_street_demand_is_constant(tidal, peak):
    peak_cross = {f["route"]: _rate(f) for f in peak if f["route"].startswith("x")}
    cross = [f for f in tidal if f["route"].startswith("x")]
    assert len(cross) == len(peak_cross), "cross flows are not split by the tide"
    for f in cross:
        assert (float(f["begin"]), float(f["end"])) == (0.0, ms.EPISODE_SECONDS)
        assert _rate(f) == pytest.approx(peak_cross[f["route"]])


def _vehicles(flows, routes) -> float:
    return sum(_rate(f) * (float(f["end"]) - float(f["begin"]))
               for f in flows if f["route"] in routes)


def test_tidal_arterial_load_matches_corridor_peak(peak, tidal):
    """Same arterial vehicles per episode as corridor_peak, pointed differently.

    This is the scenario's controlling comparison: tidal and peak differ in
    STRUCTURE, not in how much traffic there is. An earlier version capped the
    dominant direction at peak's per-approach rate, which quietly cut total
    arterial demand by a third -- tidal then measured as EASIER than peak
    (19.5 s vs 21.4 s for green_wave) and the fixed-vs-reactive margin shrank,
    which is the opposite of what the scenario exists to expose.
    """
    # rel=1e-4 absorbs the 6-decimal rounding of period="exp(rate)"
    assert _vehicles(tidal, {"eb", "wb"}) == pytest.approx(
        _vehicles(peak, {"eb", "wb"}), rel=1e-4)


def test_tidal_dominant_direction_exceeds_an_even_green_split(tidal):
    """The dominant direction must be more than an even split can serve.

    Roughly 1125 veh/h per approach gets through on an even split (2 lanes,
    ~1800 veh/h/lane saturation, ~5/16 duty at the low floors). Below that a
    fixed round-robin serves the tide perfectly well and the scenario cannot
    tell a coordinating controller from an indifferent one.
    """
    dominant = max(_rate(f) for f in tidal if f["route"] == "eb") * 3600
    assert dominant > 1125


def test_tidal_arrivals_are_stochastic(tidal, tmp_path):
    # deterministic vehsPerHour would make every "held-out demand seed" the
    # same arrival pattern -- the defect make_scenarios' docstring documents
    text = (tmp_path / "corridor_tidal.rou.xml").read_text()
    assert "vehsPerHour" not in text
    for f in tidal:
        _rate(f)


def test_tidal_rejects_unknown_flow_id(tmp_path):
    # a silent default would let a new flow in corridor.rou.xml be scaled by
    # the wrong side of the tide without anyone noticing
    src = tmp_path / "corridor.rou.xml"
    src.write_text('<routes><flow id="f_mystery" route="eb" vehsPerHour="100"/></routes>')
    with pytest.raises(KeyError):
        ms.write_tidal(str(src), str(tmp_path / "out.rou.xml"))


def test_tidal_scenario_is_registered():
    ec = pytest.importorskip("env_common")
    assert ec.SCENARIO_ROUTES["corridor_tidal"] == ms.CORRIDOR_TIDAL_DST


def test_corridor_scenarios_are_derived_from_the_route_map():
    """One list of corridor scenarios, so a new one reaches every CLI at once.

    corridor_baseline, analysis/corridor_sweep and compare each had their own
    hard-coded pair; a scenario added to only some of them is a scenario that
    silently never gets calibrated or reported.
    """
    ec = pytest.importorskip("env_common")
    assert set(ec.CORRIDOR_SCENARIOS) == {
        k for k in ec.SCENARIO_ROUTES if k.startswith("corridor_")
    }
    assert "corridor_tidal" in ec.CORRIDOR_SCENARIOS


@pytest.mark.skipif(not NET.exists(), reason="network not built")
def test_route_edges_exist_in_net():
    net = sumolib.net.readNet(str(NET))
    edge_ids = {e.getID() for e in net.getEdges()}
    text = ROU.read_text()
    for eid in re.findall(r'edges="([^"]+)"', text):
        for e in eid.split():
            assert e in edge_ids, f"route references unknown edge {e}"
