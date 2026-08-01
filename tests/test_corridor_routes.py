"""Corridor route generation: scaling produces peak/offpeak, and every edge a
route references actually exists in the built network."""
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


@pytest.mark.skipif(not NET.exists(), reason="network not built")
def test_route_edges_exist_in_net():
    net = sumolib.net.readNet(str(NET))
    edge_ids = {e.getID() for e in net.getEdges()}
    text = ROU.read_text()
    for eid in re.findall(r'edges="([^"]+)"', text):
        for e in eid.split():
            assert e in edge_ids, f"route references unknown edge {e}"
