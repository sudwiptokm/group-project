"""The built corridor network must load and expose exactly 3 traffic lights
wired along a continuous arterial."""
import pathlib

import pytest

sumolib = pytest.importorskip("sumolib")

# absolute path so the guard/readNet work regardless of pytest's invocation dir
NET = pathlib.Path(__file__).parent.parent / "corridor.net.xml"


@pytest.mark.skipif(not NET.exists(), reason="corridor.net.xml not built yet")
def test_three_traffic_lights():
    net = sumolib.net.readNet(str(NET))
    tls_ids = sorted(t.getID() for t in net.getTrafficLights())
    assert tls_ids == ["C1", "C2", "C3"]


@pytest.mark.skipif(not NET.exists(), reason="corridor.net.xml not built yet")
def test_arterial_edges_present():
    net = sumolib.net.readNet(str(NET))
    edge_ids = {e.getID() for e in net.getEdges()}
    for eid in ["W_C1", "C1_C2", "C2_C3", "C3_E", "E_C3", "C3_C2", "C2_C1", "C1_W"]:
        assert eid in edge_ids, f"missing arterial edge {eid}"
