"""SUMO actually inserts the tidal demand the route file asks for.

tests/test_corridor_routes.py checks what write_tidal emits. This checks what
SUMO does with it, because the failure mode that motivated it is invisible in
the file: SUMO reads a route file in order and silently discards any flow that
begins earlier than one already read. The route file looked correct while a
third of the corridor's demand never entered the network and no error was
raised anywhere.

Run headless SUMO directly rather than through the RL env -- the property under
test is demand insertion, which has nothing to do with the controller.
"""
import collections
import pathlib
import subprocess
import xml.etree.ElementTree as ET

import pytest

import make_scenarios as ms

pytestmark = pytest.mark.slow

sumolib = pytest.importorskip("sumolib")

ROOT = pathlib.Path(__file__).parent.parent
NET = ROOT / "corridor.net.xml"
TIDAL = ROOT / ms.CORRIDOR_TIDAL_DST

# per flow, over a 3600 s episode: 1400 veh/h for half an hour is ~700 trips,
# 700 veh/h is ~350, each cross street is ~300. Assert well below those so the
# test fails on a dropped flow, not on ordinary Poisson variation or on the
# vehicles still in the network when the episode ends (tripinfo omits those).
MIN_TRIPS = 100
# the demand ratio is 2:1; assert only that the tide is visible, since asserting
# the nominal ratio against Poisson noise is a coin flip
MIN_TIDE_RATIO = 1.5


@pytest.mark.skipif(not (NET.exists() and TIDAL.exists()),
                    reason="corridor network / tidal routes not generated")
def test_sumo_inserts_every_tidal_flow(tmp_path):
    trip = tmp_path / "tripinfo.xml"
    subprocess.run(
        [sumolib.checkBinary("sumo"),
         "-n", str(NET), "-r", str(TIDAL),
         "--additional-files", "vtypes.add.xml",
         "--lateral-resolution", "0.5",
         "--time-to-teleport", "300",
         "--tripinfo-output", str(trip),
         "--end", str(int(ms.EPISODE_SECONDS)),
         "--no-step-log"],
        cwd=str(ROOT), check=True, capture_output=True,
    )

    counts = collections.Counter(
        tp.get("id").rsplit(".", 1)[0]           # "f_eb_h1.42" -> "f_eb_h1"
        for tp in ET.parse(trip).getroot().iter("tripinfo")
    )
    expected = {"f_eb_h1", "f_eb_h2", "f_wb_h1", "f_wb_h2",
                "f_x1", "f_x2", "f_x3"}
    assert set(counts) == expected, f"flows missing from the sim: {counts}"
    for flow_id in expected:
        assert counts[flow_id] >= MIN_TRIPS, f"{flow_id} barely ran: {counts}"


@pytest.mark.skipif(not (NET.exists() and TIDAL.exists()),
                    reason="corridor network / tidal routes not generated")
def test_the_tide_actually_turns(tmp_path):
    trip = tmp_path / "tripinfo.xml"
    subprocess.run(
        [sumolib.checkBinary("sumo"),
         "-n", str(NET), "-r", str(TIDAL),
         "--additional-files", "vtypes.add.xml",
         "--lateral-resolution", "0.5",
         "--time-to-teleport", "300",
         "--tripinfo-output", str(trip),
         "--end", str(int(ms.EPISODE_SECONDS)),
         "--no-step-log"],
        cwd=str(ROOT), check=True, capture_output=True,
    )

    # departures per direction per half, straight off depart time -- this is the
    # scenario's defining property, so it is asserted on the simulation rather
    # than on the flow rates that were asked for
    per_half = collections.Counter()
    for tp in ET.parse(trip).getroot().iter("tripinfo"):
        route = tp.get("id").split("_")[1]              # eb / wb / x1..
        if route not in ("eb", "wb"):
            continue
        half = "h1" if float(tp.get("depart")) < ms.TIDAL_SWITCH_S else "h2"
        per_half[(route, half)] += 1

    assert per_half[("eb", "h1")] >= MIN_TIDE_RATIO * per_half[("wb", "h1")], per_half
    assert per_half[("wb", "h2")] >= MIN_TIDE_RATIO * per_half[("eb", "h2")], per_half
