"""SP13c follow-up: build the span=450 arm of the geometry dose-response sweep.

SP13c (docs/FINDINGS_2026-08-27-sp13c-span550.md) found the 1-crossing
(span=400) to 3-crossing (span=700) transition had already happened by
span=550 (its own three crossings sat close to span=700's, not intermediate
toward span=400's single band), bracketing the transition to (400m, 550m].
This adds a fourth span point, 450m (near the low end of that bracket), to
narrow it further: does 450m still show 3 crossings (transition happens
very close to 400m) or does it revert toward span=400's single-crossing
shape (transition is closer to 550m)?

Holds span fixed at 450m (C1@0, C3@450) and rebuilds the same 8 ratio points
every prior span in this series swept: r = 0.50, 0.55, 0.60, 0.65, 0.70,
0.75, 0.80, 0.90. None of the other spans' nets are reusable here (different
span => different absolute C2 position for every r), so all 8 are newly
built -- same node/edge template and netconvert invocation as
build_geometry_sweep_nets_span550.py, only SPAN and the output label change
(corridor_geom450_<c2>.net.xml, so this axis' net files can never collide
with the span=400/550/700 axes').

    python -m analysis.build_geometry_sweep_nets_span450
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

NETCONVERT = os.path.join(REPO, "venv", "bin", "netconvert")

SPAN = 450  # C1 at 0, C3 at this x -- near the low end of SP13c's (400,550] bracket

# Same 8 ratios every prior span in this series used, recomputed here for
# span=450's absolute C2 position (round to the nearest metre).
RATIOS = {0.50: 225, 0.55: 248, 0.60: 270, 0.65: 292, 0.70: 315,
         0.75: 338, 0.80: 360, 0.90: 405}

NOD_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!-- SP13c span=450 geometry dose-response sweep point r={ratio:.2f}
     (analysis/build_geometry_sweep_nets_span450.py). Same 3-signal arterial
     topology as corridor.nod.xml, C1@0/C3=450 fixed (near the low end of
     SP13c's (400m,550m] crossing-transition bracket), only C2's position
     varies: nominal {l1}m/{l2}m. Reuses corridor.edg.xml verbatim (edges
     reference node ids, not coordinates), same as corridor_irregular.nod.xml. -->
<nodes>
    <node id="W"  x="-200" y="0"    type="priority"/>
    <node id="C1" x="0"    y="0"    type="traffic_light"/>
    <node id="C2" x="{c2}"  y="0"    type="traffic_light"/>
    <node id="C3" x="{span}"  y="0"    type="traffic_light"/>
    <node id="E"  x="{e}"  y="0"    type="priority"/>

    <node id="N1" x="0"    y="200"  type="priority"/>
    <node id="S1" x="0"    y="-200" type="priority"/>
    <node id="N2" x="{c2}"  y="200"  type="priority"/>
    <node id="S2" x="{c2}"  y="-200" type="priority"/>
    <node id="N3" x="{span}"  y="200"  type="priority"/>
    <node id="S3" x="{span}"  y="-200" type="priority"/>
</nodes>
"""


def build_one(ratio: float, c2: int, force: bool) -> str:
    """Returns the net_file basename (e.g. 'corridor_geom450_248.net.xml')."""
    label = f"geom450_{c2}"
    nod_path = f"corridor_{label}.nod.xml"
    net_path = f"corridor_{label}.net.xml"
    if force or not os.path.exists(nod_path):
        with open(nod_path, "w") as f:
            f.write(NOD_TEMPLATE.format(ratio=ratio, l1=c2, l2=SPAN - c2,
                                        c2=c2, span=SPAN, e=SPAN + 200))
    if force or not os.path.exists(net_path):
        subprocess.run(
            [NETCONVERT, "--node-files", nod_path, "--edge-files",
             "corridor.edg.xml", "--output-file", net_path,
             "--tls.guess-signals", "true", "--tls.default-type", "static",
             "--no-turnarounds", "true"],
            check=True, cwd=REPO,
        )
        print(f"built {net_path} (r={ratio:.2f}, {c2}m/{SPAN - c2}m nominal, span={SPAN})")
    return net_path


def _verify(net_path: str) -> None:
    import sumolib
    net = sumolib.net.readNet(net_path)
    tls_ids = sorted(t.getID() for t in net.getTrafficLights())
    assert tls_ids == ["C1", "C2", "C3"], f"{net_path}: bad TLS set {tls_ids}"
    edge_ids = {e.getID() for e in net.getEdges()}
    for eid in ["W_C1", "C1_C2", "C2_C3", "C3_E", "E_C3", "C3_C2", "C2_C1", "C1_W"]:
        assert eid in edge_ids, f"{net_path}: missing arterial edge {eid}"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    for ratio, c2 in sorted(RATIOS.items()):
        net_path = build_one(ratio, c2, args.force)
        _verify(net_path)
    print(f"\nbuilt + verified {len(RATIOS)} net files")


if __name__ == "__main__":
    main()
