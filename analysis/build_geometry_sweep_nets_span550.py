"""SP13b follow-up: build the span=550 arm of the geometry dose-response sweep.

SP13 (docs/FINDINGS_2026-08-26-sp13-geometry-dose-response.md) found
green_wave's failure is a bounded band [r~0.51, r~0.80] on a span=400 axis.
SP13b (docs/FINDINGS_2026-08-27-sp13b-span700-confound.md) reran the same 8
ratios at span=700 and found the band's shape is NOT ratio-only: span=700 has
3 crossings vs span=400's 1. With only two span points sampled, whether the
crossing count/location moves monotonically with span or does something else
is unresolved -- this adds a third point (550m, midway between the two) to
start answering that.

Holds span fixed at 550m (C1@0, C3@550) and rebuilds the same 8 ratio points
SP13/SP13b swept: r = 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90. None of
the span=400 or span=700 sweeps' nets are reusable here (different span =>
different absolute C2 position for every r), so all 8 are newly built -- same
node/edge template and netconvert invocation as
build_geometry_sweep_nets_span700.py, only SPAN and the output label change
(corridor_geom550_<c2>.net.xml, so this axis' net files can never collide
with the span=400 or span=700 axes').

    python -m analysis.build_geometry_sweep_nets_span550
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

NETCONVERT = os.path.join(REPO, "venv", "bin", "netconvert")

SPAN = 550  # C1 at 0, C3 at this x -- midway between SP13's 400 and SP13b's 700

# Same 8 ratios build_geometry_sweep_nets.py / _span700.py used, recomputed
# here for span=550's absolute C2 position (round to the nearest metre).
RATIOS = {0.50: 275, 0.55: 302, 0.60: 330, 0.65: 358, 0.70: 385,
         0.75: 412, 0.80: 440, 0.90: 495}

NOD_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!-- SP13b span=550 geometry dose-response sweep point r={ratio:.2f}
     (analysis/build_geometry_sweep_nets_span550.py). Same 3-signal arterial
     topology as corridor.nod.xml, C1@0/C3=550 fixed (midway between SP13's
     400m and SP13b's 700m), only C2's position varies: nominal {l1}m/{l2}m.
     Reuses corridor.edg.xml verbatim (edges reference node ids, not
     coordinates), same as corridor_irregular.nod.xml. -->
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
    """Returns the net_file basename (e.g. 'corridor_geom550_302.net.xml')."""
    label = f"geom550_{c2}"
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
