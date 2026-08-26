"""Build the SP13 geometry dose-response sweep's network files.

Fix 3 of the three MSc-strengthening gaps identified after SP12: SP8-SP10
established that idqn beats green_wave on irregular spacing but only sampled
3 discrete hand-built geometries (regular 200m/200m, and SP8/SP10's 600m/100m,
100m/600m, 300m/100m variants -- the last two of which used a DIFFERENT total
arterial span, 700m, than the regular net's 400m, confounding spacing
asymmetry with overall corridor length). This turns "does it generalize" into
an actual measured curve: hold the total C1-to-C3 span fixed at 400m (same as
corridor.net.xml) and slide C2 along it, so only the asymmetry ratio varies.

ratio r = nominal C1-C2 length / 400 total span. r=0.50 is corridor.net.xml
itself (regular, no asymmetry). r=0.75 is corridor_irregular3.net.xml (SP10),
reused verbatim -- it already sits on this same span=400 axis. This script
only builds the 6 new points: r in {0.55, 0.60, 0.65, 0.70, 0.80, 0.90},
denser in [0.50, 0.75] since SP10's irregular3 (r=0.75) already shows idqn
ahead while regular (r=0.50) shows green_wave ahead -- the flip threshold
lies somewhere in that gap.

Each node file is generated from the same template as
corridor_irregular3.nod.xml (C1@0, C3@400 fixed, C2 slides), reuses
corridor.edg.xml verbatim (edges reference node ids, not coordinates), and is
built into a .net.xml with the exact netconvert invocation from
docs/superpowers/plans/2026-08-01-sp1-corridor-env.md Step 6.

    python -m analysis.build_geometry_sweep_nets
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

NETCONVERT = os.path.join(REPO, "venv", "bin", "netconvert")

SPAN = 400  # C1 at 0, C3 at this x -- fixed across every sweep point

# ratio -> nominal C1-C2 length (= C2's x coordinate). Excludes r=0.50
# (corridor.net.xml) and r=0.75 (corridor_irregular3.net.xml), both reused
# as-is by analysis/geometry_sweep.py.
NEW_RATIOS = {
    0.55: 220,
    0.60: 240,
    0.65: 260,
    0.70: 280,
    0.80: 320,
    0.90: 360,
}

NOD_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!-- SP13 geometry dose-response sweep point r={ratio:.2f} (analysis/
     build_geometry_sweep_nets.py). Same 3-signal arterial topology as
     corridor.nod.xml, C1@0/C3@400 fixed (same total span), only C2's
     position varies: nominal {l1}m/{l2}m instead of the regular 200m/200m.
     Reuses corridor.edg.xml verbatim (edges reference node ids, not
     coordinates), same as corridor_irregular3.nod.xml. -->
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
    """Returns the net_file basename (e.g. 'corridor_geom220.net.xml')."""
    label = f"geom{c2}"
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
        print(f"built {net_path} (r={ratio:.2f}, {c2}m/{SPAN - c2}m nominal)")
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

    for ratio, c2 in sorted(NEW_RATIOS.items()):
        net_path = build_one(ratio, c2, args.force)
        _verify(net_path)
    print(f"\nbuilt + verified {len(NEW_RATIOS)} net files")


if __name__ == "__main__":
    main()
