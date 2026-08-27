"""SP13/SP13b/SP13c/SP13d follow-up: build the r<0.50 (short block first) arm
of the geometry dose-response sweep, at span=400 (SP13's original axis).

Every doc in this series so far -- SP13 (span=400), SP13b (span=700), SP13c
(span=550), SP13d (span=450) -- swept r in [0.50, 0.90], starting at the
symmetric net and only lengthening the C1-C2 block relative to C2-C3. r<0.50
(C1-C2 shorter than C2-C3, i.e. the short block comes FIRST rather than
second) was flagged as untested in every one of those docs. This fills that
gap at span=400 specifically -- the other three spans' low-r arms are still
open (see docs/HANDOFF_2026-08-27.md).

Mirrors the original 8-point high-r sweep's spacing around r=0.50: dense
steps from 0.50 down to 0.30, then sparser to 0.10 (mirroring the original's
0.50-0.80 dense / 0.90 sparse pattern). r=0.50 itself is corridor.net.xml,
reused verbatim (build_geometry_sweep_nets.py already established this
convention). This script only builds the 7 new low-r points: r in {0.45,
0.40, 0.35, 0.30, 0.25, 0.20, 0.10}.

Same node/edge template and netconvert invocation as
build_geometry_sweep_nets.py, only C2's position moves below the 200m
midpoint instead of above it. Output labelled corridor_geomlo<c2>.net.xml
(distinct from the original high-r sweep's corridor_geom<c2>.net.xml, even
though both are span=400 -- no c2 value collides between the two ranges, but
the explicit 'lo' tag keeps the two axes visually distinguishable in
directory listings).

    python -m analysis.build_geometry_sweep_nets_lowr
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

NETCONVERT = os.path.join(REPO, "venv", "bin", "netconvert")

SPAN = 400  # C1 at 0, C3 at this x -- SP13's original span

# ratio -> nominal C1-C2 length (= C2's x coordinate). Excludes r=0.50
# (corridor.net.xml, reused as-is by analysis/geometry_sweep_lowr.py).
RATIOS = {0.45: 180, 0.40: 160, 0.35: 140, 0.30: 120, 0.25: 100,
         0.20: 80, 0.10: 40}

NOD_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!-- SP13e low-ratio (short block first) geometry dose-response sweep point
     r={ratio:.2f} (analysis/build_geometry_sweep_nets_lowr.py). Same 3-signal
     arterial topology as corridor.nod.xml, C1@0/C3@400 fixed (SP13's
     original span), only C2's position varies: nominal {l1}m/{l2}m, with
     the SHORT block now first (C1-C2) rather than second, mirroring the
     original high-r sweep's convention in the opposite direction. Reuses
     corridor.edg.xml verbatim (edges reference node ids, not coordinates). -->
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
    """Returns the net_file basename (e.g. 'corridor_geomlo180.net.xml')."""
    label = f"geomlo{c2}"
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

    for ratio, c2 in sorted(RATIOS.items(), reverse=True):
        net_path = build_one(ratio, c2, args.force)
        _verify(net_path)
    print(f"\nbuilt + verified {len(RATIOS)} net files")


if __name__ == "__main__":
    main()
