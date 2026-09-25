"""Head-to-head table for the corridor demo, from SUMO's per-completed-trip output.

    python corridor_demo_summary.py [seed] [net_tag]      net_tag e.g. irregular ('' = regular)
"""
import os
import sys
import xml.etree.ElementTree as ET

seed = sys.argv[1] if len(sys.argv) > 1 else "42"
tag = f"_net{sys.argv[2]}" if len(sys.argv) > 2 and sys.argv[2] else ""
names = {"green_wave": "green wave (fixed plan)", "max_pressure": "max-pressure (reactive)",
         "idqn": "IDQN (learned, frozen)"}
print(f"\n=== corridor_peak, 10 s floor, demand seed {seed}{tag or ' (regular 200 m spacing)'}: completed-trip metrics ===")
print(f"{'controller':26s} {'trips done':>10s} {'delay/trip':>11s}")
rows = {}
for c, label in names.items():
    p = f"logs/demo_corridor_{c}_seed{seed}{tag}_tripinfo.xml"
    if not os.path.exists(p):
        continue
    t = ET.parse(p).getroot().findall("tripinfo")
    rows[c] = (len(t), sum(float(x.get("timeLoss")) for x in t) / max(len(t), 1))
    print(f"{label:26s} {rows[c][0]:10d} {rows[c][1]:10.1f}s")
if "green_wave" in rows and "idqn" in rows:
    d = rows["idqn"][1] - rows["green_wave"][1]
    print(f"\nIDQN delay/trip is {abs(d):.1f} s {'HIGHER' if d > 0 else 'LOWER'} than the green wave.")
print("One seed is a sample. Regular spacing, n=10: green wave 13.5 s, max-pressure 26.5 s (analysis/corridor_sweep.csv).\n"
      "Caveat: green wave gets offsets re-fit to each geometry; IDQN is ONE policy trained on the regular net only.\n"
      "Irregular spacing reverses the order in bounded bands (SP9, n=10: IDQN wins 10/10 seeds, p=0.002).")
