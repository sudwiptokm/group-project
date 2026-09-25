"""Print the head-to-head table for the side-by-side demo from the tripinfo files.

    python demo_summary.py [seed] [algo] [scenario]

Reads SUMO's per-completed-trip output (the metric that cannot be gamed by
holding vehicles back), not the step CSV.
"""
import sys
import xml.etree.ElementTree as ET

seed = sys.argv[1] if len(sys.argv) > 1 else "42"
algo = sys.argv[2] if len(sys.argv) > 2 else "dqn"
scen = sys.argv[3] if len(sys.argv) > 3 else "peak"
mg = sys.argv[4] if len(sys.argv) > 4 else "60"

runs = {
    "fixed-time (60 s green)": f"logs/eval_fixedtime_{scen}_seed{seed}_g60_tripinfo.xml",
    f"actuated (no learning)": f"analysis/actuated_logs/{scen}_mg{mg}_seed{seed}_tripinfo.xml",
    f"{algo.upper()} (RL)": f"logs/eval_{algo}_{scen}_lam00_seed{seed}_mg{mg}_tripinfo.xml",
}


def stats(path):
    trips = ET.parse(path).getroot().findall("tripinfo")
    n = len(trips)
    f = lambda k, ts=trips: sum(float(t.get(k)) for t in ts) / max(len(ts), 1)
    by = {}
    for t in trips:
        by.setdefault(t.get("vType", "?").split("@")[0], []).append(t)
    return n, f("timeLoss"), f("waitingTime"), {k: f("timeLoss", v) for k, v in by.items()}


import os
rows = {name: stats(p) for name, p in runs.items() if os.path.exists(p)}
print(f"\n=== {scen} demand, demand seed {seed}: completed-trip metrics ===")
print(f"{'controller':26s} {'trips done':>10s} {'delay/trip':>11s} {'wait/trip':>10s}   delay by vehicle type (s)")
for name, (n, tl, wt, by) in rows.items():
    bt = "  ".join(f"{k} {v:5.1f}" for k, v in sorted(by.items()))
    print(f"{name:26s} {n:10d} {tl:10.1f}s {wt:9.1f}s   {bt}")
(a, ra), (b, rb) = list(rows.items())[0], list(rows.items())[-1]
d = rb[1] - ra[1]
print(f"\nRL delay/trip is {abs(d):.1f} s {'HIGHER' if d > 0 else 'LOWER'} than fixed-time "
      f"({d / ra[1] * 100:+.0f}%).")
dn = rb[0] - ra[0]
print(f"RL completed {abs(dn)} {'MORE' if dn > 0 else 'FEWER'} trips ({dn / ra[0] * 100:+.0f}%) in the same episode.")
print("Delay/trip only averages trips that FINISHED. A controller that strands traffic looks fast on\n"
      "the survivors, so read delay together with trips done: throughput is the fair scoreboard.")
print("One seed is a sample; multi-seed paired tests: analysis/headline_stats.csv, docs/PROJECT_REPORT.md.")
