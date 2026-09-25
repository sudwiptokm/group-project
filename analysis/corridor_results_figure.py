"""Results-only corridor figure: regular-spacing bars + irregular-spacing reversal.

Sources (all delay per completed trip, s; corridor_peak, min_green=10):
  green_wave / max_pressure regular: analysis/corridor_sweep.csv (n=10, computed here)
  idqn 16.56 (SP13 r=0.50, n=3), ippo 17.85 (SP4 100k check, n=3),
  mappo 18.26 (SP15, n=1 seed 42)  -- docs/PROJECT_REPORT.md
  panel B: docs/PROJECT_REPORT.md Section 5.14 table (n=3 per point)
"""
import csv, collections, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = collections.defaultdict(list)
for r in csv.DictReader(open("analysis/corridor_sweep.csv")):
    if r["scenario"] == "corridor_peak" and r["min_green"] == "10":
        d[r["controller"]].append(float(r["delay_per_trip"]))
bars = [("green_wave\n(n=10)", st.mean(d["green_wave"]), "#1b7837"),
        ("IDQN\n(n=3)", 16.56, "#2166ac"),
        ("IPPO\n(n=3)", 17.85, "#7fa7d0"),
        ("MAPPO\n(n=1)", 18.26, "#b3c9e3"),
        ("max_pressure\n(n=10)", st.mean(d["max_pressure"]), "#a6451f")]

r = [.1,.2,.25,.3,.35,.45,.5,.55,.6,.7,.75,.8,.9]
gw = [17.69,23.11,22.13,21.46,15.88,13.59,13.45,31.17,31.02,25.05,20.62,17.05,14.10]
iq = [17.87,17.19,16.93,17.12,16.88,16.97,16.56,16.92,17.01,17.10,17.26,17.31,17.84]
mp = [26.40,22.58,21.32,20.61,23.82,25.19,25.83,25.23,24.40,20.68,21.33,22.93,25.01]

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
a.bar([x[0] for x in bars], [x[1] for x in bars], color=[x[2] for x in bars])
for i, x in enumerate(bars):
    a.text(i, x[1] + .3, f"{x[1]:.1f}", ha="center", fontsize=10)
a.set_ylabel("delay per completed trip (s)")
a.set_title("A. Regular 200 m spacing: fixed green wave wins")
a.set_ylim(0, 30)
b.plot(r, gw, "o-", color="#1b7837", label="green_wave (offsets re-fit per geometry)")
b.plot(r, iq, "o-", color="#2166ac", label="IDQN (one frozen policy)")
b.plot(r, mp, "o-", color="#a6451f", label="max_pressure")
b.axvspan(.1, .34, color="0.9"); b.axvspan(.51, .8, color="0.9")
b.set_xlabel("asymmetry ratio r (C1-C2 length / 400 m span; r=0.5 regular)")
b.set_title("B. Irregular spacing: IDQN beats green_wave in shaded bands (n=3/pt)")
b.set_ylim(12, 38)
b.legend(fontsize=8, loc="upper center", ncol=1)
fig.suptitle("Corridor, peak demand, 10 s floor", y=1.0)
fig.tight_layout()
fig.savefig("docs/figures/corridor_results.png", dpi=150)
print([(x[0].replace("\n", " "), round(x[1], 2)) for x in bars])
