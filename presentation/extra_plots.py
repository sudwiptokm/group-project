"""Extra presentation plots from logs/comparison.csv (source of truth).

Writes to results/:
  bars_offpeak_lam05_logy.png  — off-peak waiting, log-y so small bars are readable
  speed_offpeak_lam05.png      — off-peak mean speed (higher = better); shows a2c collapse
  improvement_peak_lam05.png   — peak % waiting reduction vs fixed-time
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
rows = list(csv.DictReader(open(os.path.join(ROOT, "logs", "comparison.csv"))))


def cell(scenario, algo, col):
    for r in rows:
        if r["scenario"] == scenario and r["algo"] == algo and r["lam"] in ("05", "na"):
            v = r.get(col, "")
            return float(v) if v not in ("", None) else None
    return None


ORANGE, BLUE = "#e8820c", "#2f6fd0"

# 1) off-peak waiting, log-y ------------------------------------------------
order = ["fixedtime", "dqn", "ppo", "qrdqn", "a2c"]
wait = [cell("offpeak", a, "system_mean_waiting_time_mean") for a in order]
err = [cell("offpeak", a, "system_mean_waiting_time_std") or 0 for a in order]
colors = [ORANGE if a == "fixedtime" else BLUE for a in order]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(order, wait, yerr=err, color=colors, capsize=4)
ax.set_yscale("log")
ax.set_ylabel("mean waiting time (s) — log scale, lower is better")
ax.set_title("off-peak · λ=0.5 — waiting time (log scale)")
for i, v in enumerate(wait):
    ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bars_offpeak_lam05_logy.png"), dpi=130)
plt.close(fig)

# 2) off-peak mean speed ----------------------------------------------------
spd = [cell("offpeak", a, "system_mean_speed_mean") for a in order]
serr = [cell("offpeak", a, "system_mean_speed_std") or 0 for a in order]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(order, spd, yerr=serr, color=colors, capsize=4)
ax.set_ylabel("mean speed (m/s) — higher is better")
ax.set_title("off-peak · λ=0.5 — mobility (mean speed)")
for i, v in enumerate(spd):
    ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "speed_offpeak_lam05.png"), dpi=130)
plt.close(fig)

# 3) peak % improvement vs fixed-time --------------------------------------
base = cell("peak", "fixedtime", "system_mean_waiting_time_mean")
algos = ["dqn", "a2c", "ppo", "qrdqn"]
imp = [(base - cell("peak", a, "system_mean_waiting_time_mean")) / base * 100 for a in algos]
colors2 = ["#1a7f37" if v > 0 else "#b42318" for v in imp]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(algos, imp, color=colors2)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("% waiting-time reduction vs fixed-time")
ax.set_title("peak · λ=0.5 — improvement over fixed-time (higher = better)")
for i, v in enumerate(imp):
    ax.text(i, v, f"{v:+.1f}%", ha="center", va="bottom" if v > 0 else "top", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "improvement_peak_lam05.png"), dpi=130)
plt.close(fig)

print("wrote 3 plots to results/")
