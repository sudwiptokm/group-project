"""Per-vehicle-type (moto/auto/car) delay breakdown, from existing tripinfo
logs -- no new simulation. Answers: does each controller's delay advantage
hold up the same way across the heterogeneous vehicle mix, or is it carried
by one vehicle type?

Uses corridor_peak, regular net, project-default settings (lambda=0.5 for
idqn, min_green=10), the 3-5 seeds already evaluated for each controller.

Usage: python -m analysis.heterogeneity_breakdown
"""
import glob
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd

FILES = {
    "green_wave": "logs/eval_green_wave_corridor_peak_seed{seed}_mg10_tripinfo.xml",
    "max_pressure": "logs/eval_max_pressure_corridor_peak_seed{seed}_mg10_tripinfo.xml",
    "idqn": "logs/eval_idqn_corridor_peak_lam05_seed{seed}_mg10_s100000_tripinfo.xml",
}
SEEDS = {"green_wave": range(42, 47), "max_pressure": range(42, 47), "idqn": range(42, 45)}


def load(path):
    rows = []
    for tripinfo in ET.parse(path).getroot().iter("tripinfo"):
        rows.append((tripinfo.get("vType"), float(tripinfo.get("timeLoss"))))
    return rows


def main():
    records = []
    for controller, pattern in FILES.items():
        for seed in SEEDS[controller]:
            path = pattern.format(seed=seed)
            for vtype, time_loss in load(path):
                records.append((controller, seed, vtype, time_loss))

    df = pd.DataFrame(records, columns=["controller", "seed", "vtype", "time_loss"])
    df.to_csv("analysis/heterogeneity_breakdown_raw.csv", index=False)

    # mean delay per trip, per controller, per vehicle type, averaged over seeds
    per_seed = df.groupby(["controller", "vtype", "seed"])["time_loss"].mean().reset_index()
    summary = per_seed.groupby(["controller", "vtype"])["time_loss"].agg(["mean", "std", "count"])
    summary = summary.rename(columns={"count": "n_seeds"})
    print(summary.round(2))
    summary.round(3).to_csv("analysis/heterogeneity_breakdown.csv")

    # share of vehicles by type (sanity check the mix is as documented)
    counts = df.groupby(["controller", "vtype"]).size()
    shares = counts / counts.groupby(level=0).sum()
    print("\nvehicle-type share of completed trips:")
    print((shares * 100).round(1))


if __name__ == "__main__":
    main()
