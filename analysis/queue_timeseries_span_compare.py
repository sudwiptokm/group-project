"""SP13c addendum follow-up: the offset/quantization schedule was ruled out
(docs/FINDINGS_2026-08-27-sp13c-span550.md's addendum) as the cause of
span=550's anomalous r=0.50 green_wave baseline (23.15s, worse than both
span=400's 13.46s and span=700's 16.72s). That analysis was pure control-logic
algebra; ruling it out means the real cause has to be emergent SUMO dynamics
this script goes looking for directly -- per-signal queue length and phase
over time, for green_wave on the three symmetric (r=0.50) nets, same seed.

Records, once per decision step (every CORRIDOR_DELTA_TIME=5s), each of
C1/C2/C3's incoming-arterial-lane halting count and current green_phase
index, for corridor.net.xml (span=400), corridor_geom550_275.net.xml
(span=550), corridor_geom700_350.net.xml (span=700). corridor_peak demand,
min_green=10, seed=42 (single seed -- this is a diagnostic look, not a
statistical claim).

    python -m analysis.queue_timeseries_span_compare
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd

import corridor_baseline as cb
from env_common import make_corridor_env

OUT_CSV = os.path.join(REPO, "analysis", "queue_timeseries_span_compare.csv")

SCENARIO = "corridor_peak"
SEED = 42
MIN_GREEN = 10

# ts_id -> incoming arterial edge (the approach a platoon released upstream
# arrives on).
INCOMING_EDGE = {"C1": "W_C1", "C2": "C1_C2", "C3": "C2_C3"}

# Downstream of C3 -- checked separately to test whether C3's queue is a
# spillback symptom (blocked by congestion further downstream) rather than
# an arrival/discharge problem local to C3 itself.
DOWNSTREAM_EDGE = "C3_E"

NETS = {
    400: "corridor.net.xml",
    550: "corridor_geom550_275.net.xml",
    700: "corridor_geom700_350.net.xml",
}

os.environ.setdefault("TIME_TO_TELEPORT", "300")  # match geometry_sweep_*.py's protocol


def run_one(span: int, net_file: str) -> pd.DataFrame:
    env = make_corridor_env(seed=SEED, scenario=SCENARIO, lam=0.0,
                            min_green=MIN_GREEN, tripinfo=False,
                            net_file=net_file)
    env.reset()
    lanes = {}
    for ts_id, edge in INCOMING_EDGE.items():
        ts = env.traffic_signals[ts_id]
        n_lanes = ts.sumo.edge.getLaneNumber(edge)
        lanes[ts_id] = [f"{edge}_{i}" for i in range(n_lanes)]

    c3 = env.traffic_signals["C3"]
    n_down_lanes = c3.sumo.edge.getLaneNumber(DOWNSTREAM_EDGE)
    down_lanes = [f"{DOWNSTREAM_EDGE}_{i}" for i in range(n_down_lanes)]

    rows = []
    done = False
    while not done:
        actions = cb.green_wave_actions(env, net_file=net_file)
        t = env.sim_step
        for ts_id in ("C1", "C2", "C3"):
            ts = env.traffic_signals[ts_id]
            halting = sum(ts.sumo.lane.getLastStepHaltingNumber(l)
                          for l in lanes[ts_id])
            vehicles = sum(ts.sumo.lane.getLastStepVehicleNumber(l)
                          for l in lanes[ts_id])
            rows.append({"span": span, "t": t, "ts": ts_id,
                         "phase": ts.green_phase, "halting": halting,
                         "vehicles": vehicles})
        down_halting = sum(c3.sumo.lane.getLastStepHaltingNumber(l) for l in down_lanes)
        down_vehicles = sum(c3.sumo.lane.getLastStepVehicleNumber(l) for l in down_lanes)
        rows.append({"span": span, "t": t, "ts": "C3_E", "phase": -1,
                     "halting": down_halting, "vehicles": down_vehicles})
        _, _, dones, _ = env.step(actions)
        done = dones["__all__"]
    env.close()
    return pd.DataFrame(rows)


def main():
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    dfs = [run_one(span, net_file) for span, net_file in NETS.items()]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(df)} rows)")

    print("\n=== per-signal halting-count summary by span ===")
    print(df.groupby(["span", "ts"])["halting"]
            .agg(["mean", "max", lambda s: (s >= 5).mean() * 100])
            .rename(columns={"<lambda_0>": "pct_t_ge_5"})
            .to_string(float_format=lambda x: f"{x:7.2f}"))

    print("\n=== C3 halting by phase, and downstream C3_E occupancy, by span ===")
    c3 = df[df.ts == "C3"]
    print(c3.groupby(["span", "phase"])["halting"].agg(["mean", "max", "count"])
            .to_string(float_format=lambda x: f"{x:7.2f}"))
    down = df[df.ts == "C3_E"]
    print(down.groupby("span")[["halting", "vehicles"]].agg(["mean", "max"])
             .to_string(float_format=lambda x: f"{x:7.2f}"))


if __name__ == "__main__":
    main()
