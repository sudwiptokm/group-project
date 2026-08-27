"""SP15 follow-up: does retuning PPO's hyperparameters for MAPPO's wider
(57-dim) joint critic change the SP15 smoke-test verdict?

SP15 (docs/FINDINGS_2026-08-27-sp15-mappo-smoke.md) found mappo underperforms
ippo at n=1 seed, but reused train_corridor._HP unmodified for both -- those
HPs were tuned for a 19-dim (local-obs) critic, not MAPPO's 57-dim joint one.
That's a live, undiscriminated confound: the null result could be "coordination
doesn't help" or "wrong HPs for the wider critic." This is the "cheap" half of
SP15's own recommendation (retune-and-rerun vs. stop) -- a small manual sweep,
not a full Optuna search (tune.py targets algos.py's single-intersection SB3
algorithms, not this corridor PPO loop, and a search at ~30min/trial would cost
more than this smoke test's own value justifies).

Three variants, all mappo, seed=42, corridor_peak, lam=0.5, min_green=10,
100k steps -- same protocol SP15 used, changing only what a wider critic input
plausibly needs:

  lr_low   lr / 5        smaller, noisier-per-dim gradient signal from the
                          57-dim concatenated input may want a smaller step
  batch64  batch_size x2  larger minibatches average out more of the extra
                          per-agent noise in the joint state before each step
  wide512  hidden (512,512)  more capacity to fit a 3x wider input (this also
                          widens the actor, which is a disclosed asymmetry --
                          ppo_core.ActorCritic ties actor/critic hidden size)

Zero-shot evaluated on both geometries, same as SP15. Baselines (ippo, mappo
default-HP) are read from the tripinfo XMLs SP15 already produced -- not
re-run.

    python -m analysis.mappo_retune_smoke
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd

import train_corridor as tc
from analysis.tripinfo import reduce_tripinfo
from env_common import tripinfo_path

os.environ.setdefault("TIME_TO_TELEPORT", "300")

SCENARIO = "corridor_peak"
LAM = 0.5
SEED = 42
MIN_GREEN = 10
STEPS = 100_000

NETS = {"regular": "corridor.net.xml", "irregular": "corridor_irregular.net.xml"}

VARIANTS = {
    "lr_low": {"lr": tc._HP["lr"] / 5},
    "batch64": {"batch_size": tc._HP["batch_size"] * 2},
    "wide512": {"hidden": (512, 512)},
}

BASELINE_MODELS = {
    "ippo": "models/ippo_corridor_peak_lam05_seed42_mg10_s100000.pt",
    "mappo": "models/mappo_corridor_peak_lam05_seed42_mg10_s100000.pt",
}


def _out_csv(algo: str, tag_suffix: str, net_file: str) -> str:
    """Mirrors train_corridor.evaluate()'s own out_csv naming exactly."""
    tag = tc._tag(SCENARIO, LAM, SEED, MIN_GREEN, STEPS) + tag_suffix
    out_csv = f"logs/eval_{algo}_{tag}"
    if net_file != "corridor.net.xml":
        out_csv += f"_net{net_file.removesuffix('.net.xml').removeprefix('corridor_')}"
    return out_csv


def _delay(algo: str, tag_suffix: str, net_file: str, model_path: str,
           force: bool = False) -> float:
    out_csv = _out_csv(algo, tag_suffix, net_file)
    trip = tripinfo_path(out_csv)
    if force or not os.path.exists(trip):
        tc.evaluate(model_path, SCENARIO, LAM, SEED, MIN_GREEN, STEPS,
                   tripinfo=True, net_file=net_file, tag_suffix=tag_suffix)
    return reduce_tripinfo(trip)["trip_time_loss_mean"]


def run_all(force: bool = False) -> pd.DataFrame:
    rows = []

    for algo, model_path in BASELINE_MODELS.items():
        for geometry, net_file in NETS.items():
            d = _delay(algo, "", net_file, model_path)
            rows.append({"variant": algo, "geometry": geometry, "delay_per_trip": d})
            print(f"[baseline {algo:<5}/{geometry}] delay/trip={d:6.2f}s")

    for name, overrides in VARIANTS.items():
        tag_suffix = f"_retune_{name}"
        model_path = f"models/mappo_{tc._tag(SCENARIO, LAM, SEED, MIN_GREEN, STEPS)}{tag_suffix}.pt"
        if force or not os.path.exists(model_path):
            trained = tc.train(SCENARIO, LAM, SEED, STEPS, MIN_GREEN,
                              centralized=True, hp_overrides=overrides,
                              tag_suffix=tag_suffix)
            assert trained == model_path
        for geometry, net_file in NETS.items():
            d = _delay("mappo", tag_suffix, net_file, model_path, force=force)
            rows.append({"variant": f"mappo_{name}", "geometry": geometry,
                        "delay_per_trip": d})
            print(f"[{name:<7}/{geometry}] delay/trip={d:6.2f}s")

    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    wide = df.pivot_table(index="variant", columns="geometry", values="delay_per_trip")
    order = ["ippo", "mappo"] + [f"mappo_{n}" for n in VARIANTS]
    print("\n=== delay/trip (s), seed 42, corridor_peak, lam=0.5 ===")
    print(wide.reindex(order).to_string(float_format=lambda x: f"{x:6.2f}"))


if __name__ == "__main__":
    df = run_all()
    df.to_csv(os.path.join(REPO, "analysis", "mappo_retune_smoke.csv"), index=False)
    report(df)
