"""
RL training skeleton: sumo-rl + Stable-Baselines3.

Baseline of the algorithm ladder = DQN. The state is PCU-WEIGHTED (motorcycle
0.3, auto 0.5, car 1.0) so the controller actually "sees" heterogeneity instead
of raw vehicle counts.

This is the first rung. QR-DQN / PPO / SAC reuse the SAME env + SAME reward +
SAME observation (only the algorithm changes) — that comparison is the project's
contribution. Keep the protocol identical across agents:
    equal training-step budget, 3-5 seeds each, RL-Zoo default hyperparameters,
    evaluate on held-out seeds, report mean +/- std.

Prereqs:
    SUMO_HOME set; pip install sumo-rl stable-baselines3 sumolib traci
    intersection.net.xml built in netedit (Phase 2).

Run:
    python train.py --steps 100000 --seed 0
    python train.py --eval models/dqn_seed0.zip --seed 42   # held-out seed
"""

import argparse
import os
import sys

import numpy as np
from gymnasium import spaces

# sumo-rl
import sumo_rl
from sumo_rl import SumoEnvironment
from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal

import traci

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

# ----------------------------------------------------------------------------
# PCU weights  (keep in sync with vtypes.add.xml vType ids)
# ----------------------------------------------------------------------------
PCU = {"moto": 0.3, "auto": 0.5, "car": 1.0}
DEFAULT_PCU = 1.0  # unknown type -> treat as a car (conservative)


def _vehicle_pcu(type_id: str) -> float:
    # vType ids may carry a distribution suffix (e.g. "moto@..."); match prefix
    for name, w in PCU.items():
        if type_id == name or type_id.startswith(name):
            return w
    return DEFAULT_PCU


class PCUObservationFunction(ObservationFunction):
    """
    Observation per traffic signal:
        [ phase one-hot | min-green-elapsed flag | per-lane PCU density
          | per-lane PCU queue ]

    PCU density = (sum of PCU of vehicles on lane) / (lane capacity in PCU)
    PCU queue   = (sum of PCU of halting vehicles on lane) / (lane capacity)

    Lane capacity in PCU approximated as lane_length / (car_length + min_gap).
    Densities are clipped to [0, 1].
    """

    # reference car footprint for capacity normalisation (length + min_gap)
    CAR_FOOTPRINT = 4.5 + 2.0

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        ts = self.ts
        sumo = ts.sumo  # traci connection sumo-rl hands the signal

        phase_id = [1 if ts.green_phase == i else 0 for i in range(ts.num_green_phases)]
        min_green = [0 if ts.time_since_last_phase_change < ts.min_green + ts.yellow_time else 1]

        density = []
        queue = []
        for lane in ts.lanes:
            cap = max(sumo.lane.getLength(lane) / self.CAR_FOOTPRINT, 1.0)
            veh_ids = sumo.lane.getLastStepVehicleIDs(lane)
            pcu_total = 0.0
            pcu_halt = 0.0
            for vid in veh_ids:
                w = _vehicle_pcu(sumo.vehicle.getTypeID(vid))
                pcu_total += w
                if sumo.vehicle.getSpeed(vid) < 0.1:
                    pcu_halt += w
            density.append(min(pcu_total / cap, 1.0))
            queue.append(min(pcu_halt / cap, 1.0))

        return np.array(phase_id + min_green + density + queue, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        n = self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes)
        return spaces.Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32)


# ----------------------------------------------------------------------------
# Env factory — single shared definition so every algorithm gets identical env
# ----------------------------------------------------------------------------
def make_env(seed: int, gui: bool = False, out_csv: str = None) -> SumoEnvironment:
    return SumoEnvironment(
        net_file="intersection.net.xml",
        route_file="traffic.rou.xml",
        observation_class=PCUObservationFunction,
        use_gui=gui,
        num_seconds=3600,
        delta_time=5,          # seconds between agent decisions
        yellow_time=3,
        min_green=10,
        max_green=60,
        reward_fn="diff-waiting-time",   # SAME reward for all agents (held fixed)
        single_agent=True,              # one TL -> SB3 single-agent API
        sumo_seed=seed,
        out_csv_name=out_csv,
        sumo_warnings=False,
        # vtypes loaded here (no dedicated kwarg in sumo-rl) + sublane / lane-free
        additional_sumo_cmd="--additional-files vtypes.add.xml --lateral-resolution 0.5",
    )


# ----------------------------------------------------------------------------
def train(steps: int, seed: int):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = make_env(seed=seed, gui=False, out_csv=f"logs/dqn_seed{seed}")
    env = Monitor(env)

    # Hyperparameters: start from SB3 RL-Zoo DQN defaults, then tune once.
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
        seed=seed,
        tensorboard_log="logs/tb",
    )
    model.learn(total_timesteps=steps, progress_bar=True)

    path = f"models/dqn_seed{seed}.zip"
    model.save(path)
    env.close()
    print(f"saved {path}")


def evaluate(model_path: str, seed: int, gui: bool):
    env = make_env(seed=seed, gui=gui, out_csv=f"logs/eval_seed{seed}")
    model = DQN.load(model_path)
    obs, _ = env.reset()
    done = False
    total_r = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_r += reward
        done = terminated or truncated
    env.close()
    print(f"eval seed={seed} total_reward={total_r:.1f}  (metrics -> logs/eval_seed{seed}, tripinfo)")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    if "SUMO_HOME" not in os.environ:
        sys.exit("SUMO_HOME not set — see project setup (Phase 1).")

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100_000, help="training timesteps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval", type=str, default=None, help="path to a saved model to evaluate")
    p.add_argument("--gui", action="store_true", help="show sumo-gui during eval")
    args = p.parse_args()

    if args.eval:
        evaluate(args.eval, seed=args.seed, gui=args.gui)
    else:
        train(steps=args.steps, seed=args.seed)
