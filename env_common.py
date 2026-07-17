"""
Shared environment definition — ONE env for every algorithm.

The project's contribution is an apples-to-apples algorithm comparison, so every
agent (DQN / QR-DQN / PPO / A2C) must train and evaluate on the SAME env, SAME
reward, SAME observation. That invariant lives here; only the algorithm changes.

The observation is PCU-WEIGHTED (motorcycle 0.3, auto 0.5, car 1.0) so the
controller sees passenger-car equivalents instead of raw vehicle counts.
"""

import numpy as np
from gymnasium import spaces

from sumo_rl import SumoEnvironment
from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal

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


# ----------------------------------------------------------------------------
# Safety-aware reward — vulnerability weights (inverse of crash protection)
# Mirrors the PCU idea: the more exposed the rider, the higher the weight.
# ----------------------------------------------------------------------------
VULNERABILITY = {"moto": 1.0, "auto": 0.6, "car": 0.3}
DEFAULT_VULN = 0.3  # unknown type -> treat as a car (least vulnerable)

B_THRESH = 4.5      # m/s^2 : |deceleration| above this counts as an emergency brake (used in safety reward; see spec section 4)
SAFETY_SCALE = 1.0  # calibration constant; set in a later task (see spec section 4)


def _vehicle_vuln(type_id: str) -> float:
    # prefix-match so a distribution suffix (e.g. "moto@0") still resolves
    for name, w in VULNERABILITY.items():
        if type_id == name or type_id.startswith(name):
            return w
    return DEFAULT_VULN


def _internal_lanes(ts) -> list:
    # each signal index may control several connections; the via (internal
    # junction) lane is the 3rd element of each connection tuple
    links = ts.sumo.trafficlight.getControlledLinks(ts.id)
    return list({conn[2] for lk in links if lk for conn in lk if conn and conn[2]})


def _safety_penalty(ts) -> float:
    """Composite, vulnerability-weighted safety penalty for the current step.

    brake_term    : sum of vulnerability over vehicles braking harder than B_THRESH
    exposure_term : sum of vulnerability over vehicles on internal junction lanes
                    while the phase is yellow / clearing
    """
    sumo = ts.sumo

    brake_term = 0.0
    for lane in ts.lanes:
        for vid in sumo.lane.getLastStepVehicleIDs(lane):
            if sumo.vehicle.getAcceleration(vid) < -B_THRESH:
                brake_term += _vehicle_vuln(sumo.vehicle.getTypeID(vid))

    exposure_term = 0.0
    if ts.is_yellow:
        for lane in _internal_lanes(ts):
            for vid in sumo.lane.getLastStepVehicleIDs(lane):
                exposure_term += _vehicle_vuln(sumo.vehicle.getTypeID(vid))

    return brake_term + exposure_term


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
    # vtypes + sublane always; gui-settings only under sumo-gui (plain sumo rejects it)
    extra = "--additional-files vtypes.add.xml --lateral-resolution 0.5"
    if gui:
        extra += " --gui-settings-file gui-settings.xml --start --quit-on-end"
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
        additional_sumo_cmd=extra,
    )
