"""
Shared environment definition — ONE env for every algorithm.

The project's contribution is an apples-to-apples algorithm comparison, so every
agent (DQN / QR-DQN / PPO / A2C) must train and evaluate on the SAME env, SAME
reward, SAME observation. That invariant lives here; only the algorithm changes.

The observation is PCU-WEIGHTED (motorcycle 0.3, auto 0.5, car 1.0) so the
controller sees passenger-car equivalents instead of raw vehicle counts.
"""

import os
from typing import Optional

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
# calibrated: mean_safety/mean|eff| on peak seed0 (17.97/8.44); see spec section 4.
# Recalibrated after the accumulator fix — the pre-fix value (0.024) was measured
# against a safety signal that sampled one second in five and never saw yellow.
SAFETY_SCALE = 2.1298


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


def _safety_components(ts, internal_lanes=None):
    """Return (brake_term, exposure_term), the two vulnerability-weighted safety
    sub-terms **for the current simulation second**.

    brake_term    : sum of vulnerability over vehicles braking harder than B_THRESH
    exposure_term : sum of vulnerability over vehicles on internal junction lanes
                    while the phase is yellow / clearing

    `internal_lanes` may be supplied by a caller that has already cached the
    (static) junction topology, to avoid a TraCI round-trip every second.

    This is an instantaneous sample. Both sub-terms are only meaningful when
    accumulated over every second of a decision window — see _SafetyWindow.
    """
    sumo = ts.sumo

    brake_term = 0.0
    for lane in ts.lanes:
        for vid in sumo.lane.getLastStepVehicleIDs(lane):
            if sumo.vehicle.getAcceleration(vid) < -B_THRESH:
                brake_term += _vehicle_vuln(sumo.vehicle.getTypeID(vid))

    exposure_term = 0.0
    if ts.is_yellow:
        lanes = _internal_lanes(ts) if internal_lanes is None else internal_lanes
        for lane in lanes:
            for vid in sumo.lane.getLastStepVehicleIDs(lane):
                exposure_term += _vehicle_vuln(sumo.vehicle.getTypeID(vid))

    return brake_term, exposure_term


def _safety_penalty(ts) -> float:
    """Instantaneous composite penalty = brake_term + exposure_term."""
    brake_term, exposure_term = _safety_components(ts)
    return brake_term + exposure_term


class _SafetyWindow:
    """Accumulates the safety sub-terms over every second of a decision window.

    sumo-rl only computes rewards and info at action steps, i.e. after the
    simulation has already advanced `delta_time` seconds (env.step -> _run_steps).
    Sampling the safety terms at that moment is wrong twice over:

      * exposure is dead code. `set_next_phase` raises `is_yellow`, and
        `TrafficSignal.update` lowers it again after `yellow_time` seconds --
        always before the window ends, because sumo-rl asserts
        `delta_time > yellow_time`. So `is_yellow` is False at every point the
        reward is evaluated and the exposure term can never fire.
      * braking is undersampled by a factor of `delta_time`: only vehicles
        decelerating hard in the single final second are ever seen.

    The window therefore samples on each simulation second and is read (not
    drained) at the action step, so the reward and the logged metrics see the
    same totals. Junction topology is static, so internal lanes are cached per
    signal id rather than re-queried every second.
    """

    def __init__(self):
        self._acc = {}          # ts_id -> [brake, exposure] for the current window
        self._internal = {}     # ts_id -> internal lanes (cached across windows)

    def reset(self) -> None:
        """Start a new decision window. Topology cache is kept."""
        self._acc = {}

    def _internal_lanes_for(self, ts) -> list:
        if ts.id not in self._internal:
            self._internal[ts.id] = _internal_lanes(ts)
        return self._internal[ts.id]

    def accumulate(self, ts) -> None:
        """Add this simulation second's safety sample for one traffic signal."""
        brake, exposure = _safety_components(
            ts, internal_lanes=self._internal_lanes_for(ts)
        )
        cur = self._acc.setdefault(ts.id, [0.0, 0.0])
        cur[0] += brake
        cur[1] += exposure

    def for_ts(self, ts_id: str):
        """(brake, exposure) accumulated this window for one signal."""
        brake, exposure = self._acc.get(ts_id, (0.0, 0.0))
        return (brake, exposure)

    def totals(self):
        """(brake, exposure) accumulated this window, summed over all signals."""
        brake = sum(v[0] for v in self._acc.values())
        exposure = sum(v[1] for v in self._acc.values())
        return (brake, exposure)


def _step_safety_penalty(ts) -> float:
    """Safety penalty over the decision window that just elapsed.

    Falls back to the instantaneous sample when the signal's environment keeps
    no window (a plain SumoEnvironment, or a bare test stub).
    """
    window = getattr(getattr(ts, "env", None), "_safety_window", None)
    if window is None:
        return _safety_penalty(ts)
    brake, exposure = window.for_ts(ts.id)
    return brake + exposure


def _efficiency(ts) -> float:
    # sumo-rl's built-in diff-waiting-time reward; manages ts.last_measure state
    return TrafficSignal._diff_waiting_time_reward(ts)


def make_safety_reward_fn(lam: float, scale: Optional[float] = None):
    """Return a sumo-rl reward_fn(ts) = diff_waiting_time - lam * safety/scale.

    At lam == 0 the safety term is not even computed (pure efficiency, and an
    exact ablation baseline). `scale` defaults to the calibrated SAFETY_SCALE,
    resolved at call time so a later re-calibration of SAFETY_SCALE is honoured.
    """
    def reward_fn(ts):
        eff = _efficiency(ts)
        if lam == 0.0:
            return eff
        s = SAFETY_SCALE if scale is None else scale
        if s == 0:
            raise ValueError("safety scale must be non-zero")
        return eff - lam * (_step_safety_penalty(ts) / s)

    reward_fn.__name__ = f"safety_reward_lam{lam}"
    return reward_fn


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

def tripinfo_path(out_csv: str) -> str:
    """Where the tripinfo XML for a run whose step CSV stem is `out_csv` lives.

    One function so the writer (make_env) and the readers (analysis/tripinfo.py,
    compare.py) cannot drift apart. Note the stem carries no _conn/_ep suffix:
    a re-run of the same configuration overwrites its own tripinfo, exactly as
    it overwrites its own step CSV.
    """
    return f"{out_csv}_tripinfo.xml"


# The floor a green must reach before a switch request is honoured. 60 s is
# measured, not chosen: analysis/actuated.py sweeps it, and a controller with
# perfect queue information and nothing to learn is 5.6x worse than a fixed plan
# at 10 s, matches it at 60 s, and gets worse again by 90 s. Every peak training
# run this project did used 10 s because that was the default and no driver
# could override it -- which is why the peak null says nothing about RL.
DEFAULT_MIN_GREEN = 60


def resolve_min_green(min_green: int = None) -> int:
    """Explicit argument wins, else $MIN_GREEN, else DEFAULT_MIN_GREEN.

    Same precedence as TIME_TO_TELEPORT and EPISODE_SECONDS, so a sweep script
    can set the floor once for a whole grid. `is None` rather than falsiness:
    min_green=0 is a legitimate "never clamp" and must not read as unset.
    """
    if min_green is not None:
        return min_green
    return int(os.environ.get("MIN_GREEN", DEFAULT_MIN_GREEN))


def eval_csv_stem(algo: str, tag: str, seed: int, train_seed: int = None,
                  min_green: int = None) -> str:
    """Path stem for one evaluation run's step CSV.

    Every fragment goes AFTER `seed<n>`, because compare.py finds runs with
    `eval_<algo>_<scenario>_lam<lam>_seed*.csv` -- anything inserted before the
    seed drops the algorithm out of the comparison table silently.

    `seed` is the DEMAND seed. `train_seed` names the checkpoint, so several
    training seeds evaluated on one demand seed do not overwrite each other.
    `min_green` records the action space the run used: two floors are two
    different experiments, and averaging them into one row is the same class of
    confound as the seed mismatch that flipped the Stage-1 headline. Omitted
    when unspecified so pre-existing run names are unchanged.
    """
    stem = f"logs/eval_{algo}_{tag}_seed{seed}"
    if train_seed is not None:
        stem += f"_t{train_seed}"
    if min_green is not None:
        stem += f"_mg{min_green}"
    return stem


def model_path(algo: str, tag: str, seed: int, min_green: int = None) -> str:
    """Where a trained checkpoint lives.

    Carries `min_green` for the same reason the eval CSV does: a checkpoint is a
    policy FOR an action space, so two floors trained on the same algo/scenario/
    lam/seed are two different models and must not resolve to one filename.
    Omitted when unspecified, so checkpoints trained before the floor was
    settable stay loadable under the names they already have.
    """
    stem = f"models/{algo}_{tag}_seed{seed}"
    if min_green is not None:
        stem += f"_mg{min_green}"
    return f"{stem}.zip"


# route file per demand scenario (generated by make_scenarios.py)
SCENARIO_ROUTES = {
    "base": "traffic.rou.xml",
    "peak": "traffic_peak.rou.xml",
    "offpeak": "traffic_offpeak.rou.xml",
}


class SafetyLoggingEnv(SumoEnvironment):
    """SumoEnvironment that also records the raw safety sub-terms each step.

    Adds three columns to the metrics CSV (summed over all traffic signals):
        system_safety_brake     vulnerability-weighted emergency-braking events
        system_safety_exposure  vulnerability-weighted intersection exposure
        system_safety_total     brake + exposure (the raw, unscaled safety penalty)

    These are the actual safety quantities the reward penalises — logged so
    compare.py / plots.py can report them instead of a proxy like stopped count.

    Both the reward and these metrics read the same per-window accumulator
    (_SafetyWindow), sampled every simulation second rather than only at action
    steps — see _SafetyWindow for why the action-step sample is unusable.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._safety_window = _SafetyWindow()

    def step(self, action):
        # a new decision window begins; the totals read after super().step()
        # cover exactly the seconds this action was in force
        self._safety_window.reset()
        return super().step(action)

    def _sumo_step(self):
        super()._sumo_step()
        # traffic_signals do not exist yet during the reset that starts SUMO
        for ts in getattr(self, "traffic_signals", {}).values():
            self._safety_window.accumulate(ts)

    def _get_safety_info(self) -> dict:
        brake, exposure = self._safety_window.totals()
        return {
            "system_safety_brake": brake,
            "system_safety_exposure": exposure,
            "system_safety_total": brake + exposure,
        }

    def _compute_info(self):
        # super() builds the standard info dict and appends it to self.metrics;
        # patch both the returned dict and the stored row with the safety terms.
        info = super()._compute_info()
        safety = self._get_safety_info()
        info.update(safety)
        if self.metrics:
            self.metrics[-1].update(safety)
        return info


def make_env(seed: int, scenario: str = "base", lam: float = 0.0,
             gui: bool = False, out_csv: str = None,
             teleport: int = None, tripinfo: bool = False,
             min_green: int = None) -> SumoEnvironment:
    """`teleport` = SUMO --time-to-teleport, in seconds.

    `min_green` is the shortest a green may run before a switch request is
    honoured (sumo-rl silently ignores earlier requests -- see
    TrafficSignal.set_next_phase). It is a knob on the ACTION SPACE, not on the
    simulation: with a 3 s amber, a controller that switches as often as it may
    loses 23% of the cycle to clearance at a 10 s floor against 4.8% at 60 s.

    Defaults to 60 (DEFAULT_MIN_GREEN), resolved from $MIN_GREEN when not
    passed. That default is measured: analysis/actuated.py runs a controller
    with perfect queue information and nothing to learn, and it is 5.6x worse
    than a fixed plan at 10 s, matches it at 60 s, and is worse again at 90 s.
    The floor used to be hard-wired to 10 with no way to override it, so every
    peak training run this project did was conducted where no controller can
    win -- see docs/FINDINGS_2026-08-12.md.

    Callers that hold a green themselves must pass this EXPLICITLY: a static
    plan with a green shorter than the floor is clamped to the floor, which
    would silently turn the low end of analysis/static_timing.py's sweep into a
    row of identical 60 s plans.

    `tripinfo` writes SUMO's per-trip output to `<out_csv>_tripinfo.xml`
    (requires out_csv). The step CSV's system_mean_waiting_time averages over
    vehicles STILL IN the network, so a policy that strands traffic is scored
    on the survivors -- under deadlock it degenerates into a clock. tripinfo
    records one row per COMPLETED trip, which is the metric that cannot be
    gamed by holding vehicles back. Off by default: sumo-rl restarts SUMO on
    every reset, so a training run would rewrite the file each episode.

    sumo-rl defaults this to -1 (teleporting off), which makes junction
    deadlock an absorbing state: the two-phase program has permissive left
    turns, so above ~1.0x base demand the junction can lock and nothing ever
    clears it. At peak (1.5x base) that produced pure-gridlock episodes whose
    "waiting time" was just a clock. SUMO's own default (300) keeps the run
    measurable: peak settles at 42.4 +/- 10.4 s mean wait over seeds 42-46,
    ~9 teleports per episode, no deadlock.

    Resolved from the TIME_TO_TELEPORT env var when not passed explicitly, so
    the run scripts can set it once for a whole sweep. Defaults to -1 --
    sumo-rl's value -- so existing callers behave exactly as before.
    """
    if teleport is None:
        teleport = int(os.environ.get("TIME_TO_TELEPORT", "-1"))
    min_green = resolve_min_green(min_green)
    # vtypes + sublane always; gui-settings only under sumo-gui (plain sumo rejects it)
    extra = "--additional-files vtypes.add.xml --lateral-resolution 0.5"
    if tripinfo:
        if not out_csv:
            raise ValueError("tripinfo=True needs out_csv to derive the filename")
        extra += f" --tripinfo-output {tripinfo_path(out_csv)}"
    if gui:
        extra += " --gui-settings-file gui-settings.xml --start --quit-on-end"
    return SafetyLoggingEnv(
        net_file="intersection.net.xml",
        route_file=SCENARIO_ROUTES[scenario],
        observation_class=PCUObservationFunction,
        use_gui=gui,
        # episode length in sim-seconds; EPISODE_SECONDS env var lets the driver
        # shrink episodes for the fast "overnight" mode (default 3600 = full)
        num_seconds=int(os.environ.get("EPISODE_SECONDS", "3600")),
        delta_time=5,          # seconds between agent decisions
        yellow_time=3,
        min_green=min_green,
        max_green=60,
        reward_fn=make_safety_reward_fn(lam),   # SAME reward for all agents at a given lam
        single_agent=True,              # one TL -> SB3 single-agent API
        sumo_seed=seed,
        time_to_teleport=teleport,
        out_csv_name=out_csv,
        sumo_warnings=False,
        additional_sumo_cmd=extra,
    )
