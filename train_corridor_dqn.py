"""True-independent DQN on the multi-agent corridor env.

Unlike train_corridor.py's parameter-shared IPPO (one policy, pooled
transitions), SP5's IDQN gives each corridor signal its own QNetwork, target
network, replay buffer and optimizer (dqn_core.py). Agents interact only by
stepping the shared SUMO env jointly each timestep; each stores its own
transition and trains independently once its own learning_starts threshold is
hit. This matches RESCO's actual IDQN rather than this project's own
parameter-shared "IPPO" -- see
docs/superpowers/specs/2026-08-21-sp5-idqn-corridor-design.md.
"""
import argparse
import os

import numpy as np
import torch

import corridor_baseline as cb
import dqn_core as dc
from algos import ALGOS
from env_common import CORRIDOR_SCENARIOS, make_corridor_env

# The corridor net's 3 traffic-signal ids (corridor.net.xml), fixed by the
# network SP1 built. Known ahead of instantiating an env so idqn_sweep.py's
# resumability check can test for existing checkpoints without starting SUMO.
CORRIDOR_TS_IDS = ("C1", "C2", "C3")

# The single-intersection DQN pilot's tuned-for-100k config (params/*.json) is
# unrecoverable -- gitignored, cloud-only, never committed (confirmed via
# `git log --all -- params/`). Only 3 of its values survive, disclosed in
# prose in docs/FINDINGS_2026-08-12.md's "Training attempted" section. The
# rest come from algos.ALGOS['dqn']['defaults'] -- this project's own
# canonical "SB3 defaults" source, the same one validate_ppo_core.py's
# matched_hp() already reads for PPO. Best-effort reconstruction, not the
# original tuned config -- disclosed limitation, see the SP5 design spec §4.
_DISCLOSED = {
    "lr": 2.3195e-05,
    "learning_starts": 5000,
    "target_update_interval": 5000,
}


def _hp() -> dict:
    d = ALGOS["dqn"]["defaults"]()
    return {
        "lr": _DISCLOSED["lr"],
        "buffer_size": d["buffer_size"],
        "learning_starts": _DISCLOSED["learning_starts"],
        "batch_size": d["batch_size"],
        "gamma": d["gamma"],
        "train_freq": d["train_freq"],
        "target_update_interval": _DISCLOSED["target_update_interval"],
        "exploration_fraction": d["exploration_fraction"],
        "exploration_final_eps": d["exploration_final_eps"],
        "hidden": tuple(d["policy_kwargs"]["net_arch"]),
    }


def _obs_act_dims(env):
    tid = env.ts_ids[0]
    return env.observation_spaces(tid).shape[0], env.action_spaces(tid).n


# SP11's magnitude-diverse curriculum (docs/FINDINGS_2026-08-22-sp11-offpeak-
# curriculum.md): 5 demand-magnitude route files spanning corridor_offpeak's
# 0.5x to corridor_peak's 1.5x (make_scenarios.py's CORRIDOR_CURRICULUM_FACTORS
# fills the 0.75/1.0/1.25x gaps; the two endpoints reuse the existing peak/
# offpeak files rather than regenerating them). Raw filenames, not
# env_common.SCENARIO_ROUTES keys -- train_curriculum swaps env._route to one
# of these directly, bypassing the scenario-name indirection make_corridor_env
# normally does, since these files are curriculum-only training inputs, not
# scenarios anyone runs a baseline controller on.
CURRICULUM_ROUTES = (
    "corridor_offpeak.rou.xml",
    "corridor_curric_lo.rou.xml",
    "corridor_curric_mid.rou.xml",
    "corridor_curric_hi.rou.xml",
    "corridor_peak.rou.xml",
)

# Pseudo-scenario tag standing in for `scenario` in _tag/_model_path so a
# curriculum checkpoint's filename can never collide with a fixed-scenario
# checkpoint's (e.g. seed 42 curriculum vs seed 42 corridor_peak) -- same
# never-silently-glob-together discipline _eval_out_stem's `_on_`/`_incident`/
# `_net` fragments follow, just at the scenario field instead of a suffix.
CURRICULUM_TAG = "corridor_curriculum"


def _tag(scenario: str, lam: float, seed: int, min_green: int, steps: int,
        variant: str = "") -> str:
    """Filename tag shared by train (models) and evaluate (eval CSV), same
    convention as train_corridor._tag -- see that function's docstring for why
    min_green and steps are both folded in.

    `variant` (SP12) appends one more fragment, e.g. "incaware", so a
    checkpoint trained by a different curriculum than the plain scenario
    sweep (same scenario/lam/seed/min_green/steps otherwise) never collides
    with the existing file. Defaults to "" -- every pre-SP12 call site is
    unaffected."""
    tag = f"{scenario}_lam{str(lam).replace('.', '')}_seed{seed}_mg{min_green}_s{steps}"
    if variant:
        tag += f"_{variant}"
    return tag


def _model_path(agent_id: str, scenario: str, lam: float, seed: int,
                min_green: int, steps: int, variant: str = "") -> str:
    """Where one agent's checkpoint lives. Unlike train_corridor.py's single
    shared-policy path, IDQN has one file per agent -- agent_id is folded in
    right after the algo name so the 3 files for one run sort together and
    never collide with each other or with a different run's tag."""
    return f"models/idqn_{agent_id}_{_tag(scenario, lam, seed, min_green, steps, variant=variant)}.pt"


def train(scenario: str, lam: float, seed: int, steps: int, min_green: int,
         incident: tuple = None, incident_prob: float = 0.0,
         variant: str = "") -> dict:
    """Train 3 fully independent DQN agents. Returns {ts_id: model_path}.

    incident/incident_prob (SP12, docs/superpowers/specs/2026-08-22-sp7-
    corridor-incident-design.md's own deferred-decisions list): if
    `incident` is given and `incident_prob` > 0, each training episode
    independently includes that incident (the same corridor_baseline.INCIDENT
    lane closure SP7 evaluates zero-shot, applied via env.set_incident() the
    same way SP7's eval applies it via make_corridor_env) with probability
    `incident_prob`, decided by the same seeded `rng` this loop already uses
    for epsilon-greedy exploration -- reproducible per seed. A mix rather
    than incident-every-episode, so the policy sees both the ordinary
    no-incident dynamics it must still handle well and the disrupted ones,
    instead of overfitting to "an incident always happens." Defaults
    (incident=None, incident_prob=0.0) reproduce the pre-SP12 training path
    exactly: `incident_prob > 0` is checked before `rng.random()` is ever
    called (short-circuit `and`), so the default case draws nothing extra
    from `rng` and every existing call site/test is bit-for-bit unaffected.

    Limitation, disclosed rather than engineered around: SP7's incident is
    one fixed, deterministic scenario (same edge, same start/duration every
    time it fires -- see corridor_baseline.INCIDENT). This curriculum reuses
    that exact spec unmodified on every incident episode; it does not
    randomize timing/location/severity across episodes, so a policy trained
    this way could in principle learn to key off the wall-clock time the
    incident always starts at, rather than reacting to the traffic pattern
    it causes. Not addressed here -- see the SP12 findings doc.

    variant (SP12): appended to _tag()'s filename fragment (see _tag) so an
    incident-aware checkpoint never collides with the existing plain
    `corridor_peak`-trained files this same (scenario, lam, seed, min_green,
    steps) combination already names.
    """
    hp = _hp()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_corridor_env(seed=seed, scenario=scenario, lam=lam, min_green=min_green)
    ids = env.ts_ids
    obs_dim, act_dim = _obs_act_dims(env)

    def _episode_incident():
        if incident is not None and incident_prob > 0 and rng.random() < incident_prob:
            return incident
        return None

    agents = {}
    for i in ids:
        q_net = dc.QNetwork(obs_dim, act_dim, hidden=hp["hidden"])
        target_net = dc.QNetwork(obs_dim, act_dim, hidden=hp["hidden"])
        target_net.load_state_dict(q_net.state_dict())
        agents[i] = {
            "q": q_net,
            "target": target_net,
            "optim": torch.optim.Adam(q_net.parameters(), lr=hp["lr"]),
            "buffer": dc.ReplayBuffer(hp["buffer_size"], obs_dim),
            "eps": dc.EpsilonSchedule(steps, hp["exploration_fraction"],
                                      hp["exploration_final_eps"]),
        }

    # SP12: decide the first episode's incident status before the first
    # reset() too, not just on every subsequent one below -- otherwise
    # episode 0 could never be an incident episode.
    env.set_incident(_episode_incident())
    obs = env.reset()
    for t in range(steps):
        actions = {}
        for i in ids:
            a = agents[i]
            if rng.random() < a["eps"].value(t):
                actions[i] = int(rng.integers(act_dim))
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                    actions[i] = int(a["q"](obs_t).argmax(dim=-1).item())
        nobs, rewards, dones, _ = env.step(actions)
        # dones["__all__"] is True both when the episode genuinely ends and
        # when it's truncated by the episode time limit -- either way it's
        # stored as a terminal transition (done=1) in each agent's replay
        # buffer, the same convention analysis/validate_dqn_core.py's
        # train_dqn_core loop uses.
        done_all = float(dones["__all__"])
        for i in ids:
            agents[i]["buffer"].add(obs[i], actions[i], float(rewards[i]), nobs[i],
                                    done_all)
        obs = nobs
        if done_all:
            # SP12: re-roll whether the NEXT episode gets the incident before
            # resetting -- env.set_incident()'s own docstring explains why
            # this ordering (before reset(), not after) is required.
            env.set_incident(_episode_incident())
            obs = env.reset()

        if t >= hp["learning_starts"] and t % hp["train_freq"] == 0:
            for i in ids:
                a = agents[i]
                if len(a["buffer"]) < hp["batch_size"]:
                    continue
                b_obs, b_act, b_rew, b_nobs, b_done = a["buffer"].sample(
                    hp["batch_size"], rng)
                loss, _ = dc.dqn_loss(a["q"], a["target"], b_obs, b_act, b_rew,
                                      b_nobs, b_done, hp["gamma"])
                a["optim"].zero_grad()
                loss.backward()
                a["optim"].step()

        if t > 0 and t % hp["target_update_interval"] == 0:
            for i in ids:
                agents[i]["target"].load_state_dict(agents[i]["q"].state_dict())
    env.close()

    os.makedirs("models", exist_ok=True)
    paths = {}
    for i in ids:
        path = _model_path(i, scenario, lam, seed, min_green, steps, variant=variant)
        torch.save({"state_dict": agents[i]["q"].state_dict(), "hidden": hp["hidden"]},
                   path)
        paths[i] = path
    print(f"idqn models saved: {list(paths.values())}")
    return paths


def train_curriculum(lam: float, seed: int, steps: int, min_green: int,
                     routes: tuple = CURRICULUM_ROUTES,
                     tag: str = CURRICULUM_TAG) -> dict:
    """Train 3 fully independent DQN agents across a magnitude-diverse demand
    curriculum instead of one fixed scenario (SP11).

    Identical to train() in every respect (same hyperparameters, same
    architecture, same replay/target-update cadence, same total step budget)
    except for where the demand comes from: at every episode boundary a new
    route file is drawn uniformly at random from `routes` (seeded by `seed`,
    an independent RNG stream from the action-selection one so a curriculum
    run's exploration noise isn't perturbed by how many episodes have elapsed)
    and swapped into the running env before reset() -- SumoEnvironment's own
    reset() calls self._start_simulation(), which reads self._route fresh
    every time (sumo_rl/environment/env.py), so mutating that attribute
    directly is enough; no env_common.py change is needed and no new env
    object is constructed mid-run (SUMO already restarts on every reset()
    regardless, per env_common.make_env's own docstring, so this costs
    nothing extra over a fixed-scenario run's existing per-episode restart).

    `tag` stands in for `scenario` in the saved checkpoints' filenames
    (default CURRICULUM_TAG) so these checkpoints can never collide with a
    fixed-scenario run's, and so evaluate() can look them up zero-shot on any
    real scenario later exactly the way SP6 evaluated peak-only checkpoints
    zero-shot on corridor_offpeak/corridor_tidal/corridor_skew -- pass
    `scenario=tag` there.
    """
    hp = _hp()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    route_rng = np.random.default_rng(seed + 1_000_000)
    # first episode's magnitude is itself drawn from the curriculum, not
    # hard-coded to one end of it
    start_route = routes[int(route_rng.integers(len(routes)))]
    env = make_corridor_env(seed=seed, scenario="corridor_offpeak", lam=lam,
                            min_green=min_green)
    env._route = start_route
    ids = env.ts_ids
    obs_dim, act_dim = _obs_act_dims(env)

    agents = {}
    for i in ids:
        q_net = dc.QNetwork(obs_dim, act_dim, hidden=hp["hidden"])
        target_net = dc.QNetwork(obs_dim, act_dim, hidden=hp["hidden"])
        target_net.load_state_dict(q_net.state_dict())
        agents[i] = {
            "q": q_net,
            "target": target_net,
            "optim": torch.optim.Adam(q_net.parameters(), lr=hp["lr"]),
            "buffer": dc.ReplayBuffer(hp["buffer_size"], obs_dim),
            "eps": dc.EpsilonSchedule(steps, hp["exploration_fraction"],
                                      hp["exploration_final_eps"]),
        }

    obs = env.reset()
    route_counts = {r: 0 for r in routes}
    route_counts[env._route] += 1
    for t in range(steps):
        actions = {}
        for i in ids:
            a = agents[i]
            if rng.random() < a["eps"].value(t):
                actions[i] = int(rng.integers(act_dim))
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                    actions[i] = int(a["q"](obs_t).argmax(dim=-1).item())
        nobs, rewards, dones, _ = env.step(actions)
        done_all = float(dones["__all__"])
        for i in ids:
            agents[i]["buffer"].add(obs[i], actions[i], float(rewards[i]), nobs[i],
                                    done_all)
        obs = nobs
        if done_all:
            env._route = routes[int(route_rng.integers(len(routes)))]
            route_counts[env._route] += 1
            obs = env.reset()

        if t >= hp["learning_starts"] and t % hp["train_freq"] == 0:
            for i in ids:
                a = agents[i]
                if len(a["buffer"]) < hp["batch_size"]:
                    continue
                b_obs, b_act, b_rew, b_nobs, b_done = a["buffer"].sample(
                    hp["batch_size"], rng)
                loss, _ = dc.dqn_loss(a["q"], a["target"], b_obs, b_act, b_rew,
                                      b_nobs, b_done, hp["gamma"])
                a["optim"].zero_grad()
                loss.backward()
                a["optim"].step()

        if t > 0 and t % hp["target_update_interval"] == 0:
            for i in ids:
                agents[i]["target"].load_state_dict(agents[i]["q"].state_dict())
    env.close()
    print(f"curriculum route counts (episodes): {route_counts}")

    os.makedirs("models", exist_ok=True)
    paths = {}
    for i in ids:
        path = _model_path(i, tag, lam, seed, min_green, steps)
        torch.save({"state_dict": agents[i]["q"].state_dict(), "hidden": hp["hidden"]},
                   path)
        paths[i] = path
    print(f"idqn curriculum models saved: {list(paths.values())}")
    return paths


def _eval_out_stem(scenario: str, eval_scenario: str, lam: float, seed: int,
                   min_green: int, steps: int, incident: bool = False,
                   net_file: str = "corridor.net.xml", variant: str = "") -> str:
    """Eval CSV path stem. '_on_<eval_scenario>' is appended when
    eval_scenario differs from the checkpoint's training scenario (SP6
    zero-shot). '_incident' is appended when the SP7 lane closure was applied
    (docs/superpowers/specs/2026-08-22-sp7-corridor-incident-design.md) --
    both fragments can combine, since SP7's incident eval is itself zero-shot
    against the corridor_peak checkpoints. '_net<label>' is appended when
    net_file isn't the regular corridor.net.xml geometry -- same
    never-silently-glob-together discipline, and same reason
    corridor_baseline.run() tags its own non-default-net CSVs this way.
    `variant` (SP12) is folded into the tag by _tag() itself -- see that
    function's docstring."""
    tag = _tag(scenario, lam, seed, min_green, steps, variant=variant)
    stem = f"logs/eval_idqn_{tag}"
    if eval_scenario != scenario:
        stem += f"_on_{eval_scenario}"
    if incident:
        stem += "_incident"
    if net_file != "corridor.net.xml":
        label = net_file.removesuffix(".net.xml").removeprefix("corridor_")
        stem += f"_net{label}"
    return stem


def evaluate(scenario: str, lam: float, seed: int, min_green: int, steps: int,
            tripinfo: bool = False, eval_scenario: str = None,
            incident: bool = False, net_file: str = "corridor.net.xml",
            variant: str = "") -> str:
    """Run all 3 agents' greedy policies for one episode, writing one eval
    CSV in the SafetyLoggingEnv format so compare.py reads it as `idqn`. With
    tripinfo=True also writes the per-trip XML analysis/idqn_sweep.py reduces.

    Loads each agent's checkpoint by reconstructing its path from _model_path
    -- callers never pass paths directly, so train() and evaluate() can never
    disagree about where a checkpoint lives.

    eval_scenario, if given, evaluates the checkpoint's greedy policy against
    a DIFFERENT demand scenario than it was trained on -- SP6's zero-shot
    generalization eval (docs/superpowers/specs/2026-08-22-sp6-idqn-demand-shift-design.md).
    Defaults to `scenario` (today's in-distribution behaviour, unchanged): the
    checkpoint is always looked up under `scenario`, but the env runs whichever
    scenario `eval_scenario` names.

    incident=True applies corridor_baseline.INCIDENT to the eval env -- the
    same fixed lane closure every controller in the SP7 comparison faces.

    net_file, if given, runs the checkpoint's greedy policy on a DIFFERENT
    network geometry than it trained on (zero-shot, same spirit as
    eval_scenario) -- the checkpoint is still looked up under `scenario`'s
    regular-net path, only the env's geometry changes.

    variant (SP12) must match whatever `train()` was called with to produce
    the checkpoint being loaded here -- see _tag()."""
    os.makedirs("logs", exist_ok=True)
    eval_scenario = eval_scenario or scenario
    out_csv = _eval_out_stem(scenario, eval_scenario, lam, seed, min_green, steps,
                             incident=incident, net_file=net_file, variant=variant)
    env = make_corridor_env(seed=seed, scenario=eval_scenario, lam=lam,
                            min_green=min_green, out_csv=out_csv, tripinfo=tripinfo,
                            incident=cb.INCIDENT if incident else None,
                            net_file=net_file)
    ids = env.ts_ids
    obs_dim, act_dim = _obs_act_dims(env)
    policies = {}
    for i in ids:
        ckpt = torch.load(_model_path(i, scenario, lam, seed, min_green, steps,
                                      variant=variant),
                          weights_only=True)
        q_net = dc.QNetwork(obs_dim, act_dim, hidden=tuple(ckpt["hidden"]))
        q_net.load_state_dict(ckpt["state_dict"])
        q_net.eval()
        policies[i] = q_net

    obs = env.reset()
    done = False
    while not done:
        actions = {}
        for i in ids:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                actions[i] = int(policies[i](obs_t).argmax(dim=-1).item())
        obs, _, dones, _ = env.step(actions)
        done = dones["__all__"]
    env.save_csv(env.out_csv_name, env.episode)
    env.close()
    out = f"{out_csv}_conn{env.label}_ep{env.episode}.csv"
    print(f"idqn eval written: {out}")
    return out


if __name__ == "__main__":
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="corridor_offpeak",
                   choices=list(CORRIDOR_SCENARIOS))
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-scenario", default=None, choices=list(CORRIDOR_SCENARIOS),
                   help="demand scenario to evaluate on, if different from "
                        "--scenario (zero-shot; defaults to --scenario)")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--min-green", type=int, required=True,
                   help="explicit -- this script never falls back to $MIN_GREEN/DEFAULT_MIN_GREEN")
    p.add_argument("--eval", action="store_true",
                   help="evaluate existing checkpoints instead of training")
    p.add_argument("--tripinfo", action="store_true",
                   help="also write the per-trip XML (only meaningful with --eval)")
    p.add_argument("--incident", action="store_true",
                   help=f"apply the SP7 lane-closure incident ({cb.INCIDENT}) "
                        "-- eval path only; see --incident-prob for training")
    p.add_argument("--curriculum", action="store_true",
                   help="SP11: train/evaluate the magnitude-diverse curriculum "
                        f"checkpoint (CURRICULUM_ROUTES={CURRICULUM_ROUTES}) "
                        f"tagged '{CURRICULUM_TAG}', instead of a fixed --scenario "
                        "(--scenario is ignored when this is set)")
    p.add_argument("--incident-prob", type=float, default=0.0,
                   help="SP12: fraction of TRAINING episodes that include the "
                        "SP7 lane-closure incident (0.0 = never, the pre-SP12 "
                        "default; ignored with --eval)")
    p.add_argument("--variant", default="",
                   help="SP12: extra filename-tag fragment (see _tag) so a "
                        "differently-curriculumed checkpoint -- e.g. "
                        "incident-aware training -- never collides with an "
                        "existing checkpoint at the same scenario/lam/seed/"
                        "min_green/steps")
    args = p.parse_args()
    scenario = CURRICULUM_TAG if args.curriculum else args.scenario
    if args.eval:
        evaluate(scenario, args.lam, args.seed, args.min_green, args.steps,
                 tripinfo=args.tripinfo, eval_scenario=args.eval_scenario,
                 incident=args.incident, variant=args.variant)
    elif args.curriculum:
        train_curriculum(args.lam, args.seed, args.steps, args.min_green)
    else:
        train(args.scenario, args.lam, args.seed, args.steps, args.min_green,
             incident=cb.INCIDENT if args.incident_prob > 0 else None,
             incident_prob=args.incident_prob, variant=args.variant)
