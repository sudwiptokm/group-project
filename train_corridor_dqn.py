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


def _tag(scenario: str, lam: float, seed: int, min_green: int, steps: int) -> str:
    """Filename tag shared by train (models) and evaluate (eval CSV), same
    convention as train_corridor._tag -- see that function's docstring for why
    min_green and steps are both folded in."""
    return f"{scenario}_lam{str(lam).replace('.', '')}_seed{seed}_mg{min_green}_s{steps}"


def _model_path(agent_id: str, scenario: str, lam: float, seed: int,
                min_green: int, steps: int) -> str:
    """Where one agent's checkpoint lives. Unlike train_corridor.py's single
    shared-policy path, IDQN has one file per agent -- agent_id is folded in
    right after the algo name so the 3 files for one run sort together and
    never collide with each other or with a different run's tag."""
    return f"models/idqn_{agent_id}_{_tag(scenario, lam, seed, min_green, steps)}.pt"


def train(scenario: str, lam: float, seed: int, steps: int, min_green: int) -> dict:
    """Train 3 fully independent DQN agents. Returns {ts_id: model_path}."""
    hp = _hp()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_corridor_env(seed=seed, scenario=scenario, lam=lam, min_green=min_green)
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
        path = _model_path(i, scenario, lam, seed, min_green, steps)
        torch.save({"state_dict": agents[i]["q"].state_dict(), "hidden": hp["hidden"]},
                   path)
        paths[i] = path
    print(f"idqn models saved: {list(paths.values())}")
    return paths


def evaluate(scenario: str, lam: float, seed: int, min_green: int, steps: int,
            tripinfo: bool = False) -> str:
    """Run all 3 agents' greedy policies for one episode, writing one eval
    CSV in the SafetyLoggingEnv format so compare.py reads it as `idqn`. With
    tripinfo=True also writes the per-trip XML analysis/idqn_sweep.py reduces.

    Loads each agent's checkpoint by reconstructing its path from _model_path
    -- callers never pass paths directly, so train() and evaluate() can never
    disagree about where a checkpoint lives."""
    os.makedirs("logs", exist_ok=True)
    tag = _tag(scenario, lam, seed, min_green, steps)
    out_csv = f"logs/eval_idqn_{tag}"
    env = make_corridor_env(seed=seed, scenario=scenario, lam=lam,
                            min_green=min_green, out_csv=out_csv, tripinfo=tripinfo)
    ids = env.ts_ids
    obs_dim, act_dim = _obs_act_dims(env)
    policies = {}
    for i in ids:
        ckpt = torch.load(_model_path(i, scenario, lam, seed, min_green, steps),
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
    out = f"logs/eval_idqn_{tag}_conn{env.label}_ep{env.episode}.csv"
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
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--min-green", type=int, required=True,
                   help="explicit -- this script never falls back to $MIN_GREEN/DEFAULT_MIN_GREEN")
    p.add_argument("--eval", action="store_true",
                   help="evaluate existing checkpoints instead of training")
    p.add_argument("--tripinfo", action="store_true",
                   help="also write the per-trip XML (only meaningful with --eval)")
    args = p.parse_args()
    if args.eval:
        evaluate(args.scenario, args.lam, args.seed, args.min_green, args.steps,
                 tripinfo=args.tripinfo)
    else:
        train(args.scenario, args.lam, args.seed, args.steps, args.min_green)
