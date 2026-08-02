"""Parameter-shared IPPO on the multi-agent corridor env.

Custom PPO (ppo_core) trained by pooling all agents' transitions into one shared
policy. GAE is computed PER AGENT (each agent has its own temporal trajectory),
then transitions are concatenated across agents for the shared update. Same
actor/critic loop SP3's MAPPO will extend (critic input only).
"""
import argparse
import json
import os

import numpy as np
import torch

import ppo_core as pc
from env_common import make_corridor_env

PARAMS_FILE = "cloud_params/ppo.json"


def _hp() -> dict:
    """Reused single-intersection PPO hyperparameters (disclosed limitation)."""
    with open(PARAMS_FILE) as fh:
        p = json.load(fh)
    return {
        "lr": p["learning_rate"],
        "n_steps": p["n_steps"],
        "batch_size": p["batch_size"],
        "n_epochs": p["n_epochs"],
        "gamma": p["gamma"],
        "gae_lambda": p["gae_lambda"],
        "clip_range": p["clip_range"],
        "ent_coef": p["ent_coef"],
        "hidden": tuple(p["net_arch"]),
    }


def _obs_act_dims(env):
    tid = env.ts_ids[0]
    return env.observation_spaces(tid).shape[0], env.action_spaces(tid).n


def _tag(scenario: str, lam: float, seed: int) -> str:
    """Filename tag shared by train (model) and evaluate (eval CSV) so the two
    never diverge; the caller prepends the algo, matching compare.py's
    `eval_<algo>_<scenario>_lam<lam>_seed*` glob (algo = ippo | mappo)."""
    return f"{scenario}_lam{str(lam).replace('.', '')}_seed{seed}"


def build_states(obs, ts_ids, centralized):
    """Per-agent critic state. IPPO (centralized=False): each agent's own local
    observation. MAPPO (centralized=True): the joint state — all agents' local
    observations concatenated in ts_ids order — shared by every agent.
    """
    if not centralized:
        return {i: obs[i] for i in ts_ids}
    joint = np.concatenate([obs[i] for i in ts_ids])
    return {i: joint for i in ts_ids}


def collect_rollout(env, policy, obs, n_steps, centralized):
    """Step the env n_steps, storing a SEPARATE temporal buffer per agent. Each
    transition keeps the actor's local `obs` and the critic's `state` (local obs
    for IPPO, joint state for MAPPO). Returns (per_agent_buffers, trailing_obs)."""
    ids = env.ts_ids
    per = {i: {"obs": [], "state": [], "act": [], "logp": [], "rew": [],
               "val": [], "done": []} for i in ids}
    for _ in range(n_steps):
        obs_t = torch.as_tensor(np.stack([obs[i] for i in ids]), dtype=torch.float32)
        states = build_states(obs, ids, centralized)
        state_t = torch.as_tensor(np.stack([states[i] for i in ids]), dtype=torch.float32)
        with torch.no_grad():
            actions_t, logp_t = policy.act(obs_t)   # actor: local obs
            vals_t = policy.value(state_t)          # critic: state (local or joint)
        actions = {i: int(a) for i, a in zip(ids, actions_t)}
        nobs, rewards, dones, _ = env.step(actions)
        done_all = float(dones["__all__"])
        for k, i in enumerate(ids):
            per[i]["obs"].append(obs_t[k])
            per[i]["state"].append(state_t[k])
            per[i]["act"].append(actions_t[k])
            per[i]["logp"].append(logp_t[k])
            per[i]["val"].append(float(vals_t[k]))
            per[i]["rew"].append(float(rewards[i]))
            per[i]["done"].append(done_all)
        obs = nobs
        if done_all:
            obs = env.reset()
    return per, obs


def update(policy, optim, per, hp, last_states):
    """One PPO update: per-agent GAE (bootstrapping V from each agent's trailing
    critic state), then concatenated minibatch updates. The actor is trained on
    local `obs`; the critic on `state` (local for IPPO, joint for MAPPO)."""
    all_obs, all_state, all_act, all_logp, all_adv, all_ret = [], [], [], [], [], []
    for agent, b in per.items():
        with torch.no_grad():
            lv = float(policy.value(
                torch.as_tensor(last_states[agent], dtype=torch.float32)))
        adv, ret = pc.compute_gae(b["rew"], b["val"], b["done"],
                                  hp["gamma"], hp["gae_lambda"], last_value=lv)
        all_obs += b["obs"]
        all_state += b["state"]
        all_act += b["act"]
        all_logp += b["logp"]
        all_adv += adv
        all_ret += ret

    obs = torch.stack(all_obs)
    state = torch.stack(all_state)
    act = torch.stack(all_act)
    old_logp = torch.stack(all_logp).detach()
    adv_t = torch.as_tensor(all_adv, dtype=torch.float32)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t = torch.as_tensor(all_ret, dtype=torch.float32)

    n = obs.shape[0]
    idx = np.arange(n)
    last_info = {}
    for _ in range(hp["n_epochs"]):
        np.random.shuffle(idx)
        for start in range(0, n, hp["batch_size"]):
            bi = idx[start:start + hp["batch_size"]]
            dist = policy.policy(obs[bi])          # actor: local obs
            vals = policy.value(state[bi])         # critic: state
            loss, info = pc.ppo_loss(dist, act[bi], old_logp[bi], adv_t[bi], vals,
                                     ret_t[bi], clip=hp["clip_range"],
                                     ent_coef=hp["ent_coef"])
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optim.step()
            last_info = info
    return last_info


def train(scenario: str, lam: float, seed: int, steps: int,
          centralized: bool = False) -> str:
    hp = _hp()
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = make_corridor_env(seed=seed, scenario=scenario, lam=lam)
    obs_dim, act_dim = _obs_act_dims(env)
    n_agents = len(env.ts_ids)
    state_dim = obs_dim * n_agents if centralized else obs_dim
    policy = pc.ActorCritic(obs_dim, act_dim, state_dim=state_dim, hidden=hp["hidden"])
    optim = torch.optim.Adam(policy.parameters(), lr=hp["lr"])

    obs = env.reset()
    collected = 0  # ENV-steps (matches SB3 total_timesteps semantics)
    while collected < steps:
        per, obs = collect_rollout(env, policy, obs, hp["n_steps"], centralized)
        last_states = build_states(obs, env.ts_ids, centralized)
        update(policy, optim, per, hp, last_states)
        collected += hp["n_steps"]
    env.close()

    algo = "mappo" if centralized else "ippo"
    os.makedirs("models", exist_ok=True)
    path = f"models/{algo}_{_tag(scenario, lam, seed)}.pt"
    torch.save({"state_dict": policy.state_dict(), "hidden": hp["hidden"],
                "centralized": centralized, "state_dim": state_dim}, path)
    print(f"{algo} model saved: {path}")
    return path


def evaluate(model_path: str, scenario: str, lam: float, seed: int) -> str:
    """Greedy eval of a saved policy on a held-out seed. The entity (ippo/mappo)
    and the critic width are read from the checkpoint so the net is rebuilt exactly.
    The actor uses local obs, so eval is identical in shape for IPPO and MAPPO."""
    ckpt = torch.load(model_path, weights_only=True)
    is_dict = isinstance(ckpt, dict) and "state_dict" in ckpt
    centralized = bool(ckpt.get("centralized", False)) if is_dict else False
    algo = "mappo" if centralized else "ippo"

    os.makedirs("logs", exist_ok=True)
    tag = _tag(scenario, lam, seed)
    out_csv = f"logs/eval_{algo}_{tag}"
    env = make_corridor_env(seed=seed, scenario=scenario, lam=lam, out_csv=out_csv)
    obs_dim, act_dim = _obs_act_dims(env)
    if is_dict:
        hidden = tuple(ckpt["hidden"])
        state_dim = ckpt.get("state_dim", obs_dim)
        state = ckpt["state_dict"]
    else:
        hidden, state_dim, state = _hp()["hidden"], obs_dim, ckpt
    policy = pc.ActorCritic(obs_dim, act_dim, state_dim=state_dim, hidden=hidden)
    policy.load_state_dict(state)
    policy.eval()

    obs = env.reset()
    done = False
    while not done:
        ids = env.ts_ids
        obs_t = torch.as_tensor(np.stack([obs[i] for i in ids]), dtype=torch.float32)
        with torch.no_grad():
            logits = policy.actor(obs_t)          # greedy, decentralised (local obs)
        actions = {i: int(a) for i, a in zip(ids, logits.argmax(dim=-1))}
        obs, _, dones, _ = env.step(actions)
        done = dones["__all__"]
    env.save_csv(env.out_csv_name, env.episode)
    env.close()
    out = f"logs/eval_{algo}_{tag}_conn{env.label}_ep{env.episode}.csv"
    print(f"{algo} eval written: {out}")
    return out


if __name__ == "__main__":
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="corridor_offpeak",
                   choices=["corridor_peak", "corridor_offpeak"])
    p.add_argument("--algo", default="ippo", choices=["ippo", "mappo"])
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--eval", type=str, default=None,
                   help="path to a saved model to evaluate instead of training")
    args = p.parse_args()
    if args.eval:
        evaluate(args.eval, args.scenario, args.lam, args.seed)
    else:
        train(args.scenario, args.lam, args.seed, args.steps,
              centralized=(args.algo == "mappo"))
