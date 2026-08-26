"""Parameter-shared IPPO on the multi-agent corridor env.

Custom PPO (ppo_core) trained by pooling all agents' transitions into one shared
policy. GAE is computed PER AGENT (each agent has its own temporal trajectory),
then transitions are concatenated across agents for the shared update. Same
actor/critic loop SP3's MAPPO will extend (critic input only).
"""
import argparse
import os

import numpy as np
import torch

import ppo_core as pc
from env_common import CORRIDOR_SCENARIOS, make_corridor_env

# Reused single-intersection PPO hyperparameters (disclosed limitation, see
# docs/superpowers/plans/2026-08-02-sp2-independent-marl.md header). These came
# from a cloud tuning run whose params/ directory was scp'd to cloud_params/
# (docs/AWS_CLOUD_GUIDE.md); that directory is gitignored and does not exist on
# a local-only checkout, so the exact values are inlined here rather than read
# from a file that would silently vanish on a fresh clone.
_HP = {
    "lr": 2.3195e-05,
    "n_steps": 128,
    "batch_size": 32,
    "n_epochs": 10,
    "gamma": 0.95,
    "gae_lambda": 0.9525,
    "clip_range": 0.1,
    "ent_coef": 0.0081,
    "hidden": (256, 256),
}


def _hp() -> dict:
    return dict(_HP)


def _obs_act_dims(env):
    tid = env.ts_ids[0]
    return env.observation_spaces(tid).shape[0], env.action_spaces(tid).n


def _tag(scenario: str, lam: float, seed: int, min_green: int, steps: int) -> str:
    """Filename tag shared by train (model) and evaluate (eval CSV) so the two
    never diverge. min_green is folded in (env_common's own eval_csv_stem/
    model_path convention) because a checkpoint or eval CSV is FOR one floor;
    two floors trained on the same scenario/lam/seed must not collide. steps
    is folded in for the same reason: a checkpoint or eval CSV is also FOR one
    step budget, and two budgets trained on the same scenario/lam/seed/floor
    must not collide either -- a re-run at a different budget (e.g. a 100k
    confirmatory check after a 16k sweep) must not silently overwrite or be
    mistaken for the other budget's checkpoint/eval log."""
    return f"{scenario}_lam{str(lam).replace('.', '')}_seed{seed}_mg{min_green}_s{steps}"


def build_states(obs: dict, ts_ids, centralized: bool) -> dict:
    """Per-agent critic input for one step (SP3, MAPPO). IPPO (centralized=False):
    each agent's own local observation, unchanged. MAPPO (centralized=True): the
    SAME joint state -- every agent's local observation concatenated in `ts_ids`
    order, NOT dict-insertion order -- shared by every agent. The actor always
    consumes local `obs` regardless; only the critic's input differs between the
    two algorithms."""
    if not centralized:
        return {i: obs[i] for i in ts_ids}
    joint = np.concatenate([obs[i] for i in ts_ids])
    return {i: joint for i in ts_ids}


def collect_rollout(env, policy, obs, n_steps, centralized: bool = False):
    """Step the env n_steps, storing a SEPARATE temporal buffer per agent (so GAE
    stays within each agent's trajectory). Each transition keeps both the actor's
    local `obs` and the critic's `state` (local obs for IPPO, joint state for
    MAPPO, via build_states) -- the value estimate is computed from `state`.
    Returns (per_agent_buffers, trailing_obs)."""
    ids = env.ts_ids
    per = {i: {"obs": [], "state": [], "act": [], "logp": [], "rew": [], "val": [],
               "done": []} for i in ids}
    for _ in range(n_steps):
        obs_t = torch.as_tensor(np.stack([obs[i] for i in ids]), dtype=torch.float32)
        states = build_states(obs, ids, centralized)
        state_t = torch.as_tensor(np.stack([states[i] for i in ids]), dtype=torch.float32)
        with torch.no_grad():
            actions_t, logp_t = policy.act(obs_t)      # actor: local obs
            vals_t = policy.value(state_t)              # critic: state (local or joint)
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
    critic STATE, not obs -- local for IPPO, joint for MAPPO), then concatenated
    minibatch updates over the shared policy. Returns the last minibatch info
    dict. The actor is trained on local `obs`; the critic on `state`.

    last_states is the trailing per-agent critic-state dict from the rollout
    (build_states applied to the rollout's trailing obs); each agent's GAE
    bootstraps V(s_T) from it so truncated (mid-episode) rollouts are not biased by
    an implicit V=0. The per-step done flag masks this bootstrap when the final
    collected step was actually terminal.
    """
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
            b = idx[start:start + hp["batch_size"]]
            dist = policy.policy(obs[b])
            vals = policy.value(state[b])
            loss, info = pc.ppo_loss(dist, act[b], old_logp[b], adv_t[b], vals,
                                     ret_t[b], clip=hp["clip_range"],
                                     ent_coef=hp["ent_coef"])
            optim.zero_grad()
            loss.backward()
            # match SB3's default max_grad_norm=0.5 (the HPs were tuned under it)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optim.step()
            last_info = info
    return last_info


def train(scenario: str, lam: float, seed: int, steps: int, min_green: int,
         centralized: bool = False) -> str:
    """centralized=True is MAPPO (SP3): the critic sees the joint observation
    of all agents. centralized=False (default) is IPPO, unchanged behaviour."""
    hp = _hp()
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = make_corridor_env(seed=seed, scenario=scenario, lam=lam, min_green=min_green)
    ids = env.ts_ids
    obs_dim, act_dim = _obs_act_dims(env)
    # SP3: MAPPO's critic sees the joint observation of every agent; the actor
    # (obs_dim, unchanged) stays local-obs-only either way -- see build_states.
    state_dim = obs_dim * len(ids) if centralized else obs_dim
    policy = pc.ActorCritic(obs_dim, act_dim, state_dim=state_dim, hidden=hp["hidden"])
    optim = torch.optim.Adam(policy.parameters(), lr=hp["lr"])

    obs = env.reset()
    collected = 0  # counted in ENV-steps (matches SB3 total_timesteps semantics)
    while collected < steps:
        per, obs = collect_rollout(env, policy, obs, hp["n_steps"], centralized=centralized)
        last_states = build_states(obs, ids, centralized)
        update(policy, optim, per, hp, last_states=last_states)
        collected += hp["n_steps"]
    env.close()

    algo = "mappo" if centralized else "ippo"
    os.makedirs("models", exist_ok=True)
    path = f"models/{algo}_{_tag(scenario, lam, seed, min_green, steps)}.pt"
    # store the architecture (+ centralized/state_dim, SP3) alongside the weights
    # so evaluate() rebuilds the exact net without depending on _hp()'s current
    # defaults matching what this checkpoint was actually trained with.
    torch.save({"state_dict": policy.state_dict(), "hidden": hp["hidden"],
               "centralized": centralized, "state_dim": state_dim}, path)
    print(f"{algo} model saved: {path}")
    return path


def evaluate(model_path: str, scenario: str, lam: float, seed: int, min_green: int,
             steps: int, tripinfo: bool = False) -> str:
    """Run the trained shared policy greedily on a held-out seed, writing an eval
    CSV in the SafetyLoggingEnv format so compare.py reads it as `ippo`/`mappo`
    (SP3: the entity is read from the checkpoint's `centralized` flag, not passed
    in -- eval doesn't need to know which algorithm trained the checkpoint it's
    given). With tripinfo=True also writes the per-trip XML
    analysis/ippo_sweep.py reduces.

    steps is not used for inference -- it exists purely so this function
    reconstructs the exact same tag train() used to name the checkpoint it is
    now loading, per _tag()'s docstring."""
    os.makedirs("logs", exist_ok=True)
    tag = _tag(scenario, lam, seed, min_green, steps)
    # checkpoints save {"state_dict", "hidden", "centralized", "state_dim"};
    # tolerate a bare state_dict, or a pre-SP3 dict missing the two new keys,
    # both of which default to IPPO (centralized=False, state_dim=obs_dim) --
    # obs_dim isn't known yet at this point, so that fallback is resolved once
    # the env below gives us obs_dim.
    ckpt = torch.load(model_path, weights_only=True)
    has_arch = isinstance(ckpt, dict) and "state_dict" in ckpt
    hidden = tuple(ckpt["hidden"]) if has_arch else _hp()["hidden"]
    centralized = bool(ckpt.get("centralized", False)) if has_arch else False
    state = ckpt["state_dict"] if has_arch else ckpt
    algo = "mappo" if centralized else "ippo"
    out_csv = f"logs/eval_{algo}_{tag}"

    env = make_corridor_env(seed=seed, scenario=scenario, lam=lam,
                            min_green=min_green, out_csv=out_csv, tripinfo=tripinfo)
    obs_dim, act_dim = _obs_act_dims(env)
    state_dim = ckpt.get("state_dim", obs_dim) if has_arch else obs_dim
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
                   choices=list(CORRIDOR_SCENARIOS))
    p.add_argument("--algo", default="ippo", choices=["ippo", "mappo"],
                   help="ippo (default): local-obs critic. mappo (SP3): joint-"
                        "observation critic. Ignored with --eval, which reads "
                        "the algorithm off the checkpoint instead.")
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--min-green", type=int, required=True,
                   help="explicit -- this script never falls back to $MIN_GREEN/DEFAULT_MIN_GREEN")
    p.add_argument("--eval", type=str, default=None,
                   help="path to a saved model to evaluate instead of training")
    p.add_argument("--tripinfo", action="store_true",
                   help="also write the per-trip XML (only meaningful with --eval)")
    args = p.parse_args()
    if args.eval:
        evaluate(args.eval, args.scenario, args.lam, args.seed, args.min_green,
                 args.steps, tripinfo=args.tripinfo)
    else:
        train(args.scenario, args.lam, args.seed, args.steps, args.min_green,
             centralized=(args.algo == "mappo"))
