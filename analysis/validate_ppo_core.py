"""Head-to-head: hand-rolled ppo_core vs SB3 PPO, matched hyperparameters and
step budget, on the single-agent intersection env (scenario 'base').

ppo_core/train_corridor were built for the multi-agent corridor's dict-based
API (obs/reward/dones keyed by agent id). The single-intersection env is
single-agent sumo-rl (Gymnasium-style: obs, reward, terminated, truncated), so
this file adapts ppo_core's ActorCritic/compute_gae/ppo_loss to that API rather
than reusing train_corridor.collect_rollout/update directly.

Not a pass/fail gate: a from-scratch reimplementation is not expected to exactly
match a mature library's sample efficiency. It reports held-out tripinfo delay
and wall-clock timing for both so the risk this comparison exists to surface --
"is ppo_core's gradient step actually equivalent to SB3's" -- has evidence
either way before any corridor number is trusted.

    python -m analysis.validate_ppo_core --steps 100000 --seed 0
"""
import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor

import ppo_core as pc
from algos import ALGOS, build
from analysis.tripinfo import reduce_tripinfo
from env_common import make_env, tripinfo_path

MIN_GREEN = 60  # env_common.DEFAULT_MIN_GREEN -- explicit here for the same
                # reason it must be explicit everywhere else in this project


def matched_hp() -> dict:
    """ppo_core's hyperparameter names <- algos.ALGOS['ppo']['defaults']()."""
    d = ALGOS["ppo"]["defaults"]()
    return {
        "lr": d["learning_rate"], "n_steps": d["n_steps"],
        "batch_size": d["batch_size"], "n_epochs": d["n_epochs"],
        "gamma": d["gamma"], "gae_lambda": d["gae_lambda"],
        "clip_range": d["clip_range"], "ent_coef": d["ent_coef"],
        "hidden": tuple(d["policy_kwargs"]["net_arch"]),
    }


def train_ppo_core(seed: int, steps: int) -> str:
    """Single-agent PPO training loop using ppo_core, matched to SB3 PPO's
    hyperparameters. Mirrors train_corridor.collect_rollout/update but against
    the single-agent (obs, reward, terminated, truncated) API."""
    hp = matched_hp()
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = make_env(seed=seed, scenario="base", min_green=MIN_GREEN)
    policy = pc.ActorCritic(env.observation_space.shape[0], env.action_space.n,
                            hidden=hp["hidden"])
    optim = torch.optim.Adam(policy.parameters(), lr=hp["lr"])

    obs, _ = env.reset()
    collected = 0
    while collected < steps:
        buf = {k: [] for k in ("obs", "act", "logp", "rew", "val", "done")}
        for _ in range(hp["n_steps"]):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_t, logp_t = policy.act(obs_t)
                val_t = policy.value(obs_t)
            nobs, reward, terminated, truncated, _ = env.step(int(action_t[0]))
            done = terminated or truncated
            buf["obs"].append(obs_t[0]); buf["act"].append(action_t[0])
            buf["logp"].append(logp_t[0]); buf["val"].append(float(val_t[0]))
            buf["rew"].append(float(reward)); buf["done"].append(float(done))
            obs = nobs
            if done:
                obs, _ = env.reset()

        with torch.no_grad():
            last_val = float(policy.value(
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0])
        adv, ret = pc.compute_gae(buf["rew"], buf["val"], buf["done"],
                                  hp["gamma"], hp["gae_lambda"], last_value=last_val)
        obs_b = torch.stack(buf["obs"]); act_b = torch.stack(buf["act"])
        old_logp = torch.stack(buf["logp"]).detach()
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)

        n = obs_b.shape[0]
        idx = np.arange(n)
        for _ in range(hp["n_epochs"]):
            np.random.shuffle(idx)
            for start in range(0, n, hp["batch_size"]):
                b = idx[start:start + hp["batch_size"]]
                dist = policy.policy(obs_b[b]); vals = policy.value(obs_b[b])
                loss, _ = pc.ppo_loss(dist, act_b[b], old_logp[b], adv_t[b], vals,
                                      ret_t[b], clip=hp["clip_range"],
                                      ent_coef=hp["ent_coef"])
                optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optim.step()
        collected += hp["n_steps"]
    env.close()

    os.makedirs("models", exist_ok=True)
    path = f"models/validate_ppo_core_seed{seed}.pt"
    torch.save({"state_dict": policy.state_dict(), "hidden": hp["hidden"]}, path)
    return path


def eval_ppo_core(model_path: str, seed: int) -> float:
    out_csv = f"logs/eval_validate_ppo_core_seed{seed}"
    env = make_env(seed=seed, scenario="base", min_green=MIN_GREEN,
                   out_csv=out_csv, tripinfo=True)
    ckpt = torch.load(model_path, weights_only=True)
    policy = pc.ActorCritic(env.observation_space.shape[0], env.action_space.n,
                            hidden=tuple(ckpt["hidden"]))
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    obs, _ = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            logits = policy.actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        obs, _, terminated, truncated, _ = env.step(int(logits.argmax(dim=-1)[0]))
        done = terminated or truncated
    env.save_csv(env.out_csv_name, env.episode)
    env.close()
    return reduce_tripinfo(tripinfo_path(out_csv))["trip_time_loss_mean"]


def train_eval_sb3(seed: int, steps: int) -> float:
    params = ALGOS["ppo"]["defaults"]()
    env = Monitor(make_env(seed=seed, scenario="base", min_green=MIN_GREEN))
    model = build("ppo", env, params, seed=seed, tb_log="logs/tb")
    model.learn(total_timesteps=steps)
    env.close()

    out_csv = f"logs/eval_validate_sb3_ppo_seed{seed}"
    eval_env = make_env(seed=seed, scenario="base", min_green=MIN_GREEN,
                        out_csv=out_csv, tripinfo=True)
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
    eval_env.save_csv(eval_env.out_csv_name, eval_env.episode)
    eval_env.close()
    return reduce_tripinfo(tripinfo_path(out_csv))["trip_time_loss_mean"]


if __name__ == "__main__":
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.monotonic()
    ppo_core_model = train_ppo_core(args.seed, args.steps)
    ppo_core_delay = eval_ppo_core(ppo_core_model, seed=args.seed + 1000)
    t1 = time.monotonic()
    sb3_delay = train_eval_sb3(args.seed, args.steps)
    t2 = time.monotonic()

    print(f"\nppo_core:  delay/trip={ppo_core_delay:.1f}s  wall={t1 - t0:.0f}s")
    print(f"sb3 PPO:   delay/trip={sb3_delay:.1f}s  wall={t2 - t1:.0f}s")
    print(f"ppo_core - sb3: {ppo_core_delay - sb3_delay:+.1f}s")
