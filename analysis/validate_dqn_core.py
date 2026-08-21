"""Head-to-head: hand-rolled dqn_core vs SB3 DQN, matched hyperparameters and
step budget, on the single-agent intersection env (scenario 'base').

dqn_core/train_corridor_dqn were built for the multi-agent corridor's
independent-per-agent training. The single-intersection env is single-agent
sumo-rl (Gymnasium-style: obs, reward, terminated, truncated), so this file
adapts dqn_core's QNetwork/ReplayBuffer/dqn_loss to that API directly rather
than reusing train_corridor_dqn's multi-agent training loop.

Not a pass/fail gate: a from-scratch reimplementation is not expected to
exactly match a mature library's sample efficiency. It reports held-out
tripinfo delay and wall-clock timing for both so the risk this comparison
exists to surface -- "is dqn_core's gradient step actually equivalent to
SB3's" -- has evidence either way before any corridor number is trusted.

    python -m analysis.validate_dqn_core --steps 100000 --seed 0
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

import dqn_core as dc
from algos import ALGOS, build
from analysis.tripinfo import reduce_tripinfo
from env_common import make_env, tripinfo_path

MIN_GREEN = 60  # env_common.DEFAULT_MIN_GREEN -- explicit here for the same
                # reason it must be explicit everywhere else in this project


def matched_hp() -> dict:
    """dqn_core's hyperparameter names <- algos.ALGOS['dqn']['defaults']().

    Plain SB3 DQN defaults, NOT train_corridor_dqn._hp()'s 3 disclosed
    overrides -- this validation isolates "is dqn_core's math right" from
    "are these the hyperparameters SP5's real corridor run uses."
    """
    d = ALGOS["dqn"]["defaults"]()
    return {
        "lr": d["learning_rate"], "buffer_size": d["buffer_size"],
        "learning_starts": d["learning_starts"], "batch_size": d["batch_size"],
        "gamma": d["gamma"], "train_freq": d["train_freq"],
        "target_update_interval": d["target_update_interval"],
        "exploration_fraction": d["exploration_fraction"],
        "exploration_final_eps": d["exploration_final_eps"],
        "hidden": tuple(d["policy_kwargs"]["net_arch"]),
    }


def train_dqn_core(seed: int, steps: int) -> str:
    """Single-agent DQN training loop using dqn_core, matched to SB3 DQN's
    hyperparameters."""
    hp = matched_hp()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env(seed=seed, scenario="base", min_green=MIN_GREEN)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.n
    q_net = dc.QNetwork(obs_dim, act_dim, hidden=hp["hidden"])
    target_net = dc.QNetwork(obs_dim, act_dim, hidden=hp["hidden"])
    target_net.load_state_dict(q_net.state_dict())
    optim = torch.optim.Adam(q_net.parameters(), lr=hp["lr"])
    buffer = dc.ReplayBuffer(hp["buffer_size"], obs_dim)
    eps_sched = dc.EpsilonSchedule(steps, hp["exploration_fraction"],
                                   hp["exploration_final_eps"])

    obs, _ = env.reset()
    for t in range(steps):
        eps = eps_sched.value(t)
        if rng.random() < eps:
            action = int(rng.integers(act_dim))
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = int(q_net(obs_t).argmax(dim=-1).item())
        nobs, reward, terminated, truncated, _ = env.step(action)
        done = float(terminated or truncated)
        buffer.add(obs, action, float(reward), nobs, done)
        obs = nobs
        if done:
            obs, _ = env.reset()

        if t >= hp["learning_starts"] and t % hp["train_freq"] == 0 \
                and len(buffer) >= hp["batch_size"]:
            b_obs, b_act, b_rew, b_nobs, b_done = buffer.sample(hp["batch_size"], rng)
            loss, _ = dc.dqn_loss(q_net, target_net, b_obs, b_act, b_rew, b_nobs,
                                  b_done, hp["gamma"])
            optim.zero_grad()
            loss.backward()
            optim.step()

        if t > 0 and t % hp["target_update_interval"] == 0:
            target_net.load_state_dict(q_net.state_dict())
    env.close()

    os.makedirs("models", exist_ok=True)
    path = f"models/validate_dqn_core_seed{seed}.pt"
    torch.save({"state_dict": q_net.state_dict(), "hidden": hp["hidden"]}, path)
    return path


def eval_dqn_core(model_path: str, seed: int) -> float:
    out_csv = f"logs/eval_validate_dqn_core_seed{seed}"
    env = make_env(seed=seed, scenario="base", min_green=MIN_GREEN,
                   out_csv=out_csv, tripinfo=True)
    ckpt = torch.load(model_path, weights_only=True)
    q_net = dc.QNetwork(env.observation_space.shape[0], env.action_space.n,
                        hidden=tuple(ckpt["hidden"]))
    q_net.load_state_dict(ckpt["state_dict"])
    q_net.eval()
    obs, _ = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            q = q_net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        obs, _, terminated, truncated, _ = env.step(int(q.argmax(dim=-1)[0]))
        done = terminated or truncated
    env.save_csv(env.out_csv_name, env.episode)
    env.close()
    return reduce_tripinfo(tripinfo_path(out_csv))["trip_time_loss_mean"]


def train_eval_sb3(seed: int, steps: int) -> float:
    params = ALGOS["dqn"]["defaults"]()
    env = Monitor(make_env(seed=seed, scenario="base", min_green=MIN_GREEN))
    model = build("dqn", env, params, seed=seed, tb_log="logs/tb")
    model.learn(total_timesteps=steps)
    env.close()

    out_csv = f"logs/eval_validate_sb3_dqn_seed{seed}"
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
    dqn_core_model = train_dqn_core(args.seed, args.steps)
    dqn_core_delay = eval_dqn_core(dqn_core_model, seed=args.seed + 1000)
    t1 = time.monotonic()
    sb3_delay = train_eval_sb3(args.seed, args.steps)
    t2 = time.monotonic()

    print(f"\ndqn_core:  delay/trip={dqn_core_delay:.1f}s  wall={t1 - t0:.0f}s")
    print(f"sb3 DQN:   delay/trip={sb3_delay:.1f}s  wall={t2 - t1:.0f}s")
    print(f"dqn_core - sb3: {dqn_core_delay - sb3_delay:+.1f}s")
