"""One live sumo-gui corridor run with the metric HUD, for the side-by-side corridor demo.

    python corridor_demo.py --controller {green_wave,max_pressure,idqn} [--seed 42]
                            [--net corridor.net.xml] [--headless]

All three run the same episode (corridor_peak demand, 10 s floor -- this network's own
measured optimum, not the single junction's 60 s). idqn loads the frozen 100k-step
checkpoint trained on the REGULAR net (lam=0.5); on another --net it is zero-shot,
exactly as in SP8-SP13e. --headless skips the GUI and HUD (summary-table run).
"""
import argparse
import os

import torch

import corridor_baseline as cb
import dqn_core as dc
from env_common import make_corridor_env

CKPT = "models/idqn_{i}_corridor_peak_lam05_seed{s}_mg10_s100000.pt"
LABEL = {"green_wave": "FIXED GREEN WAVE  (offset-coordinated plan, no learning)",
         "max_pressure": "MAX-PRESSURE  (reactive rule, no learning)",
         "idqn": "IDQN  (3 independent learned policies, frozen)"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", required=True, choices=list(LABEL))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--net", default="corridor.net.xml")
    p.add_argument("--ckpt-seed", type=int, default=42, help="which trained IDQN seed to load")
    p.add_argument("--headless", action="store_true")
    a = p.parse_args()
    gui = not a.headless
    net_tag = "" if a.net == "corridor.net.xml" else "_net" + a.net.removesuffix(".net.xml").removeprefix("corridor_")
    out = f"logs/demo_corridor_{a.controller}_seed{a.seed}{net_tag}"
    os.makedirs("logs", exist_ok=True)
    env = make_corridor_env(seed=a.seed, scenario="corridor_peak", lam=0.0, out_csv=out,
                            min_green=10, tripinfo=True, gui=gui, net_file=a.net)
    obs = env.reset()
    policies = {}
    if a.controller == "idqn":
        obs_dim, act_dim = (env.observation_spaces(env.ts_ids[0]).shape[0],
                            env.action_spaces(env.ts_ids[0]).n)
        for i in env.ts_ids:
            ck = torch.load(CKPT.format(i=i, s=a.ckpt_seed), weights_only=True)
            q = dc.QNetwork(obs_dim, act_dim, hidden=tuple(ck["hidden"]))
            q.load_state_dict(ck["state_dict"])
            q.eval()
            policies[i] = q
    hud = None
    if gui:
        from demo_hud import Hud
        sub = f"peak demand, seed {a.seed}, 10 s floor" + (
            "" if not net_tag else f", {net_tag[4:]} spacing")
        hud = Hud(env, LABEL[a.controller], sub, x_off=250.0 if not net_tag else 380.0)   # wide title clears the legend bar
    done, info = False, {}
    while not done:
        if a.controller == "green_wave":
            acts = cb.green_wave_actions(env, net_file=a.net)
        elif a.controller == "max_pressure":
            acts = cb._max_pressure_actions(env)
        else:
            acts = {}
            for i in env.ts_ids:
                with torch.no_grad():
                    o = torch.as_tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                    acts[i] = int(policies[i](o).argmax(dim=-1).item())
        obs, _, dones, info = env.step(acts)
        done = dones["__all__"]
        if hud:
            hud.update(info, done)
    env.save_csv(env.out_csv_name, env.episode)
    if hud:
        hud.hold()
    env.close()


if __name__ == "__main__":
    main()
