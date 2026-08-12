"""Capture a trained DQN agent controlling the PEAK-demand intersection, with a
live metrics HUD — the RL counterpart to capture_demo.py (fixed-time).

Builds the same shared env (PCU obs, safety reward) on the peak scenario with
sumo-gui, loads the tuned DQN model, and lets it drive the signal. SUMO runs at
0.25 s steps for smooth motion; a hook on simulationStep screenshots every step
and overlays live metrics. Output: /tmp/demo_frames_rl/seq_*.png -> mp4.
"""
import os
import sys
import glob
import sumo

os.environ.setdefault("SUMO_HOME", sumo.SUMO_HOME)
os.environ.setdefault("EPISODE_SECONDS", "3600")
import traci  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_common import (  # noqa: E402
    SafetyLoggingEnv, PCUObservationFunction, make_safety_reward_fn, SCENARIO_ROUTES,
)
from stable_baselines3 import DQN  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DEMO_CONTROLLER=dqn (default) or fixed — same env/rendering, different signal policy
CONTROLLER = os.environ.get("DEMO_CONTROLLER", "dqn")
HUD_TITLE = "DQN agent - peak demand" if CONTROLLER == "dqn" else "Fixed-time signal - peak demand"
FIXED_HOLD_DECISIONS = 6   # ~30 s green per phase at delta_time=5 (fixed-time baseline)
FRAMES = f"/tmp/demo_frames_{CONTROLLER}"
os.makedirs(FRAMES, exist_ok=True)
for f in glob.glob(f"{FRAMES}/*.png"):
    os.remove(f)

MODEL = os.path.join(ROOT, "models", "dqn_peak_lam05_seed0.zip")
SCENARIO = "peak"
FRAME_CAP = 300        # ~ frames to capture (0.25 s each -> ~75 s sim)
STEP_LENGTH = 0.25
EMERGENCY_DECEL = -4.5
VIEW = "View #0"

try:
    import matplotlib
    _p = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    FONT = ImageFont.truetype(os.path.join(_p, "DejaVuSans.ttf"), 26)
    FONT_B = ImageFont.truetype(os.path.join(_p, "DejaVuSans-Bold.ttf"), 30)
except Exception:
    FONT = FONT_B = ImageFont.load_default()

# NB: sumo-rl already appends --start/--quit-on-end for use_gui — don't repeat them
extra = ("--additional-files vtypes.add.xml --lateral-resolution 0.5 "
         "--gui-settings-file gui-settings.xml "
         f"--step-length {STEP_LENGTH} --collision.action warn")
env = SafetyLoggingEnv(
    net_file=os.path.join(ROOT, "intersection.net.xml"),
    route_file=os.path.join(ROOT, SCENARIO_ROUTES[SCENARIO]),
    observation_class=PCUObservationFunction,
    use_gui=True,
    num_seconds=1200,
    delta_time=5, yellow_time=3, min_green=10, max_green=60,
    reward_fn=make_safety_reward_fn(0.5),
    single_agent=True, sumo_seed=42, sumo_warnings=False,
    additional_sumo_cmd=extra,
)
model = DQN.load(MODEL) if CONTROLLER == "dqn" else None
nphases = env.action_space.n
obs, _ = env.reset()
conn = env.sumo  # traci connection sumo-rl opened

try:
    (x0, y0), (x1, y1) = conn.simulation.getNetBoundary()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    conn.gui.setBoundary(VIEW, cx - 70, cy - 70, cx + 70, cy + 70)
except Exception as e:
    print("zoom err", e, file=sys.stderr)

state = {"i": 0, "brakes": 0, "coll": 0, "departed": 0, "pending": None}


def collect():
    ids = conn.vehicle.getIDList()
    n = len(ids)
    speeds = [conn.vehicle.getSpeed(v) for v in ids]
    waits = [conn.vehicle.getWaitingTime(v) for v in ids]
    avg_wait = sum(waits) / n if n else 0.0
    avg_speed = sum(speeds) / n if n else 0.0
    queue = sum(1 for s in speeds if s < 0.1)
    brakes = sum(1 for v in ids if conn.vehicle.getAcceleration(v) <= EMERGENCY_DECEL)
    return n, avg_wait, avg_speed, queue, brakes


def draw_hud(path, m):
    im = Image.open(path).convert("RGB")
    im.load()
    if im.width < 2 or im.height < 2:
        return False
    d = ImageDraw.Draw(im, "RGBA")
    coll_pct = (state["coll"] / state["departed"] * 100.0) if state["departed"] else 0.0
    lines = [
        (HUD_TITLE, None),
        (f"Sim time:        {m['t']:6.1f} s", None),
        (f"Vehicles active: {m['n']:4d}", None),
        (f"Avg wait time:   {m['w']:7.1f} s", (255, 210, 90)),
        (f"Queue (stopped): {m['q']:4d}", (255, 210, 90)),
        (f"Avg speed:       {m['s']:6.2f} m/s", (150, 220, 150)),
        (f"Emerg. brakes:   {state['brakes']:4d}", (255, 150, 120)),
        (f"Collisions:      {state['coll']:4d}  ({coll_pct:.2f}%)", (255, 150, 120)),
    ]
    pad, lh, w = 16, 34, 430
    h = pad * 2 + lh * len(lines)
    d.rectangle([12, 12, 12 + w, 12 + h], fill=(15, 22, 33, 205))
    y = 12 + pad
    for text, color in lines:
        f = FONT_B if text == HUD_TITLE else FONT
        d.text((28, y), text, font=f, fill=color or (235, 240, 245))
        y += lh
    im.save(path)
    return True


orig_step = conn.simulationStep


def hooked_step(*a, **k):
    orig_step(*a, **k)  # advance sim (also flushes the previous pending screenshot)
    p = state["pending"]
    if p and os.path.exists(p[0]):
        draw_hud(p[0], p[1])
    state["pending"] = None
    state["departed"] += conn.simulation.getDepartedNumber()
    state["coll"] += conn.simulation.getCollidingVehiclesNumber()
    if state["i"] < FRAME_CAP:
        n, w, s, q, b = collect()
        state["brakes"] += b
        m = {"t": conn.simulation.getTime(), "n": n, "w": w, "s": s, "q": q}
        path = f"{FRAMES}/frame_{state['i']:04d}.png"
        conn.gui.screenshot(VIEW, path)
        state["pending"] = (path, m)
        state["i"] += 1


conn.simulationStep = hooked_step

done = False
dcount = 0
while not done and state["i"] < FRAME_CAP:
    if CONTROLLER == "dqn":
        action, _ = model.predict(obs, deterministic=True)
    else:
        # fixed-time baseline: round-robin the green phases on a fixed timer
        action = (dcount // FIXED_HOLD_DECISIONS) % nphases
    obs, reward, terminated, truncated, _ = env.step(action)
    dcount += 1
    done = terminated or truncated
# flush the last pending screenshot with one more raw step
try:
    orig_step()
    p = state["pending"]
    if p and os.path.exists(p[0]):
        draw_hud(p[0], p[1])
except Exception:
    pass
env.close()

frames = sorted(glob.glob(f"{FRAMES}/*.png"))
print(f"captured {len(frames)} frames")
for f in frames:
    try:
        im = Image.open(f); im.load()
        if im.width < 2 or im.height < 2:
            os.remove(f)
    except Exception:
        os.remove(f)
good = sorted(glob.glob(f"{FRAMES}/*.png"))
for i, f in enumerate(good):
    dst = f"{FRAMES}/seq_{i:04d}.png"
    if f != dst:
        os.rename(f, dst)
print(f"kept {len(good)} clean frames as seq_%04d.png")
