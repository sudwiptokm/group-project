"""Capture a sumo-gui demo of the intersection with a LIVE metrics HUD.

Runs the fixed-time SUMO config (heterogeneous traffic + nice gui-settings)
under sumo-gui via TraCI, screenshots the canvas each step, and overlays live
metrics (avg wait, queue, speed, emergency brakes, collisions) computed from
TraCI onto every frame with PIL. Requires an X display (e.g. XQuartz :1).
Output: contiguous /tmp/demo_frames/seq_*.png -> encode to mp4 with ffmpeg.
"""
import os
import sys
import glob
import sumo

os.environ.setdefault("SUMO_HOME", sumo.SUMO_HOME)
import traci  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = "/tmp/demo_frames"
os.makedirs(FRAMES, exist_ok=True)
for f in glob.glob(f"{FRAMES}/*.png"):
    os.remove(f)

N_FRAMES = 260        # number of screenshots
STEPS_PER_FRAME = 1   # sim steps between screenshots
STEP_LENGTH = 0.25    # seconds of sim time per step -> small, realistic moves/frame
EMERGENCY_DECEL = -4.5  # m/s^2 threshold counted as an emergency brake

# a legible truetype font (ships with matplotlib)
try:
    import matplotlib
    _font_path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
    _font_bold = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")
    FONT = ImageFont.truetype(_font_path, 26)
    FONT_B = ImageFont.truetype(_font_bold, 30)
except Exception:
    FONT = FONT_B = ImageFont.load_default()

sumo_binary = os.path.join(sumo.SUMO_HOME, "bin", "sumo-gui")
cmd = [
    sumo_binary,
    "-c", os.path.join(ROOT, "intersection.sumocfg"),
    "--start", "--quit-on-end",
    "--step-length", str(STEP_LENGTH),
    "--collision.action", "warn",   # report collisions instead of teleporting
    "--window-size", "1400,950",
    "--delay", "0",
]
traci.start(cmd)
view = traci.gui.DEFAULT_VIEW
try:
    (x0, y0), (x1, y1) = traci.simulation.getNetBoundary()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = 70.0
    traci.gui.setBoundary(view, cx - half, cy - half, cx + half, cy + half)
except Exception as e:
    print("zoom err", e, file=sys.stderr)


def collect_metrics():
    """Instantaneous + cumulative metrics from the current sim state."""
    ids = traci.vehicle.getIDList()
    n = len(ids)
    waits = [traci.vehicle.getWaitingTime(v) for v in ids]
    speeds = [traci.vehicle.getSpeed(v) for v in ids]
    avg_wait = sum(waits) / n if n else 0.0
    avg_speed = sum(speeds) / n if n else 0.0
    queue = sum(1 for s in speeds if s < 0.1)
    brakes = sum(1 for v in ids if traci.vehicle.getAcceleration(v) <= EMERGENCY_DECEL)
    colliding = traci.simulation.getCollidingVehiclesNumber()
    return n, avg_wait, avg_speed, queue, brakes, colliding


def draw_hud(path, t, active, avg_wait, avg_speed, queue, brakes_cum, coll_cum, departed_cum):
    im = Image.open(path).convert("RGB")
    im.load()
    if im.width < 2 or im.height < 2:
        return False
    d = ImageDraw.Draw(im, "RGBA")
    coll_pct = (coll_cum / departed_cum * 100.0) if departed_cum else 0.0
    lines = [
        ("Fixed-time signal - live metrics", None),
        (f"Sim time:        {t:6.1f} s", None),
        (f"Vehicles active: {active:4d}", None),
        (f"Avg wait time:   {avg_wait:6.2f} s", (255, 210, 90)),
        (f"Queue (stopped): {queue:4d}", (255, 210, 90)),
        (f"Avg speed:       {avg_speed:6.2f} m/s", (150, 220, 150)),
        (f"Emerg. brakes:   {brakes_cum:4d}", (255, 150, 120)),
        (f"Collisions:      {coll_cum:4d}  ({coll_pct:.2f}%)", (255, 150, 120)),
    ]
    pad, lh = 16, 34
    w = 430
    h = pad * 2 + lh * len(lines)
    d.rectangle([12, 12, 12 + w, 12 + h], fill=(15, 22, 33, 205))
    y = 12 + pad
    for text, color in lines:
        f = FONT_B if color is None and text.startswith("Fixed") else FONT
        d.text((28, y), text, font=f, fill=color or (235, 240, 245))
        y += lh
    im.save(path)
    return True


saved = 0
brakes_cum = 0
coll_cum = 0
departed_cum = 0
for i in range(N_FRAMES):
    for _ in range(STEPS_PER_FRAME):
        if traci.simulation.getMinExpectedNumber() <= 0:
            break
        traci.simulationStep()
    departed_cum += traci.simulation.getDepartedNumber()
    active, avg_wait, avg_speed, queue, brakes, colliding = collect_metrics()
    brakes_cum += brakes
    coll_cum += colliding
    t = traci.simulation.getTime()
    path = f"{FRAMES}/frame_{i:04d}.png"
    try:
        traci.gui.screenshot(view, path)
        traci.simulationStep()  # let the screenshot flush to disk
        departed_cum += traci.simulation.getDepartedNumber()
        if os.path.exists(path):
            if draw_hud(path, t, active, avg_wait, avg_speed, queue, brakes_cum, coll_cum, departed_cum):
                saved += 1
    except Exception as e:
        print("frame err", e, file=sys.stderr)
    if traci.simulation.getMinExpectedNumber() <= 0:
        break
traci.close()

frames = sorted(glob.glob(f"{FRAMES}/*.png"))
print(f"captured {len(frames)} frames (saved counter {saved})")
if not frames:
    sys.exit("no frames captured")

# drop any zero-dim/corrupt screenshots, then renumber contiguously for ffmpeg
for f in frames:
    try:
        im = Image.open(f)
        im.load()
        if im.width < 2 or im.height < 2:
            os.remove(f)
    except Exception:
        os.remove(f)
good = sorted(glob.glob(f"{FRAMES}/*.png"))
kept = 0
for i, f in enumerate(good):
    dst = f"{FRAMES}/seq_{i:04d}.png"
    if f != dst:
        os.rename(f, dst)
    kept += 1
print(f"kept {kept} clean frames as seq_%04d.png")
