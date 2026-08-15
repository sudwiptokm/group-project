# Live demo guide — sumo-gui

**Primary demo: `presentation/demo_dqn_peak.mp4`** — the trained DQN agent
controlling the peak-demand intersection, with a live metrics HUD. Prefer the
recorded clip; sumo-gui over macOS X11 plays back laggy (software OpenGL).

## IMPORTANT — how to frame the demo

The clip is **qualitative**: it shows the agent working. Its HUD numbers are
*instantaneous* over a ~75 s window. The headline **−24%** result is an
**episode-mean over a full 3600 s run** — present that from the plots
(`results/bars_peak_lam05.png`, `results/improvement_peak_lam05.png`), **not** the
clip's live numbers.

**Do NOT show a live fixed-time-vs-DQN race.** On a short early window the
instantaneous avg-wait can invert the aggregate result (peak congestion hasn't
built yet, and DQN peak has high seed variance ±402). It would look like
fixed-time beats the agent. The rigorous comparison is the aggregate table/plots.

## Option A (recommended): play the recorded clip

`presentation/demo_dqn_peak.mp4` — 30 s, 1390×764, zoomed 4-arm intersection,
heterogeneous traffic (orange moto, blue auto, white/grey cars), red/green
stop-line phases, live HUD. Recorded at 0.25 s sim-steps (smooth). H264/yuv420p →
embeds in PowerPoint/Keynote.

Also available: `presentation/demo.mp4` — fixed-time signal on the base scenario
with the same HUD, usable as a neutral "here is the environment" intro (no
comparison claim attached).

### Rebuild the clips

```bash
# start ONE X server on :1, shared memory OFF (avoids BadShmSeg blank window)
pkill -9 -f Xquartz; rm -f /tmp/.X1-lock /tmp/.X11-unix/X1   # clean any stale server
/opt/X11/bin/Xquartz :1 -extension MIT-SHM &
DISPLAY=:1 /opt/X11/bin/quartz-wm &          # after /tmp/.X11-unix/X1 appears

source venv/bin/activate
export SUMO_HOME=$(python -c 'import sumo; print(sumo.SUMO_HOME)')

# trained DQN agent on peak (the primary demo)
DISPLAY=:1 python presentation/capture_demo_rl.py           # -> /tmp/demo_frames_dqn/seq_*.png
ffmpeg -y -framerate 10 -i /tmp/demo_frames_dqn/seq_%04d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 18 presentation/demo_dqn_peak.mp4

# fixed-time environment intro (base scenario)
DISPLAY=:1 python presentation/capture_demo.py             # -> /tmp/demo_frames/seq_*.png
ffmpeg -y -framerate 10 -i /tmp/demo_frames/seq_%04d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 18 presentation/demo.mp4
```

`capture_demo_rl.py` also has a `DEMO_CONTROLLER=fixed` mode (round-robin signal)
— but do not present it against the DQN clip (see caveat above).

Slower/faster: change `-framerate` (lower = slower). Finer motion: lower
`STEP_LENGTH` in the capture script.

## Option B (live, backup): trained agent in sumo-gui

```bash
/opt/X11/bin/Xquartz :1 -extension MIT-SHM &
DISPLAY=:1 /opt/X11/bin/quartz-wm &
source venv/bin/activate
export SUMO_HOME=$(python -c 'import sumo; print(sumo.SUMO_HOME)')
DISPLAY=:1 python train.py --algo dqn --eval models/dqn_peak_lam05_seed0.zip --seed 42 --gui
```

- Hit **Run** (▶) if it doesn't auto-start; raise the **Delay (ms)** box to slow it.
- Blank window + `BadShmSeg` spam → you're on `:0`. Use `:1` with `-extension MIT-SHM`.
- `Fontconfig error` on launch is harmless.

## Talking points while it plays

- Vehicles are **PCU-weighted** in the agent's observation — 3 motorbikes ≠ 3
  cars in road-space.
- Watch queues build on the red approaches; the agent switches phase to clear the
  heaviest PCU-weighted queue.
- Red/green bars on the stop lines = current signal phase.
- HUD tracks avg wait, queue, speed, emergency brakes, collisions (≈0 — SUMO
  enforces safe car-following; safety is measured via emergency braking instead).
