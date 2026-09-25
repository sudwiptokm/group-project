#!/usr/bin/env bash
# Two sumo-gui windows, same demand seed: fixed-time baseline (left) vs trained RL (right).
# usage: ./demo_side_by_side.sh [seed] [algo]
set -e
cd "$(dirname "$0")"
source venv/bin/activate
export SUMO_HOME=$(python -c 'import sumo; print(sumo.SUMO_HOME)')
SEED=${1:-42}; ALGO=${2:-dqn}; MG=${MG:-60}   # RL min-green floor: 60 = corrected fair setup; 10 = the report's defective Stage-1 setup
# README "Option A": X on :1, MIT-SHM off, indirect GLX on. Quit any running XQuartz first.
if [ ! -e /tmp/.X11-unix/X1 ]; then
  /opt/X11/bin/Xquartz :1 -extension MIT-SHM +iglx &
  until [ -e /tmp/.X11-unix/X1 ]; do sleep 1; done
  DISPLAY=:1 /opt/X11/bin/quartz-wm &
  sleep 2
fi
export DISPLAY=:1
export TIME_TO_TELEPORT=${TIME_TO_TELEPORT:-300}   # report's protocol; -1 lets gridlock lock forever
export EPISODE_SECONDS=${EPISODE_SECONDS:-1200}

SUMO_GUI_WINDOW="--window-size 720,700 --window-pos 0,50" \
  python baseline.py --scenario peak --seed $SEED --gui &
P1=$!
SUMO_GUI_WINDOW="--window-size 720,700 --window-pos 730,50" \
  python train.py --algo $ALGO --eval models/${ALGO}_seed0.zip --scenario peak --seed $SEED --min-green $MG --gui &
P2=$!
# non-learning reference (headless), same seed and episode length, for the summary table
python analysis/actuated.py --min-green $MG --seed $SEED --teleport $TIME_TO_TELEPORT > /dev/null 2>&1 &
P3=$!
wait $P1 $P2 $P3   # not bare `wait`: Xquartz/quartz-wm are children too and never exit
python demo_summary.py $SEED $ALGO peak $MG
