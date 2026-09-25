#!/usr/bin/env bash
# Corridor demo: two sumo-gui windows, same demand seed: fixed green wave (left) vs IDQN (right).
# Max-pressure runs headless for the summary table.
# usage: ./demo_corridor.sh [seed] [regular|irregular]
#   regular   200 m/200 m spacing  -> green wave wins (the report's headline)
#   irregular 578 m/78 m spacing   -> IDQN edges the green wave (SP8/SP9, n=10, 10/10 seeds)
# env: EPISODE_SECONDS (default 3600 = the report's episode; 1200 for a shorter live run),
#      DEMO_HOLD (seconds windows stay open at the end, default 600)
set -e
cd "$(dirname "$0")"
source venv/bin/activate
export SUMO_HOME=$(python -c 'import sumo; print(sumo.SUMO_HOME)')
SEED=${1:-42}; GEOM=${2:-regular}
case $GEOM in
  regular)   NET=corridor.net.xml;           TAG="" ;;
  irregular) NET=corridor_irregular.net.xml; TAG=irregular ;;
  *) echo "geometry must be regular|irregular"; exit 1 ;;
esac
# README "Option A": X on :1, MIT-SHM off, indirect GLX on. Quit any running XQuartz first.
if [ ! -e /tmp/.X11-unix/X1 ]; then
  /opt/X11/bin/Xquartz :1 -extension MIT-SHM +iglx &
  until [ -e /tmp/.X11-unix/X1 ]; do sleep 1; done
  DISPLAY=:1 /opt/X11/bin/quartz-wm &
  sleep 2
fi
export DISPLAY=:1
export TIME_TO_TELEPORT=${TIME_TO_TELEPORT:--1}
export EPISODE_SECONDS=${EPISODE_SECONDS:-3600}

SUMO_GUI_WINDOW="--window-size 720,520 --window-pos 0,50" \
  python corridor_demo.py --controller green_wave --seed $SEED --net $NET > /dev/null 2>&1 &
P1=$!
SUMO_GUI_WINDOW="--window-size 720,520 --window-pos 730,50" \
  python corridor_demo.py --controller idqn --seed $SEED --net $NET > /dev/null 2>&1 &
P2=$!
python corridor_demo.py --controller max_pressure --seed $SEED --net $NET --headless > /dev/null 2>&1 &
P3=$!
wait $P1 $P2 $P3   # not bare `wait`: Xquartz/quartz-wm are children too and never exit
python corridor_demo_summary.py $SEED $TAG
