#!/usr/bin/env bash
# Headless multi-seed runs for the demo claim: fixed-time vs actuated (non-learning) vs DQN, all at
# the corrected 60 s floor, peak demand, teleport=300 (the report's protocol). Then demo_stats.py.
# usage: ./demo_stats.sh [first_seed] [last_seed]     (default 42 51 = n=10)
set -e
cd "$(dirname "$0")"
source venv/bin/activate
export SUMO_HOME=$(python -c 'import sumo; print(sumo.SUMO_HOME)')
export EPISODE_SECONDS=${EPISODE_SECONDS:-1200} TIME_TO_TELEPORT=300
A=${1:-42}; B=${2:-51}
for s in $(seq $A $B); do
  echo "python baseline.py --scenario peak --seed $s --green 60"
  echo "python train.py --algo dqn --eval models/dqn_seed0.zip --scenario peak --seed $s --min-green 60"
  echo "python analysis/actuated.py --min-green 60 --seed $s --teleport 300"
done | xargs -P 4 -I{} sh -c '{} >/dev/null 2>&1 || echo "FAILED: {}"'
python demo_stats.py $A $B
