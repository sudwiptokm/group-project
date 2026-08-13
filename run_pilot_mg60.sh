#!/usr/bin/env bash
# Pilot retrain at the corrected action-space floor.
#
# Every peak training run this project did used min_green=10, and
# analysis/actuated.py showed that floor is unwinnable: a controller with
# perfect queue information and nothing to learn is 5.6x worse than a fixed
# plan there. So the peak null is over-determined and says nothing about RL.
# This is the smallest run that can say something.
#
# What it must beat, on delay per completed trip at peak (seeds 42-46):
#     static 60 s plan          91.8 +/- 19.9 s
#     queue-actuated, mg 60     82.5 +/- 10.1 s   <- the real bar
# Matching the static plan is not a result; the actuated controller already
# does that without training. Only beating the actuated row means anything.
#
# Deliberate choices:
#   * 3600 s episodes throughout. Peak 1.5x is oversaturated, so the queue
#     deepens over the hour -- training on short episodes would never show the
#     agent the backlog states it is evaluated on.
#   * --defaults, not params/*.json. Those were tuned for a 100k budget
#     (lr 2.3e-5, learning_starts 5000) and burn a 30k budget on warmup.
#   * dqn and ppo only: the two already probed at 20k, so a change here is
#     attributable to the floor rather than to a new algorithm.
#
# KNOWN LIMITATION, state it with any result: 30k steps at 3600 s episodes is
# ~42 episodes. A positive result at this budget is informative; a null is
# weak, because it cannot distinguish "no signal" from "too few episodes".
set -uo pipefail
cd "$(dirname "$0")"
[ -d venv ] && source venv/bin/activate
export SUMO_HOME="${SUMO_HOME:-$(python -c 'import sumo; print(sumo.SUMO_HOME)')}"

MIN_GREEN="${MIN_GREEN:-60}"
STEPS="${STEPS:-30000}"
ALGOS="${ALGOS:-dqn ppo}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2}"
EVAL_SEEDS="${EVAL_SEEDS:-42 43 44 45 46}"
LAM="${LAM:-0.5}"
JOBS="${JOBS:-3}"
export MIN_GREEN
export EPISODE_SECONDS="${EPISODE_SECONDS:-3600}"
export TIME_TO_TELEPORT="${TIME_TO_TELEPORT:-300}"

echo "pilot: min_green=$MIN_GREEN steps=$STEPS algos='$ALGOS' seeds='$TRAIN_SEEDS'"
echo "       episode=${EPISODE_SECONDS}s teleport=$TIME_TO_TELEPORT lam=$LAM jobs=$JOBS"

mkdir -p logs models

# Plain job control rather than xargs -I: macOS xargs refuses to assemble a
# replacement line this long ("command line cannot be assembled, too long").
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || sleep 2; done; }

train_one() {
  local a="$1" s="$2"
  # Resumable: three long background runs have been SIGINT'd on this machine,
  # and completed checkpoints survive a kill. Skipping them means a restart
  # costs the interrupted run rather than the whole arm. FORCE=1 overrides.
  local ckpt="models/${a}_peak_lam05_seed${s}_mg${MIN_GREEN}.zip"
  if [ -f "$ckpt" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "TRAIN have  $a seed$s (checkpoint exists)"; return 0
  fi
  if python train.py --algo "$a" --seed "$s" --steps "$STEPS" --scenario peak \
        --lam "$LAM" --defaults --min-green "$MIN_GREEN" \
        > "logs/pilot_train_${a}_seed${s}.log" 2>&1
  then echo "TRAIN done  $a seed$s"; else echo "TRAIN FAIL  $a seed$s (see logs/pilot_train_${a}_seed${s}.log)"; fi
}

eval_one() {
  local a="$1" t="$2" e="$3"
  local m="models/${a}_peak_lam05_seed${t}_mg${MIN_GREEN}.zip"
  if [ ! -f "$m" ]; then echo "EVAL skip  $a t$t (no checkpoint)"; return 0; fi
  if python train.py --algo "$a" --eval "$m" --seed "$e" --scenario peak \
        --lam "$LAM" --train-seed "$t" --min-green "$MIN_GREEN" \
        > "logs/pilot_eval_${a}_t${t}_seed${e}.log" 2>&1
  then echo "EVAL done  $a t$t seed$e"; else echo "EVAL FAIL  $a t$t seed$e (see logs/pilot_eval_${a}_t${t}_seed${e}.log)"; fi
}

# ---- train -----------------------------------------------------------------
for algo in $ALGOS; do
  for s in $TRAIN_SEEDS; do
    throttle
    echo "TRAIN start $algo seed$s"
    train_one "$algo" "$s" &
  done
done
wait

# ---- evaluate --------------------------------------------------------------
# Each checkpoint on every held-out demand seed. --train-seed tags the eval CSV
# so several checkpoints can share a demand seed without overwriting.
for algo in $ALGOS; do
  for t in $TRAIN_SEEDS; do
    for e in $EVAL_SEEDS; do
      throttle
      eval_one "$algo" "$t" "$e" &
    done
  done
done
wait

echo "PILOT COMPLETE — now: python compare.py ; python analysis/headroom.py"
