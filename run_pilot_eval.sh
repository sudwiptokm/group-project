#!/usr/bin/env bash
# Evaluate whatever pilot checkpoints exist, without retraining.
#
# Split out of run_pilot_mg60.sh because that script always trains first, so
# re-running it after a partial kill would overwrite the checkpoints that
# survived. Long background runs on this machine have now been SIGINT'd three
# times, so the recovery path has to be resumable and each stage short.
set -uo pipefail
cd "$(dirname "$0")"
[ -d venv ] && source venv/bin/activate
export SUMO_HOME="${SUMO_HOME:-$(python -c 'import sumo; print(sumo.SUMO_HOME)')}"

MIN_GREEN="${MIN_GREEN:-60}"
ALGOS="${ALGOS:-dqn ppo}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2}"
EVAL_SEEDS="${EVAL_SEEDS:-42 43 44 45 46}"
LAM="${LAM:-0.5}"
JOBS="${JOBS:-2}"          # 2, not 3: swap was at 4.2 GB of 5.1 GB after the kill
export MIN_GREEN
export EPISODE_SECONDS="${EPISODE_SECONDS:-3600}"
export TIME_TO_TELEPORT="${TIME_TO_TELEPORT:-300}"

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || sleep 2; done; }

eval_one() {
  local a="$1" t="$2" e="$3"
  local m="models/${a}_peak_lam05_seed${t}_mg${MIN_GREEN}.zip"
  local out="logs/eval_${a}_peak_lam05_seed${e}_t${t}_mg${MIN_GREEN}_conn0_ep1.csv"
  [ -f "$m" ] || { echo "skip  $a t$t (no checkpoint)"; return 0; }
  [ -f "$out" ] && { echo "have  $a t$t seed$e (already evaluated)"; return 0; }
  if python train.py --algo "$a" --eval "$m" --seed "$e" --scenario peak \
        --lam "$LAM" --train-seed "$t" --min-green "$MIN_GREEN" \
        > "logs/pilot_eval_${a}_t${t}_seed${e}.log" 2>&1
  then echo "done  $a t$t seed$e"; else echo "FAIL  $a t$t seed$e"; fi
}

for algo in $ALGOS; do
  for t in $TRAIN_SEEDS; do
    for e in $EVAL_SEEDS; do
      throttle
      eval_one "$algo" "$t" "$e" &
    done
  done
done
wait
echo "EVAL STAGE COMPLETE"
