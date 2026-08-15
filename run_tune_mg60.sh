#!/usr/bin/env bash
# Re-tune hyperparameters at the corrected action-space floor (min_green 60).
#
# WHY THIS RUN EXISTS (docs/FINDINGS_2026-08-12.md, next step 5). Every peak
# params/*.json was selected against a 10 s floor -- the floor the actuated
# probe showed no controller can win at (517.5 s/trip vs 91.8 s for a fixed
# plan) -- and for a 100k budget. The mg-60 pilot then ran --defaults at 30k.
# So the pilot's losses are against hyperparameters that were never chosen for
# this action space. That is the cheapest remaining explanation for the deficit
# and it has to be eliminated before a full-budget retrain is worth its cost.
#
# What the resulting params must eventually beat, on delay per completed trip
# at peak (seeds 42-46):
#     static 60 s plan          91.8 +/- 19.9 s
#     queue-actuated, mg 60     82.5 +/- 10.1 s   <- the real bar
#
# PROTOCOL. Identical to the tuning that produced the current params/*.json
# (30 trials x 20k steps, Optuna TPE, objective = mean eval waiting time) with
# two deliberate changes:
#   1. --min-green 60. The point of the run.
#   2. Selection seeds 7 8, NOT the default 42 43. The final comparison scores
#      on 42-46, so tuning on 42/43 would select hyperparameters on two of the
#      five seeds it is later judged by. The old params have that leak; there
#      is no reason to reproduce it in the file that replaces them.
# The environment matches the retrain exactly (3600 s episodes, teleport 300,
# lam 0.5) -- HPs selected under one episode length do not transfer either.
#
# COST. ~30 min of wall clock per trial (20k steps at ~10 it/s, plus two eval
# episodes), so 30 trials x 2 algorithms is ~30 CPU-hours. With WORKERS_PER_ALGO
# workers sharing each study that is roughly 30 / (2 * WORKERS_PER_ALGO) hours.
#
# INTERRUPTION. Background jobs on this machine get SIGINT'd at unpredictable
# intervals, cause unknown (see docs). The study lives in params/<study>.db, so
# --trials is a target for the STUDY and a restarted worker tops it up instead
# of starting over. Each worker is wrapped in a retry loop; re-running this
# script after a kill is always safe and never repeats completed trials.
set -uo pipefail
cd "$(dirname "$0")"
[ -d venv ] && source venv/bin/activate
export SUMO_HOME="${SUMO_HOME:-$(python -c 'import sumo; print(sumo.SUMO_HOME)')}"

MIN_GREEN="${MIN_GREEN:-60}"
TRIALS="${TRIALS:-30}"
STEPS="${STEPS:-20000}"
ALGOS="${ALGOS:-dqn ppo}"
SCENARIO="${SCENARIO:-peak}"
LAM="${LAM:-0.5}"
TUNE_EVAL_SEEDS="${TUNE_EVAL_SEEDS:-7 8}"
WORKERS_PER_ALGO="${WORKERS_PER_ALGO:-3}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"

export EPISODE_SECONDS="${EPISODE_SECONDS:-3600}"
export TIME_TO_TELEPORT="${TIME_TO_TELEPORT:-300}"
# Per-eval-episode wall cap. The default 120 s was set for 10 s-floor episodes;
# a healthy 3600 s peak episode already takes ~70 s alone and several workers
# share the machine, so 120 s would prune trials for being scheduled against
# load rather than for being bad policies. TIME_TO_TELEPORT=300 stops a real
# gridlock from crawling forever, so a generous cap costs little.
export EVAL_WALL_CAP="${EVAL_WALL_CAP:-900}"
# One BLAS thread per worker: these are small MLPs, and letting each process
# grab every core makes N workers slower than one.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "tune: min_green=$MIN_GREEN trials=$TRIALS steps=$STEPS algos='$ALGOS'"
echo "      scenario=$SCENARIO lam=$LAM select_seeds='$TUNE_EVAL_SEEDS'"
echo "      episode=${EPISODE_SECONDS}s teleport=$TIME_TO_TELEPORT wall_cap=${EVAL_WALL_CAP}s"
echo "      workers/algo=$WORKERS_PER_ALGO"

mkdir -p logs params

worker() {
  local algo="$1" idx="$2"
  local log="logs/tune_${algo}_${SCENARIO}_mg${MIN_GREEN}_w${idx}.log"
  local attempt=1
  while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    echo "TUNE start $algo w$idx (attempt $attempt)"
    if python tune.py --algo "$algo" --scenario "$SCENARIO" --lam "$LAM" \
          --trials "$TRIALS" --steps "$STEPS" --min-green "$MIN_GREEN" \
          --eval-seeds $TUNE_EVAL_SEEDS --sampler-seed "$idx" \
          >> "$log" 2>&1
    then echo "TUNE done  $algo w$idx"; return 0; fi
    # Non-zero means killed mid-trial (or a genuine error). Completed trials are
    # already in the study db, so restarting resumes rather than repeats.
    echo "TUNE retry $algo w$idx (exit $?, see $log)"
    attempt=$((attempt + 1))
    sleep 5
  done
  echo "TUNE GAVE UP $algo w$idx after $MAX_ATTEMPTS attempts"
  return 1
}

for algo in $ALGOS; do
  for i in $(seq 1 "$WORKERS_PER_ALGO"); do
    worker "$algo" "$i" &
  done
done
wait

echo
echo "TUNING COMPLETE — params written:"
for algo in $ALGOS; do
  f="params/${algo}_${SCENARIO}.json"
  [ -f "$f" ] && echo "  $f (min_green $(python -c "import json;print(json.load(open('$f')).get('_min_green'))"))"
done
echo "next: full-budget retrain at the same floor, scored against the actuated"
echo "      controller (82.5 +/- 10.1 s), not the static plan."
