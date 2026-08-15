#!/usr/bin/env bash
# Run the mg-60 searches one algorithm at a time, DQN first.
#
# WHY SEQUENTIAL, AND WHY ONLY THREE WORKERS. Measured here, not guessed: this
# is an 8 GB machine with 4 performance cores. Six concurrent workers averaged
# a third of a core each, drove swap to 2.4 GB of 3, and completed nothing in
# five hours. Three concurrent SUMO+SB3 jobs is the configuration the mg-60
# pilot already ran successfully, so that is the ceiling. With the machine
# usable at all, a 20k-step trial is ~40 minutes, so 30 trials across 3 workers
# is ~7 hours per algorithm. Running both at once would make that ~15 hours with
# nothing usable until the end; running them in order answers DQN first.
#
# WHY DQN FIRST. It is the arm with something to gain: at the corrected floor
# DQN scored 88.3 +/- 8.0 s against the actuated bar of 82.5 +/- 10.1, while PPO
# was 30 s behind it. If re-tuned DQN still loses, PPO's result adds little; if
# it wins, PPO matters and runs next. No arm is dropped -- only deferred.
set -uo pipefail
trap '' INT HUP                 # same group-kill survival as run_tune_mg60.sh
cd "$(dirname "$0")"
[ -d venv ] && source venv/bin/activate

MIN_GREEN="${MIN_GREEN:-60}"
SCENARIO="${SCENARIO:-peak}"
TRIALS="${TRIALS:-30}"
ORDER="${ORDER:-dqn ppo}"
WORKERS="${WORKERS:-3}"    # see the header: more than 3 swaps this machine to a halt

completed() {   # completed trials in a study, 0 if it does not exist yet
  python - "$1" <<'PY' 2>/dev/null || echo 0
import sys, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
algo, scenario, mg = sys.argv[1].split(":")
name = f"{algo}_{scenario}_mg{mg}"
try:
    s = optuna.load_study(study_name=name, storage=f"sqlite:///params/{name}.db")
    print(sum(1 for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE))
except Exception:
    print(0)
PY
}

for algo in $ORDER; do
  n=$(completed "${algo}:${SCENARIO}:${MIN_GREEN}")
  if [ "${n:-0}" -ge "$TRIALS" ]; then
    echo "CHAIN skip $algo — study already has $n/$TRIALS trials"
    continue
  fi
  echo "CHAIN start $algo at $(date +%H:%M) ($n/$TRIALS trials done)"
  ALGOS="$algo" WORKERS_PER_ALGO="$WORKERS" TRIALS="$TRIALS" \
    MIN_GREEN="$MIN_GREEN" SCENARIO="$SCENARIO" ./run_tune_mg60.sh
  echo "CHAIN done  $algo at $(date +%H:%M) ($(completed "${algo}:${SCENARIO}:${MIN_GREEN}")/$TRIALS trials)"
done

echo "CHAIN COMPLETE — params written for: $ORDER"
echo "next: full-budget retrain at min_green $MIN_GREEN with these params,"
echo "      scored against the actuated controller (82.5 +/- 10.1 s)."
