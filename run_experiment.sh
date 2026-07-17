#!/usr/bin/env bash
#
# Full experiment driver: tune -> train (multi-seed) -> eval (held-out) -> compare.
#
# Runs the whole algorithm ladder unattended. Resumable: existing params/models/
# eval CSVs are skipped, so re-running continues where it left off. One failed run
# is logged and skipped — it does not abort the batch.
#
#   ./run_experiment.sh                 # full ladder, default budgets
#   ./run_experiment.sh --skip-tune     # use current params/ (or defaults)
#   ALGOS="dqn ppo" ./run_experiment.sh # subset of algorithms
#   STEPS=50000 TRAIN_SEEDS="0 1" ./run_experiment.sh   # smaller run
#   ./run_experiment.sh --force         # ignore existing artifacts, redo everything
#
# Config via env vars (defaults below).

set -uo pipefail

# ---- config -----------------------------------------------------------------
ALGOS="${ALGOS:-dqn qrdqn ppo a2c}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2}"
EVAL_SEEDS="${EVAL_SEEDS:-42 43 44}"
STEPS="${STEPS:-100000}"          # per training run
TUNE_TRIALS="${TUNE_TRIALS:-30}"
TUNE_STEPS="${TUNE_STEPS:-20000}" # per Optuna trial
TUNE_EVAL_SEEDS="${TUNE_EVAL_SEEDS:-42 43}"

SKIP_TUNE=0
SKIP_TRAIN=0
SKIP_EVAL=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --skip-tune)  SKIP_TUNE=1 ;;
        --skip-train) SKIP_TRAIN=1 ;;
        --skip-eval)  SKIP_EVAL=1 ;;
        --force)      FORCE=1 ;;
        *) echo "unknown arg: $arg"; exit 2 ;;
    esac
done

# eval loads the seed-0 model of each algo (the reference checkpoint)
REF_SEED="$(echo "$TRAIN_SEEDS" | awk '{print $1}')"

# ---- setup ------------------------------------------------------------------
cd "$(dirname "$0")"
if [ -d venv ]; then source venv/bin/activate; fi
export SUMO_HOME="${SUMO_HOME:-$(python -c 'import sumo; print(sumo.SUMO_HOME)')}"

mkdir -p logs models params
RUNLOG="logs/experiment_$(date +%Y%m%d_%H%M%S).log"

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUNLOG"; }
run()  { log "RUN: $*"; if "$@" >>"$RUNLOG" 2>&1; then log "OK"; else log "FAILED (see $RUNLOG) — continuing"; fi; }

log "SUMO_HOME=$SUMO_HOME"
log "algos=[$ALGOS] train_seeds=[$TRAIN_SEEDS] eval_seeds=[$EVAL_SEEDS] steps=$STEPS ref_seed=$REF_SEED"

# ---- 1. tune ----------------------------------------------------------------
if [ "$SKIP_TUNE" -eq 0 ]; then
    for a in $ALGOS; do
        if [ "$FORCE" -eq 0 ] && [ -f "params/$a.json" ]; then
            log "SKIP tune $a (params/$a.json exists)"; continue
        fi
        log "=== TUNE $a ($TUNE_TRIALS trials x $TUNE_STEPS steps) ==="
        run python tune.py --algo "$a" --trials "$TUNE_TRIALS" --steps "$TUNE_STEPS" --eval-seeds $TUNE_EVAL_SEEDS
    done
else
    log "skip tuning (--skip-tune)"
fi

# ---- 2. train ---------------------------------------------------------------
if [ "$SKIP_TRAIN" -eq 0 ]; then
    for a in $ALGOS; do
        for s in $TRAIN_SEEDS; do
            if [ "$FORCE" -eq 0 ] && [ -f "models/${a}_seed${s}.zip" ]; then
                log "SKIP train $a seed$s (model exists)"; continue
            fi
            log "=== TRAIN $a seed$s ($STEPS steps) ==="
            run python train.py --algo "$a" --seed "$s" --steps "$STEPS"
        done
    done
else
    log "skip training (--skip-train)"
fi

# ---- 3. eval (held-out seeds, reference checkpoint per algo) -----------------
if [ "$SKIP_EVAL" -eq 0 ]; then
    for a in $ALGOS; do
        model="models/${a}_seed${REF_SEED}.zip"
        if [ ! -f "$model" ]; then log "SKIP eval $a (no $model)"; continue; fi
        for s in $EVAL_SEEDS; do
            csv="logs/eval_${a}_seed${s}_conn0_ep1.csv"
            if [ "$FORCE" -eq 0 ] && [ -f "$csv" ]; then
                log "SKIP eval $a seed$s (csv exists)"; continue
            fi
            log "=== EVAL $a seed$s ==="
            run python train.py --algo "$a" --eval "$model" --seed "$s"
        done
    done
else
    log "skip eval (--skip-eval)"
fi

# ---- 4. compare -------------------------------------------------------------
log "=== COMPARE ==="
python compare.py 2>&1 | tee -a "$RUNLOG"

log "done. full log: $RUNLOG"
