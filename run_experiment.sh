#!/usr/bin/env bash
#
# Full experiment driver: tune -> train (multi-seed) -> eval (held-out) -> compare.
#
# Runs the whole algorithm ladder unattended over SCENARIO × LAMBDA axes.
# Resumable: existing params/models/eval CSVs are skipped, so re-running
# continues where it left off. One failed run is logged and skipped — it does
# not abort the batch. A fixed-time baseline is generated once per scenario.
#
#   ./run_experiment.sh                 # full ladder, default budgets
#   ./run_experiment.sh --skip-tune     # use current params/ (or defaults)
#   ALGOS="dqn ppo" ./run_experiment.sh # subset of algorithms
#   STEPS=50000 TRAIN_SEEDS="0 1" ./run_experiment.sh   # smaller run
#   ./run_experiment.sh --force         # ignore existing artifacts, redo everything
#   SCENARIOS="peak" LAMBDAS="0.0 0.5" ./run_experiment.sh   # subset of axes
#
# Config via env vars (defaults below).
#
# IMPORTANT — LAMBDAS formatting: always write lambda values WITH the decimal
# point (e.g. "0.0 0.5 1.0", NOT "0 0.5 1").  The shell tag uses ${lam//./}
# (strip dots) to get "00","05","10" — this must match what train.py's _tag()
# produces via str(float(lam)).replace('.','').  "0" → shell tag "0" but
# train.py tag "00" → MISMATCH.  Use "0.0" to get "00" in both.

set -uo pipefail

# ---- config -----------------------------------------------------------------
ALGOS="${ALGOS:-dqn qrdqn ppo a2c}"
SCENARIOS="${SCENARIOS:-peak offpeak}"
# Stage 1 reference: single lambda. Stage 2 sweep: set to "0.0 0.5 1.0"
# Values MUST include the decimal point — see note above.
LAMBDAS="${LAMBDAS:-0.5}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2 3 4}"
EVAL_SEEDS="${EVAL_SEEDS:-42 43 44 45 46}"
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
log "algos=[$ALGOS] scenarios=[$SCENARIOS] lambdas=[$LAMBDAS]"
log "train_seeds=[$TRAIN_SEEDS] eval_seeds=[$EVAL_SEEDS] steps=$STEPS ref_seed=$REF_SEED"

# ---- 1. tune ----------------------------------------------------------------
# Tuning is done once per algo at the Stage-1 reference point (peak, lam=0.5).
# This produces params/<algo>.json which is then reused across all scenarios/lambdas.
if [ "$SKIP_TUNE" -eq 0 ]; then
    for a in $ALGOS; do
        if [ "$FORCE" -eq 0 ] && [ -f "params/$a.json" ]; then
            log "SKIP tune $a (params/$a.json exists)"; continue
        fi
        log "=== TUNE $a ($TUNE_TRIALS trials x $TUNE_STEPS steps) ==="
        run python tune.py --algo "$a" --trials "$TUNE_TRIALS" --steps "$TUNE_STEPS" \
            --eval-seeds $TUNE_EVAL_SEEDS --scenario peak --lam 0.5
    done
else
    log "skip tuning (--skip-tune)"
fi

# ---- 2. train ---------------------------------------------------------------
if [ "$SKIP_TRAIN" -eq 0 ]; then
    for scenario in $SCENARIOS; do
        for lam in $LAMBDAS; do
            # Shell tag: strip dots to match train.py's _tag() — requires decimal values
            # e.g. "0.5" -> "05", "1.0" -> "10", "0.0" -> "00"
            tag="${scenario}_lam${lam//./}"
            for algo in $ALGOS; do
                for s in $TRAIN_SEEDS; do
                    if [ "$FORCE" -eq 0 ] && [ -f "models/${algo}_${tag}_seed${s}.zip" ]; then
                        log "SKIP train $algo $tag seed$s (model exists)"; continue
                    fi
                    log "=== TRAIN $algo $tag seed$s ($STEPS steps) ==="
                    run python train.py --algo "$algo" --scenario "$scenario" --lam "$lam" \
                        --seed "$s" --steps "$STEPS"
                done
            done
        done
    done
else
    log "skip training (--skip-train)"
fi

# ---- 3. eval (held-out seeds, reference checkpoint per algo) -----------------
if [ "$SKIP_EVAL" -eq 0 ]; then
    for scenario in $SCENARIOS; do
        # Fixed-time baseline: once per scenario, resumable.
        # baseline.py writes: logs/eval_fixedtime_<scenario>_seed0_conn<N>_ep<M>.csv
        if [ "$FORCE" -eq 0 ] && ls "logs/eval_fixedtime_${scenario}_seed0"*.csv >/dev/null 2>&1; then
            log "SKIP baseline $scenario (csv exists)"
        else
            log "=== BASELINE fixedtime $scenario ==="
            run python baseline.py --scenario "$scenario" --seed 0
        fi

        for lam in $LAMBDAS; do
            tag="${scenario}_lam${lam//./}"
            for algo in $ALGOS; do
                ref="models/${algo}_${tag}_seed${REF_SEED}.zip"
                if [ ! -f "$ref" ]; then
                    log "SKIP eval $algo $tag (no $ref)"; continue
                fi
                for s in $EVAL_SEEDS; do
                    # train.py writes: logs/eval_<algo>_<tag>_seed<s>_conn<N>_ep<M>.csv
                    if [ "$FORCE" -eq 0 ] && ls "logs/eval_${algo}_${tag}_seed${s}"*.csv >/dev/null 2>&1; then
                        log "SKIP eval $algo $tag seed$s (csv exists)"; continue
                    fi
                    log "=== EVAL $algo $tag seed$s ==="
                    run python train.py --algo "$algo" --eval "$ref" \
                        --scenario "$scenario" --lam "$lam" --seed "$s"
                done
            done
        done
    done
else
    log "skip eval (--skip-eval)"
fi

# ---- 4. compare -------------------------------------------------------------
log "=== COMPARE ==="
python compare.py 2>&1 | tee -a "$RUNLOG"

log "done. full log: $RUNLOG"
