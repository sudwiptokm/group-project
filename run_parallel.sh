#!/usr/bin/env bash
#
# Parallel experiment driver: same pipeline as run_experiment.sh
# (tune -> train -> eval -> compare -> plots) but each stage fans its
# independent jobs across CPU cores. Turns the ~11-day serial `full` run
# into roughly 1 day on a 16-vCPU box.
#
# Why this is safe to parallelise: every train/eval/tune invocation is a
# separate Python process with its own in-process libsumo simulator (no shared
# TraCI port), so the runs are fully independent. Only within a stage do we run
# in parallel; stages stay ordered because train reads tune's params/*.json and
# eval reads train's models/*.zip.
#
# Resumable + interrupt-safe (spot/preemptible friendly): jobs whose output
# artifact already exists are skipped, exactly like run_experiment.sh. After a
# preemption just re-run the same command.
#
#   MODE=full  JOBS=16 ./run_parallel.sh      # publication budget, 16 parallel
#   MODE=overnight ./run_parallel.sh          # quick preset (default MODE)
#   JOBS=8 ./run_parallel.sh                  # cap parallelism (RAM/core limit)
#   ./run_parallel.sh --skip-tune             # reuse existing params/
#   ALGOS="dqn ppo" SCENARIOS=peak ./run_parallel.sh   # subset
#   ./run_parallel.sh --force                 # ignore existing artifacts, redo
#
# Each job logs to logs/parallel_<stamp>/<label>.log ; the driver prints a
# one-line start/ok/FAIL per job to the console + run log.
#
# RAM note: each job is ~0.5-1 GB (SUMO + torch CPU). JOBS=16 => ~8-16 GB.
# A 16-vCPU box usually has 32-64 GB, fine. Lower JOBS if you see OOM kills.

set -uo pipefail
cd "$(dirname "$0")"
if [ -d venv ]; then source venv/bin/activate; fi
export SUMO_HOME="${SUMO_HOME:-$(python -c 'import sumo; print(sumo.SUMO_HOME)')}"

# ---- parallelism ------------------------------------------------------------
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 4)}"

# ---- mode presets (identical to run_experiment.sh) --------------------------
MODE="${MODE:-overnight}"
case "$MODE" in
    overnight)
        _EPISODE=1200; _STEPS=30000;  _TRIALS=12; _TSTEPS=10000
        _TRAIN_SEEDS="0 1 2";       _EVAL_SEEDS="42 43 44" ;;
    full)
        _EPISODE=3600; _STEPS=100000; _TRIALS=30; _TSTEPS=20000
        _TRAIN_SEEDS="0 1 2 3 4";   _EVAL_SEEDS="42 43 44 45 46" ;;
    *) echo "unknown MODE: $MODE (use 'overnight' or 'full')"; exit 2 ;;
esac

# ---- config (explicit env vars override the MODE preset) --------------------
ALGOS="${ALGOS:-dqn qrdqn ppo a2c}"
SCENARIOS="${SCENARIOS:-peak offpeak}"
LAMBDAS="${LAMBDAS:-0.5}"                          # decimal point required (see run_experiment.sh)
TRAIN_SEEDS="${TRAIN_SEEDS:-$_TRAIN_SEEDS}"
EVAL_SEEDS="${EVAL_SEEDS:-$_EVAL_SEEDS}"
STEPS="${STEPS:-$_STEPS}"
TUNE_TRIALS="${TUNE_TRIALS:-$_TRIALS}"
TUNE_STEPS="${TUNE_STEPS:-$_TSTEPS}"
TUNE_EVAL_SEEDS="${TUNE_EVAL_SEEDS:-42 43}"
EPISODE_SECONDS="${EPISODE_SECONDS:-$_EPISODE}"
export EPISODE_SECONDS                            # consumed by env_common.make_env

SKIP_TUNE=0; SKIP_TRAIN=0; SKIP_EVAL=0; FORCE=0
for arg in "$@"; do
    case "$arg" in
        --skip-tune)  SKIP_TUNE=1 ;;
        --skip-train) SKIP_TRAIN=1 ;;
        --skip-eval)  SKIP_EVAL=1 ;;
        --force)      FORCE=1 ;;
        *) echo "unknown arg: $arg"; exit 2 ;;
    esac
done

REF_SEED="$(echo "$TRAIN_SEEDS" | awk '{print $1}')"

mkdir -p logs models params
STAMP="$(date +%Y%m%d_%H%M%S)"
JOBDIR="logs/parallel_${STAMP}"
mkdir -p "$JOBDIR"
RUNLOG="${JOBDIR}/driver.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUNLOG"; }

# run_phase NAME JOBFILE
#   JOBFILE = null-delimited records, each "label<TAB>command".
#   Runs up to $JOBS at once; each job's stdout/stderr -> $JOBDIR/<label>.log.
run_phase() {
    local name="$1" jobfile="$2"
    local n; n="$(tr -cd '\0' < "$jobfile" | wc -c | tr -d ' ')"
    if [ "$n" -eq 0 ]; then log "[$name] nothing to do (all artifacts present)"; return; fi
    log "[$name] $n jobs, up to $JOBS in parallel"
    # -0: records null-delimited. The runner splits label<TAB>command on the
    # first tab, so commands may contain spaces/flags freely.
    JOBDIR="$JOBDIR" xargs -0 -P "$JOBS" -n1 -I REC bash -c '
        rec="$1"
        label="${rec%%	*}"        # up to first literal tab
        cmd="${rec#*	}"           # after first literal tab
        echo "[start] $label"
        if eval "$cmd" > "$JOBDIR/$label.log" 2>&1; then
            echo "[ok]   $label"
        else
            echo "[FAIL] $label  -> $JOBDIR/$label.log"
        fi
    ' _ REC < "$jobfile" | tee -a "$RUNLOG"
}

# emit LABEL COMMAND -> append a null-terminated "label<TAB>command" record
emit() { printf '%s\t%s\0' "$1" "$2" >> "$JOBFILE"; }

log "MODE=$MODE JOBS=$JOBS SUMO_HOME=$SUMO_HOME"
log "algos=[$ALGOS] scenarios=[$SCENARIOS] lambdas=[$LAMBDAS]"
log "train_seeds=[$TRAIN_SEEDS] eval_seeds=[$EVAL_SEEDS] steps=$STEPS ref_seed=$REF_SEED"

# ---- 1. tune (parallel over algos) ------------------------------------------
if [ "$SKIP_TUNE" -eq 0 ]; then
    JOBFILE="$JOBDIR/jobs_tune"; : > "$JOBFILE"
    for a in $ALGOS; do
        if [ "$FORCE" -eq 0 ] && [ -f "params/$a.json" ]; then
            log "SKIP tune $a (params/$a.json exists)"; continue
        fi
        emit "tune_$a" "python tune.py --algo $a --trials $TUNE_TRIALS --steps $TUNE_STEPS --eval-seeds $TUNE_EVAL_SEEDS --scenario peak --lam 0.5"
    done
    run_phase tune "$JOBFILE"
else
    log "skip tuning (--skip-tune)"
fi

# ---- 2. train (parallel over algo x scenario x lambda x seed) ---------------
if [ "$SKIP_TRAIN" -eq 0 ]; then
    JOBFILE="$JOBDIR/jobs_train"; : > "$JOBFILE"
    for scenario in $SCENARIOS; do
        for lam in $LAMBDAS; do
            tag="${scenario}_lam${lam//./}"
            for algo in $ALGOS; do
                for s in $TRAIN_SEEDS; do
                    if [ "$FORCE" -eq 0 ] && [ -f "models/${algo}_${tag}_seed${s}.zip" ]; then
                        log "SKIP train $algo $tag seed$s (model exists)"; continue
                    fi
                    emit "train_${algo}_${tag}_seed${s}" \
                        "python train.py --algo $algo --scenario $scenario --lam $lam --seed $s --steps $STEPS"
                done
            done
        done
    done
    run_phase train "$JOBFILE"
else
    log "skip training (--skip-train)"
fi

# ---- 3. eval (parallel; baselines + held-out seeds) -------------------------
if [ "$SKIP_EVAL" -eq 0 ]; then
    JOBFILE="$JOBDIR/jobs_eval"; : > "$JOBFILE"
    for scenario in $SCENARIOS; do
        if [ "$FORCE" -eq 0 ] && ls "logs/eval_fixedtime_${scenario}_seed0"*.csv >/dev/null 2>&1; then
            log "SKIP baseline $scenario (csv exists)"
        else
            emit "baseline_${scenario}" "python baseline.py --scenario $scenario --seed 0"
        fi
        for lam in $LAMBDAS; do
            tag="${scenario}_lam${lam//./}"
            for algo in $ALGOS; do
                ref="models/${algo}_${tag}_seed${REF_SEED}.zip"
                if [ ! -f "$ref" ]; then
                    log "SKIP eval $algo $tag (no $ref)"; continue
                fi
                for s in $EVAL_SEEDS; do
                    if [ "$FORCE" -eq 0 ] && ls "logs/eval_${algo}_${tag}_seed${s}"*.csv >/dev/null 2>&1; then
                        log "SKIP eval $algo $tag seed$s (csv exists)"; continue
                    fi
                    emit "eval_${algo}_${tag}_seed${s}" \
                        "python train.py --algo $algo --eval $ref --scenario $scenario --lam $lam --seed $s"
                done
            done
        done
    done
    run_phase eval "$JOBFILE"
else
    log "skip eval (--skip-eval)"
fi

# ---- 4. compare + plots (serial) --------------------------------------------
log "=== COMPARE ==="
python compare.py 2>&1 | tee -a "$RUNLOG"
log "=== PLOTS ==="
python plots.py 2>&1 | tee -a "$RUNLOG"

log "done. driver log: $RUNLOG ; per-job logs: $JOBDIR/*.log"
