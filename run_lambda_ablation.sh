#!/usr/bin/env bash
#
# SP14: train the missing arms of the safety-weight (lambda) ablation.
#
# The project is framed as "safety-aware" signal control -- the corridor reward
# is efficiency - lambda * safety_penalty (env_common.make_safety_reward_fn) --
# but every corridor experiment so far (SP4-SP12) was run at the single value
# lambda=0.5. The tradeoff the framing rests on has never been measured.
#
# This trains IDQN at lambda in {0.0, 0.25, 0.75, 1.0} x seeds {42, 43, 44} on
# corridor_peak (12 runs). lambda=0.5 x the same 3 seeds already exists from
# SP5 (models/idqn_{C1,C2,C3}_corridor_peak_lam05_seed4{2,3,4}_mg10_s100000.pt)
# and is deliberately NOT retrained -- reusing it keeps the 0.5 arm identical
# to every published corridor result.
#
# Everything except lambda is held at the SP5 settings: corridor_peak,
# min_green=10, steps=100000, TIME_TO_TELEPORT=300, hyperparameters from
# train_corridor_dqn._hp(). Holding HPs constant across arms is the right
# control for an ablation, but it is a disclosed limitation: those HPs were
# selected at lambda=0.5.
#
# train() has no net_file parameter -- training only ever happens on
# corridor.net.xml. The cross-geometry half of this ablation is zero-shot
# evaluation of these checkpoints, done by analysis/lambda_ablation.py.
#
# Resumable: a run whose 3 agent checkpoints all exist is skipped, so an
# interruption costs one run, not the batch. Re-run the same command.
#
#   caffeinate -is ./run_lambda_ablation.sh      # keeps the mac awake
#   JOBS=3 ./run_lambda_ablation.sh              # lighter on the machine
#
# Ctrl-C is deliberately ignored (an ignored disposition survives exec, so the
# python/SUMO children inherit it too). Stop with:
#   pkill -f run_lambda_ablation.sh; pkill -f train_corridor_dqn.py
set -uo pipefail
trap '' INT HUP

cd "$(dirname "$0")"
if [ -d venv ]; then source venv/bin/activate; fi
export SUMO_HOME="${SUMO_HOME:-$(python -c 'import sumo; print(sumo.SUMO_HOME)')}"

# Must match the environment SP5 trained the lambda=0.5 arm under
# (analysis/idqn_sweep.py sets exactly this); with SUMO's -1 default, junction
# deadlock becomes an absorbing state and the arms are not comparable.
export TIME_TO_TELEPORT="${TIME_TO_TELEPORT:-300}"
# one BLAS thread per job: the parallelism here is across processes, and torch
# fanning out over all cores inside each of $JOBS jobs just oversubscribes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

JOBS="${JOBS:-6}"
LAMBDAS="${LAMBDAS:-0.0 0.25 0.75 1.0}"   # 0.5 already trained (SP5)
SEEDS="${SEEDS:-42 43 44}"
SCENARIO="${SCENARIO:-corridor_peak}"
MIN_GREEN="${MIN_GREEN:-10}"
STEPS="${STEPS:-100000}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUNLOG="logs/lambda_ablation_${STAMP}"
mkdir -p "$RUNLOG" models logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUNLOG/driver.log"; }

# mirrors train_corridor_dqn._tag: the decimal point is stripped, so 0.0 -> 00,
# 0.25 -> 025, 1.0 -> 10. Pass lambdas WITH the decimal point.
tag_of() { echo "${SCENARIO}_lam$(echo "$1" | tr -d '.')_seed${2}_mg${MIN_GREEN}_s${STEPS}"; }

JOBLIST="$RUNLOG/train_jobs.txt"
: > "$JOBLIST"
for lam in $LAMBDAS; do
    for s in $SEEDS; do
        tag="$(tag_of "$lam" "$s")"
        missing=0
        for a in C1 C2 C3; do
            [ -f "models/idqn_${a}_${tag}.pt" ] || missing=1
        done
        if [ "$missing" -eq 0 ]; then
            log "skip (all 3 checkpoints exist): $tag"
            continue
        fi
        echo "python train_corridor_dqn.py --scenario $SCENARIO --lam $lam --seed $s --min-green $MIN_GREEN --steps $STEPS > $RUNLOG/train_${tag}.log 2>&1" >> "$JOBLIST"
    done
done

N=$(wc -l < "$JOBLIST" | tr -d ' ')
log "lambda ablation: lambdas=[$LAMBDAS] seeds=[$SEEDS] scenario=$SCENARIO mg=$MIN_GREEN steps=$STEPS"
log "=== TRAIN: $N runs, $JOBS at a time (teleport=$TIME_TO_TELEPORT) ==="
if [ "$N" -gt 0 ]; then
    xargs -P "$JOBS" -I{} sh -c '{}' < "$JOBLIST"
fi
log "train stage done -- per-run logs in $RUNLOG"
log "next: python -m analysis.lambda_ablation"
