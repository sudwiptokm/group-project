#!/usr/bin/env bash
#
# Stage-2 lambda ablation at peak, on the recalibrated environment.
#
# What changed vs Stage 1, and why:
#   * TIME_TO_TELEPORT=300 (SUMO's own default). sumo-rl ships -1, which turns
#     junction deadlock into an absorbing state -- the 2-phase program has
#     permissive left turns, so peak locked up and "waiting time" degenerated
#     into a clock. At 300 peak settles at 42.4 +/- 10.4 s over seeds 42-46
#     with ~9 teleports/episode and no deadlock.
#   * Demand is UNCHANGED (peak = 1.5x base). The bistability was the teleport
#     setting, not the flow rates.
#   * lambda is swept over {0.0, 0.5, 1.0}; Stage 1 only ever ran 0.5, so the
#     safety/efficiency tradeoff central to the project was never measured.
#   * Hyperparameters are held at the Stage-1 tuned values across all lambda
#     arms. Re-tuning per lambda costs about as much as the whole sweep. Held
#     constant is the right control for an ablation, but it must be disclosed:
#     those HPs were selected at lambda=0.5 under the old (broken) reward.
#
# Resumable: any run whose model .zip already exists is skipped, so an
# interruption costs one run, not the batch. Re-run the same command to
# continue. Pass --force to redo everything.
#
#   caffeinate -is ./run_lambda_sweep.sh              # keeps the mac awake
#   JOBS=2 ./run_lambda_sweep.sh                      # lighter on the machine
#   STEPS=10000 ./run_lambda_sweep.sh                 # quicker, rougher

set -uo pipefail

# A multi-hour sweep must not die because someone hit Ctrl-C in some other
# terminal. An ignored disposition survives exec, so every python/SUMO child
# inherits the immunity too. Consequence: Ctrl-C will NOT stop this.
#   stop with:  pkill -f run_lambda_sweep.sh; pkill -f train.py
trap '' INT HUP

cd "$(dirname "$0")"
if [ -d venv ]; then source venv/bin/activate; fi
export SUMO_HOME="${SUMO_HOME:-$(python -c 'import sumo; print(sumo.SUMO_HOME)')}"

JOBS="${JOBS:-3}"                       # 3 keeps ~2 GB free on an 8 GB machine
STEPS="${STEPS:-20000}"
# ppo first: an 8k-step pilot showed ppo learning (episode mean wait 25.7 -> 20.3
# across quartiles) while dqn stayed flat and noisy on two seeds. Ordering the
# arm with a real signal first means an out-of-time run still yields a complete
# ppo lambda ablation rather than two half-finished ones.
ALGOS="${ALGOS:-ppo dqn}"
# params/{dqn,ppo}.json were tuned for a 100k-step budget: lr ~2.3e-5,
# dqn learning_starts=5000 and target_update_interval=5000. Against a 20k run
# that is 25% of the budget spent on random actions and 3 target updates, and
# nothing learns. The repo defaults in algos.py (dqn lr 1e-4 / starts 1000 /
# target 1000, ppo lr 3e-4 / n_steps 256) are sized for short runs. Held
# identical across every lambda arm, which is what the ablation needs.
USE_DEFAULTS="${USE_DEFAULTS:-1}"
LAMBDAS="${LAMBDAS:-0.0 0.5 1.0}"       # MUST keep the decimal point (_tag strips it)
TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2}"
EVAL_SEEDS="${EVAL_SEEDS:-42 43 44}"
SCENARIO="${SCENARIO:-peak}"
# Baseline green, in seconds. 60 is the best static plan in the peak sweep
# (analysis/static_timing.py: 11.5 s mean wait vs 26.7 s for the 10 s cycler
# Stage 1 called "fixed-time"). The comparison is only fair against a
# competently timed plan.
BASELINE_GREEN="${BASELINE_GREEN:-60}"

export EPISODE_SECONDS="${EPISODE_SECONDS:-1200}"   # mean wait over 0-1200s
export TIME_TO_TELEPORT="${TIME_TO_TELEPORT:-300}"  # tracks the 0-3600s mean

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

STAMP="$(date +%Y%m%d_%H%M%S)"
RUNLOG="logs/lambda_sweep_${STAMP}"
mkdir -p "$RUNLOG" models logs

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUNLOG/driver.log"; }

# tag mirrors train.py::_tag -- "peak_lam05" etc
tag_of() { echo "${SCENARIO}_lam$(echo "$1" | tr -d '.')"; }

DEFAULTS_FLAG=""
[ "$USE_DEFAULTS" = "1" ] && DEFAULTS_FLAG="--defaults"

log "lambda sweep: algos=[$ALGOS] lambdas=[$LAMBDAS] seeds=[$TRAIN_SEEDS] steps=$STEPS jobs=$JOBS defaults=$USE_DEFAULTS"
log "env: EPISODE_SECONDS=$EPISODE_SECONDS TIME_TO_TELEPORT=$TIME_TO_TELEPORT scenario=$SCENARIO baseline_green=$BASELINE_GREEN"

# ---- stage 1: train ---------------------------------------------------------
JOBLIST="$RUNLOG/train_jobs.txt"
: > "$JOBLIST"
for algo in $ALGOS; do
    for lam in $LAMBDAS; do
        tag="$(tag_of "$lam")"
        for s in $TRAIN_SEEDS; do
            model="models/${algo}_${tag}_seed${s}.zip"
            if [ -f "$model" ] && [ "$FORCE" -eq 0 ]; then
                log "skip (exists): $model"
                continue
            fi
            echo "python train.py --algo $algo --scenario $SCENARIO --lam $lam --seed $s --steps $STEPS $DEFAULTS_FLAG > $RUNLOG/train_${algo}_${tag}_seed${s}.log 2>&1" >> "$JOBLIST"
        done
    done
done

NTRAIN=$(wc -l < "$JOBLIST" | tr -d ' ')
log "=== TRAIN: $NTRAIN runs, $JOBS at a time ==="
if [ "$NTRAIN" -gt 0 ]; then
    xargs -P "$JOBS" -I{} sh -c '{}' < "$JOBLIST"
    log "train stage done"
fi

# ---- stage 2: eval ----------------------------------------------------------
# Each trained model is evaluated on held-out demand seeds. Baselines run on
# the SAME seeds -- Stage 1 compared RL on seeds 42-46 against a fixed-time
# baseline on seed 0 alone, and that confound alone flipped the headline.
JOBLIST="$RUNLOG/eval_jobs.txt"
: > "$JOBLIST"
for algo in $ALGOS; do
    for lam in $LAMBDAS; do
        tag="$(tag_of "$lam")"
        for ts in $TRAIN_SEEDS; do
            model="models/${algo}_${tag}_seed${ts}.zip"
            [ -f "$model" ] || { log "MISSING model, skipping eval: $model"; continue; }
            # every checkpoint on every demand seed: Stage 1 evaluated only the
            # seed-0 model, so its "std over 5 seeds" was demand spread from a
            # single policy. train.py --train-seed keeps the CSVs distinct.
            for es in $EVAL_SEEDS; do
                out="logs/eval_${algo}_${tag}_seed${es}_t${ts}_conn0_ep1.csv"
                if [ -f "$out" ] && [ "$FORCE" -eq 0 ]; then continue; fi
                echo "python train.py --algo $algo --eval $model --scenario $SCENARIO --lam $lam --seed $es --train-seed $ts > $RUNLOG/eval_${algo}_${tag}_train${ts}_seed${es}.log 2>&1" >> "$JOBLIST"
            done
        done
    done
done

for es in $EVAL_SEEDS; do
    out="logs/eval_fixedtime_${SCENARIO}_seed${es}_g${BASELINE_GREEN}_conn0_ep1.csv"
    if [ -f "$out" ] && [ "$FORCE" -eq 0 ]; then continue; fi
    echo "python baseline.py --scenario $SCENARIO --seed $es --green $BASELINE_GREEN > $RUNLOG/eval_fixedtime_seed${es}_g${BASELINE_GREEN}.log 2>&1" >> "$JOBLIST"
done

NEVAL=$(wc -l < "$JOBLIST" | tr -d ' ')
log "=== EVAL: $NEVAL runs, $JOBS at a time ==="
if [ "$NEVAL" -gt 0 ]; then
    xargs -P "$JOBS" -I{} sh -c '{}' < "$JOBLIST"
    log "eval stage done"
fi

log "sweep complete -- logs in $RUNLOG"
