#!/usr/bin/env bash
# ONE setting (reg, lr, dim, patience) for BOTH datasets, chosen on validation,
# then every experiment on ml-1m and yelp with it. Every tuning cell is cached
# (logs/tunegrid_<ds>.jsonl) and finished runs are kept, so the script can be
# stopped and started again at any time.
#
#   1) tune_reg.py on ml-1m and on yelp: reg x lr at dim 64 (the same grid,
#      the same arms, the same early stopping) - yelp reuses the ytune cells
#   2) pick_joint.py: the (reg, lr) best on both datasets (each dataset's
#      score relative to its own best, averaged)
#   3) dims at that (reg, lr) on both datasets, pick_joint.py again
#   4) ml-1m then yelp: one model per arm, unlearned 5 times, single / most /
#      fewest, every method (Retrain, RecEraser included)
#   5) per-seed test (5 seeds), single / most / fewest, both datasets
#   6) results and paired tests -> logs/final_joint_results.txt / _paired.txt
#
#   nohup setsid bash code/queue_joint.sh > logs/queue_joint.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.." || exit 1
export RECUNLEARN_ENGINE="${RECUNLEARN_ENGINE:-gpu}"

wait_free () {
  while pgrep -f "run_blocks\.sh (regauto|ytune|esauto|yone|yseeds|compare|patience|kyelp|epochs|ablate|full|fullwmf|main|modes|single|all)$" >/dev/null \
        || pgrep -f "code/tune_reg\.py" >/dev/null; do
    sleep 120
  done
}
step () { echo "[$(date '+%F %T')] $*"; }
run_step () { wait_free; step "$1"; shift; "$@"; sleep 60; }

DATASETS="ml-1m yelp2018"
TARMS=SAFER_P,SAFER_PI,PROPOSED,PROPOSED_BAL,RANDOM_PERM,SISA_FULL
REGS="${J_REGS:-0 0.001 0.003 0.01 0.03}"
LRS="${J_LRS:-0.05 0.02 0.01 0.005}"
DIMS="${J_DIMS:-32 64 128 256}"
P="${J_PATIENCE:-15}"                  # the patience every tuning cell ran with
TUNE="--arms $TARMS --eval_every 5 --patience $P"

# the tuner holds the run lock, so nothing measured runs beside it
tune () { flock logs/.run.lock python code/tune_reg.py "$@"; }

for ds in $DATASETS; do
  run_step "step 1 $ds: reg x lr at dim 64 (validation)" \
    tune --dataset "$ds" $TUNE --regs "$REGS" --lrs "$LRS" --dims 64
done > >(tee -a logs/jtune_s1.out) 2>&1
python code/pick_joint.py --datasets "$DATASETS" --arms "$TARMS" --patience "$P" --dims 64 \
  | tee logs/jtune_pick1.out
R=$(grep -o "^JOINT RECOMMENDED reg=[0-9.e-]*" logs/jtune_pick1.out | cut -d= -f2)
L=$(grep -o "^JOINT RECOMMENDED reg=[0-9.e-]* lr=[0-9.e-]*" logs/jtune_pick1.out | sed 's/.*lr=//')
if [ -z "$R" ] || [ -z "$L" ]; then step "!! no JOINT RECOMMENDED in logs/jtune_pick1.out"; exit 1; fi

for ds in $DATASETS; do
  run_step "step 3 $ds: dims $DIMS at reg $R lr $L (validation)" \
    tune --dataset "$ds" $TUNE --regs "$R" --lrs "$L" --dims "$DIMS"
done > >(tee -a logs/jtune_s3.out) 2>&1
python code/pick_joint.py --datasets "$DATASETS" --arms "$TARMS" --patience "$P" \
  | tee logs/jtune_pick2.out
line=$(grep "^JOINT RECOMMENDED" logs/jtune_pick2.out | tail -1)
R=$(echo "$line" | grep -o "reg=[0-9.e-]*" | cut -d= -f2)
L=$(echo "$line" | grep -o "lr=[0-9.e-]*" | cut -d= -f2)
D=$(echo "$line" | grep -o "dim=[0-9]*" | cut -d= -f2)
if [ -z "$R" ] || [ -z "$L" ] || [ -z "$D" ]; then step "!! no JOINT RECOMMENDED in logs/jtune_pick2.out"; exit 1; fi
step "ONE SETTING for $DATASETS: reg $R lr $L dim $D patience $P"
[ "${J_TUNE_ONLY:-0}" = 1 ] && { step "J_TUNE_ONLY=1: stop after tuning"; exit 0; }

ALL_ARMS=SAFER_P,SAFER_PI,PROPOSED,PROPOSED_BAL,RANDOM_PERM,RANDOM_EMB,SISA,SISA_FULL,GLOBAL,RECERASER
TEST_ARMS=SAFER_P,RANDOM_PERM,SAFER_PI,SISA,SISA_FULL,RANDOM_EMB,PROPOSED_BAL,PROPOSED
PAIRS_ALL="SAFER_P:RANDOM_PERM SAFER_P:SISA SAFER_P:SISA_FULL SAFER_P:RANDOM_EMB SAFER_P:PROPOSED_BAL SAFER_P:PROPOSED SAFER_PI:SAFER_P"
CFG="EARLY_STOP=1 PATIENCE=$P REG=$R LR=$L DIM=$D"

for ds in $DATASETS; do
  run_step "step 4 $ds: one model per arm, all methods, single/most/fewest" \
    env $CFG YO_DATASET="$ds" YO_ARMS="$ALL_ARMS" bash code/run_blocks.sh start yone
done
for ds in $DATASETS; do
  run_step "step 5 $ds: per-seed test, single/most/fewest" \
    env $CFG YS_DATASET="$ds" YS_MODES="single most fewest" YS_FIRST="$TEST_ARMS" \
        YS_STEPS=1 PAIRS="$PAIRS_ALL" bash code/run_blocks.sh start yseeds
done

wait_free
step "step 6 writing logs/final_joint_results.txt and logs/final_joint_paired.txt"
bash code/run_blocks.sh results > logs/final_joint_results.txt 2>&1
for ds in $DATASETS; do
  echo "### $ds (one model per seed, single / most / fewest)"
  env $CFG YS_DATASET="$ds" YS_MODES="single most fewest" PAIRS="$PAIRS_ALL" \
    bash code/run_blocks.sh paired
  echo ""
done > logs/final_joint_paired.txt 2>&1
step "all done: reg $R lr $L dim $D patience $P on $DATASETS"
