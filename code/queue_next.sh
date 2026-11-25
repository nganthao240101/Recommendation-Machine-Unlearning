#!/usr/bin/env bash
# Next queue, run in order once the chain running now is done:
#   1) ml-1m, mode most, ONE MODEL PER SEED (5 seeds): SAFER-PI vs Random
#      and SAFER-PIE vs Random+E, then the paired test (p-values)
#   2) yelp2018 (its own setting: lr 0.02, patience 15): the new arms added to
#      the single / most / fewest logs - confirms ml-1m without any change
#   3) ml-1m: Retrain and RecEraser in the same setting, for reference
#
#   nohup setsid bash code/queue_next.sh > logs/queue_next.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.." || exit 1

wait_free () {
  while pgrep -f "run_blocks\.sh (regauto|ytune|esauto|yone|yseeds|compare|patience|kyelp|epochs|full|fullwmf|main|modes|single|all)$" >/dev/null; do
    sleep 120
  done
}
step () { echo "[$(date '+%F %T')] $*"; }

P_ML=$(grep -o "RECOMMENDED patience = [0-9]*" logs/patience_ml1m_reg0.03.out | grep -o "[0-9]*$" | tail -1)
P_ML=${P_ML:-30}

# ---- 1
wait_free
step "step 1 ml-1m per-seed test, mode most, patience=$P_ML"
YS_DATASET=ml-1m EARLY_STOP=1 PATIENCE="$P_ML" REG=0.03 YS_MODES="most" \
  YS_FIRST=SAFER_PI,RANDOM_PERM,SAFER_PIE,RANDOM_E YS_STEPS=1 \
  PAIRS="SAFER_PI:RANDOM_PERM SAFER_PIE:RANDOM_E" \
  bash code/run_blocks.sh start yseeds
sleep 60

# ---- 2
wait_free
step "step 2 yelp2018 new arms, lr0.02 patience=15"
YO_DATASET=yelp2018 EARLY_STOP=1 PATIENCE=15 LR=0.02 \
  YO_ARMS=SAFER_P,SAFER_PI,SAFER_PIE,SAFER_BE,RANDOM_E,RANDOM_EU,SISA_FULL \
  bash code/run_blocks.sh start yone
sleep 60

# ---- 3
wait_free
step "step 3 ml-1m Retrain and RecEraser, reg0.03 patience=$P_ML"
YO_DATASET=ml-1m EARLY_STOP=1 PATIENCE="$P_ML" REG=0.03 YO_ARMS=GLOBAL,RECERASER \
  bash code/run_blocks.sh start yone
step "all steps queued"
