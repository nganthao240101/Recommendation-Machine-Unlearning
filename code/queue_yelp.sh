#!/usr/bin/env bash
# yelp again with the setting chosen on VALIDATION by ytune (one setting for
# every method): reg 0, lr 0.005, dim 128 by default, patience 15 (the
# early-stopping protocol ytune was run with). Finished runs are kept and
# skipped, so the script can be started again at any time.
#
#   1) one model per arm, unlearned 5 times, single / most / fewest:
#      every method, Retrain and RecEraser included
#   2) per-seed test (5 seeds, one model per seed), single / most / fewest
#   3) results and paired tests -> logs/final_yelp_results.txt,
#      logs/final_yelp_paired.txt
#
#   nohup setsid bash code/queue_yelp.sh > logs/queue_yelp.out 2>&1 < /dev/null &
#   other setting: YREG=0.003 YLR=0.005 YDIM=64 nohup setsid bash code/queue_yelp.sh ...
cd "$(dirname "$0")/.." || exit 1

wait_free () {
  while pgrep -f "run_blocks\.sh (regauto|ytune|esauto|yone|yseeds|compare|patience|kyelp|epochs|ablate|full|fullwmf|main|modes|single|all)$" >/dev/null \
        || pgrep -f "code/tune_reg\.py" >/dev/null; do
    sleep 120
  done
}
step () { echo "[$(date '+%F %T')] $*"; }
run_step () { wait_free; step "$1"; shift; "$@"; sleep 60; }

R=${YREG:-0}; L=${YLR:-0.005}; D=${YDIM:-128}; P=${YPAT:-15}
ALL_ARMS=SAFER_P,SAFER_PI,PROPOSED,PROPOSED_BAL,RANDOM_PERM,RANDOM_EMB,SISA,SISA_FULL,GLOBAL,RECERASER
TEST_ARMS=SAFER_P,RANDOM_PERM,SAFER_PI,SISA,SISA_FULL,RANDOM_EMB,PROPOSED_BAL,PROPOSED
PAIRS_ALL="SAFER_P:RANDOM_PERM SAFER_P:SISA SAFER_P:SISA_FULL SAFER_P:RANDOM_EMB SAFER_P:PROPOSED_BAL SAFER_P:PROPOSED SAFER_PI:SAFER_P"
step "yelp setting: reg $R lr $L dim $D patience $P"

run_step "step 1 yelp one model per arm, all methods, single/most/fewest" \
  env YO_DATASET=yelp2018 EARLY_STOP=1 PATIENCE="$P" REG="$R" LR="$L" DIM="$D" \
      YO_ARMS="$ALL_ARMS" bash code/run_blocks.sh start yone

run_step "step 2 yelp per-seed test, single/most/fewest" \
  env YS_DATASET=yelp2018 EARLY_STOP=1 PATIENCE="$P" REG="$R" LR="$L" DIM="$D" \
      YS_MODES="single most fewest" YS_FIRST="$TEST_ARMS" YS_STEPS=1 PAIRS="$PAIRS_ALL" \
      bash code/run_blocks.sh start yseeds

wait_free
step "step 3 writing logs/final_yelp_results.txt and logs/final_yelp_paired.txt"
bash code/run_blocks.sh results > logs/final_yelp_results.txt 2>&1
YS_DATASET=yelp2018 EARLY_STOP=1 REG="$R" LR="$L" DIM="$D" YS_MODES="single most fewest" \
  PAIRS="$PAIRS_ALL" bash code/run_blocks.sh paired > logs/final_yelp_paired.txt 2>&1
step "all done"
