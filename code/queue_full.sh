#!/usr/bin/env bash
# Everything still to run for the paper, in order, once the chain running now
# is done. Runs that are already finished are kept and skipped (every step
# resumes its logs), so the script can be started again at any time.
#
#   1) ml-1m   per-seed test (5 seeds, mode most): SAFER-PI vs Random,
#              SAFER-PIE vs Random+E  -> paired t-test
#   2) yelp    the new arms in the single / most / fewest logs (one model),
#              yelp's own setting (lr 0.02, patience 15), nothing changed
#   3) yelp    per-seed test (5 seeds, mode most), same pairs
#   4) ml-1m   Retrain and RecEraser (same setting, for reference)
#   5) yelp    Retrain and RecEraser
#   6) results and paired tests written to logs/final_results.txt and
#      logs/final_paired.txt
#
#   nohup setsid bash code/queue_full.sh > logs/queue_full.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.." || exit 1

wait_free () {
  while pgrep -f "run_blocks\.sh (regauto|esauto|yone|yseeds|compare|patience|kyelp|epochs|full|fullwmf|main|modes|single|all)$" >/dev/null; do
    sleep 120
  done
}
step () { echo "[$(date '+%F %T')] $*"; }
run_step () { wait_free; step "$1"; shift; "$@"; sleep 60; }

P_ML=$(grep -o "RECOMMENDED patience = [0-9]*" logs/patience_ml1m_reg0.03.out 2>/dev/null | grep -o "[0-9]*$" | tail -1)
P_ML=${P_ML:-30}
P_YE=15                                  # the patience of the yelp lr 0.02 logs
NEW_ARMS=SAFER_P,SAFER_PI,SAFER_PIE,SAFER_BE,RANDOM_E,RANDOM_EU,SISA_FULL
TEST_ARMS=SAFER_PI,RANDOM_PERM,SAFER_PIE,RANDOM_E
PAIRS_ALL="SAFER_PI:RANDOM_PERM SAFER_PIE:RANDOM_E SAFER_P:RANDOM_PERM"


run_step "step 1 ml-1m per-seed test, mode most (patience $P_ML)" \
  env YS_DATASET=ml-1m EARLY_STOP=1 PATIENCE="$P_ML" REG=0.03 YS_MODES=most \
      YS_FIRST="$TEST_ARMS" YS_STEPS=1 PAIRS="SAFER_PI:RANDOM_PERM SAFER_PIE:RANDOM_E" \
      bash code/run_blocks.sh start yseeds

run_step "step 2 yelp new arms, single/most/fewest (lr 0.02, patience $P_YE)" \
  env YO_DATASET=yelp2018 EARLY_STOP=1 PATIENCE="$P_YE" LR=0.02 YO_ARMS="$NEW_ARMS" \
      bash code/run_blocks.sh start yone

run_step "step 3 yelp per-seed test, mode most" \
  env YS_DATASET=yelp2018 EARLY_STOP=1 PATIENCE="$P_YE" LR=0.02 YS_MODES=most \
      YS_FIRST="$TEST_ARMS" YS_STEPS=1 PAIRS="SAFER_PI:RANDOM_PERM SAFER_PIE:RANDOM_E" \
      bash code/run_blocks.sh start yseeds

run_step "step 4 ml-1m Retrain and RecEraser" \
  env YO_DATASET=ml-1m EARLY_STOP=1 PATIENCE="$P_ML" REG=0.03 YO_ARMS=GLOBAL,RECERASER \
      bash code/run_blocks.sh start yone

run_step "step 5 yelp Retrain and RecEraser" \
  env YO_DATASET=yelp2018 EARLY_STOP=1 PATIENCE="$P_YE" LR=0.02 YO_ARMS=GLOBAL,RECERASER \
      bash code/run_blocks.sh start yone

wait_free
step "step 6 writing logs/final_results.txt and logs/final_paired.txt"
bash code/run_blocks.sh results > logs/final_results.txt 2>&1
{
  echo "### ml-1m (one model per seed, mode most)"
  YS_DATASET=ml-1m EARLY_STOP=1 REG=0.03 YS_MODES=most PAIRS="$PAIRS_ALL" bash code/run_blocks.sh paired
  echo ""
  echo "### yelp2018 (one model per seed, mode most)"
  YS_DATASET=yelp2018 EARLY_STOP=1 LR=0.02 YS_MODES=most PAIRS="$PAIRS_ALL" bash code/run_blocks.sh paired
} > logs/final_paired.txt 2>&1
step "all done"
