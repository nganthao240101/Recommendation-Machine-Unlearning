#!/usr/bin/env bash
# Everything still to run for the paper, in order, once the chain running now
# is done. Runs that are already finished are kept and skipped (every step
# resumes its logs), so the script can be started again at any time.
#
#   0) ml-1m   SISA as defined (5 slices) in the single / most / fewest logs
#   1) ml-1m   per-seed test (5 seeds, single/most/fewest; a seed's model is
#              trained once and shared by the 3 modes): SAFER-PI vs Random,
#              and the other baselines  -> paired t-test
#   2) yelp    the new arms in the single / most / fewest logs (one model),
#              yelp's own setting (lr 0.02, patience 15), nothing changed
#   3) yelp    per-seed test (5 seeds, single/most/fewest), same pairs
#   4) ml-1m   Retrain and RecEraser (same setting, for reference)
#   5) yelp    Retrain and RecEraser
#   6) results and paired tests written to logs/final_results.txt and
#      logs/final_paired.txt
#
#   nohup setsid bash code/queue_full.sh > logs/queue_full.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.." || exit 1

wait_free () {
  while pgrep -f "run_blocks\.sh (regauto|ytune|esauto|yone|yseeds|compare|patience|kyelp|epochs|full|fullwmf|main|modes|single|all)$" >/dev/null; do
    sleep 120
  done
}
step () { echo "[$(date '+%F %T')] $*"; }
run_step () { wait_free; step "$1"; shift; "$@"; sleep 60; }

P_ML=$(grep -o "RECOMMENDED patience = [0-9]*" logs/patience_ml1m_reg0.03.out 2>/dev/null | grep -o "[0-9]*$" | tail -1)
P_ML=${P_ML:-30}
P_YE=15                                  # the patience of the yelp lr 0.02 logs
NEW_ARMS=SAFER_P,SAFER_PI,SISA_FULL
# the per-seed test: SAFER against every baseline it can be run against on 5
# seeds, and the ablations (Retrain and RecEraser: one-model logs, steps 4-5)
TEST_ARMS=SAFER_P,RANDOM_PERM,SAFER_PI,SISA,SISA_FULL,RANDOM_EMB,PROPOSED_BAL
PAIRS_ALL="SAFER_P:RANDOM_PERM SAFER_P:SISA SAFER_P:SISA_FULL SAFER_P:RANDOM_EMB SAFER_P:PROPOSED_BAL SAFER_PI:SAFER_P"


run_step "step 0 ml-1m SISA as defined (5 slices), single/most/fewest (patience $P_ML)" \
  env YO_DATASET=ml-1m EARLY_STOP=1 PATIENCE="$P_ML" REG=0.03 YO_ARMS=SISA_FULL \
      bash code/run_blocks.sh start yone

run_step "step 1 ml-1m per-seed test, all modes (patience $P_ML)" \
  env YS_DATASET=ml-1m EARLY_STOP=1 PATIENCE="$P_ML" REG=0.03 YS_MODES="single most fewest" \
      YS_FIRST="$TEST_ARMS" YS_STEPS=1 PAIRS="$PAIRS_ALL" \
      bash code/run_blocks.sh start yseeds

run_step "step 2 yelp new arms, single/most/fewest (lr 0.02, patience $P_YE)" \
  env YO_DATASET=yelp2018 EARLY_STOP=1 PATIENCE="$P_YE" LR=0.02 YO_ARMS="$NEW_ARMS" \
      bash code/run_blocks.sh start yone

run_step "step 3 yelp per-seed test, all modes" \
  env YS_DATASET=yelp2018 EARLY_STOP=1 PATIENCE="$P_YE" LR=0.02 YS_MODES="single most fewest" \
      YS_FIRST="$TEST_ARMS" YS_STEPS=1 PAIRS="$PAIRS_ALL" \
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
  echo "### ml-1m (one model per seed, single / most / fewest)"
  YS_DATASET=ml-1m EARLY_STOP=1 REG=0.03 YS_MODES="single most fewest" PAIRS="$PAIRS_ALL" bash code/run_blocks.sh paired
  echo ""
  echo "### yelp2018 (one model per seed, single / most / fewest)"
  YS_DATASET=yelp2018 EARLY_STOP=1 LR=0.02 YS_MODES="single most fewest" PAIRS="$PAIRS_ALL" bash code/run_blocks.sh paired
} > logs/final_paired.txt 2>&1
step "all done"
