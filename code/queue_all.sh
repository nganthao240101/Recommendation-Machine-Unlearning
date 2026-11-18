#!/usr/bin/env bash
# One queue, run in order once the chain running now is done:
#   1) ml-1m reg 0.03: run the unlearning again (saved models are loaded) so the
#      log has the train / eval split of the unlearning time
#   2) yelp2018 with the ml-1m setting (reg 0.03, lr 0.05): patience chosen on
#      validation, then single / most / fewest (sensitivity analysis)
#   3) yelp2018 lr 0.02, patience 15: run the unlearning again (split)
#   4) yelp2018: wider reg x lr grid on validation, then the chosen setting
# A step that was already started (its old logs renamed *.nosplit) is skipped.
#
#   nohup setsid bash code/queue_all.sh > logs/queue_all.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.." || exit 1

wait_free () {
  while pgrep -f "run_blocks\.sh (regauto|esauto|yone|yseeds|compare|patience|kyelp|epochs|full|fullwmf|main|modes|single|all)$" >/dev/null; do
    sleep 120
  done
}
step () { echo "[$(date '+%F %T')] $*"; }

# ---- 1
wait_free
if ls logs/mode_*_ml1m_esu_reg0.03.out.nosplit >/dev/null 2>&1; then
  step "step 1 ml-1m: already started before, skipped"
else
  P=$(grep -o "RECOMMENDED patience = [0-9]*" logs/patience_ml1m_reg0.03.out | grep -o "[0-9]*$" | tail -1)
  step "step 1 ml-1m reg0.03 patience=$P: unlearning again"
  for m in single most fewest; do
    f="logs/mode_${m}_ml1m_esu_reg0.03.out"
    [ -f "$f" ] && mv "$f" "$f.nosplit"
  done
  YO_DATASET=ml-1m EARLY_STOP=1 PATIENCE="$P" REG=0.03 bash code/run_blocks.sh start yone
  sleep 60
fi

# ---- 2
wait_free
step "step 2 yelp with the ml-1m setting reg0.03 lr0.05"
YO_DATASET=yelp2018 EARLY_STOP=1 REG=0.03 LR=0.05 bash code/run_blocks.sh start esauto
sleep 60

# ---- 3
wait_free
if ls logs/mode_*_yelp_esu_lr0.02.out.nosplit >/dev/null 2>&1; then
  step "step 3 yelp lr0.02: already started before, skipped"
else
  step "step 3 yelp lr0.02 patience=15: unlearning again"
  for m in single most fewest; do
    f="logs/mode_${m}_yelp_esu_lr0.02.out"
    [ -f "$f" ] && mv "$f" "$f.nosplit"
  done
  YO_DATASET=yelp2018 EARLY_STOP=1 PATIENCE=15 LR=0.02 bash code/run_blocks.sh start yone
  sleep 60
fi

# ---- 4
wait_free
step "step 4 yelp: wider reg x lr grid"
[ -f logs/tunereg_yelp_grid1.out ] || mv logs/tunereg_yelp.out logs/tunereg_yelp_grid1.out 2>/dev/null
YO_DATASET=yelp2018 REG_GRID="0 0.001 0.003 0.01 0.03" LR_GRID="0.05 0.02 0.01 0.005" \
  bash code/run_blocks.sh start regauto
step "all steps queued"
