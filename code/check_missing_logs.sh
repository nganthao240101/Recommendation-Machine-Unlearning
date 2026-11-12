#!/usr/bin/env bash
# Which experiment logs are already there, and which are still missing.
#
#   bash code/check_missing_logs.sh          # report only
#   bash code/check_missing_logs.sh --run    # report, then run the missing ones
#
# A log counts as DONE when it contains the CONSOLE SUMMARY block. A log that
# was interrupted is reported separately, so it is never mistaken for a
# finished run.

cd "$(dirname "$0")/.."
mkdir -p logs

# name | command to produce it
declare -a SPECS=(
  "main_ml1m|python code/exp_partition_quality.py --dataset ml-1m --unlearn_mode random --n_shards 8 --emb_dim 64 --sig_dim 64 --epochs 50 --retrain_epochs 50 --lr 0.05 --batch_size 512 --unlearn_ratio 0.01 --seeds 0,1,2,3,4 --arms PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,GLOBAL"
  "main_yelp|python code/exp_partition_quality.py --dataset yelp2018 --unlearn_mode random --n_shards 8 --emb_dim 64 --sig_dim 64 --epochs 50 --retrain_epochs 50 --lr 0.05 --batch_size 512 --unlearn_ratio 0.01 --seeds 0,1,2,3,4 --arms PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,GLOBAL"
  "single_request_ml1m|python code/exp_single_request.py --dataset ml-1m --n_shards 8 --emb_dim 64 --sig_dim 64 --epochs 50 --retrain_epochs 50 --lr 0.05 --batch_size 512 --n_users 20 --seeds 0,1,2,3,4 --arms PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,RECERASER,GLOBAL"
  "receraser_ml1m|python code/exp_partition_quality.py --dataset ml-1m --unlearn_mode random --n_shards 8 --emb_dim 64 --sig_dim 64 --epochs 50 --retrain_epochs 50 --lr 0.05 --batch_size 512 --unlearn_ratio 0.01 --seeds 0,1,2,3,4 --arms RECERASER"
  "receraser_yelp|python code/exp_partition_quality.py --dataset yelp2018 --unlearn_mode random --n_shards 8 --emb_dim 64 --sig_dim 64 --epochs 50 --retrain_epochs 50 --lr 0.05 --batch_size 512 --unlearn_ratio 0.01 --seeds 0,1,2,3,4 --arms RECERASER"
  "receraser_ml1m_most|python code/exp_partition_quality.py --dataset ml-1m --unlearn_mode most --n_shards 8 --emb_dim 64 --sig_dim 64 --epochs 50 --retrain_epochs 50 --lr 0.05 --batch_size 512 --unlearn_ratio 0.01 --seeds 0,1,2,3,4 --arms RECERASER"
  "receraser_ml1m_fewest|python code/exp_partition_quality.py --dataset ml-1m --unlearn_mode fewest --n_shards 8 --emb_dim 64 --sig_dim 64 --epochs 50 --retrain_epochs 50 --lr 0.05 --batch_size 512 --unlearn_ratio 0.01 --seeds 0,1,2,3,4 --arms RECERASER"
  "mode_most_ml1m|python code/exp_partition_quality.py --dataset ml-1m --unlearn_mode most --n_shards 8 --emb_dim 64 --sig_dim 64 --epochs 50 --retrain_epochs 50 --lr 0.05 --batch_size 512 --unlearn_ratio 0.01 --seeds 0,1,2,3,4 --arms PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,GLOBAL"
  "mode_fewest_ml1m|python code/exp_partition_quality.py --dataset ml-1m --unlearn_mode fewest --n_shards 8 --emb_dim 64 --sig_dim 64 --epochs 50 --retrain_epochs 50 --lr 0.05 --batch_size 512 --unlearn_ratio 0.01 --seeds 0,1,2,3,4 --arms PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,GLOBAL"
  "dim32_ml1m|python code/exp_partition_quality.py --dataset ml-1m --unlearn_mode random --n_shards 8 --emb_dim 32 --sig_dim 64 --epochs 50 --retrain_epochs 50 --lr 0.05 --batch_size 512 --unlearn_ratio 0.01 --seeds 0,1,2,3,4 --arms PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,GLOBAL"
  "dim128_ml1m|python code/exp_partition_quality.py --dataset ml-1m --unlearn_mode random --n_shards 8 --emb_dim 128 --sig_dim 64 --epochs 50 --retrain_epochs 50 --lr 0.05 --batch_size 512 --unlearn_ratio 0.01 --seeds 0,1,2,3,4 --arms PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,GLOBAL"
  "dim256_ml1m|python code/exp_partition_quality.py --dataset ml-1m --unlearn_mode random --n_shards 8 --emb_dim 256 --sig_dim 64 --epochs 50 --retrain_epochs 50 --lr 0.05 --batch_size 512 --unlearn_ratio 0.01 --seeds 0,1,2,3,4 --arms PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,GLOBAL"
)

RUN=0
[ "${1:-}" = "--run" ] && RUN=1

DONE=(); MISSING=(); PARTIAL=()

for spec in "${SPECS[@]}"; do
  name="${spec%%|*}"
  cmd="${spec#*|}"
  f="logs/${name}.out"
  if [ ! -f "$f" ]; then
    MISSING+=("$name")
    printf '  %-26s MISSING\n' "$name"
  elif grep -q 'CONSOLE SUMMARY' "$f"; then
    DONE+=("$name")
    printf '  %-26s DONE      (%s arms, %s seeds)\n' "$name" \
      "$(grep -c '^--- ARM' "$f")" \
      "$(grep -o 'seed [0-9]*' "$f" | sort -u | wc -l)"
  else
    PARTIAL+=("$name")
    printf '  %-26s PARTIAL   (%s/%s arms, no CONSOLE SUMMARY)\n' "$name" \
      "$(grep -c '^--- ARM' "$f")" \
      "$(echo "$cmd" | grep -o 'PROPOSED[^ ]*' | tr ',' '\n' | wc -l)"
  fi
done

echo ""
echo "done=${#DONE[@]}  partial=${#PARTIAL[@]}  missing=${#MISSING[@]}"

if [ "$RUN" = "1" ] && [ ${#MISSING[@]} -gt 0 ]; then
  echo ""
  echo "=== running the missing ones, one at a time ==="
  for name in "${MISSING[@]}"; do
    for spec in "${SPECS[@]}"; do
      [ "${spec%%|*}" = "$name" ] || continue
      cmd="${spec#*|}"
      echo ""
      echo "--------------------------------------------------------------"
      echo "START $name"
      echo "--------------------------------------------------------------"
      t0=$SECONDS
      $cmd 2>&1 | tee "logs/${name}.out"
      echo "DONE $name ($((SECONDS-t0))s) -> logs/${name}.out"
    done
  done
fi
