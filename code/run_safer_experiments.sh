#!/usr/bin/env bash
# Runner for the SAFER paper experiments.
#
# Every stage runs one job at a time, so the reported unlearn_time is a
# clean single-job measurement and is not inflated by competing for the GPU.
#
#   bash code/run_safer_experiments.sh <stage>
#
# Stages:
#   verify   1 seed, 1 dataset, few epochs - a smoke test of the whole chain
#   main     R1: ml-1m + yelp2018, K=8, random deletion, 5 seeds
#   single   N2: time of one single deletion request, ml-1m
#   receraser N1: RecEraser arm, ml-1m + yelp2018 (+ most/fewest on ml-1m)
#   modes    R2: most-active and fewest-active deletion, ml-1m
#   dims     R3: BPRMF embedding dim 32 / 128 / 256, ml-1m
#   all      main -> single -> receraser -> modes -> dims

set -u
cd "$(dirname "$0")/.."
mkdir -p logs

STAGE="${1:-verify}"

# ---- shared settings (spec section 0) ----
COMMON="--n_shards 8 --emb_dim 64 --sig_dim 64 --epochs 50 --retrain_epochs 50 \
--lr 0.05 --batch_size 512 --unlearn_ratio 0.01 --seeds 0,1,2,3,4"

# the main arms: SAFER A, SAFER B, SISA, Random, RandomEmb, Retrain
MAIN_ARMS="PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,GLOBAL"

banner () {
  echo ""
  echo "=============================================================="
  echo "$1"
  echo "=============================================================="
}

run () {
  local name="$1"; shift
  banner "START $name"
  local t0=$SECONDS
  "$@" 2>&1 | tee "logs/${name}.out"
  local rc=${PIPESTATUS[0]}
  echo "--------------------------------------------------------------"
  echo "DONE $name  (rc=$rc, $((SECONDS-t0))s)  -> logs/${name}.out"
  echo "--------------------------------------------------------------"
  return $rc
}

# ------------------------------------------------------------------ stages
stage_pretrain () {
  # the RECERASER arm needs pretrained WMF embeddings for its partition and
  # to seed the model. They are produced once per dataset and reused.
  for DS in ml-1m yelp2018; do
    if [ -f "data/${DS}/wmf_embeddings.npz" ]; then
      echo "[skip] data/${DS}/wmf_embeddings.npz already exists"
      continue
    fi
    banner "PRETRAIN WMF - $DS"
    python code/pretrain_wmf.py --dataset "$DS" --emb_dim 64 \
      --batch_size 512 --learning_rate 0.05 --max_epochs 100 \
      2>&1 | tee "logs/pretrain_wmf_${DS}.out"
  done
}

stage_verify () {
  # smoke test: 1 seed, 3 epochs, every new code path exercised once
  banner "VERIFY - 1 seed, 3 epochs, ml-1m, all arms + RECERASER"
  python code/exp_partition_quality.py --dataset ml-1m --n_shards 8 \
    --emb_dim 64 --sig_dim 64 --epochs 3 --retrain_epochs 3 --lr 0.05 \
    --batch_size 512 --unlearn_ratio 0.01 --unlearn_mode random --seeds 0 \
    --arms PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,GLOBAL,RECERASER \
    2>&1 | tee logs/verify_ml1m.out
  echo ""
  echo "--- the lines that matter ---"
  grep -E "instability:|per-shard retrain|S5 rate|cost:|R@10=" logs/verify_ml1m.out
}

stage_main () {
  run main_ml1m python code/exp_partition_quality.py --dataset ml-1m \
    --unlearn_mode random $COMMON --arms $MAIN_ARMS
  run main_yelp python code/exp_partition_quality.py --dataset yelp2018 \
    --unlearn_mode random $COMMON --arms $MAIN_ARMS
}

stage_single () {
  run single_request_ml1m python code/exp_single_request.py --dataset ml-1m \
    --n_shards 8 --emb_dim 64 --sig_dim 64 --epochs 50 --retrain_epochs 50 \
    --lr 0.05 --batch_size 512 --n_users 20 --seeds 0,1,2,3,4 \
    --arms PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,RECERASER,GLOBAL
}

stage_receraser () {
  run receraser_ml1m python code/exp_partition_quality.py --dataset ml-1m \
    --unlearn_mode random $COMMON --arms RECERASER
  run receraser_yelp python code/exp_partition_quality.py --dataset yelp2018 \
    --unlearn_mode random $COMMON --arms RECERASER
  run receraser_ml1m_most python code/exp_partition_quality.py --dataset ml-1m \
    --unlearn_mode most $COMMON --arms RECERASER
  run receraser_ml1m_fewest python code/exp_partition_quality.py --dataset ml-1m \
    --unlearn_mode fewest $COMMON --arms RECERASER
}

stage_modes () {
  run mode_most_ml1m python code/exp_partition_quality.py --dataset ml-1m \
    --unlearn_mode most $COMMON --arms $MAIN_ARMS
  run mode_fewest_ml1m python code/exp_partition_quality.py --dataset ml-1m \
    --unlearn_mode fewest $COMMON --arms $MAIN_ARMS
}

stage_dims () {
  # spec S4: the BPRMF embedding dim changes, the signature dim stays at 64
  for D in 32 128 256; do
    run "dim${D}_ml1m" python code/exp_partition_quality.py --dataset ml-1m \
      --unlearn_mode random --n_shards 8 --emb_dim $D --sig_dim 64 \
      --epochs 50 --retrain_epochs 50 --lr 0.05 --batch_size 512 \
      --unlearn_ratio 0.01 --seeds 0,1,2,3,4 --arms $MAIN_ARMS
  done
}

case "$STAGE" in
  pretrain)  stage_pretrain ;;
  verify)    stage_verify ;;
  main)      stage_main ;;
  single)    stage_single ;;
  receraser) stage_receraser ;;
  modes)     stage_modes ;;
  dims)      stage_dims ;;
  all)       stage_main; stage_single; stage_receraser; stage_modes; stage_dims ;;
  *) echo "unknown stage: $STAGE (use pretrain|verify|main|single|receraser|modes|dims|all)"; exit 1 ;;
esac

echo ""
echo "=============================================================="
echo "ALL DONE ($STAGE)"
echo "=============================================================="
