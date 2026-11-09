#!/usr/bin/env bash
# Run every experiment for the partition-quality study, SERIALLY.
#
# Why serial: the experiment prints train_time / unlearn_time, and those are
# wall-clock measurements. Running several jobs on the same GPU inflates them
# (measured: ~2x), which would make the cost column of the paper wrong. Every
# other number (Recall/NDCG, IntraSim, NR@q, load balance, affected shards) is
# independent of timing, but the cost column has to be measured on a quiet
# GPU, so everything here runs one job at a time.
#
# Usage:
#   bash code/run_all_experiments.sh              # everything, ml-1m + yelp2018
#   bash code/run_all_experiments.sh main         # main tables only
#   bash code/run_all_experiments.sh ablation     # ablations only
#   bash code/run_all_experiments.sh yelp         # yelp main only
#
# Re-running a stage overwrites its log, so the stage can be resumed by
# running that stage alone.

set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
mkdir -p logs

STAGE="${1:-all}"

# common experiment settings (identical for every arm and every stage)
COMMON="--sig_dim 64 --emb_dim 64 --epochs 50 --retrain_epochs 50 --lr 0.05 \
        --batch_size 512 --unlearn_ratio 0.01 --unlearn_mode random --seeds 0,1,2,3,4"

MAIN_ARMS="PROPOSED,PROPOSED_BAL,RANDOM_EMB,RANDOM_PERM,RANDOM_BAL,SISA,GLOBAL"
ABL_ARMS="PROPOSED,PROPOSED_BAL,RANDOM_EMB,RANDOM_PERM,RANDOM_BAL,SISA"
SMALL_ARMS="PROPOSED,RANDOM_EMB,RANDOM_PERM"

# one job at a time - never change this
run () {
    local log="$1"; shift
    echo ""
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] START  $log"
    echo "  cmd: $*"
    echo "=============================================================="
    python "$@" > "logs/$log.out" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] DONE   $log  (exit=$rc)"
    if [ $rc -ne 0 ]; then
        echo "  !! FAILED - tail of logs/$log.out:"
        tail -n 20 "logs/$log.out" | sed 's/^/     /'
    fi
    return $rc
}

wait_for_gpu () {
    # refuse to start while another experiment of this repo is still running
    while pgrep -f "exp_partition_quality.py" > /dev/null 2>&1; do
        echo "  waiting for a free slot (another exp_partition_quality.py is running)..."
        sleep 60
    done
}

banner () { echo; echo "############################################################"; echo "# $1"; echo "############################################################"; }

# ---------------------------------------------------------------- main tables
stage_main () {
    banner "MAIN - ml-1m, K=8, 5 seeds (all arms, includes GLOBAL)"
    run ex1_ml1m_serial code/exp_partition_quality.py --dataset ml-1m \
        --n_shards 8 $COMMON --arms $MAIN_ARMS

    banner "MAIN - ml-1m, K=8, unlearn_mode=most (ablation: deletion mode)"
    run abl_mode_most_serial code/exp_partition_quality.py --dataset ml-1m \
        --n_shards 8 --sig_dim 64 --emb_dim 64 --epochs 50 --retrain_epochs 50 \
        --lr 0.05 --batch_size 512 --unlearn_ratio 0.01 --unlearn_mode most \
        --seeds 0,1,2,3,4 --arms $ABL_ARMS
}

# ---------------------------------------------------------------------- yelp
stage_yelp () {
    banner "MAIN - yelp2018, K=8, 5 seeds (all arms)"
    run ex1_yelp_serial code/exp_partition_quality.py --dataset yelp2018 \
        --n_shards 8 $COMMON --arms $MAIN_ARMS
}

# ------------------------------------------------------------------ ablation
stage_ablation () {
    banner "ABLATION - number of shards K = 4 / 16 (K=8 comes from the main run)"
    for K in 4 16; do
        run "abl_K${K}_serial" code/exp_partition_quality.py --dataset ml-1m \
            --n_shards $K $COMMON --arms $ABL_ARMS
    done

    banner "ABLATION - signature dimension d = 32 / 64 / 128 / 256"
    for D in 32 64 128 256; do
        run "abl_dim${D}_serial" code/exp_partition_quality.py --dataset ml-1m \
            --n_shards 8 --sig_dim $D --emb_dim 64 --epochs 50 --retrain_epochs 50 \
            --lr 0.05 --batch_size 512 --unlearn_ratio 0.01 --unlearn_mode random \
            --seeds 0,1,2,3,4 --arms $SMALL_ARMS
    done

    banner "ABLATION - router: cosine LSH vs MinHash (Jaccard) LSH"
    run abl_router_serial code/exp_partition_quality.py --dataset ml-1m \
        --n_shards 8 --router minhash --sig_dim 64 --emb_dim 64 --epochs 50 \
        --retrain_epochs 50 --lr 0.05 --batch_size 512 --unlearn_ratio 0.01 \
        --unlearn_mode random --seeds 0,1,2,3,4 \
        --arms PROPOSED_MINHASH,PROPOSED,RANDOM_PERM

    banner "ABLATION - signature weighting: binary vs log-count"
    run abl_weight_serial code/exp_partition_quality.py --dataset ml-1m \
        --n_shards 8 --sig_weighting log_count --sig_dim 64 --emb_dim 64 \
        --epochs 50 --retrain_epochs 50 --lr 0.05 --batch_size 512 \
        --unlearn_ratio 0.01 --unlearn_mode random --seeds 0,1,2,3,4 \
        --arms PROPOSED,RANDOM_PERM
}

# ------------------------------------------------------------------- summary
stage_summary () {
    banner "SUMMARY - rebuild every table from the logs"
    python code/summarize_partition_logs.py \
        logs/ex1_ml1m_serial.out \
        logs/abl_mode_most_serial.out \
        logs/abl_K4_serial.out logs/abl_K16_serial.out \
        logs/ex1_yelp_serial.out \
        logs/abl_dim32_serial.out logs/abl_dim64_serial.out \
        logs/abl_dim128_serial.out logs/abl_dim256_serial.out \
        logs/abl_router_serial.out logs/abl_weight_serial.out \
        --baseline RANDOM_PERM --json logs/SUMMARY.json
}

case "$STAGE" in
    main)      stage_main ;;
    yelp)      stage_yelp ;;
    ablation)  stage_ablation ;;
    summary)   stage_summary ;;
    all)       stage_main; stage_yelp; stage_ablation; stage_summary ;;
    *) echo "unknown stage: $STAGE (use main|yelp|ablation|summary|all)"; exit 1 ;;
esac

echo ""
echo "=============================================================="
echo "ALL DONE"
echo "=============================================================="
