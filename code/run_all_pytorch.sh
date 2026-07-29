#!/bin/bash
# Run all 4 partitions x 2 aggregation types, save log, print summary
# Usage: bash code/run_all_pytorch.sh [epochs] [part_type]
#   epochs: 5 (default), 10, 30
#   part_type: 0 = all (default), 1=InP, 2=UBP, 3=Random, 4=IBP

EPOCHS=${1:-5}
PART_ARG=${2:-0}

LOG_FILE="results_pytorch_e${EPOCHS}.log"

echo "============================================================" | tee -a "$LOG_FILE"
echo " RecEraser PyTorch - All experiments (epochs=$EPOCHS)" | tee -a "$LOG_FILE"
echo " Started: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Function to extract metrics from output
extract_metrics() {
    local agg=$1
    local part=$2
    local output=$3
    local recall=$(echo "$output" | grep -A 1 "AGGREGATION FINAL" | grep "recall@" | head -3)
    local ndcg=$(echo "$output" | grep -A 1 "AGGREGATION FINAL" | grep "ndcg@" | head -3)
    echo "[$agg p$part] $recall $ndcg" | tee -a "$LOG_FILE"
}

# Determine partitions to run
if [ "$PART_ARG" -eq 0 ]; then
    PARTS=(1 2 3 4)
else
    PARTS=($PART_ARG)
fi

for part in "${PARTS[@]}"; do
    case $part in
        1) PART_NAME="InP";;
        2) PART_NAME="UBP";;
        3) PART_NAME="Random";;
        4) PART_NAME="IBP";;
    esac

    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo " Partition: $PART_NAME (part_type=$part)" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"

    for agg in attention mean; do
        echo "" | tee -a "$LOG_FILE"
        echo ">>> Running $agg for $PART_NAME..." | tee -a "$LOG_FILE"

        OUT=$(python code/RecEraser_BPR_pytorch.py \
            --dataset ml-1m \
            --part_type $part \
            --part_num 10 \
            --agg_type $agg \
            --epoch $EPOCHS \
            --data_path ./data/ \
            --save_flag 1 2>&1)

        echo "$OUT" >> "$LOG_FILE"
        extract_metrics "$agg" "$part" "$OUT"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " FINISHED: $(date)" | tee -a "$LOG_FILE"
echo " Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"