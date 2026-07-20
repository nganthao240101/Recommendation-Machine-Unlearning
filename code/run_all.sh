#!/bin/bash
# Run all experiments and print comparison results

EPOCHS=30
DATA_PATH=./data/

echo "============================================"
echo "Running all experiments with $EPOCHS epochs"
echo "============================================"

declare -A RESULTS

# Function to extract final recall and ndcg
extract_results() {
    local part_name=$1
    local agg=$2
    local output=$3

    # Extract last recall line (format: recall=[0.xxx, 0.xxx, 0.xxx])
    local recall_line=$(echo "$output" | grep "recall=\[0\." | tail -1)
    local recall10=$(echo "$recall_line" | grep -oP 'recall=\[\K[0-9.]+')
    local recall20=$(echo "$recall_line" | grep -oP 'recall=\[[0-9.]+,\s*\K[0-9.]+')
    local recall50=$(echo "$recall_line" | grep -oP 'recall=\[[0-9.]+,\s*[0-9.]+,\s*\K[0-9.]+')

    # Extract NDCG
    local ndcg_line=$(echo "$output" | grep "ndcg=\[" | tail -1)
    local ndcg20=$(echo "$ndcg_line" | grep -oP 'ndcg=\[[0-9.]+,\s*\K[0-9.]+')

    RESULTS["${part_name}_${agg}_r10"]=$recall10
    RESULTS["${part_name}_${agg}_r20"]=$recall20
    RESULTS["${part_name}_${agg}_r50"]=$recall50
    RESULTS["${part_name}_${agg}_n20"]=$ndcg20
}

# InP
echo ""
echo ">>> InP Attention"
OUT=$(python code/RecEraser_BPR.py --dataset ml-1m --part_type 1 --part_num 10 --agg_type attention --epoch $EPOCHS --data_path $DATA_PATH --save_flag 1 2>&1)
extract_results "InP" "attn" "$OUT"
echo "$OUT" | tail -1

echo ">>> InP MEAN"
OUT=$(python code/RecEraser_BPR.py --dataset ml-1m --part_type 1 --part_num 10 --agg_type mean --epoch $EPOCHS --data_path $DATA_PATH --save_flag 1 2>&1)
extract_results "InP" "mean" "$OUT"
echo "$OUT" | tail -1

# UBP
echo ""
echo ">>> UBP Attention"
OUT=$(python code/RecEraser_BPR.py --dataset ml-1m --part_type 2 --part_num 10 --agg_type attention --epoch $EPOCHS --data_path $DATA_PATH --save_flag 1 2>&1)
extract_results "UBP" "attn" "$OUT"
echo "$OUT" | tail -1

echo ">>> UBP MEAN"
OUT=$(python code/RecEraser_BPR.py --dataset ml-1m --part_type 2 --part_num 10 --agg_type mean --epoch $EPOCHS --data_path $DATA_PATH --save_flag 1 2>&1)
extract_results "UBP" "mean" "$OUT"
echo "$OUT" | tail -1

# Random
echo ""
echo ">>> Random Attention"
OUT=$(python code/RecEraser_BPR.py --dataset ml-1m --part_type 3 --part_num 10 --agg_type attention --epoch $EPOCHS --data_path $DATA_PATH --save_flag 1 2>&1)
extract_results "Random" "attn" "$OUT"
echo "$OUT" | tail -1

echo ">>> Random MEAN"
OUT=$(python code/RecEraser_BPR.py --dataset ml-1m --part_type 3 --part_num 10 --agg_type mean --epoch $EPOCHS --data_path $DATA_PATH --save_flag 1 2>&1)
extract_results "Random" "mean" "$OUT"
echo "$OUT" | tail -1

# IBP
echo ""
echo ">>> IBP Attention"
OUT=$(python code/RecEraser_BPR.py --dataset ml-1m --part_type 4 --part_num 10 --agg_type attention --epoch $EPOCHS --data_path $DATA_PATH --save_flag 1 2>&1)
extract_results "IBP" "attn" "$OUT"
echo "$OUT" | tail -1

echo ">>> IBP MEAN"
OUT=$(python code/RecEraser_BPR.py --dataset ml-1m --part_type 4 --part_num 10 --agg_type mean --epoch $EPOCHS --data_path $DATA_PATH --save_flag 1 2>&1)
extract_results "IBP" "mean" "$OUT"
echo "$OUT" | tail -1

echo ""
echo "============================================"
echo "COMPARISON TABLE (30 epochs)"
echo "============================================"
printf "%-12s %-10s %-10s %-10s %-10s %-10s %-10s\n" "Partition" "Attn R@20" "Mean R@20" "Attn N@20" "Mean N@20" "Winner" "Improvement"
echo "--------------------------------------------"

for part in InP UBP Random IBP; do
    attn_r20=${RESULTS["${part}_attn_r20"]}
    mean_r20=${RESULTS["${part}_mean_r20"]}
    attn_n20=${RESULTS["${part}_attn_n20"]}
    mean_n20=${RESULTS["${part}_mean_n20"]}

    # Calculate winner and improvement
    winner=""
    improvement=""
    if [ ! -z "$attn_r20" ] && [ ! -z "$mean_r20" ]; then
        # Convert to float and compare using awk
        diff=$(awk "BEGIN {printf \"%.4f\", $attn_r20 - $mean_r20}")
        pct=$(awk "BEGIN {printf \"%.1f\", ($attn_r20 - $mean_r20) / $mean_r20 * 100}")

        if (( $(echo "$diff > 0" | bc -l 2>/dev/null || echo 0) )); then
            winner="Attention"
            improvement="+${pct}%"
        else
            winner="MEAN"
            pct=$(awk "BEGIN {printf \"%.1f\", ($mean_r20 - $attn_r20) / $attn_r20 * 100}")
            improvement="+${pct}%"
        fi
    fi

    printf "%-12s %-10s %-10s %-10s %-10s %-10s %-10s\n" "$part" "$attn_r20" "$mean_r20" "$attn_n20" "$mean_n20" "$winner" "$improvement"
done

echo "============================================"
echo "All experiments completed!"
echo "Weights saved in weights/ml-1m/RecEraser_BPR/"
echo "============================================"
