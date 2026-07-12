#!/bin/bash
# Quick test with 10 epochs to compare Attention vs MEAN

EPOCHS=10
DATA_PATH=./data/

echo "========================================"
echo "Quick Test: Attention vs MEAN (${EPOCHS} epochs)"
echo "========================================"

declare -A RESULTS

for part_type in 1 2 3 4; do
    case $part_type in
        1) part_name="InP" ;;
        2) part_name="UBP" ;;
        3) part_name="Random" ;;
        4) part_name="IBP" ;;
    esac

    for agg in attention mean; do
        echo ">>> Training ${part_name} ${agg}..."
        output=$(python code/RecEraser_BPR.py --dataset ml-1m --part_type $part_type --part_num 10 --agg_type $agg --epoch $EPOCHS --data_path $DATA_PATH --save_flag 0 2>&1 | grep "recall=\[")
        echo "$output"

        # Extract recall values
        recall=$(echo "$output" | grep -oP 'recall=\[\K[0-9.]+' | head -1)
        recall20=$(echo "$output" | grep -oP 'recall=\[[0-9.]+, [0-9.]+, [0-9.]+\]' | grep -oP '\d+\.\d+' | sed -n '2p')
        ndcg20=$(echo "$output" | grep -oP 'ndcg=\[[0-9.]+, [0-9.]+, [0-9.]+\]' | grep -oP '\d+\.\d+' | sed -n '2p')

        key="${part_name}_${agg}"
        RESULTS[$key]="$recall20 $ndcg20"
    done
    echo ""
done

echo "========================================"
echo "SUMMARY TABLE (Recall@20, NDCG@20)"
echo "========================================"
printf "%-12s %-12s %-12s %-12s\n" "Partition" "Attention R@20" "MEAN R@20" "Winner"
echo "----------------------------------------"

for part_type in 1 2 3 4; do
    case $part_type in
        1) part_name="InP" ;;
        2) part_name="UBP" ;;
        3) part_name="Random" ;;
        4) part_name="IBP" ;;
    esac

    attn_data=${RESULTS[${part_name}_attention]}
    mean_data=${RESULTS[${part_name}_mean]}

    attn_r20=$(echo $attn_data | awk '{print $1}')
    mean_r20=$(echo $mean_data | awk '{print $1}')
    attn_n20=$(echo $attn_data | awk '{print $2}')
    mean_n20=$(echo $mean_data | awk '{print $2}')

    # Determine winner
    winner=""
    if (( $(echo "$attn_r20 > $mean_r20" | bc -l) )); then
        winner="Attention"
    elif (( $(echo "$mean_r20 > $attn_r20" | bc -l) )); then
        winner="MEAN"
    else
        winner="Tie"
    fi

    printf "%-12s %-12.4f %-12.4f %-12s\n" "$part_name" "$attn_r20" "$mean_r20" "$winner"
done
echo "========================================"
