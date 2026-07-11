#!/bin/bash
# Run all experiments with same settings and show results

EPOCHS=30
DATA_PATH=./data/

echo "============================================"
echo "Running all experiments with $EPOCHS epochs"
echo "============================================"

# InP
echo ""
echo ">>> InP Attention"
python code/RecEraser_BPR.py --dataset ml-1m --part_type 1 --part_num 10 --agg_type attention --epoch $EPOCHS --data_path $DATA_PATH --save_flag 1

echo ">>> InP MEAN"
python code/RecEraser_BPR.py --dataset ml-1m --part_type 1 --part_num 10 --agg_type mean --epoch $EPOCHS --data_path $DATA_PATH --save_flag 1

# UBP
echo ""
echo ">>> UBP Attention"
python code/RecEraser_BPR.py --dataset ml-1m --part_type 2 --part_num 10 --agg_type attention --epoch $EPOCHS --data_path $DATA_PATH --save_flag 1

echo ">>> UBP MEAN"
python code/RecEraser_BPR.py --dataset ml-1m --part_type 2 --part_num 10 --agg_type mean --epoch $EPOCHS --data_path $DATA_PATH --save_flag 1

# Random
echo ""
echo ">>> Random Attention"
python code/RecEraser_BPR.py --dataset ml-1m --part_type 3 --part_num 10 --agg_type attention --epoch $EPOCHS --data_path $DATA_PATH --save_flag 1

echo ">>> Random MEAN"
python code/RecEraser_BPR.py --dataset ml-1m --part_type 3 --part_num 10 --agg_type mean --epoch $EPOCHS --data_path $DATA_PATH --save_flag 1

# IBP
echo ""
echo ">>> IBP Attention"
python code/RecEraser_BPR.py --dataset ml-1m --part_type 4 --part_num 10 --agg_type attention --epoch $EPOCHS --data_path $DATA_PATH --save_flag 1

echo ">>> IBP MEAN"
python code/RecEraser_BPR.py --dataset ml-1m --part_type 4 --part_num 10 --agg_type mean --epoch $EPOCHS --data_path $DATA_PATH --save_flag 1

echo ""
echo "============================================"
echo "All experiments completed!"
echo "============================================"

# Show results
echo ""
python code/show_results.py
