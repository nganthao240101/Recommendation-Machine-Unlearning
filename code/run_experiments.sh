#!/bin/bash
# Run experiments for InP and UBP partitions with WMF embeddings

echo "=========================================="
echo "STEP 1: Train WMF (generate embeddings)"
echo "=========================================="

# Train WMF
nohup python code/WMF.py \
    --dataset ml-1m \
    --embed_size 64 \
    --lr 0.05 \
    --epoch 1000 \
    > logs_WMF.log 2>&1 &

# Wait for WMF to finish
echo "Waiting for WMF to finish..."
wait

echo "WMF done! Checking embeddings..."
ls -la data/ml-1m/*_pretrain*.pk

echo ""
echo "=========================================="
echo "STEP 2: Create Partitions (InP and UBP)"
echo "=========================================="

# Create InP partition
echo "Creating InP partition..."
python -c "
import sys
sys.path.insert(0, 'code')
from utility.data_partition import data_partition_1
from utility.load_data import Data
import pickle

data = Data(path='data/ml-1m/', batch_size=512, part_type=1, part_num=10, part_T=5)
C, users, items = data_partition_1(data.train_items, 10, 5)
with open('data/ml-1m/C_type-1_num-10.pk', 'wb') as f:
    pickle.dump(C, f)
print('InP partition created!')
"

# Create UBP partition
echo "Creating UBP partition..."
python -c "
import sys
sys.path.insert(0, 'code')
from utility.data_partition import data_partition_2
from utility.load_data import Data
import pickle

data = Data(path='data/ml-1m/', batch_size=512, part_type=2, part_num=10, part_T=5)
C, users, items = data_partition_2(data.train_items, 10, 5)
with open('data/ml-1m/C_type-2_num-10.pk', 'wb') as f:
    pickle.dump(C, f)
print('UBP partition created!')
"

echo ""
echo "=========================================="
echo "STEP 3: Train RecEraser - InP (Partition 1)"
echo "=========================================="

nohup python code/RecEraser_BPR_pytorch.py \
    --dataset ml-1m \
    --part_type 1 \
    --part_num 10 \
    --agg_type attention \
    --epoch 1000 \
    --epoch_agg 50 \
    --embed_size 64 \
    --lr 0.05 \
    --regs 0.01 \
    --pretrain 0 \
    > logs_attention_p1.log 2>&1 &

echo "InP training started in background"

echo ""
echo "=========================================="
echo "STEP 4: Train RecEraser - UBP (Partition 2)"
echo "=========================================="

nohup python code/RecEraser_BPR_pytorch.py \
    --dataset ml-1m \
    --part_type 2 \
    --part_num 10 \
    --agg_type attention \
    --epoch 1000 \
    --epoch_agg 50 \
    --embed_size 64 \
    --lr 0.05 \
    --regs 0.01 \
    --pretrain 0 \
    > logs_attention_p2.log 2>&1 &

echo "UBP training started in background"

echo ""
echo "=========================================="
echo "All jobs started! Check status with:"
echo "  ps aux | grep python"
echo "  tail -f logs_attention_p1.log"
echo "  tail -f logs_attention_p2.log"
echo "=========================================="
