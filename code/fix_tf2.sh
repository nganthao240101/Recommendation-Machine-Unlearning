#!/bin/bash
# Script tự động sửa code RecEraser để chạy với TF 2.x (GPU support)
# Chạy trên Linux sau khi cd vào code directory

echo "=========================================="
echo "Auto-fix RecEraser code for TF 2.x GPU"
echo "=========================================="

# Check directory
if [ ! -f "RecEraser_BPR.py" ]; then
    echo "ERROR: Must be in code/ directory!"
    echo "Usage: cd /home/khanh/Recommendation-Machine-Unlearning/code"
    echo "       bash fix_tf2.sh"
    exit 1
fi

# Backup files first
echo ""
echo "Step 1: Backing up original files..."
mkdir -p backup_tf1
for file in RecEraser_BPR.py RecEraser_LightGCN.py RecEraser_WMF.py BPRMF.py WMF.py LightGCN.py utility/batch_test.py utility/load_data.py; do
    if [ -f "$file" ]; then
        cp "$file" "backup_tf1/$(basename $file).bak"
        echo " Backed up: $file"
    fi
done

# Fix TF 1.15 API to TF 2.x compat.v1
echo ""
echo "Step 2: Replacing TF 1.15 API with TF 2.x compat..."

for file in RecEraser_BPR.py RecEraser_LightGCN.py RecEraser_WMF.py BPRMF.py WMF.py LightGCN.py utility/batch_test.py utility/load_data.py; do
    if [ ! -f "$file" ]; then
        continue
    fi
    echo ""
    echo " Fixing: $file"

    # Replace tf.X with tf.compat.v1.X for deprecated APIs
    sed -i 's/tf\.truncated_normal/tf.compat.v1.truncated_normal/g' "$file"
    sed -i 's/tf\.train\.AdagradOptimizer/tf.compat.v1.train.AdagradOptimizer/g' "$file"
    sed -i 's/tf\.train\.AdamOptimizer/tf.compat.v1.train.AdamOptimizer/g' "$file"
    sed -i 's/tf\.train\.GradientDescentOptimizer/tf.compat.v1.train.GradientDescentOptimizer/g' "$file"
    sed -i 's/tf\.train\.Saver/tf.compat.v1.train.Saver/g' "$file"
    sed -i 's/tf\.ConfigProto/tf.compat.v1.ConfigProto/g' "$file"
    sed -i 's/tf\.Session/tf.compat.v1.Session/g' "$file"
    sed -i 's/tf\.global_variables_initializer/tf.compat.v1.global_variables_initializer/g' "$file"
    sed -i 's/tf\.placeholder/tf.compat.v1.placeholder/g' "$file"
    sed -i 's/tf\.nn\.dropout/tf.nn.dropout/g' "$file"
    sed -i 's/tf\.contrib/tf.compat.v1.estimator/g' "$file"
    # Fix any double compat
    sed -i 's/tf\.compat\.v1\.compat\.v1/tf.compat.v1/g' "$file"

    # Fix keep_dims to keepdims
    sed -i 's/keep_dims=True/keepdims=True/g' "$file"

done

# Install TF 2.x
echo ""
echo "Step 3: Installing TensorFlow 2.x..."
pip install tensorflow-gpu==2.10.0

echo ""
echo "Step 4: Verifying fixes..."
python -c "
import os
files = ['RecEraser_BPR.py', 'utility/batch_test.py', 'utility/load_data.py']
for f in files:
    if os.path.exists(f):
        with open(f, 'r') as file:
            content = file.read()
        has_tf1 = 'tf.truncated_normal' in content or 'tf.train.Saver' in content
        has_tf2 = 'tf.compat.v1' in content
        status = '[OK]' if has_tf2 and not has_tf1 else '[NEED_FIX]'
        print(f' {status} {f}')
"

echo ""
echo "=========================================="
echo "Step 5: Test GPU"
echo "=========================================="
python check_gpu.py

echo ""
echo "Step 6: Try to run UBP if GPU is available"
python -c "
import tensorflow as tf
if tf.config.list_physical_devices('GPU'):
    print('GPU available - ready to run!')
    print('')
    print('Run:')
    print(' python RecEraser_BPR.py --dataset ml-1m --part_type 2 --part_num 5 --epoch 30 --embed_size 64 --lr 0.05 --regs \"[0.01]\" --batch_size 256')
else:
    print('GPU not available - need to use CPU or fix driver')
    print('')
    print('Option 1: Run on CPU:')
    print(' CUDA_VISIBLE_DEVICES=\"\" python RecEraser_BPR.py --dataset ml-1m --part_type 2 --part_num 5 --epoch 30 --embed_size 64 --lr 0.05 --regs \"[0.01]\" --batch_size 256')
    print('')
    print('Option 2: Downgrade driver:')
    print(' sudo apt install nvidia-driver-470')
    print(' sudo reboot')
"