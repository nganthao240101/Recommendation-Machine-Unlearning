#!/bin/bash

# ============================================================================
# Script chạy ngầm với screen - không bị mất kết nối SSH
# ============================================================================

# Cài đặt
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
SESSION_NAME="unlearning_exp"

# Tạo thư mục log
mkdir -p "$LOG_DIR"

# Thời gian
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Config - có thể thay đổi
MAX_EPOCHS=${1:-50}
RETRAIN_EPOCHS=${2:-20}
N_SHARDS=${3:-8}
DATASET=${4:-ml-1m}
UNLEARN_RATIO=${5:-0.1}

# ============================================================================
# Kiểm tra và cài đặt screen
# ============================================================================

if ! command -v screen &> /dev/null; then
    echo "screen not found. Installing..."
    apt-get update && apt-get install -y screen
fi

# ============================================================================
# Kiểm tra xem có screen session đang chạy không
# ============================================================================

if screen -list | grep -q "$SESSION_NAME"; then
    echo "Screen session '$SESSION_NAME' already exists!"
    echo "To attach: screen -r $SESSION_NAME"
    echo "To kill: screen -S $SESSION_NAME -X quit"
    read -p "Kill existing session? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        screen -S "$SESSION_NAME" -X quit
        sleep 1
    else
        echo "Keeping existing session. Exiting."
        exit 0
    fi
fi

# ============================================================================
# Tạo screen session và chạy script
# ============================================================================

echo "============================================================"
echo "Starting experiment in screen session: $SESSION_NAME"
echo "Timestamp: $TIMESTAMP"
echo "Config:"
echo "  - MAX_EPOCHS: $MAX_EPOCHS"
echo "  - RETRAIN_EPOCHS: $RETRAIN_EPOCHS"
echo "  - N_SHARDS: $N_SHARDS"
echo "  - DATASET: $DATASET"
echo "  - UNLEARN_RATIO: $UNLEARN_RATIO"
echo "============================================================"
echo ""
echo "To monitor progress: screen -r $SESSION_NAME"
echo "To detach from screen: Ctrl+A, then D"
echo "To kill session: screen -S $SESSION_NAME -X quit"
echo ""

# Tạo file cấu hình cho script chạy bên trong screen
CONFIG_FILE="$LOG_DIR/config_${TIMESTAMP}.sh"
cat > "$CONFIG_FILE" << EOF
export MAX_EPOCHS=$MAX_EPOCHS
export RETRAIN_EPOCHS=$RETRAIN_EPOCHS
export N_SHARDS=$N_SHARDS
export DATASET=$DATASET
export UNLEARN_RATIO=$UNLEARN_RATIO
export TIMESTAMP=$TIMESTAMP
export SCRIPT_DIR=$SCRIPT_DIR
export LOG_DIR=$LOG_DIR
EOF

# Chạy script trong screen
screen -dmS "$SESSION_NAME" bash -c "
    cd '$SCRIPT_DIR'
    source '$CONFIG_FILE'

    LOG_FILE='$LOG_DIR/run_${TIMESTAMP}.log'

    log() {
        echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$1\" | tee -a \"\$LOG_FILE\"
    }

    log_section() {
        echo '' | tee -a \"\$LOG_FILE\"
        echo '============================================================' | tee -a \"\$LOG_FILE\"
        echo \"\$1\" | tee -a \"\$LOG_FILE\"
        echo '============================================================' | tee -a \"\$LOG_FILE\"
        echo '' | tee -a \"\$LOG_FILE\"
    }

    log_section 'BẮT ĐẦU CHẠY EXPERIMENT'

    cd '$SCRIPT_DIR'

    # Method 1: Full Retrain
    log_section 'METHOD 1: FULL RETRAIN'
    python method_1_full_retrain.py \
        --model_name BPRMF \
        --dataset \$DATASET \
        --batch_size 512 \
        --learning_rate 0.05 \
        --emb_dim 64 \
        --max_epochs \$MAX_EPOCHS \
        --early_stopping False \
        --unlearn_ratio \$UNLEARN_RATIO \
        --output_suffix \${DATASET}_s64 2>&1 | tee -a \$LOG_FILE

    # Method 2: SISA
    log_section 'METHOD 2: SISA'
    python method_2_sisa.py \
        --model_name BPRMF \
        --dataset \$DATASET \
        --batch_size 512 \
        --learning_rate 0.05 \
        --emb_dim 64 \
        --n_shards \$N_SHARDS \
        --max_epochs \$MAX_EPOCHS \
        --unlearn_ratio \$UNLEARN_RATIO \
        --retrain_epochs \$RETRAIN_EPOCHS \
        --output_suffix \${DATASET}_s64 2>&1 | tee -a \$LOG_FILE

    # Method 3: RecEraser
    log_section 'METHOD 3: RECERASER'
    python method_3_receraser.py \
        --dataset \$DATASET \
        --batch_size 512 \
        --learning_rate 0.05 \
        --emb_dim 64 \
        --attention_size 32 \
        --n_shards \$N_SHARDS \
        --max_epochs_local \$MAX_EPOCHS \
        --max_epochs_agg \$MAX_EPOCHS \
        --agg_type attention \
        --unlearn_ratio \$UNLEARN_RATIO \
        --retrain_epochs \$RETRAIN_EPOCHS \
        --output_suffix \${DATASET}_s64 2>&1 | tee -a \$LOG_FILE

    # Method 4: Ours
    log_section 'METHOD 4: OURS (3 Components)'
    python method_4_ours.py \
        --dataset \$DATASET \
        --batch_size 512 \
        --learning_rate 0.05 \
        --emb_dim 64 \
        --n_shards \$N_SHARDS \
        --max_epochs \$MAX_EPOCHS \
        --unlearn_ratio \$UNLEARN_RATIO \
        --retrain_epochs \$RETRAIN_EPOCHS \
        --output_suffix \${DATASET}_s64 2>&1 | tee -a \$LOG_FILE

    log_section 'HOÀN THÀNH!'
    echo 'All experiments done!' | tee -a \$LOG_FILE
    echo 'Press Enter to close...' | tee -a \$LOG_FILE
    read
"

echo "Screen session started: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  screen -r $SESSION_NAME    # Xem tiến trình"
echo "  screen -d $SESSION_NAME    # Thoát screen"
echo "  screen -S $SESSION_NAME -X quit   # Dừng và xóa session"
echo ""
echo "Log file: $LOG_DIR/run_${TIMESTAMP}.log"
