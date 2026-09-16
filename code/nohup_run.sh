#!/bin/bash

# ============================================================================
# Script chạy ngầm với nohup - không bị mất kết nối SSH
# ============================================================================

# Cài đặt
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Tạo thư mục log
mkdir -p "$LOG_DIR"

# Config - có thể thay đổi khi gọi script
MAX_EPOCHS=${1:-50}
RETRAIN_EPOCHS=${2:-20}
N_SHARDS=${3:-8}
DATASET=${4:-ml-1m}
UNLEARN_RATIO=${5:-0.1}

# ============================================================================
# Bắt đầu chạy
# ============================================================================

LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"
CMD_FILE="$LOG_DIR/command_${TIMESTAMP}.sh"

echo "============================================================"
echo "Starting experiment with nohup"
echo "Timestamp: $TIMESTAMP"
echo "Config:"
echo "  - MAX_EPOCHS: $MAX_EPOCHS"
echo "  - RETRAIN_EPOCHS: $RETRAIN_EPOCHS"
echo "  - N_SHARDS: $N_SHARDS"
echo "  - DATASET: $DATASET"
echo "  - UNLEARN_RATIO: $UNLEARN_RATIO"
echo "============================================================"
echo ""
echo "Log file: $LOG_FILE"
echo "Command file: $CMD_FILE"
echo ""
echo "To check progress: tail -f $LOG_FILE"
echo "To stop: kill \$(cat $LOG_DIR/experiment_${TIMESTAMP}.pid)"
echo ""

# Tạo script chạy
cat > "$CMD_FILE" << 'SCRIPT_EOF'
#!/bin/bash

LOG_FILE="__LOG_FILE__"
MAX_EPOCHS="__MAX_EPOCHS__"
RETRAIN_EPOCHS="__RETRAIN_EPOCHS__"
N_SHARDS="__N_SHARDS__"
DATASET="__DATASET__"
UNLEARN_RATIO="__UNLEARN_RATIO__"
SCRIPT_DIR="__SCRIPT_DIR__"

cd "$SCRIPT_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "$1" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

log_section "BẮT ĐẦU CHẠY EXPERIMENT"

# Method 1: Full Retrain
log_section "METHOD 1: FULL RETRAIN"
python method_1_full_retrain.py \
    --model_name BPRMF \
    --dataset $DATASET \
    --batch_size 512 \
    --learning_rate 0.05 \
    --emb_dim 64 \
    --max_epochs $MAX_EPOCHS \
    --early_stopping False \
    --unlearn_ratio $UNLEARN_RATIO \
    --output_suffix ${DATASET}_s64 2>&1 | tee -a "$LOG_FILE"

# Method 2: SISA
log_section "METHOD 2: SISA"
python method_2_sisa.py \
    --model_name BPRMF \
    --dataset $DATASET \
    --batch_size 512 \
    --learning_rate 0.05 \
    --emb_dim 64 \
    --n_shards $N_SHARDS \
    --max_epochs $MAX_EPOCHS \
    --unlearn_ratio $UNLEARN_RATIO \
    --retrain_epochs $RETRAIN_EPOCHS \
    --output_suffix ${DATASET}_s64 2>&1 | tee -a "$LOG_FILE"

# Method 3: RecEraser
log_section "METHOD 3: RECERASER"
python method_3_receraser.py \
    --dataset $DATASET \
    --batch_size 512 \
    --learning_rate 0.05 \
    --emb_dim 64 \
    --attention_size 32 \
    --n_shards $N_SHARDS \
    --max_epochs_local $MAX_EPOCHS \
    --max_epochs_agg $MAX_EPOCHS \
    --agg_type attention \
    --unlearn_ratio $UNLEARN_RATIO \
    --retrain_epochs $RETRAIN_EPOCHS \
    --output_suffix ${DATASET}_s64 2>&1 | tee -a "$LOG_FILE"

# Method 4: Ours
log_section "METHOD 4: OURS (3 Components)"
python method_4_ours.py \
    --dataset $DATASET \
    --batch_size 512 \
    --learning_rate 0.05 \
    --emb_dim 64 \
    --n_shards $N_SHARDS \
    --max_epochs $MAX_EPOCHS \
    --unlearn_ratio $UNLEARN_RATIO \
    --retrain_epochs $RETRAIN_EPOCHS \
    --output_suffix ${DATASET}_s64 2>&1 | tee -a "$LOG_FILE"

log_section "HOÀN THÀNH!"
SCRIPT_EOF

# Thay thế placeholder
sed -i "s|__LOG_FILE__|$LOG_FILE|g" "$CMD_FILE"
sed -i "s|__MAX_EPOCHS__|$MAX_EPOCHS|g" "$CMD_FILE"
sed -i "s|__RETRAIN_EPOCHS__|$RETRAIN_EPOCHS|g" "$CMD_FILE"
sed -i "s|__N_SHARDS__|$N_SHARDS|g" "$CMD_FILE"
sed -i "s|__DATASET__|$DATASET|g" "$CMD_FILE"
sed -i "s|__UNLEARN_RATIO__|$UNLEARN_RATIO|g" "$CMD_FILE"
sed -i "s|__SCRIPT_DIR__|$SCRIPT_DIR|g" "$CMD_FILE"

# Chạy với nohup
chmod +x "$CMD_FILE"
nohup bash "$CMD_FILE" > /dev/null 2>&1 &

# Lưu PID
echo $! > "$LOG_DIR/experiment_${TIMESTAMP}.pid"

echo "Experiment started!"
echo "PID: $(cat $LOG_DIR/experiment_${TIMESTAMP}.pid)"
echo ""
echo "Check progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check if running:"
echo "  ps aux | grep experiment_${TIMESTAMP}"
echo ""
echo "Stop:"
echo "  kill $(cat $LOG_DIR/experiment_${TIMESTAMP}.pid)"
