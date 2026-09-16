#!/bin/bash

# ============================================================================
# Script chạy ngầm tất cả methods - không bị gián đoạn khi mất SSH
# ============================================================================

# Cài đặt
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
RESULTS_DIR="$SCRIPT_DIR"

# Tạo thư mục log
mkdir -p "$LOG_DIR"

# Thời gian
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/run_${TIMESTAMP}.pid"

# Config - có thể thay đổi
MAX_EPOCHS=${1:-50}          # Default 50 epochs cho test nhanh
RETRAIN_EPOCHS=${2:-20}      # Default 20 epochs cho retrain
N_SHARDS=${3:-8}             # Default 8 shards
DATASET=${4:-ml-1m}         # Default ml-1m
UNLEARN_RATIO=${5:-0.1}      # Default 10%

# ============================================================================
# Hàm ghi log
# ============================================================================

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

# ============================================================================
# Chạy một method
# ============================================================================

run_method() {
    local method_name=$1
    local command=$2
    local output_file=$3

    log "Bắt đầu: $method_name"
    log "Command: $command"

    # Chạy với timeout 24h, log output
    timeout 86400 bash -c "$command" >> "$LOG_FILE" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log "Hoàn thành: $method_name"
    elif [ $exit_code -eq 124 ]; then
        log "TIMEOUT: $method_name (quá 24h)"
    else
        log "LỖI: $method_name (exit code: $exit_code)"
    fi

    echo $exit_code > "$LOG_DIR/${method_name}_exit_code.txt"
}

# ============================================================================
# MAIN
# ============================================================================

{
    log_section "BẮT ĐẦU CHẠY EXPERIMENT"
    log "Timestamp: $TIMESTAMP"
    log "Config:"
    log "  - MAX_EPOCHS: $MAX_EPOCHS"
    log "  - RETRAIN_EPOCHS: $RETRAIN_EPOCHS"
    log "  - N_SHARDS: $N_SHARDS"
    log "  - DATASET: $DATASET"
    log "  - UNLEARN_RATIO: $UNLEARN_RATIO"
    log "  - Log file: $LOG_FILE"

    cd "$SCRIPT_DIR"

    # Method 1: Full Retrain
    log_section "METHOD 1: FULL RETRAIN"
    CMD1="python method_1_full_retrain.py \
        --model_name BPRMF \
        --dataset $DATASET \
        --batch_size 512 \
        --learning_rate 0.05 \
        --emb_dim 64 \
        --max_epochs $MAX_EPOCHS \
        --early_stopping False \
        --unlearn_ratio $UNLEARN_RATIO \
        --output_suffix ${DATASET}_s64"
    run_method "Method1_FullRetrain" "$CMD1" "results_full_retrain_bprmf_${DATASET}_s64.json"

    # Method 2: SISA
    log_section "METHOD 2: SISA"
    CMD2="python method_2_sisa.py \
        --model_name BPRMF \
        --dataset $DATASET \
        --batch_size 512 \
        --learning_rate 0.05 \
        --emb_dim 64 \
        --n_shards $N_SHARDS \
        --max_epochs $MAX_EPOCHS \
        --unlearn_ratio $UNLEARN_RATIO \
        --retrain_epochs $RETRAIN_EPOCHS \
        --output_suffix ${DATASET}_s64"
    run_method "Method2_SISA" "$CMD2" "results_sisa_bprmf_${DATASET}_s64.json"

    # Method 3: RecEraser
    log_section "METHOD 3: RECERASER"
    CMD3="python method_3_receraser.py \
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
        --output_suffix ${DATASET}_s64"
    run_method "Method3_RecEraser" "$CMD3" "results_receraser_attention_${DATASET}_s64.json"

    # Method 4: Ours
    log_section "METHOD 4: OURS (3 Components)"
    CMD4="python method_4_ours.py \
        --dataset $DATASET \
        --batch_size 512 \
        --learning_rate 0.05 \
        --emb_dim 64 \
        --n_shards $N_SHARDS \
        --max_epochs $MAX_EPOCHS \
        --unlearn_ratio $UNLEARN_RATIO \
        --retrain_epochs $RETRAIN_EPOCHS \
        --output_suffix ${DATASET}_s64"
    run_method "Method4_Ours" "$CMD4" "results_ours_3components_${DATASET}_s64.json"

    # Tổng kết
    log_section "TỔNG KẾT"
    log "Hoàn thành tất cả methods!"

    # Copy log to results
    cp "$LOG_FILE" "$SCRIPT_DIR/results_experiment_${TIMESTAMP}.log"

} 2>&1 | tee -a "$LOG_FILE"

echo "============================================================"
echo "PID của script: $$"
echo "Log file: $LOG_FILE"
echo "============================================================"
