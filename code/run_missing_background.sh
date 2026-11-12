#!/usr/bin/env bash
# ============================================================================
# run_missing_background.sh - chạy TẤT CẢ thí nghiệm SAFER còn thiếu, NỀN.
#
#   Nhân sẻ chạy 1 job / 1 lần (để unlearn_time là số đo 1-job sạch, không
#   bị tranh GPU), và SKIP log đã có "CONSOLE SUMMARY" -> chạy lại được sau
#   khi bị gián đoạn mà không mất tiến độ.
#
#   nohup bash code/run_missing_background.sh > logs/full.out 2>&1 &
#   tail -f logs/full.out
# ============================================================================
set -u
cd "$(dirname "$0")/.."
mkdir -p logs

COMMON="--n_shards 8 --emb_dim 64 --sig_dim 64 --epochs 50 --retrain_epochs 50 \
--lr 0.05 --batch_size 512 --unlearn_ratio 0.01 --seeds 0,1,2,3,4"
MAIN_ARMS="PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,GLOBAL"

banner () {
  echo ""
  echo "=============================================================="
  echo "$1   ($(date +%H:%M:%S))"
  echo "=============================================================="
}

run () {
  local name="$1"; shift
  if [ -f "logs/${name}.out" ] && grep -q "CONSOLE SUMMARY" "logs/${name}.out" 2>/dev/null; then
    echo "[skip] $name  (đã xong - log đã có CONSOLE SUMMARY)"
    return 0
  fi
  banner "START $name"
  local t0=$SECONDS
  python "$@" 2>&1 | tee "logs/${name}.out"
  local rc=${PIPESTATUS[0]}
  echo "--------------------------------------------------------------"
  echo "DONE $name  (rc=$rc, $((SECONDS-t0))s)  -> logs/${name}.out"
  echo "--------------------------------------------------------------"
}

# ---------------------------------------------------------------------------
# [0] PREFLIGHT - in ra cảnh báo, KHÔNG chặn job.
#     Quan trọng: nếu torch.cuda.is_available() == False thì mọi thí nghiệm
#     chạy CPU và số đo thời gian sẽ tệ hơn hẳn bản GPU.
# ---------------------------------------------------------------------------
banner "PREFLIGHT"
python - <<'PY' 2>&1 | sed 's/^/  /'
import torch, sys
print(f"torch {torch.__version__}  cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}  n_gpu={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    print("!! CANH BAO: chay CPU. Ket qua/thoi gian se khac he so voi ban GPU.")
    print("!! Fix: pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128")
for m in ("numpy", "scipy", "sklearn"):
    try:
        __import__(m); print(f"  {m}: ok")
    except Exception as e:
        print(f"  {m}: THIEU -> {e}")
PY
echo ""

# ---------------------------------------------------------------------------
# [1] WMF embeddings cho RecEraser.
#     Đã có sẵn trên máy (data/<ds>/wmf_embeddings.npz) -> không cần train lại.
#     Nếu chỉ có .pk thì convert (convert_wmf_pk.py), vẫn không train lại.
#     Chỉ khi thiếu cả hai mới train WMF từ đầu (chậm nhất).
# ---------------------------------------------------------------------------
for DS in ml-1m yelp2018; do
  NPZ="data/${DS}/wmf_embeddings.npz"
  if [ -f "$NPZ" ]; then
    echo "[skip] ${DS}: $NPZ đã tồn tại"
    continue
  fi
  if [ -f "data/${DS}/user_pretrain.pk" ] && [ -f "data/${DS}/item_pretrain.pk" ]; then
    banner "CONVERT WMF PK -> NPZ - $DS"
    python code/convert_wmf_pk.py --dataset "$DS" 2>&1 | tee "logs/convert_wmf_${DS}.out"
    continue
  fi
  banner "PRETRAIN WMF - $DS"
  python code/pretrain_wmf.py --dataset "$DS" --emb_dim 64 \
    --batch_size 512 --learning_rate 0.05 --max_epochs 100 \
    2>&1 | tee "logs/pretrain_wmf_${DS}.out"
done

# ---------------------------------------------------------------------------
# [2] R1 - main: random deletion, 5 seeds, 6 arms, 2 dataset
# ---------------------------------------------------------------------------
run main_ml1m python code/exp_partition_quality.py --dataset ml-1m \
  --unlearn_mode random $COMMON --arms $MAIN_ARMS
run main_yelp python code/exp_partition_quality.py --dataset yelp2018 \
  --unlearn_mode random $COMMON --arms $MAIN_ARMS

# ---------------------------------------------------------------------------
# [3] N2 - một request xóa đơn lẻ (ml-1m)
# ---------------------------------------------------------------------------
run single_request_ml1m python code/exp_single_request.py --dataset ml-1m \
  --n_shards 8 --emb_dim 64 --sig_dim 64 --epochs 50 --retrain_epochs 50 \
  --lr 0.05 --batch_size 512 --n_users 20 --seeds 0,1,2,3,4 \
  --arms PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,RECERASER,GLOBAL

# ---------------------------------------------------------------------------
# [4] N1 - RecEraser arm (random / most / fewest)
# ---------------------------------------------------------------------------
run receraser_ml1m python code/exp_partition_quality.py --dataset ml-1m \
  --unlearn_mode random $COMMON --arms RECERASER
run receraser_yelp python code/exp_partition_quality.py --dataset yelp2018 \
  --unlearn_mode random $COMMON --arms RECERASER
run receraser_ml1m_most python code/exp_partition_quality.py --dataset ml-1m \
  --unlearn_mode most $COMMON --arms RECERASER
run receraser_ml1m_fewest python code/exp_partition_quality.py --dataset ml-1m \
  --unlearn_mode fewest $COMMON --arms RECERASER

# ---------------------------------------------------------------------------
# [5] R2 - deletion mode: most-active / fewest-active
# ---------------------------------------------------------------------------
run mode_most_ml1m python code/exp_partition_quality.py --dataset ml-1m \
  --unlearn_mode most $COMMON --arms $MAIN_ARMS
run mode_fewest_ml1m python code/exp_partition_quality.py --dataset ml-1m \
  --unlearn_mode fewest $COMMON --arms $MAIN_ARMS

# ---------------------------------------------------------------------------
# [6] R3 - backbone dim ablation (emb_dim đổi, sig_dim giữ 64)
# ---------------------------------------------------------------------------
for D in 32 128 256; do
  run dim${D}_ml1m python code/exp_partition_quality.py --dataset ml-1m \
    --unlearn_mode random --n_shards 8 --emb_dim $D --sig_dim 64 \
    --epochs 50 --retrain_epochs 50 --lr 0.05 --batch_size 512 \
    --unlearn_ratio 0.01 --seeds 0,1,2,3,4 --arms $MAIN_ARMS
done

banner "ALL DONE"
echo "Tổng hợp: python code/summarize_logs.py"
