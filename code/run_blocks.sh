#!/usr/bin/env bash
# ============================================================================
# run_blocks.sh - chạy thí nghiệm SAFER TỪNG KHỐI một, NỀN.
#
#   Mỗi khối là 1 lệnh nohup độc lập, log riêng (logs/block_<tên>.out),
#   nên có thể chạy theo thứ tự, xem/xử lý từng khối, hoặc bỏ qua khối nào
#   thiếu. SKIP tự động: log đã có "CONSOLE SUMMARY" thì bỏ qua.
#
#   Cách dùng (trên máy SSH, tại thư mục repo):
#     bash code/run_blocks.sh torch        # kiểm tra CUDA trước khi chạy
#     bash code/run_blocks.sh main         # R1
#     bash code/run_blocks.sh single       # N2
#     bash code/run_blocks.sh receraser    # N1
#     bash code/run_blocks.sh modes        # R2
#     bash code/run_blocks.sh dims         # R3
#     bash code/run_blocks.sh all          # liên tiếp các khối (vẫn 1 job/lần)
#     bash code/run_blocks.sh status       # khối nào xong / đang chạy / thiếu
#     bash code/run_blocks.sh csv          # in bảng kết quả ra results/csv/*.csv
#
#   Chạy NGẦM (để thoát kết nối mà job vẫn chạy):
#     nohup bash code/run_blocks.sh main > logs/block_main.out 2>&1 &
#     tail -f logs/block_main.out
# ============================================================================
set -u
cd "$(dirname "$0")/.."
mkdir -p logs

COMMON="--n_shards 8 --emb_dim 64 --sig_dim 64 --epochs 50 --retrain_epochs 50 \
--lr 0.05 --batch_size 512 --unlearn_ratio 0.01 --seeds 0,1,2,3,4"
MAIN_ARMS="PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,GLOBAL"
REC_ARMS="PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,RECERASER,GLOBAL"

banner () {
  echo ""
  echo "=============================================================="
  echo "$1   ($(date +%H:%M:%S))"
  echo "=============================================================="
}

run () {
  local name="$1"; shift
  if [ -f "logs/${name}.out" ] && grep -q "CONSOLE SUMMARY" "logs/${name}.out" 2>/dev/null; then
    echo "[skip] $name  (log đã có CONSOLE SUMMARY)"
    return 0
  fi
  banner "START $name"
  local t0=$SECONDS
  python "$@" 2>&1 | tee "logs/${name}.out"
  local rc=${PIPESTATUS[0]}
  echo "--------------------------------------------------------------"
  echo "DONE $name  (rc=$rc, $((SECONDS-t0))s)  -> logs/${name}.out"
  echo "--------------------------------------------------------------"
  return $rc
}

# ------------------------------------------------------------------ preflight
# CUDA là bắt buộc: số đo thời gian chỉ so sánh được trong cùng một máy.
# Thiếu embedding cho RECERASER thì cũng dừng, thay vì để job crash giữa chừng.
preflight () {
  local need_emb="${1:-0}"
  local rc=0
  PREFLIGHT_LOG="logs/_preflight.log"
  if ! python - > "$PREFLIGHT_LOG" 2>&1 <<'PY'
import torch, sys
print(f"torch {torch.__version__}  cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}  n_gpu={torch.cuda.device_count()}")
for m in ("numpy", "scipy", "sklearn"):
    try:
        __import__(m); print(f"  {m}: ok")
    except Exception as e:
        print(f"  {m}: THIEU -> {e}")
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    sed 's/^/  /' "$PREFLIGHT_LOG"
    if [ "${RECUNLEARN_ALLOW_CPU:-0}" != "1" ]; then
      echo ""
      echo "!! DUNG: CUDA chua bat -> ket qua se do tren CPU, khong so sanh duoc."
      echo "   Fix: pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128"
      echo "   (dat RECUNLEARN_ALLOW_CPU=1 de chi smoke-test)"
      exit 1
    fi
    echo "  -> RECUNLEARN_ALLOW_CPU=1: van chay tren CPU."
  else
    sed 's/^/  /' "$PREFLIGHT_LOG"
  fi

  if [ "$need_emb" = "1" ]; then
    for DS in ml-1m yelp2018; do
      NPZ="data/${DS}/wmf_embeddings.npz"
      if [ -f "$NPZ" ]; then
        echo "  [ok] ${DS}: $NPZ"
      else
        echo "  [THIEU] ${DS}: $NPZ"; rc=1
      fi
    done
    if [ "$rc" = "1" ]; then
      echo "!! Thiếu wmf_embeddings.npz -> khong chay."
      exit 1
    fi
  fi
  echo ""
}

# ------------------------------------------------------------------ torch
block_torch () {
  banner "Torch / CUDA + embedding check"
  preflight 1
  echo "OK - co the chay cac khoi."
}

# ------------------------------------------------------------------ main (R1)
block_main () {
  preflight 0
  run main_ml1m  python code/exp_partition_quality.py --dataset ml-1m   \
    --unlearn_mode random $COMMON --arms $MAIN_ARMS
  run main_yelp  python code/exp_partition_quality.py --dataset yelp2018 \
    --unlearn_mode random $COMMON --arms $MAIN_ARMS
}

# ------------------------------------------------------------------ single (N2)
block_single () {
  preflight 1
  run single_request_ml1m python code/exp_single_request.py --dataset ml-1m \
    --n_shards 8 --emb_dim 64 --sig_dim 64 --epochs 50 --retrain_epochs 50 \
    --lr 0.05 --batch_size 512 --n_users 20 --seeds 0,1,2,3,4 \
    --arms $REC_ARMS
}

# ------------------------------------------------------------------ receraser (N1)
block_receraser () {
  preflight 1
  run receraser_ml1m      python code/exp_partition_quality.py --dataset ml-1m    \
    --unlearn_mode random $COMMON --arms RECERASER
  run receraser_yelp      python code/exp_partition_quality.py --dataset yelp2018 \
    --unlearn_mode random $COMMON --arms RECERASER
  run receraser_ml1m_most python code/exp_partition_quality.py --dataset ml-1m    \
    --unlearn_mode most   $COMMON --arms RECERASER
  run receraser_ml1m_fewest python code/exp_partition_quality.py --dataset ml-1m  \
    --unlearn_mode fewest $COMMON --arms RECERASER
}

# ------------------------------------------------------------------ modes (R2)
block_modes () {
  preflight 0
  run mode_most_ml1m   python code/exp_partition_quality.py --dataset ml-1m \
    --unlearn_mode most   $COMMON --arms $MAIN_ARMS
  run mode_fewest_ml1m python code/exp_partition_quality.py --dataset ml-1m \
    --unlearn_mode fewest $COMMON --arms $MAIN_ARMS
}

# ------------------------------------------------------------------ dims (R3)
block_dims () {
  preflight 0
  for D in 32 128 256; do
    run dim${D}_ml1m python code/exp_partition_quality.py --dataset ml-1m \
      --unlearn_mode random --n_shards 8 --emb_dim $D --sig_dim 64 \
      --epochs 50 --retrain_epochs 50 --lr 0.05 --batch_size 512 \
      --unlearn_ratio 0.01 --seeds 0,1,2,3,4 --arms $MAIN_ARMS
  done
}

# ------------------------------------------------------------------ status
ALL_LOGS="main_ml1m main_yelp single_request_ml1m receraser_ml1m receraser_yelp
receraser_ml1m_most receraser_ml1m_fewest mode_most_ml1m mode_fewest_ml1m
dim32_ml1m dim128_ml1m dim256_ml1m"

block_status () {
  banner "STATUS"
  local done=0 run=0 miss=0
  for n in $ALL_LOGS; do
    f="logs/${n}.out"
    if [ ! -f "$f" ]; then
      printf '  %-24s MISSING\n' "$n"; miss=$((miss+1))
    elif grep -q "CONSOLE SUMMARY" "$f"; then
      printf '  %-24s DONE\n' "$n"; done=$((done+1))
    elif grep -q "Traceback" "$f"; then
      printf '  %-24s CRASHED  (xem: tail -30 %s)\n' "$n" "$f"; run=$((run+1))
    else
      printf '  %-24s RUNNING  | %s\n' "$n" "$(grep -E '^--- ARM' "$f" | tail -1)"
      run=$((run+1))
    fi
  done
  echo ""
  echo "  done=$done  running/crashed=$run  missing=$miss  (tong 12)"
  echo ""
  ps -eo pid,etime,cmd | grep -E "run_blocks|exp_partition_quality|exp_single_request" \
    | grep -v grep || echo "  (khong co job nao dang chay)"
}

# ------------------------------------------------------------------ csv
# In bảng kết quả ra results/csv/. Chỉ đọc log, không train gì.
#   <khoi>_raw.csv    1 dòng / (log, arm, seed)
#   <khoi>_mean.csv   1 dòng / (log, arm): mean, std, delta vs RANDOM_PERM, p
#   <khoi>_tables.txt bảng đầy đủ dạng text (utility, partition, instability, cost, paired)
#   all_*.csv         gộp mọi log partition-quality
#   single_request.csv thời gian 1 request xóa, mỗi arm
# Log chưa xong vẫn được đọc phần đã chạy; cột n_seeds cho biết đủ 5 seed chưa.
csv_group () {   # csv_group <ten> <log1> [log2 ...]
  local name="$1"; shift
  local have=()
  for n in "$@"; do [ -f "logs/${n}.out" ] && have+=("logs/${n}.out"); done
  if [ ${#have[@]} -eq 0 ]; then
    echo "  [skip] $name - chua co log nao"
    return
  fi
  python code/summarize_partition_logs.py "${have[@]}" --csv "results/csv/$name" \
    > "results/csv/${name}_tables.txt" 2>&1
  grep -E "^\[written\]|^\[skip\]|Traceback|Error" "results/csv/${name}_tables.txt" | sed 's/^/  /'
}

block_csv () {
  banner "EXPORT CSV -> results/csv/"
  mkdir -p results/csv

  csv_group main      main_ml1m main_yelp
  csv_group receraser receraser_ml1m receraser_yelp receraser_ml1m_most receraser_ml1m_fewest
  csv_group modes     mode_most_ml1m mode_fewest_ml1m
  csv_group dims      dim32_ml1m dim128_ml1m dim256_ml1m
  csv_group all       main_ml1m main_yelp receraser_ml1m receraser_yelp \
                      receraser_ml1m_most receraser_ml1m_fewest \
                      mode_most_ml1m mode_fewest_ml1m dim32_ml1m dim128_ml1m dim256_ml1m

  if [ -f logs/single_request_ml1m.out ]; then
    python code/export_single_request_csv.py logs/single_request_ml1m.out \
      --out results/csv/single_request.csv | sed 's/^/  /'
  else
    echo "  [skip] single_request - chua co log"
  fi

  echo ""
  ls -la results/csv
}

case "${1:-main}" in
  torch)     block_torch ;;
  main)      block_main ;;
  single)    block_single ;;
  receraser) block_receraser ;;
  modes)     block_modes ;;
  dims)      block_dims ;;
  all)       block_main; block_single; block_receraser; block_modes; block_dims; block_csv ;;
  status)    block_status ;;
  csv)       block_csv ;;
  *) echo "dùng: bash code/run_blocks.sh <torch|main|single|receraser|modes|dims|all|status|csv>"; exit 1 ;;
esac
