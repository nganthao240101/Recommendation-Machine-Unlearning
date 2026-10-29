#!/usr/bin/env bash
# =============================================================================
# CHẠY TOÀN BỘ THÍ NGHIỆM TRÊN YELP2018 (gợi ý kiểm tra trend so ml-1m)
#
# Yelp2018 LỚN hơn ml-1m ~5-10x (31668 users, 38047 items).
# => Chạy theo LÔ (batches = 2 job nặng cùng lúc) để tránh OOM.
#    Mặc định KHÔNG chạy tất cả song song.
#
# Cách chạy:
#   git pull
#   bash code/run_yelp.sh
#
# Có thể override bằng biến môi trường, ví dụ:
#   BATCHES=1 bash code/run_yelp.sh        # chạy tuần tự (an toàn nhất RAM)
#   EPOCHS=100 bash code/run_yelp.sh      # tăng epochs
# =============================================================================

set -u

cd "$(dirname "$0")/.."        # về thư mục gốc repo
CODE=code
LOGS=logs
mkdir -p "$LOGS"

# ---- Tham số điều chỉnh được (override bằng env) ---------------------------
DS="${DS:-yelp2018}"
EMB="${EMB:-64}"
EP="${EP:-50}"                 # max_epochs (yelp sparse => có thể tăng lên 100)
SIG="${SIG:-64}"               # signature_dim (Ours)
K="${K:-8}"                    # n_shards
RATIO="${RATIO:-0.01}"        # unlearn_ratio
MODE="${MODE:-most}"          # unlearn_mode
BATCHES="${BATCHES:-2}"       # số job nặng chạy song song
WMF_EPOCHS="${WMF_EPOCHS:-100}"

# Cờ cho Rec v2 (partition_type: 1=InP, 2=UBP, 3=Random; max_epochs_wmf; max_epochs_local; use_attention)
REC_LOCAL="${REC_LOCAL:-50}"
REC_ATT="${REC_ATT:-1}"

echo "###############################################################"
echo " RUN YELP SUITE | dataset=$DS emb=$EMB ep=$EP K=$K ratio=$RATIO mode=$MODE"
echo " batches=$BATCHES  (giới hạn job nặng chạy song song)"
echo "###############################################################"

# ---- Kiểm tra data ---------------------------------------------------------
if [ ! -d "data/$DS" ]; then
  echo "[LỖI] Không có thư mục data/$DS. Kiểm tra data/yelp2018/ trước."
  exit 1
fi
echo "[data] data/$DS hợp lệ: $(ls data/$DS | tr '\n' ' ')"

# Chờ cả 2 job (chạy nền rồi wait) => giữ BATCHES job song song
run_batch() {  # mỗi dòng = 1 job; script chạy từng lô BATCHES job rồi mới tiếp
  # (được dùng bên dưới qua while + fifo đơn giản)
  :
}

# ---- (0) WMF pretrain: BẮT BUỘC trước Rec-UBP & C6 (tốn ~1-2h trên yelp) ----
if [ ! -f "data/$DS/wmf_embeddings.npz" ]; then
  echo "==== [0/7] Pretrain WMF ($WMF_EPOCHS ep) - cần cho Rec UBP + C6 ===="
  python "$CODE/pretrain_wmf.py" --dataset "$DS" --emb_dim "$EMB" \
    --max_epochs "$WMF_EPOCHS" > "$LOGS/0_wmf.log" 2>&1
  echo "    -> WMF xong (logs/0_wmf.log)"
else
  echo "==== [0/7] WMF đã có, bỏ qua ===="
fi

# Chạy lô: gửi BATCHES job vào hàng đợi, chạy tuần tự theo lô
run_loops() {
  # $@ = danh sách các "command" (gọi bằng eval từng cái)
  local -a cmds=("$@")
  local running=0 i=0
  # đơn giản: chạy từng lô BATCHES job song song, wait trước lô kế
  while (( i < ${#cmds[@]} )); do
    local start=$i
    local end=$(( i + BATCHES ))
    (( end > ${#cmds[@]} )) && end=${#cmds[@]}
    for (( j=start; j<end; j++ )); do
      eval "${cmds[$j]} &"
    done
    echo "    ...chờ ${end}-$start job hoàn thành (lô bắt đầu job #$start)..."
    wait
    i=$end
  done
}

# =============================================================================
# (1) 4 PHƯƠNG PHÁP (K=8, RATIO, MODE)
# =============================================================================
echo "==== [1/7] 4 phương pháp, K=$K ===="
run_loops \
  "python $CODE/method_4_ours.py --dataset $DS --emb_dim $EMB --n_shards $K --max_epochs $EP --signature_dim $SIG --unlearn_ratio $RATIO --unlearn_mode $MODE > $LOGS/1_ours.log 2>&1" \
  "python $CODE/method_2_sisa.py --dataset $DS --emb_dim $EMB --n_shards $K --max_epochs $EP --unlearn_ratio $RATIO --unlearn_mode $MODE > $LOGS/1_sisa.log 2>&1" \
  "python $CODE/method_3_receraser_v2.py --dataset $DS --emb_dim $EMB --n_shards $K --partition_type 3 --max_epochs_wmf 0 --max_epochs_local $REC_LOCAL --use_attention $REC_ATT --unlearn_ratio $RATIO --unlearn_mode $MODE > $LOGS/1_rec_random.log 2>&1" \
  "python $CODE/method_3_receraser_v2.py --dataset $DS --emb_dim $EMB --n_shards $K --partition_type 2 --max_epochs_wmf $WMF_EPOCHS --max_epochs_local $REC_LOCAL --use_attention $REC_ATT --unlearn_ratio $RATIO --unlearn_mode $MODE > $LOGS/1_rec_ubp.log 2>&1"

# =============================================================================
# (2) N_SHARDS (Ours vs SISA ở K=4 và K=16)
# =============================================================================
echo "==== [2/7] n_shards = 4,16 (Ours vs SISA) ===="
run_loops \
  "python $CODE/method_4_ours.py --dataset $DS --emb_dim $EMB --n_shards 4 --max_epochs $EP --signature_dim $SIG --unlearn_ratio $RATIO --unlearn_mode $MODE > $LOGS/2_ours_k4.log 2>&1" \
  "python $CODE/method_2_sisa.py --dataset $DS --emb_dim $EMB --n_shards 4 --max_epochs $EP --unlearn_ratio $RATIO --unlearn_mode $MODE > $LOGS/2_sisa_k4.log 2>&1" \
  "python $CODE/method_4_ours.py --dataset $DS --emb_dim $EMB --n_shards 16 --max_epochs $EP --signature_dim $SIG --unlearn_ratio $RATIO --unlearn_mode $MODE > $LOGS/2_ours_k16.log 2>&1" \
  "python $CODE/method_2_sisa.py --dataset $DS --emb_dim $EMB --n_shards 16 --max_epochs $EP --unlearn_ratio $RATIO --unlearn_mode $MODE > $LOGS/2_sisa_k16.log 2>&1"

# =============================================================================
# (3) TỶ LỆ UNLEARN (Ours/SISA/Rec UBP ở 0.1% & 2%)
# =============================================================================
echo "==== [3/7] unlearn_ratio = 0.001,0.02 ===="
run_loops \
  "python $CODE/method_4_ours.py --dataset $DS --emb_dim $EMB --n_shards $K --max_epochs $EP --signature_dim $SIG --unlearn_ratio 0.001 --unlearn_mode $MODE > $LOGS/3_ours_001.log 2>&1" \
  "python $CODE/method_4_ours.py --dataset $DS --emb_dim $EMB --n_shards $K --max_epochs $EP --signature_dim $SIG --unlearn_ratio 0.02 --unlearn_mode $MODE > $LOGS/3_ours_02.log 2>&1" \
  "python $CODE/method_2_sisa.py --dataset $DS --emb_dim $EMB --n_shards $K --max_epochs $EP --unlearn_ratio 0.001 --unlearn_mode $MODE > $LOGS/3_sisa_001.log 2>&1" \
  "python $CODE/method_2_sisa.py --dataset $DS --emb_dim $EMB --n_shards $K --max_epochs $EP --unlearn_ratio 0.02 --unlearn_mode $MODE > $LOGS/3_sisa_02.log 2>&1" \
  "python $CODE/method_3_receraser_v2.py --dataset $DS --emb_dim $EMB --n_shards $K --partition_type 2 --max_epochs_wmf $WMF_EPOCHS --max_epochs_local $REC_LOCAL --use_attention $REC_ATT --unlearn_ratio 0.001 --unlearn_mode $MODE > $LOGS/3_rec_ubp_001.log 2>&1" \
  "python $CODE/method_3_receraser_v2.py --dataset $DS --emb_dim $EMB --n_shards $K --partition_type 2 --max_epochs_wmf $WMF_EPOCHS --max_epochs_local $REC_LOCAL --use_attention $REC_ATT --unlearn_ratio 0.02 --unlearn_mode $MODE > $LOGS/3_rec_ubp_02.log 2>&1"

# =============================================================================
# (4) UNLEARN MODE (RANDOM / FEWEST)
# =============================================================================
echo "==== [4/7] unlearn_mode = random, fewest ===="
run_loops \
  "python $CODE/method_4_ours.py --dataset $DS --emb_dim $EMB --n_shards $K --max_epochs $EP --signature_dim $SIG --unlearn_ratio $RATIO --unlearn_mode random > $LOGS/4_ours_rand.log 2>&1" \
  "python $CODE/method_4_ours.py --dataset $DS --emb_dim $EMB --n_shards $K --max_epochs $EP --signature_dim $SIG --unlearn_ratio $RATIO --unlearn_mode fewest > $LOGS/4_ours_fewest.log 2>&1" \
  "python $CODE/method_2_sisa.py --dataset $DS --emb_dim $EMB --n_shards $K --max_epochs $EP --unlearn_ratio $RATIO --unlearn_mode random > $LOGS/4_sisa_rand.log 2>&1"

# =============================================================================
# (5) SIGNATURE_DIM (32 / 128)
# =============================================================================
echo "==== [5/7] signature_dim = 32,128 ===="
run_loops \
  "python $CODE/method_4_ours.py --dataset $DS --emb_dim $EMB --n_shards $K --max_epochs $EP --signature_dim 32 --unlearn_ratio $RATIO --unlearn_mode $MODE > $LOGS/5_ours_sig32.log 2>&1" \
  "python $CODE/method_4_ours.py --dataset $DS --emb_dim $EMB --n_shards $K --max_epochs $EP --signature_dim 128 --unlearn_ratio $RATIO --unlearn_mode $MODE > $LOGS/5_ours_sig128.log 2>&1"

# =============================================================================
# (6) FAIR 4-METHOD (C1/C2/C3/C4/C6) + K=4
# =============================================================================
echo "==== [6/7] fair compare cùng capacity (K=8 và K=4) ===="
run_loops \
  "python $CODE/benchmark_fair_compare.py --dataset $DS --emb_dim $EMB --n_shards 8 --max_epochs $EP --unlearn_ratio $RATIO --unlearn_mode random --skip_rec > $LOGS/6_fair_k8.log 2>&1" \
  "python $CODE/benchmark_fair_compare.py --dataset $DS --emb_dim $EMB --n_shards 4 --max_epochs $EP --unlearn_ratio $RATIO --unlearn_mode random --skip_rec > $LOGS/6_fair_k4.log 2>&1"

# =============================================================================
# (7) DELETION-STABILITY (không cần train, nhanh)
# =============================================================================
echo "==== [7/7] deletion-stability + 2 figure ===="
run_loops \
  "python $CODE/measure_stability.py --dataset $DS --emb_dim $EMB --n_shards $K --signature_dim $SIG --unlearn_ratio $RATIO --unlearn_mode $MODE > $LOGS/7_stab_most.log 2>&1" \
  "python $CODE/measure_stability.py --dataset $DS --emb_dim $EMB --n_shards $K --signature_dim $SIG --unlearn_ratio $RATIO --unlearn_mode random > $LOGS/7_stab_rand.log 2>&1" \
  "python $CODE/visualize_stability.py --dataset $DS --n_shards $K --signature_dim $SIG --unlearn_mode $MODE --ratios 0.001,0.005,0.01,0.02,0.05 --out figures/$DS > $LOGS/7_vis.log 2>&1"

# =============================================================================
# TÓM TẮT (in ra bảng nhanh để so với ml-1m)
# =============================================================================
echo ""
echo "###############################################################"
echo " XONG! Tóm tắt nhanh (R@10 / NDCG@10 AFTER-RETAINED):"
echo "###############################################################"
for f in "$LOGS"/1_ours "$LOGS"/1_sisa "$LOGS"/1_rec_random "$LOGS"/1_rec_ubp \
         "$LOGS"/2_ours_k4 "$LOGS"/2_sisa_k4 "$LOGS"/2_ours_k16 "$LOGS"/2_sisa_k16; do
  echo "--- ${f##*/}"
  grep -E "Before.*FULL|After.*RETAINED|Unlearn time|Assignment.*min|Affected" "$f.log" 2>/dev/null
done
echo ""
echo " Stablity (naive vs effective):"
grep -E "Method.*sim|inst%|PHÂN TÍCH|Ours|SISA|Rec" "$LOGS"/7_stab_most.log 2>/dev/null | head -20
echo ""
echo " Fair compare (K=8):"
grep -E "C1|C3|C4|C6|ENSEMBLE|SHARDING|so Full|Rec vs SISA|Ours vs Rec" "$LOGS"/6_fair_k8.log 2>/dev/null
echo ""
echo " Xem chi tiết từng log:  ls -la $LOGS/*.log"
