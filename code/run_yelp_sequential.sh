#!/usr/bin/env bash
###############################################################################
# RUN YELP SEQUENTIAL (1 job 1 lần - KHÔNG tranh CPU như run song song)
#
# Dùng khi: đã kill hết job, muốn chạy LẠI toàn bộ thí nghiệm yelp một cách
# an toàn (tuần tự, không OOM, không đua tranh GPU).
#
# Cách chạy (1 dòng, chạy nền):
#   nohup bash code/run_yelp_sequential.sh > logs/yelp_seq.log 2>&1 &
#   tail -f logs/yelp_seq.log
#
# Nguyên tắc:
#   - MIỄN ml-1m (đã xong: EX1 + EX5 + EX6) - chỉ chạy YELP.
#   - Rec UBP yelp dùng MEAN (use_attention 0) -> tránh NaN của attention.
#   - Chạy TUẦN TỰ -> 1 job lúc 1 thời điểm -> không tranh CPU/GPU.
###############################################################################
set -u

# ==== cấu hình (đổi ở đây nếu cần) ====
DS=yelp2018
EMB=64
EP=50            # epochs local (Ours/SISA/Rec)
WMF=100          # epochs WMF pretrain (chỉ cho Rec UBP)
RATIO=0.01
MODE=most
LOGS=logs
CODE=code

# màu (nếu stdout)
R=$'\e[31m'; G=$'\e[32m'; Y=$'\e[33m'; N=$'\e[0m'
say(){ echo; echo "${G}================== $* ${N}"; }
run(){ echo -e "${Y}\$ $*${N}"; }

cd "$(dirname "$0")/.."   # về thư mục gốc repo (code/..)

# ============================================================================
# STEP 0: KILL HẾT job cũ (để chạy lại từ đầu, sạch)
# ============================================================================
say "STEP 0: kill hết tiến trình cũ"
pkill -9 -f "code/method_1_full_retrain.py|code/method_2_sisa.py|code/method_3_receraser_v2.py|code/method_4_ours.py|code/benchmark_fair_compare.py|code/measure_stability.py|code/visualize_stability.py|code/pretrain_wmf.py" 2>/dev/null
sleep 3
echo "  Số process còn lại (phải = 0 hoặc chỉ còn shell này):"
ps aux | grep -E "method_|benchmark_fair|measure_stability" | grep -v grep | grep -v run_yelp_sequential || echo "  (rỗng - sạch)"

# ============================================================================
# STEP 1: WMF embeddings (yelp) - cần cho Rec UBP + fair C6 + stability
#   - convert .pk -> .npz nếu chưa có (skip nếu đã có)
# ============================================================================
say "STEP 1: WMF embeddings (yelp2018)"
if [ ! -f "data/$DS/wmf_embeddings.npz" ]; then
  run "python $CODE/convert_wmf_pk.py --dataset $DS"
  python "$CODE/convert_wmf_pk.py" --dataset "$DS" || { echo "  convert fail"; }
fi
run "ls -la data/$DS/wmf_embeddings.npz"
ls -la "data/$DS/wmf_embeddings.npz"

# ============================================================================
# STEP 2: 4 PHƯƠNG PHÁP (K=8, 1%, MOST) - chạy TUẦN TỰ
# ============================================================================
say "STEP 2/6: Full Retrain (trần oracle)"
run "python $CODE/method_1_full_retrain.py ... (yelp, 100ep, no early-stop)"
python "$CODE/method_1_full_retrain.py" --dataset "$DS" --emb_dim "$EMB" \
  --max_epochs 100 --early_stopping False --unlearn_ratio "$RATIO" --unlearn_mode "$MODE" \
  > "$LOGS/y_ex1_full.log" 2>&1

say "STEP 3/6: Ours (deletion-stable)"
run "python $CODE/method_4_ours.py ... (yelp)"
python "$CODE/method_4_ours.py" --dataset "$DS" --emb_dim "$EMB" --n_shards 8 \
  --max_epochs "$EP" --signature_dim "$EMB" --unlearn_ratio "$RATIO" --unlearn_mode "$MODE" \
  > "$LOGS/y_ex1_ours.log" 2>&1

say "STEP 4/6: SISA (modulo)"
run "python $CODE/method_2_sisa.py ... (yelp)"
python "$CODE/method_2_sisa.py" --dataset "$DS" --emb_dim "$EMB" --n_shards 8 \
  --max_epochs "$EP" --unlearn_ratio "$RATIO" --unlearn_mode "$MODE" \
  > "$LOGS/y_ex1_sisa.log" 2>&1

say "STEP 5/6: Rec UBP (mean, tránh NaN)"
run "python $CODE/method_3_receraser_v2.py ... (yelp, use_attention 0)"
python "$CODE/method_3_receraser_v2.py" --dataset "$DS" --emb_dim "$EMB" --n_shards 8 \
  --partition_type 2 --max_epochs_wmf "$WMF" --max_epochs_local "$EP" --use_attention 0 \
  --unlearn_ratio "$RATIO" --unlearn_mode "$MODE" \
  > "$LOGS/y_ex1_rec_ubp.log" 2>&1

# ============================================================================
# STEP 6: STABILITY + FAIR (đóng góp chính + cùng capacity)
# ============================================================================
say "STEP 6/6: Deletion-stability + Fair compare"
run "python $CODE/measure_stability.py ... (yelp)"
python "$CODE/measure_stability.py" --dataset "$DS" --n_shards 8 --signature_dim "$EMB" \
  --unlearn_ratio "$RATIO" --unlearn_mode "$MODE" > "$LOGS/y_ex6_stab.log" 2>&1

run "python $CODE/benchmark_fair_compare.py ... (yelp, skip_rec)"
python "$CODE/benchmark_fair_compare.py" --dataset "$DS" --emb_dim "$EMB" --n_shards 8 \
  --max_epochs "$EP" --unlearn_ratio "$RATIO" --unlearn_mode random --skip_rec \
  > "$LOGS/y_ex5_fair.log" 2>&1

# ============================================================================
# TÓM TẮT (điền số vào bảng so ml-1m)
# ============================================================================
say "XONG - tóm tắt yelp"
echo "  4 method (After R@10):"
grep -hE "After.*RETAINED" "$LOGS"/y_ex1_full.log "$LOGS"/y_ex1_sisa.log \
  "$LOGS"/y_ex1_ours.log "$LOGS"/y_ex1_rec_ubp.log 2>/dev/null || echo "    (chưa đủ)"
echo "  Fair C1-C6:"
grep -hE "C1 |C3 |C4 |C6 |ENSEMBLE|SHARDING" "$LOGS"/y_ex5_fair.log 2>/dev/null || echo "    (chưa đủ)"
echo "  Stability (inst%):"
grep -hE "inst%|Ours \(signature\)|Rec UBP" "$LOGS"/y_ex6_stab.log 2>/dev/null || echo "    (chưa đủ)"
echo
echo "${G}HOÀN TẤT. Xem log: $LOGS/y_ex1_*.log $LOGS/y_ex6_stab.log $LOGS/y_ex5_fair.log${N}"
