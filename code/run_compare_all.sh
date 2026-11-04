#!/usr/bin/env bash
# ============================================================================
# run_compare_all.sh - CHẠY ĐỦ SỐ so sánh PHƯƠNG PHÁP × PARTITION × MODEL(WMF/BPR)
# ----------------------------------------------------------------------------
# Nguyên tắc: MỌI run có log riêng; NẾU log đã ra "After (RETAINED)" -> BỎ QUA.
# Chọn khối:  BLOCKS="1 2 3 4" (default tất cả).  --grep để in mọi số.
#
#   1) METHODS   : Full / Ours(sig) / SISA / Rec(UBP WMF-in, Rec thật)  - 2 ds
#   2) OURS×PART : Ours × {signature,Random,UBP,InP}(frozen) × {wmf,bprfull}
#   3) REC×PART  : Rec  × {InP,UBP,Random} × {wmf,bprfull} (no-WMF, mean, lr 0.05)
#   4) EX6       : stability (flip%) 4 Ours-partition (không train, ~5p/ds)
#   comb        : = KHỐI 1 + KHỐI 3 gộp 1 chỗ (4 method + Rec×3-partition×2-model,
#                 CẢ 2 dataset, CẢ 2 model) -> "1 lệnh so tất cả"
#
# Cách dùng:
#   bash code/run_compare_all.sh                     # tất cả (1 2 3 4)
#   BLOCKS="comb" bash code/run_compare_all.sh       # 4 method + Rec×3partition×2model (đúng ý "khối1+3")
#   BLOCKS="comb 4" bash code/run_compare_all.sh     # + thêm EX6
#   BLOCKS="2 3" bash code/run_compare_all.sh        # chỉ so partition × model
#   bash code/run_compare_all.sh --grep              # in mọi số
# ============================================================================
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
K=8; E=50; D=64; R=0.01; M=most; EL=60; EWMF=100
export CUDA_VISIBLE_DEVICES=0
BLOCKS="${BLOCKS:-1 2 3 4}"

log(){ echo "=== $(date +%H:%M:%S) $1 ==="; }

# Run 1 lệnh, ghi log; SKIP nếu log đã có "After (RETAINED)"
run(){  # $1=logfile  $2=label  $...=command
  local logfile="$1"; shift; local label="$1"; shift
  if [ -f "$logfile" ] && grep -q "After (RETAINED" "$logfile" 2>/dev/null; then
    echo "  [SKIP] $label (log đã có kết quả: $logfile)"
    return 0
  fi
  log "$label"
  "$@" > "$logfile" 2>&1
  echo "  [done] $label -> $logfile (exit=$?)"
}

# ============================================================================
# KHỐI 1 - 4 PHƯƠNG PHÁP (2 dataset, lr=0.05; Rec-UBP WMF-in = "Rec thật")
# ============================================================================
if echo " $BLOCKS " | grep -q " 1 "; then
for ds in ml-1m yelp2018; do
  run "logs/cmp1_${ds}_full.log"  "M1 $ds FULL" \
    python code/method_1_full_retrain.py --dataset $ds --emb_dim 64 --max_epochs $E \
      --unlearn_ratio $R --unlearn_mode $M
  run "logs/cmp1_${ds}_ours.log" "M1 $ds OURS(sig)" \
    python code/method_4_ours.py --dataset $ds --emb_dim 64 --n_shards $K \
      --signature_dim $D --max_epochs $E --unlearn_ratio $R --unlearn_mode $M
  run "logs/cmp1_${ds}_sisa.log" "M1 $ds SISA" \
    python code/method_2_sisa.py --dataset $ds --emb_dim 64 --n_shards $K \
      --max_epochs $E --unlearn_ratio $R --unlearn_mode $M
  run "logs/cmp1_${ds}_rec.log"  "M1 $ds REC-UBP (WMF-in, attn) [Rec thật]" \
    python code/method_3_receraser_v2.py --dataset $ds --emb_dim 64 --n_shards $K \
      --partition_type 2 --max_epochs_wmf $EWMF --max_epochs_local $EL \
      --use_attention 1 --lr_finetune 0.05 --unlearn_ratio $R --unlearn_mode $M
done
fi

# ============================================================================
# KHỐI 2 - Ours × 4 PARTITION × {wmf,bprfull} (frozen -> stable 0%)
#   signature/randomfrozen: KHÔNG cần embedding (hash/modulo) -> chỉ chạy 1 lần
#   ubpfrozen/inpfrozen: chạy CẢ 2 nguồn embedding (wmf & bprfull)
# ============================================================================
if echo " $BLOCKS " | grep -q " 2 "; then
for ds in ml-1m yelp2018; do
  for part in signature randomfrozen; do
    run "logs/cmp2_${ds}_${part}.log" "M2 $ds Ours-$part (no-emb)" \
      python code/method_4_ours.py --dataset $ds --emb_dim 64 --n_shards $K \
        --signature_dim $D --max_epochs $E --unlearn_ratio $R --unlearn_mode $M \
        --partition $part --max_epochs_wmf $EWMF
  done
  for part in ubpfrozen inpfrozen; do
    for emb in wmf bprfull; do
      run "logs/cmp2_${ds}_${part}_emb${emb}.log" "M2 $ds Ours-$part emb=$emb" \
        python code/method_4_ours.py --dataset $ds --emb_dim 64 --n_shards $K \
          --signature_dim $D --max_epochs $E --unlearn_ratio $R --unlearn_mode $M \
          --partition $part --max_epochs_wmf $EWMF --embed_source $emb
    done
  done
done
fi

# ============================================================================
# KHỐI 3 - Rec × 3 PARTITION × {wmf,bprfull} (no-WMF, MEAN, lr 0.05)
#   InP=pt1, UBP=pt2, Random=pt3. Random KHÔNG cần embedding (cả 2 emb cùng modulo)
#   InP/UBP: chạy CẢ 2 nguồn embedding (wmf & bprfull)
# ============================================================================
if echo " $BLOCKS " | grep -q " 3 "; then
for ds in ml-1m yelp2018; do
  # Random (pt3): 1 lần (modulo, không dùng embedding)
  run "logs/cmp3_${ds}_pt3_random.log" "M3 $ds Rec-Random (no-emb)" \
    python code/method_3_receraser_v2.py --dataset $ds --emb_dim 64 --n_shards $K \
      --partition_type 3 --max_epochs_wmf 0 --max_epochs_local $EL \
      --use_attention 0 --no_wmf_in_model --lr_finetune 0.05 \
      --unlearn_ratio $R --unlearn_mode $M
  # InP(pt1) / UBP(pt2): chạy cả 2 model
  for pt in 1 2; do
    for emb in wmf bprfull; do
      run "logs/cmp3_${ds}_pt${pt}_emb${emb}.log" "M3 $ds Rec-pt${pt} emb=$emb" \
        python code/method_3_receraser_v2.py --dataset $ds --emb_dim 64 --n_shards $K \
          --partition_type $pt --max_epochs_wmf $EWMF --max_epochs_local $EL \
          --use_attention 0 --no_wmf_in_model --lr_finetune 0.05 \
          --embed_source $emb --unlearn_ratio $R --unlearn_mode $M
    done
  done
done
fi

# ============================================================================
# KHỐI "comb" = KHỐI 1 (4 method) + KHỐI 3 (Rec x 3 partition x 2 model),
# CẢ 2 DATASET, CẢ 2 MODEL (wmf/bprfull). Chạy: BLOCKS="comb"
#   - 4 method: Full / Ours(sig) / SISA / Rec-UBP (WMF-in, "Rec thật")
#   - Rec x 3 partition (InP/UBP/Random) x 2 model (wmf & bprfull)
# ============================================================================
if echo " $BLOCKS " | grep -q " comb "; then
for ds in ml-1m yelp2018; do
  # ---- 4 PHƯƠNG PHÁP (khối 1) ----
  run "logs/cmp1_${ds}_full.log"  "comb $ds FULL" \
    python code/method_1_full_retrain.py --dataset $ds --emb_dim 64 --max_epochs $E \
      --unlearn_ratio $R --unlearn_mode $M
  run "logs/cmp1_${ds}_ours.log" "comb $ds OURS(sig)" \
    python code/method_4_ours.py --dataset $ds --emb_dim 64 --n_shards $K \
      --signature_dim $D --max_epochs $E --unlearn_ratio $R --unlearn_mode $M
  run "logs/cmp1_${ds}_sisa.log" "comb $ds SISA" \
    python code/method_2_sisa.py --dataset $ds --emb_dim 64 --n_shards $K \
      --max_epochs $E --unlearn_ratio $R --unlearn_mode $M
  run "logs/cmp1_${ds}_rec.log"  "comb $ds REC-UBP (WMF-in, attn) [Rec thật]" \
    python code/method_3_receraser_v2.py --dataset $ds --emb_dim 64 --n_shards $K \
      --partition_type 2 --max_epochs_wmf $EWMF --max_epochs_local $EL \
      --use_attention 1 --lr_finetune 0.05 --unlearn_ratio $R --unlearn_mode $M
  # ---- Rec x 3 PARTITION x 2 MODEL (khối 3) ----
  run "logs/cmp3_${ds}_pt3_random.log" "comb $ds Rec-Random (no-emb)" \
    python code/method_3_receraser_v2.py --dataset $ds --emb_dim 64 --n_shards $K \
      --partition_type 3 --max_epochs_wmf 0 --max_epochs_local $EL \
      --use_attention 0 --no_wmf_in_model --lr_finetune 0.05 \
      --unlearn_ratio $R --unlearn_mode $M
  for pt in 1 2; do
    for emb in wmf bprfull; do
      run "logs/cmp3_${ds}_pt${pt}_emb${emb}.log" "comb $ds Rec-pt${pt} emb=$emb" \
        python code/method_3_receraser_v2.py --dataset $ds --emb_dim 64 --n_shards $K \
          --partition_type $pt --max_epochs_wmf $EWMF --max_epochs_local $EL \
          --use_attention 0 --no_wmf_in_model --lr_finetune 0.05 \
          --embed_source $emb --unlearn_ratio $R --unlearn_mode $M
    done
  done
done
fi

# ============================================================================
# KHỐI 4 - EX6 STABILITY (flip% 4 Ours-partition, KHÔNG train ~5p/dataset)
# ============================================================================
if echo " $BLOCKS " | grep -q " 4 "; then
for ds in ml-1m yelp2018; do
  run "logs/cmp4_ex6_${ds}.log" "M4 EX6 $ds (stability)" \
    python code/measure_stability.py --dataset $ds --n_shards $K --signature_dim $D \
      --unlearn_ratio $R --unlearn_mode $M
done
fi

# ============================================================================
# IN MỌI SỐ
# ============================================================================
rnum(){  # $1=logfile  $2=prefix
  grep -hE "After \(RETAINED" "$1" 2>/dev/null | grep -oE "R@10: [0-9.]+" || echo "(thiếu)"
}
if [ "${1:-}" = "--grep" ]; then
  echo "########## KHỐI 1 - 4 PHƯƠNG PHÁP (Before / After / Unlearn time) ##########"
  for ds in ml-1m yelp2018; do
    echo "----- $ds -----"
    for m in full ours sisa rec; do
      echo "[$ds:$m]"; grep -hE "Before \(FULL|After \(RETAINED|Unlearn time" logs/cmp1_${ds}_${m}.log 2>/dev/null || echo "  (thiếu)"
    done
  done
  echo ""
  echo "########## KHỐI 2 - Ours × 4 PARTITION × {wmf,bprfull} ##########"
  for ds in ml-1m yelp2018; do
    echo "----- $ds -----"
    printf "%-16s : " "sig"        "$(rnum logs/cmp2_${ds}_signature.log)"
    printf "  randomfrozen : " "$(rnum logs/cmp2_${ds}_randomfrozen.log)"
    for part in ubpfrozen inpfrozen; do
      printf "%-14s wmf : " "$part" "$(rnum logs/cmp2_${ds}_${part}_embwmf.log)"
      printf "  bprfull : " "$(rnum logs/cmp2_${ds}_${part}_emb${part#frozen}.log)"
    done
  done
  echo ""
  echo "########## KHỐI 3 - Rec × 3 PARTITION × {wmf,bprfull} ##########"
  for ds in ml-1m yelp2018; do
    echo "----- $ds -----"
    printf "%-14s : " "Random(no-emb)" "$(rnum logs/cmp3_${ds}_pt3_random.log)"
    for pt in 1 2; do
      p=$((pt==1?1:2)); pnm=$((pt==1?InP:UBP))
      printf "%-14s wmf : " "$pnm" "$(rnum logs/cmp3_${ds}_pt${pt}_embwmf.log)"
      printf "  bprfull : " "$(rnum logs/cmp3_${ds}_pt${pt}_emb${pt}.log)"
    done
  done
  echo ""
  echo "########## KHỐI 4 - EX6 (flip% 4 Ours-partition) ##########"
  for ds in ml-1m yelp2018; do
    echo "----- $ds -----"; grep -hE "Ours|Instability" logs/cmp4_ex6_${ds}.log 2>/dev/null || echo "  (thiếu)"
  done
  exit 0
fi

echo "XONG TẤT CẢ $(date +%H:%M:%S) - xem số: bash code/run_compare_all.sh --grep"
