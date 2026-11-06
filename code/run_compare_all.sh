#!/usr/bin/env bash
# ============================================================================
# run_compare_all.sh - CHẠY ĐỦ SỐ so sánh PHƯƠNG PHÁP × PARTITION × MODEL(WMF/BPR)
# ----------------------------------------------------------------------------
# Nguyên tắc: MỌI run có log riêng; NẾU log đã ra "After (RETAINED)" -> BỎ QUA.
# Chọn khối:  BLOCKS="1 2 3 4" (default tất cả).  --grep để in mọi số.
#
#   1) METHODS   : Full / Ours(sig) / SISA / Rec(UBP NGUYÊN TÁC: random-init,
#                  WMF chỉ dùng cho partition, không nạp vào model)          - 2 ds
#   2) OURS×PART : Ours × {signature,Random,UBP,InP}(frozen) × {wmf,bprfull}
#   3) REC×PART  : Rec  × {InP,UBP,Random} × {wmf,bprfull} (no-WMF, mean, lr 0.05)
#   4) EX6       : stability (flip%) các method × partition (không train, ~5p/ds)
#   5) ABLATION  : Ours (ml-1m, signature): K {4,8,16} · ratio {0.1..2%} ·
#                   mode {random,most,fewest} · sig-dim {32,64,128}
#   6) ATTN×MEAN : EX7 - Rec-UBP aggregator: mean vs attention (quét agg lr/ep)
#   comb        : = KHỐI 1 + KHỐI 3 gộp 1 chỗ (4 method + Rec×3-partition×2-model,
#                 CẢ 2 dataset, CẢ 2 model) -> "1 lệnh so tất cả"
#
# LƯU Ý NGYUÊN TÁC: RecEraser gốc LUÔN random-init model; WMF/BPR embedding CHỈ dùng
# cho partition (C/C_U/C_I), KHÔNG bao giờ nạp vào model. --pretrain 1 (nạp emb vào
# model) là variant TỰ THÊM để đo lợi thế pretrain, KHÔNG phải config của tác giả.
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

# ---------------------------------------------------------------------------
# --grep: CHỈ IN SỐ, KHÔNG CHẠY GÌ. Phải kiểm TRƯỚC khi chạy các khối,
# nếu không lệnh này sẽ chạy lại toàn bộ (rất tốn thời gian/GPU).
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--grep" ]; then
  # R@10 theo cả 2 format: cũ "R@10: 0.1234" và mới "R@10=0.1234 | ..."
  rnum(){ grep -hE "After \(RETAINED" "$1" 2>/dev/null | grep -oE "R@10[ =:]+[0-9.]+" | grep -oE "[0-9.]+$" | tail -1 || echo "(thiếu)"; }
  # Dòng metrics đầy đủ (R/NDCG @10,20,50) nếu log có
  rfull(){ grep -hE "After \(RETAINED" "$1" 2>/dev/null | tail -1 || echo "(thiếu)"; }
  echo "########## KHỐI 1 - 4 PHƯƠNG PHÁP (Before / After / Unlearn time) ##########"
  for ds in ml-1m yelp2018; do
    echo "----- $ds -----"
    for m in full ours sisa rec_ubp_author; do
      echo "[$ds:$m]"; grep -hE "Before \(FULL|After \(RETAINED|Unlearn time" logs/cmp1_${ds}_${m}.log 2>/dev/null || echo "  (thiếu)"
    done
  done
  echo ""
  echo "########## KHỐI 2 - Ours × 4 PARTITION × {wmf,bprfull} ##########"
  for ds in ml-1m yelp2018; do
    echo "----- $ds -----"
    printf "  %-14s : %s\n" "sig"          "$(rnum logs/cmp2_${ds}_signature.log)"
    printf "  %-14s : %s\n" "randomfrozen" "$(rnum logs/cmp2_${ds}_randomfrozen.log)"
    for part in ubpfrozen inpfrozen; do
      printf "  %-14s wmf     : %s\n" "$part" "$(rnum logs/cmp2_${ds}_${part}_embwmf.log)"
      printf "  %-14s bprfull : %s\n" "$part" "$(rnum logs/cmp2_${ds}_${part}_embbprfull.log)"
    done
  done
  echo ""
  echo "########## KHỐI 3 - Rec × 3 PARTITION × {wmf,bprfull} ##########"
  for ds in ml-1m yelp2018; do
    echo "----- $ds -----"
    printf "  %-14s : %s\n" "Random(no-emb)" "$(rnum logs/cmp3_${ds}_pt3_random.log)"
    for pt in 1 2; do
      pnm=$([ "$pt" = "1" ] && echo InP || echo UBP)
      printf "  %-14s wmf     : %s\n" "$pnm" "$(rnum logs/cmp3_${ds}_pt${pt}_embwmf.log)"
      printf "  %-14s bprfull : %s\n" "$pnm" "$(rnum logs/cmp3_${ds}_pt${pt}_embbprfull.log)"
    done
  done
  echo ""
  echo "########## KHỐI 4 - EX6 (flip% các method × partition) ##########"
  for ds in ml-1m yelp2018; do
    echo "----- $ds -----"; grep -hE "Ours|Rec|SISA|Instab" logs/cmp4_ex6_${ds}.log 2>/dev/null || echo "  (thiếu)"
  done
  echo ""
  echo "########## BẢNG ĐẦY ĐỦ: Recall/NDCG @10,20,50 (After) ##########"
  for ds in ml-1m yelp2018; do
    echo "----- $ds -----"
    echo "[$ds:full]";        rfull logs/cmp1_${ds}_full.log
    echo "[$ds:ours]";        rfull logs/cmp1_${ds}_ours.log
    echo "[$ds:sisa]";        rfull logs/cmp1_${ds}_sisa.log
    echo "[$ds:rec_ubp_author]"; rfull logs/cmp1_${ds}_rec_ubp_author.log
    for part in signature randomfrozen; do
      echo "[$ds:ours-$part]"; rfull logs/cmp2_${ds}_${part}.log
    done
    for part in ubpfrozen inpfrozen; do
      for emb in wmf bprfull; do
        echo "[$ds:ours-$part-$emb]"; rfull logs/cmp2_${ds}_${part}_emb${emb}.log
      done
    done
    echo "[$ds:rec-random]";  rfull logs/cmp3_${ds}_pt3_random.log
    for pt in 1 2; do
      pnm=$([ "$pt" = "1" ] && echo InP || echo UBP)
      for emb in wmf bprfull; do
        echo "[$ds:rec-$pnm-$emb]"; rfull logs/cmp3_${ds}_pt${pt}_emb${emb}.log
      done
    done
  done
  echo ""
  echo "########## KHỐI 5 - ABLATION (Ours, ml-1m) ##########"
  echo "-- Shard K --"
  for k in 4 8 16; do
    printf "  K=%-4s : %s\n" "$k" "$(rnum logs/cmp5_k${k}.log)"
  done
  echo "-- Unlearn ratio --"
  for r in 0.001 0.005 0.01 0.02; do
    printf "  r=%-6s : %s\n" "$r" "$(rnum logs/cmp5_r${r}.log)"
  done
  echo "-- Unlearn mode --"
  for m in random most fewest; do
    printf "  %-8s : %s\n" "$m" "$(rnum logs/cmp5_mode_${m}.log)"
  done
  echo "-- Signature dim --"
  for d in 32 64 128; do
    printf "  d=%-4s : %s\n" "$d" "$(rnum logs/cmp5_sigd${d}.log)"
  done
  echo ""
  echo "########## KHỐI 6 - EX7: ATTENTION vs MEAN (Rec-UBP) ##########"
  for ds in ml-1m yelp2018; do
    echo "----- $ds -----"
    printf "  %-28s : %s\n" "MEAN (baseline)" "$(rnum logs/cmp6_${ds}_ubp_mean.log)"
    for cfg in "0.05 60" "0.005 60" "0.005 200" "0.001 200"; do
      alr=$(echo "$cfg" | cut -d' ' -f1)
      aep=$(echo "$cfg" | cut -d' ' -f2)
      printf "  %-28s : %s\n" "ATTN lr=$alr ep=$aep" "$(rnum logs/cmp6_${ds}_ubp_attn_lr${alr}_ep${aep}.log)"
    done
  done
  exit 0
fi

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
  run "logs/cmp1_${ds}_rec_ubp_author.log"  "M1 $ds REC-UBP (NGUYÊN TÁC: random-init, WMF-chỉ-partition, attn)" \
    python code/method_3_receraser_v2.py --dataset $ds --emb_dim 64 --n_shards $K \
      --partition_type 2 --max_epochs_wmf $EWMF --max_epochs_local $EL \
      --use_attention 1 --pretrain 0 --lr_finetune 0.05 --unlearn_ratio $R --unlearn_mode $M
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
  run "logs/cmp1_${ds}_rec_ubp_author.log"  "comb $ds REC-UBP (NGUYÊN TÁC: random-init, WMF-chỉ-partition, attn)" \
    python code/method_3_receraser_v2.py --dataset $ds --emb_dim 64 --n_shards $K \
      --partition_type 2 --max_epochs_wmf $EWMF --max_epochs_local $EL \
      --use_attention 1 --pretrain 0 --lr_finetune 0.05 --unlearn_ratio $R --unlearn_mode $M
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
# KHỐI 5 - ABLATION (Ours, ml-1m, signature base): mỗi yếu tố đổi 1, giữ nguyên
#   K (shard):   4 / 8 / 16
#   ratio:       0.001 / 0.005 / 0.01 / 0.02
#   mode:        random / most / fewest
#   sig_dim:     32 / 64 / 128
# ============================================================================
if echo " $BLOCKS " | grep -q " 5 "; then
for k in 4 8 16; do
  run "logs/cmp5_k${k}.log" "M5 K=${k}" \
    python code/method_4_ours.py --dataset ml-1m --emb_dim 64 --n_shards $k \
      --signature_dim $D --max_epochs $E --unlearn_ratio $R --unlearn_mode $M
done
# NOTE: ablation chạy trên PARTITION SIGNATURE (native Ours) - build rất nhanh
#       (matrix sparses x R), KẾT QUẢ DETERMINISTIC (seed 42) -> mọi giá trị
#       ratio/mode đã dùng CHUNG 1 partition. KHÔNG cần --persist_partition.
#       (Chỉ bật --persist_partition 1 khi sweep trên InP/UBP - partition k-means
#        / bprfull đắt tiền, để reuse 1 partition + audit qua file .npz.)
for r in 0.001 0.005 0.01 0.02; do
  run "logs/cmp5_r${r}.log" "M5 ratio=${r}" \
    python code/method_4_ours.py --dataset ml-1m --emb_dim 64 --n_shards $K \
      --signature_dim $D --max_epochs $E --unlearn_ratio $r --unlearn_mode $M
done
for m in random most fewest; do
  run "logs/cmp5_mode_${m}.log" "M5 mode=${m}" \
    python code/method_4_ours.py --dataset ml-1m --emb_dim 64 --n_shards $K \
      --signature_dim $D --max_epochs $E --unlearn_ratio $R --unlearn_mode $m
done
for d in 32 64 128; do
  run "logs/cmp5_sigd${d}.log" "M5 sigdim=${d}" \
    python code/method_4_ours.py --dataset ml-1m --emb_dim 64 --n_shards $K \
      --signature_dim $d --max_epochs $E --unlearn_ratio $R --unlearn_mode $M
done
fi

# ============================================================================
# KHỐI 6 - EX7: ATTENTION vs MEAN (RecEraser aggregator)
#   Cùng partition (UBP), cùng model (random-init, WMF chỉ partition),
#   chỉ đổi aggregator: attention vs mean, và quét lr/epochs của aggregator.
#   Sau fix stop-gradient (khớp code gốc tác giả), attention dùng
#   --agg_lr nhỏ + --agg_epochs nhiều hơn.
# ============================================================================
if echo " $BLOCKS " | grep -q " 6 "; then
for ds in ml-1m yelp2018; do
  # --- mean (baseline, hard-routing 1 shard) ---
  run "logs/cmp6_${ds}_ubp_mean.log" "M6 $ds Rec-UBP MEAN" \
    python code/method_3_receraser_v2.py --dataset $ds --emb_dim 64 --n_shards $K \
      --partition_type 2 --max_epochs_wmf $EWMF --max_epochs_local $EL \
      --use_attention 0 --no_wmf_in_model --lr_finetune 0.05 \
      --unlearn_ratio $R --unlearn_mode $M
  # --- attention: quét (agg_lr, agg_epochs) ---
  for cfg in "0.05 60" "0.005 60" "0.005 200" "0.001 200"; do
    alr=$(echo "$cfg" | cut -d' ' -f1)
    aep=$(echo "$cfg" | cut -d' ' -f2)
    run "logs/cmp6_${ds}_ubp_attn_lr${alr}_ep${aep}.log" "M6 $ds Rec-UBP ATTN lr=$alr ep=$aep" \
      python code/method_3_receraser_v2.py --dataset $ds --emb_dim 64 --n_shards $K \
        --partition_type 2 --max_epochs_wmf $EWMF --max_epochs_local $EL \
        --use_attention 1 --no_wmf_in_model --lr_finetune 0.05 \
        --agg_lr $alr --agg_epochs $aep \
        --unlearn_ratio $R --unlearn_mode $M
  done
done
fi

# ============================================================================
# IN MỌI SỐ: đã xử lý ở đầu script (--grep) để KHÔNG chạy lại các khối.
# ============================================================================
echo "XONG TẤT CẢ $(date +%H:%M:%S) - xem số: bash code/run_compare_all.sh --grep"
