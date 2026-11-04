#!/usr/bin/env bash
# ============================================================================
# run_compare_all.sh - CHẠY ĐỦ SỐ so sánh PHƯƠNG PHÁP × MODEL (WMF vs BPR)
# ----------------------------------------------------------------------------
# 4 khối (chạy nối tiếp, 1 GPU; block nào chạy được chọn bằng $BLOCK):
#   1) METHODS      : Full / Ours(sign) / SISA / Rec-UBP(WMF-in)  - 2 dataset
#   2) PARTITIONS   : Ours × {signature,Random,UBP,InP}(frozen, WMF-emb) - 2 ds
#   3) EMBEDSOURCES : Ours × {InP,UBP} × {wmf,bprfull}  -> "so 2 model" - ml-1m
#   4) EX6          : stability (flip%) cho 4 Ours-partition (không train, ~5p)
#
# Cách dùng:
#   bash code/run_compare_all.sh              # chạy tất cả (khóa đêm)
#   BLOCK=3 bash code/run_compare_all.sh      # chỉ chạy khối so WMF vs BPR
#   BLOCKS="1 3 4" bash code/run_compare_all.sh
#
# Sau khi xong (hoặc từng khối), xem số:
#   bash code/run_compare_all.sh --grep      # in mọi số (Before/After/Time/flip)
# ============================================================================
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
K=8; E=50; D=64; R=0.01; M=most; EL=60; EWMF=100
export CUDA_VISIBLE_DEVICES=0

BLOCKS="${BLOCKS:-1 2 3 4}"

log(){ echo "=== $(date +%H:%M:%S) $1 ==="; }

# ============================================================================
# KHỐI 1 - METHODS (4 phương pháp, 2 dataset, lr=0.05 fair, Rec dùng --lr_finetune)
# ============================================================================
if echo " $BLOCKS " | grep -q " 1 "; then
for ds in ml-1m yelp2018; do
  log "M1 $ds FULL"
  python code/method_1_full_retrain.py --dataset $ds --emb_dim 64 --max_epochs $E \
    --unlearn_ratio $R --unlearn_mode $M > logs/cmp1_${ds}_full.log 2>&1
  log "M1 $ds OURS(sig)"
  python code/method_4_ours.py --dataset $ds --emb_dim 64 --n_shards $K \
    --signature_dim $D --max_epochs $E --unlearn_ratio $R --unlearn_mode $M \
    > logs/cmp1_${ds}_ours.log 2>&1
  log "M1 $ds SISA"
  python code/method_2_sisa.py --dataset $ds --emb_dim 64 --n_shards $K \
    --max_epochs $E --unlearn_ratio $R --unlearn_mode $M > logs/cmp1_${ds}_sisa.log 2>&1
  log "M1 $ds REC-UBP (WMF-in, attn) [Rec thật]"
  python code/method_3_receraser_v2.py --dataset $ds --emb_dim 64 --n_shards $K \
    --partition_type 2 --max_epochs_wmf $EWMF --max_epochs_local $EL \
    --use_attention 1 --lr_finetune 0.05 --unlearn_ratio $R --unlearn_mode $M \
    > logs/cmp1_${ds}_rec.log 2>&1
done
fi

# ============================================================================
# KHỐI 2 - Ours × 4 PARTITION (frozen, WMF-embedding; cùng cơ chế -> stable 0%)
# ============================================================================
if echo " $BLOCKS " | grep -q " 2 "; then
for ds in ml-1m yelp2018; do
for part in signature randomfrozen ubpfrozen inpfrozen; do
  log "M2 $ds $part (WMF-emb)"
  python code/method_4_ours.py --dataset $ds --emb_dim 64 --n_shards $K \
    --signature_dim $D --max_epochs $E --unlearn_ratio $R --unlearn_mode $M \
    --partition $part --max_epochs_wmf $EWMF \
    > logs/cmp2_${ds}_${part}.log 2>&1
done
done
fi

# ============================================================================
# KHỐI 3 - "SO 2 MODEL" : Ours × {InP,UBP} × {WMF, BPR-full}
#   wmf     = load wmf_embeddings.npz (BPR-trọng-1, KHÔNG reg)  -> partition
#   bprfull = train BPRMF (BPR + L2 reg 0.01 + clamp) FULL data -> partition
#   Cùng partition, đổi NGUỒN EMBEDDING => xem BPR có tốt hơn WMF không
# ============================================================================
if echo " $BLOCKS " | grep -q " 3 "; then
for ds in ml-1m yelp2018; do
for part in inpfrozen ubpfrozen; do
for emb in wmf bprfull; do
  log "M3 $ds $part emb=$emb"
  python code/method_4_ours.py --dataset $ds --emb_dim 64 --n_shards $K \
    --signature_dim $D --max_epochs $E --unlearn_ratio $R --unlearn_mode $M \
    --partition $part --max_epochs_wmf $EWMF --embed_source $emb \
    > logs/cmp3_${ds}_${part}_emb${emb}.log 2>&1
done
done
done
fi

# ============================================================================
# KHỐI 4 - EX6 STABILITY (flip% cho 4 Ours-partition, KHÔNG train ~5p/dataset)
# ============================================================================
if echo " $BLOCKS " | grep -q " 4 "; then
for ds in ml-1m yelp2018; do
  log "M4 EX6 $ds"
  python code/measure_stability.py --dataset $ds --n_shards $K --signature_dim $D \
    --unlearn_ratio $R --unlearn_mode $M > logs/cmp4_ex6_${ds}.log 2>&1
done
fi

# ============================================================================
# IN MỌI SỐ (chạy "bash code/run_compare_all.sh --grep" để xem kết quả)
# ============================================================================
if [ "${1:-}" = "--grep" ]; then
  echo "########## KHỐI 1 - 4 PHƯƠNG PHÁP (Before / After / Unlearn time) ##########"
  for ds in ml-1m yelp2018; do
    echo "----- $ds -----"
    for m in full ours sisa rec; do
      echo "[$ds:$m]"; grep -hE "Before \(FULL|After \(RETAINED|Unlearn time" logs/cmp1_${ds}_${m}.log 2>/dev/null || echo "  (thiếu)"
    done
  done
  echo ""
  echo "########## KHỐI 2 - Ours × 4 PARTITION (WMF-emb, stable 0%) ##########"
  for ds in ml-1m yelp2018; do
    for part in signature randomfrozen ubpfrozen inpfrozen; do
      f="logs/cmp2_${ds}_${part}.log"
      printf "%s %-14s : " "$ds" "$part"
      grep -hE "After \(RETAINED" "$f" 2>/dev/null | grep -oE "R@10: [0-9.]+" || echo "(thiếu)"
    done
  done
  echo ""
  echo "########## KHỐI 3 - SO 2 MODEL (WMF vs BPR-full, InP/UBP) ##########"
  for ds in ml-1m yelp2018; do
    for part in inpfrozen ubpfrozen; do
      for emb in wmf bprfull; do
        f="logs/cmp3_${ds}_${part}_emb${emb}.log"
        printf "%s %-14s emb=%s : " "$ds" "$part" "$emb"
        grep -hE "After \(RETAINED" "$f" 2>/dev/null | grep -oE "R@10: [0-9.]+" || echo "(thiếu)"
      done
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
