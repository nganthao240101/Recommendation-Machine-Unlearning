#!/usr/bin/env bash
###############################################################################
# CHẠY CODE GỐC TÁC GIẢ (RecEraser_BPR_pytorch.py) - AN TOÀN, TUẦN TỰ
#
# Script này tự chữa CẢ 3 bug gặp phải:
#   (1) ENV VAR ghi đè dataset : batch_test.py đọc RECUNLEARN_DATASET /
#                                 RECUNLEARN_DATA_PATH. Script SET env đúng
#                                 cho MỖI dataset -> Data load đúng, không nhầm.
#   (2) SIZE .pk sai           : preflight so shape user_pretrain.pk với
#                                 n_users (từ train/test.txt). Nếu sai mà có
#                                 .bak_dict -> re-convert; không -> fallback
#                                 --pretrain 0 (train WMF từ đầu, đúng size).
#   (3) EARLY-STOP / epoch     : dùng --epoch 1000 (đủ chín, không dừng sớm
#                                 như lần trước ra 0.0030).
#
# Cách chạy (nền, 1 dòng):
#   nohup bash code/run_orig_receraser.sh > logs/orig_safe.log 2>&1 &
#   tail -f logs/orig_safe.log
#
# Datasets chạy: ml-1m, yelp2018  (UBP partition_type=2, attention, K=8).
# Kết quả: logs/orig_<ds>_rec_ubp_attn.log  (box "AGGREGATION FINAL").
###############################################################################
set -u

ROOT="/workspace/Recommendation-Machine-Unlearning"
CODE="$ROOT/code"
DATA="$ROOT/data"
LOGS="$ROOT/logs"
mkdir -p "$LOGS"

K=8
EPOCH=1000        # đủ epoch local (không early-stop sớm)
EPOCH_AGG=50      # aggregator
EMB=64
LR=0.05
REGS=0.01
BATCH=512
AGG=attention

cd "$CODE"

# ---------------------------------------------------------------------------
# PREFLIGHT: size .pk so với n_users/n_items thật (từ train/test.txt)
# ---------------------------------------------------------------------------
check_pk(){
  # $1=dataset  -> in ra: "OK" hoặc "FIXED" hoặc "BAD"
  ds=$1
  pkuser="$DATA/$ds/user_pretrain.pk"
  pkitm="$DATA/$ds/item_pretrain.pk"

  # n_users / n_items thật (0-indexed: max_id + 1)
  read NU NI < <(python - "$DATA/$ds" <<'PY'
import sys, os
d=sys.argv[1]; nu=ni=0
for fn in ('train.txt','test.txt'):
    p=os.path.join(d,fn)
    if not os.path.exists(p): continue
    for l in open(p):
        parts=l.split()
        if not parts: continue
        try:
            nu=max(nu,int(parts[0])+1)
            for it in parts[1:]: ni=max(ni,int(it)+1)
        except: pass
print(nu,ni)
PY
)

  # shape .pk hiện tại
  read SU SI < <(python - "$pkuser" "$pkitm" <<'PY'
import sys, pickle, numpy as np
def sh(p):
    try:
        o=pickle.load(open(p,'rb'))
        return o.shape[0] if isinstance(o,np.ndarray) else -1
    except: return -1
print(sh(sys.argv[1]), sh(sys.argv[2]))
PY
)

  echo "  preflight $ds: n_users=$NU (user .pk=$SU), n_items=$NI (item .pk=$SI)" >&2
  if [ "$SU" = "$NU" ] && [ "$SI" = "$NI" ]; then
    echo "  OK - .pk đúng size, chạy --pretrain 1" >&2
    echo "pretrain"
    return
  fi
  # size sai -> thử re-convert từ .bak_dict (nếu có)
  if [ -f "$pkuser.bak_dict" ]; then
    echo "  SIZE SAI -> re-convert từ .bak_dict" >&2
    python "$CODE/make_array_pk.py" --dataset "$ds" >&2 || true
    # kiểm tra lại
    SU2=$(python -c "import pickle,numpy as np;print(pickle.load(open('$pkuser','rb')).shape[0])" 2>/dev/null || echo -1)
    if [ "$SU2" = "$NU" ]; then
      echo "  FIXED - .pk đúng $NU dòng, chạy --pretrain 1" >&2
      echo "pretrain"
      return
    fi
  fi
  echo "  KHÔNG FIX ĐƯỢC size (thiếu .bak_dict) -> fallback --pretrain 0 (train WMF từ đầu)" >&2
  echo "scratch"
}

run_ds(){
  local ds=$1
  echo; echo "############ ORIGINAL RecEraser | $ds | UBP($K) attention | pretrain-check ############"

  # (1) SET ENV ĐÚNG cho dataset này (chữa bug ghi đè)
  export RECUNLEARN_DATA_PATH="$DATA"
  export RECUNLEARN_DATASET="$ds"

  local mode
  mode=$(check_pk "$ds")

  # (3) chạy đủ epoch
  if [ "$mode" = "pretrain" ]; then
    echo "  -> chạy --pretrain 1 (WMF .pk), epoch=$EPOCH, agg=$EPOCH_AGG"
    python RecEraser_BPR_pytorch.py --dataset "$ds" --data_path "$DATA/" \
      --part_type 2 --part_num "$K" --agg_type "$AGG" \
      --epoch "$EPOCH" --epoch_agg "$EPOCH_AGG" --embed_size "$EMB" \
      --lr "$LR" --regs "$REGS" --pretrain 1 --save_flag 0 \
      --batch_size "$BATCH" --verbose 100 --Ks "[10,20,50]" \
      > "$LOGS/orig_${ds}_rec_ubp_attn.log" 2>&1
    local rc=$?
    # nếu crash (NaN / size), tự fallback --pretrain 0
    if [ $rc -ne 0 ]; then
      echo "  --pretrain 1 crash (exit=$rc) -> fallback --pretrain 0"
      python RecEraser_BPR_pytorch.py --dataset "$ds" --data_path "$DATA/" \
        --part_type 2 --part_num "$K" --agg_type "$AGG" \
        --epoch "$EPOCH" --epoch_agg "$EPOCH_AGG" --embed_size "$EMB" \
        --lr "$LR" --regs "$REGS" --pretrain 0 --save_flag 0 \
        --batch_size "$BATCH" --verbose 100 --Ks "[10,20,50]" \
        > "$LOGS/orig_${ds}_rec_ubp_attn_scratch.log" 2>&1
      echo "  -> log: logs/orig_${ds}_rec_ubp_attn_scratch.log"
    else
      echo "  OK (exit=0) -> log: logs/orig_${ds}_rec_ubp_attn.log"
    fi
  else
    echo "  -> chạy --pretrain 0 (train WMF từ đầu, size đúng), epoch=$EPOCH"
    python RecEraser_BPR_pytorch.py --dataset "$ds" --data_path "$DATA/" \
      --part_type 2 --part_num "$K" --agg_type "$AGG" \
      --epoch "$EPOCH" --epoch_agg "$EPOCH_AGG" --embed_size "$EMB" \
      --lr "$LR" --regs "$REGS" --pretrain 0 --save_flag 0 \
      --batch_size "$BATCH" --verbose 100 --Ks "[10,20,50]" \
      > "$LOGS/orig_${ds}_rec_ubp_attn_scratch.log" 2>&1
    echo "  -> log: logs/orig_${ds}_rec_ubp_attn_scratch.log"
  fi
}

# ---------------------------------------------------------------------------
# CHẠY TUẦN TỰ (1 dataset xong mới dataset kia - không tranh GPU/CPU)
# ---------------------------------------------------------------------------
for ds in ml-1m yelp2018; do
  run_ds "$ds"
done

echo; echo "############ XONG. Kết quả (box FINAL) ############"
for ds in ml-1m yelp2018; do
  for f in "$LOGS/orig_${ds}_rec_ubp_attn.log" "$LOGS/orig_${ds}_rec_ubp_attn_scratch.log"; do
    if [ -f "$f" ] && grep -q "AGGREGATION FINAL" "$f"; then
      echo "=== $ds ($(basename "$f")) ==="
      grep -E "AGGREGATION FINAL|recall@10|ndcg@10" "$f" | head -4
    fi
  done
done
echo "(log chi tiết: logs/orig_*.log, theo dõi: logs/orig_safe.log)"
