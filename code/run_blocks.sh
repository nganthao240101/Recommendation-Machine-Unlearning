#!/usr/bin/env bash
# ============================================================================
# run_blocks.sh - chạy thí nghiệm SAFER TỪNG KHỐI một, NỀN.
#
#   7 arm chính, chạy trên CẢ 2 dataset (ml-1m, yelp2018):
#     PROPOSED      SAFER A - partition theo signature
#     PROPOSED_BAL  SAFER B - signature + cân bằng tải
#     SISA          random user%K, suy luận = trung bình K model
#     RANDOM_PERM   partition ngẫu nhiên (baseline cho delta / p-value)
#     RANDOM_EMB    partition theo embedding ngẫu nhiên (đối chứng)
#     RECERASER     RecEraser: partition UBP (k-means trên WMF) + attention,
#                   model khởi tạo NGẪU NHIÊN như bản gốc, WMF chỉ để chia
#     GLOBAL        retrain toàn bộ (trần)
#
#   Cấu hình gốc: K=8, emb_dim=64, sig_dim=64, xóa 1% user ngẫu nhiên,
#   50 epoch train + 50 epoch retrain, lr 0.05, batch 512, 5 seed.
#   Mỗi khối chỉ đổi MỘT yếu tố so với cấu hình gốc:
#
#     main     mode random                    main_<ds>
#     modes    mode most / fewest             mode_most_<ds>, mode_fewest_<ds>
#     single   xóa TỪNG user (20 request)     single_<ds>
#     shards   K = 4 / 16   (K=8 là main)     k4_<ds>, k16_<ds>
#     dims     emb_dim = 32 / 128 (64 là main) dim32_<ds>, dim128_<ds>
#     recpart  [chạy sau] chỉ RECERASER, partition random  recrand_<ds>
#              (InP chưa có: InP của RecEraser chia TƯƠNG TÁC, pipeline này chia
#               theo user - xem exp_receraser_arm.py, kind='inp')
#   <ds> = ml1m | yelp
#
#   Mọi log đo: R/NDCG @10/20/50, chất lượng partition (IntraSim, NR@k,
#   độ lệch tải), instability, thời gian train/unlearn/parallel, số shard bị
#   ảnh hưởng. Riêng single đo thời gian 1 request xóa.
#
#   Cách dùng (trên máy SSH, tại thư mục repo):
#     bash code/run_blocks.sh reset        # XÓA HẾT log + kết quả cũ (nén backup trước)
#     bash code/run_blocks.sh start full   # MỘT LỆNH CHẠY HẾT: ml-1m rồi yelp, Rec-UBP trước, engine GPU
#     bash code/run_blocks.sh start fullwmf  # như full, nhưng mọi arm dùng model WMF (log *_wmf)
#     bash code/run_blocks.sh start        # chạy NGẦM toàn bộ (all) trên GPU, 1 job/lần
#     bash code/run_blocks.sh results      # in kết quả: xong trước in trước, chưa xong thì báo
#     bash code/run_blocks.sh status       # tiến độ từng log
#     bash code/run_blocks.sh gpu          # có đang chạy thí nghiệm trên GPU không
#     bash code/run_blocks.sh smoke        # chạy thử 7 arm trên dataset tí hon (start tự chạy)
#     bash code/run_blocks.sh stop         # dừng chuỗi đang chạy ngầm
#     bash code/run_blocks.sh torch        # chỉ kiểm tra CUDA + WMF embedding
#     bash code/run_blocks.sh main         # 1 khối (tương tự: modes single shards dims recpart)
#     bash code/run_blocks.sh all          # mọi khối, chạy trực tiếp (start = all chạy ngầm)
#     bash code/run_blocks.sh csv          # xuất bảng kết quả ra results/csv/
#     bash code/run_blocks.sh check        # tiến độ + log cũ/mới + kết quả
#     bash code/run_blocks.sh clean        # xóa log CŨ, giữ log mới (nén backup trước)
#   Chỉ 1 dataset:  DATASETS="ml-1m" bash code/run_blocks.sh start
#   Ablation K/dim chỉ trên ml-1m:  ABLATE_DATASETS="ml-1m" bash code/run_blocks.sh start
#   Engine GPU (nhanh hơn trên GPU, xem bench):  RECUNLEARN_ENGINE=gpu bash code/run_blocks.sh start
#     bash code/run_blocks.sh bench        # đo engine cũ vs GPU trên dữ liệu thật (vài phút)
#
#   Tránh sai thời gian: mọi khối giữ khóa logs/.run.lock, nên KHÔNG bao giờ
#   có 2 thí nghiệm cùng chạy trên GPU; lệnh thứ 2 bị từ chối. Đầu mỗi log
#   ghi trạng thái GPU lúc bắt đầu (bộ nhớ đã dùng, % tải) để đối chiếu.
# ============================================================================
set -u
cd "$(dirname "$0")/.."
mkdir -p logs

DATASETS="${DATASETS:-ml-1m yelp2018}"
# Ablation K / số chiều chạy trên các dataset nào (mặc định: như DATASETS).
# ABLATE_DATASETS="ml-1m" = ablation chỉ trên ml-1m (yelp chỉ main/modes/single/recpart).
ABLATE_DATASETS="${ABLATE_DATASETS:-$DATASETS}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
LOCK=logs/.run.lock
PIDFILE=logs/.run.pid
QUEUEFILE=logs/.queue.pid
SEEDS="0,1,2,3,4"
ARMS="PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,RECERASER,GLOBAL"
# cấu hình gốc; mỗi khối đè lên đúng 1 yếu tố
K=8; DIM=64; SIG=64; MODE=random

# Backbone mà mọi arm train: bprmf (mặc định) hoặc wmf (WMF của tác giả,
# xem code/wmf_backbone.py). Log WMF có đuôi _wmf, không đè log BPRMF.
BACKBONE="${BACKBONE:-bprmf}"
# Khối single: số yêu cầu xóa (mỗi yêu cầu 1 user) mỗi seed. 5 x 5 seed = 25
# lần đo cho mỗi method; RecEraser phải train lại aggregator ở MỖI yêu cầu nên
# đây là phần chậm nhất. Retrain đo trên min(3, SINGLE_N) yêu cầu đầu.
SINGLE_N="${SINGLE_N:-5}"
# Như bài RecEraser: train MỘT model gốc, rồi đo từng yêu cầu xóa ngẫu nhiên và
# lấy trung bình; không lặp lại theo seed (thời gian xóa phụ thuộc kích thước
# shard, không phụ thuộc seed). SINGLE_SEEDS="0,1,2" để lặp lại nếu cần.
SINGLE_SEEDS="${SINGLE_SEEDS:-0}"
bsuffix () { [ "$BACKBONE" = wmf ] && echo _wmf; }

short () { case "$1" in ml-1m) echo ml1m ;; yelp2018) echo yelp ;; *) echo "$1" ;; esac; }

banner () {
  echo ""
  echo "=============================================================="
  echo "$1   ($(date +%H:%M:%S))"
  echo "=============================================================="
}

# log do code HIỆN TẠI sinh ra? (xem code/clean_old_results.py)
is_new () { python code/clean_old_results.py --is-new "$1"; }

run () {
  local name="$1"; shift
  local f="logs/${name}.out"
  local tee_opt="" skip=""
  if [ -f "$f" ]; then
    if grep -q "CONSOLE SUMMARY" "$f" 2>/dev/null && is_new "$f"; then
      echo "[skip] $name  (đã xong bằng code hiện tại)"
      return 0
    fi
    # Log dở, hoặc log đã xong nhưng có run RecEraser cũ/hỏng: giữ mọi cặp
    # arm/seed còn hợp lệ, bỏ run hỏng, chỉ chạy phần còn thiếu và ghi NỐI
    # vào log. Log của code cũ (trước a716452) thì chạy lại từ đầu.
    skip=$(python code/clean_old_results.py --prepare-resume "$f")
    if [ -n "$skip" ]; then
      tee_opt="-a"
      set -- "$@" --skip_runs "$skip"
      echo "[resume] $name: giữ $(echo "$skip" | tr ',' ' ' | wc -w) run hợp lệ, chạy phần còn lại"
      grep "^\[resume\]" "$f" | tail -1
    else
      local old="logs/${name}.out.old_$(date +%Y%m%d_%H%M%S)"
      mv "$f" "$old"
      echo "[rerun] $name: không giữ được run nào (code cũ / chưa xong run nào) -> chạy lại từ đầu, log cũ -> $old"
    fi
  fi
  banner "START $name"
  local t0=$SECONDS
  { echo "[gpu] $(gpu_state)"; "$@"; } 2>&1 | tee $tee_opt "logs/${name}.out"
  local rc=${PIPESTATUS[0]}
  echo "--------------------------------------------------------------"
  echo "DONE $name  (rc=$rc, $((SECONDS-t0))s)  -> logs/${name}.out"
  echo "--------------------------------------------------------------"
  # Hai thí nghiệm lỗi liên tiếp = lỗi hệ thống (đường dẫn, code), không phải
  # lỗi ngẫu nhiên: dừng cả chuỗi thay vì đốt GPU cho các thí nghiệm sau.
  if [ "$rc" != 0 ]; then
    FAILS=$((FAILS + 1))
    echo "!! $name lỗi: $(grep -E '^[A-Za-z_.]*(Error|Exception)' "logs/${name}.out" | tail -1)"
    if [ "$FAILS" -ge 2 ]; then
      echo "!! $FAILS thí nghiệm lỗi liên tiếp - DỪNG cả chuỗi. Sửa lỗi rồi start lại."
      exit 3
    fi
  else
    FAILS=0
  fi
  return $rc
}
FAILS=0

# exp_partition_quality với cấu hình gốc, đè bằng: k=<K> dim=<D> mode=<M> arms=<..> extra=<..>
pq () {   # pq <log> <ds> [k=..] [dim=..] [mode=..] [arms=..] [-- extra args]
  local name="$1" ds="$2"; shift 2
  local k=$K d=$DIM m=$MODE a=$ARMS
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    case "$1" in k=*) k="${1#k=}" ;; dim=*) d="${1#dim=}" ;;
                 mode=*) m="${1#mode=}" ;; arms=*) a="${1#arms=}" ;; esac
    shift
  done
  [ "${1:-}" = "--" ] && shift
  run "${name}$(bsuffix)" python code/exp_partition_quality.py --dataset "$ds" \
    --unlearn_mode "$m" --n_shards "$k" --emb_dim "$d" --sig_dim $SIG \
    --epochs 50 --retrain_epochs 50 --lr 0.05 --batch_size 512 \
    --unlearn_ratio 0.01 --seeds $SEEDS --arms "$a" --backbone "$BACKBONE" "$@"
}

# trạng thái GPU: tên, bộ nhớ đã dùng, % tải, số process đang dùng GPU
gpu_state () {
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; return; }
  local q n
  q=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" \
      --query-gpu=name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader 2>/dev/null)
  n=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-compute-apps=pid \
      --format=csv,noheader 2>/dev/null | grep -c .)
  echo "$q | compute processes: $n"
}

# Chỉ 1 chuỗi thí nghiệm chạy tại một thời điểm (giữ khóa tới khi xong).
take_lock () {
  exec 9>"$LOCK"
  if [ "${WAIT_LOCK:-0}" = "1" ]; then
    if ! flock -n 9; then
      echo "[queue] $(date '+%F %T') đang có chuỗi khác chạy - chờ nó xong rồi chạy tiếp"
      flock 9
      echo "[queue] $(date '+%F %T') chuỗi trước đã xong - bắt đầu"
    fi
    echo $$ > "$PIDFILE"; rm -f "$QUEUEFILE"
    return
  fi
  if ! flock -n 9; then
    echo "!! Đang có thí nghiệm khác chạy (khóa $LOCK). Không chạy song song"
    echo "   để thời gian unlearn đo trên GPU không bị nhiễu."
    echo "   Xem: bash code/run_blocks.sh status   |   dừng: bash code/run_blocks.sh stop"
    exit 1
  fi
}
lock_busy () { ( exec 9>"$LOCK"; ! flock -n 9 ); }

# ------------------------------------------------------------------ preflight
# CUDA là bắt buộc: số đo thời gian chỉ so sánh được trong cùng một máy.
# RECERASER cần data/<ds>/wmf_embeddings.npz cho partition: thiếu thì dừng.
preflight () {
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

  echo "  GPU ${CUDA_VISIBLE_DEVICES}: $(gpu_state)"
  local busy
  busy=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=memory.used \
         --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  if [ -n "$busy" ] && [ "$busy" -gt 2000 ] 2>/dev/null; then
    echo "  !! GPU đã dùng ${busy} MiB trước khi chạy: có tiến trình khác (có thể ở"
    echo "     container khác, nên nvidia-smi không liệt kê). Thời gian đo sẽ bị nhiễu"
    echo "     nếu nó đang tính toán. Mỗi log ghi dòng [gpu] lúc bắt đầu để đối chiếu."
  fi
  for DS in $DATASETS; do
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
  echo ""
}

# ------------------------------------------------------------------ blocks
block_torch () { banner "Torch / CUDA + embedding check"; preflight; echo "OK - co the chay cac khoi."; }

block_main () {
  preflight
  for ds in $DATASETS; do pq "main_$(short $ds)" "$ds"; done
}

block_modes () {
  preflight
  for ds in $DATASETS; do
    for m in most fewest; do pq "mode_${m}_$(short $ds)" "$ds" mode=$m; done
  done
}

block_single () {
  preflight
  for ds in $DATASETS; do
    run "single_$(short $ds)$(bsuffix)" python code/exp_single_request.py --dataset "$ds" \
      --n_shards $K --emb_dim $DIM --sig_dim $SIG --epochs 50 --retrain_epochs 50 \
      --lr 0.05 --batch_size 512 --n_users $SINGLE_N --seeds $SINGLE_SEEDS --arms "$ARMS" \
      --backbone "$BACKBONE"
  done
}

in_ablate () { case " $ABLATE_DATASETS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

block_shards () {
  preflight
  for ds in $DATASETS; do
    in_ablate "$ds" || { echo "[skip] k4/k16 $ds (ABLATE_DATASETS=$ABLATE_DATASETS)"; continue; }
    for k in 4 16; do pq "k${k}_$(short $ds)" "$ds" k=$k; done
  done
}

block_dims () {
  preflight
  for ds in $DATASETS; do
    in_ablate "$ds" || { echo "[skip] dim32/dim128 $ds (ABLATE_DATASETS=$ABLATE_DATASETS)"; continue; }
    for d in 32 128; do pq "dim${d}_$(short $ds)" "$ds" dim=$d; done
  done
}

# [chạy sau] RecEraser với partition ngẫu nhiên: cùng cấu hình gốc, chỉ arm
# RECERASER. So với RECERASER (UBP) trong main_<ds> để tách phần đóng góp của
# partition. InP không chạy ở đây: --receraser_partition inp chỉ là bản thay
# thế theo user, không phải InP (chia tương tác) của RecEraser.
block_recpart () {
  preflight
  for ds in $DATASETS; do
    pq "recrand_$(short $ds)" "$ds" arms=RECERASER -- --receraser_partition random
  done
}

# ------------------------------------------------------------------ full
# MỘT LỆNH CHẠY HẾT, theo thứ tự:
#   1. ml-1m : main -> modes -> single -> shards (K=4,16) -> dims (32,128)
#   2. yelp  : main -> modes -> single -> shards (K=4,16) -> dims (32,128)
#   3. Rec với partition khác (recpart): ml-1m rồi yelp  [ưu tiên thấp nhất]
#   4. xuất CSV
# Rec-UBP trước: trong mọi thí nghiệm, arm RECERASER (partition UBP) chạy đầu
# tiên ở mỗi seed. Rec với partition khác để cuối cùng. Mỗi arm tự đặt lại
# seed và tập user bị xóa chỉ phụ thuộc vào seed, nên thứ tự không đổi kết quả.
# Engine GPU mặc định (RECUNLEARN_ENGINE=python để dùng vòng lặp cũ).
# Log đã xong thì bỏ qua, log dở thì chạy tiếp phần còn thiếu.
block_full () {
  export RECUNLEARN_ENGINE="${RECUNLEARN_ENGINE:-gpu}"
  ARMS="RECERASER,PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,GLOBAL"
  local all_ds="${FULL_DATASETS:-ml-1m yelp2018}" ds
  ABLATE_DATASETS="${FULL_ABLATE:-$all_ds}"
  banner "FULL: $all_ds | Rec-UBP trước | engine=$RECUNLEARN_ENGINE | ablation trên: $ABLATE_DATASETS"
  for ds in $all_ds; do
    DATASETS="$ds"
    banner "=== DATASET $ds ==="
    block_main; block_modes; block_single; block_shards; block_dims
  done
  banner "=== Rec với partition khác (ưu tiên thấp nhất) ==="
  for ds in $all_ds; do
    DATASETS="$ds"
    block_recpart
  done
  DATASETS="$all_ds"
  block_csv
  banner "FULL XONG - xem: bash code/run_blocks.sh results"
}

# ------------------------------------------------------------------ fullwmf
# Đúng các thí nghiệm của full, chỉ đổi model BPRMF -> WMF ở MỌI arm.
# Log: <tên>_wmf (main_ml1m_wmf, ...). Mọi thứ khác như full.
block_fullwmf () {
  BACKBONE=wmf
  block_full
}

# ------------------------------------------------------------------ status
block_status () {
  banner "STATUS"
  # DONE / RUNNING (co job dang ghi) / FAILED (dung giua chung) / OLD / MISSING
  python code/clean_old_results.py --status
  echo ""
  echo "  Job dang chay:"
  ps -ww -eo pid,etime,args | grep -E "exp_partition_quality|exp_single_request|run_blocks\.sh (main|modes|single|shards|dims|recpart|all|full|fullwmf)" \
    | grep -v grep | sed 's/^/    /' || echo "    (khong co job nao)"
}

# ------------------------------------------------------------------ csv
# In bảng kết quả ra results/csv/. Chỉ đọc log, không train gì.
#   <khoi>_raw.csv    1 dòng / (log, arm, seed)
#   <khoi>_mean.csv   1 dòng / (log, arm): mean, std, delta vs RANDOM_PERM, p
#   <khoi>_tables.txt bảng đầy đủ dạng text (utility, partition, instability, cost, paired)
#   all_*.csv         gộp mọi log của exp_partition_quality
#   single.csv        thời gian 1 request xóa, mỗi arm, mỗi dataset
# Log chưa xong vẫn được đọc phần đã chạy; cột n_seeds cho biết đủ 5 seed chưa.
csv_group () {   # csv_group <ten> <log1> [log2 ...]
  local name="$1"; shift
  local have=()
  # chỉ đọc log của code hiện tại, để không trộn số cũ vào bảng
  for n in "$@"; do
    [ -f "logs/${n}.out" ] && is_new "logs/${n}.out" && have+=("logs/${n}.out")
  done
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
  local all_ds="ml1m yelp" x
  x="$(bsuffix)"
  local L_main="" L_modes="" L_shards="" L_dims="" L_rec="" L_single="" L_all=""
  for s in $all_ds; do
    L_main="$L_main main_$s$x"
    L_modes="$L_modes mode_most_$s$x mode_fewest_$s$x"
    L_shards="$L_shards k4_$s$x main_$s$x k16_$s$x"      # K=8 là main
    L_dims="$L_dims dim32_$s$x main_$s$x dim128_$s$x"    # dim=64 là main
    L_rec="$L_rec main_$s$x recrand_$s$x"                # RECERASER UBP nằm trong main
    L_single="$L_single logs/single_$s$x.out"
    L_all="$L_all main_$s$x mode_most_$s$x mode_fewest_$s$x k4_$s$x k16_$s$x dim32_$s$x dim128_$s$x recrand_$s$x"
  done
  csv_group "main$x"    $L_main
  csv_group "modes$x"   $L_main $L_modes
  csv_group "shards$x"  $L_shards
  csv_group "dims$x"    $L_dims
  csv_group "recpart$x" $L_rec
  csv_group "all$x"     $L_all

  local have=()
  for f in $L_single; do [ -f "$f" ] && is_new "$f" && have+=("$f"); done
  if [ ${#have[@]} -gt 0 ]; then
    python code/export_single_request_csv.py "${have[@]}" --out "results/csv/single$x.csv" | sed 's/^/  /'
  else
    echo "  [skip] single - chua co log"
  fi
  echo ""
  ls -la results/csv
}

# ------------------------------------------------------------------ check
# Kiểm tra log + kết quả mới nhất: tiến độ từng log, log nào CŨ / MỚI,
# rồi xuất lại CSV và in bảng kết quả gọn. Không train, không xóa gì.
block_check () {
  block_status
  banner "LOG CU / MOI (chi liet ke, khong xoa)"
  python code/clean_old_results.py
  block_csv > /dev/null
  banner "KET QUA MOI NHAT"
  python code/show_latest_results.py
}

# ------------------------------------------------------------------ clean
# Xóa log/kết quả CŨ. Mặc định nén vào ../old_results_<giờ>.tar.gz trước
# rồi mới xóa; đặt NO_BACKUP=1 để xóa hẳn. Sau đó xuất lại CSV từ log mới.
block_clean () {
  banner "XOA KET QUA CU"
  if [ "${NO_BACKUP:-0}" = "1" ]; then
    python code/clean_old_results.py --yes --no-backup
  else
    python code/clean_old_results.py --yes
  fi
  block_csv > /dev/null
  echo ""
  echo "Da xuat lai results/csv/ tu cac log con lai."
}

# ------------------------------------------------------------------ start / stop
# start: kiểm tra trước (CUDA, embedding), rồi chạy NGẦM toàn bộ 'all' trong
# 1 process group riêng; thoát SSH vẫn chạy. Từng thí nghiệm chạy tuần tự.
block_smoke () {
  banner "SMOKE TEST - 7 arm, dataset tí hon, cả 2 script, trên CPU (1-2 phút)"
  # CPU: không chen vào GPU của thí nghiệm đang đo thời gian
  CUDA_VISIBLE_DEVICES="" python code/smoke_test.py
}

# Đang có chuỗi chạy thì KHÔNG dừng nó: xếp hàng, chờ nó xong rồi tự chạy
# tiếp. Lúc đó all bỏ qua log đã xong, chạy tiếp log dở từ run còn thiếu.
block_start () {
  local blk="${1:-all}"
  case "$blk" in all|full|fullwmf|main|modes|single|shards|dims|recpart) ;;
    *) echo "!! start chỉ nhận: all | full | fullwmf | main | modes | single | shards | dims | recpart"; exit 1 ;;
  esac
  if [ -f "$QUEUEFILE" ] && kill -0 "$(cat "$QUEUEFILE")" 2>/dev/null; then
    echo "!! Đã có một chuỗi đang xếp hàng (PID $(cat "$QUEUEFILE")). Không cần start thêm."; exit 1
  fi
  local busy=0 out
  lock_busy && busy=1
  case "$blk" in full|fullwmf) DATASETS="${FULL_DATASETS:-ml-1m yelp2018}" ;; esac
  banner "START (ngầm) - khối $blk - DATASETS=${FULL_DATASETS:-$DATASETS}"
  preflight
  block_smoke || { echo "!! Smoke test hỏng - KHÔNG chạy ngầm. Sửa lỗi trên rồi start lại."; exit 1; }
  out="logs/block_${blk}_$(date +%Y%m%d_%H%M%S).out"
  WAIT_LOCK=1 nohup setsid bash code/run_blocks.sh "$blk" > "$out" 2>&1 < /dev/null &
  local pid=$!
  sleep 3
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "!! Chuỗi dừng ngay khi vừa chạy. Xem: tail -30 $out"; exit 1
  fi
  if [ "$busy" = 1 ]; then
    echo "$pid" > "$QUEUEFILE"
    echo "Đang có chuỗi chạy - KHÔNG dừng nó. Chuỗi mới (PID $pid) đã XẾP HÀNG:"
    echo "  chờ chuỗi hiện tại xong, rồi tự chạy các thí nghiệm còn thiếu/dở."
  else
    echo "$pid" > "$PIDFILE"
    echo "Đã chạy ngầm, PID $pid."
  fi
  echo "  Log tổng: $out"
  echo "  tiến độ : bash code/run_blocks.sh status"
  echo "  kết quả : bash code/run_blocks.sh results"
  echo "  dừng    : bash code/run_blocks.sh stop"
}

block_stop () {
  if [ -f "$QUEUEFILE" ] && kill -0 "$(cat "$QUEUEFILE")" 2>/dev/null; then
    kill -TERM -- "-$(cat "$QUEUEFILE")" 2>/dev/null || kill -TERM "$(cat "$QUEUEFILE")"
    echo "Đã hủy chuỗi đang xếp hàng (PID $(cat "$QUEUEFILE"))."
  fi
  rm -f "$QUEUEFILE"
  if [ ! -f "$PIDFILE" ] || ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Không có chuỗi nào đang chạy (từ start)."; rm -f "$PIDFILE"; return
  fi
  local pid; pid=$(cat "$PIDFILE")
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
  sleep 3
  kill -0 "$pid" 2>/dev/null && kill -KILL -- "-$pid" 2>/dev/null
  rm -f "$PIDFILE"
  echo "Đã dừng chuỗi PID $pid. Log đang chạy dở sẽ hiện FAILED và được chạy lại lần sau."
}

# ------------------------------------------------------------------ reset
# XÓA HẾT log + kết quả (kể cả log mới). Mặc định nén backup trước vào
# ../old_results_<giờ>.tar.gz ; NO_BACKUP=1 để xóa hẳn.
# Từ chối nếu đang có thí nghiệm chạy. Bỏ qua .nfs* (NFS tự tạo lại).
block_reset () {
  if lock_busy; then
    echo "!! Đang có thí nghiệm chạy - dừng trước: bash code/run_blocks.sh stop"; exit 1
  fi
  banner "RESET: xóa toàn bộ logs/ và results/"
  mkdir -p results
  local n
  n=$(find logs results -mindepth 1 ! -name '.nfs*' ! -name '.run.lock' 2>/dev/null | wc -l)
  echo "  $n file/thư mục sẽ bị xóa"
  if [ "${NO_BACKUP:-0}" != "1" ] && [ "$n" -gt 0 ]; then
    local tar="../old_results_$(date +%Y%m%d_%H%M%S).tar.gz"
    tar czf "$tar" --exclude='.nfs*' --exclude='.run.lock' logs results 2>/dev/null
    echo "  [backup] $tar ($(du -h "$tar" | cut -f1))"
  fi
  find logs results -mindepth 1 -type f ! -name '.nfs*' ! -name '.run.lock' -delete 2>/dev/null
  find logs results -mindepth 1 -depth -type d -empty -delete 2>/dev/null
  mkdir -p logs results
  echo "  [removed] xong. Còn lại: $(find logs results -mindepth 1 ! -name '.run.lock' | wc -l) mục (chỉ .nfs* nếu có)"
}

# ------------------------------------------------------------------ gpu
# Có đang chạy thí nghiệm trên GPU không? Trong container, nvidia-smi thường
# KHÔNG liệt kê process (khác PID namespace), nên kiểm tra 3 cách độc lập:
#   [1] process thí nghiệm nào đang sống, chạy log nào
#   [2] process đó có mở /dev/nvidia* không  -> bằng chứng đang dùng GPU
#   [3] log của nó ghi device=cuda hay cpu, và GPU có đang tải không
block_gpu () {
  banner "GPU CHECK"
  echo "[0] Chuỗi run_blocks:"
  if lock_busy; then
    echo "    ĐANG CHẠY (giữ khóa $LOCK)$( [ -f "$PIDFILE" ] && echo ", PID $(cat "$PIDFILE")")"
  else
    echo "    KHÔNG chạy"
  fi
  echo ""
  echo "[1-3] Process thí nghiệm:"
  local found=0 pid args log dev gpu_fds
  for pid in $(pgrep -f "code/exp_(partition_quality|single_request)\.py" 2>/dev/null); do
    args=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null) || continue
    case "$args" in *python*) ;; *) continue ;; esac
    found=1
    gpu_fds=$(ls -l /proc/$pid/fd 2>/dev/null | grep -c '/dev/nvidia')
    # log = file mà tee cùng nhóm đang ghi
    log=$(for t in $(pgrep -x tee); do
            [ "$(ps -o pgid= -p $t | tr -d ' ')" = "$(ps -o pgid= -p $pid | tr -d ' ')" ] && \
              tr '\0' ' ' < /proc/$t/cmdline | awk '{for (i = 2; i <= NF; i++) if ($i !~ /^-/) { print $i; exit }}'
          done | head -1)
    dev=$( [ -n "$log" ] && grep -m1 -o -e 'device=[a-z]*' -- "$log" < /dev/null 2>/dev/null)
    echo "    PID $pid  chạy $(ps -o etime= -p $pid | tr -d ' ')  log=${log:-?}"
    echo "        dataset/arm : $(echo "$args" | grep -o -- '--dataset [^ ]*') $(echo "$args" | grep -o -- '--n_shards [^ ]*') $(echo "$args" | grep -o -- '--emb_dim [^ ]*') $(echo "$args" | grep -o -- '--unlearn_mode [^ ]*')"
    if [ "$gpu_fds" -gt 0 ]; then
      echo "        /dev/nvidia*: mở $gpu_fds file  -> ĐANG DÙNG GPU"
    else
      echo "        /dev/nvidia*: KHÔNG mở  -> KHÔNG dùng GPU (hoặc chưa khởi tạo CUDA)"
    fi
    echo "        log ghi     : ${dev:-chưa in device}  | đang ở: $( [ -n "$log" ] && grep -o -e '--- ARM [A-Z_]* (seed [0-9]*)' -- "$log" < /dev/null | tail -1)"
  done
  [ "$found" = 0 ] && echo "    Không có process thí nghiệm nào đang chạy."
  echo ""
  echo "[GPU] nvidia-smi (tổng của cả GPU, gồm cả tiến trình ngoài container):"
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" \
      --query-gpu=name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader | sed 's/^/    lúc 1: /'
    sleep 2
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" \
      --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed 's/^/    lúc 2: /'
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
      | sed 's/^/    process: /' | grep . || echo "    process: (nvidia-smi không thấy - bình thường trong container)"
  else
    echo "    nvidia-smi không có"
  fi
}

# ------------------------------------------------------------------ results
# Kết quả đọc thẳng từ log: xong trước in trước; chưa xong thì thông báo
# (đang chạy tới đâu / chưa chạy / dừng giữa chừng). BRIEF=1 để in gọn.
block_results () {
  if [ "${BRIEF:-0}" = "1" ]; then
    python code/show_latest_results.py --brief
  else
    python code/show_latest_results.py
  fi
}

case "${1:-}" in
  main|modes|single|shards|dims|recpart|all|full|fullwmf) take_lock ;;
esac

case "${1:-}" in
  torch)   block_torch ;;
  main)    block_main ;;
  modes)   block_modes ;;
  single)  block_single ;;
  shards)  block_shards ;;
  dims)    block_dims ;;
  recpart) block_recpart ;;
  all)     block_main; block_modes; block_single; block_shards; block_dims; block_recpart; block_csv ;;
  start)   block_start "${2:-all}" ;;
  full)    block_full ;;
  fullwmf) block_fullwmf ;;
  stop)    block_stop ;;
  reset)   block_reset ;;
  results) block_results ;;
  gpu)     block_gpu ;;
  smoke)   block_smoke ;;
  bench)   python code/bench_engine.py --dataset "${BENCH_DATASET:-ml-1m}" ;;
  status)  block_status ;;
  csv)     block_csv ;;
  check)   block_check ;;
  clean)   block_clean ;;
  *) echo "dùng: bash code/run_blocks.sh <reset|start [full]|results|status|gpu|smoke|bench|stop|torch|full|main|modes|single|shards|dims|recpart|all|csv|check|clean>"; exit 1 ;;
esac
