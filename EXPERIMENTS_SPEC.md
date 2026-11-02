# SPEC THÍ NGHIỆM — Recommendation Machine Unlearning
## Mỗi thí nghiệm: chạy gì · tham số · đầu ra · cách đánh giá

4 method: **Full · Ours · SISA · Rec**. 2 dataset: **ml-1m** · **yelp2018**.
Protocol chung (EX1): train FULL → **Before (FULL test)** → unlearn (retrain shard bị ảnh hưởng) → **After (RETAINED test, đã loại user xóa)** + **Unlearn time**.
**lr=0.05 cho MỌI method (fair)**; Rec thêm **`--lr_finetune 0.05`**. K=8, emb=64, 50ep, xóa 1% MOST.

---

## EX1 — 4 METHOD (utility + performance, cả 2 dataset)

### EX1.1 Full Retrain — `method_1_full_retrain.py`
- **Làm gì:** 1 BPRMF full-data. Unlearn = **retrain CẢ model** trên data retained.
- **Tham số:** `--dataset {ml-1m|yelp2018} --emb_dim 64 --max_epochs 50 --unlearn_ratio 0.01 --unlearn_mode most --learning_rate 0.05`
- **Đầu ra (log):** `Before (FULL) R@10` · `Before (RETAINED) R@10` · `After (RETAINED, retrain) R@10` · `Unlearn time` · (json).
- **Đánh giá:** R@10/after = **trần utility** (retrain cả model → cao nhất). Unlearn time = **đắt nhất** (chi phí reference).

### EX1.2 Ours — `method_4_ours.py`
- **Làm gì:** K BPRMF, partition **signature (deletion-stable, 0% flip)**, inference **hard-routing** (default). Unlearn = retrain chỉ shard bị ảnh hưởng.
- **Tham số:** `--dataset ... --n_shards 8 --signature_dim 64 --max_epochs 50 --unlearn_ratio 0.01 --unlearn_mode most --learning_rate 0.05`
- **Đầu ra:** `Before (FULL)` · `After (RETAINED)` · `Unlearn time` · json.
- **Đánh giá:** utility ≈ SISA (dưới trần); unlearn time **rẻ hơn Full ~57%**; **stable 0% + đúng scope** (điểm mới, đo ở EX6).

### EX1.3 SISA — `method_2_sisa.py`
- **Làm gì:** K BPRMF, partition **modulo `user % K`** (không similarity-aware). Unlearn = retrain shard chứa user xóa (modulo → **mọi shard**, 8/8).
- **Tham số:** `--dataset ... --n_shards 8 --max_epochs 50 --unlearn_ratio 0.01 --unlearn_mode most --learning_rate 0.05`
- **Đầu ra:** `Before (FULL)` · `After (RETAINED)` · `Unlearn time` · json.
- **Đánh giá:** utility ≈ Ours; **stable 0% (modulo cố định) nhưng KHÔNG sim-aware**; scope đúng (8/8 thật, nói đúng 8/8).

### EX1.4 Rec (UBP, WMF-in-model, attention) — `method_3_receraser_v2.py`
- **Làm gì:** RecEraser "thật" (K× capacity, **nạp WMF vào model**). Partition UBP (k-means). Aggregator **attention**.
- **Tham số:** `--dataset ... --n_shards 8 --partition_type 2 --max_epochs_wmf 100 --max_epochs_local 50 --use_attention 1 --lr_finetune 0.05 --unlearn_ratio 0.01 --unlearn_mode most`
- **Đầu ra:** `Before (FULL)` · `After (RETAINED)` · `Unlearn time` · json.
- **Đánh giá:** Rec "thật" (extra WMF → **không cùng capacity** với Ours/SISA). Utility thấp hơn Ours (WMF-in chưa fair); **giấu shard (84% instability, EX6)**.

> **Command EX1 (8 runs, 1 GPU, nền):**
> ```bash
> cd /workspace/Recommendation-Machine-Unlearning
> nohup bash -c '
> cd /workspace/Recommendation-Machine-Unlearning
> K=8;E=50;R=0.01;M=most;export CUDA_VISIBLE_DEVICES=0
> for ds in ml-1m yelp2018; do
>   echo "=== $ds $(date +%H:%M) ==="
>   python code/method_1_full_retrain.py --dataset $ds --emb_dim 64 --max_epochs $E \
>     --unlearn_ratio $R --unlearn_mode $M > logs/A_${ds}_full.log 2>&1
>   python code/method_4_ours.py --dataset $ds --emb_dim 64 --n_shards $K \
>     --signature_dim 64 --max_epochs $E --unlearn_ratio $R --unlearn_mode $M \
>     > logs/A_${ds}_ours.log 2>&1
>   python code/method_2_sisa.py --dataset $ds --emb_dim 64 --n_shards $K \
>     --max_epochs $E --unlearn_ratio $R --unlearn_mode $M > logs/A_${ds}_sisa.log 2>&1
>   python code/method_3_receraser_v2.py --dataset $ds --emb_dim 64 --n_shards $K \
>     --partition_type 2 --max_epochs_wmf 100 --max_epochs_local $E --use_attention 1 \
>     --lr_finetune 0.05 --unlearn_ratio $R --unlearn_mode $M > logs/A_${ds}_rec.log 2>&1
> done
> echo "EX1 XONG $(date +%H:%M)"
> ' > logs/EX1.out 2>&1 &
> ```
> **Đọc:** `grep -HnE "Before \(FULL|After \(RETAINED|Unlearn time" logs/A_*.log`

---

## EX5 — FAIR COMPARE (C1–C6, tách ENSEMBLE vs SHARDING)

### `benchmark_fair_compare.py` (không WMF-in-model, cùng capacity K)
- **C1** Full single (1 model, full data) · **C2** Full ensemble (K model, full data, mean)
- **C3** SISA (user%K, mean) · **C4** Ours (signature, hard) · **C5** Rec attn (random, no WMF) · **C6** Rec mean (UBP, no WMF)
- **Tham số:** `--dataset ... --emb_dim 64 --n_shards 8 --max_epochs 50 --unlearn_ratio 0.01 --unlearn_mode random`
- **Đầu ra:** mỗi C: `Cn R@10=... time=...s` + 2 dòng **`ENSEMBLE = C2−C1`** / **`SHARDING = C3−C2`**.
- **Đánh giá:** chứng minh lợi thế "nhiều model" **chủ yếu ensemble (+0.024), bị sharding bù (−0.063)** → không phải cơ chế unlearn.

> **Command:** `python code/benchmark_fair_compare.py --dataset ml-1m --emb_dim 64 --n_shards 8 --max_epochs 50 --unlearn_ratio 0.01 --unlearn_mode random`
> (yelp: đổi `--dataset yelp2018`). **Đọc:** `grep -hE "C[1-6] .*R@10|ENSEMBLE|SHARDING" logs/EX5_ml.log`

---

## REC-12 — REC MATRIX (3 partition × 2 WMF × 2 agg), cùng capacity, lr=0.05

### `method_3_receraser_v2.py` × 12 config
- **Biến (3×2×2):**
  - `--partition_type 1/2/3` = **InP / UBP / Random**
  - WMF: có flag `--no_wmf_in_model` = **no-WMF (FAIR, cùng capacity)**; KHÔNG flag = **WMF-in (Rec thật, không fair)**
  - `--use_attention 0/1` = **mean / attention**
- **Tham số (cố định):** `--emb_dim 64 --n_shards 8 --max_epochs_wmf 100 --max_epochs_local 50 --lr_finetune 0.05 --unlearn_ratio 0.01 --unlearn_mode most`
- **Đầu ra:** `Before (FULL)` · `After (RETAINED)` · `Unlearn time` (mỗi config 1 log).
- **Đánh giá:**
  - **6 ô no-WMF** so trực tiếp Ours/SISA (cùng capacity).
  - So **`(attn − mean)`** giữa **WMF-in vs no-WMF** theo từng partition → trả lời *"WMF-in có làm attention cao hơn mean không?"*
  - **InP "trần" Rec** (≈ full) mà UBP/Random ≈ Ours → **lợi Rec = partition, không = unlearn**.

> **Command (12 config, ml-1m):**
> ```bash
> cd /workspace/Recommendation-Machine-Unlearning
> nohup bash -c '
> cd /workspace/Recommendation-Machine-Unlearning
> K=8;E=50;R=0.01;M=most;export CUDA_VISIBLE_DEVICES=0
> for ds in ml-1m yelp2018; do
>   for pt in 1 2 3; do for wmf in nw wmf; do for attn in 0 1; do
>     flag=""; [ "$wmf" = "nw" ] && flag="--no_wmf_in_model"
>     echo "=== $ds pt=$pt $wmf attn=$attn $(date +%H:%M) ==="
>     python code/method_3_receraser_v2.py --dataset $ds --emb_dim 64 --n_shards $K \
>       --partition_type $pt --max_epochs_wmf 100 --max_epochs_local $E \
>       --use_attention $attn $flag --lr_finetune 0.05 \
>       --unlearn_ratio $R --unlearn_mode $M \
>       > logs/B_${ds}_pt${pt}_${wmf}_attn${attn}.log 2>&1
>   done; done; done
> done
> echo "REC-12 XONG $(date +%H:%M)"
> ' > logs/REC12.out 2>&1 &
> ```
> **Đọc (6 ô FAIR = `nw`):**
> ```bash
> for ds in ml-1m yelp2018; do for pt in 1 2 3; do for a in 0 1; do
>   f="logs/B_${ds}_pt${pt}_nw_attn${a}.log"
>   printf "%s pt=%s attn=%s : " "$ds" "$pt" "$a"
>   grep -hE "After \(RETAINED" "$f" 2>/dev/null | grep -oE "R@10: [0-9.]+" || echo "(thiếu)"
> done; done; done
> ```

---

## EX6 — STABILITY (delete-stable, đúng scope) — `measure_stability.py`

> ⚠️ **KHÔNG phải `--ex6`** (flag không tồn tại). Đây là **file riêng, không train** (chỉ tính partition, chạy vài phút).

- **Làm gì:** với mỗi method, partition **trên FULL** (A_full) rồi **re-partition trên RETAINED** (A_ret) → đếm user retained **đổi shard**.
- **Tham số:** `--dataset ... --emb_dim 64 --n_shards 8 --signature_dim 64 --unlearn_ratio 0.01 --unlearn_mode most`
- **Đầu ra (mỗi method):** `similarity-aware` · `deletion-stable` · **`Instability%`** (user retained đổi shard) · **`Naive / Effective affected shards`** · shard load min/max. + bảng tổng hợp + json.
- **Đánh giá:**
  - **Ours:** sim-aware ✅ + deletion-stable ✅ → **Instability ≈ 0%, naive = effective (đúng scope).**
  - **SISA:** không sim-aware, modulo cố định → Instability 0%, đúng scope.
  - **Rec UBP:** sim-aware ✅ nhưng **re-cluster khi xóa** → **Instability ~84–85%**, **naive < effective (giấu shard)**.
  - → **Ours duy nhất đạt "sim-aware + deletion-stable + đúng scope" cả 3.**

> **Command:** `python code/measure_stability.py --dataset ml-1m --n_shards 8 --signature_dim 64 --unlearn_ratio 0.01 --unlearn_mode most`
> (yelp: đổi dataset). **Đọc:** `grep -HnE "similarity-aware|deletion-stable|Instability|Naive|Effective" logs/EX6_ml.log`

---

## OURS-MEAN — đòn bẩy utility (giữ stable 0%)

- **Làm gì:** Ours **partition không đổi** (stable 0%, đúng scope), chỉ đổi **aggregation = mean** (`--use_mean 1`) → tận dụng đủ K model (nên ≥ hard-routing).
- **Tham số:** `python code/method_4_ours.py --dataset ... --n_shards 8 --signature_dim 64 --max_epochs 50 --unlearn_ratio 0.01 --unlearn_mode most --use_mean 1`
- **Đầu ra:** `After (RETAINED) R@10` + `Unlearn time`.
- **Đánh giá:** nếu **Ours-mean ≈ Rec InP-mean (0.105) mà vẫn stable 0% + đúng scope** → **"Ours-mean: utility bắt kịp Rec, ổn định hơn hẳn"** (câu chuyện cải tiến chính).

---

## REC-UNBAL — partition unbalanced (quick win, no-WMF mean)

- **Làm gì:** `--partition_type 4` (UBP-unbal) / `5` (InP-unbal) — k-means **không ép cân bằng** → shard "đầy data" hơn → utility cao hơn (log cũ UBP-unbal ≈ 0.1092).
- **Tham số:** `python code/method_3_receraser_v2.py --dataset ml-1m --n_shards 8 --partition_type 4/5 --max_epochs_wmf 100 --max_epochs_local 50 --use_attention 0 --no_wmf_in_model --lr_finetune 0.05 --unlearn_ratio 0.01 --unlearn_mode most`
- **Đầu ra:** `After (RETAINED)` + `shard sizes min/max` (để thấy không cân bằng) + `Unlearn time`.
- **Đánh giá:** so utility vs partition cân bằng (InP-mean 0.1053); ghi rõ "unbalanced → utility cao hơn nhưng shard lệch, scope rộng hơn".

---

## ABLATION — Ours (4 yếu tố, ml-1m)

`method_4_ours.py`, đổi 1 tham số mỗi lần (K / ratio / mode / sig-dim), còn lại cố định (E=50, R=0.01, M=most, D=64):

| Yếu tố | Điểm chạy | Lệnh (khác nhau) | Kết luận mong đợi |
|--------|-----------|-------------------|-------------------|
| **Shard K** | 4 / 8 / 16 | `--n_shards {4,8,16}` | K↑ → utility↓ (data vụn) |
| **Ratio** | 0.001 / 0.005 / 0.01 / 0.02 | `--unlearn_ratio {..}` | **FLAT** (không catastrophic) |
| **Mode** | most / fewest / random | `--unlearn_mode {..}` | Ours ổn mọi mode |
| **Sig-dim** | 32 / 64 / 128 | `--signature_dim {..}` | **dim 64 đủ** (128 không cải thiện) |

- **Đầu ra mỗi run:** `After (RETAINED) R@10` + `Unlearn time`.
- **Đánh giá:** chứng minh Ours **robust theo 4 hướng** (không phụ thuộc 1 siêu-hyper).

---

# BẢNG TỔNG HỢP: MỤC ĐÍCH MỖI THÍ NGHIỆM (để "thấy điểm mới")

| Thí nghiệm | Trả lời câu hỏi | Số liệu chính |
|------------|-----------------|---------------|
| **EX1** (4 method × 2 ds) | Utility + performance trước/sau unlearn | After R@10 + Unlearn time (Full=trần+đắt; Ours≈SISA+giảm; Rec thấp+giấu) |
| **EX5** (C1–C6) | "Nhiều model" giúp thật không? Tách ensemble vs sharding | ENSEMBLE=+0.024 · SHARDING=−0.063 |
| **REC-12** (3×2×2) | Rec thắng do partition hay unlearn? WMF-in giúp attention? | 6 ô no-WMF (fair) + attn−mean (WMF vs no-WMF, theo InP/UBP/Random) |
| **EX6** (stability) | Ai stable + đúng scope khi xóa? | Instability% (Ours 0% · Rec ~84%) + naive/effective affected |
| **OURS-MEAN** | Cải tiến Ours để ≥ Rec mà giữ stable? | Ours-mean After R@10 (so Rec InP-mean 0.105, stable 0%) |
| **REC-UNBAL** | Partition unbalanced có giúp utility? | After R@10 + shard sizes |
| **ABLATION** | Ours robust theo hyper không? | K / ratio / mode / sig-dim → After R@10 |

**Điểm mới (nổi bật nhất, đọc từ EX6 + EX5):**
> Ours **không bán utility cao** (ngang SISA, dưới trần) — **bán TIÊU CHÍ ③: unique "similarity-aware + deletion-stable (0%) + đúng scope"**, ở chi phí unlearn **rẻ hơn Full ~57%**. Rec "thắng utility" chỉ nhờ **partition InP (≈ trần, EX5/REC-12)**, không nhờ cơ chế unlearn, **trả giá bằng 84% instability + giấu shard**.
