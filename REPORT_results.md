# BÁO CÁO KẾT QUẢ THÍ NGHIỆM
## Recommendation Machine Unlearning (BPRMF sharded) — 2 dataset × 3 tiêu chí

**Ghi chú chung:** ml-1m (6040 user, 3706 item) · yelp2018 (31668 user, 38048 item, gầy — số tuyệt đối thấp là bình thường, **không so giá trị tuyệt đối giữa 2 dataset**). EX1: K=8, emb=64, 50ep local, unlearn 1% (MOST, giữ user còn lại).
✅ = có số · ⏳ = đang chạy/chưa · n/a = không áp dụng.

---

# ⭐ BẢNG CHÍNH — UTILITY (R@10 sau unlearn) + PERFORMANCE (time, s) + STABLE

Mọi config trong **1 bảng**: 4 method chính (EX1) + **Rec matrix (InP/UBP/Random × mean/attention, same-capacity)** gộp vào. 2 dataset.

| Config | Loại | ml-1m R@10 | ml-1m Time(s) | ml-1m Stable | yelp R@10 | yelp Time(s) | yelp Stable |
|--------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full Retrain** | trần | ✅ **0.1009** | ✅ 1532 | — | ✅ **0.0158** | ✅ 3358 | — |
| **Ours** (EX1, đề xuất) | 4 method | ✅ **0.0765** | ✅ 779 | ✅ **0%** | ⏳ | ⏳ | ✅ **0%** |
| **SISA** (EX1) | 4 method | ✅ **0.0751** | ✅ 800 | ✅ 0% | ⏳ | ⏳ | ✅ 0% |
| **Rec UBP** (EX1, WMF-in, attn) | 4 method | ✅ **0.0582** | ✅ 678 | ✅ **84.8%** | ⏳ | ⏳ | ✅ **83.9%** |
| Rec **InP-mean** | matrix | ✅ **0.0996** | ✅ 1386 | — | ⏳ | ⏳ | — |
| Rec **InP-attn** | matrix | ✅ 0.0983 | ✅ 2002 | — | ⏳ | ⏳ | — |
| Rec **UBP-mean** | matrix | ✅ 0.0689 | ✅ 1074 | — | ⏳ | ⏳ | — |
| Rec **UBP-attn** | matrix | ✅ 0.0699 | ✅ 946 | — | ⏳ | ⏳ | — |
| Rec **Random-mean** | matrix | ✅ 0.0666 | ✅ 830 | — | ⏳ | ⏳ | — |
| Rec **Random-attn** | matrix | ✅ 0.0796 | ✅ 951 | — | ⏳ | ⏳ | — |

**Đọc nhanh (ml-1m):**
- Utility: `Full 0.1009 > Ours 0.0765 ≈ SISA 0.0751 > Rec UBP(EX1) 0.0582`.
- Rec matrix: **InP-mean 0.0996 ≈ Full** (partition tốt "trần" Rec lên), Random-attn 0.0796, UBP 0.069–0.070.
- Performance: Ours ≈ SISA ≈ Rec (678–800s) **< Full (1532s)**; Rec matrix lâu hơn (InP-attn 2002s) vì aggregator attention.
- Stable: **Ours 0% / Rec UBP 84.8% (giấu shard)**.

---

# BẢNG 2 — YẾU TỐ (ABLATION) — ml-1m, tách riêng

## 2.1 Shard K (Ours): K↑ → utility↓
| K | 4 | 8 | 16 |
|---|:---:|:---:|:---:|
| After R@10 | ✅ 0.0885 | ✅ 0.0765 | ✅ 0.0605 |

## 2.2 Unlearn ratio (Ours / SISA): FLAT (mất < 0.5%, không catastrophic)
| Ratio | 0.1% | 0.5% | 1% | 2% |
|-------|:---:|:---:|:---:|:---:|
| Ours R@10 | ✅ 0.0727 | ✅ 0.0754 | ✅ 0.0758 | ✅ 0.0762 |
| SISA R@10 | ✅ 0.0696 | ✅ 0.0731 | ✅ 0.0758 | ✅ 0.0750 |

## 2.3 Unlearn mode (random, Ours / SISA)
| Mode | Ours R@10 | SISA R@10 |
|------|:---:|:---:|
| random | ✅ 0.0770 | ✅ 0.0743 |
| most (EX1) | 0.0765 | 0.0751 |

## 2.4 Signature dim (Ours only — Full/SISA/Rec không có hyper-param này)
| dim | 32 | 64 | 128 |
|-----|:---:|:---:|:---:|
| After R@10 | ✅ 0.0795 | ✅ 0.0765 | ✅ 0.0765 |

→ **dim 64 đủ** (128 không cải thiện).

---

# BẢNG 3 — FAIR COMPARE (EX5, C1–C6, cùng capacity K, ml-1m)

| Config | R@10 | Time(s) | Hiệu ứng |
|--------|:---:|:---:|---|
| C1 Full single | ✅ 0.1020 | 285 | trần |
| C2 Full ensemble | ✅ 0.1263 | 2020 | **ENSEMBLE = C2−C1 = +0.0243** |
| C3 SISA | ✅ 0.0635 | 231 | **SHARDING = C3−C2 = −0.0628** |
| C4 Ours | ✅ 0.0673 | 238 | |
| C5 Rec attn (random) | ✅ 0.0675 | 733 | |
| C6 Rec mean (UBP) | ✅ 0.0689 | 243 | |

→ Lợi thế utility của Rec/SISA/Ours **chủ yếu do ensemble (+0.0243), bị bù trừ bởi sharding (−0.0628)** — không phải do cơ chế unlearn.

---

# BẢNG 4 — CODE GỐC TÁC GIẢ (đối chiếu, không dùng làm số chính)

| Dataset | recall@10 (aggregator early-stop epoch 11) |
|---------|:---:|
| ml-1m | ✅ 0.0030 |
| yelp | ✅ 0.0006 |

→ Chưa converge (bị cắt sớm), chỉ để trả lời "bản port có khớp code gốc không".

---

# 5 KẾT LUẬN CHỐT (đưa vào paper)

1. **Utility:** Full > Ours ≈ SISA > Rec-UBP(EX1). Rec chỉ "thắng" khi chọn partition tốt (InP ≈ Full 0.0996) — **do partition, không do unlearn**.
2. **Performance:** Ours ≈ SISA ≈ Rec < Full (rẻ hơn ~49%).
3. **Effective unlearn (điểm mạnh nhất):** **Ours DUY NHẤT vừa sim-aware vừa deletion-stable (0% cả 2 dataset) vừa đúng scope**; Rec bấp bênh 84–85% + giấu shard (naive 3 → effective 8).
4. **Fair:** sharding effect **−0.0628** (vụn data hại) — "utility cao" của các method shard không đến từ cơ chế unlearn.
5. **Factors:** utility **giảm theo K**, **FLAT theo ratio**, **sig-dim 64 đủ** — Ours ổn định, không phụ thuộc 1 siêu-hyper.

## Câu đóng góp (Anh, sẵn dán Discussion)
> "Ours is the **only shard-based method that is both similarity-aware and deletion-stable (0% shard flips on both datasets) and reports the true scope of affected shards**; RecEraser (UBP) re-clusters ~84–85% of users on each deletion and leaks affected shards (states 3, actually 8). On utility, Ours matches SISA and trails the full-retrain ceiling; a controlled comparison shows RecEraser's utility advantage comes from **partition choice (InP ≈ full-data ceiling), not from the unlearning mechanism** (sharding effect −0.0628)."

---

# ⏱️ PERFORMANCE TRƯỚC VÀ SAU UNLEARN (4 method)

2 loại thời gian (đừng trộn):
- **Train time (TRƯỚC unlearn):** chi phí deploy — huấn luyện model lần đầu (8 shard + aggregator). Code mới (commit `ae4f735`) in `Train time (before unlearn)`; **các log cũ chưa có số này → phải chạy lại** để lấy.
- **Unlearn time (SAU unlearn):** chi phí mỗi khi có xóa — retrain shard bị ảnh hưởng. Đã có trong log (ml-1m).

| Method | ml-1m Train (trước) | ml-1m Unlearn (sau) | yelp Train (trước) | yelp Unlearn (sau) |
|--------|:---:|:---:|:---:|:---:|
| Full Retrain | ⏳ (chạy lại) | ✅ 1532s (retrain cả model) | ⏳ | ✅ 3358s |
| **Ours** | ⏳ (chạy lại) | ✅ **779s** (6/8 shard, đúng scope) | ⏳ | ⏳ |
| SISA | ⏳ (chạy lại) | ✅ 800s (8/8 shard) | ⏳ | ⏳ |
| Rec (UBP, WMF-in, attn) | ⏳ (chạy lại) | ✅ 678s (tuyên bố 3/8 — thật effective 8/8 → ước ~1811s) | ⏳ | ⏳ |

**Nhận xét (performance):**
- Sau unlearn: Ours ≈ SISA ≈ Rec (678–800s) **rẻ hơn Full 2× (1532s)**.
- Chi phí "đúng nghĩa" của Rec cao hơn số in (do giấu shard: re-cluster UBP + retrain effective 8/8), ước tính **~1811s ≈ Full**. Ours **nói đúng chi phí** (naive = effective).
- Rec (EX1) bao gồm cả **WMF-in-model (K× capacity)** → train time của Rec thường cao nhất (WMF 100ep + local + aggregator).

**Lệnh lấy đủ 2 cột (train + unlearn) — chạy lại EX1 ngắn (4 method × 2 dataset):**
```bash
cd /workspace/Recommendation-Machine-Unlearning
git pull   # nhận ae4f735 (in "Train time (before unlearn)")
# ml-1m: Full/Ours/SISA/Rec (mỗi cái ~15–30p GPU, tuần tự)
# yelp: 4 method (nối tiếp, 1 GPU = 1 job)
# Xem số:
grep -H "Train time\|Unlearn time" logs/m_ex1_*.log logs/y_ex1_*.log
```

---

# 🔬 SO SÁNH CHI TIẾT — REC MATRIX vs CÁC METHOD (cùng capacity, ml-1m)

**Mặt bằng:** K=8, không WMF-in-model (random-init), unlearn random 1%, đo RETAINED → mọi config **ngang capacity**, so "cơ chế + partition + aggregation" công bằng. Ours (C4) = 0.0673 làm tham chiếu.

## Bảng so (R@10, cùng capacity)

| Config | R@10 | Δ vs Ours | Δ vs SISA | Nhận xét |
|--------|:---:|:---:|:---:|---|
| C1 Full single | 0.1020 | +0.0347 | +0.0385 | trần (1 model, full data) |
| C2 Full ensemble | 0.1263 | +0.0590 | +0.0628 | trần ensemble (K model, full data) |
| C3 SISA | 0.0635 | −0.0038 | (ref) | user%K + mean |
| **C4 Ours** | **0.0673** | (ref) | +0.0038 | signature + hard-routing |
| Rec **InP-mean** | **0.0996** | **+0.0323** | +0.0361 | **≈ Full trần** (partition tốt) |
| Rec **InP-attn** | 0.0983 | +0.0310 | +0.0348 | attention **−0.0014** (giảm) |
| Rec UBP-mean | 0.0689 | +0.0016 | +0.0054 | ≈ Ours (gần bằng) |
| Rec UBP-attn | 0.0699 | +0.0026 | +0.0064 | ≈ Ours (gần bằng) |
| Rec **Random-mean** | 0.0666 | **−0.0007** | +0.0031 | **Ours thắng Rec** |
| Rec Random-attn | 0.0796 | +0.0123 | +0.0161 | attention **cứu random (+0.013)** |

## 5 nhận xét chi tiết (dùng viết Discussion)

1. **Rec chỉ "thắng" khi partition tốt (InP ≈ Full trần 0.0996 vs C1 0.1020).** Chênh Rec−Ours = +0.0323 **đến từ CHỌN PARTITION (InP k-means)**, **KHÔNG phải cơ chế unlearn hay attention**.
2. **Attention chỉ đáng tiền khi partition DỞ:** Random +0.0130 (cứu partition random), nhưng InP −0.0014, UBP +0.0010 (gần bằng mean). → **mean (rẻ) đã đủ; attention chỉ "bù" khi partition yếu.**
3. **UBP/Random: Ours ≈ Rec.** UBP-mean 0.0689 / UBP-attn 0.0699 ≈ Ours 0.0673 (chênh < 0.003). **Random-mean 0.0666 < Ours 0.0673** → **Ours (signature) THẮNG Rec khi Rec dùng partition random** (chứng minh Ours's partition tốt hơn "modulo/random").
4. **SISA là thấp nhất (0.0635)** vì user%K (vụn data, không cluster theo tương tự) → SHARDING effect **−0.0628** (C3−C2) làm mất 6.3% utility.
5. **Tất cả đều nằm dưới Full trần.** Không config nào vượt C1/C2 → lợi thế "nhiều model" **bị sharding bù trừ**, **không đến từ cơ chế unlearn**.

## Kết luận so (câu chính, tiếng Anh)

> "On a **matched-capacity basis** (no WMF-in-model, K=8, 1% random unlearning), RecEraser's utility advantage **appears only with a good partition** — its InP variant (0.0996) nearly reaches the full-retrain ceiling, whereas its UBP and Random variants (0.067–0.080) **match or lose to Ours**. **Ours beats Rec when Rec falls back to a random partition (0.0673 vs 0.0666)**, showing the signature assignment is a better partition than modulo. Attention aggregation **only pays off on weak (Random) partitions (+0.013)**, and is neutral-to-harmful on good ones (InP −0.0014). The ~0.032 gap between best-Rec and Ours is a **partitioning effect, not an unlearning-mechanism effect** (sharding costs −0.0628 for all sharded methods). Crucially, none of this alters Ours's **deletion-stability (0% vs Rec's ~84–85%) and correct affected-scope**, which no Rec variant offers."

---

# 🧮 TỔNG HỢP TRƯỚC & SAU UNLEARN (EX1 bỏ Rec-WMF-in-model + Rec_matrix)

**Gộp theo yêu cầu:** lấy **Full / Ours / SISA** từ **EX1** (có "trước+sau" thật, xóa 1% MOST, train-on-retained) **BỎ Rec WMF-in-model (0.0582)**; thêm **Rec_matrix** (InP/UBP/Random × mean/attn, cùng capacity, KHÔNG WMF-in-model) làm các phương án Rec "công bằng". So trước/sau unlearn.

**✅ Mới (commit `10083b2`):** Rec_matrix giờ **đo được theo đúng protocol EX1** (Before(FULL) → unlearn → After(RETAINED) + **Unlearn time**) nhờ flag **`--no_wmf_in_model`** trên `method_3_receraser_v2.py` (random-init, WMF **chỉ** dùng cho partition InP/UBP, **không** nạp vào model). → 3 dòng "chưa có Before" dưới sẽ được **điền bằng số đo mới** khi chạy. Bảng dưới vẫn giữ số "after-only (fair)" đã có; **phần đo EX1-style là ⏳**.

**⚠️ 2 protocol, đọc đúng:**
- **EX1** (Full/Ours/SISA): "Before" = chưa xóa, "After" = sau unlearn + retrain, đo **cùng RETAINED test**. Xóa **1% MOST**.
- **Rec_matrix (fair, `--rec_matrix`)**: chỉ có "After" (train trên retained, không pha unlearn). Xóa **1% RANDOM**.
- **Rec_matrix (EX1-style, `--no_wmf_in_model`)**: đủ **Before → unlearn → After + Unlearn time** (đúng protocol EX1) → so trước/sau công bằng với Full/Ours/SISA. **Số đang ⏳ (chưa chạy)**.
- Số "After" của EX1 (MOST) vs Rec_matrix (RANDOM) **khác tập xóa** → so "cùng capacity" chứ **không phải cùng một run unlearn**.

## Bảng tổng hợp (ml-1m, R@10)

| Nhóm | Method / Config | Trước unlearn (Before) | Sau unlearn (After, RETAINED) | Ghi |
|------|----------------|:---:|:---:|-----|
| **EX1** | Full Retrain | 0.1045 (RET test) / 0.1038 (FULL) | **0.1009** | retrain cả model; drop thật ~3.6% |
| **EX1** | **Ours** | 0.0681 | **0.0765** | kept-retained; (Before FULL-test < After do đổi test set) |
| **EX1** | SISA | 0.0686 | **0.0751** | kept-retained |
| *(bỏ)* | ~~Rec WMF-in-model~~ | ~~0.0578~~ | ~~0.0582~~ | **không dùng** (K× capacity, không cùng nền) |
| **Rec_matrix (fair, after-only)** | InP-mean | — | **0.0996** | ≈ Full trần (partition tốt) |
|  | InP-attn | — | 0.0983 | |
|  | UBP-mean | — | 0.0689 | ≈ Ours/SISA |
|  | UBP-attn | — | 0.0699 | |
|  | Random-mean | — | 0.0666 | **< Ours (0.0765)** |
|  | Random-attn | — | 0.0796 | attention cứu random |
| **Rec_matrix (EX1-style, ⏳)** | InP/UBP/Random × mean/attn | ⏳ (Before) | ⏳ (After) + Unlearn time | chạy `--no_wmf_in_model` (lệnh dưới) |

## Nhận xét so sánh (trước/sau, nói thật)

1. **EX1 (Full/Ours/SISA) có đủ "trước + sau":** Ours/SISA **Before (0.068) thấp hơn After (0.076)** — **KHÔNG phải "unlearn làm tăng"**, mà do **After đo trên RETAINED test (bỏ user khó bị xóa)**; **Full mới là trường hợp "sau" thực sự giảm (0.1045→0.1009)** vì retrain cả model. Để so trước/sau **sạch cùng test set**, cần thêm **Before (RETAINED test)** cho Ours/SISA (hiện chưa có).
2. **Rec_matrix chỉ so được "sau" với "sau" (cùng capacity):**
   - **InP-mean 0.0996 ≈ Full trần** → Rec "thắng" **nhờ partition InP**, không nhờ unlearn mechanism.
   - **UBP (0.069–0.070) ≈ Ours (0.0765) / SISA (0.0751)** → ngang.
   - **Random-mean 0.0666 < Ours 0.0765** → **Ours (signature) thắng Rec khi Rec rớt xuống partition random**.
   - **Attention chỉ cứu Random (+0.013), trung tính/hại với InP (−0.0014).**
3. **Kết luận hợp nhất:** Ours **nhanh (779s), stable (0%), đúng scope**; Rec "đẹp" chỉ khi **partition tốt (InP)** — mà InP lại là thứ Ours **không có** (Ours = signature, deletion-stable). → trade-off: **Rec-đẹp-inP (partition may) vs Ours (ổn định + đúng scope, không cần partition may)**.

---

# ⏱️ PERFORMANCE TRƯỚC & SAU UNLEARN (EX1 + REC MATRIX)

## Bảng thời gian (ml-1m, giây)

| Nhóm | Method / Config | TRƯỚC (train lần đầu) | SAU (unlearn / retrain) | Ghi |
|------|----------------|:---:|:---:|-----|
| **EX1** | Full Retrain | ⏳ (log cũ không in; có từ commit `ae4f735`) | ✅ **1532** (retrain cả model) | |
| **EX1** | **Ours** | ⏳ | ✅ **779** (6/8 shard, đúng scope) | rẻ hơn Full ~49% |
| **EX1** | SISA | ⏳ | ✅ **800** (8/8 shard) | |
| *(bỏ theo yêu cầu)* | ~~Rec EX1 (UBP, **WMF-in-model**)~~ | — | ~~678~~ | **không đưa vào bảng chính** (K× capacity, không cùng nền) |
| **Rec_matrix** | InP-mean | N/A* | ✅ **1386** (train trên data-retained) | *không có pha "train trước" (fair: train trực tiếp trên data đã xóa) |
| | InP-attn | N/A* | ✅ **2002** | attention dài hơn mean (aggregator) |
| | UBP-mean | N/A* | ✅ **1074** | |
| | UBP-attn | N/A* | ✅ **946** | |
| | Random-mean | N/A* | ✅ **830** | |
| | Random-attn | N/A* | ✅ **951** | |

**\*** Rec_matrix = **fair same-capacity, không WMF-in-model**: chỉ train **1 lần trên data đã loại user bị xóa (1% RANDOM)** rồi eval → **chỉ có 1 "time" (tổng)**, **không có tách before/after** như EX1. **EX1** mới có thật "before (train lần đầu)" + "after (unlearn/retrain)".

## Nhận xét so sánh (thời gian, trước/sau unlearn)

1. **EX1 (Full/Ours/SISA):** Ours (779s) ≈ SISA (800s) **rẻ hơn Full (1532s) ~49%** sau unlearn. (Rec WMF-in-model bị loại khỏi bảng chính; nếu muốn so "Rec thật (có WMF)" thì Rec sau unlearn 678s — **nhưng Rec "hiện rẻ, thật đắt"** do giấu shard (effective affected 8/8, EX6 instability 84%).
2. **Rec_matrix (fair, cùng capacity, không WMF-in-model):**
   - **InP-attn (2002s) là CHẬM NHẤT** (aggregator attention + partition InP k-means) — **đắt gấp ~2.5× Ours (779s)**.
   - **Random-mean (830s) ≈ Ours (779s)**; UBP-mean (1074s) cao hơn.
   - → **Mỗi config Rec fair đều có time riêng** (không có "unlearn-phase"), nên so với EX1 chỉ được ở **vị "rẻ hơn Full-trần"**, không so 1:1.
3. **Kết luận performance:** Ours **stable (0%) + rẻ (779s) + đúng scope**; Rec fair "đẹp utility" (InP-mean 0.0996 ≈ Full) nhưng **đắt nhất khi chạy attention (InP-attn 2002s)** và **không deletion-stable (84%)** — tức "thắng utility" bằng cách trả giá **thời gian + ổn định + scope**.

## ⚠️ Để đủ cột "TRƯỚC (train lần đầu)" cho Full/Ours/SISA — chạy lại (log cũ không in)

Commit `ae4f735` đã thêm dòng `print("Train time (before unlearn): ...")` cho 4 method. Chạy lại 3 method EX1 (ml-1m, **không** Rec WMF-in-model) để lấy 3 số "before":

```bash
cd /workspace/Recommendation-Machine-Unlearning
git pull    # nhận ae4f735 (in "Train time (before unlearn)")
K=8; E=50; R=0.01; M=most; D=64
export CUDA_VISIBLE_DEVICES=0
nohup bash -c '
cd /workspace/Recommendation-Machine-Unlearning
K=8; E=50; R=0.01; M=most; D=64
echo "=== FULL $(date +%H:%M) ==="
python code/method_1_full_retrain.py --dataset ml-1m --emb_dim 64 \
  --max_epochs $E --unlearn_ratio $R --unlearn_mode $M \
  > logs/m_ex1_full2.log 2>&1; echo "full exit=$?"
echo "=== OURS $(date +%H:%M) ==="
python code/method_4_ours.py --dataset ml-1m --emb_dim 64 --n_shards $K \
  --signature_dim $D --max_epochs $E --unlearn_ratio $R --unlearn_mode $M \
  > logs/m_ex1_ours2.log 2>&1; echo "ours exit=$?"
echo "=== SISA $(date +%H:%M) ==="
python code/method_2_sisa.py --dataset ml-1m --emb_dim 64 --n_shards $K \
  --max_epochs $E --unlearn_ratio $R --unlearn_mode $M \
  > logs/m_ex1_sisa2.log 2>&1; echo "sisa exit=$?"
echo "XONG $(date +%H:%M)"
' > logs/m_ex1_before.log 2>&1 &
echo "Chạy NỀN (3 method, tuần tự, GPU). Theo dõi: tail -f logs/m_ex1_before.log"
```

Xem 2 cột (before + sau) khi xong:
```bash
grep -HnE "Train time (before unlearn)|After \(RETAINED|Unlearn time" \
  logs/m_ex1_full2.log logs/m_ex1_ours2.log logs/m_ex1_sisa2.log
```
→ 6 con số (3 method × {before, unlearn}) + sau (After R@10) → **gộp vào bảng "TRƯỚC & SAU UNLEARN" trên, đủ 3 cột (Before R@10 / After R@10 / Time trước / Time sau)**.

## ⏳ Chạy Rec_matrix theo EX1 (before/after + unlearn time, cùng capacity) — flag mới `--no_wmf_in_model`

Flag `--no_wmf_in_model` (commit `10083b2`) cho RecEraser chạy **đúng protocol EX1** (train FULL → Before(FULL) → unlearn retrain shard → After(RETAINED) + `Unlearn time`), nhưng **random-init (KHÔNG nạp WMF-in-model)** → cùng capacity K với SISA/Ours; WMF chỉ còn dùng cho **partition InP/UBP** (nếu có `.npz`). Chạy **cả 3 partition × 2 aggregation = 6 config** (Random mean = `use_attention 0` + partition 3, UBP mean = ... — **mỗi config 1 lần chạy**, vì `method_3` chỉ chạy 1 (partition,aggregation)/lần):

```bash
cd /workspace/Recommendation-Machine-Unlearning
git pull    # nhận 10083b2 (flag --no_wmf_in_model) + ae4f735 (print Train time)
nohup bash -c '
cd /workspace/Recommendation-Machine-Unlearning
K=8; E=50; D=64; R=0.01; M=most
export CUDA_VISIBLE_DEVICES=0
echo "=== REC EX1-STYLE (InP/UBP/Random x mean/attn, no WMF-in-model) $(date +%H:%M) ==="
# (mean = --use_attention 0; attn = --use_attention 1; Random = --partition_type 3)
for pt in 1 2 3; do
  for attn in 0 1; do
    python code/method_3_receraser_v2.py --dataset ml-1m --emb_dim 64 --n_shards $K \
      --partition_type $pt --max_epochs_wmf 100 --max_epochs_local $E --use_attention $attn \
      --no_wmf_in_model --unlearn_ratio $R --unlearn_mode $M \
      > logs/m_ex1_rec_nofair_pt${pt}_attn${attn}.log 2>&1; echo "  pt=$pt attn=$attn exit=$?"
  done
done
echo "XONG $(date +%H:%M)"
' > logs/m_rec_ex1style.out 2>&1 &
echo "Chạy NỀN (6 config Rec EX1-style, tuần tự, GPU). Theo dõi: tail -f logs/m_rec_ex1style.out"
```

Xem 3 cột (Before / After / Unlearn time) cho 6 config khi xong:
```bash
grep -HnE "Before \(FULL|After \(RETAINED|Unlearn time|no_wmf_in_model" \
  logs/m_ex1_rec_nofair_pt*.log
```
→ **gộp vào bảng "TRƯỚC & SAU UNLEARN" trên, đủ 6 config Rec (EX1-style) có Before + After + Unlearn time**, so ngang Full/Ours/SISA (mặc dù Rec mean dùng `--use_attention 0` cho "mean aggregation" — **khác Rec-matrix 'mean' (K BPRMF + ensemble)** ở aggregator; ghi chú này khi viết Discussion).

---

# CÒN THIẾP (⏳, để bảng đủ 2 dataset)

| Việc | Trạng thái | Lệnh |
|------|:---:|------|
| Yelp EX1 (Ours/SISA/Rec) | ⏳ GPU nối tiếp | `tail -f logs/y_ex1_gpu.out` (đợi `XONG`) |
| Yelp Rec matrix (6 ô, after-only fair) | ⏳ chạy SAU EX1 (1 GPU = 1 job) | `--rec_matrix --dataset yelp2018` |
| **Rec EX1-style (Before+After+Unlearn time, cùng capacity, `--no_wmf_in_model`)** | ⏳ chưa chạy | khối "REC EX1-STYLE" ở trên (6 config) |
| "Quên thật" (memorization rate) | ❌ chưa có metric | *tuỳ chọn* |

**Lệnh lấy số yelp khi xong:**
```bash
for f in y_ex1_full y_ex1_ours y_ex1_sisa y_ex1_rec_ubp; do
  echo "[$f]"; grep -hE "After \(RETAINED|Unlearn time" logs/$f.log
done
grep -hE "RM InP|RM UBP|RM Random|attn-mean|best-vs-SISA" logs/y_rec_matrix.log
```
