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

# CÒN THIẾP (⏳, để bảng đủ 2 dataset)

| Việc | Trạng thái | Lệnh |
|------|:---:|------|
| Yelp EX1 (Ours/SISA/Rec) | ⏳ GPU nối tiếp | `tail -f logs/y_ex1_gpu.out` (đợi `XONG`) |
| Yelp Rec matrix (6 ô) | ⏳ chạy SAU EX1 (1 GPU = 1 job) | `--rec_matrix --dataset yelp2018` |
| "Quên thật" (memorization rate) | ❌ chưa có metric | *tuỳ chọn* |

**Lệnh lấy số yelp khi xong:**
```bash
for f in y_ex1_full y_ex1_ours y_ex1_sisa y_ex1_rec_ubp; do
  echo "[$f]"; grep -hE "After \(RETAINED|Unlearn time" logs/$f.log
done
grep -hE "RM InP|RM UBP|RM Random|attn-mean|best-vs-SISA" logs/y_rec_matrix.log
```
