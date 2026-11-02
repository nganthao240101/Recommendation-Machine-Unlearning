# BÁO CÁO KẾT QUẢ THÍ NGHIỆM
## Recommendation Machine Unlearning (BPRMF sharded) — 2 dataset, 3 tiêu chí, "ours"

> File tổng hợp sạch cho paper. 4 method: **Full Retrain · Ours (Deletion-Stable 3-Component) · SISA · RecEraser**.
> 2 dataset: **ml-1m** (6040 user, 3706 item, ~1M tương tác) · **yelp2018** (31668 user, 38048 item, ~1.2M, gầy).
> 3 tiêu chí: **① Utility · ② Performance (thời gian trước/sau unlearn) · ③ Effective Unlearn (ổn định khi xóa)**.
> **⚠️ Nguyên tắc viết báo cáo:** chỉ dùng số **ĐÃ CÓ trong log**. Ô còn thiếu (yelp EX1 Ours/SISA/Rec + yelp C7) đánh **⏳ CHƯA CÓ**, tuyệt đối không bịa số placeholder.

---

# 0. QUY ƯỚC & SETUP (đọc trước)

| Mục | Giá trị |
|-----|---------|
| Model core | BPRMF (user emb + item emb, BPR loss), 8 shards |
| Embedding dim | 64 |
| Local epochs | 50 (EX1) / 100 (Rec WMF-in) |
| Unlearn ratio (EX1) | **1% MOST** (xóa user nhiều tương tác nhất), giữ user còn lại |
| Metric chính | **R@10 trên RETAINED test** (user bị xóa đã loại khỏi data + test) |
| Trước/sau | `Before (FULL test, chưa xóa)` · `After (RETAINED test, sau unlearn + retrain shard)` |
| Device | CUDA (GPU) |
| 2 loại "Rec" (đừng lẫn) | **Rec WMF-in-model** (EX1, K× capacity, RecEraser thật) vs **Rec fair (no WMF-in-model)** (cùng capacity K, random-init, so công bằng) |

---

# ① TIÊU CHÍ 1 — UTILITY (R@10, NDCG@10 sau unlearn, RETAINED)

## EX1 · ml-1m ✅ ĐỦ (4 method)

| Method | Before (FULL) | After (RETAINED) R@10 | NDCG@10 | Ghi |
|--------|:---:|:---:|:---:|-----|
| Full Retrain | 0.1038 | **0.1009** | 0.2556 | trần (retrain cả model) |
| **Ours** | 0.0681 | **0.0765** | 0.1980 | đề xuất |
| SISA | 0.0686 | **0.0751** | 0.1925 | |
| Rec (UBP, WMF-in-model) | 0.0578 | **0.0582** | 0.1605 | Rec "thật" (K× capacity) |

→ **Order utility:** Full (0.1009) > Ours (0.0765) ≈ SISA (0.0751) > Rec WMF-in (0.0582).
→ **Ours dưới Full (trần), ngang SISA, thắng Rec (WMF-in).**

## EX1 · yelp — ⚠️ CHỈ ĐỦ FULL (Ours/SISA/Rec CÒN THIẾU)

| Method | After (RETAINED) R@10 | Trạng thái |
|--------|:---:|---|
| Full Retrain | **0.0158** | ✅ (Unlearn time 3358s) |
| Ours | ⏳ **CHƯA CÓ** | đang chạy |
| SISA | ⏳ **CHƯA CÓ** | đang chạy |
| Rec (UBP, WMF-in) | ⏳ **CHƯA CÓ** | đang chạy |

→ **Không điền số** cho Ours/SISA/Rec yelp (log chưa ra). Khi chạy xong, thay ⏳ bằng số thật.
→ (yelp gầy: Full = 0.0158 là trần, các method dự kiến rất thấp ~0.00x — **chờ số thật**.)

## EX5 · FAIR COMPARE (cùng capacity K, không WMF-in-model, ml-1m, xóa 1% RANDOM) ✅

Cắt 2 "hiệu ứng" tách nhau:

| Config | R@10 | Time(s) | Ý nghĩa |
|--------|:---:|:---:|---|
| C1 Full Single (1 model) | 0.1020 | 285 | trần "không ensemble" |
| C2 Full Ensemble (K model, full data, mean) | 0.1263 | 2020 | trần "nhiều model, data không vụn" |
| C3 SISA (user%K, mean) | 0.0635 | 231 | shard thật |
| C4 Ours (signature, hard-routing) | **0.0673** | 238 | Ours |
| C5 Rec attn (random, no WMF) | 0.0675 | 733 | Rec attention |
| C6 Rec mean (UBP, no WMF) | 0.0689 | 243 | Rec mean |

**Hiệu ứng tách được (đúng yêu cầu "tách ensemble vs sharding"):**
- **ENSEMBLE effect = C2 − C1 = +0.0243** → nhiều model (data FULL) thật sự giúp +2.4%.
- **SHARDING effect = C3 − C2 = −0.0628** → **vụn data 1/K MẤT 6.3% utility**.

→ Lợi thế "nhiều model" của SISA/Ours/Rec **chủ yếu từ ensemble (+0.0243), bị bù trừ bởi sharding (−0.0628)** — **không phải do cơ chế unlearn**.

## C7 · REC MATRIX (InP/UBP/Random × mean/attention, same-capacity, không WMF-in-model)

### (a) fair after-only (train trên data đã xóa) — ml-1m ✅

| Partition | mean | attention | attn − mean | best − SISA(C3) |
|-----------|:---:|:---:|:---:|:---:|
| **InP** | **0.0996** | 0.0983 | −0.0014 | **+0.0362** |
| UBP | 0.0689 | 0.0699 | +0.0010 | +0.0065 |
| Random | 0.0666 | **0.0796** | +0.0130 | +0.0161 |

**Đọc:** ① InP "trần" Rec lên **0.0996 ≈ Full trần (0.1020)** → Rec chỉ "thắng" nhờ **partition InP tốt**, không nhờ unlearn/attention. ② Attention chỉ "cứu" partition DỞ (Random +0.013), **giảm** khi partition tốt (InP −0.0014). ③ Cùng capacity, Rec best (InP) > Ours (C4 0.0673).

### (b) EX1-style (Before→After + unlearn time, cùng tập xóa MOST, no WMF-in-model) — ml-1m ✅

| Config | Before (FULL) | After (RETAINED) | Δ | Unlearn time |
|--------|:---:|:---:|:---:|:---:|
| InP-mean | 0.0640 | 0.0652 | +0.0012 | 357s |
| InP-**attn** | 0.0524 | **0.0481** | −0.0043 | 416s (tệ nhất) |
| UBP-mean | 0.0634 | **0.0665** | +0.0031 | 405s (Rec tốt nhất) |
| UBP-attn | 0.0590 | 0.0618 | +0.0028 | 382s |
| Random-mean | 0.0620 | 0.0634 | +0.0014 | 395s |
| Random-attn | 0.0643 | 0.0651 | +0.0008 | 383s |

**Đọc (so cùng capacity + cùng tập xóa với Ours/SISA):** Ours (0.0765) và SISA (0.0751) **thắng cả 6 config Rec** (Rec tốt nhất = UBP-mean 0.0665). **Attention KHÔNG hơn mean** (InP-attn 0.0481 tệ nhất) → "mean robust hơn khi unlearn" (khác paper gốc: họ đo WMF-init + 1000ep + before).

> ⚠️ **Đừng lẫn 2 bảng Rec:** (a) fair after-only (oracle, RANDOM) cho **InP-mean 0.0996**; (b) EX1-style (MOST, unlearn thật) cho **InP-mean 0.0652**. Hai số đo **2 điều kiện khác nhau**, không so trực tiếp.

---

# ② TIÊU CHÍ 2 — PERFORMANCE (thời gian TRƯỚC & SAU unlearn, ml-1m)

> 2 loại (đừng trộn): **Train time (TRƯỚC, deploy lần đầu)** vs **Unlearn time (SAU, chi phí mỗi lần xóa)**.

| Method | After R@10 (EX1) | Unlearn time (SAU) | Ghi |
|--------|:---:|:---:|-----|
| Full Retrain | 0.1009 | **1532s** | retrain cả model |
| **Ours** | 0.0765 | **779s** | 6/8 shard, đúng scope |
| SISA | 0.0751 | **800s** | 8/8 shard |
| Rec (UBP, WMF-in) | 0.0582 | **678s** | tuyên bố 3/8 — thật effective 8/8 (giấu shard) |
| Rec fair (no WMF) | 0.0665 | ~357–416s | từ bảng C7(b) (6 config) |

→ **Ours ≈ SISA ≈ Rec (678–800s) RẺ HƠN Full (1532s) ~49%** sau unlearn.
→ **Rec "hiện rẻ (678s) nhưng thật đắt"** (giấu shard, instability 84% → chi phí đúng nghĩa ≈ Full). **Ours nói đúng chi phí (naive = effective, stable 0%).**

---

# ③ TIÊU CHÍ 3 — EFFECTIVE UNLEARN (ổn định khi xóa) · EX6 · CẢ 2 DATASET ✅

| Method | ml-1m instability | yelp instability | sim-aware | đúng scope |
|--------|:---:|:---:|:---:|:---:|
| **Ours** | **0.00%** (0 user) | **0.00%** (0 user) | ✅ | ✅ (naive = effective) |
| SISA | 0.00% | 0.00% | ❌ (modulo) | ✅ |
| Rec (UBP) | **84.82%** (5072 user, naive 3→eff 8) | **83.90%** (26305 user) | ✅ | ❌ **giấu shard** |

→ **Ours là method DUY NHẤT vừa similarity-aware VỪA deletion-stable (0% cả 2 dataset) VỪA đúng scope.** Rec: mỗi lần xóa **re-cluster ~84–85% user** + **giấu shard** (nói đổi 3, thật 8).

---

# ABLATION / FACTORS (ml-1m, Ours) ✅

| Yếu tố | Giá trị → After R@10 | Kết luận |
|--------|----------------------|:---:|
| **Shard K** | k4 0.0885 → k8 0.0765 → k16 0.0605 | K lớn → utility giảm (data vụn) |
| **Ratio** (Ours) | 0.1% 0.0727 → 0.5% 0.0754 → 1% 0.0758 → 2% 0.0762 | **FLAT** (mất < 0.5%) |
| **Ratio** (SISA) | 0.1% 0.0696 → 0.5% 0.0731 → 1% 0.0758 → 2% 0.0750 | FLAT |
| **Mode (random)** | Ours 0.0770 / SISA 0.0743 | Ours ≥ SISA |
| **Sig-dim** (Ours only) | d32 0.0795 / d64 0.0765 / d128 0.0765 | dim 64 đủ |

> Sig-dim là hyper-param **riêng của Ours**; Full/SISA/Rec **không có** tham số này.

---

# CODE GỐC TÁC GIẢ (đối chiếu, không dùng làm số chính)

| Dataset | recall@10 (aggregator, early-stop epoch 11) | Ghi |
|---------|:---:|---|
| ml-1m | 0.0030 | chưa converge (bị cắt sớm) |
| yelp | 0.0006 | chưa converge |

→ Số gốc **thấp do early-stop** — chỉ để trả lời "bản port có khớp code gốc không". **Không dùng làm cột Rec chính.**

---

# KẾT LUẬN (đặt vào 3 tiêu chí)

1. **Utility:** Ours **ngang SISA, dưới Full trần, trên Rec (WMF-in)**. Fair-compare chứng minh lợi thế utility của Rec **đến từ partition tốt (InP ≈ Full trần) + ensemble, KHÔNG phải cơ chế unlearn** (sharding effect −0.0628).
2. **Performance:** Ours **rẻ hơn Full ~49%** (779s vs 1532s) sau unlearn; **Rec "hiện rẻ, thật đắt"** (giấu shard); **Ours nói đúng chi phí** (naive = effective).
3. **Effective unlearn (điểm mạnh nhất → đóng góp chính):** Ours **duy nhất sim-aware + deletion-stable (0% cả 2 dataset) + đúng scope**.

## Câu đóng góp (Anh, cho Discussion)

> "Ours is the **only shard-based method that is both similarity-aware and deletion-stable (0% shard flips on both datasets) and reports the true scope of affected shards**; RecEraser (UBP) re-clusters ~84–85% of users on each deletion and leaks affected shards (states 3, actually 8). On utility, Ours matches SISA and trails the full-retrain ceiling; a controlled comparison shows RecEraser's utility advantage comes from **partition choice (InP ≈ full-data ceiling) and ensemble, not from the unlearning mechanism** (sharding effect −0.0628)."

---

# HƯỚNG NGHIÊN CỨU TIẾP (mở, chưa làm)

1. **"Quên thật" (memorization rate):** đo After/Before **trên user bị xóa** (tỷ lệ item của user bị xóa còn bị gợi ý) → đủ tiêu chí ③ thực (hiện chỉ "stable + đúng scope", chưa đo "quên thật").
2. **Attention vs mean (đúng điều kiện):** Rec WMF-in + 200ep (mean/attn) → kiểm tra "attention chỉ có giá khi train đủ + khởi tạo tốt" (giải thích khác biệt với paper gốc).
3. **Scaling theo K/dataset:** EX6 + EX1 trên dataset to hơn (Amazon, CiteULike).
4. **Tối ưu Ours:** chọn K tối ưu (k4/k8/k16); sig-dim 64 đủ.

---

# CÒN THIẾP (⏳, phải điền khi có log)

| Việc | Trạng thái | Lệnh lấy số |
|------|:---:|------|
| **yelp EX1** (Ours/SISA/Rec WMF-in) | ⏳ | `grep -hE "After \(RETAINED\|Unlearn time" logs/y_ex1_ours.log logs/y_ex1_sisa.log logs/y_ex1_rec_ubp.log` |
| **yelp C7 fair matrix** (6 ô) | ⏳ | `grep -hE "RM InP\|RM UBP\|RM Random\|attn-mean\|best-vs-SISA" logs/y_rec_matrix.log` |
| **"Quên thật"** (memorization rate) | ❌ chưa có metric | *tuỳ chọn* |

> **Chỉ khi đủ 2 dòng yelp trên, bảng EX1-yelp và C7-yelp mới hoàn chỉnh. Chưa đủ thì giữ ⏳, KHÔNG điền số ước tính.**
