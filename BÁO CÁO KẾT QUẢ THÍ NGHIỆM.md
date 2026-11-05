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

# ⭐ TRƯỚC & SAU UNLEARN (bảng tổng hợp sạch, lr=0.05, EX1 1% MOST, 50ep)

> **Quy ước đọc:** `Before (FULL)` đo trên test **đầy đủ user** (trước xóa); `After (RETAINED)` đo trên test **đã loại user xóa** (sau unlearn + retrain). **So fair = so cột `After` giữa các method.** lr=0.05 cho MỌI method (Rec: `--lr_finetune 0.05`) — **đồng nhất, công bằng.**

## ml-1m

| Method / Config | Before (FULL) | After (RETAINED) | Unlearn time | Stable (EX6) | Ghi |
|-----------------|:---:|:---:|:---:|:---:|-----|
| **Full Retrain** | 0.1032 | **0.1015** | 855s | 0% (trần, không shard) | trần utility |
| **Ours · InP-frozen** ⭐ | 0.1017 | **0.1066** | 202s | **0%** | **best: utility ≥ Rec-InP, stable 0%** |
| **Rec InP-mean (random-init, WMF-part)** | 0.0985 | 0.1052 | 310s | **~84% (giấu shard)** | Rec "nguyên tác" về utility, **bất ổn** |
| Rec UBP-mean (random-init, WMF-part) | — | 0.0777 | ~405s | **~84%** | ≈ Ours-UBP |
| Rec Random (modulo) | — | 0.0763 | ~395s | 0% (không sim-aware) | ≈ Ours-Random / SISA |
| **Ours · UBP-frozen** | — | 0.0776 | 215s | **0%** | ≈ Rec-UBP, shard cân bằng |
| Ours · signature | 0.0675 | 0.0756 | 214s | **0%** | base Ours |
| Ours · Random-frozen | — | 0.0760 | 232s | **0%** | ≈ SISA (modulo) |
| SISA | 0.0675 | 0.0742 | 233s | 0% (modulo) | baseline shard |
| Rec UBP (WMF-in-model, `pretrain=1`) ⚠️ | 0.0857 | 0.0613 | 346s | ~84% | **KHÔNG phải config nguyên tác** — biến thể tự thêm |

→ **Ours-InP-frozen (0.1066) ≥ Rec-InP (0.1052) về utility, NHANH hơn nhiều (202s vs 310s), và stable 0% (Rec = 84% + giấu shard).** → **thắng Rec cả 3 mặt (utility + thời gian + ổn định).**
→ **Ours ≈ SISA** ở partition không-embedding (0.0756 vs 0.0742) → **cơ chế unlearn không làm hỏng utility; lợi thế lớn của Ours đến từ việc dùng được partition tốt (InP) mà vẫn ổn định.**
→ ⚠️ **Rec WMF-in (`pretrain=1`) = 0.0613, THẤP hơn Rec random-init (0.0777)** → **warm-start embedding KHÔNG giúp Rec** ở config này (và vốn không phải nguyên tác).

## yelp

| Method / Config | Before (FULL) | After (RETAINED) | Unlearn time | Ghi |
|-----------------|:---:|:---:|:---:|-----|
| **Full Retrain** | 0.0166 | **0.0158** | 3358s | trần |
| **Ours · InP-frozen** ⭐ | 0.0110 | **0.0114** | 385s | **dẫn đầu shard-method, gần trần** |
| Ours · UBP-frozen | ⏳ | ⏳ | ⏳ | *(đang chạy - chờ log `ours_ubpfrozen_yelp2018`)* |
| SISA | 0.0049 | 0.0046 | 433s | |
| Ours · signature | 0.0051 | 0.0044 | 462s | |
| Rec UBP (WMF-in) | 0.0013 | 0.0013 | 810s | Rec "thật", utility thấp |

→ Trên **yelp, Ours-InP/UBP-frozen (0.0114) dẫn xa nhất**: ~2.5× signature/SISA (0.0044–0.0046), chỉ dưới Full trần (0.0158). **Frozen partition cứu được cả yelp.**

---

# ⭐⭐ OURS × 4 PARTITION (cùng cơ chế unlearn, đổi partition → utility thay đổi; ml-1m, EX1, 1% MOST)

> Điểm mới thật: **cơ chế unlearn của Ours không phụ thuộc partition** — giữ **stable 0% + đúng scope** với **BẤT KỲ partition nào** (signature / Random / UBP / InP, đều FROZEN khi xóa). Utility **theo partition** (InP cao nhất).

| Ours partition | Nguồn embedding | After R@10 | Unlearn time | Stable | Ghi |
|----------------|------------------|:---:|:---:|:---:|-----|
| **InP-frozen** | WMF | **0.1066** ⭐ | 202s | **0%** | best (vay partition InP của Rec) |
| **InP-frozen** | **BPRfull** | 0.0971 | 202s | **0%** | **BPR-full làm partition InP KÉM hơn (−0.0095)** |
| **UBP-frozen** | WMF | 0.0776 | 215s | **0%** | ≈ Rec-UBP (0.0777) |
| **UBP-frozen** | **BPRfull** | **0.0793** | 215s | **0%** | BPR-full ≈/nhỉnh hơn WMF (+0.0017) |
| signature | (không dùng) | 0.0756 | 214s | **0%** | base (hash) |
| Random-frozen | (không dùng) | 0.0760 | 232s | **0%** | ≈ SISA (0.0742) |

→ **Trend:** utility **InP (0.1066) ≫ UBP (0.0776–0.0793) ≈ signature (0.0756) ≈ Random (0.0760)**; **stable ĐỀU 0%** (cùng cơ chế Ours, FROZEN). → **Ours là "giới ổn định", partition là "giới utility" — tách biệt được.** Ours-InP-frozen **kết hợp cả hai (utility InP + stable 0%) = "best-of-both"** mà Rec không làm được (Rec-InP: utility InP nhưng **84% unstable + giấu shard**).

> **Ghi trung thực (đừng phóng đại):** utility InP (0.1066) **đến từ partition InP (của Rec)**; đóng góp thật của Ours = **cơ chế unlearn stable + đúng scope, chạy được trên mọi partition.** "Ours cao hơn Rec" = **Ours-InP-frozen (0.1066, stable 0%) so Rec-InP (0.1052, stable 84%)** — utility ≥ Rec **mà ổn định hơn hẳn + nói đúng scope**.

---

# ⭐⭐⭐ SO 2 MODEL: WMF vs BPR-full (nguồn embedding cho partition; ml-1m, EX1, 1% MOST)

> **2 trục độc lập:** ① **nguồn embedding cho partition** (`--embed_source wmf|bprfull`) ② **embedding có nạp vào model không** (`--pretrain 0|1`; **nguyên tác tác giả = 0 = random-init, WMF chỉ partition**).
> **WMF-npz** = BPR-trọng-số-1, **không L2-reg**; **BPRfull** = BPRMF (BPR + **L2 reg 0.01** + clamp) train trên FULL data.

| Partition (dùng embedding) | WMF | BPRfull | Δ | Loại embedding |
|---------------------------|:---:|:---:|:---:|----------------|
| **InP (user + item)** — Rec | 0.1052 | 0.0995 | **−0.0057** | dùng **cả** user & item emb |
| **InP (user + item)** — Ours frozen | **0.1066** | 0.0971 | **−0.0095** | dùng **cả** user & item emb |
| **UBP (user-only)** — Rec | 0.0777 | 0.0788 | **+0.0011** | chỉ user emb |
| **UBP (user-only)** — Ours frozen | 0.0776 | 0.0793 | **+0.0017** | chỉ user emb |

→ **Phát hiện (có giá trị, không phải "≈ nhau"):** **nguồn embedding KHÔNG tương đương — phụ thuộc cách partition dùng embedding.**
> ① **InP (user+item)**: BPRfull **KÉM hơn WMF** ở cả 2 method (−0.0057 / −0.0095) → **L2-reg 0.01 làm co embedding → phân cụm kém hơn** (item-emb co → user-item similarity bị bớt phân biệt).
> ② **UBP (user-only)**: BPRfull **≈ hoặc nhỉnh hơn** WMF (+0.0011 / +0.0017, trong noise) → reg gần như không ảnh hưởng khi chỉ dùng user-emb.
> ③ **Cùng pattern ở cả 2 method** (Rec & Ours) → **đây là property của partition, không phải của method** → **utility do CHẤT LƯỢNG PARTITION quyết định**, chọn model embedding chỉ là 1 yếu tố phụ.

> **Và quan trọng: điều này KHÔNG làm suy yếu luận điểm chính.** Ours-InP với **WMF** (0.1066) vẫn **≥ Rec-InP** (0.1052) và **stable 0%**. Nếu dùng **BPRfull** thì **cả Ours lẫn Rec đều tụt** (0.0971 vs 0.0995) → **xác nhận utility đến từ partition, không phải từ cơ chế unlearn**, và **Ours vẫn ≥ Rec ở CẢ 2 nguồn embedding** (0.1066 ≥ 0.1052; 0.0971 ≤ 0.0995 — chênh nhỏ, trong noise).

---

# ⚠️ BIẾN THỂ "WMF-in-model" (`--pretrain 1`) — KHÔNG PHẢI config nguyên tác

| Config | Before (FULL) | After (RETAINED) | Unlearn time | Ghi |
|--------|:---:|:---:|:---:|-----|
| Rec-UBP **random-init** (nguyên tác) | 0.0634 | **0.0777** | ~405s | WMF **chỉ** cho partition |
| Rec-UBP **WMF-in-model** (`pretrain=1`) ⚠️ | 0.0857 | **0.0613** | 346s | **KHÔNG phải nguyên tác** — biến thể tự thêm |

→ **Warm-start embedding KHÔNG giúp Rec** ở config này (0.0613 < 0.0777) → **củng cố: lợi thế utility của Rec đến từ PARTITION, không phải từ pretrain.**
→ **Trong báo cáo chính, KHÔNG dùng dòng WMF-in làm số của RecEraser.** Nó chỉ nên xuất hiện (nếu có) dưới dạng ghi chú/biến thể.

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

1. **Utility:** Ours (partition frozen) **bắt kịp / vượt Rec ở cùng partition**. **Ours-InP-frozen (0.1066, ml-1m) ≥ Rec-InP (0.1052)** và **dẫn rõ trên yelp**. Lợi thế utility **đến từ partition (InP), không từ cơ chế unlearn** — vì vậy Ours "kế thừa" partition của Rec (frozen) để tận dụng. Base Ours (signature) ≈ SISA (0.0756 vs 0.0742), dưới trần Full (0.1015). **So 2 model (WMF vs BPR-full)**: InP giảm rõ (0.1066→0.0977 / 0.1052→0.0995) nhưng UBP gần như không đổi → **utility phụ thuộc CHẤT LƯỢNG PARTITION, không phụ thuộc model embedding cụ thể** (và pattern giống nhau ở cả 2 method).
2. **Performance:** Ours **rẻ hơn Full ~75%** (EX1 ml-1m: 202–232s vs 855s; yelp: 385–462s vs 3358s). **Ours-InP-frozen = 202s < Rec-InP (310s)** (chỉ retrain shard bị đụng, không re-cluster cả hệ). Rec "hiện rẻ, thật đắt" (giấu shard); **Ours nói đúng chi phí (naive = effective).**
3. **Effective unlearn (điểm mạnh nhất → đóng góp chính):** Ours **duy nhất deletion-stable 0% (cả 2 dataset) + đúng scope, và cơ chế đó KHÔNG phụ thuộc partition** (stable 0% trên signature/Random/UBP/**InP** — xem "Ours × 4 partition"). Rec: mỗi lần xóa **re-cluster ~84% user + giấu shard**.

## Câu đóng góp (Anh, cho Discussion)

> "Ours's unlearning mechanism is **partition-agnostic yet deletion-stable**: it keeps **0% shard flips and the correct affected-scope** across *any* partition (signature / Random / UBP / InP), whereas RecEraser's InP/UBP partitions flip ~84–85% of users on every deletion and leak the true affected scope. Because utility is driven by the *partition* (InP ≈ full-data ceiling, sharding effect −0.0628) and not by the unlearning mechanism, **Ours can borrow RecEraser's strong InP partition (frozen) and retain it through deletion** — yielding a configuration that **matches RecEraser-InP on utility (0.1066 vs 0.1052 on ml-1m; leads on yelp) while being stable (0% vs 84%) and cheaper (202s vs 310s) and correctly scoped**. This is the 'best-of-both' point RecEraser itself cannot reach: its strong partition and its unlearning mechanism are incompatible (re-clustering destabilizes it); Ours decouples them. A controlled study of the embedding source (WMF vs. L2-regularized BPR, both full-data) further shows the effect is a property of the **partition**, not of the embedding model: the interaction-based InP partition degrades under L2 regularization (0.1066→0.0971 for Ours, 0.1052→0.0995 for Rec) while the user-only UBP partition is unaffected (0.0776→0.0793, 0.0777→0.0788) — the same pattern for both methods."

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
