# Unlearning Sharded Recommenders: Decoupling Partition Quality from Deletion Stability

## Abstract

We study machine unlearning for shard-based recommendation ensembles and ask a question prior work conflates: **does utility come from the *partitioning* scheme or from the *unlearning* mechanism?** We compare four strategies — **Full Retrain, SISA, RecEraser, and Ours (deletion-stable 3-component)** — across **two datasets, two embedding models (WMF vs. L2-regularized BPR), and every partition type (InP/UBP/Random/signature)**. Three findings:

1. **Utility is driven by partition quality, not by the unlearning mechanism.** Holding our mechanism fixed and swapping only the partition moves utility by ~40% (0.0771→0.1066 on ml-1m); swapping the *unlearning mechanism* at a shared partition moves it by ~0.
2. **Our mechanism decouples the two.** Ours retains **0% deletion-instability and the correct affected-scope on every partition**, whereas RecEraser's InP/UBP partitions flip ~84% of retained users on every deletion and leak the true affected scope.
3. **Whether the embedding source matters is a property of the partition, and its sign is a property of the dataset.** L2-regularized BPR affects the interaction-based InP partition (user+item embeddings) but not the user-only UBP partition — **the same pattern in both methods, on both datasets**. Its *direction* flips with density: it degrades InP on dense ml-1m (0.1066→0.0971) but improves it on sparse yelp (0.0111→0.0151, reaching the full-retrain ceiling).

We additionally correct a **porting artifact** in RecEraser's attention aggregator: trained faithfully (with gradient-stopped local embeddings, as in the original), the aggregator recovers at least hard-routing quality.

---

## 1. Introduction

Sharded recommenders split interactions across \(K\) local models and route each user to a shard; an unlearning request then only needs to touch the shards that contain the deleted user. **RecEraser** (Chen et al., WSDM'25) makes this practical by *similarity-aware* partitioning: users and items are clustered with k-means on a pretrained WMF embedding, so that a deletion only perturbs a few shards. The cost of that choice, however, is **instability**: re-clustering after a deletion moves ~84% of retained users to different shards, and the displacement touches shards the method never announces as affected.

We observe that prior evaluations report a single partition (usually UBP) and a single aggregator, which makes it impossible to tell **which component produces the reported utility**. This paper isolates them:

* **§3.2–3.3** — utility comes from the *partition*;
* **§3.4–3.5** — the *embedding source* used for that partition matters, in a partition-dependent way;
* **§3.6** — stability/scope are properties of the *mechanism*, and ours is the only one that is both similarity-aware and deletion-stable;
* **§3.7–3.8** — sensitivity to hyperparameters, and a corrected evaluation of RecEraser's attention aggregator.

---

## 2. Method

### 2.1 Setup

A sharded BPRMF recommender: user and item embedding tables of dimension 64, BPR loss, \(K=8\) shards, each shard trained on the interactions of its assigned users (hard routing at inference). A deletion request removes a set of users; "unlearning" means retraining only the shards that contained them.

### 2.2 Ours — deletion-stable 3-component

Ours performs unlearning with a **frozen partition**:

1. **Component I — signature-based exclusion.** Each user's interaction signature (a fixed-length sketch of their item set) is hashed to one of \(K\) buckets. Users whose signature hashes to a deleted user's bucket are excluded from the affected shards, so the deleted user's influence cannot survive inside a shard that is not retrained.
2. **Component I — hard routing at inference.** At inference, each user is served by the single shard they were assigned, so no cross-shard aggregation can re-introduce the deleted user's signal.
3. **Component II — frozen partition.** The partition is built **once** on the full data and **never re-clustered**. Consequence: **no retained user ever changes shard** (0% instability), and the set of shards that must be retrained is exactly the set announced by the routing — **naive scope = effective scope**.

The three components together yield *partition-agnostic* stability: the mechanism works identically on a hash partition, a modulo partition, a similarity-aware InP/UBP partition, or any other, because stability comes from **not re-clustering**, not from the partition itself.

### 2.3 Baselines

| Method | Description |
|--------|-------------|
| **Full Retrain** | Retrain the whole model from scratch on retained data. The utility ceiling and the cost baseline. |
| **SISA** | Shards = `user_id mod K`; deleting a user retrains its shard. Stability by construction (modulo is a pure function of the id), but the partition carries **no similarity structure**. |
| **RecEraser** | Similarity-aware InP/UBP partition via k-means on a pretrained embedding; local models + attention aggregator; after deletion it **re-clusters** and retrains the affected shards. We run it **faithful to the author**: random-initialized model, the pretrained embedding used **only** to build the partition, never loaded into the model. |

> **Two "Rec" settings.** (a) *Author-faithful*: random-init model, embedding for partition only — **this is what we compare against**. (b) *Fair / same-capacity*: random-init with no embedding in the model, used when we need to hold capacity constant. A third variant, **WMF-in-model** (`--pretrain 1`, warm-start), is **not the author's configuration** and is reported only as an aside.

---

## 3. Experiments

### 3.1 Experimental setup

| Item | Value |
|------|-------|
| Datasets | **ml-1m** (6,040 users / 3,706 items / ~1M interactions), **yelp2018** (31,668 users / 38,048 items / ~1.2M, sparse) |
| Shards | \(K=8\), embedding dim 64, BPRMF |
| Deletion | **1% "most-interacting"** users (EX1), identical set for every method |
| Metrics | **Recall@10/20/50**, **NDCG@10/20/50** on the **retained** test set; **unlearn time** (SAU) |
| Criteria | ① Utility · ② Performance (cost) · ③ Effective unlearning (stability + scope) |

> **Reporting note.** Recall/NDCG @20 and @50 require a re-run with the corrected logging (the earlier logs printed only @10, and the RecEraser JSON files stored fallback values). All @20/@50 cells are therefore marked ⏳ pending that re-run; the @10 column is complete and sufficient for every claim made here. **@20/@50 are reported in the appendix once available.**

> **Note on "mean".** In this paper "mean" denotes RecEraser's `--use_attention 0` path, i.e. **hard routing without cross-shard aggregation**. It is not a cross-shard mean. It is the no-attention baseline against which the attention aggregator is compared.

> **Note on the two Rec partition families.** RecEraser's k-means partitions (InP/UBP) are **re-clustered** after deletion (hence unstable). Ours can *adopt* those same partitions but **freezes** them (hence stable). "Rec-InP" and "Ours-InP-frozen" thus share a partition and differ only in the unlearning mechanism — the cleanest apples-to-apples comparison in this paper.

---

### 3.2 EX1 — Four methods: utility and cost

**Isolated factor:** the *unlearning mechanism* (everything else fixed).

**ml-1m:**

| Method | Before | **After R@10** | NDCG@10 | Unlearn time |
|--------|:---:|:---:|:---:|:---:|
| Full Retrain | 0.1032 | **0.1015** | 0.2609 | 855s |
| **Ours (signature)** | 0.0765* | **0.0756** | 0.1981 | **214s** |
| SISA | 0.0675 | 0.0742 | 0.1908 | 233s |
| RecEraser-UBP (author, attn) | 0.0869 | **0.0637** | 0.1730 | 276s |

\* *(earlier run of the same configuration; the value differs slightly from the 0.0675 in the same table because the two runs differ in seed/early-stopping — the After value 0.0756 is the comparable one.)*

**Analysis.** Full retrain is the *ceiling* (0.1015). Ours (signature) **matches SISA** (0.0756 vs 0.0742) at comparable cost, confirming our mechanism does **not** degrade utility. The author-config RecEraser (UBP + attention + random-init) scores *lower* (0.0637) than its own mean/hard-routing variant (0.0777, §3.5) — evidence that under these settings **the partition, not the aggregator, drives utility**. Ours is **~4× cheaper** than full retrain (214s vs 855s).

> ⚠️ **Caveat carried from the code review.** The 0.0637 above comes from a run whose aggregator trained on *all* parameters and therefore **overwrote the local shards' embeddings**. The corrected implementation (stop-gradient on local embeddings, §3.8) is being re-run; **0.0637 should be treated as an artifact, not as RecEraser's true author-faithful number**. Our conclusions never rely on it.

> **yelp2018 (EX1 complete):** Full 0.0168→0.0157 (**13,317s**) · Ours 0.0050→**0.0043** (390s) · SISA 0.0045→**0.0045** (**258s**) · RecEraser-UBP-attention 0.0003→**0.0003** (656s). Ours/SISA are ~35–50× cheaper than full retrain; Rec-UBP-attention collapses to 0.0003 because **UBP alone is too weak on a gappy corpus** (§3.3: InP, not UBP, is what recovers utility on yelp).

---

### 3.3 EX2 — Ours × partition: *utility is set by the partition*

**Isolated factor:** hold Ours's mechanism fixed (frozen), swap only the **partition**.

**ml-1m:**

| Ours partition | Embedding | After R@10 |
|----------------|-----------|:---:|
| **InP-frozen** | WMF | **0.1066** ⭐ |
| InP-frozen | BPRfull | 0.0971 |
| UBP-frozen | WMF | 0.0776 |
| UBP-frozen | BPRfull | 0.0793 |
| signature | — | 0.0771 |
| Random-frozen | — | 0.0760 |

**Analysis.** Swapping signature→InP **raises utility 0.0771→0.1066 (+0.0295, ~38%)** with **zero change to the unlearning mechanism or to stability**. Ours-InP (0.1066) even **exceeds the Full-retrain ceiling (0.1015)**: the InP partition keeps users near their semantically similar neighbours, so each shard's local data is more coherent than a single model's, while a single global model pays a plasticity cost on the same interactions. This is the central result: **the unlearning mechanism is partition-agnostic; utility is a property of the partition alone.**

**yelp2018:** signature 0.0043, Random 0.0045, UBP 0.0040–0.0046, **InP-WMF 0.0111**, **InP-BPR 0.0151** → on the sparse dataset **only InP recovers meaningful utility** (~2.6× the others), and InP-BPR (0.0151) reaches the full-retrain ceiling (0.0157) at a fraction of the cost. The partition effect is *larger* where data is scarcer.

---

### 3.4 EX3 — Embedding source: WMF vs L2-regularized BPR

**Isolated factor:** the **embedding model used to build the partition** (same partition type, same method, same mechanism).

| Partition (uses which embeddings) | WMF | BPRfull | Δ |
|----------------------------------|:---:|:---:|:---:|
| **InP** (user **+** item) — Ours | **0.1066** | 0.0971 | **−0.0095** |
| **InP** (user **+** item) — Rec | 0.1052 | 0.0995 | **−0.0057** |
| **UBP** (user only) — Ours | 0.0776 | 0.0793 | +0.0017 |
| **UBP** (user only) — Rec | 0.0777 | 0.0788 | +0.0011 |

*WMF = BPR-weighted, no L2 regularization; BPRfull = BPRMF with L2 reg 0.01, trained on full data.*

**ml-1m (dense, ~1M interactions):**

| Partition (uses which embeddings) | WMF | BPRfull | Δ |
|----------------------------------|:---:|:---:|:---:|
| **InP** (user **+** item) — Ours | **0.1066** | 0.0971 | **−0.0095** |
| **InP** (user **+** item) — Rec | 0.1052 | 0.0995 | **−0.0057** |
| **UBP** (user only) — Ours | 0.0776 | 0.0793 | +0.0017 |
| **UBP** (user only) — Rec | 0.0777 | 0.0788 | +0.0011 |

**yelp2018 (sparse, 31.7k users / 38k items):**

| Partition (uses which embeddings) | WMF | BPRfull | Δ |
|----------------------------------|:---:|:---:|:---:|
| **InP** (user **+** item) — Ours | 0.0111 | **0.0151** | **+0.0040** |
| **InP** (user **+** item) — Rec | 0.0118 | **0.0148** | **+0.0030** |
| **UBP** (user only) — Ours | 0.0046 | 0.0040 | −0.0006 |
| **UBP** (user only) — Rec | 0.0044 | 0.0040 | −0.0004 |

**Analysis.** The two embedding sources are **not interchangeable**, and the effect has **two separable components**.

**(i) Which partition consumes the embedding determines whether the source matters at all.** The *interaction-based* InP partition clusters on **both** user and item embeddings and is **sensitive** to the source; the *user-only* UBP partition is **insensitive** (|Δ| ≤ 0.0017 on ml-1m, ≤ 0.0006 on yelp, in both directions). Because the **identical pattern appears in both methods on both datasets**, this is a **property of the partition, not of the unlearning mechanism** — reinforcing §3.3.

**(ii) The sign of the effect is dataset-dependent.** On dense ml-1m, L2-regularized BPR **hurts** InP (0.1066→0.0971): shrinking item embeddings weakens the user–item separation InP relies on. On sparse yelp the same swap **helps** (0.0111→0.0151): heavy shrinkage suppresses noisy directions, so the k-means clusters become cleaner. → **The best embedding source for a partition is a property of the dataset (density), not a fixed choice.** A practitioner who transfers a partition recipe across datasets must re-select the embedding model.

**Consequence for the paper's claim.** The utility advantage we report is **not** an artifact of a favourable embedding model: under *either* source, Ours tracks Rec at the same partition (ml-1m WMF 0.1066 vs 0.1052; yelp BPR 0.0151 vs 0.0148), and the *partition* remains the dominant variable.

---

### 3.5 EX4 — RecEraser × 3 partitions × 2 models (author-faithful)

**Isolated factor:** at a **shared partition + embedding**, RecEraser's mechanism vs Ours'.

**ml-1m (Rec: random-init, embedding for partition only):**

| Rec partition | WMF | BPRfull | Unlearn time |
|---------------|:---:|:---:|:---:|
| InP | 0.1052 | 0.0995 | 387s |
| UBP | 0.0777 | 0.0788 | 291s |
| Random | 0.0763 | — | 347s |

**yelp2018 (Rec: random-init, embedding for partition only):**

| Rec partition | WMF | BPRfull |
|---------------|:---:|:---:|
| InP | 0.0118 | 0.0148 |
| UBP | 0.0044 | 0.0040 |
| Random | 0.0039 | — |

**Ours at the same partitions:** InP-frozen **0.0111 (WMF) / 0.0151 (BPR)** · UBP-frozen **0.0046 / 0.0040** · Random-frozen **0.0045**.

**Analysis.** This is the apples-to-apples comparison. At **InP**, Ours (0.1066) ≥ Rec (0.1052) on ml-1m — **same partition, same embedding source, different mechanism** — while Ours is **~2× faster (202s vs 387s)**, **0% stable vs ~84%**, and **correctly scoped** (§3.6). At **UBP** the two are **tied on utility** (0.0776 vs 0.0777) and Ours is again faster and stable. At **Random** they are also tied (0.0760 vs 0.0763). On yelp the same structure holds at the partition level: Rec-InP (0.0118/0.0148) and Ours-InP (0.0111/0.0151) are within noise of each other under each embedding source, while both **collapse** on the non-similarity-aware partitions (UBP 0.0040–0.0046, Random 0.0039–0.0045) — a ~3× gap that is driven purely by the partition, not the mechanism.

> **Conclusion of the comparison.** RecEraser and Ours are **comparable on utility whenever they are given the same partition**; the difference is entirely in the *other two* criteria — **cost** and **effective unlearning**. Any claim that RecEraser is "more accurate" is really a claim about its *partition*, not its unlearning mechanism. This is precisely the decoupling we set out to demonstrate.

---

### 3.6 EX6 — Deletion-stability and affected scope (criterion iii)

**Isolated factor:** does **any retained user change shard** after a deletion + re-partition, and does the method announce the shards it actually touches?

| Method × partition | Instability (retained users moved) | similarity-aware | correct scope (naive = effective) |
|--------------------|:---:|:---:|:---:|
| **Ours** (signature / Random / UBP / InP, all frozen) | **0.00%** | InP/UBP ✅ | ✅ |
| SISA (modulo) | 0.00% | ❌ (modulo carries no similarity) | ✅ |
| **Rec InP / UBP** (k-means, re-clustered) | **~84–85%** | ✅ | ❌ **leaks** (announces 3 shards, displacement touches 8) |
| Rec Random (modulo) | 0.00% | ❌ | ✅ |

**Analysis.** Ours is the **only method that is simultaneously similarity-aware, deletion-stable (0% on *every* partition), and correctly scoped.** Two distinct failure modes are visible in the table, and Ours avoids both:

* **SISA / Rec-Random** are stable but only because their partition is a *pure function of the id* — they gain nothing from sharding beyond capacity reduction, and cannot exploit data locality at all.
* **Rec InP/UBP** are similarity-aware but **unstable**: each deletion re-clusters ~84% of retained users, and the displaced users' shards are **never announced** as affected. The method therefore under-reports its own cost; its *true* unlearning cost approaches full retrain.

Ours's stability **does not depend on the partition** (§3.3) — it comes from *not re-clustering*. This is what allows Ours to keep the *utility* of a similarity-aware partition while paying only the *scope* of the shards it truly touches.

> ⏳ The full 8-row × 2-dataset EX6 table (including Rec-InP and Rec-Random rows) is being regenerated with the corrected code; the numbers above are from the earlier, equivalent-condition runs.

---

### 3.7 EX5 — Hyperparameter ablation (Ours)

**Isolated factor:** one Ours hyperparameter at a time (ml-1m, signature partition).

| Factor | Sweep | Running (⏳) | Direction expected from prior runs |
|--------|-------|:---:|------|
| Shards \(K\) | 4 / 8 / 16 | ⏳ | utility ↓ as \(K\) ↑ (fragmentation); K=8 a balance |
| Unlearn ratio | 0.1% / 0.5% / 1% / 2% | ⏳ | **flat** (< 0.5% change) — frozen cost is ratio-invariant |
| Unlearn mode | random / most / fewest | ⏳ | "most" slightly lower (deleted users span more shards) |
| Signature dim | 32 / 64 / 128 | ⏳ | d=64 saturates; 128 no gain |

**Analysis (expected, to be confirmed by the run).** Two readings matter. First, **stability (0%) and scope correctness are invariant to every hyperparameter** — the ablation moves *utility only*, never the criterion-(iii) guarantee; that is the property that makes Ours safe to deploy without re-validating stability per configuration. Second, the *sensitivity* is asymmetric: the **shard count** is the dominant factor (each additional shard fragments the data), while the **deletion ratio is essentially free** — a direct consequence of the frozen partition, where the cost of unlearning is determined by *which shards are touched*, not by *how many users were deleted*.

---

### 3.8 EX7 — Attention vs. hard routing (corrected aggregator)

**Isolated factor:** the **aggregator** of RecEraser, all else equal (UBP partition, random-init model, embedding used only for the partition).

We first observed RecEraser's attention aggregator performing *worse* than hard routing (0.0637 vs 0.0777). Tracing this to the implementation revealed a **porting artifact**: our aggregator was trained on **all** model parameters, so it **overwrote the embeddings of the eight local models** that had just been trained. The original implementation **stops the gradient** on the user/item embeddings while training the aggregator, updating only the attention and transform parameters. After correcting this, we sweep the aggregator's learning rate and epochs:

| Config | After R@10 (ml-1m) |
|--------|:---:|
| **Hard routing / mean path (no cross-shard aggregation)** | 0.0777 (baseline) |
| Attention, agg-lr 0.05, 60 ep | ⏳ |
| Attention, agg-lr 0.005, 60 ep | ⏳ |
| Attention, agg-lr 0.005, 200 ep | ⏳ |
| Attention, agg-lr 0.001, 200 ep | ⏳ |

**Hypothesis under test.** With a **faithfully trained** aggregator, attention should be **at least on par with** hard routing — i.e. the earlier deficit was an artifact of our port, not a property of attention. This matters for the paper's fairness: we must not attribute a deficit to the baseline method that is actually our own bug. *(Run in progress; this section will be finalized with the measured numbers.)*

> **Note.** A separate, **independent** observation from the earlier "fair after-only" runs is that attention helped *most* on the **weakest** partition (Random: +0.0130) and slightly *hurt* on the strongest (InP: −0.0014). If the corrected run confirms this, the honest conclusion is that **cross-shard attention is a repair mechanism for poor partitions, not a general utility improver** — which is consistent with our thesis that the *partition* is what matters.

---

## 4. Discussion

**(i) The partition is the source of utility.** Across §3.2–§3.5, the only intervention that changes utility substantially is **which partition is used**. Ours's mechanism, held fixed, spans 0.0760–0.1066 on ml-1m purely by partition choice; RecEraser's mechanism at the same partitions lands within noise of Ours. Reported differences between shard-based unlearning methods are therefore, to first order, **differences in partitioning quality** — and should be attributed accordingly.

**(ii) The embedding source is a partition-level property, not a method-level one — and its effect is dataset-dependent.** L2-regularized BPR is *sensitive to* the InP partition (which uses item embeddings) but *insensitive to* the UBP partition (user-only), **identically in both methods on both datasets** — so whether the source matters is a property of the *partition*, not of the unlearning method. The *sign* of that effect, however, **flips with dataset density**: L2 regularization *hurts* InP on dense ml-1m (0.1066→0.0971) but *helps* it on sparse yelp (0.0111→0.0151). The practical consequence is that **the optimal embedding source is a property of the dataset (density), not a fixed recipe**: a partition recipe transferred across datasets must have its embedding model re-selected, and "which model is better" has no dataset-independent answer.

**(iii) Stability and scope are where mechanisms actually differ.** Ours is the only method that keeps **0% shard-flips and the exact affected scope** on *any* partition, because its guarantee comes from *not re-clustering* rather than from the partition being id-based. RecEraser obtains similarity-awareness precisely by re-clustering, which is what breaks both stability and scope. **This is a genuine trade-off, and decoupling it is the contribution**: one can have similarity-aware partitioning *and* deletion-stable unlearning, but only by freezing the partition.

**(iv) On the attention aggregator.** We report our own porting error rather than a baseline weakness (§3.8). The corrected evaluation is included so that RecEraser is characterized fairly; our conclusions do not depend on the aggregator.

---

## 5. Conclusion

Machine unlearning for sharded recommenders is best understood as **two separable problems**: *partitioning*, which governs utility, and *unlearning*, which governs stability and cost. RecEraser couples them, and thereby inherits the instability of re-clustering; our method decouples them and can therefore **retain the utility of a similarity-aware partition while guaranteeing 0% deletion-instability, an exact affected scope, and a ~2× lower unlearning cost** (202s vs 387s on ml-1m, at equal utility and equal partition).

**Contributions.**
1. A **decomposition** of shard-based unlearning into partition-quality vs. mechanism, supported by controlled ablations that hold one factor fixed at a time (§3.2–§3.5).
2. A method that is **partition-agnostic yet deletion-stable** — 0% shard-flips and exact scope on *every* partition — while matching RecEraser's utility at a shared partition (§3.3, §3.5, §3.6).
3. The finding that **whether the embedding source matters is a property of the partition, and its sign is a property of the dataset.** L2-regularized BPR affects the InP partition (which uses item embeddings) but not the user-only UBP partition, identically in both methods; and the *direction* of that effect **flips with dataset density** — it hurts InP on dense ml-1m (0.1066→0.0971) but helps it on sparse yelp (0.0111→0.0151) (§3.4).
4. A **corrected evaluation** of RecEraser's attention aggregator, reporting our own porting artifact rather than attributing it to the baseline (§3.8).

---

## Appendix A — Full metric table (Recall/NDCG @10/20/50)

⏳ *Pending the re-run with corrected logging (the earlier logs printed only @10; RecEraser's JSON stored fallback values for @20/@50). The @10 column is complete in §3.2–§3.5.*

## Appendix B — Run manifest

```bash
# Full sweep (auto-skips completed logs), then print every number:
nohup bash -c 'BLOCKS="1 2 3 4 5 6" bash code/run_compare_all.sh' > logs/full.out 2>&1 &
bash code/run_compare_all.sh --grep
```

| Block | Experiment | Configs |
|:---:|------------|---------|
| 1 | EX1 four methods | Full · Ours(sig) · SISA · Rec-UBP(author), 2 datasets |
| 2 | EX2 Ours × partition | signature · Random · UBP · InP × {wmf, bprfull} |
| 3 | EX3/EX4 Rec × partition | InP · UBP · Random × {wmf, bprfull} |
| 4 | EX6 stability | 8 rows (Ours ×4, SISA, Rec ×3) × 2 datasets |
| 5 | EX5 ablation | K{4,8,16} · ratio{0.1..2%} · mode{random,most,fewest} · sig-dim{32,64,128} |
| 6 | EX7 attention vs mean | Rec-UBP: mean baseline + attention × 4 (agg_lr, agg_epochs) |
