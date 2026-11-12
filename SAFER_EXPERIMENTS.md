# SAFER - code changes and how to run the experiments

Commit `d8074b0`. Everything below was checked against the code, not assumed.

## 1. What was wrong, and what each fix does

### S1 - the blank N@20 / N@50 cells: the numbers were correct

The suspicion was that NDCG@20 had been copied from NDCG@10. Two independent
checks were written to settle it:

- `code/check_ndcg_evaluator.py` - reimplements Recall and NDCG from scratch
  and compares against `evaluate_hard_routing` on synthetic data.
- `code/check_ndcg_realdata.py` - the same comparison on the real ml-1m test
  split, where the test list lengths and the distinct item counts matter.

Both agree to the last digit (`|diff| = 0.00e+00` for Recall and NDCG at
K = 10, 20, 50), and on the real data NDCG@10 = 0.008759 differs from
NDCG@20 = 0.009516. The old CSV agrees too (`ndcg_10=0.19922` against
`ndcg_20=0.19799`).

**So the log values are right.** The two cells were a reporting gap: the
per-run print only showed `R@10, NDCG@10, R@20, R@50`. All six metrics are
now printed per run and the summary table shows all six columns.

Why NDCG@10 and NDCG@20 come out close is a property of the metric, not a
copy: the ideal DCG changes with K, and on this data most users have fewer
than 10 test items, so the two cut-offs rank much of the same list.

### S2 - instability rate (new)

`instability_rate()` in `code/exp_partition_quality.py`. After the deletion
the partition is **re-run on the remaining data** - it is not reused - and two
numbers are reported:

- **label-matched**: the shard labels of the new partition are aligned with
  the old ones using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`)
  on the shard co-occurrence matrix, so a pure permutation of the labels does
  not count as instability. **This is the number to quote.**
- **pairwise**: label-invariant. The share of retained user *pairs* whose
  co-location flipped (same shard before, different after, or the reverse).
  A permutation of the labels cannot change this, so it is the conservative
  check.

`_rebuild_partition()` rebuilds each arm's own partition the same way it
built the first one, so the comparison is between like and like.

### S3 - the retrain reference never retrained

`run_global_arm()` trained one model on the reduced data and returned that
single duration for both columns:

```python
return out, train_time, train_time     # old
```

So `train_time` and `unlearn_time` were identical by construction and the
cost of the retrain was never measured. It now trains the original model on
the full data (`train_time`) and then a fresh model on the reduced data
(`unlearn_time`), and evaluates the retrained model - the one that would
actually be deployed after the request.

### S4 - the dimension ablation never moved the backbone

`--emb_dim` and `--sig_dim` were already separate arguments. The problem was
in the runner: the ablation stage only changed `--sig_dim`, so the BPRMF
backbone stayed at 64 while the signature changed. The dimension stage now
moves `--emb_dim` and holds `--sig_dim 64` fixed.

That RANDOM_PERM gave the identical result at every dimension is expected
rather than a bug: the random partition does not read either dimension. What
mattered was that PROPOSED was never re-run at a different backbone
dimension, and now it is.

### S5 - timing

- `train_shard_model()` now returns `(loss, elapsed, n_samples)`.
- `run_sharded_arm()` records the retrain time and the interaction count of
  **each** affected shard, and reports `parallel_time`, the wall time if the
  affected shards were retrained concurrently (the slowest shard).
- The per-shard times are printed next to the interaction counts, and their
  sum is printed next to the reported `unlearn_time`.

A worked check from the smoke test:

```
per-shard retrain (s): s0=1.3s/35395int, s1=5.0s/142425int, s2=3.8s/108184int,
s3=7.4s/211755int, s4=0.7s/21187int, s5=2.0s/60295int, s6=1.7s/50649int,
s7=2.9s/82306int  | sum=24.9s vs unlearn_time=25.2s
```

The sum matches the reported time, and the time scales with the shard size,
so the retrain really does iterate over the shard and not over the whole
dataset.

### S6 - the router ablation compared an arm with itself

`PROPOSED` took its router from `--router`, while `PROPOSED_MINHASH` was
hard-coded to minhash. Running the ablation with `--router minhash` therefore
turned **both** arms into minhash, which is why they came out identical.

`PROPOSED` now always uses the hyperplane router; `PROPOSED_MINHASH` carries
the variant. A separate issue is worth noting for the paper: `route_minhash`
collapses the bands into a single 64-bit key, so every user gets their own
key and `np.unique` orders them by hash value. The resulting partition is
ordered by hash rather than by similarity, which is why its IntraSim sat at
the random level (0.0794). Fixing that is a change to the router, not to the
experiment, and is left for a later pass.

### The SISA arm was not SISA

The SISA arm was evaluated with hard routing, which drops the aggregation
that is the defining part of SISA. `method_2_sisa.py`, the original
implementation in this repository, averages the scores of **all K** models at
inference time. `evaluate_sisa_mean_aggregation()` now does the same, so the
arm in the table is the method the paper names.

The other arms keep the hard routing that the spec fixes for the experiment.

### The backbone is consistent across the arms

Every arm trains BPRMF:

| arm | model | loss |
|---|---|---|
| PROPOSED, PROPOSED_BAL, RANDOM_EMB, RANDOM_PERM, SISA, RANDOM_MODULO | `method_4_ours.BPRMF` | `-log(sigmoid(diff)) + 0.01 L2` |
| GLOBAL (Retrain) | `method_4_ours.BPRMF` | the same |
| RECERASER | `RecEraserModel` | `softplus(-diff) + 0.01 L2` |

`softplus(-x)` and `-log(sigmoid(x))` are the same function: checked
numerically, the largest difference over a range of scores is 2.98e-08, which
is float rounding.

The `WMF` class in the repository is **not** a backbone here. It produces the
pretrained embeddings (`wmf_embeddings.npz`) that feed RecEraser's partition,
which is how the original RecEraser works. No arm is trained or evaluated
with it.

`code/check_stale_retrain_cost.py` also reports the backbone of every log, so
a table cannot end up mixing a BPRMF arm with a WMF arm.

### The retrain cost in the old logs is not a measurement

The old `run_global_arm` trained one model on the reduced data and returned
that single duration for both the train and the unlearn column. In those logs
the retrain cost of the reference arm reads exactly the same as its training
cost, so it was never measured.

`code/check_stale_retrain_cost.py` finds those logs:

```
log                                 seed    train(s)   unlearn(s)   rel.diff  verdict
main_yelp_old                        0      1544.6       1544.6     0.0000  STALE (train == unlearn)
main_ml1m_new                        0      1544.6       1602.3     0.0374  ok
```

Those logs must not be merged with the new ones. Either re-run them with the
fixed code, or keep them apart and say so in the paper.

### One more thing that was wrong: the retrain did not remove the user

The affected shards were retrained **from their current weights**. The retrain
only updates the users that are still present, so the deleted user's
embedding row stayed in the table. The deleted user was therefore still in
the model, which contradicts the exact-unlearning claim.

The affected shards are now retrained **from a fresh model** by default. The
old behaviour is still available behind `--keep_weights_on_retrain`, because
the earlier logs were produced with it and those logs are the baseline the
new numbers are compared against.

This is why the smoke test gives a lower R@10 (0.0657 for PROPOSED with 2
epochs) than the old logs: a fresh model has to relearn the shard from
scratch, while the old path kept most of what it had learned.

## 2. New experiments

### RECERASER arm - `code/exp_receraser_arm.py`

RecEraser is structurally different from the other arms, so it gets its own
runner rather than being forced into the shard loop:

- the partition is the **user-based partition** (k-means over pretrained WMF
  user embeddings, balanced by interaction count), recomputed from the data
  it is given, so the instability rate is meaningful;
- the model holds all K shards' embeddings side by side;
- inference uses the **attention aggregation** over all K shards, not hard
  routing;
- deleting a user forces the **aggregator** to be retrained as well, since
  the attention weights depend on every user's embedding. That cost is
  included in `unlearn_time`, because it is part of what the request costs.

The partition needs pretrained WMF embeddings, produced once per dataset:

```bash
python code/pretrain_wmf.py --dataset ml-1m --emb_dim 64 --batch_size 512 \
    --learning_rate 0.05 --max_epochs 100
```

### Single deletion request - `code/exp_single_request.py`

The batch experiments delete 1% of the users at once. This measures what a
deployed system answers: how long to remove **one** user. The same 20 users
are used for every arm and every seed, so the requests are comparable.

For each request the original model is trained once, then for each victim the
affected shard is copied, the victim removed, the shard retrained from
scratch (and the aggregator retrained too, for RecEraser), and the time
recorded. The copy is discarded, so every request starts from the same
original model.

## 3. Running it

Every stage runs **one job at a time**, so the reported `unlearn_time` is a
clean single-job measurement.

```bash
cd /workspace/Recommendation-Machine-Unlearning
git pull origin main

# check the GPU first
python -c "import torch; print('cuda:', torch.cuda.is_available()); \
print('GPUs:', torch.cuda.device_count())"

# smoke test: 1 seed, 3 epochs, every new code path once
bash code/run_safer_experiments.sh verify

# the stages
bash code/run_safer_experiments.sh pretrain     # WMF embeddings, once per dataset
bash code/run_safer_experiments.sh main         # R1: ml-1m + yelp, random, 5 seeds
bash code/run_safer_experiments.sh single       # N2: one deletion request, ml-1m
bash code/run_safer_experiments.sh receraser    # N1: RecEraser arm
bash code/run_safer_experiments.sh modes        # R2: most-active, fewest-active
bash code/run_safer_experiments.sh dims         # R3: backbone dim 32 / 128 / 256
bash code/run_safer_experiments.sh all          # all of the above, in order
```

Each stage writes `logs/<name>.out` with the four blocks the spec asks for:
quality, partition, cost, and the paired comparison against RANDOM_PERM.

Follow a running stage with:

```bash
tail -f logs/main_ml1m.out
```

### Log names

| Stage | Log files |
|---|---|
| main | `main_ml1m.out`, `main_yelp.out` |
| single | `single_request_ml1m.out` |
| receraser | `receraser_ml1m.out`, `receraser_yelp.out`, `receraser_ml1m_most.out`, `receraser_ml1m_fewest.out` |
| modes | `mode_most_ml1m.out`, `mode_fewest_ml1m.out` |
| dims | `dim32_ml1m.out`, `dim128_ml1m.out`, `dim256_ml1m.out` |

## 4. What to read out of a run

The summary table now has four blocks:

```
arm    recall_10  ndcg_10  recall_20  ndcg_20  recall_50  ndcg_50
arm    intra_similar  nr_10  nr_20  nr_50  user_load_cv  interaction_l  max_mean_rati
arm    instability_lab  instability_pai
arm    train_time  unlearn_time  parallel_time  affected_shard
```

and the paired block compares each arm against RANDOM_PERM with the delta,
the standard deviation of the delta and the p-value, for every metric.

## 5. Read from the code, no run needed

- **Train / validation / test split.** The split is fixed on disk and shared
  by every arm and every seed. `code/exp_partition_quality.py:load_data()`
  reads `data/<dataset>/train.txt` and `data/<dataset>/test.txt`. There is no
  separate validation split: the data is already split into train and test,
  and the model is trained on the train part for a fixed number of epochs with
  no early stopping, so no validation set is used anywhere.
- **Signature dimension.** 64. `build_signature()` in
  `code/partition_variants.py` draws a random `+-1` projection matrix `R` of
  shape `(d, n_items)` with `d = sig_dim`, and the main experiment runs with
  `--sig_dim 64`.
- **Optimizer.** Adagrad, `lr = 0.05`, `initial_accumulator_value = 1e-8`,
  batch size 512, 50 epochs for both the initial training and the retrain,
  no early stopping. The BPRMF loss carries an L2 regularisation of 0.01.
