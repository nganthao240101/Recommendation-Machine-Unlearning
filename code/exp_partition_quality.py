"""
EXPERIMENT: does the proposed partition beat a random partition on utility?

Answering H1: Utility(PROPOSED) > Utility(RANDOM)
against   H0: Utility(PROPOSED) <= Utility(RANDOM)

Design (paired runs - the only valid way to compare two partitions):
  For each seed s:
    1. ONE split and ONE deletion set, shared by every arm.
    2. Every arm trains with the same backbone, same dim, same lr, same epochs,
       same random-init rule, same hard-routing inference, same frozen-mapping
       unlearning (retrain only the shards that contain deleted users).
    3. The ONLY thing that differs between PROPOSED and RANDOM is the shard map.
  Arms differ only in the partition; everything else is held fixed.

Arms
  PROPOSED       Ours signature + deletion-stable similarity-aware routing (LSH)
  PROPOSED_BAL   same, but the cell->shard mapping is load-balanced by
                 interaction count (removes the "one huge shard" confound)
  RANDOM_EMB    random user embedding routed through the SAME LSH router -
                 the control that isolates the value of the signature itself
  RANDOM_PERM    random permutation of user ids -> K shards   (primary baseline)
  RANDOM_BAL     random permutation, greedy-balanced by interaction load
                 (the load-matched control)
  RANDOM_MODULO  user_id % K                                   (sanity check)
  SISA           user_id % K, frozen mapping, evaluated with the mean
                 aggregation over all K models, which is the original SISA
  INP / UBP      embedding-based (k-means over WMF) partition, frozen -
                 UPPER REFERENCE for partition quality, not part of the
                 PROPOSED-vs-RANDOM claim
  GLOBAL         one BPRMF trained on all retained data         (utility/cost reference)

Partition-quality metrics (explain WHY utility differs)
  intra_similarity  mean cosine similarity of sampled within-shard user pairs
  nr_q              fraction of a user's q nearest neighbours in the same shard
  user_load_cv      std/mean of users per shard
  interaction_load_cv  std/mean of interactions per shard
  max_mean_ratio    max_k W_k / mean_k W_k   (W_k = interactions in shard k)

Output: one CSV row per (seed, arm) - never overwrite, aggregate separately.
"""

import os
import sys

# Write every line to the log as it is printed. Piped through tee, Python
# otherwise buffers about 8 KB, so a log could sit hours behind the run and
# a progress check would read a live run as stuck.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass
import time
import json
import random
import argparse
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

import gpu_probe
import fast_train
import wmf_backbone
from partition_variants import (
    build_signature as pv_build_signature,
    random_embedding as pv_random_embedding,
    route_hyperplane,
    route_minhash,
    random_perm as pv_random_perm,
    random_perm_balanced,
    random_modulo as pv_random_modulo,
)
from method_4_ours import (
    UserSignatureBuilder,
    DeletionStableAssignment,
    ShardModels,
    train_shard_model,
    evaluate_hard_routing,
)

KS = [10, 20, 50]


# ----------------------------------------------------------------------------
# seeding
# ----------------------------------------------------------------------------
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_data(dataset, data_root):
    """Load train/test. Split is FIXED on disk, so it is shared by every seed.

    The spec asks for a split per seed; the provided train.txt/test.txt are
    already a single fixed split. We keep it fixed (identical for all arms and
    seeds) because a per-seed resplit would change the data seen by every arm
    equally and would multiply the cost of an already long run. The seed
    variation therefore comes from the deletion set, the partition, and the
    model initialisation - the three factors that actually differ.
    """
    d = os.path.join(data_root, dataset)
    train_items, test_set = {}, {}
    n_users, n_items = 0, 0

    with open(os.path.join(d, 'train.txt')) as f:
        for line in f:
            parts = line.strip().split(' ')
            if not parts or not parts[0]:
                continue
            uid = int(parts[0])
            items = [int(i) for i in parts[1:] if i]
            train_items[uid] = items
            n_users = max(n_users, uid + 1)
            for i in items:
                n_items = max(n_items, i + 1)

    with open(os.path.join(d, 'test.txt')) as f:
        for line in f:
            parts = line.strip().split(' ')
            if not parts or not parts[0]:
                continue
            uid = int(parts[0])
            items = [int(i) for i in parts[1:] if i]
            if items:
                test_set[uid] = items

    return train_items, test_set, n_users, n_items


def pick_deletion_users(train_items, ratio, mode, seed):
    """Deletion set for this seed - shared by every arm (paired)."""
    rng = random.Random(seed)
    users = [u for u, it in train_items.items() if len(it) > 0]
    n_del = max(1, int(len(users) * ratio))

    if mode == 'most':
        users.sort(key=lambda u: len(train_items[u]), reverse=True)
        return set(users[:n_del])
    if mode == 'fewest':
        users.sort(key=lambda u: len(train_items[u]))
        return set(users[:n_del])
    return set(rng.sample(users, n_del))


# ----------------------------------------------------------------------------
# partitions  (the ONLY factor that differs between arms)
# ----------------------------------------------------------------------------
def partition_random_perm(n_users, n_shards, seed):
    """Random permutation of user ids, then split into K shards.

    Not modulo: user ids are often ordered (registration order / temporal
    drift), so uid % K would place correlated users together and would not be
    a genuinely random user-level partition.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_users)
    out = np.zeros(n_users, dtype=np.int32)
    # deal round-robin over the permutation so shards stay balanced
    out[perm] = (np.arange(n_users) % n_shards).astype(np.int32)
    return out


def partition_random_modulo(n_users, n_shards, seed):
    """user_id % K (sanity check; matches the 'random' arm used elsewhere)."""
    return (np.arange(n_users) % n_shards).astype(np.int32)


def partition_proposed(train_items, n_users, n_items, n_shards, sig_dim, seed,
                       router='hyperplane', sig_weighting='binary'):
    """Ours: user-local signature + deletion-stable similarity-aware routing.

    router:
      'hyperplane'  cosine LSH via random hyperplanes (shipped Ours Component II)
      'minhash'     MinHash/LSH banding - Jaccard-style, better suited to the
                    binary user-item vectors that Ours builds (spec ablation 8)
    sig_weighting:
      'binary'      phi(y)=1  (shipped)
      'log_count'   phi(y)=1+log(count) - magnitude-aware preference profile
                   (spec ablation 8)
    """
    sigs = pv_build_signature(train_items, n_users, n_items, d=sig_dim,
                              seed=seed, weighting=sig_weighting, verbose=True)
    if router == 'minhash':
        return route_minhash(train_items, n_users, n_items, n_shards,
                             num_perm=64, bands=8, seed=seed,
                             weighting=sig_weighting)
    return route_hyperplane(sigs, n_shards, seed=seed)


def partition_random_embedding(train_items, n_users, n_items, n_shards,
                                sig_dim, seed, router='hyperplane'):
    """RANDOM_EMB control: random user embedding, SAME LSH router as Ours.

    Only the source of the user representation differs from Ours-signature:
    z_u ~ N(0,I) instead of z_u = R*x_u/||R*x_u||. Everything downstream
    (hyperplane LSH, balanced cell map, isolated shard training, hard routing)
    is identical, so a difference in utility is attributable to the signature
    carrying user information (or not).
    """
    sigs = pv_random_embedding(n_users, d=sig_dim, seed=seed)
    if router == 'minhash':
        return route_minhash(train_items, n_users, n_items, n_shards,
                             num_perm=64, bands=8, seed=seed,
                             weighting='binary')
    return route_hyperplane(sigs, n_shards, seed=seed)


def partition_proposed_balanced(train_items, n_users, n_items, n_shards, sig_dim, seed):
    """PROPOSED but LOAD-BALANCED: same LSH cells (so similar users still
    co-locate), only the cell->shard mapping is rebalanced by interaction load.

    Motivation (measured on ml-1m, K=8): the default mapping in
    DeletionStableAssignment balances the NUMBER OF CELLS per shard, not the
    number of interactions, so shard loads came out very uneven
    (interaction CV 0.668, max/mean 2.39) against 0.047 / 1.06 for a modulo
    partition. With hard routing, the lean shards drag the average utility
    down. This arm isolates that: if it beats a random partition, the
    similarity-aware routing does carry utility and the imbalance was the
    confound; if it ties random, similarity in user space does not help here.
    """
    builder = UserSignatureBuilder(n_users, n_items, d=sig_dim, seed=seed)
    sigs = builder.build_signature(train_items)
    assigner = DeletionStableAssignment(n_shards=n_shards, d=sig_dim, seed=seed)

    # Use MORE hash cells than shards so the cell->shard mapping has freedom
    # to balance load. With B == K the default mapping gives each shard exactly
    # one cell, and no rebalancing is possible (measured: same CV as PROPOSED).
    extra_bits = max(1, int(np.ceil(np.log2(n_shards))))
    b = assigner.b + extra_bits
    hp = np.random.RandomState(seed).randn(b, sig_dim)

    h_bits = (sigs @ hp.T) >= 0
    h_codes = np.zeros(n_users, dtype=np.int32)
    for t in range(b):
        h_codes += h_bits[:, t].astype(np.int32) * (2 ** (b - 1 - t))
    B = 2 ** b

    # interaction load per cell
    cell_load = np.zeros(B, dtype=np.float64)
    for u, items in train_items.items():
        cell_load[h_codes[u]] += len(items)

    # greedy: repeatedly give the heaviest cell to the currently lightest shard
    order = np.argsort(-cell_load)
    cell_to_shard = np.zeros(B, dtype=np.int32)
    shard_load = np.zeros(n_shards, dtype=np.float64)
    for c in order:
        s = int(np.argmin(shard_load))
        cell_to_shard[c] = s
        shard_load[s] += cell_load[c]

    umap = cell_to_shard[h_codes]
    counts = np.bincount(umap, minlength=n_shards)
    print(f"    [PROPOSED_BAL] b={b} bits, {B} cells -> {n_shards} shards, "
          f"users min={counts.min()} max={counts.max()} mean={counts.mean():.1f}")
    return umap


# ----------------------------------------------------------------------------
# instability rate  (spec S2)
# ----------------------------------------------------------------------------
def instability_rate(before, after, retained_ids, n_shards):
    """Fraction of retained users whose shard changed after re-running the
    partition on the post-deletion data.

    Two numbers are reported, because a clustering can relabel its shards:

      label_matched : the shard labels are first aligned with the Hungarian
                      algorithm, so a pure permutation of the labels does not
                      count as instability. This is the number to quote.

      pairwise      : label-invariant. The fraction of retained user PAIRS whose
                      co-location flipped (same shard before, different after,
                      or the other way round). A permutation of the labels
                      cannot change this, so it is the conservative check.

    Returns (label_matched_pct, pairwise_pct).
    """
    before = np.asarray(before, dtype=np.int64)
    after = np.asarray(after, dtype=np.int64)
    retained_ids = np.asarray(retained_ids, dtype=np.int64)

    if len(retained_ids) == 0:
        return float('nan'), float('nan')

    # ---- label-matched: align the after-labels to the before-labels ----
    # co-occurrence matrix rows = before shard, cols = after shard
    M = np.zeros((n_shards, n_shards), dtype=np.int64)
    b = before[retained_ids]
    a = after[retained_ids]
    np.add.at(M, (b, a), 1)
    row, col = linear_sum_assignment(-M)
    mapping = dict(zip(col.tolist(), row.tolist()))   # after-shard -> before-shard
    a_aligned = np.array([mapping.get(int(x), int(x)) for x in a], dtype=np.int64)
    changed = float((a_aligned != b).mean() * 100.0)

    # ---- pairwise: label-invariant ----
    # Compare the agreement of the two partitions on all retained user pairs.
    # Subsample for tractability: the statistic is a mean over pairs, so a
    # random sample of pairs gives the same expectation at a fraction of the
    # cost.
    n_ret = len(retained_ids)
    rng = np.random.RandomState(0)
    n_pairs = 200000
    n_pairs = min(n_pairs, n_ret * (n_ret - 1) // 2)
    if n_pairs <= 0:
        return changed, float('nan')
    i = rng.randint(0, n_ret, size=n_pairs)
    j = rng.randint(0, n_ret, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    same_before = (b[i] == b[j])
    same_after = (a[i] == a[j])
    flip = float((same_before != same_after).mean() * 100.0)

    return changed, flip


def _rebuild_partition(arm, retained, n_users, n_items, args, seed, data_root):
    """Re-run the partition of `arm` on the retained data.

    This is what the instability rate measures: the partition is NOT reused,
    it is recomputed from scratch on the data that remains after the deletion.
    Each arm rebuilds its own partition the same way it built the first one,
    so the comparison is between like and like.
    """
    if arm == 'PROPOSED':
        return partition_proposed(retained, n_users, n_items, args.n_shards,
                                  args.sig_dim, args.legacy_seed,
                                  router='hyperplane',
                                  sig_weighting=args.sig_weighting)
    if arm == 'PROPOSED_MINHASH':
        return partition_proposed(retained, n_users, n_items, args.n_shards,
                                  args.sig_dim, args.legacy_seed,
                                  router='minhash',
                                  sig_weighting=args.sig_weighting)
    if arm == 'PROPOSED_BAL':
        return partition_proposed_balanced(retained, n_users, n_items,
                                           args.n_shards, args.sig_dim,
                                           args.legacy_seed)
    if arm == 'RANDOM_EMB':
        return partition_random_embedding(retained, n_users, n_items,
                                          args.n_shards, args.sig_dim, seed,
                                          router=args.router)
    if arm == 'RANDOM_PERM':
        return pv_random_perm(n_users, args.n_shards, seed)
    if arm == 'RANDOM_BAL':
        return random_perm_balanced(retained, n_users, args.n_shards, seed)
    if arm in ('RANDOM_MODULO', 'SISA'):
        return pv_random_modulo(n_users, args.n_shards, seed)
    if arm == 'INP':
        return partition_embedding_based(retained, n_users, n_items,
                                         args.n_shards, seed, kind='inp',
                                         data_root=data_root)
    if arm == 'UBP':
        return partition_embedding_based(retained, n_users, n_items,
                                         args.n_shards, seed, kind='ubp',
                                         data_root=data_root)
    if arm == 'RECERASER':
        from exp_receraser_arm import receraser_ubp_partition
        return receraser_ubp_partition(retained, n_users, n_items,
                                       args.n_shards, seed,
                                       data_root=data_root,
                                       kind=args.receraser_partition)
    raise ValueError(f'cannot rebuild partition for arm {arm}')


def partition_embedding_based(train_items, n_users, n_items, n_shards, seed,
                              kind='inp', data_root=None):
    """Embedding-based partition (InP / UBP) as an UPPER REFERENCE.

    The spec asks for an embedding-based partition as a reference point for
    partition quality; it is explicitly NOT part of the exact claim, which is
    PROPOSED vs RANDOM. This is the frozen version of RecEraser's partition:
    k-means over WMF embeddings, then the mapping is held fixed, so the
    unlearning mechanism stays identical to every other arm here.

      kind='inp' : k-means over (user emb, item-weighted profile)
      kind='ubp' : k-means over user embeddings only
    """
    import numpy as _np
    from sklearn.cluster import KMeans

    if data_root is None:
        data_root = os.path.join(os.path.dirname(PROJ), 'data')
    emb_path = os.path.join(data_root, 'wmf_embeddings.npz')
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f'WMF embeddings not found at {emb_path}. Train them first with '
            f'code/train_wmf_pretrain.py --dataset <ds>')

    data = _np.load(emb_path)
    user_emb = data['user_embeddings']
    item_emb = data['item_embeddings']
    u = user_emb[:n_users]
    u = u / _np.maximum(_np.linalg.norm(u, axis=1, keepdims=True), 1e-10)

    if kind == 'inp':
        # interaction-based: append the (normalised) item-embedding centroid
        # of each user's consumed items
        prof = _np.zeros((n_users, item_emb.shape[1]), dtype=_np.float32)
        for uid, its in train_items.items():
            if uid >= n_users:
                continue
            idx = [i for i in its if i < item_emb.shape[0]]
            if idx:
                prof[uid] = item_emb[idx].mean(axis=0)
        prof = prof / _np.maximum(_np.linalg.norm(prof, axis=1, keepdims=True), 1e-10)
        feats = _np.hstack([u, prof])
    else:
        feats = u

    km = KMeans(n_clusters=n_shards, n_init=10, random_state=seed).fit(feats)
    umap = km.labels_.astype(_np.int32)
    counts = _np.bincount(umap, minlength=n_shards)
    print(f"    [{kind.upper()}-frozen] k-means over WMF -> users "
          f"min={counts.min()} max={counts.max()} mean={counts.mean():.1f}")
    return umap


# ----------------------------------------------------------------------------
# partition-quality metrics
# ----------------------------------------------------------------------------
def _l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-10)


def intra_similarity(user_emb, user_to_shard, n_shards, max_pairs=200000, seed=42):
    """Mean cosine similarity of within-shard user pairs (sampled)."""
    rng = np.random.RandomState(seed)
    pairs = []
    for s in range(n_shards):
        idx = np.where(user_to_shard == s)[0]
        if len(idx) < 2:
            continue
        # sample pairs within this shard
        n_pairs = min(len(idx) * (len(idx) - 1) // 2, max_pairs // max(1, n_shards) + 1)
        if n_pairs <= 0:
            continue
        a = rng.choice(idx, size=n_pairs)
        b = rng.choice(idx, size=n_pairs)
        keep = a != b
        pairs.append(np.stack([a[keep], b[keep]], axis=1))
    if not pairs:
        return float('nan')
    P = np.concatenate(pairs, axis=0)
    if len(P) > max_pairs:
        P = P[rng.choice(len(P), size=max_pairs, replace=False)]
    u = _l2norm(user_emb[P[:, 0]])
    v = _l2norm(user_emb[P[:, 1]])
    return float(np.mean(np.sum(u * v, axis=1)))


def neighbour_retention(user_emb, user_to_shard, q=20, max_users=2000, seed=42):
    """NR@q: mean fraction of a user's q nearest neighbours in the same shard."""
    rng = np.random.RandomState(seed)
    U = _l2norm(user_emb)
    users = np.where(user_to_shard >= 0)[0]
    if len(users) > max_users:
        users = rng.choice(users, size=max_users, replace=False)

    vals = []
    for start in range(0, len(users), 256):
        block = users[start:start + 256]
        sims = U[block] @ U.T                      # (b, n_users)
        # exclude self
        sims[np.arange(len(block)), block] = -np.inf
        k = min(q + 1, sims.shape[1] - 1)
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        for bi, u in enumerate(block):
            neigh = top[bi]
            neigh = neigh[neigh != u]
            neigh = neigh[:q]
            if len(neigh) == 0:
                continue
            vals.append(float(np.mean(user_to_shard[neigh] == user_to_shard[u])))
    return float(np.mean(vals)) if vals else float('nan')


def load_balance(user_to_shard, train_items, n_shards):
    users_per = np.bincount(user_to_shard, minlength=n_shards).astype(np.float64)
    inter_per = np.zeros(n_shards, dtype=np.float64)
    for u, items in train_items.items():
        inter_per[user_to_shard[u]] += len(items)
    ucv = users_per.std() / max(users_per.mean(), 1e-9)
    icv = inter_per.std() / max(inter_per.mean(), 1e-9)
    mmr = inter_per.max() / max(inter_per.mean(), 1e-9)
    return float(ucv), float(icv), float(mmr)


# ----------------------------------------------------------------------------
# train / unlearn / evaluate
# ----------------------------------------------------------------------------
def evaluate_sisa_mean_aggregation(shard_models, train_data, test_data,
                                   n_users, n_items, device, Ks=KS):
    """Evaluate SISA the way SISA does it: mean aggregation over all shards.

    The SISA paper splits the users over K shards, trains one model per
    shard, and at inference time averages the scores of all K models. The
    arm SISA in this experiment used to be evaluated with hard routing, which
    is not SISA: it drops the aggregation that is the defining part of the
    method. This is the original aggregation, ported from method_2_sisa.py.

    A user that lives in shard j is trained in shard j's model; the other
    models still hold a row for that user, which stays at its initial value.
    Averaging therefore mixes the trained representation with K-1 untrained
    ones, which is exactly what the original method does.
    """
    import heapq
    n_shards = len(shard_models.models)
    if fast_train.use_gpu_engine():
        live = [shard_models.get_model(s) for s in range(n_shards)]
        live = [m for m in live if m is not None]

        def score_block(ut):
            sc = torch.zeros((len(ut), n_items), device=device)
            for m in live:
                sc += m.user_embedding.weight[ut] @ m.item_embedding.weight.T
            return sc / max(n_shards, 1)
        res = fast_train.evaluate_topk(score_block, list(range(n_users)), train_data,
                                       test_data, n_items, device, Ks)
        return {'recall': res['recall'], 'ndcg': res['ndcg']}
    rec = {k: [] for k in Ks}
    ndcg = {k: [] for k in Ks}

    with torch.no_grad():
        for user in range(n_users):
            if user not in test_data or not test_data[user]:
                continue
            scores_sum = np.zeros(n_items)
            for sid in range(n_shards):
                model = shard_models.get_model(sid)
                if model is None:
                    continue
                user_t = torch.LongTensor([user]).to(device)
                scores = []
                for i in range(0, n_items, 256):
                    batch_items = torch.arange(i, min(i + 256, n_items),
                                               device=device)
                    s = model.predict_user_items(user_t, batch_items).cpu().numpy()
                    scores.extend(s.tolist())
                scores_sum += np.array(scores)
            scores = scores_sum / max(n_shards, 1)

            for it in set(train_data.get(user, [])):
                scores[it] = -np.inf

            rank_list = heapq.nlargest(max(Ks), range(len(scores)),
                                       key=scores.__getitem__)
            item_set = set(test_data[user])
            for k in Ks:
                hit = rank_list[:k]
                hits = len(set(hit) & item_set)
                rec[k].append(hits / len(item_set) if item_set else 0)
                dcg = sum(1.0 / np.log2(i + 2) for i, x in enumerate(hit)
                          if x in item_set)
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(item_set), k)))
                ndcg[k].append(dcg / idcg if idcg > 0 else 0)

    return {
        'recall': [float(np.nanmean(rec[k])) if rec[k] else 0.0 for k in Ks],
        'ndcg': [float(np.nanmean(ndcg[k])) if ndcg[k] else 0.0 for k in Ks],
    }


def _mean(vals):
    """Mean of a list, or None if it is empty."""
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def build_shard_data(train_items, user_to_shard, n_shards):
    shard_data = [{} for _ in range(n_shards)]
    for u, items in train_items.items():
        shard_data[user_to_shard[u]][u] = list(items)
    return shard_data


def run_sharded_arm(train_items, test_data, n_users, n_items, n_shards,
                    emb_dim, user_to_shard, del_users, epochs, lr, batch_size,
                    device, seed, retrain_epochs, reset_on_retrain=False,
                    eval_mode='hard', cache=None):
    """Train K shard models, delete users, retrain affected shards, evaluate.

    The per-shard retrain times and interaction counts are returned so the
    caller can report them and check that they add up to unlearn_time
    (spec S5). parallel_time is the wall time if the affected shards were
    retrained concurrently, i.e. the slowest one.
    """
    set_all_seeds(seed)
    shard_data = build_shard_data(train_items, user_to_shard, n_shards)
    shard_models = ShardModels(n_users, n_items, emb_dim, n_shards)

    if cache and 'states' in cache:
        # one model for every deletion: start again from the trained shards
        for sid in range(n_shards):
            shard_models.create_model(sid, device).load_state_dict(cache['states'][sid])
        train_time = cache['train_time']
        # the retrain of this deletion gets its own random stream: with the
        # same seed every repeat would retrain identically (most/fewest even
        # delete the same users), and the 5 deletions would be one
        set_all_seeds(cache['rng_seed'])
        print("       model: the trained shards are reused (trained once), "
              f"train time {train_time:.1f}s from that training")
    else:
        t0 = time.time()
        for sid in range(n_shards):
            model = shard_models.create_model(sid, device)
            train_shard_model(model, shard_data[sid], n_items, device,
                              batch_size=batch_size, lr=lr, n_epochs=epochs)
        train_time = time.time() - t0
        if cache is not None:
            cache['states'] = [{k: v.detach().clone() for k, v in
                                shard_models.get_model(sid).state_dict().items()}
                               for sid in range(n_shards)]
            cache['train_time'] = train_time

    # ---- unlearn: remove deleted users, retrain ONLY affected shards ----
    # reset_on_retrain controls whether the affected shard starts from a FRESH
    # model or from its current weights.
    #
    # The fresh model is the one the spec asks for ("retrain from scratch the
    # submodels whose shard holds a deleted user"). It is also the only one
    # that removes the deleted user completely: training from the current
    # weights leaves the deleted user's embedding row in the table, because
    # the retrain only touches the users that are still present, so the
    # deleted user's vector would survive the deletion.
    #
    # The old behaviour (keep the weights) is kept behind the flag because
    # the earlier logs were produced with it, and those logs are the baseline
    # the new numbers are compared against.
    affected = sorted({int(user_to_shard[u]) for u in del_users if u < len(user_to_shard)})
    retained = {u: it for u, it in train_items.items() if u not in del_users}
    retained_shard_data = build_shard_data(retained, user_to_shard, n_shards)

    per_shard_time = {}
    per_shard_inter = {}
    t0 = time.time()
    for sid in affected:
        model = shard_models.get_model(sid)
        if reset_on_retrain:
            model = shard_models.create_model(sid, device)
        _, elapsed, n_samples = train_shard_model(
            model, retained_shard_data[sid], n_items, device,
            batch_size=batch_size, lr=lr, n_epochs=retrain_epochs)
        per_shard_time[sid] = elapsed
        per_shard_inter[sid] = n_samples
    unlearn_time = time.time() - t0
    parallel_time = max(per_shard_time.values()) if per_shard_time else 0.0

    # ---- evaluate on retained test users only ----
    test_retained = {u: it for u, it in test_data.items() if u not in del_users}
    if eval_mode == 'sisa_mean':
        # the original SISA inference: average the scores of all K models
        res = evaluate_sisa_mean_aggregation(shard_models, retained,
                                             test_retained, n_users, n_items,
                                             device, Ks=KS)
    else:
        res = evaluate_hard_routing(shard_models, user_to_shard, retained,
                                     test_retained, n_users, n_items, device,
                                     Ks=KS)
    return (res, train_time, unlearn_time, affected, shard_models,
            parallel_time, per_shard_time, per_shard_inter)


def _train_bprmf(train_items, n_users, n_items, emb_dim, epochs, lr,
                 batch_size, device, seed):
    """Train one BPRMF on the given interactions.

    Returns (model, elapsed_seconds, n_samples, per_epoch_seconds).

    The per-epoch times are what make the cost comparable across arms: the
    seconds per epoch divided by the number of interactions is a rate that
    should be the same for every shard and for the global model. If a small
    shard takes as long per epoch as the global model, the training loop is
    iterating over more than that shard's data, and the reported cost would
    be measuring the wrong thing.
    """
    from method_4_ours import BPRMF
    set_all_seeds(seed)
    if wmf_backbone.use_wmf():
        model = wmf_backbone.WMF(n_users, n_items, emb_dim).to(device)
        if not any(train_items.values()):
            return model, 0.0, 0, []
        per_epoch, t0 = [], time.time()
        last = [t0]

        def epoch_end(ep):
            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
            now = time.time()
            per_epoch.append(now - last[0])
            last[0] = now
        _, n = wmf_backbone.train_wmf(lambda u: model.user_embedding(u),
                                      lambda: model.item_embedding.weight,
                                      model.parameters(), train_items, device, lr,
                                      epochs, epoch_end=epoch_end)
        return model, time.time() - t0, n, per_epoch
    model = BPRMF(n_users, n_items, emb_dim).to(device)
    opt = torch.optim.Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    if fast_train.use_gpu_engine():
        samples = fast_train.draw_samples(train_items, n_items)
        if not samples[0]:
            return model, 0.0, 0, []
        per_epoch = []
        t0 = time.time()
        last = [t0]

        def step(u, p, n):
            loss = model(u, p, n)
            opt.zero_grad()
            loss.backward()
            opt.step()
            return loss

        def epoch_end(ep):
            fast_train.sync(device)
            now = time.time()
            per_epoch.append(now - last[0])
            last[0] = now
        fast_train.train_epochs(step, samples, device, batch_size, epochs,
                                epoch_end=epoch_end)
        fast_train.sync(device)
        return model, time.time() - t0, len(samples[0]), per_epoch

    samples = []
    for u, items in train_items.items():
        for pos in items:
            neg = random.randint(0, n_items - 1)
            while neg in items:
                neg = random.randint(0, n_items - 1)
            samples.append((u, pos, neg))
    if not samples:
        return model, 0.0, 0, []

    per_epoch = []
    t0 = time.time()
    for ep in range(epochs):
        random.shuffle(samples)
        ep_t0 = time.time()
        nb = max(1, len(samples) // batch_size)
        for b in range(nb):
            chunk = samples[b * batch_size:(b + 1) * batch_size]
            if not chunk:
                continue
            us = torch.LongTensor([c[0] for c in chunk]).to(device)
            ps = torch.LongTensor([c[1] for c in chunk]).to(device)
            ns = torch.LongTensor([c[2] for c in chunk]).to(device)
            loss = model(us, ps, ns)
            opt.zero_grad()
            loss.backward()
            opt.step()
        per_epoch.append(time.time() - ep_t0)
    return model, time.time() - t0, len(samples), per_epoch


def run_global_arm(train_items, test_data, n_users, n_items, del_users,
                   emb_dim, epochs, lr, batch_size, device, seed, cache=None):
    """Retrain reference: one BPRMF on the whole retained dataset.

    The global model has no shards, so unlearning it means retraining a fresh
    model on the data with the deleted users removed. Both models are trained
    here so that the two costs can be separated:

      train_time    : training the ORIGINAL model on the full dataset
      unlearn_time  : training the RETRAINED model on the reduced dataset

    The old code trained only once, on the reduced data, and returned that
    single duration for both fields - so the cost of the retrain was never
    measured and train_time and unlearn_time were identical by construction.
    The evaluation uses the retrained model, which is the model that would
    actually be deployed after the deletion request.

    The per-epoch times of both models are returned as well, so the retrain
    can be compared with the shard retrains on a rate rather than on a total.
    """
    retained = {u: it for u, it in train_items.items() if u not in del_users}

    # ---- the original model, on the full dataset ----
    # (with one model for every deletion it is trained once and its time
    # reused: the retrain below does not start from it)
    if cache and 'train_time' in cache:
        train_time, n_full, ep_full = cache['train_time'], cache['n_full'], cache['ep_full']
        cache['reused'] = True
        print("       model: the original model is reused (trained once), "
              f"train time {train_time:.1f}s from that training")
    else:
        _, train_time, n_full, ep_full = _train_bprmf(train_items, n_users, n_items,
                                                       emb_dim, epochs, lr,
                                                       batch_size, device, seed)
        if cache is not None:
            cache.update(train_time=train_time, n_full=n_full, ep_full=ep_full)
    # ---- the retrained model, on the data with the deleted users removed ----
    rseed = cache['rng_seed'] if cache and 'reused' in cache else seed
    model, unlearn_time, n_ret, ep_ret = _train_bprmf(
        retained, n_users, n_items, emb_dim, epochs, lr, batch_size, device, rseed)

    test_retained = {u: it for u, it in test_data.items() if u not in del_users}
    if fast_train.use_gpu_engine():
        res = fast_train.evaluate_topk(
            lambda ut: model.user_embedding.weight[ut] @ model.item_embedding.weight.T,
            list(range(n_users)), retained, test_retained, n_items, device, KS)
        out = {'recall': res['recall'], 'ndcg': res['ndcg']}
        return (out, train_time, unlearn_time, unlearn_time, n_full, n_ret,
                ep_full, ep_ret)
    rec, ndcg = {k: [] for k in KS}, {k: [] for k in KS}
    import heapq
    with torch.no_grad():
        for user in range(n_users):
            if user not in test_retained or not test_retained[user]:
                continue
            scores = model.predict_user_items(
                user, torch.arange(n_items, device=device)).cpu().numpy()
            for it in set(retained.get(user, [])):
                scores[it] = -np.inf
            rank_list = heapq.nlargest(max(KS), range(len(scores)), key=scores.__getitem__)
            item_set = set(test_retained[user])
            for k in KS:
                hit_list = rank_list[:k]
                hit_num = len(set(hit_list) & item_set)
                rec[k].append(hit_num / len(item_set) if item_set else 0)
                dcg = sum(1.0 / np.log2(i + 2) for i, x in enumerate(hit_list) if x in item_set)
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(item_set), k)))
                ndcg[k].append(dcg / idcg if idcg > 0 else 0)
    out = {'recall': [np.nanmean(rec[k]) if rec[k] else 0 for k in KS],
           'ndcg': [np.nanmean(ndcg[k]) if ndcg[k] else 0 for k in KS]}
    # the retrain is a single model, so parallel_time == unlearn_time
    return (out, train_time, unlearn_time, unlearn_time, n_full, n_ret,
            ep_full, ep_ret)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ml-1m')
    ap.add_argument('--emb_dim', type=int, default=64)
    ap.add_argument('--sig_dim', type=int, default=64)
    ap.add_argument('--n_shards', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--retrain_epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--unlearn_ratio', type=float, default=0.01)
    ap.add_argument('--unlearn_mode', default='most')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--train_seed', type=int, default=-1,
                    help='>= 0: train every arm ONCE with this seed and use --seeds '
                         'only to pick the deleted users (each deletion starts from '
                         'that one model). -1 (default): one model per seed.')
    ap.add_argument('--arms', default='PROPOSED,PROPOSED_BAL,RANDOM_EMB,RANDOM_PERM,RANDOM_BAL,RANDOM_MODULO,SISA,GLOBAL')
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--backbone', default='bprmf', choices=['bprmf', 'wmf'],
                    help='the model every arm trains: bprmf, or wmf (the authors\' '
                         'WMF.py loss, see wmf_backbone.py); nothing else changes')
    ap.add_argument('--wmf_c0', type=float, default=None,
                    help='WMF negative weight c0 (default 0.2 ml-1m, 0.05 yelp2018)')
    ap.add_argument('--wmf_keep', type=float, default=0.7, help='WMF dropout keep probability')
    ap.add_argument('--wmf_batch_users', type=int, default=256, help='WMF users per batch')
    ap.add_argument('--skip_runs', default='',
                    help='ARM:seed,ARM:seed,... already finished in an earlier, '
                         'interrupted run of this log; they are not run again. '
                         'run_blocks.sh passes them when it resumes a log.')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--legacy_seed', type=int, default=42,
                    help='Seed used INSIDE the partition builders (signature matrix R and '
                         'LSH hyperplanes). The original runs hard-code 42; keep it to stay '
                         'comparable with the existing logs.')
    ap.add_argument('--legacy_mode', action='store_true',
                    help='Pick the deletion set exactly like the original method_4_ours.py: '
                         'sort by interaction count and take the top n (mode=most/fewest), '
                         'ignoring the run seed. Use this to reproduce an existing log '
                         'exactly; otherwise the deletion set varies per seed as the '
                         'experiment spec requires.')
    ap.add_argument('--router', default='hyperplane',
                    choices=['hyperplane', 'minhash'],
                    help='LSH router for the PROPOSED partition (spec ablation 8). '
                         'hyperplane = shipped cosine LSH; minhash = Jaccard-style.')
    ap.add_argument('--sig_weighting', default='binary',
                    choices=['binary', 'log_count'],
                    help='User profile weighting for the signature (spec ablation 8).')
    ap.add_argument('--instability', type=int, default=1,
                    help='Measure the instability rate: re-run the partition on '
                         'the retained data and report the %% of retained users '
                         'whose shard changed (spec S2).')
    ap.add_argument('--receraser_partition', default='ubp',
                    choices=['ubp', 'inp', 'random'],
                    help='Partition used by the RECERASER arm: ubp = '
                         'user-based partition over WMF user embeddings, inp = '
                         'user + item-profile centroid, random = a random '
                         'equal split (the control).')
    ap.add_argument('--receraser_init', default='random',
                    choices=['random', 'wmf'],
                    help='How the RECERASER shards start: random (the original '
                         'RecEraser and every other arm) or wmf (a warm start '
                         'from the WMF vectors, needs emb_dim == 64).')
    ap.add_argument('--agg_epochs', type=int, default=50,
                    help='Epochs for the RecEraser attention aggregator.')
    ap.add_argument('--keep_weights_on_retrain', action='store_true',
                    help='When retraining an affected shard, start from its '
                         'current weights instead of a fresh model. This '
                         'reproduces the earlier logs, where the retrain '
                         'never touched the deleted user embedding row. The '
                         'default (a fresh model) is the one that actually '
                         'removes the user, as the spec requires.')
    args = ap.parse_args()

    data_root = args.data_root or os.path.join(os.path.dirname(PROJ), 'data')
    wmf_backbone.configure(args.backbone, args.wmf_c0, args.wmf_keep,
                           args.wmf_batch_users, dataset=args.dataset)
    tag = args.tag or f'{args.dataset}_K{args.n_shards}'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_items, test_data, n_users, n_items = load_data(args.dataset, data_root)
    print(f"  {args.dataset}: {n_users} users, {n_items} items, device={device}")
    # From here on data_root only locates the dataset's own files (the WMF
    # embeddings of the RECERASER partition), which live in data/<dataset>/,
    # not in the parent data/ that load_data takes.
    data_root = os.path.join(data_root, args.dataset)
    emb = os.path.join(data_root, 'wmf_embeddings.npz')
    print(f"  WMF embeddings: {emb} ({'found' if os.path.exists(emb) else 'MISSING'})")
    print(f"  engine: {fast_train.engine_name()}")
    print(f"  backbone: {wmf_backbone.describe()}")

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    skip = {(a, int(s)) for a, s in (x.split(':') for x in args.skip_runs.split(',') if x)}
    if skip:
        print(f"  resuming: {len(skip)} run(s) already done earlier, not run again")
    arms = [a.strip() for a in args.arms.split(',') if a.strip()]

    # Results are printed to the console as they are produced; no CSV is written.
    RESULTS = []

    # One model (--train_seed >= 0): each arm is trained ONCE with that seed
    # and every seed in --seeds only picks a different set of deleted users;
    # each deletion starts again from the same trained model (the unaffected
    # shards keep the original weights, the affected ones are retrained from
    # scratch). Otherwise every seed trains its own model, as before.
    one_model = args.train_seed >= 0
    # The signature the partition-quality metrics are measured on does not
    # depend on the run seed (legacy_seed), so it is built once for every arm.
    # The metrics use this ORIGINAL signature - not the post-train
    # embeddings, which carry no useful user similarity yet at few epochs
    # (NR@q would read ~1.0 for every arm and tell us nothing).
    sigs_for_metrics = pv_build_signature(
        train_items, n_users, n_items, d=args.sig_dim,
        seed=args.legacy_seed, weighting=args.sig_weighting, verbose=False)
    if one_model:
        print(f"  one model: every arm trained once (seed {args.train_seed}), then "
              f"unlearned {len(seeds)} times from that model "
              f"(deletion seeds {','.join(map(str, seeds))})")
        pairs = [(a, s) for a in arms for s in seeds]
    else:
        pairs = [(a, s) for s in seeds for a in arms]

    dels_by_seed, caches, shown = {}, {}, None
    for arm, seed in pairs:
        if (arm, seed) in skip:
            continue
        tseed = args.train_seed if one_model else seed
        cache = None
        if one_model:
            if arm not in caches:
                caches.clear()            # the previous arm's model is done
            cache = caches.setdefault(arm, {})
            cache['rng_seed'] = seed      # the retrain of a reused model
        if seed not in dels_by_seed:
            if args.legacy_mode:
                # exactly reproduce the original method_4_ours.py deletion set
                ui = sorted(train_items.items(), key=lambda kv: len(kv[1]), reverse=True)
                dels_by_seed[seed] = {u for u, _ in ui[:int(len(train_items) * args.unlearn_ratio)]}
            else:
                dels_by_seed[seed] = pick_deletion_users(
                    train_items, args.unlearn_ratio, args.unlearn_mode, seed)
        del_users = dels_by_seed[seed]
        if seed != shown or one_model:
            shown = seed
            print(f"\n{'='*70}\nSEED {seed} | deleted {len(del_users)} users "
                  f"({args.unlearn_ratio*100:.1f}%, {args.unlearn_mode})"
                  + (f" | model trained with seed {tseed}" if one_model else "")
                  + f"\n{'='*70}")

        print(f"\n--- ARM {arm} (seed {seed}) ---")
        bg0 = gpu_probe.background_load()
        row = {
            'dataset': args.dataset, 'seed': seed, 'arm': arm, 'K': args.n_shards,
            'deletion_ratio': args.unlearn_ratio, 'deletion_mode': args.unlearn_mode,
            'backbone': 'BPRMF', 'embedding_dim': args.emb_dim,
            'epochs': args.epochs, 'learning_rate': args.lr,
            'regularization': 0.01,
        }

        if arm == 'GLOBAL':
            res, tt, ut, ptime, n_full, n_ret, ep_full, ep_ret = run_global_arm(
                train_items, test_data, n_users, n_items, del_users,
                args.emb_dim, args.epochs, args.lr, args.batch_size, device, tseed,
                cache=cache)
            row.update({
                'partition': 'none', 'train_time': tt, 'unlearn_time': ut,
                'parallel_time': ptime,
                'affected_shards': 1, 'intra_similarity': '',
                'nr_10': '', 'nr_20': '', 'nr_50': '',
                'user_load_cv': '', 'interaction_load_cv': '', 'max_mean_ratio': '',
                'instability_label': '', 'instability_pairwise': '',
                'shard_interactions_total': n_full,
                'shard_interactions_unlearn': n_ret,
                'sec_per_epoch_global': (_mean(ep_ret) if ep_ret else ''),
                'sec_per_interaction_global': (
                    (_mean(ep_ret) / n_ret) if ep_ret and n_ret else ''),
            })
            per_shard_time, per_shard_inter = {}, {}
        else:
            if arm == 'PROPOSED':
                # legacy_seed keeps the signature matrix R and the LSH
                # hyperplanes identical to the earlier runs (seed 42),
                # so PROPOSED is directly comparable with the old logs.
                #
                # The router is FIXED to hyperplane here (spec S6): the old
                # code took it from --router, so running the router ablation
                # with --router minhash turned BOTH this arm and
                # PROPOSED_MINHASH into minhash and the two arms came out
                # identical. PROPOSED_MINHASH now carries the variant.
                umap = partition_proposed(train_items, n_users, n_items,
                                          args.n_shards, args.sig_dim,
                                          args.legacy_seed,
                                          router='hyperplane',
                                          sig_weighting=args.sig_weighting)
                pname = f'signature_hyperplane_{args.sig_weighting}'
            elif arm == 'PROPOSED_MINHASH':
                umap = partition_proposed(train_items, n_users, n_items,
                                          args.n_shards, args.sig_dim,
                                          args.legacy_seed,
                                          router='minhash',
                                          sig_weighting=args.sig_weighting)
                pname = f'signature_minhash_{args.sig_weighting}'
            elif arm == 'PROPOSED_BAL':
                umap = partition_proposed_balanced(
                    train_items, n_users, n_items, args.n_shards,
                    args.sig_dim, args.legacy_seed)
                pname = 'signature_lsh_balanced'
            elif arm == 'RANDOM_EMB':
                # control for the signature: random user embedding routed
                # through the SAME LSH router as Ours-signature
                umap = partition_random_embedding(
                    train_items, n_users, n_items, args.n_shards,
                    args.sig_dim, tseed, router=args.router)
                pname = f'random_embedding_{args.router}'
            elif arm == 'RANDOM_PERM':
                umap = pv_random_perm(n_users, args.n_shards, tseed)
                pname = 'random_perm'
            elif arm == 'RANDOM_BAL':
                # random, greedy-balanced by interaction load: the control
                # that removes the "one large shard ~= global model" confound
                umap = random_perm_balanced(train_items, n_users,
                                            args.n_shards, tseed)
                pname = 'random_balanced'
            elif arm == 'RANDOM_MODULO':
                umap = pv_random_modulo(n_users, args.n_shards, tseed)
                pname = 'random_modulo'
            elif arm == 'SISA':
                umap = pv_random_modulo(n_users, args.n_shards, tseed)
                pname = 'sisa_modulo'
            elif arm == 'INP':
                # embedding-based upper reference (frozen k-means over WMF)
                umap = partition_embedding_based(train_items, n_users, n_items,
                                                 args.n_shards, tseed,
                                                 kind='inp',
                                                 data_root=data_root)
                pname = 'inp_frozen'
            elif arm == 'UBP':
                umap = partition_embedding_based(train_items, n_users, n_items,
                                                 args.n_shards, tseed,
                                                 kind='ubp',
                                                 data_root=data_root)
                pname = 'ubp_frozen'
            elif arm == 'RECERASER':
                # RecEraser: UBP partition over WMF embeddings + attention
                # aggregation. Structurally different from the other arms,
                # so it gets its own runner.
                from exp_receraser_arm import receraser_ubp_partition
                umap = receraser_ubp_partition(
                    train_items, n_users, n_items, args.n_shards, tseed,
                    data_root=data_root, kind=args.receraser_partition)
                pname = f'receraser_{args.receraser_partition}_attention'
            else:
                raise ValueError(f'unknown arm {arm}')

            if arm == 'RECERASER':
                from exp_receraser_arm import run_receraser_arm
                (res, tt, ut, affected, _sm, ptime,
                 per_shard_time, per_shard_inter) = run_receraser_arm(
                    train_items, test_data, n_users, n_items, args.n_shards,
                    args.emb_dim, umap, del_users, args.epochs, args.lr,
                    args.batch_size, device, tseed, args.retrain_epochs,
                    data_root=data_root, agg_epochs=args.agg_epochs,
                    init=args.receraser_init, cache=cache)
            else:
                # SISA is evaluated with its own mean aggregation, which
                # is what makes it SISA. Every other arm keeps the hard
                # routing the spec fixes for the experiment.
                eval_mode = ('sisa_mean' if arm == 'SISA' else 'hard')
                (res, tt, ut, affected, _sm, ptime,
                 per_shard_time, per_shard_inter) = run_sharded_arm(
                    train_items, test_data, n_users, n_items, args.n_shards,
                    args.emb_dim, umap, del_users, args.epochs, args.lr,
                    args.batch_size, device, tseed, args.retrain_epochs,
                    reset_on_retrain=not args.keep_weights_on_retrain,
                    eval_mode=eval_mode, cache=cache)

            # Partition-quality metrics are measured on the ORIGINAL
            # signature (built once per seed above), not on post-train
            # embeddings: the latter carry no useful user similarity at a
            # few epochs, so NR@q reads ~1.0 for every arm and proves nothing
            # about the partition. The signature is the quantity the
            # partition was actually designed to co-locate, so it is the
            # honest measure of co-location quality (spec section 6).
            isim = intra_similarity(sigs_for_metrics, umap,
                                    args.n_shards, seed=tseed)
            nr10 = neighbour_retention(sigs_for_metrics, umap, q=10, seed=tseed)
            nr20 = neighbour_retention(sigs_for_metrics, umap, q=20, seed=tseed)
            nr50 = neighbour_retention(sigs_for_metrics, umap, q=50, seed=tseed)
            ucv, icv, mmr = load_balance(umap, train_items, args.n_shards)

            # ---- instability rate (spec S2) ----
            # Re-run the partition on the RETAINED data, then measure how
            # many retained users end up in a different shard.
            inst_lab, inst_pair = '', ''
            if args.instability:
                retained_now = {u: it for u, it in train_items.items()
                                if u not in del_users}
                try:
                    umap_after = _rebuild_partition(
                        arm, retained_now, n_users, n_items, args,
                        tseed, data_root)
                    retained_ids = [u for u in range(n_users) if u not in del_users]
                    inst_lab, inst_pair = instability_rate(
                        umap, umap_after, retained_ids, args.n_shards)
                except Exception as exc:      # a partition that needs
                    inst_lab = inst_pair = f'n/a ({type(exc).__name__})'
                    print(f"       [instability] not measurable: {exc}")

            row.update({
                'partition': pname, 'train_time': tt, 'unlearn_time': ut,
                'parallel_time': ptime,
                'affected_shards': len(affected),
                'intra_similarity': isim, 'nr_10': nr10, 'nr_20': nr20, 'nr_50': nr50,
                'user_load_cv': ucv, 'interaction_load_cv': icv, 'max_mean_ratio': mmr,
                'instability_label': inst_lab, 'instability_pairwise': inst_pair,
                'shard_interactions_total': sum(per_shard_inter.values()),
                'shard_interactions_unlearn': sum(
                    per_shard_inter[s] for s in affected if s in per_shard_inter),
                'sec_per_epoch_global': '',
                'sec_per_interaction_global': '',
            })

        for i, k in enumerate(KS):
            row[f'recall_{k}'] = res['recall'][i]
            row[f'ndcg_{k}'] = res['ndcg'][i]

        RESULTS.append(row)
        print(f"  [{arm}] R@10={row['recall_10']:.4f} NDCG@10={row['ndcg_10']:.4f} "
              f"| R@20={row['recall_20']:.4f} NDCG@20={row['ndcg_20']:.4f} "
              f"| R@50={row['recall_50']:.4f} NDCG@50={row['ndcg_50']:.4f}")
        print(f"       cost: train={row['train_time']:.1f}s "
              f"unlearn={row['unlearn_time']:.1f}s "
              f"parallel={row.get('parallel_time', 0.0):.1f}s "
              f"shards={row['affected_shards']}")
        # Unlearning cost in training samples processed: the interactions
        # of every retrained shard times its epochs (plus the aggregator
        # pass for RecEraser, the whole retained set for the retrain).
        # Unlike the seconds above it does not depend on the hardware or
        # on what else runs on the GPU.
        if arm == 'GLOBAL':
            work = int(row.get('shard_interactions_unlearn') or 0) * args.epochs
        else:
            work = sum(per_shard_inter.values()) * args.retrain_epochs
            if arm == 'RECERASER':
                n_kept = sum(len(it) for u, it in train_items.items()
                             if u not in del_users)
                work += n_kept * args.agg_epochs
        bg1 = gpu_probe.background_load()
        row['unlearn_samples'] = work
        row['bg_load_before'], row['bg_load_after'] = bg0, bg1
        print(f"       work: unlearn_samples={work}")
        shared = any(v is not None and v >= gpu_probe.THRESHOLD for v in (bg0, bg1))
        print(f"       bg-load: before={gpu_probe.fmt(bg0)} after={gpu_probe.fmt(bg1)}"
              + ("  [SHARED GPU - other work was running, times may be inflated]"
                 if shared else ""))
        print(f"       partition quality: IntraSim={row['intra_similarity']} "
              f"NR@10={row['nr_10']} NR@20={row['nr_20']} NR@50={row['nr_50']}")
        print(f"       load: user_cv={row['user_load_cv']} "
              f"inter_cv={row['interaction_load_cv']} max/mean={row['max_mean_ratio']}")
        if row.get('instability_label') != '':
            print(f"       instability: label-matched={row['instability_label']}% "
                  f"pairwise={row['instability_pairwise']}%")
        if per_shard_time:
            print("       per-shard retrain (s): "
                  + ", ".join(f"s{s}={per_shard_time[s]:.1f}s/"
                              f"{per_shard_inter.get(s, 0)}int"
                              for s in sorted(per_shard_time))
                  + f"  | sum={sum(per_shard_time.values()):.1f}s "
                    f"vs unlearn_time={row['unlearn_time']:.1f}s")

            # S5 check: the time per epoch should scale with the number
            # of interactions a shard holds. If a small shard takes about
            # as long per epoch as a large one (or as the global model),
            # the training loop is iterating over more than that shard's
            # data.
            rate = {}
            for s in per_shard_time:
                n_int = per_shard_inter.get(s, 0)
                per_epoch = per_shard_time[s] / max(args.retrain_epochs, 1)
                rate[s] = (per_epoch / n_int) if n_int else float('nan')
            if len(rate) > 1:
                ok = [r for r in rate.values() if r == r]
                span = (max(ok) - min(ok)) / max(_mean(ok), 1e-12) \
                    if ok else 0.0
                flag = "OK" if span < 1.5 else \
                    "CHECK: per-epoch time does not scale with the " \
                    f"shard size (spread {span:.2f}x)"
                print(f"       S5 rate check: sec/epoch per interaction "
                      f"= " + ", ".join(f"s{s}={rate[s]:.2e}"
                                        for s in sorted(rate))
                      + f"  [{flag}]")
                if row.get('sec_per_epoch_global'):
                    print(f"       global (GLOBAL arm) "
                          f"sec/epoch/interaction="
                          f"{row.get('sec_per_interaction_global', ''):.2e}")

    print_console_summary(RESULTS, seeds, arms, args)


def print_console_summary(results, seeds, arms, args):
    """Print the per-arm summary and the paired PROPOSED-vs-RANDOM comparison."""
    import statistics

    print()
    print('=' * 78)
    print('CONSOLE SUMMARY')
    print('=' * 78)

    def get(arm, seed, key):
        for r in results:
            if r['arm'] == arm and int(r['seed']) == int(seed):
                v = r.get(key, '')
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return float('nan')
        return float('nan')

    metrics = ['recall_10', 'ndcg_10', 'recall_20', 'ndcg_20', 'recall_50', 'ndcg_50']
    print(f"\n{'arm':<18}" + ''.join(f'{m:>16}' for m in metrics))
    for arm in arms:
        line = f'{arm:<18}'
        for m in metrics:
            vals = [get(arm, s, m) for s in seeds]
            vals = [v for v in vals if v == v]
            if vals:
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                line += f'{statistics.mean(vals):.4f}+-{sd:.4f}'.rjust(16)
            else:
                line += ''.rjust(16)
        print(line)

    # partition quality + load balance + instability
    pqm = ['intra_similarity', 'nr_10', 'nr_20', 'nr_50',
           'user_load_cv', 'interaction_load_cv', 'max_mean_ratio']
    print(f"\n{'arm':<18}" + ''.join(f'{m[:13]:>15}' for m in pqm))
    for arm in arms:
        line = f'{arm:<18}'
        for m in pqm:
            vals = [get(arm, s, m) for s in seeds]
            vals = [v for v in vals if v == v]
            if vals:
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                line += f'{statistics.mean(vals):.4f}+-{sd:.4f}'.rjust(15)
            else:
                line += ''.rjust(15)
        print(line)

    # INSTABILITY RATE: % of retained users whose shard changed after the
    # partition was re-run on the remaining data. The label-matched number is
    # the one to quote; the pairwise number is label-invariant and is the
    # conservative check.
    inst = ['instability_label', 'instability_pairwise']
    print(f"\n{'arm':<18}" + ''.join(f'{m[:15]:>16}' for m in inst))
    for arm in arms:
        line = f'{arm:<18}'
        for m in inst:
            vals = [get(arm, s, m) for s in seeds]
            vals = [v for v in vals if v == v]
            if vals:
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                line += f'{statistics.mean(vals):.2f}+-{sd:.2f}'.rjust(16)
            else:
                line += ''.rjust(16)
        print(line)

    # COST: train time / unlearn time / parallel time / affected shards.
    # Kept in its own table because the spec asks for the cost of one
    # deletion alongside utility: a partition that wins on recall but has to
    # retrain every shard is not a win. affected_shards is the mechanistic
    # driver of unlearn cost; parallel_time is the wall time if the affected
    # shards were retrained concurrently.
    cost = ['train_time', 'unlearn_time', 'parallel_time', 'affected_shards']
    print(f"\n{'arm':<18}" + ''.join(f'{m[:14]:>16}' for m in cost))
    for arm in arms:
        line = f'{arm:<18}'
        for m in cost:
            vals = [get(arm, s, m) for s in seeds]
            vals = [v for v in vals if v == v]
            if vals:
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                line += f'{statistics.mean(vals):.2f}+-{sd:.2f}'.rjust(16)
            else:
                line += ''.rjust(16)
        print(line)

    # Print every meaningful paired comparison. The load-matched pair
    # (PROPOSED_BAL vs RANDOM_BAL) is the decisive one: it removes the
    # "one large shard ~= global model" confound, so any difference left is
    # attributable to the partition's similarity structure.
    available = {r['arm'] for r in results}
    comparisons = [
        ('PROPOSED', 'RANDOM_EMB', 'SIGNATURE vs random embedding (same router)'),
        ('PROPOSED_BAL', 'RANDOM_EMB', 'load-matched LSH vs random embedding'),
        ('PROPOSED', 'RANDOM_PERM', 'raw LSH vs random'),
        ('PROPOSED_BAL', 'RANDOM_BAL', 'LOAD-MATCHED: LSH vs random'),
        ('PROPOSED_MINHASH', 'RANDOM_PERM', 'minhash LSH vs random'),
        ('PROPOSED_BAL', 'RANDOM_PERM', 'load-matched LSH vs random perm'),
    ]

    for prop, base, label in comparisons:
        if prop not in available or base not in available:
            continue
        paired = [s for s in seeds
                  if any(r['arm'] == prop and int(r['seed']) == int(s) for r in results)
                  and any(r['arm'] == base and int(r['seed']) == int(s) for r in results)]
        if not paired:
            continue

        print(f"\nPAIRED: {prop} vs {base}   [{label}]   seeds={paired} (n={len(paired)})")
        for m in metrics:
            pairs = [(get(prop, s, m), get(base, s, m)) for s in paired]
            pairs = [(x, y) for x, y in pairs if x == x and y == y]
            if not pairs:
                continue
            p = [x for x, _ in pairs]
            b = [y for _, y in pairs]
            d = [x - y for x, y in pairs]
            md = statistics.mean(d)
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            rg = (statistics.mean([(x - y) / y * 100 for x, y in pairs])
                  if all(y != 0 for y in b) else float('nan'))
            # paired t-test on the per-seed differences (n>=3)
            pval = float('nan')
            if len(d) >= 3:
                try:
                    from scipy import stats as _st
                    pval = float(_st.ttest_1samp(d, 0.0).pvalue)
                except Exception:
                    pval = float('nan')
            ptag = '' if pval != pval else (f"  p={pval:.4f}" + (" *" if pval < 0.05 else ""))
            print(f'  {m:<12} {prop}={statistics.mean(p):.4f}  {base}={statistics.mean(b):.4f}  '
                  f'rel_gain={rg:+.2f}%  mean_delta={md:+.4f} (sd={sd:.4f}){ptag}')

        # mechanism check: does the partition actually co-locate similar users?
        for m in ['intra_similarity', 'nr_10', 'nr_20', 'nr_50']:
            pv = [x for x in [get(prop, s, m) for s in paired] if x == x]
            bv = [x for x in [get(base, s, m) for s in paired] if x == x]
            if pv and bv:
                print(f'  {m:<12} {prop}={statistics.mean(pv):.4f}  {base}={statistics.mean(bv):.4f}  '
                      f'{"HIGHER" if statistics.mean(pv) > statistics.mean(bv) else "NOT higher"}')

        # cost of ONE deletion: how long the unlearning took and how many
        # shards it had to retrain. A partition that wins on recall but
        # retrains more shards is not a free win, so this is reported next
        # to the utility comparison rather than in a separate table.
        for m in ['unlearn_time', 'affected_shards']:
            pv = [x for x in [get(prop, s, m) for s in paired] if x == x]
            bv = [x for x in [get(base, s, m) for s in paired] if x == x]
            if pv and bv:
                mp, mb = statistics.mean(pv), statistics.mean(bv)
                print(f'  {m:<12} {prop}={mp:.2f}  {base}={mb:.2f}  '
                      f'delta={mp - mb:+.2f}  '
                      f'{"CHEAPER" if mp < mb else ("costlier" if mp > mb else "equal")}')


if __name__ == '__main__':
    main()
