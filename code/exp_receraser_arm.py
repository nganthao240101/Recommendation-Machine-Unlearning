"""
The RECERASER arm for the partition experiment.

RecEraser is structurally different from every other arm in
exp_partition_quality.py:

  * one model holds all K shards' embeddings side by side
    (user_embedding is (n_users, K*dim), sliced per shard),
  * inference uses an ATTENTION AGGREGATION over all K shard embeddings
    instead of hard routing to a single shard,
  * the partition is a k-means over pretrained WMF user embeddings
    (UBP: user-based, balanced by interaction count).

Because of the aggregation, deleting a user forces the aggregator to be
retrained as well: the attention weights depend on every user's embedding,
so removing a user changes the aggregated representation of every other user.
That is exactly the cost the paper is measuring, so it is included in
unlearn_time.

The partition is recomputed on the retained data when the instability rate is
measured, so the partition here is a function of the data it is given.
"""

import os
import sys
import time
import random

import numpy as np
import torch

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

KS = [10, 20, 50]


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# partition: UBP over pretrained WMF user embeddings
# ----------------------------------------------------------------------------
def receraser_ubp_partition(train_data, n_users, n_items, n_shards, seed,
                            data_root=None, kind='ubp'):
    """UBP partition exactly as in RecEraser: k-means over user embeddings,
    balanced by interaction count.

    kind='ubp'  : k-means over the user embedding only (user-based partition)
    kind='inp'  : k-means over the user embedding concatenated with the
                  item-embedding centroid of the user's items
    kind='random': a random permutation of the users split into K equal
                  parts, the control that isolates what the partition
                  contributes to RecEraser

    Returns user_to_shard (int32 array of length n_users).
    """
    if kind == 'random':
        rng = np.random.RandomState(seed)
        umap = np.empty(n_users, dtype=np.int32)
        umap[rng.permutation(n_users)] = np.arange(n_users) % n_shards
        return umap

    from sklearn.cluster import KMeans

    if data_root is None:
        data_root = os.path.join(os.path.dirname(PROJ), 'data')
    emb_path = os.path.join(data_root, 'wmf_embeddings.npz')
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f'WMF embeddings not found at {emb_path}. Train them first with '
            f'code/train_wmf_pretrain.py --dataset <ds>')

    data = np.load(emb_path)
    user_emb = data['user_embeddings'][:n_users]
    item_emb = data['item_embeddings']
    user_emb = user_emb / np.maximum(np.linalg.norm(user_emb, axis=1, keepdims=True), 1e-10)

    if kind == 'inp':
        prof = np.zeros((n_users, item_emb.shape[1]), dtype=np.float32)
        for uid, its in train_data.items():
            if uid >= n_users:
                continue
            idx = [i for i in its if i < item_emb.shape[0]]
            if idx:
                prof[uid] = item_emb[idx].mean(axis=0)
        prof = prof / np.maximum(np.linalg.norm(prof, axis=1, keepdims=True), 1e-10)
        feats = np.hstack([user_emb, prof])
    else:
        feats = user_emb

    # RecEraser balances the shards by INTERACTION count, not by user count.
    km = KMeans(n_clusters=n_shards, n_init=10, random_state=seed).fit(feats)
    centers = km.cluster_centers_
    labels = km.labels_

    loads = np.zeros(n_shards, dtype=np.int64)
    for u, its in train_data.items():
        if u < n_users:
            loads[labels[u]] += len(its)

    # Greedy rebalancing: move the users of the heaviest shard to the
    # lightest shard until the loads are close enough. This is the balancing
    # step of RecEraser's UBP.
    umap = labels.astype(np.int32).copy()
    order = np.argsort(-loads)
    for src in order:
        while loads[src] > 0:
            dst = int(np.argmin(loads))
            if dst == src or loads[dst] == 0:
                break
            # move the users of the heaviest shard with the smallest load
            users_in_src = np.where(umap == src)[0]
            if len(users_in_src) == 0:
                break
            # pick the user whose removal balances best
            best_u, best_delta = None, None
            for u in users_in_src:
                w = len(train_data.get(int(u), []))
                delta = abs(loads[src] - w) - abs(loads[dst] + w)
                if best_delta is None or delta < best_delta:
                    best_u, best_delta = int(u), delta
            umap[best_u] = dst
            loads[src] -= len(train_data.get(best_u, []))
            loads[dst] += len(train_data.get(best_u, []))

    counts = np.bincount(umap, minlength=n_shards)
    print(f"    [RecEraser] {kind.upper()} partition: users min={counts.min()} "
          f"max={counts.max()} mean={counts.mean():.1f} "
          f"inter_cv={loads.std()/(loads.mean()+1e-10):.3f}")
    return umap


# ----------------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------------
class RecEraserArm:
    """One model holding all K shards' embeddings, plus the attention
    aggregator that RecEraser uses at inference time."""

    def __init__(self, n_users, n_items, emb_dim, n_shards, device, seed):
        from method_3_receraser_v2 import RecEraserModel
        set_all_seeds(seed)
        self.model = RecEraserModel(n_users, n_items, emb_dim,
                                    num_local=n_shards, use_attention=True).to(device)
        self.n_shards = n_shards
        self.device = device

    def load_pretrained(self, user_emb, item_emb):
        """Seed every shard's slice with the pretrained vector.

        The table holds the K slices side by side, (n, K*d), so the vector is
        tiled K times. np.repeat would repeat each column instead
        ([a,a,b,b,...]), which hands every shard a scrambled vector.
        """
        d = self.model.emb_dim
        if user_emb.shape[1] != d:
            raise ValueError(
                f'--receraser_init wmf needs emb_dim == the WMF dim '
                f'({user_emb.shape[1]}), got emb_dim={d}. Use the default '
                f'--receraser_init random for the other dims.')
        u = np.concatenate([user_emb] * self.n_shards, axis=1)
        i = np.concatenate([item_emb] * self.n_shards, axis=1)
        m = self.model
        m.user_embedding.weight.data = torch.as_tensor(u, dtype=torch.float32,
                                                       device=self.device)
        m.item_embedding.weight.data = torch.as_tensor(i, dtype=torch.float32,
                                                       device=self.device)

    def shard_parameters(self, shard_id):
        """The slice of the embedding tables that belongs to one shard."""
        m = self.model
        d = m.emb_dim
        u = m.user_embedding.weight.data
        i = m.item_embedding.weight.data
        us = u[:, shard_id * d:(shard_id + 1) * d].clone()
        is_ = i[:, shard_id * d:(shard_id + 1) * d].clone()
        return us, is_

    def set_shard_parameters(self, shard_id, us, is_):
        m = self.model
        d = m.emb_dim
        m.user_embedding.weight.data[:, shard_id * d:(shard_id + 1) * d] = us
        m.item_embedding.weight.data[:, shard_id * d:(shard_id + 1) * d] = is_

    def train_shard(self, shard_id, shard_data, n_items, batch_size=512,
                    lr=0.05, n_epochs=50):
        """Train one shard's slice of the embedding table."""
        m = self.model
        optimizer = torch.optim.Adagrad(m.parameters(), lr=lr,
                                        initial_accumulator_value=1e-8)
        samples = []
        for user, items in shard_data.items():
            for pos_item in items:
                neg_item = random.randint(0, n_items - 1)
                while neg_item in items:
                    neg_item = random.randint(0, n_items - 1)
                samples.append((user, pos_item, neg_item))
        if not samples:
            return 0.0, time.time(), 0

        t0 = time.time()
        total_loss = 0.0
        for epoch in range(n_epochs):
            random.shuffle(samples)
            nb = max(1, len(samples) // batch_size)
            for b in range(nb):
                chunk = samples[b * batch_size:(b + 1) * batch_size]
                if not chunk:
                    continue
                users = torch.LongTensor([c[0] for c in chunk]).to(self.device)
                pos = torch.LongTensor([c[1] for c in chunk]).to(self.device)
                neg = torch.LongTensor([c[2] for c in chunk]).to(self.device)
                optimizer.zero_grad()
                loss = m(users, pos, neg, shard_id)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        return total_loss, time.time() - t0, len(samples)

    def train_aggregator(self, train_data, n_users, n_items, batch_size=512,
                         lr=0.005, n_epochs=50):
        """Retrain the attention aggregator.

        The aggregator's attention weights are a function of every user's
        embedding, so deleting a user changes the representation of every
        other user. RecEraser therefore has to retrain it, and that cost
        belongs to the unlearning time.
        """
        from method_3_receraser_v2 import train_aggregator
        t0 = time.time()
        train_aggregator(self.model, train_data, n_users, n_items, self.device,
                         batch_size=batch_size, lr=lr, max_epochs=n_epochs)
        return time.time() - t0

    @torch.no_grad()
    def evaluate(self, user_to_shard, train_data, test_data, n_users, n_items,
                 Ks=KS):
        """Evaluate with the attention aggregation (RecEraser's inference)."""
        import heapq
        m = self.model
        rec = {k: [] for k in Ks}
        ndcg = {k: [] for k in Ks}
        m.eval()
        for user in range(n_users):
            if user not in test_data or not test_data[user]:
                continue
            user_t = torch.LongTensor([user]).to(self.device)
            scores = []
            for i in range(0, n_items, 256):
                items = torch.arange(i, min(i + 256, n_items), device=self.device)
                s = m.predict(user_t, items, shard=0, use_aggregation=True).cpu().numpy()
                scores.extend(s.tolist())
            scores = np.array(scores)
            for it in set(train_data.get(user, [])):
                scores[it] = -np.inf
            rank_list = heapq.nlargest(max(Ks), range(len(scores)), key=scores.__getitem__)
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


def run_receraser_arm(train_items, test_data, n_users, n_items, n_shards,
                      emb_dim, user_to_shard, del_users, epochs, lr, batch_size,
                      device, seed, retrain_epochs, data_root=None,
                      agg_epochs=50, init='random'):
    """Full RecEraser arm: train K shard slices + aggregator, delete users,
    retrain the affected shards + the aggregator, evaluate.

    init='random' (default) starts every shard from a random model, as the
    original RecEraser does and as every other arm here does; the WMF
    embeddings only feed the partition. init='wmf' seeds the shards with the
    WMF vectors, a warm start the original configuration does not use.

    Returns the same tuple shape as run_sharded_arm.
    """
    from method_4_ours import BPRMF
    set_all_seeds(seed)

    # the WMF embeddings give the partition its input; with init='wmf' they
    # also seed the model
    emb_path = os.path.join(data_root or os.path.join(os.path.dirname(PROJ), 'data'),
                            'wmf_embeddings.npz')
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f'WMF embeddings not found at {emb_path}. Train them first with '
            f'code/train_wmf_pretrain.py --dataset <ds>')
    emb = np.load(emb_path)
    user_emb = emb['user_embeddings'][:n_users]
    item_emb = emb['item_embeddings']

    arm = RecEraserArm(n_users, n_items, emb_dim, n_shards, device, seed)
    if init == 'wmf':
        arm.load_pretrained(user_emb, item_emb)
    print(f'    [RecEraser] init={init} (emb_dim={emb_dim}, K={n_shards})')

    shard_data = [{} for _ in range(n_shards)]
    for u, items in train_items.items():
        if u < n_users:
            shard_data[user_to_shard[u]][u] = list(items)

    # ---- train all shards + the aggregator ----
    t0 = time.time()
    for sid in range(n_shards):
        arm.train_shard(sid, shard_data[sid], n_items, batch_size=batch_size,
                         lr=lr, n_epochs=epochs)
    arm.train_aggregator(train_items, n_users, n_items, batch_size=batch_size,
                         n_epochs=agg_epochs)
    train_time = time.time() - t0

    # ---- unlearn ----
    affected = sorted({int(user_to_shard[u]) for u in del_users if u < len(user_to_shard)})
    retained = {u: it for u, it in train_items.items() if u not in del_users}
    retained_shard_data = [{} for _ in range(n_shards)]
    for u, items in retained.items():
        if u < n_users:
            retained_shard_data[user_to_shard[u]][u] = list(items)

    per_shard_time = {}
    per_shard_inter = {}
    t0 = time.time()
    for sid in affected:
        # the shard is retrained FROM SCRATCH: its slice of the embedding
        # table is replaced by a fresh one, so the deleted user's embedding
        # is genuinely gone
        d = emb_dim
        fresh_u = torch.nn.Embedding(n_users, d).weight.data.to(device)
        fresh_i = torch.nn.Embedding(n_items, d).weight.data.to(device)
        nn_init = torch.nn.init.xavier_uniform_
        nn_init(fresh_u)
        nn_init(fresh_i)
        arm.set_shard_parameters(sid, fresh_u, fresh_i)
        _, elapsed, n_samples = arm.train_shard(
            sid, retained_shard_data[sid], n_items, batch_size=batch_size,
            lr=lr, n_epochs=retrain_epochs)
        per_shard_time[sid] = elapsed
        per_shard_inter[sid] = n_samples
    # the aggregator must be retrained too: its attention depends on every
    # user's embedding, including the deleted ones
    agg_time = arm.train_aggregator(retained, n_users, n_items,
                                    batch_size=batch_size, n_epochs=agg_epochs)
    unlearn_time = time.time() - t0
    parallel_time = max(per_shard_time.values()) if per_shard_time else 0.0

    test_retained = {u: it for u, it in test_data.items() if u not in del_users}
    res = arm.evaluate(user_to_shard, retained, test_retained, n_users, n_items)

    print(f"    [RecEraser] unlearn: {len(affected)} shard(s) retrained "
          f"({sum(per_shard_time.values()):.1f}s) + aggregator {agg_time:.1f}s "
          f"= {unlearn_time:.1f}s")
    return (res, train_time, unlearn_time, affected, None,
            parallel_time, per_shard_time, per_shard_inter)
