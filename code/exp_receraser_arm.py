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
import fast_train

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
                            data_root=None, kind='ubp', n_iter=50):
    """The partition of the RECERASER arm.

    kind='ubp'   : RecEraser's User-based Balanced Partition, as in the
                   authors' utility/data_partition.py (data_partition_2):
                   balanced k-means over the pretrained user embeddings.
                   K random users are the first centroids. Each round scores
                   every (user, centroid) pair by squared Euclidean distance,
                   walks the pairs from the closest and puts a user in that
                   shard unless it already holds 1.2 * n / K users, then moves
                   each centroid to the mean of its users. n_iter rounds
                   (the authors' part_T = 50), stopping early once no user
                   changes shard. Two departures from that file, both
                   deliberate: the centroids come from a seeded generator, so
                   a seed gives the same partition, and the centroid is the
                   mean VECTOR of its users - the file takes np.mean without an
                   axis, which collapses each centroid to one number.
    kind='random': a random permutation of the users split into K equal
                   parts, the control that isolates what the partition
                   contributes to RecEraser.
    kind='inp'   : NOT RecEraser's InP. That partition assigns INTERACTIONS,
                   so one user's interactions spread over several shards, and
                   this pipeline keys every shard on users. What runs here is
                   a user-level stand-in (k-means over the user vector joined
                   with the mean of its items' vectors). Kept for reference;
                   do not report it as InP.

    Returns user_to_shard (int32 array of length n_users).
    """
    rng = np.random.RandomState(seed)
    if kind == 'random':
        umap = np.empty(n_users, dtype=np.int32)
        umap[rng.permutation(n_users)] = np.arange(n_users) % n_shards
        return umap

    if data_root is None:
        data_root = os.path.join(os.path.dirname(PROJ), 'data')
    emb_path = os.path.join(data_root, 'wmf_embeddings.npz')
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f'WMF embeddings not found at {emb_path}. Train them first with '
            f'code/train_wmf_pretrain.py --dataset <ds>')
    data = np.load(emb_path)
    user_emb = np.asarray(data['user_embeddings'][:n_users], dtype=np.float64)

    if kind == 'inp':
        return _user_level_inp_standin(train_data, n_users, n_shards, seed,
                                       user_emb, data['item_embeddings'])
    if kind != 'ubp':
        raise ValueError(f'unknown RecEraser partition {kind}')

    # the users the authors partition: those with training data
    users = np.array(sorted(u for u in train_data if u < n_users), dtype=np.int64)
    X = user_emb[users]
    n, k = len(users), n_shards
    cap = 1.2 * n / k
    cent = X[rng.choice(n, k, replace=False)].copy()

    def assign_round(cent):
        dist = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(-1)      # (n, k)
        order = np.argsort(dist, axis=None, kind='stable')
        new = np.full(n, -1, dtype=np.int64)
        counts = np.zeros(k, dtype=np.int64)
        left = n
        for idx in order:
            u, j = divmod(int(idx), k)
            if new[u] < 0 and counts[j] < cap:
                new[u] = j
                counts[j] += 1
                left -= 1
                if left == 0:
                    break
        return new, dist

    assign = np.full(n, -1, dtype=np.int64)
    reseeded = 0
    for it in range(n_iter):
        new, dist = assign_round(cent)
        changed = int((new != assign).sum())
        assign = new
        empty = []
        for j in range(k):
            members = X[assign == j]
            if len(members):
                cent[j] = members.mean(axis=0)
            else:
                empty.append(j)
        # A shard that ends a round with no user would train on nothing. As in
        # standard k-means, its centroid restarts at the user farthest from
        # its own centroid, so the next round fills it. (The authors' file
        # takes the mean of an empty list there, which is NaN.)
        if empty:
            own = dist[np.arange(n), assign]
            for j, u in zip(empty, np.argsort(-own)[:len(empty)]):
                cent[j] = X[u]
            reseeded += len(empty)
            changed = max(changed, 1)
        if changed == 0:
            break
    if (np.bincount(assign, minlength=k) == 0).any():
        assign, _ = assign_round(cent)

    umap = np.empty(n_users, dtype=np.int32)
    # users with no training data never enter a shard's training; route them
    # to the nearest centroid so every id has a shard
    umap[:] = ((user_emb[:, None, :] - cent[None, :, :]) ** 2).sum(-1).argmin(1)
    umap[users] = assign
    counts = np.bincount(assign, minlength=k)
    loads = np.zeros(k)
    for u, j in zip(users, assign):
        loads[j] += len(train_data[int(u)])
    print(f"    [RecEraser] UBP partition: {it + 1} round(s), cap={cap:.0f} users, "
          f"{reseeded} empty-shard restart(s), "
          f"users min={counts.min()} max={counts.max()} "
          f"inter_cv={loads.std() / (loads.mean() + 1e-10):.3f}")
    return umap


def _user_level_inp_standin(train_data, n_users, n_shards, seed, user_emb, item_emb):
    """See kind='inp' above: a user-level stand-in, not RecEraser's InP."""
    from sklearn.cluster import KMeans
    u = user_emb / np.maximum(np.linalg.norm(user_emb, axis=1, keepdims=True), 1e-10)
    prof = np.zeros((n_users, item_emb.shape[1]), dtype=np.float64)
    for uid, its in train_data.items():
        if uid < n_users:
            idx = [i for i in its if i < item_emb.shape[0]]
            if idx:
                prof[uid] = item_emb[idx].mean(axis=0)
    prof = prof / np.maximum(np.linalg.norm(prof, axis=1, keepdims=True), 1e-10)
    km = KMeans(n_clusters=n_shards, n_init=10, random_state=seed).fit(np.hstack([u, prof]))
    print("    [RecEraser] user-level InP STAND-IN (not RecEraser's interaction InP)")
    return km.labels_.astype(np.int32)


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
        if fast_train.use_gpu_receraser_training():
            samples = fast_train.draw_samples(shard_data, n_items)
            if not samples[0]:
                return 0.0, 0.0, 0
            t0 = time.time()

            def step(u, p, n):
                optimizer.zero_grad()
                loss = m(u, p, n, shard_id)
                loss.backward()
                optimizer.step()
                return loss
            total, _ = fast_train.train_epochs(step, samples, self.device,
                                               batch_size, n_epochs)
            fast_train.sync(self.device)
            return total, time.time() - t0, len(samples[0])
        samples = []
        for user, items in shard_data.items():
            for pos_item in items:
                neg_item = random.randint(0, n_items - 1)
                while neg_item in items:
                    neg_item = random.randint(0, n_items - 1)
                samples.append((user, pos_item, neg_item))
        if not samples:
            return 0.0, 0.0, 0

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
        if fast_train.use_gpu_receraser_training():
            # method_3_receraser_v2.train_aggregator on the GPU: only the
            # attention and transform parameters move (the embeddings are
            # detached in the forward), a non-finite batch is skipped, the
            # gradient is clipped at 1.0
            m = self.model
            agg_params = [m.WA, m.BA, m.HA, m.WB, m.BB, m.HB, m.trans_W, m.trans_B]
            opt = torch.optim.Adagrad(agg_params, lr=lr, initial_accumulator_value=1.0)
            samples = fast_train.draw_samples(train_data, n_items)
            skipped = [0]

            def step(u, p, n):
                opt.zero_grad()
                loss = m(u, p, n, shard=0, use_aggregation=True)
                if not torch.isfinite(loss):
                    skipped[0] += 1
                    return None
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                return loss
            fast_train.train_epochs(
                step, samples, self.device, batch_size, n_epochs, log_every=10,
                log_fmt='    Agg Epoch {ep}: loss={loss:.4f}')
            fast_train.sync(self.device)
            if skipped[0]:
                print(f"    [RecEraser] aggregator skipped {skipped[0]} non-finite batches")
            return time.time() - t0
        train_aggregator(self.model, train_data, n_users, n_items, self.device,
                         batch_size=batch_size, lr=lr, max_epochs=n_epochs)
        return time.time() - t0

    @torch.no_grad()
    def evaluate(self, user_to_shard, train_data, test_data, n_users, n_items,
                 Ks=KS):
        """Evaluate with the attention aggregation (RecEraser's inference)."""
        import heapq
        m = self.model
        if fast_train.use_gpu_engine():
            m.eval()
            K, D = m.num_local, m.emb_dim

            def agg(emb):        # what predict(..., use_aggregation=True) does
                es = emb.view(-1, K, D)
                es = torch.einsum('bkd,kde->bke', es, m.trans_W) + m.trans_B
                return m._attention_aggregate(es)[0]
            with torch.no_grad():
                items = torch.cat([agg(m.item_embedding.weight[i:i + 8192])
                                   for i in range(0, n_items, 8192)])
            res = fast_train.evaluate_topk(
                lambda ut: agg(m.user_embedding.weight[ut]) @ items.T,
                list(range(n_users)), train_data, test_data, n_items, self.device, Ks)
            return {'recall': res['recall'], 'ndcg': res['ndcg']}
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
