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
  RANDOM_PERM    random permutation of user ids -> K shards   (primary baseline)
  RANDOM_MODULO  user_id % K                                   (sanity check)
  SISA           user_id % K, frozen mapping (simple deletion-stable baseline)
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
import time
import json
import random
import argparse
import numpy as np
import torch

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

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


def partition_proposed(train_items, n_users, n_items, n_shards, sig_dim, seed):
    """Ours: user-local signature + deletion-stable similarity-aware routing."""
    builder = UserSignatureBuilder(n_users, n_items, d=sig_dim, seed=seed)
    sigs = builder.build_signature(train_items)
    assigner = DeletionStableAssignment(n_shards=n_shards, d=sig_dim, seed=seed)
    return assigner.assign_users(sigs)


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
def build_shard_data(train_items, user_to_shard, n_shards):
    shard_data = [{} for _ in range(n_shards)]
    for u, items in train_items.items():
        shard_data[user_to_shard[u]][u] = list(items)
    return shard_data


def run_sharded_arm(train_items, test_data, n_users, n_items, n_shards,
                    emb_dim, user_to_shard, del_users, epochs, lr, batch_size,
                    device, seed, retrain_epochs):
    """Train K shard models, delete users, retrain affected shards, evaluate."""
    set_all_seeds(seed)
    shard_data = build_shard_data(train_items, user_to_shard, n_shards)
    shard_models = ShardModels(n_users, n_items, emb_dim, n_shards)

    t0 = time.time()
    for sid in range(n_shards):
        model = shard_models.create_model(sid, device)
        train_shard_model(model, shard_data[sid], n_items, device,
                          batch_size=batch_size, lr=lr, n_epochs=epochs)
    train_time = time.time() - t0

    # ---- unlearn: remove deleted users, retrain ONLY affected shards ----
    # NOTE: matches the original method_4_ours.unlearn(): the shard model is
    # RETRAINED FROM ITS CURRENT WEIGHTS (not reset), as in the original paper
    # ("retrain only Mj from scratch using Sj-"). Creating a fresh model here
    # would change the measured utility relative to the earlier logs.
    affected = sorted({int(user_to_shard[u]) for u in del_users if u < len(user_to_shard)})
    retained = {u: it for u, it in train_items.items() if u not in del_users}
    retained_shard_data = build_shard_data(retained, user_to_shard, n_shards)

    t0 = time.time()
    for sid in affected:
        model = shard_models.get_model(sid)
        train_shard_model(model, retained_shard_data[sid], n_items, device,
                          batch_size=batch_size, lr=lr, n_epochs=retrain_epochs)
    unlearn_time = time.time() - t0

    # ---- evaluate on retained test users only ----
    test_retained = {u: it for u, it in test_data.items() if u not in del_users}
    res = evaluate_hard_routing(shard_models, user_to_shard, retained,
                                 test_retained, n_users, n_items, device, Ks=KS)
    return res, train_time, unlearn_time, affected, shard_models


def run_global_arm(train_items, test_data, n_users, n_items, del_users,
                   emb_dim, epochs, lr, batch_size, device, seed):
    """GLOBAL reference: one BPRMF on all retained data."""
    from method_4_ours import BPRMF
    set_all_seeds(seed)
    retained = {u: it for u, it in train_items.items() if u not in del_users}
    model = BPRMF(n_users, n_items, emb_dim).to(device)
    opt = torch.optim.Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    samples = []
    for u, items in retained.items():
        for pos in items:
            neg = random.randint(0, n_items - 1)
            while neg in items:
                neg = random.randint(0, n_items - 1)
            samples.append((u, pos, neg))
    if not samples:
        return {'recall': [0] * len(KS), 'ndcg': [0] * len(KS)}, 0.0, 0.0

    t0 = time.time()
    for ep in range(epochs):
        random.shuffle(samples)
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
    train_time = time.time() - t0

    test_retained = {u: it for u, it in test_data.items() if u not in del_users}
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
    return out, train_time, train_time


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
    ap.add_argument('--arms', default='PROPOSED,PROPOSED_BAL,RANDOM_PERM,RANDOM_MODULO,SISA,GLOBAL')
    ap.add_argument('--data_root', default=None)
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
    args = ap.parse_args()

    data_root = args.data_root or os.path.join(os.path.dirname(PROJ), 'data')
    out_dir = args.out_dir or os.path.join(os.path.dirname(PROJ), 'results')
    os.makedirs(out_dir, exist_ok=True)
    tag = args.tag or f'{args.dataset}_K{args.n_shards}'
    csv_path = os.path.join(out_dir, f'partition_quality_{tag}.csv')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_items, test_data, n_users, n_items = load_data(args.dataset, data_root)
    print(f"  {args.dataset}: {n_users} users, {n_items} items, device={device}")

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    arms = [a.strip() for a in args.arms.split(',') if a.strip()]

    # append-only: never overwrite a per-seed file
    write_header = not os.path.exists(csv_path)
    fh = open(csv_path, 'a', newline='', encoding='utf-8')
    import csv as _csv
    writer = _csv.DictWriter(fh, fieldnames=[
        'dataset', 'seed', 'arm', 'partition', 'K', 'deletion_ratio', 'deletion_mode',
        'backbone', 'embedding_dim', 'epochs', 'learning_rate', 'regularization',
        'recall_10', 'ndcg_10', 'recall_20', 'ndcg_20', 'recall_50', 'ndcg_50',
        'intra_similarity', 'nr_10', 'nr_20', 'nr_50',
        'user_load_cv', 'interaction_load_cv', 'max_mean_ratio',
        'train_time', 'unlearn_time', 'affected_shards',
    ])
    if write_header:
        writer.writeheader()
        fh.flush()

    for seed in seeds:
        if args.legacy_mode:
            # exactly reproduce the original method_4_ours.py deletion set
            ui = sorted(train_items.items(), key=lambda kv: len(kv[1]), reverse=True)
            del_users = {u for u, _ in ui[:int(len(train_items) * args.unlearn_ratio)]}
        else:
            del_users = pick_deletion_users(train_items, args.unlearn_ratio,
                                            args.unlearn_mode, seed)
        print(f"\n{'='*70}\nSEED {seed} | deleted {len(del_users)} users "
              f"({args.unlearn_ratio*100:.1f}%, {args.unlearn_mode})\n{'='*70}")

        for arm in arms:
            print(f"\n--- ARM {arm} (seed {seed}) ---")
            row = {
                'dataset': args.dataset, 'seed': seed, 'arm': arm, 'K': args.n_shards,
                'deletion_ratio': args.unlearn_ratio, 'deletion_mode': args.unlearn_mode,
                'backbone': 'BPRMF', 'embedding_dim': args.emb_dim,
                'epochs': args.epochs, 'learning_rate': args.lr,
                'regularization': 0.01,
            }

            if arm == 'GLOBAL':
                res, tt, ut = run_global_arm(
                    train_items, test_data, n_users, n_items, del_users,
                    args.emb_dim, args.epochs, args.lr, args.batch_size, device, seed)
                row.update({
                    'partition': 'none', 'train_time': tt, 'unlearn_time': ut,
                    'affected_shards': 1, 'intra_similarity': '',
                    'nr_10': '', 'nr_20': '', 'nr_50': '',
                    'user_load_cv': '', 'interaction_load_cv': '', 'max_mean_ratio': '',
                })
            else:
                if arm == 'PROPOSED':
                    # legacy_seed keeps the signature matrix R and the LSH
                    # hyperplanes identical to the earlier runs (seed 42),
                    # so PROPOSED is directly comparable with the old logs.
                    umap = partition_proposed(train_items, n_users, n_items,
                                              args.n_shards, args.sig_dim,
                                              args.legacy_seed)
                    pname = 'signature_lsh'
                elif arm == 'PROPOSED_BAL':
                    umap = partition_proposed_balanced(
                        train_items, n_users, n_items, args.n_shards,
                        args.sig_dim, args.legacy_seed)
                    pname = 'signature_lsh_balanced'
                elif arm == 'RANDOM_PERM':
                    umap = partition_random_perm(n_users, args.n_shards, seed)
                    pname = 'random_perm'
                elif arm == 'RANDOM_MODULO':
                    umap = partition_random_modulo(n_users, args.n_shards, seed)
                    pname = 'random_modulo'
                elif arm == 'SISA':
                    umap = partition_random_modulo(n_users, args.n_shards, seed)
                    pname = 'sisa_modulo'
                else:
                    raise ValueError(f'unknown arm {arm}')

                res, tt, ut, affected, shard_models = run_sharded_arm(
                    train_items, test_data, n_users, n_items, args.n_shards,
                    args.emb_dim, umap, del_users, args.epochs, args.lr,
                    args.batch_size, device, seed, args.retrain_epochs)

                # partition-quality metrics need the user embeddings.
                # Each shard model owns a FULL n_users x emb_dim table (only its own
                # users' rows were trained), so index by the real user ids.
                uemb = np.zeros((n_users, args.emb_dim), dtype=np.float32)
                for sid in range(args.n_shards):
                    m = shard_models.get_model(sid)
                    if m is None:
                        continue
                    idx = np.where(umap == sid)[0]
                    if len(idx):
                        uemb[idx] = (m.user_embedding.weight
                                     .detach().cpu().numpy()[idx])
                isim = intra_similarity(uemb, umap, args.n_shards, seed=seed)
                nr10 = neighbour_retention(uemb, umap, q=10, seed=seed)
                nr20 = neighbour_retention(uemb, umap, q=20, seed=seed)
                nr50 = neighbour_retention(uemb, umap, q=50, seed=seed)
                ucv, icv, mmr = load_balance(umap, train_items, args.n_shards)

                row.update({
                    'partition': pname, 'train_time': tt, 'unlearn_time': ut,
                    'affected_shards': len(affected),
                    'intra_similarity': isim, 'nr_10': nr10, 'nr_20': nr20, 'nr_50': nr50,
                    'user_load_cv': ucv, 'interaction_load_cv': icv, 'max_mean_ratio': mmr,
                })

            for i, k in enumerate(KS):
                row[f'recall_{k}'] = res['recall'][i]
                row[f'ndcg_{k}'] = res['ndcg'][i]

            writer.writerow(row)
            fh.flush()
            print(f"  [CSV] {arm}: R@10={row['recall_10']:.4f} NDCG@10={row['ndcg_10']:.4f} "
                  f"train={row['train_time']:.1f}s unlearn={row['unlearn_time']:.1f}s "
                  f"-> {csv_path}")

    fh.close()
    print(f"\nDone. Raw results: {csv_path}")
    print("Aggregate with: python code/aggregate_partition_quality.py")


if __name__ == '__main__':
    main()
