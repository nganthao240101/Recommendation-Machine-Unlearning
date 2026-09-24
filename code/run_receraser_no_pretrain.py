"""
RECERASER KHÔNG DÙNG WMF PRETRAINED
Để so sánh công bằng với các methods khác

Sự khác biệt với run_receraser_like_original.py:
- Không load WMF pretrained embeddings
- Khởi tạo embeddings ngẫu nhiên (Xavier uniform)
- Vẫn dùng attention aggregation
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adagrad
import heapq
import math

PROJ = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(PROJ), 'data')


# ============================================================================
# DATA LOADER
# ============================================================================

class SimpleDataLoader:
    def __init__(self, data_dir, batch_size=512):
        train_file = os.path.join(data_dir, 'train.txt')
        test_file = os.path.join(data_dir, 'test.txt')

        self.n_users, self.n_items = 0, 0
        self.train_items = {}
        self.test_set = {}

        with open(train_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip('\n').split(' ')
                uid = int(parts[0])
                items = [int(i) for i in parts[1:]]
                self.train_items[uid] = items
                self.n_users = max(self.n_users, uid + 1)
                self.n_items = max(self.n_items, max(items) + 1 if items else 0)

        with open(test_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip('\n').split(' ')
                uid = int(parts[0])
                items = [int(i) for i in parts[1:]]
                self.test_set[uid] = items

        print(f"  Loaded: {self.n_users} users, {self.n_items} items")


def load_data(dataset='ml-1m', batch_size=512):
    data_dir = os.path.join(DATA_DIR, dataset)
    return SimpleDataLoader(data_dir, batch_size)


# ============================================================================
# RECERASER MODEL
# ============================================================================

class RecEraserBPR(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, num_local, agg_type='attention', attention_size=32):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.num_local = num_local
        self.agg_type = agg_type

        # Per-shard embeddings (shared)
        self.user_embedding = nn.Embedding(n_users, num_local * emb_dim)
        self.item_embedding = nn.Embedding(n_items, num_local * emb_dim)

        # Xavier uniform initialization - KHÔNG dùng pretrained
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Attention parameters
        self.WA = nn.Parameter(torch.empty(emb_dim, attention_size))
        self.BA = nn.Parameter(torch.zeros(attention_size))
        self.HA = nn.Parameter(torch.ones(attention_size, 1) * 0.1)
        self.WB = nn.Parameter(torch.empty(emb_dim, attention_size))
        self.BB = nn.Parameter(torch.zeros(attention_size))
        self.HB = nn.Parameter(torch.ones(attention_size, 1) * 0.1)

        std_w = math.sqrt(2.0 / (emb_dim + attention_size))
        nn.init.trunc_normal_(self.WA, mean=0.0, std=std_w, a=-2*std_w, b=2*std_w)
        nn.init.trunc_normal_(self.WB, mean=0.0, std=std_w, a=-2*std_w, b=2*std_w)

        self.trans_W = nn.Parameter(torch.empty(num_local, emb_dim, emb_dim))
        self.trans_B = nn.Parameter(torch.zeros(num_local, emb_dim))
        for k in range(num_local):
            self.trans_W.data[k] = torch.eye(emb_dim)

    def _user_emb_for_shard(self, users, shard):
        emb = self.user_embedding(users).view(-1, self.num_local, self.emb_dim)
        return emb[:, shard, :]

    def _item_emb_for_shard(self, items, shard):
        emb = self.item_embedding(items).view(-1, self.num_local, self.emb_dim)
        return emb[:, shard, :]

    def _per_shard_user_emb(self, users):
        return self.user_embedding(users).view(-1, self.num_local, self.emb_dim)

    def _per_shard_item_emb(self, items):
        return self.item_embedding(items).view(-1, self.num_local, self.emb_dim)

    def local_loss(self, users, pos, neg, shard, decay=0.01):
        u_emb = self._user_emb_for_shard(users, shard)
        pos_emb = self._item_emb_for_shard(pos, shard)
        neg_emb = self._item_emb_for_shard(neg, shard)

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        reg = (u_emb.pow(2).sum() + pos_emb.pow(2).sum() + neg_emb.pow(2).sum()) / users.size(0)
        diff = torch.clamp(pos_scores - neg_scores, -50.0, 50.0)
        mf = torch.mean(torch.nn.functional.softplus(-diff))
        reg_loss = decay * reg

        return mf, reg_loss, mf + reg_loss

    def _attention_aggregate(self, embs, which='user'):
        if which == 'user':
            W, B, H = self.WA, self.BA, self.HA
        else:
            W, B, H = self.WB, self.BB, self.HB

        hidden = torch.einsum('bkd,dc->bkc', embs, W) + B
        hidden = torch.relu(hidden)
        score = torch.einsum('bkc,ca->bka', hidden, H)
        attn = torch.softmax(score, dim=1)
        agg = (attn * embs).sum(dim=1)
        return agg, attn

    def batch_ratings_full(self, users, items):
        if self.agg_type == 'attention':
            with torch.no_grad():
                u_es = self._per_shard_user_emb(users)
                i_es = self._per_shard_item_emb(items)
                u_e = torch.einsum('bkd,kde->bke', u_es, self.trans_W) + self.trans_B
                i_e = torch.einsum('bkd,kde->bke', i_es, self.trans_W) + self.trans_B
                u_agg, _ = self._attention_aggregate(u_e, 'user')
                i_agg, _ = self._attention_aggregate(i_e, 'item')
                return u_agg @ i_agg.t()
        else:
            u_es = self._per_shard_user_emb(users)
            i_es = self._per_shard_item_emb(items)
            rs = []
            for k in range(self.num_local):
                u_k = u_es[:, k, :]
                i_k = i_es[:, k, :]
                rs.append(u_k @ i_k.t())
            return torch.stack(rs, dim=0).mean(dim=0)


# ============================================================================
# PARTITIONER
# ============================================================================

class UserPartitioner:
    def __init__(self, n_shards, seed=42):
        self.n_shards = n_shards
        self.seed = seed
        self.user_to_shard = None
        self.shard_data = None

    def partition_users(self, train_data, n_users):
        np.random.seed(self.seed)
        self.user_to_shard = np.zeros(n_users, dtype=np.int32)
        for user_id in train_data.keys():
            if user_id < n_users:
                self.user_to_shard[user_id] = user_id % self.n_shards
        return self.user_to_shard

    def build_shard_data(self, train_data):
        self.shard_data = [{} for _ in range(self.n_shards)]
        for user_id, items in train_data.items():
            shard_id = self.user_to_shard[user_id]
            self.shard_data[shard_id][user_id] = items.copy()
        return self.shard_data

    def get_affected_shards(self, unlearn_user_ids):
        affected = set()
        for uid in unlearn_user_ids:
            if uid < len(self.user_to_shard):
                affected.add(self.user_to_shard[uid])
        return sorted(list(affected))

    def filter_shard_data(self, shard_id, unlearn_user_ids):
        return {u: items for u, items in self.shard_data[shard_id].items()
                if u not in unlearn_user_ids}


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, train_data, test_data, n_users, n_items, device, Ks=[10, 20, 50]):
    model.eval()
    pre_log = {k: [] for k in Ks}
    rec_log = {k: [] for k in Ks}
    ndcg_log = {k: [] for k in Ks}

    item_ids = np.arange(n_items)
    item_batch_t = torch.from_numpy(item_ids).long().to(device)

    with torch.no_grad():
        for user in range(n_users):
            if user not in test_data or not test_data[user]:
                continue

            users_t = torch.LongTensor([user]).to(device)
            rate_batch = model.batch_ratings_full(users_t, item_batch_t)
            rate_batch = rate_batch.cpu().numpy()[0]

            train_items = train_data.get(user, [])
            rate_batch[train_items] = -np.inf

            item_pos = test_data.get(user, [])
            item_set = set(item_pos)

            rank_list = heapq.nlargest(max(Ks), range(len(rate_batch)), key=rate_batch.__getitem__)

            for k in Ks:
                hit_list = rank_list[:k]
                hit_num = len(set(hit_list) & item_set)
                pre = hit_num / k if k > 0 else 0
                rec = hit_num / len(item_pos) if len(item_pos) > 0 else 0
                dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(hit_list) if item in item_set)
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(item_pos), k)))
                ndcg = dcg / idcg if idcg > 0 else 0
                pre_log[k].append(pre)
                rec_log[k].append(rec)
                ndcg_log[k].append(ndcg)

    model.train()
    return {
        'precision': [np.mean(pre_log[k]) for k in Ks],
        'recall': [np.mean(rec_log[k]) for k in Ks],
        'ndcg': [np.mean(ndcg_log[k]) for k in Ks]
    }


# ============================================================================
# TRAINING
# ============================================================================

def train_local_model(model, shard_data, n_items, device, shard_id, batch_size=512, lr=0.05, max_epochs=100, verbose=True):
    n_samples = sum(len(items) for items in shard_data.values())
    if n_samples == 0:
        return

    optimizer = Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    for epoch in range(max_epochs):
        samples = []
        for user, items in shard_data.items():
            for pos_item in items:
                neg_item = random.randint(0, n_items - 1)
                while neg_item in items:
                    neg_item = random.randint(0, n_items - 1)
                samples.append((user, pos_item, neg_item))

        random.shuffle(samples)
        loss_sum = 0.0
        n_batches = max(1, len(samples) // batch_size)

        for _ in range(n_batches):
            indices = random.sample(range(len(samples)), min(batch_size, len(samples)))
            batch = [samples[i] for i in indices]

            users = torch.LongTensor([s[0] for s in batch]).to(device)
            pos_items = torch.LongTensor([s[1] for s in batch]).to(device)
            neg_items = torch.LongTensor([s[2] for s in batch]).to(device)

            optimizer.zero_grad()
            mf, reg, total = model.local_loss(users, pos_items, neg_items, shard_id)
            total.backward()
            optimizer.step()
            loss_sum += total.item()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    [Local {shard_id}] Epoch {epoch+1}: loss={loss_sum/n_batches:.4f}")


def train_aggregator(model, train_data, n_users, n_items, device, batch_size=512, lr=0.05, max_epochs=100, verbose=True):
    optimizer = Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    samples = []
    for user, items in train_data.items():
        for pos_item in items:
            neg_item = random.randint(0, n_items - 1)
            while neg_item in items:
                neg_item = random.randint(0, n_items - 1)
            samples.append((user, pos_item, neg_item))

    n_samples = len(samples)
    n_batches = max(1, n_samples // batch_size)

    for epoch in range(max_epochs):
        random.shuffle(samples)
        total_loss = 0.0

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            batch = samples[start:end]

            users = torch.LongTensor([s[0] for s in batch]).to(device)
            pos_items = torch.LongTensor([s[1] for s in batch]).to(device)
            neg_items = torch.LongTensor([s[2] for s in batch]).to(device)

            optimizer.zero_grad()
            mf, reg, total = model.local_loss(users, pos_items, neg_items, 0)
            total.backward()
            optimizer.step()
            total_loss += total.item()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    [Aggregator] Epoch {epoch+1}: loss={total_loss/n_batches:.4f}")


# ============================================================================
# MAIN
# ============================================================================

def run_receraser_no_pretrain(dataset='ml-1m', emb_dim=64, n_shards=8,
                               batch_size=512, lr=0.05, attention_size=32,
                               max_epochs_local=50, max_epochs_agg=50,
                               unlearn_ratio=0.1, unlearn_mode='random',
                               unlearn_user_id=None, retrain_epochs=50,
                               agg_type='attention', output_suffix=''):
    print(f"\n{'='*70}")
    print(f"RECERASER (NO WMF PRETRAIN) - CUNG SETTING VOI CAC METHODS KHAC")
    print(f"{'='*70}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load data
    print(f"\nLoading data...")
    data = load_data(dataset=dataset, batch_size=batch_size)
    n_users, n_items = data.n_users, data.n_items
    train_data, test_data = data.train_items, data.test_set

    # Select unlearn users
    random.seed(42)
    all_users = list(train_data.keys())
    n_unlearn = int(len(all_users) * unlearn_ratio)

    if unlearn_mode == 'single' and unlearn_user_id is not None:
        unlearn_users = {unlearn_user_id}
        n_unlearn = 1
    else:
        unlearn_users = set(random.sample(all_users, n_unlearn))

    print(f"\nUnlearn: {len(unlearn_users)} users ({unlearn_ratio*100}%)")

    test_data_retained = {u: items for u, items in test_data.items() if u not in unlearn_users}

    # =========================================================================
    # TRAIN RECERASER WITHOUT WMF PRETRAIN
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP: TRAIN RECERASER (NO WMF PRETRAIN)")
    print(f"{'='*70}")

    # Create partitioner
    partitioner = UserPartitioner(n_shards=n_shards, seed=42)
    partitioner.partition_users(train_data, n_users)
    partitioner.build_shard_data(train_data)

    # Create model - NO PRETRAIN (Xavier init)
    print(f"  Creating model with Xavier initialization (NO WMF)...")
    model = RecEraserBPR(n_users, n_items, emb_dim, num_local=n_shards,
                          agg_type=agg_type, attention_size=attention_size).to(device)

    # Debug embeddings
    u_emb = model.user_embedding.weight.data
    i_emb = model.item_embedding.weight.data
    print(f"    [DEBUG] User emb: mean={u_emb.mean():.4f}, std={u_emb.std():.4f}")
    print(f"    [DEBUG] Item emb: mean={i_emb.mean():.4f}, std={i_emb.std():.4f}")

    # Local training
    print(f"\n  Phase 1: Local Training ({n_shards} shards, {max_epochs_local} epochs each)")
    t0 = time.time()
    for shard_id in range(n_shards):
        shard_data = partitioner.shard_data[shard_id]
        print(f"    Training shard {shard_id}...")
        train_local_model(model, shard_data, n_items, device, shard_id,
                          batch_size, lr, max_epochs_local, verbose=True)
    local_time = time.time() - t0
    print(f"  Local training time: {local_time:.1f}s")

    # Aggregator training
    print(f"\n  Phase 2: Aggregator Training ({max_epochs_agg} epochs)")
    t0 = time.time()
    train_aggregator(model, train_data, n_users, n_items, device,
                     batch_size, lr, max_epochs_agg, verbose=True)
    agg_time = time.time() - t0
    print(f"  Aggregator training time: {agg_time:.1f}s")

    train_time = local_time + agg_time

    # Evaluate BEFORE unlearn
    print(f"\n  Evaluating BEFORE unlearn on FULL test set...")
    results_before = evaluate_model(model, train_data, test_data, n_users, n_items, device)
    print(f"    Before (FULL) - R@10: {results_before['recall'][0]:.4f}, NDCG@10: {results_before['ndcg'][0]:.4f}")

    # =========================================================================
    # UNLEARN
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP: UNLEARN")
    print(f"{'='*70}")

    affected_shards = partitioner.get_affected_shards(unlearn_users)
    print(f"  Affected shards: {affected_shards}")

    t0 = time.time()
    for shard_id in affected_shards:
        filtered_data = partitioner.filter_shard_data(shard_id, unlearn_users)
        print(f"  Retraining shard {shard_id}...")
        train_local_model(model, filtered_data, n_items, device, shard_id,
                          batch_size, lr, retrain_epochs, verbose=True)
        print(f"  Retraining aggregator...")
        train_aggregator(model, filtered_data, n_users, n_items, device,
                         batch_size, lr, retrain_epochs, verbose=True)

    unlearn_time = time.time() - t0
    print(f"  Unlearn time: {unlearn_time:.1f}s")

    # Evaluate AFTER unlearn
    print(f"\n  Evaluating AFTER unlearn on FULL test set...")
    results_after = evaluate_model(model, train_data, test_data, n_users, n_items, device)
    print(f"    After (FULL) - R@10: {results_after['recall'][0]:.4f}, NDCG@10: {results_after['ndcg'][0]:.4f}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results = {
        'method': 'RecEraser_NoPretrain',
        'dataset': dataset,
        'hyperparameters': {
            'emb_dim': emb_dim,
            'n_shards': n_shards,
            'batch_size': batch_size,
            'lr': lr,
            'attention_size': attention_size,
            'max_epochs_local': max_epochs_local,
            'max_epochs_agg': max_epochs_agg,
            'pretrain_wmf': False  # KEY DIFFERENCE
        },
        'unlearn_ratio': unlearn_ratio,
        'n_unlearn': n_unlearn,
        'train_time': train_time,
        'unlearn_time': unlearn_time,
        'before': {
            'recall@10': results_before['recall'][0],
            'recall@20': results_before['recall'][1],
            'recall@50': results_before['recall'][2],
            'ndcg@10': results_before['ndcg'][0],
        },
        'after': {
            'recall@10': results_after['recall'][0],
            'recall@20': results_after['recall'][1],
            'recall@50': results_after['recall'][2],
            'ndcg@10': results_after['ndcg'][0],
        }
    }

    output_path = f'results_receraser_no_pretrain{output_suffix}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Before R@10: {results_before['recall'][0]:.4f}")
    print(f"  After R@10:  {results_after['recall'][0]:.4f}")
    print(f"  Retention:   {(results_after['recall'][0]/results_before['recall'][0]*100):.2f}%")
    print(f"  Train time:  {train_time:.1f}s")
    print(f"  Unlearn time: {unlearn_time:.1f}s")
    print(f"\n  Results saved to: {output_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--n_shards', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--attention_size', type=int, default=32)
    parser.add_argument('--max_epochs_local', type=int, default=50)
    parser.add_argument('--max_epochs_agg', type=int, default=50)
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)
    parser.add_argument('--unlearn_mode', type=str, default='random')
    parser.add_argument('--unlearn_user_id', type=int, default=None)
    parser.add_argument('--retrain_epochs', type=int, default=50)
    parser.add_argument('--agg_type', type=str, default='attention')
    parser.add_argument('--output_suffix', type=str, default='')

    args = parser.parse_args()

    run_receraser_no_pretrain(
        dataset=args.dataset,
        emb_dim=args.emb_dim,
        n_shards=args.n_shards,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        attention_size=args.attention_size,
        max_epochs_local=args.max_epochs_local,
        max_epochs_agg=args.max_epochs_agg,
        unlearn_ratio=args.unlearn_ratio,
        unlearn_mode=args.unlearn_mode,
        unlearn_user_id=args.unlearn_user_id,
        retrain_epochs=args.retrain_epochs,
        agg_type=args.agg_type,
        output_suffix=args.output_suffix
    )
