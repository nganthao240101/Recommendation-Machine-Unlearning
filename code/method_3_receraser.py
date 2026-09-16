"""
Method 3: RecEraser - GIỮ NGUYÊN CODE CỦA TÁC GIẢ (RecEraser_BPR_pytorch.py)

Hyper-parameters theo bài báo:
- Batch size: 512
- Learning rate: 0.05
- Embedding size: 64
- Attention size k: 32
- Max epochs: 1000
- Early stopping: Recall@10 không tăng trong 10 epochs liên tiếp
"""

import os
import sys
import time
import json
import random
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adagrad
import heapq

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

from utility.load_data import Data


# ============================================================================
# HYPER-PARAMETERS THEO BÀI BÁO
# ============================================================================

DEFAULT_CONFIG = {
    'batch_size': 512,
    'learning_rate': 0.05,
    'embedding_dim': 64,
    'attention_size': 32,  # k trong paper
    'max_epochs': 1000,
    'max_epochs_agg': 1000,
    'early_stopping_patience': 10,
    'Ks': [10, 20, 50]
}


# ============================================================================
# RECERASER - CODE GỐC CỦA TÁC GIẢ
# ============================================================================

class RecEraserBPR(nn.Module):
    """
    RecEraser BPR Model - GIỮ NGUYÊN CODE GỐC CỦA TÁC GIẢ
    """

    def __init__(self, n_users, n_items, emb_dim, num_local, agg_type='attention',
                 attention_size=32):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.attention_size = attention_size
        self.num_local = num_local
        self.agg_type = agg_type

        # Per-shard embeddings
        self.user_embedding = nn.Embedding(n_users, num_local * emb_dim)
        self.item_embedding = nn.Embedding(n_items, num_local * emb_dim)

        for emb in (self.user_embedding, self.item_embedding):
            nn.init.xavier_uniform_(emb.weight)

        # Attention parameters
        self.WA = nn.Parameter(torch.empty(emb_dim, self.attention_size))
        self.BA = nn.Parameter(torch.zeros(self.attention_size))
        self.HA = nn.Parameter(torch.ones(self.attention_size, 1) * 0.1)

        self.WB = nn.Parameter(torch.empty(emb_dim, self.attention_size))
        self.BB = nn.Parameter(torch.zeros(self.attention_size))
        self.HB = nn.Parameter(torch.ones(self.attention_size, 1) * 0.1)

        std_w = math.sqrt(2.0 / (emb_dim + self.attention_size))
        nn.init.trunc_normal_(self.WA, mean=0.0, std=std_w, a=-2*std_w, b=2*std_w)
        nn.init.trunc_normal_(self.WB, mean=0.0, std=std_w, a=-2*std_w, b=2*std_w)

        self.trans_W = nn.Parameter(torch.empty(num_local, emb_dim, emb_dim))
        self.trans_B = nn.Parameter(torch.zeros(num_local, emb_dim))
        for k in range(num_local):
            self.trans_W.data[k] = torch.eye(emb_dim)

    def _user_emb_for_shard(self, users, shard):
        emb = self.user_embedding(users)
        emb = emb.view(-1, self.num_local, self.emb_dim)
        return emb[:, shard, :]

    def _item_emb_for_shard(self, items, shard):
        emb = self.item_embedding(items)
        emb = emb.view(-1, self.num_local, self.emb_dim)
        return emb[:, shard, :]

    def _per_shard_user_emb(self, users):
        return self.user_embedding(users).view(-1, self.num_local, self.emb_dim)

    def _per_shard_item_emb(self, items):
        return self.item_embedding(items).view(-1, self.num_local, self.emb_dim)

    def _bpr_loss(self, users, pos, neg, decay=0.01):
        pos_scores = (users * pos).sum(dim=1)
        neg_scores = (users * neg).sum(dim=1)
        reg = (users.pow(2).sum() + pos.pow(2).sum() + neg.pow(2).sum()) / users.size(0)
        diff = torch.clamp(pos_scores - neg_scores, -50.0, 50.0)
        mf = torch.mean(F.softplus(-diff))
        reg_loss = decay * reg
        return mf, reg_loss, mf + reg_loss

    def local_loss(self, users, pos_items, neg_items, shard):
        u_e = self._user_emb_for_shard(users, shard)
        pos_e = self._item_emb_for_shard(pos_items, shard)
        neg_e = self._item_emb_for_shard(neg_items, shard)
        mf, reg, total = self._bpr_loss(u_e, pos_e, neg_e)
        return mf, reg, total

    def _attention_aggregate(self, embs, which='user'):
        if which == 'user':
            W, B, H = self.WA, self.BA, self.HA
        else:
            W, B, H = self.WB, self.BB, self.HB

        hidden = torch.einsum('bkd,dc->bkc', embs, W) + B
        hidden = F.relu(hidden)
        score = torch.einsum('bkc,ca->bka', hidden, H)
        attn = F.softmax(score, dim=1)
        agg = (attn * embs).sum(dim=1)
        return agg, attn

    def agg_loss_attention(self, users, pos_items, neg_items):
        u_es = self._per_shard_user_emb(users).detach()
        pos_i_es = self._per_shard_item_emb(pos_items).detach()
        neg_i_es = self._per_shard_item_emb(neg_items).detach()

        u_e = torch.einsum('bkd,kde->bke', u_es, self.trans_W) + self.trans_B
        pos_e = torch.einsum('bkd,kde->bke', pos_i_es, self.trans_W) + self.trans_B
        neg_e = torch.einsum('bkd,kde->bke', neg_i_es, self.trans_W) + self.trans_B

        u_agg, u_w = self._attention_aggregate(u_e, 'user')
        pos_agg, _ = self._attention_aggregate(pos_e, 'item')
        neg_agg, _ = self._attention_aggregate(neg_e, 'item')

        pos_scores = (u_agg * pos_agg).sum(dim=1)
        neg_scores = (u_agg * neg_agg).sum(dim=1)
        diff = torch.clamp(pos_scores - neg_scores, -50.0, 50.0)
        mf = torch.mean(F.softplus(-diff))

        attn_reg = 1e-6 * (self.HA.pow(2).sum() + self.HB.pow(2).sum())
        trans_reg = 1e-6 * (self.trans_W.pow(2).sum() + self.trans_B.pow(2).sum())
        reg = attn_reg + trans_reg

        return mf, reg, mf + reg, attn_reg, u_w

    def agg_loss_mean(self, users, pos_items, neg_items):
        u_es = self._per_shard_user_emb(users)
        pos_i_es = self._per_shard_item_emb(pos_items)
        neg_i_es = self._per_shard_item_emb(neg_items)

        pos_scores = (u_es * pos_i_es).sum(dim=2)
        neg_scores = (u_es * neg_i_es).sum(dim=2)
        pos_score = pos_scores.mean(dim=1)
        neg_score = neg_scores.mean(dim=1)
        diff = torch.clamp(pos_score - neg_score, -50.0, 50.0)
        mf = torch.mean(F.softplus(-diff))

        return mf, torch.zeros(1, device=users.device), mf, torch.zeros(1), None

    @torch.no_grad()
    def predict(self, users, items):
        if self.agg_type == 'attention':
            u_es = self._per_shard_user_emb(users)
            i_es = self._per_shard_item_emb(items)

            u_e = torch.einsum('bkd,kde->bke', u_es, self.trans_W) + self.trans_B
            i_e = torch.einsum('bkd,kde->bke', i_es, self.trans_W) + self.trans_B

            u_agg, _ = self._attention_aggregate(u_e, 'user')
            i_agg, _ = self._attention_aggregate(i_e, 'item')

            scores = (u_agg * i_agg).sum(dim=1)
        else:
            u_es = self._per_shard_user_emb(users)
            i_es = self._per_shard_item_emb(items)

            u_agg = u_es.mean(dim=1)
            i_agg = i_es.mean(dim=1)

            scores = (u_agg * i_agg).sum(dim=1)

        return scores


# ============================================================================
# DATA PARTITIONER
# ============================================================================

class DataPartitioner:
    """Partition data into shards using stable hash."""

    def __init__(self, n_shards=8, seed=42):
        self.n_shards = n_shards
        self.seed = seed
        self.user_to_shard = None
        self.shard_data = None

    def partition_users(self, train_data, n_users):
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.user_to_shard = np.zeros(n_users, dtype=np.int32)

        for user_id in train_data.keys():
            if user_id < n_users:
                self.user_to_shard[user_id] = user_id % self.n_shards

        shard_counts = np.bincount(self.user_to_shard, minlength=self.n_shards)
        print(f"    [RecEraser] Shard sizes: min={shard_counts.min()}, "
              f"max={shard_counts.max()}, mean={shard_counts.mean():.1f}")

        return self.user_to_shard

    def build_shard_data(self, train_data):
        self.shard_data = [{} for _ in range(self.n_shards)]

        for user_id, items in train_data.items():
            shard_id = self.user_to_shard[user_id]
            self.shard_data[shard_id][user_id] = items.copy()

        for sid in range(self.n_shards):
            n_users = len(self.shard_data[sid])
            n_interactions = sum(len(items) for items in self.shard_data[sid].values())
            print(f"    [RecEraser] Shard {sid}: {n_users} users, {n_interactions} interactions")

        return self.shard_data

    def get_affected_shards(self, unlearn_user_ids):
        affected = set()
        for uid in unlearn_user_ids:
            if uid < len(self.user_to_shard):
                affected.add(self.user_to_shard[uid])
        return sorted(list(affected))

    def filter_shard_data(self, shard_id, unlearn_user_ids):
        if self.shard_data is None:
            return {}
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

    with torch.no_grad():
        for user in range(n_users):
            if user not in test_data or not test_data[user]:
                continue

            user_t = torch.LongTensor([user]).to(device)
            all_items = list(range(n_items))

            scores = []
            for i in range(0, n_items, 256):
                batch_items = torch.LongTensor(all_items[i:i+256]).to(device)
                score = model.predict(user_t, batch_items).cpu().numpy()
                scores.extend(score.tolist())

            scores = np.array(scores)

            train_items = set(train_data.get(user, []))
            for item in train_items:
                scores[item] = -np.inf

            rank_list = heapq.nlargest(max(Ks), range(len(scores)), key=scores.__getitem__)

            item_pos = test_data.get(user, [])
            item_set = set(item_pos)

            for k_idx, k in enumerate(Ks):
                hit_list = rank_list[:k]
                hit_num = len(set(hit_list) & item_set)

                pre = hit_num / k if k > 0 else 0
                rec = hit_num / len(item_pos) if len(item_pos) > 0 else 0

                dcg = 0.0
                for i, item in enumerate(hit_list):
                    if item in item_set:
                        dcg += 1.0 / np.log2(i + 2.0)

                idcg = sum(1.0 / np.log2(i + 2.0) for i in range(min(len(item_pos), k)))
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

def train_local_model(model, shard_data, n_items, device, shard_id,
                    batch_size=512, lr=0.05, n_epochs=10):
    optimizer = Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    samples = []
    for user, items in shard_data.items():
        for pos_item in items:
            neg_item = random.randint(0, n_items - 1)
            while neg_item in items:
                neg_item = random.randint(0, n_items - 1)
            samples.append((user, pos_item, neg_item))

    if not samples:
        return 0.0

    for epoch in range(n_epochs):
        random.shuffle(samples)
        total_loss = 0
        n_batches = max(1, len(samples) // batch_size)

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(samples))
            batch = samples[start:end]

            if not batch:
                continue

            users = torch.LongTensor([s[0] for s in batch]).to(device)
            pos_items = torch.LongTensor([s[1] for s in batch]).to(device)
            neg_items = torch.LongTensor([s[2] for s in batch]).to(device)

            optimizer.zero_grad()
            mf, reg, total = model.local_loss(users, pos_items, neg_items, shard_id)
            total.backward()
            optimizer.step()

            total_loss += total.item()

    return total_loss


def train_aggregator(model, train_data, n_items, device,
                    batch_size=512, lr=0.05, n_epochs=10):
    optimizer = Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    samples = []
    for user, items in train_data.items():
        for pos_item in items:
            neg_item = random.randint(0, n_items - 1)
            while neg_item in items:
                neg_item = random.randint(0, n_items - 1)
            samples.append((user, pos_item, neg_item))

    if not samples:
        return 0.0

    for epoch in range(n_epochs):
        random.shuffle(samples)
        total_loss = 0
        n_batches = max(1, len(samples) // batch_size)

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(samples))
            batch = samples[start:end]

            if not batch:
                continue

            users = torch.LongTensor([s[0] for s in batch]).to(device)
            pos_items = torch.LongTensor([s[1] for s in batch]).to(device)
            neg_items = torch.LongTensor([s[2] for s in batch]).to(device)

            optimizer.zero_grad()
            if model.agg_type == 'attention':
                mf, reg, total, attn, _ = model.agg_loss_attention(users, pos_items, neg_items)
            else:
                mf, reg, total, attn, _ = model.agg_loss_mean(users, pos_items, neg_items)
            total.backward()
            optimizer.step()

            total_loss += total.item()

        if (epoch + 1) % 20 == 0:
            print(f"    Aggregator Epoch {epoch+1}: loss={total_loss/n_batches:.4f}")

    return total_loss


# ============================================================================
# RECERASER METHOD
# ============================================================================

class RecEraserMethod:
    """
    Method 3: RecEraser - GIỮ NGUYÊN CODE GỐC CỦA TÁC GIẢ

    Đặc điểm:
    - Per-shard embeddings
    - Two-phase training: local → aggregator
    - Attention mechanism
    - Unlearn: Retrain affected shards + aggregator
      (attention weights THAY ĐỔI - điểm khác biệt với Ours)
    """

    def __init__(self, n_users, n_items, emb_dim, n_shards=8, agg_type='attention',
                 attention_size=32, batch_size=512, lr=0.05,
                 max_epochs_local=500, max_epochs_agg=500):
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_shards = n_shards
        self.agg_type = agg_type
        self.attention_size = attention_size
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs_local = max_epochs_local
        self.max_epochs_agg = max_epochs_agg
        self.model = None
        self.partitioner = DataPartitioner(n_shards)

    def train(self, train_data, device):
        print("    [RecEraser] Partitioning users...")
        self.partitioner.partition_users(train_data, self.n_users)

        print("    [RecEraser] Building shard data...")
        self.partitioner.build_shard_data(train_data)

        print("    [RecEraser] Creating model...")
        self.model = RecEraserBPR(
            self.n_users, self.n_items, self.emb_dim,
            num_local=self.n_shards, agg_type=self.agg_type,
            attention_size=self.attention_size
        ).to(device)

        # Phase 1: Train local models
        print(f"    [RecEraser] Phase 1: Local training ({self.max_epochs_local} epochs)...")
        for shard_id in range(self.n_shards):
            shard_data = self.partitioner.shard_data[shard_id]
            print(f"    [RecEraser] Training shard {shard_id}...")
            train_local_model(self.model, shard_data, self.n_items, device,
                           shard_id, batch_size=self.batch_size, lr=self.lr,
                           n_epochs=self.max_epochs_local)

        # Phase 2: Train aggregator
        print(f"    [RecEraser] Phase 2: Aggregator training ({self.max_epochs_agg} epochs)...")
        train_aggregator(self.model, train_data, self.n_items, device,
                        batch_size=self.batch_size, lr=self.lr,
                        n_epochs=self.max_epochs_agg)

        return self.model

    def unlearn(self, unlearn_user_ids, train_data, device, retrain_epochs=50):
        affected_shards = self.partitioner.get_affected_shards(unlearn_user_ids)
        print(f"    [RecEraser] Affected shards: {affected_shards}")

        # Retrain affected local shards
        print("    [RecEraser] Retraining affected shards...")
        for shard_id in affected_shards:
            filtered_data = self.partitioner.filter_shard_data(shard_id, set(unlearn_user_ids))
            print(f"    [RecEraser] Retraining shard {shard_id}...")
            train_local_model(self.model, filtered_data, self.n_items, device,
                            shard_id, batch_size=self.batch_size, lr=self.lr,
                            n_epochs=retrain_epochs)

        # Retrain aggregator (attention weights THAY ĐỔI)
        print("    [RecEraser] Retraining aggregator (attention weights will change)...")
        train_aggregator(self.model, train_data, self.n_items, device,
                        batch_size=self.batch_size, lr=self.lr,
                        n_epochs=retrain_epochs)

        return self.model, affected_shards

    def evaluate(self, train_data, test_data, device, Ks=[10, 20, 50]):
        return evaluate_model(self.model, train_data, test_data,
                           self.n_users, self.n_items, device, Ks)


# ============================================================================
# MAIN
# ============================================================================

def run_receraser(dataset='ml-1m', emb_dim=64, n_shards=8,
                batch_size=512, lr=0.05, attention_size=32,
                max_epochs_local=500, max_epochs_agg=500,
                unlearn_ratio=0.1, retrain_epochs=50,
                agg_type='attention', output_suffix=''):
    """Run RecEraser method."""
    print(f"\n{'='*60}")
    print(f"METHOD 3: RECERASER (GIỮ NGUYÊN CODE GỐC CỦA TÁC GIẢ)")
    print(f"{'='*60}")
    print(f"Hyper-parameters:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Embedding dim: {emb_dim}")
    print(f"  - Attention size k: {attention_size}")
    print(f"  - Max epochs (local): {max_epochs_local}")
    print(f"  - Max epochs (aggregator): {max_epochs_agg}")
    print(f"  - Aggregation type: {agg_type}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    data_path = os.path.join(os.path.dirname(PROJ), 'data', dataset)
    print(f"\nLoading data from {data_path}...")

    data = Data(
        path=data_path,
        batch_size=batch_size,
        part_type=1,
        part_num=1,
        part_T=5
    )

    n_users = data.n_users
    n_items = data.n_items
    train_data = data.train_items
    test_data = data.test_set

    print(f"Users: {n_users}, Items: {n_items}")
    print(f"Train interactions: {sum(len(v) for v in train_data.values())}")
    print(f"Test users: {len(test_data)}")

    random.seed(42)
    all_users = list(train_data.keys())
    n_unlearn = int(len(all_users) * unlearn_ratio)
    unlearn_users = set(random.sample(all_users, n_unlearn))

    print(f"\nUnlearn ratio: {unlearn_ratio} ({n_unlearn} users)")

    method = RecEraserMethod(
        n_users, n_items, emb_dim, n_shards, agg_type,
        attention_size=attention_size,
        batch_size=batch_size, lr=lr,
        max_epochs_local=max_epochs_local,
        max_epochs_agg=max_epochs_agg
    )

    print(f"\n--- Phase 1: Train BEFORE unlearning ---")
    t0 = time.time()
    method.train(train_data, device)
    train_time = time.time() - t0

    results_before = method.evaluate(train_data, test_data, device)
    print(f"  Before - R@10: {results_before['recall'][0]:.4f}, "
          f"NDCG@10: {results_before['ndcg'][0]:.4f}")

    print(f"\n--- Phase 2: Unlearn (retrain shards + aggregator) ---")
    t0 = time.time()
    method.unlearn(unlearn_users, train_data, device, retrain_epochs=retrain_epochs)
    unlearn_time = time.time() - t0

    results_after = method.evaluate(train_data, test_data, device)
    print(f"  After - R@10: {results_after['recall'][0]:.4f}, "
          f"NDCG@10: {results_after['ndcg'][0]:.4f}")
    print(f"  Unlearn time: {unlearn_time:.2f}s")

    results = {
        'method': f'RecEraser_{agg_type}',
        'dataset': dataset,
        'hyperparameters': {
            'batch_size': batch_size,
            'learning_rate': lr,
            'embedding_dim': emb_dim,
            'attention_size': attention_size,
            'max_epochs_local': max_epochs_local,
            'max_epochs_agg': max_epochs_agg,
            'agg_type': agg_type
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
            'ndcg@20': results_before['ndcg'][1],
            'ndcg@50': results_before['ndcg'][2],
        },
        'after': {
            'recall@10': results_after['recall'][0],
            'recall@20': results_after['recall'][1],
            'recall@50': results_after['recall'][2],
            'ndcg@10': results_after['ndcg'][0],
            'ndcg@20': results_after['ndcg'][1],
            'ndcg@50': results_after['ndcg'][2],
        }
    }

    suffix = f"_{output_suffix}" if output_suffix else ""
    output_path = os.path.join(PROJ, f'results_receraser_{agg_type}{suffix}.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Method 3: RecEraser')
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--attention_size', type=int, default=32)
    parser.add_argument('--n_shards', type=int, default=8)
    parser.add_argument('--max_epochs_local', type=int, default=500)
    parser.add_argument('--max_epochs_agg', type=int, default=500)
    parser.add_argument('--agg_type', type=str, default='attention', choices=['attention', 'mean'])
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)
    parser.add_argument('--retrain_epochs', type=int, default=50)
    parser.add_argument('--output_suffix', type=str, default='')
    args = parser.parse_args()

    run_receraser(
        dataset=args.dataset,
        emb_dim=args.emb_dim,
        n_shards=args.n_shards,
        batch_size=args.batch_size,
        lr=args.lr,
        attention_size=args.attention_size,
        max_epochs_local=args.max_epochs_local,
        max_epochs_agg=args.max_epochs_agg,
        agg_type=args.agg_type,
        unlearn_ratio=args.unlearn_ratio,
        retrain_epochs=args.retrain_epochs,
        output_suffix=args.output_suffix
    )
