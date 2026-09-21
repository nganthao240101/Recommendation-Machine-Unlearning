"""
Method 3: RecEraser - Theo code gốc của tác giả

Hyper-parameters theo bài báo:
- Batch size: 512
- Learning rate: 0.05
- Embedding size: 64
- Attention size k: 32
- Max epochs: 500
- Early stopping: flag_step=10
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


# ============================================================================
# EARLY STOPPING (giống code gốc utility/helper.py)
# ============================================================================

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=10):
    """Early stopping strategy - giống code gốc."""
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or \
       (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print(f"    [Early Stopping] Triggered at step {flag_step}, log={log_value:.6f}")
        should_stop = True
    else:
        should_stop = False

    return best_value, stopping_step, should_stop


# ============================================================================
# CUSTOM DATA LOADER
# ============================================================================

class SimpleDataLoader:
    def __init__(self, data_dir, batch_size=512):
        self.path = data_dir
        self.batch_size = batch_size

        train_file = os.path.join(data_dir, 'train.txt')
        test_file = os.path.join(data_dir, 'test.txt')

        self.n_users, self.n_items = 0, 0
        self.train_items = {}
        self.test_set = {}

        with open(train_file, 'r') as f:
            for line in f.readlines():
                if len(line) > 0:
                    parts = line.strip('\n').split(' ')
                    uid = int(parts[0])
                    items = [int(i) for i in parts[1:]]
                    self.train_items[uid] = items
                    self.n_users = max(self.n_users, uid + 1)
                    self.n_items = max(self.n_items, max(items) + 1 if items else 0)

        with open(test_file, 'r') as f:
            for line in f.readlines():
                if len(line) > 0:
                    parts = line.strip('\n').split(' ')
                    uid = int(parts[0])
                    items = [int(i) for i in parts[1:]]
                    self.test_set[uid] = items

        print(f"  Loaded: {self.n_users} users, {self.n_items} items")


def load_data(dataset='ml-1m', batch_size=512):
    data_path = os.environ.get('RECUNLEARN_DATA_PATH', None)

    if data_path:
        dataset_name = os.environ.get('RECUNLEARN_DATASET', dataset)
        data_dir = os.path.join(data_path, dataset_name)
    else:
        base_dir = os.path.dirname(PROJ)
        data_dir = os.path.join(base_dir, 'data', dataset)

    print(f"  Loading data from: {data_dir}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    return SimpleDataLoader(data_dir, batch_size)


# ============================================================================
# RECERASER MODEL - GIỐNG CODE GỐC
# ============================================================================

class RecEraserBPR(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, num_local, agg_type='attention', attention_size=32):
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

        # Xavier uniform initialization
        for emb in (self.user_embedding, self.item_embedding):
            nn.init.xavier_uniform_(emb.weight)

        # Attention parameters (small init = 0.1)
        self.WA = nn.Parameter(torch.empty(emb_dim, self.attention_size))
        self.BA = nn.Parameter(torch.zeros(self.attention_size))
        self.HA = nn.Parameter(torch.ones(self.attention_size, 1) * 0.1)

        self.WB = nn.Parameter(torch.empty(emb_dim, self.attention_size))
        self.BB = nn.Parameter(torch.zeros(self.attention_size))
        self.HB = nn.Parameter(torch.ones(self.attention_size, 1) * 0.1)

        # Truncated normal init (như code gốc)
        std_w = math.sqrt(2.0 / (emb_dim + self.attention_size))
        nn.init.trunc_normal_(self.WA, mean=0.0, std=std_w, a=-2*std_w, b=2*std_w)
        nn.init.trunc_normal_(self.WB, mean=0.0, std=std_w, a=-2*std_w, b=2*std_w)

        # Transformation parameters (identity init)
        self.trans_W = nn.Parameter(torch.empty(num_local, emb_dim, emb_dim))
        self.trans_B = nn.Parameter(torch.zeros(num_local, emb_dim))
        for k in range(num_local):
            self.trans_W.data[k] = torch.eye(emb_dim)

    def load_pretrained_embeddings(self, user_emb, item_emb):
        """Load pretrained embeddings cho tất cả shards.

        Args:
            user_emb: (n_users, emb_dim) numpy array
            item_emb: (n_items, emb_dim) numpy array
        """
        # user_emb và item_emb có shape (n, emb_dim)
        # Nhưng trong model, embeddings có shape (n, num_local * emb_dim)
        # Chúng ta lặp lại embedding cho mỗi shard

        n_users, emb_dim = user_emb.shape
        n_items = item_emb.shape[0]

        # Reshape: (n, emb_dim) -> (n, num_local * emb_dim)
        # Lặp lại cùng embedding cho mỗi shard
        user_emb_expanded = np.repeat(user_emb, self.num_local, axis=1)  # (n_users, num_local * emb_dim)
        item_emb_expanded = np.repeat(item_emb, self.num_local, axis=1)  # (n_items, num_local * emb_dim)

        # Load vào model
        self.user_embedding.weight.data = torch.FloatTensor(user_emb_expanded)
        self.item_embedding.weight.data = torch.FloatTensor(item_emb_expanded)

        print(f"    [RecEraser] Loaded pretrained embeddings: users={n_users}, items={n_items}, shards={self.num_local}")

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
        """BPR loss với softplus - giống code gốc."""
        pos_scores = (users * pos).sum(dim=1)
        neg_scores = (users * neg).sum(dim=1)
        reg = (users.pow(2).sum() + pos.pow(2).sum() + neg.pow(2).sum()) / users.size(0)
        diff = torch.clamp(pos_scores - neg_scores, -50.0, 50.0)
        mf = torch.mean(F.softplus(-diff))
        reg_loss = decay * reg
        return mf, reg_loss, mf + reg_loss

    def local_loss(self, users, pos_items, neg_items, shard):
        """Local loss cho một shard - giống code gốc."""
        u_e = self._user_emb_for_shard(users, shard)
        pos_e = self._item_emb_for_shard(pos_items, shard)
        neg_e = self._item_emb_for_shard(neg_items, shard)
        mf, reg, total = self._bpr_loss(u_e, pos_e, neg_e)
        return mf, reg, total

    def _attention_aggregate(self, embs, which='user'):
        """Attention aggregation - giống code gốc."""
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
        """Attention aggregation loss - giống code gốc (có stop_gradient)."""
        # Stop gradient như code gốc
        u_es = self._per_shard_user_emb(users).detach()
        pos_i_es = self._per_shard_item_emb(pos_items).detach()
        neg_i_es = self._per_shard_item_emb(neg_items).detach()

        # Apply transformation
        u_e = torch.einsum('bkd,kde->bke', u_es, self.trans_W) + self.trans_B
        pos_e = torch.einsum('bkd,kde->bke', pos_i_es, self.trans_W) + self.trans_B
        neg_e = torch.einsum('bkd,kde->bke', neg_i_es, self.trans_W) + self.trans_B

        # Attention aggregate
        u_agg, u_w = self._attention_aggregate(u_e, 'user')
        pos_agg, _ = self._attention_aggregate(pos_e, 'item')
        neg_agg, _ = self._attention_aggregate(neg_e, 'item')

        # BPR loss
        pos_scores = (u_agg * pos_agg).sum(dim=1)
        neg_scores = (u_agg * neg_agg).sum(dim=1)
        diff = torch.clamp(pos_scores - neg_scores, -50.0, 50.0)
        mf = torch.mean(F.softplus(-diff))

        # Regularization (chỉ attention params như code gốc)
        attn_reg = 1e-6 * (self.HA.pow(2).sum() + self.HB.pow(2).sum())
        trans_reg = 1e-6 * (self.trans_W.pow(2).sum() + self.trans_B.pow(2).sum())
        reg = attn_reg + trans_reg

        return mf, reg, mf + reg, attn_reg, u_w

    def agg_loss_mean(self, users, pos_items, neg_items):
        """Mean aggregation loss - giống code gốc."""
        u_es = self._per_shard_user_emb(users)
        pos_i_es = self._per_shard_item_emb(pos_items)
        neg_i_es = self._per_shard_item_emb(neg_items)

        pos_scores = (u_es * pos_i_es).sum(dim=2)
        neg_scores = (u_es * neg_i_es).sum(dim=2)
        pos_score = pos_scores.mean(dim=1)
        neg_score = neg_scores.mean(dim=1)
        diff = torch.clamp(pos_score - neg_score, -50.0, 50.0)
        mf = torch.mean(F.softplus(-diff))

        reg = 0.01 * (u_es.pow(2).sum() + pos_i_es.pow(2).sum() +
                       neg_i_es.pow(2).sum()) / u_es.size(0)

        return mf, reg, mf + reg, torch.zeros(1), None

    @torch.no_grad()
    def predict(self, users, items):
        """Predict scores - giống code gốc."""
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
# DATA PARTITIONER - GIỐNG CODE GỐC
# ============================================================================

class DataPartitioner:
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
        print(f"    [RecEraser] Shard sizes: min={shard_counts.min()}, max={shard_counts.max()}")

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
# EVALUATION - GIỐNG CODE GỐC
# ============================================================================

def evaluate_model(model, train_data, test_data, n_users, n_items, device, Ks=[10, 20, 50]):
    """Evaluate model - giống cách đánh giá trong code gốc."""
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
# TRAINING VỚI EARLY STOPPING - GIỐNG CODE GỐC
# ============================================================================

def train_local_model(model, shard_data, n_items, device, shard_id, test_data,
                    train_data, n_users, batch_size=512, lr=0.05, max_epochs=500,
                    early_stopping_patience=10, verbose=True):
    """Train local model với early stopping - giống code gốc."""

    # Count samples
    n_samples = sum(len(items) for items in shard_data.values())
    if n_samples == 0:
        return 0.0

    optimizer = Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    # Early stopping variables
    cur_best = 0.0
    stopping_step = 0

    if verbose:
        print(f"    [Local {shard_id}] Training with early stopping (patience={early_stopping_patience})...")

    for epoch in range(max_epochs):
        t1 = time.time()

        # Build samples
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
            # Sample batch
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

        # Evaluate mỗi 5 epochs (như code gốc)
        if (epoch + 1) % 5 == 0:
            metrics = evaluate_model(model, train_data, test_data, n_users, n_items, device)
            recall = metrics['recall'][0]

            elapsed = time.time() - t1
            if verbose:
                print(f"    [Local {shard_id}] Epoch {epoch+1} [{elapsed:.1f}s]: "
                      f"loss={loss_sum:.5f}, recall={recall:.5f}")

            # Early stopping
            cur_best, stopping_step, should_stop = early_stopping(
                recall, cur_best, stopping_step, expected_order='acc', flag_step=early_stopping_patience)

            if should_stop:
                if verbose:
                    print(f"    [Local {shard_id}] Early stopping at epoch {epoch+1}")
                break

    return loss_sum


def train_local_model_fast(model, shard_data, n_items, device, shard_id, batch_size=512, lr=0.05, max_epochs=50):
    """Train local model NHANH - không evaluate, không early stopping.
    Dùng cho unlearn để đo thời gian thực tế.
    """
    n_samples = sum(len(items) for items in shard_data.values())
    if n_samples == 0:
        return 0.0

    optimizer = Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    print(f"    [Local {shard_id}] Training (fast, no eval)...")

    for epoch in range(max_epochs):
        # Build samples
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

    return loss_sum


def train_aggregator(model, train_data, test_data, n_users, n_items, device,
                    batch_size=512, lr=0.05, max_epochs=500,
                    early_stopping_patience=10):
    """Train aggregator với early stopping - giống code gốc."""

    n_samples = sum(len(items) for items in train_data.values())
    if n_samples == 0:
        return 0.0

    optimizer = Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    # Early stopping variables
    cur_best = 0.0
    stopping_step = 0

    print(f"    [Aggregator] Training with early stopping (patience={early_stopping_patience})...")

    for epoch in range(max_epochs):
        t1 = time.time()

        # Build samples
        samples = []
        for user, items in train_data.items():
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
            if model.agg_type == 'attention':
                mf, reg, total, attn, _ = model.agg_loss_attention(users, pos_items, neg_items)
            else:
                mf, reg, total, attn, _ = model.agg_loss_mean(users, pos_items, neg_items)
            total.backward()
            optimizer.step()

            loss_sum += total.item()

        # Evaluate mỗi 5 epochs
        if (epoch + 1) % 5 == 0:
            metrics = evaluate_model(model, train_data, test_data, n_users, n_items, device)
            recall = metrics['recall'][0]

            elapsed = time.time() - t1
            print(f"    [Aggregator] Epoch {epoch+1} [{elapsed:.1f}s]: "
                  f"loss={loss_sum:.5f}, recall={recall:.5f}")

            # Early stopping
            cur_best, stopping_step, should_stop = early_stopping(
                recall, cur_best, stopping_step, expected_order='acc', flag_step=early_stopping_patience)

            if should_stop:
                print(f"    [Aggregator] Early stopping at epoch {epoch+1}")
                break

    return loss_sum


def train_aggregator_fast(model, train_data, n_items, device, batch_size=512, lr=0.05, max_epochs=50):
    """Train aggregator NHANH - không evaluate, không early stopping.
    Dùng cho unlearn để đo thời gian thực tế.
    """
    n_samples = sum(len(items) for items in train_data.values())
    if n_samples == 0:
        return 0.0

    optimizer = Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    print(f"    [Aggregator] Training (fast, no eval)...")

    for epoch in range(max_epochs):
        # Build samples
        samples = []
        for user, items in train_data.items():
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
            if model.agg_type == 'attention':
                mf, reg, total, attn, _ = model.agg_loss_attention(users, pos_items, neg_items)
            else:
                mf, reg, total, attn, _ = model.agg_loss_mean(users, pos_items, neg_items)
            total.backward()
            optimizer.step()

            loss_sum += total.item()

    return loss_sum


# ============================================================================
# RECERASER METHOD
# ============================================================================

class RecEraserMethod:
    def __init__(self, n_users, n_items, emb_dim, n_shards=8, agg_type='attention',
                 attention_size=32, batch_size=512, lr=0.05,
                 max_epochs_local=500, max_epochs_agg=500,
                 early_stopping_patience=10):
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
        self.early_stopping_patience = early_stopping_patience
        self.model = None
        self.partitioner = DataPartitioner(n_shards)

    def train(self, train_data, test_data, device, pretrained_emb_path=None):
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

        # Load pretrained embeddings if provided
        if pretrained_emb_path is not None and os.path.exists(pretrained_emb_path):
            print(f"    [RecEraser] Loading pretrained embeddings from {pretrained_emb_path}...")
            data = np.load(pretrained_emb_path)
            user_emb = data['user_embeddings']
            item_emb = data['item_embeddings']
            self.model.load_pretrained_embeddings(user_emb, item_emb)
        else:
            print("    [RecEraser] No pretrained embeddings, using Xavier init")

        # Debug: Print embedding stats before training
        user_emb = self.model.user_embedding.weight.data
        item_emb = self.model.item_embedding.weight.data
        print(f"    [DEBUG] Before train - User emb: mean={user_emb.mean():.4f}, std={user_emb.std():.4f}, norm={user_emb.norm():.2f}")
        print(f"    [DEBUG] Before train - Item emb: mean={item_emb.mean():.4f}, std={item_emb.std():.4f}, norm={item_emb.norm():.2f}")

        # Phase 1: Local training với early stopping
        print(f"\n    [RecEraser] Phase 1: Local training (max {self.max_epochs_local} epochs)...")
        for shard_id in range(self.n_shards):
            shard_data = self.partitioner.shard_data[shard_id]
            print(f"    [RecEraser] Training shard {shard_id}...")
            train_local_model(
                self.model, shard_data, self.n_items, device, shard_id,
                test_data, train_data, self.n_users,
                batch_size=self.batch_size, lr=self.lr,
                max_epochs=self.max_epochs_local,
                early_stopping_patience=self.early_stopping_patience
            )

        # Phase 2: Aggregator training với early stopping
        print(f"\n    [RecEraser] Phase 2: Aggregator training (max {self.max_epochs_agg} epochs)...")
        train_aggregator(
            self.model, train_data, test_data, self.n_users, self.n_items, device,
            batch_size=self.batch_size, lr=self.lr,
            max_epochs=self.max_epochs_agg,
            early_stopping_patience=self.early_stopping_patience
        )

        # Debug: Print embedding stats after training
        user_emb = self.model.user_embedding.weight.data
        item_emb = self.model.item_embedding.weight.data
        print(f"    [DEBUG] After train - User emb: mean={user_emb.mean():.4f}, std={user_emb.std():.4f}, norm={user_emb.norm():.2f}")
        print(f"    [DEBUG] After train - Item emb: mean={item_emb.mean():.4f}, std={item_emb.std():.4f}, norm={item_emb.norm():.2f}")

        return self.model

    def unlearn(self, unlearn_user_ids, train_data, test_data, device, retrain_epochs=50):
        """Unlearn với thời gian thực tế - chỉ train, không evaluate."""
        affected_shards = self.partitioner.get_affected_shards(unlearn_user_ids)
        print(f"    [RecEraser] Affected shards: {affected_shards}")

        # Retrain affected shards - dùng fast version (không evaluate)
        for shard_id in affected_shards:
            filtered_data = self.partitioner.filter_shard_data(shard_id, set(unlearn_user_ids))
            print(f"    [RecEraser] Retraining shard {shard_id} (fast, no eval)...")
            train_local_model_fast(
                self.model, filtered_data, self.n_items, device, shard_id,
                batch_size=self.batch_size, lr=self.lr,
                max_epochs=retrain_epochs
            )

        # Retrain aggregator - dùng fast version (không evaluate)
        print(f"    [RecEraser] Retraining aggregator (fast, no eval)...")
        train_aggregator_fast(
            self.model, train_data, self.n_items, device,
            batch_size=self.batch_size, lr=self.lr,
            max_epochs=retrain_epochs
        )

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
                early_stopping_patience=10,
                unlearn_ratio=0.1, unlearn_mode='random', unlearn_user_id=None,
                retrain_epochs=50,
                agg_type='attention', pretrained_emb_path=None, output_suffix=''):
    print(f"\n{'='*60}")
    print(f"METHOD 3: RECERASER (giống code gốc)")
    print(f"{'='*60}")
    print(f"Hyper-parameters:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Embedding dim: {emb_dim}")
    print(f"  - Attention size k: {attention_size}")
    print(f"  - Max epochs (local): {max_epochs_local}")
    print(f"  - Max epochs (aggregator): {max_epochs_agg}")
    print(f"  - Early stopping patience: {early_stopping_patience}")
    print(f"  - Aggregation type: {agg_type}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    print(f"\nLoading data from dataset: {dataset}...")
    data = load_data(dataset=dataset, batch_size=batch_size)

    n_users = data.n_users
    n_items = data.n_items
    train_data = data.train_items
    test_data = data.test_set

    random.seed(42)
    all_users = list(train_data.keys())
    n_unlearn = int(len(all_users) * unlearn_ratio)

    # Unlearn mode
    if unlearn_mode == 'single':
        if unlearn_user_id is None:
            raise ValueError("--unlearn_user_id is required when using --unlearn_mode single")
        unlearn_users = {unlearn_user_id}
        n_unlearn = 1
        n_interactions = len(train_data.get(unlearn_user_id, []))
        print(f"\nUnlearn mode: SINGLE USER (ID={unlearn_user_id}, interactions={n_interactions})")
    elif unlearn_mode == 'fewest':
        user_interactions = [(u, len(items)) for u, items in train_data.items()]
        user_interactions.sort(key=lambda x: x[1])
        unlearn_users = set([u for u, _ in user_interactions[:n_unlearn]])
        print(f"\nUnlearn mode: FEWEST interactions ({n_unlearn} users)")
        print(f"  Min: user {user_interactions[0][0]} ({user_interactions[0][1]} interactions)")
    elif unlearn_mode == 'most':
        user_interactions = [(u, len(items)) for u, items in train_data.items()]
        user_interactions.sort(key=lambda x: x[1], reverse=True)
        unlearn_users = set([u for u, _ in user_interactions[:n_unlearn]])
        print(f"\nUnlearn mode: MOST interactions ({n_unlearn} users)")
        print(f"  Max: user {user_interactions[0][0]} ({user_interactions[0][1]} interactions)")
    else:
        unlearn_users = set(random.sample(all_users, n_unlearn))
        print(f"\nUnlearn mode: RANDOM ({n_unlearn} users)")

    method = RecEraserMethod(
        n_users, n_items, emb_dim, n_shards, agg_type,
        attention_size=attention_size,
        batch_size=batch_size, lr=lr,
        max_epochs_local=max_epochs_local,
        max_epochs_agg=max_epochs_agg,
        early_stopping_patience=early_stopping_patience
    )

    print(f"\n--- Phase 1: Train BEFORE unlearning ---")
    t0 = time.time()
    method.train(train_data, test_data, device, pretrained_emb_path=pretrained_emb_path)
    train_time = time.time() - t0

    # Filter out unlearned users from test set for fair comparison (BEFORE and AFTER same set)
    test_data_retained = {u: items for u, items in test_data.items() if u not in unlearn_users}

    # Evaluate before (chỉ trên retained users)
    results_before = method.evaluate(train_data, test_data_retained, device)
    print(f"  Before - R@10: {results_before['recall'][0]:.4f}, NDCG@10: {results_before['ndcg'][0]:.4f}")

    print(f"\n--- Phase 2: Unlearn ---")
    t0 = time.time()
    method.unlearn(unlearn_users, train_data, test_data, device, retrain_epochs=retrain_epochs)
    unlearn_time = time.time() - t0

    # Evaluate after (chỉ trên retained users - cùng tập với before)
    results_after = method.evaluate(train_data, test_data_retained, device)
    print(f"  After - R@10: {results_after['recall'][0]:.4f}, NDCG@10: {results_after['ndcg'][0]:.4f}")
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
            'early_stopping_patience': early_stopping_patience,
            'agg_type': agg_type
        },
        'unlearn_ratio': unlearn_ratio,
        'unlearn_mode': unlearn_mode,
        'n_unlearn': n_unlearn,
        'unlearn_users_sample': list(unlearn_users)[:10],
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
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--attention_size', type=int, default=32)
    parser.add_argument('--n_shards', type=int, default=8)
    parser.add_argument('--max_epochs_local', type=int, default=500)
    parser.add_argument('--max_epochs_agg', type=int, default=500)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--agg_type', type=str, default='attention', choices=['attention', 'mean'])
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)
    parser.add_argument('--unlearn_mode', type=str, default='random',
                       choices=['random', 'fewest', 'most', 'single'],
                       help='random: ngau nhien, fewest: it interaction nhat, most: nhieu interaction nhat, single: 1 user')
    parser.add_argument('--unlearn_user_id', type=int, default=None,
                       help='Chi dinh user ID cu the de unlearn (dung voi --unlearn_mode single)')
    parser.add_argument('--retrain_epochs', type=int, default=50)
    parser.add_argument('--pretrained_emb_path', type=str, default=None,
                       help='Duong dan den file pretrained embeddings (npz)')
    parser.add_argument('--output_suffix', type=str, default='')

    args = parser.parse_args()

    run_receraser(
        dataset=args.dataset,
        emb_dim=args.emb_dim,
        n_shards=args.n_shards,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        attention_size=args.attention_size,
        max_epochs_local=args.max_epochs_local,
        max_epochs_agg=args.max_epochs_agg,
        early_stopping_patience=args.early_stopping_patience,
        agg_type=args.agg_type,
        pretrained_emb_path=args.pretrained_emb_path,
        unlearn_ratio=args.unlearn_ratio,
        unlearn_mode=args.unlearn_mode,
        unlearn_user_id=args.unlearn_user_id,
        retrain_epochs=args.retrain_epochs,
        output_suffix=args.output_suffix
    )
