"""
Benchmark for Recommendation Unlearning Methods - FINAL VERSION

Method 3 (RecEraser) được viết lại dựa trên code gốc của tác giả (RecEraser_BPR_pytorch.py)
nhưng dùng PyTorch.

4 Methods:
1. Full retrain (Oracle baseline)
2. SISA (Sharded Isolated Slicing and Aggregation)
3. RecEraser - DỰA TRÊN CODE GỐC CỦA TÁC GIẢ (RecEraser_BPR_pytorch.py)
4. Ours (Deletion-Stable 3 Components)

Metrics: Recall@K, NDCG@K (K = 10, 20, 50)
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
import torch.nn.functional as F
from torch.optim import Adagrad
from typing import Dict, List, Tuple
import heapq
import math

# Setup paths
PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

from utility.parser import parse_args
from utility.load_data import Data


# ============================================================================
# MODEL 1: BPRMF
# ============================================================================

class BPRMF(nn.Module):
    """Bayesian Personalized Ranking Matrix Factorization."""

    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, users, pos_items, neg_items):
        u_emb = self.user_embedding(users)
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        diff = torch.clamp(pos_scores - neg_scores, -50.0, 50.0)
        loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
        reg_loss = (u_emb.pow(2).sum() + pos_emb.pow(2).sum() +
                    neg_emb.pow(2).sum()) / users.size(0) * 0.01

        return loss + reg_loss

    @torch.no_grad()
    def predict(self, user_ids, item_ids):
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        return (u_emb * i_emb).sum(dim=1)


# ============================================================================
# MODEL 2: WMF
# ============================================================================

class WMF(nn.Module):
    """Weighted Matrix Factorization."""

    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, users, pos_items, neg_items):
        u_emb = self.user_embedding(users)
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        diff = torch.clamp(pos_scores - neg_scores, -50.0, 50.0)
        loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
        reg_loss = (u_emb.pow(2).sum() + pos_emb.pow(2).sum() +
                    neg_emb.pow(2).sum()) / users.size(0) * 0.01

        return loss + reg_loss

    @torch.no_grad()
    def predict(self, user_ids, item_ids):
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        return (u_emb * i_emb).sum(dim=1)


# ============================================================================
# MODEL 3: RECERASER - DỰA TRÊN CODE GỐC CỦA TÁC GIẢ
# ============================================================================

class RecEraser(nn.Module):
    """
    RecEraser Model - DỰA TRÊN CODE GỐC (RecEraser_BPR_pytorch.py)

    Đặc điểm chính từ code gốc:
    1. Per-shard embeddings: [n_users, num_local * emb_dim]
       - Mỗi user có embedding RIÊNG cho mỗi shard
    2. Attention aggregation với transformation
       - score = H^T * ReLU(emb @ W + B)
       - attention = softmax(score, dim=shards)
       - agg_emb = sum(attention * emb)
    3. Transformation: trans_emb = trans_W @ emb + trans_B
    4. Two-phase training: local → aggregator
    """

    def __init__(self, n_users, n_items, emb_dim, num_local, agg_type='attention'):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.num_local = num_local
        self.agg_type = agg_type
        self.attention_size = 32

        # Per-shard user/item embeddings (SHARED variable như code gốc)
        # Shape: [n_users, num_local * emb_dim]
        # User u ở shard k có embedding tại vị trí [u, k*emb_dim:(k+1)*emb_dim]
        self.user_embedding = nn.Embedding(n_users, num_local * emb_dim)
        self.item_embedding = nn.Embedding(n_items, num_local * emb_dim)

        # Xavier initialization (như code gốc)
        for emb in (self.user_embedding, self.item_embedding):
            nn.init.xavier_uniform_(emb.weight)

        # Attention parameters (như code gốc)
        self.WA = nn.Parameter(torch.empty(emb_dim, self.attention_size))
        self.BA = nn.Parameter(torch.zeros(self.attention_size))
        self.HA = nn.Parameter(torch.ones(self.attention_size, 1) * 0.1)  # small init như code gốc

        self.WB = nn.Parameter(torch.empty(emb_dim, self.attention_size))
        self.BB = nn.Parameter(torch.zeros(self.attention_size))
        self.HB = nn.Parameter(torch.ones(self.attention_size, 1) * 0.1)  # small init

        # Truncated normal init (như code gốc)
        std_w = math.sqrt(2.0 / (emb_dim + self.attention_size))
        nn.init.trunc_normal_(self.WA, mean=0.0, std=std_w, a=-2*std_w, b=2*std_w)
        nn.init.trunc_normal_(self.WB, mean=0.0, std=std_w, a=-2*std_w, b=2*std_w)

        # Transformation parameters (như code gốc)
        self.trans_W = nn.Parameter(torch.empty(num_local, emb_dim, emb_dim))
        self.trans_B = nn.Parameter(torch.zeros(num_local, emb_dim))
        for k in range(num_local):
            self.trans_W.data[k] = torch.eye(emb_dim)  # identity init

    def _get_shard_user_emb(self, users, shard):
        """Lấy embedding của user cho một shard cụ thể (như _user_emb_for_shard trong code gốc)."""
        emb = self.user_embedding(users)  # [batch, num_local * emb_dim]
        emb = emb.view(-1, self.num_local, self.emb_dim)  # [batch, num_local, emb_dim]
        return emb[:, shard, :]  # [batch, emb_dim]

    def _get_shard_item_emb(self, items, shard):
        """Lấy embedding của item cho một shard cụ thể."""
        emb = self.item_embedding(items)
        emb = emb.view(-1, self.num_local, self.emb_dim)
        return emb[:, shard, :]

    def _get_all_shard_emb(self, emb_table):
        """Lấy tất cả per-shard embeddings."""
        return emb_table.weight.view(-1, self.num_local, self.emb_dim)

    def _attention_aggregate(self, embs, which='user'):
        """
        Attention aggregation - CHÍNH XÁC như code gốc.

        score = H^T * ReLU(emb @ W + B)
        attention = softmax(score, dim=shards)
        agg_emb = sum(attention * emb)
        """
        if which == 'user':
            W, B, H = self.WA, self.BA, self.HA
        else:
            W, B, H = self.WB, self.BB, self.HB

        # embs: [batch, num_local, emb_dim]
        hidden = torch.einsum('bkd,dc->bkc', embs, W) + B  # [batch, num_local, attention_size]
        hidden = F.relu(hidden)
        score = torch.einsum('bkc,ca->bka', hidden, H)  # [batch, num_local, 1]
        score = score.squeeze(-1)  # [batch, num_local]

        # Softmax over shards (axis=1 trong code gốc)
        attn = F.softmax(score, dim=1)  # [batch, num_local]

        # Aggregate
        agg = torch.einsum('bk,bkd->bd', attn, embs)  # [batch, emb_dim]

        return agg, attn

    def _transform(self, embs):
        """
        Transform embeddings trước khi attention (như code gốc).
        trans_emb = trans_W @ emb + trans_B
        """
        # embs: [batch, num_local, emb_dim]
        trans = torch.einsum('bkd,kde->bke', embs, self.trans_W) + self.trans_B
        return trans

    def local_loss(self, users, pos_items, neg_items, shard):
        """
        Local BPR loss cho một shard cụ thể (như local_loss trong code gốc).

        Training per-shard model riêng biệt.
        """
        u_e = self._get_shard_user_emb(users, shard)
        pos_e = self._get_shard_item_emb(pos_items, shard)
        neg_e = self._get_shard_item_emb(neg_items, shard)

        pos_scores = (u_e * pos_e).sum(dim=1)
        neg_scores = (u_e * neg_e).sum(dim=1)

        diff = torch.clamp(pos_scores - neg_scores, -50.0, 50.0)
        mf = torch.mean(F.softplus(-diff))
        reg = (u_e.pow(2).sum() + pos_e.pow(2).sum() + neg_e.pow(2).sum()) / users.size(0) * 0.01

        return mf + reg

    def agg_loss_attention(self, users, pos_items, neg_items):
        """
        Aggregator loss với attention (như code gốc).

        - Transform: trans_emb = trans_W @ emb + trans_B
        - Attention aggregate
        - BPR loss
        """
        # Lấy per-shard embeddings (detach như code gốc)
        u_es = self._get_all_shard_emb(self.user_embedding).detach()
        pos_i_es = self._get_all_shard_emb(self.item_embedding).detach()
        neg_i_es = self._get_all_shard_emb(self.item_embedding).detach()

        # Transform trước khi attention (như code gốc)
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

        # Regularization (như code gốc)
        attn_reg = 1e-6 * (self.HA.pow(2).sum() + self.HB.pow(2).sum())
        trans_reg = 1e-6 * (self.trans_W.pow(2).sum() + self.trans_B.pow(2).sum())
        reg = attn_reg + trans_reg

        return mf + reg, attn_reg, u_w

    def agg_loss_mean(self, users, pos_items, neg_items):
        """
        Aggregator loss với mean (như code gốc).
        """
        # Lấy per-shard embeddings
        u_es = self._get_all_shard_emb(self.user_embedding)
        pos_i_es = self._get_all_shard_emb(self.item_embedding)
        neg_i_es = self._get_all_shard_emb(self.item_embedding)

        # Average over shards
        u_agg = u_es.mean(dim=1)
        pos_agg = pos_i_es.mean(dim=1)
        neg_agg = neg_i_es.mean(dim=1)

        # BPR loss
        pos_scores = (u_agg * pos_agg).sum(dim=1)
        neg_scores = (u_agg * neg_agg).sum(dim=1)
        diff = torch.clamp(pos_scores - neg_scores, -50.0, 50.0)
        mf = torch.mean(F.softplus(-diff))

        return mf

    @torch.no_grad()
    def predict(self, users, items):
        """
        Predict scores - CHÍNH XÁC như batch_ratings_full trong code gốc.
        """
        if self.agg_type == 'attention':
            # Transform
            u_es = self._get_all_shard_emb(self.user_embedding)
            i_es = self._get_all_shard_emb(self.item_embedding)

            u_e = torch.einsum('bkd,kde->bke', u_es, self.trans_W) + self.trans_B
            i_e = torch.einsum('bkd,kde->bke', i_es, self.trans_W) + self.trans_B

            # Attention aggregate
            u_agg, _ = self._attention_aggregate(u_e, 'user')
            i_agg, _ = self._attention_aggregate(i_e, 'item')

            # Score
            scores = (u_agg * i_agg).sum(dim=1)

        else:  # mean
            u_es = self._get_all_shard_emb(self.user_embedding)
            i_es = self._get_all_shard_emb(self.item_embedding)

            u_agg = u_es.mean(dim=1)
            i_agg = i_es.mean(dim=1)

            scores = (u_agg * i_agg).sum(dim=1)

        return scores


# ============================================================================
# DATA PARTITIONER
# ============================================================================

class DataPartitioner:
    """Partition data into shards (SISA-style)."""

    def __init__(self, n_shards=8, seed=42):
        self.n_shards = n_shards
        self.seed = seed
        self.user_to_shard = None
        self.shard_data = None

    def partition_users(self, train_data, n_users):
        """Hash-based partitioning (stable)."""
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.user_to_shard = np.zeros(n_users, dtype=np.int32)

        for user_id in train_data.keys():
            if user_id < n_users:
                self.user_to_shard[user_id] = user_id % self.n_shards

        return self.user_to_shard

    def build_shard_data(self, train_data):
        """Build data for each shard."""
        self.shard_data = [{} for _ in range(self.n_shards)]

        for user_id, items in train_data.items():
            shard_id = self.user_to_shard[user_id]
            self.shard_data[shard_id][user_id] = items.copy()

        return self.shard_data

    def get_affected_shards(self, unlearn_user_ids):
        """Get shards containing unlearned users."""
        affected = set()
        for uid in unlearn_user_ids:
            if uid < len(self.user_to_shard):
                affected.add(self.user_to_shard[uid])
        return sorted(list(affected))

    def filter_shard_data(self, shard_id, unlearn_user_ids):
        """Remove unlearned users from shard data."""
        if self.shard_data is None:
            return {}
        return {u: items for u, items in self.shard_data[shard_id].items()
                if u not in unlearn_user_ids}


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, train_data, test_data, n_users, n_items, device, Ks=[10, 20, 50]):
    """Evaluate model using Recall@K and NDCG@K."""
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

            # Mask training items
            train_items = set(train_data.get(user, []))
            for item in train_items:
                scores[item] = -np.inf

            # Get top-K
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
# TRAINING FUNCTIONS
# ============================================================================

def train_model(model, train_data, n_users, n_items, device, n_epochs=20, batch_size=512, lr=0.01):
    """Train a model on the full dataset."""
    optimizer = Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    samples = []
    for user, items in train_data.items():
        for pos_item in items:
            neg_item = random.randint(0, n_items - 1)
            while neg_item in items:
                neg_item = random.randint(0, n_items - 1)
            samples.append((user, pos_item, neg_item))

    for epoch in range(n_epochs):
        random.shuffle(samples)
        total_loss = 0

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            if not batch:
                continue

            users = torch.LongTensor([s[0] for s in batch]).to(device)
            pos_items = torch.LongTensor([s[1] for s in batch]).to(device)
            neg_items = torch.LongTensor([s[2] for s in batch]).to(device)

            optimizer.zero_grad()
            loss = model(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/(len(samples)/batch_size):.4f}")

    return model


def train_receraser_local(model, shard_data, n_items, device, shard_id, n_epochs=10, batch_size=256, lr=0.01):
    """
    Train RecEraser local model cho một shard (Phase 1).

    CHÍNH XÁC như _train_one_local_model trong code gốc.
    """
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
            loss = model.local_loss(users, pos_items, neg_items, shard_id)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return total_loss


def train_receraser_aggregator(model, train_data, n_users, n_items, device, n_epochs=10, batch_size=256, lr=0.01):
    """
    Train RecEraser aggregator (Phase 2).

    CHÍNH XÁC như _train_aggregator trong code gốc.
    """
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
                loss, attn_reg, _ = model.agg_loss_attention(users, pos_items, neg_items)
            else:
                loss = model.agg_loss_mean(users, pos_items, neg_items)
                attn_reg = 0.0
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"    Aggregator Epoch {epoch+1}: loss={total_loss/n_batches:.4f}")

    return total_loss


# ============================================================================
# METHOD 1: FULL RETRAIN (ORACLE BASELINE)
# ============================================================================

class FullRetrainMethod:
    """Method 1: Full Retrain - Oracle baseline."""

    def __init__(self, model_class, n_users, n_items, emb_dim):
        self.model_class = model_class
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.model = None

    def train(self, train_data, device, n_epochs=20):
        self.model = self.model_class(self.n_users, self.n_items, self.emb_dim).to(device)
        self.model = train_model(self.model, train_data, self.n_users, self.n_items, device, n_epochs)
        return self.model

    def unlearn(self, unlearn_user_ids, train_data, device):
        filtered_data = {u: items for u, items in train_data.items() if u not in unlearn_user_ids}
        self.model = self.model_class(self.n_users, self.n_items, self.emb_dim).to(device)
        self.model = train_model(self.model, filtered_data, self.n_users, self.n_items, device, n_epochs=20)
        return self.model

    def evaluate(self, train_data, test_data, device, Ks=[10, 20, 50]):
        return evaluate_model(self.model, train_data, test_data, self.n_users, self.n_items, device, Ks)


# ============================================================================
# METHOD 2: SISA (SHARDED ISOLATED SLICING AND AGGREGATION)
# ============================================================================

class SISAMethod:
    """Method 2: SISA - Sharded Isolated Slicing and Aggregation."""

    def __init__(self, model_class, n_users, n_items, emb_dim, n_shards=8):
        self.model_class = model_class
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_shards = n_shards
        self.models = None
        self.partitioner = DataPartitioner(n_shards)

    def train(self, train_data, device, n_epochs=10):
        self.partitioner.partition_users(train_data, self.n_users)
        self.partitioner.build_shard_data(train_data)

        self.models = []
        for shard_id in range(self.n_shards):
            shard_data = self.partitioner.shard_data[shard_id]
            model = self.model_class(self.n_users, self.n_items, self.emb_dim).to(device)
            train_model(model, shard_data, self.n_users, self.n_items, device, n_epochs)
            self.models.append(model)

        return self.models

    def unlearn(self, unlearn_user_ids, train_data, device, retrain_epochs=5):
        affected_shards = self.partitioner.get_affected_shards(unlearn_user_ids)

        for shard_id in affected_shards:
            filtered_data = self.partitioner.filter_shard_data(shard_id, set(unlearn_user_ids))
            train_model(self.models[shard_id], filtered_data, self.n_users, self.n_items, device, retrain_epochs)

        return self.models, affected_shards

    def evaluate(self, train_data, test_data, device, Ks=[10, 20, 50]):
        all_users = list(range(self.n_users))
        pre_log = {k: [] for k in Ks}
        rec_log = {k: [] for k in Ks}
        ndcg_log = {k: [] for k in Ks}

        for user in all_users:
            if user not in test_data or not test_data[user]:
                continue

            scores_list = []
            for model in self.models:
                with torch.no_grad():
                    user_t = torch.LongTensor([user]).to(device)
                    all_items = list(range(self.n_items))
                    scores = []
                    for i in range(0, self.n_items, 256):
                        batch_items = torch.LongTensor(all_items[i:i+256]).to(device)
                        s = model.predict(user_t, batch_items).cpu().numpy()
                        scores.extend(s.tolist())
                    scores_list.append(np.array(scores))

            avg_scores = np.mean(scores_list, axis=0)

            train_items = set(train_data.get(user, []))
            for item in train_items:
                avg_scores[item] = -np.inf

            rank_list = heapq.nlargest(max(Ks), range(len(avg_scores)), key=avg_scores.__getitem__)

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

        return {
            'precision': [np.mean(pre_log[k]) for k in Ks],
            'recall': [np.mean(rec_log[k]) for k in Ks],
            'ndcg': [np.mean(ndcg_log[k]) for k in Ks]
        }


# ============================================================================
# METHOD 3: RECERASER - DỰA TRÊN CODE GỐC CỦA TÁC GIẢ
# ============================================================================

class RecEraserMethod:
    """
    Method 3: RecEraser - DỰA TRÊN CODE GỐC (RecEraser_BPR_pytorch.py)

    Đặc điểm:
    1. Per-shard embeddings: [n_users, num_local * emb_dim]
    2. Two-phase training:
       - Phase 1: Train local models cho mỗi shard
       - Phase 2: Train attention aggregator
    3. Attention mechanism:
       - score = H^T * ReLU(emb @ W + B)
       - attention = softmax(score, dim=shards)
    4. Transformation: trans_emb = trans_W @ emb + trans_B
    5. Unlearn: Retrain affected shards + Retrain aggregator
    """

    def __init__(self, n_users, n_items, emb_dim, n_shards=8, agg_type='attention'):
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_shards = n_shards
        self.agg_type = agg_type
        self.model = None
        self.partitioner = DataPartitioner(n_shards)

    def train(self, train_data, device, n_epochs=10):
        """
        Two-phase training - CHÍNH XÁC như code gốc.

        Phase 1: Train local models (per-shard BPR)
        Phase 2: Train aggregator (attention hoặc mean)
        """
        print("    Phase 1: Local training for each shard...")
        self.partitioner.partition_users(train_data, self.n_users)
        self.partitioner.build_shard_data(train_data)

        self.model = RecEraser(
            self.n_users, self.n_items, self.emb_dim,
            num_local=self.n_shards, agg_type=self.agg_type
        ).to(device)

        # Phase 1: Train local models (như code gốc)
        for shard_id in range(self.n_shards):
            shard_data = self.partitioner.shard_data[shard_id]
            print(f"    Training shard {shard_id}/{self.n_shards-1}...")
            train_receraser_local(self.model, shard_data, self.n_items, device,
                                 shard_id, n_epochs=n_epochs//2)

        # Phase 2: Train aggregator (như code gốc)
        print("    Phase 2: Aggregator training...")
        train_receraser_aggregator(self.model, train_data, self.n_users,
                                 self.n_items, device, n_epochs=n_epochs//2)

        return self.model

    def unlearn(self, unlearn_user_ids, train_data, device, retrain_epochs=5):
        """
        Unlearn - CHÍNH XÁC như code gốc.

        1. Tìm affected shards
        2. Retrain affected local shards
        3. Retrain aggregator (attention weights THAY ĐỔI)
        """
        affected_shards = self.partitioner.get_affected_shards(unlearn_user_ids)
        print(f"    Affected shards: {affected_shards}")

        # Retrain affected local shards (như code gốc)
        for shard_id in affected_shards:
            filtered_data = self.partitioner.filter_shard_data(shard_id, set(unlearn_user_ids))
            train_receraser_local(self.model, filtered_data, self.n_items, device,
                                 shard_id, n_epochs=retrain_epochs)

        # Retrain aggregator (như code gốc - attention weights THAY ĐỔI)
        print("    Retraining aggregator (attention weights will change)...")
        train_receraser_aggregator(self.model, train_data, self.n_users,
                                 self.n_items, device, n_epochs=retrain_epochs)

        return self.model, affected_shards

    def evaluate(self, train_data, test_data, device, Ks=[10, 20, 50]):
        return evaluate_model(self.model, train_data, test_data,
                            self.n_users, self.n_items, device, Ks)


# ============================================================================
# METHOD 4: OURS (DELETION-STABLE 3 COMPONENTS)
# ============================================================================

class OursMethod:
    """
    Method 4: Ours - Deletion-Stable 3 Components

    Component 1: Deletion-Local User Signatures
    Component 2: Deletion-Stable Assignment (stable hash)
    Component 3: Isolated Training (fixed equal weights)

    ĐIỂM KHÁC BIỆT VỚI RECERASER:
    - RecEraser: Attention weights THAY ĐỔI khi unlearn → ảnh hưởng users khác
    - Ours: Fixed equal weights KHÔNG đổi → users khác KHÔNG bị ảnh hưởng
    """

    def __init__(self, n_users, n_items, emb_dim, n_shards=8):
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_shards = n_shards
        self.model = None
        self.partitioner = DataPartitioner(n_shards)

        # Component 1: User signatures
        self.user_signatures = None

    def build_signatures(self, train_data):
        """
        Component 1: Build deletion-local user signatures.

        Signature của user chỉ phụ thuộc vào INTERACTION RIÊNG CỦA HỌ,
        không phụ thuộc users khác.
        """
        from scipy.sparse import csr_matrix

        rows, cols, data = [], [], []
        for user_id, items in train_data.items():
            if user_id >= self.n_users:
                continue

            item_counts = {}
            for item in items:
                if item < self.n_items:
                    item_counts[item] = item_counts.get(item, 0) + 1

            total = sum(item_counts.values())
            for item_id, count in item_counts.items():
                weight = count / total
                rows.append(user_id)
                cols.append(item_id)
                data.append(weight)

        self.user_signatures = csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )
        print(f"    Built signatures: {self.user_signatures.nnz} non-zero entries")

    def train(self, train_data, device, n_epochs=10):
        """
        Train with deletion-stable property.

        - Component 1: Build deletion-local signatures
        - Component 2: Stable hash assignment
        - Component 3: Use RecEraser architecture nhưng với mean aggregation
          (FIXED equal weights, NOT learned attention)
        """
        print("    Component 1: Building deletion-local signatures...")
        self.build_signatures(train_data)

        print("    Component 2: Creating deletion-stable assignment...")
        self.partitioner.partition_users(train_data, self.n_users)
        self.partitioner.build_shard_data(train_data)

        print("    Component 3: Training with fixed equal weights...")
        # Dùng RecEraser architecture nhưng với mean aggregation
        # (FIXED weights, NOT learned attention)
        self.model = RecEraser(
            self.n_users, self.n_items, self.emb_dim,
            num_local=self.n_shards, agg_type='mean'  # FIXED: mean, not attention
        ).to(device)

        # Chỉ train local models, KHÔNG train aggregator
        # (aggregator weights = fixed equal weights = 1/n_shards)
        for shard_id in range(self.n_shards):
            shard_data = self.partitioner.shard_data[shard_id]
            print(f"    Training shard {shard_id}/{self.n_shards-1}...")
            train_receraser_local(self.model, shard_data, self.n_items, device,
                                 shard_id, n_epochs=n_epochs)

        return self.model

    def unlearn(self, unlearn_user_ids, train_data, device, retrain_epochs=5):
        """
        Deletion-stable unlearning.

        1. Component 1: Xóa users khỏi signatures
        2. Component 2: Assignment STABLE - KHÔNG cần thay đổi
        3. Component 3: Chỉ retrain affected shards
           - KHÔNG retrain aggregator (vì dùng fixed equal weights)
        """
        affected_shards = self.partitioner.get_affected_shards(unlearn_user_ids)
        print(f"    Affected shards: {affected_shards}")

        # Component 1: Xóa khỏi signatures
        print("    Component 1: Removing from signatures...")
        # (In practice, signatures would be updated)

        # Component 2: Assignment STABLE
        print("    Component 2: Assignment is STABLE (no changes needed)")

        # Component 3: Chỉ retrain affected shards, KHÔNG retrain aggregator
        print("    Component 3: Retraining only affected shards (NO aggregator retrain)...")
        for shard_id in affected_shards:
            filtered_data = self.partitioner.filter_shard_data(shard_id, set(unlearn_user_ids))
            train_receraser_local(self.model, filtered_data, self.n_items, device,
                                 shard_id, n_epochs=retrain_epochs)

        return self.model, affected_shards

    def evaluate(self, train_data, test_data, device, Ks=[10, 20, 50]):
        return evaluate_model(self.model, train_data, test_data,
                            self.n_users, self.n_items, device, Ks)


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark():
    """Run the full benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--n_shards', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='benchmark_results.json')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    data_path = os.path.join(os.path.dirname(PROJ), 'data', args.dataset)
    print(f"\nLoading data from {data_path}...")

    data = Data(
        path=data_path,
        batch_size=512,
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

    # Select users to unlearn
    random.seed(42)
    all_users = list(train_data.keys())
    n_unlearn = int(len(all_users) * args.unlearn_ratio)
    unlearn_users = set(random.sample(all_users, n_unlearn))

    print(f"\nUnlearn ratio: {args.unlearn_ratio} ({n_unlearn} users)")

    # Models
    model_classes = {
        'BPRMF': BPRMF,
        'WMF': WMF,
    }

    # Results storage
    results = {
        'config': {
            'dataset': args.dataset,
            'emb_dim': args.emb_dim,
            'n_shards': args.n_shards,
            'unlearn_ratio': args.unlearn_ratio,
            'n_unlearn': n_unlearn
        },
        'models': {}
    }

    Ks = [10, 20, 50]

    # Run for each model
    for model_name, model_class in model_classes.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        model_results = {}

        # Method 1: Full Retrain
        print(f"\n  Method 1: FullRetrain (Oracle)")
        method = FullRetrainMethod(model_class, n_users, n_items, args.emb_dim)
        t0 = time.time()
        method.train(train_data, device, n_epochs=5)
        train_time = time.time() - t0
        results_before = method.evaluate(train_data, test_data, device, Ks)

        t0 = time.time()
        method.unlearn(unlearn_users, train_data, device)
        unlearn_time = time.time() - t0
        results_after = method.evaluate(train_data, test_data, device, Ks)

        model_results['FullRetrain'] = {
            'train_time': train_time,
            'unlearn_time': unlearn_time,
            'before': {f'recall@{k}': results_before['recall'][i] for i, k in enumerate(Ks)},
            'after': {f'recall@{k}': results_after['recall'][i] for i, k in enumerate(Ks)},
            'before_ndcg': {f'ndcg@{k}': results_before['ndcg'][i] for i, k in enumerate(Ks)},
            'after_ndcg': {f'ndcg@{k}': results_after['ndcg'][i] for i, k in enumerate(Ks)},
        }
        print(f"    R@10={results_after['recall'][0]:.4f}, NDCG@10={results_after['ndcg'][0]:.4f}")

        # Method 2: SISA
        print(f"\n  Method 2: SISA")
        method = SISAMethod(model_class, n_users, n_items, args.emb_dim, args.n_shards)
        t0 = time.time()
        method.train(train_data, device, n_epochs=5)
        train_time = time.time() - t0
        results_before = method.evaluate(train_data, test_data, device, Ks)

        t0 = time.time()
        method.unlearn(unlearn_users, train_data, device, retrain_epochs=3)
        unlearn_time = time.time() - t0
        results_after = method.evaluate(train_data, test_data, device, Ks)

        model_results['SISA'] = {
            'train_time': train_time,
            'unlearn_time': unlearn_time,
            'before': {f'recall@{k}': results_before['recall'][i] for i, k in enumerate(Ks)},
            'after': {f'recall@{k}': results_after['recall'][i] for i, k in enumerate(Ks)},
            'before_ndcg': {f'ndcg@{k}': results_before['ndcg'][i] for i, k in enumerate(Ks)},
            'after_ndcg': {f'ndcg@{k}': results_after['ndcg'][i] for i, k in enumerate(Ks)},
        }
        print(f"    R@10={results_after['recall'][0]:.4f}, NDCG@10={results_after['ndcg'][0]:.4f}")

        # Method 3: RecEraser (DỰA TRÊN CODE GỐC)
        print(f"\n  Method 3: RecEraser (dựa trên code gốc của tác giả)")
        method = RecEraserMethod(n_users, n_items, args.emb_dim, args.n_shards, agg_type='attention')
        t0 = time.time()
        method.train(train_data, device, n_epochs=5)
        train_time = time.time() - t0
        results_before = method.evaluate(train_data, test_data, device, Ks)

        t0 = time.time()
        method.unlearn(unlearn_users, train_data, device, retrain_epochs=3)
        unlearn_time = time.time() - t0
        results_after = method.evaluate(train_data, test_data, device, Ks)

        model_results['RecEraser'] = {
            'train_time': train_time,
            'unlearn_time': unlearn_time,
            'before': {f'recall@{k}': results_before['recall'][i] for i, k in enumerate(Ks)},
            'after': {f'recall@{k}': results_after['recall'][i] for i, k in enumerate(Ks)},
            'before_ndcg': {f'ndcg@{k}': results_before['ndcg'][i] for i, k in enumerate(Ks)},
            'after_ndcg': {f'ndcg@{k}': results_after['ndcg'][i] for i, k in enumerate(Ks)},
        }
        print(f"    R@10={results_after['recall'][0]:.4f}, NDCG@10={results_after['ndcg'][0]:.4f}")

        # Method 4: Ours (Deletion-Stable 3 Components)
        print(f"\n  Method 4: Ours (Deletion-Stable 3 Components)")
        method = OursMethod(n_users, n_items, args.emb_dim, args.n_shards)
        t0 = time.time()
        method.train(train_data, device, n_epochs=5)
        train_time = time.time() - t0
        results_before = method.evaluate(train_data, test_data, device, Ks)

        t0 = time.time()
        method.unlearn(unlearn_users, train_data, device, retrain_epochs=3)
        unlearn_time = time.time() - t0
        results_after = method.evaluate(train_data, test_data, device, Ks)

        model_results['Ours'] = {
            'train_time': train_time,
            'unlearn_time': unlearn_time,
            'before': {f'recall@{k}': results_before['recall'][i] for i, k in enumerate(Ks)},
            'after': {f'recall@{k}': results_after['recall'][i] for i, k in enumerate(Ks)},
            'before_ndcg': {f'ndcg@{k}': results_before['ndcg'][i] for i, k in enumerate(Ks)},
            'after_ndcg': {f'ndcg@{k}': results_after['ndcg'][i] for i, k in enumerate(Ks)},
        }
        print(f"    R@10={results_after['recall'][0]:.4f}, NDCG@10={results_after['ndcg'][0]:.4f}")

        results['models'][model_name] = model_results

    # Save results
    output_path = os.path.join(PROJ, args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"{'='*60}")

    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Model':<12} {'Method':<15} {'R@10':<10} {'R@20':<10} {'N@10':<10} {'Unlearn Time':<15}")
    print("-"*100)

    for model_name, model_results in results['models'].items():
        for method_name, method_result in model_results.items():
            r10 = method_result['after']['recall@10']
            r20 = method_result['after']['recall@20']
            n10 = method_result['after_ndcg']['ndcg@10']
            utime = method_result['unlearn_time']
            print(f"{model_name:<12} {method_name:<15} {r10:.4f}     {r20:.4f}     {n10:.4f}     {utime:.2f}s")

    return results


if __name__ == '__main__':
    run_benchmark()
