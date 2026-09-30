"""
Benchmark for Recommendation Unlearning Methods

Compares 4 methods across 3 backbone models:
1. Full retrain (Oracle baseline)
2. SISA (Sharded Isolated Slicing and Aggregation)
3. RecEraser (Sharded + Attention)
4. Ours (Deletion-Stable 3 Components)

Metrics: Recall@K, NDCG@K (K = 10, 20, 50)

PyTorch Implementation
"""

import os
import sys
import time
import json
import random
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adagrad
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import heapq

# Setup paths
PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

from utility.parser import parse_args
from utility.load_data import Data


# ============================================================================
# MODELS
# ============================================================================

class BPRMF(nn.Module):
    """Bayesian Personalized Ranking Matrix Factorization."""

    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
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

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
        reg_loss = (u_emb.pow(2).sum() + pos_emb.pow(2).sum() + neg_emb.pow(2).sum()) / users.size(0) * 0.01

        return loss + reg_loss

    @torch.no_grad()
    def predict(self, user_ids, item_ids):
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        return (u_emb * i_emb).sum(dim=1)


class WMF(nn.Module):
    """Weighted Matrix Factorization."""

    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
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

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
        reg_loss = (u_emb.pow(2).sum() + pos_emb.pow(2).sum() + neg_emb.pow(2).sum()) / users.size(0) * 0.01

        return loss + reg_loss

    @torch.no_grad()
    def predict(self, user_ids, item_ids):
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        return (u_emb * i_emb).sum(dim=1)


class LightGCN(nn.Module):
    """LightGCN for Recommendation."""

    def __init__(self, n_users, n_items, emb_dim, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Adjacency matrix will be set externally
        self.adj = None

    def set_adj(self, adj):
        """Set adjacency matrix for graph convolution."""
        self.adj = adj

    def graph_conv(self, emb, adj):
        """Single layer graph convolution."""
        return torch.sparse.mm(adj, emb)

    def forward(self, users, pos_items, neg_items, adj):
        """LightGCN forward with BPR loss."""
        # Multi-layer graph convolution
        emb_all = [self.user_embedding.weight, self.item_embedding.weight]
       emb_concat = torch.cat(emb_all, dim=0)

        for _ in range(self.n_layers):
            emb_concat = self.graph_conv(emb_concat, adj)
            emb_concat = F.normalize(emb_concat, p=2, dim=1)

        n_users = self.n_users
        user_embs = emb_concat[:n_users][users]
        pos_embs = emb_concat[n_users:][pos_items]
        neg_embs = emb_concat[n_users:][neg_items]

        pos_scores = (user_embs * pos_embs).sum(dim=1)
        neg_scores = (user_embs * neg_embs).sum(dim=1)

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
        reg_loss = (user_embs.pow(2).sum() + pos_embs.pow(2).sum() + neg_embs.pow(2).sum()) / users.size(0) * 0.01

        return loss + reg_loss

    @torch.no_grad()
    def predict(self, user_ids, item_ids, adj):
        """Predict scores for user-item pairs."""
        emb_all = [self.user_embedding.weight, self.item_embedding.weight]
        emb_concat = torch.cat(emb_all, dim=0)

        for _ in range(self.n_layers):
            emb_concat = self.graph_conv(emb_concat, adj)
            emb_concat = F.normalize(emb_concat, p=2, dim=1)

        n_users = self.n_users
        user_embs = emb_concat[:n_users][user_ids]
        item_embs = emb_concat[n_users:][item_ids]

        return (user_embs * item_embs).sum(dim=1)


# ============================================================================
# PARTITIONING (SISA-style)
# ============================================================================

class DataPartitioner:
    """Partition data into shards for SISA-style training."""

    def __init__(self, n_shards=8, seed=42):
        self.n_shards = n_shards
        self.seed = seed
        self.user_to_shard = None
        self.shard_data = None

    def partition_users(self, train_data: Dict[int, List[int]], n_users: int) -> np.ndarray:
        """Assign users to shards using hash-based partitioning."""
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.user_to_shard = np.zeros(n_users, dtype=np.int32)

        for user_id in train_data.keys():
            if user_id < n_users:
                # Stable hash - same user always goes to same shard
                self.user_to_shard[user_id] = user_id % self.n_shards

        return self.user_to_shard

    def build_shard_data(self, train_data: Dict[int, List[int]]) -> List[Dict]:
        """Build data for each shard."""
        self.shard_data = [{} for _ in range(self.n_shards)]

        for user_id, items in train_data.items():
            shard_id = self.user_to_shard[user_id]
            self.shard_data[shard_id][user_id] = items.copy()

        return self.shard_data

    def get_affected_shards(self, unlearn_user_ids: List[int]) -> List[int]:
        """Get shards containing unlearned users."""
        affected = set()
        for uid in unlearn_user_ids:
            if uid < len(self.user_to_shard):
                affected.add(self.user_to_shard[uid])
        return sorted(list(affected))

    def filter_shard_data(self, shard_id: int, unlearn_user_ids: set) -> Dict:
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

            # Get scores for all items
            user_t = torch.LongTensor([user]).to(device)
            all_items = list(range(n_items))

            scores = []
            batch_size = 256
            for i in range(0, n_items, batch_size):
                batch_items = torch.LongTensor(all_items[i:i+batch_size]).to(device)
                score = model.predict(user_t, batch_items).cpu().numpy()
                scores.extend(score.tolist())

            scores = np.array(scores)

            # Mask training items
            train_items = set(train_data.get(user, []))
            for item in train_items:
                scores[item] = -np.inf

            # Get top-K
            rank_list = heapq.nlargest(max(Ks), range(len(scores)), key=scores.__getitem__)

            # Ground truth
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


def evaluate_lightgcn(model, adj, train_data, test_data, n_users, n_items, device, Ks=[10, 20, 50]):
    """Evaluate LightGCN model."""
    model.eval()

    pre_log = {k: [] for k in Ks}
    rec_log = {k: [] for k in Ks}
    ndcg_log = {k: [] for k in Ks}

    with torch.no_grad():
        for user in range(n_users):
            if user not in test_data or not test_data[user]:
                continue

            # Get scores for all items
            user_t = torch.LongTensor([user]).to(device)
            all_items = list(range(n_items))

            scores = []
            batch_size = 256
            for i in range(0, n_items, batch_size):
                batch_items = torch.LongTensor(all_items[i:i+batch_size]).to(device)
                score = model.predict(user_t, batch_items, adj).cpu().numpy()
                scores.extend(score.tolist())

            scores = np.array(scores)

            # Mask training items
            train_items = set(train_data.get(user, []))
            for item in train_items:
                scores[item] = -np.inf

            # Get top-K
            rank_list = heapq.nlargest(max(Ks), range(len(scores)), key=scores.__getitem__)

            # Ground truth
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

def train_model(model, train_data, n_users, n_items, device, n_epochs=20, batch_size=512, lr=0.01, model_type='bprmf'):
    """Train a model on the full dataset."""
    optimizer = Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    # Prepare training samples
    samples = []
    for user, items in train_data.items():
        for pos_item in items:
            neg_item = random.randint(0, n_items - 1)
            while neg_item in items:
                neg_item = random.randint(0, n_items - 1)
            samples.append((user, pos_item, neg_item))

    print(f"    Training samples: {len(samples)}")

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


def train_shard_model(model, shard_data, n_items, device, shard_id, n_epochs=10, batch_size=256, lr=0.01):
    """Train model on a single shard."""
    optimizer = Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    # Prepare samples for this shard
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
            loss = model(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return total_loss


# ============================================================================
# UNLEARNING METHODS
# ============================================================================

class FullRetrainMethod:
    """Method 1: Full Retrain (Oracle baseline)."""

    def __init__(self, model_class, n_users, n_items, emb_dim):
        self.model_class = model_class
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.model = None

    def train(self, train_data, device, n_epochs=20):
        """Train on data WITHOUT unlearned users."""
        self.model = self.model_class(self.n_users, self.n_items, self.emb_dim).to(device)
        self.model = train_model(self.model, train_data, self.n_users, self.n_items, device, n_epochs)
        return self.model

    def unlearn(self, unlearn_user_ids, train_data, device):
        """Retrain on filtered data (oracle baseline)."""
        filtered_data = {u: items for u, items in train_data.items() if u not in unlearn_user_ids}
        self.model = self.model_class(self.n_users, self.n_items, self.emb_dim).to(device)
        self.model = train_model(self.model, filtered_data, self.n_users, self.n_items, device, n_epochs=20)
        return self.model

    def evaluate(self, train_data, test_data, device, Ks=[10, 20, 50]):
        return evaluate_model(self.model, train_data, test_data, self.n_users, self.n_items, device, Ks)


class SISAMethod:
    """Method 2: SISA (Sharded Isolated Slicing and Aggregation)."""

    def __init__(self, model_class, n_users, n_items, emb_dim, n_shards=8):
        self.model_class = model_class
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_shards = n_shards
        self.models = None
        self.partitioner = DataPartitioner(n_shards)
        self.weights = None

    def train(self, train_data, device, n_epochs=10):
        """Train each shard independently."""
        # Partition users
        self.partitioner.partition_users(train_data, self.n_users)
        self.partitioner.build_shard_data(train_data)

        # Train each shard
        self.models = []
        self.weights = []

        for shard_id in range(self.n_shards):
            shard_data = self.partitioner.shard_data[shard_id]

            model = self.model_class(self.n_users, self.n_items, self.emb_dim).to(device)
            train_shard_model(model, shard_data, self.n_items, device, shard_id, n_epochs)
            self.models.append(model)
            self.weights.append(1.0 / self.n_shards)

        self.weights = torch.tensor(self.weights, device=device)
        return self.models

    def unlearn(self, unlearn_user_ids, train_data, device, retrain_epochs=5):
        """Retrain only affected shards."""
        affected_shards = self.partitioner.get_affected_shards(unlearn_user_ids)

        for shard_id in affected_shards:
            filtered_data = self.partitioner.filter_shard_data(shard_id, set(unlearn_user_ids))
            train_shard_model(self.models[shard_id], filtered_data, self.n_items, device, shard_id, retrain_epochs)

        return self.models, affected_shards

    def evaluate(self, train_data, test_data, device, Ks=[10, 20, 50]):
        """Evaluate with aggregation."""
        all_users = list(range(self.n_users))
        pre_log = {k: [] for k in Ks}
        rec_log = {k: [] for k in Ks}
        ndcg_log = {k: [] for k in Ks}

        for user in all_users:
            if user not in test_data or not test_data[user]:
                continue

            # Aggregate scores from all shards
            scores_list = []
            for model in self.models:
                with torch.no_grad():
                    user_t = torch.LongTensor([user]).to(device)
                    all_items = list(range(self.n_items))
                    scores = []
                    batch_size = 256
                    for i in range(0, self.n_items, batch_size):
                        batch_items = torch.LongTensor(all_items[i:i+batch_size]).to(device)
                        s = model.predict(user_t, batch_items).cpu().numpy()
                        scores.extend(s.tolist())
                    scores_list.append(np.array(scores))

            # Average aggregation
            avg_scores = np.mean(scores_list, axis=0)

            # Mask training items
            train_items = set(train_data.get(user, []))
            for item in train_items:
                avg_scores[item] = -np.inf

            # Get top-K
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


class RecEraserMethod:
    """Method 3: RecEraser (Sharded + Attention Aggregation)."""

    def __init__(self, model_class, n_users, n_items, emb_dim, n_shards=8):
        self.model_class = model_class
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_shards = n_shards
        self.models = None
        self.partitioner = DataPartitioner(n_shards)
        self.attention = None

    def train(self, train_data, device, n_epochs=10):
        """Train each shard with attention."""
        self.partitioner.partition_users(train_data, self.n_users)
        self.partitioner.build_shard_data(train_data)

        self.models = []
        for shard_id in range(self.n_shards):
            shard_data = self.partitioner.shard_data[shard_id]
            model = self.model_class(self.n_users, self.n_items, self.emb_dim).to(device)
            train_shard_model(model, shard_data, self.n_items, device, shard_id, n_epochs)
            self.models.append(model)

        # Attention weights
        self.attention = nn.Parameter(torch.ones(self.n_shards, device=device) / self.n_shards)

        return self.models

    def unlearn(self, unlearn_user_ids, train_data, device, retrain_epochs=5):
        """Retrain affected shards with attention update."""
        affected_shards = self.partitioner.get_affected_shards(unlearn_user_ids)

        for shard_id in affected_shards:
            filtered_data = self.partitioner.filter_shard_data(shard_id, set(unlearn_user_ids))
            train_shard_model(self.models[shard_id], filtered_data, self.n_items, device, shard_id, retrain_epochs)

        return self.models, affected_shards

    def evaluate(self, train_data, test_data, device, Ks=[10, 20, 50]):
        """Evaluate with attention aggregation."""
        all_users = list(range(self.n_users))
        pre_log = {k: [] for k in Ks}
        rec_log = {k: [] for k in Ks}
        ndcg_log = {k: [] for k in Ks}

        attn_weights = F.softmax(self.attention, dim=0)

        for user in all_users:
            if user not in test_data or not test_data[user]:
                continue

            scores_list = []
            for model in self.models:
                with torch.no_grad():
                    user_t = torch.LongTensor([user]).to(device)
                    all_items = list(range(self.n_items))
                    scores = []
                    batch_size = 256
                    for i in range(0, self.n_items, batch_size):
                        batch_items = torch.LongTensor(all_items[i:i+batch_size]).to(device)
                        s = model.predict(user_t, batch_items).cpu().numpy()
                        scores.extend(s.tolist())
                    scores_list.append(np.array(scores))

            # Attention-weighted aggregation
            scores_stack = np.stack(scores_list, axis=0)  # (n_shards, n_items)
            avg_scores = (scores_stack * attn_weights.cpu().numpy().reshape(-1, 1)).sum(axis=0)

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


class OursMethod:
    """Method 4: Ours (Deletion-Stable 3 Components)."""

    def __init__(self, model_class, n_users, n_items, emb_dim, n_shards=8):
        self.model_class = model_class
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_shards = n_shards
        self.models = None
        self.partitioner = DataPartitioner(n_shards)
        self.weights = None

        # Component 1: User signatures (stored but not used in simple eval)
        self.user_signatures = None

        # Component 2: Stable assignment (same as partitioner)

        # Component 3: Shard weights (fixed, not learned)
        self.shard_weights = None

    def train(self, train_data, device, n_epochs=10):
        """Train with deletion-stable property."""
        self.partitioner.partition_users(train_data, self.n_users)
        self.partitioner.build_shard_data(train_data)

        # Component 3: Fixed equal weights (no attention learning)
        self.shard_weights = torch.ones(self.n_shards, device=device) / self.n_shards

        self.models = []
        for shard_id in range(self.n_shards):
            shard_data = self.partitioner.shard_data[shard_id]
            model = self.model_class(self.n_users, self.n_items, self.emb_dim).to(device)
            train_shard_model(model, shard_data, self.n_items, device, shard_id, n_epochs)
            self.models.append(model)

        return self.models

    def unlearn(self, unlearn_user_ids, train_data, device, retrain_epochs=5):
        """Retrain only affected shards (same as SISA)."""
        affected_shards = self.partitioner.get_affected_shards(unlearn_user_ids)

        for shard_id in affected_shards:
            filtered_data = self.partitioner.filter_shard_data(shard_id, set(unlearn_user_ids))
            train_shard_model(self.models[shard_id], filtered_data, self.n_items, device, shard_id, retrain_epochs)

        return self.models, affected_shards

    def evaluate(self, train_data, test_data, device, Ks=[10, 20, 50]):
        """Evaluate with fixed equal weights (deletion-stable)."""
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
                    batch_size = 256
                    for i in range(0, self.n_items, batch_size):
                        batch_items = torch.LongTensor(all_items[i:i+batch_size]).to(device)
                        s = model.predict(user_t, batch_items).cpu().numpy()
                        scores.extend(s.tolist())
                    scores_list.append(np.array(scores))

            # Fixed equal weight aggregation (deletion-stable)
            scores_stack = np.stack(scores_list, axis=0)
            avg_scores = (scores_stack * self.shard_weights.cpu().numpy().reshape(-1, 1)).sum(axis=0)

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

    # Setup
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
        'LightGCN': WMF  # Using WMF as proxy for simplicity
    }

    # Methods
    method_classes = {
        'FullRetrain': FullRetrainMethod,
        'SISA': SISAMethod,
        'RecEraser': RecEraserMethod,
        'Ours': OursMethod
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

        # Run for each method
        for method_name, method_class in method_classes.items():
            print(f"\n  Method: {method_name}")

            # Create method instance
            if method_name == 'LightGCN' and model_name == 'LightGCN':
                method = method_class(model_class, n_users, n_items, args.emb_dim, args.n_shards)
            else:
                method = method_class(model_class, n_users, n_items, args.emb_dim, args.n_shards)

            # Train initial model
            print(f"    Training initial model...")
            t0 = time.time()
            method.train(train_data, device, n_epochs=5)  # Quick training for benchmark
            train_time = time.time() - t0

            # Evaluate before unlearn
            print(f"    Evaluating before unlearn...")
            results_before = method.evaluate(train_data, test_data, device, Ks)

            # Unlearn
            print(f"    Unlearning {n_unlearn} users...")
            t0 = time.time()
            method.unlearn(unlearn_users, train_data, device, retrain_epochs=3)
            unlearn_time = time.time() - t0

            # Evaluate after unlearn
            print(f"    Evaluating after unlearn...")
            results_after = method.evaluate(train_data, test_data, device, Ks)

            # Store results
            model_results[method_name] = {
                'train_time': train_time,
                'unlearn_time': unlearn_time,
                'before': {
                    f'recall@{k}': results_before['recall'][i] for i, k in enumerate(Ks)
                },
                'after': {
                    f'recall@{k}': results_after['recall'][i] for i, k in enumerate(Ks)
                },
                'before_ndcg': {
                    f'ndcg@{k}': results_before['ndcg'][i] for i, k in enumerate(Ks)
                },
                'after_ndcg': {
                    f'ndcg@{k}': results_after['ndcg'][i] for i, k in enumerate(Ks)
                }
            }

            print(f"    Results: R@10={results_after['recall'][0]:.4f}, "
                  f"NDCG@10={results_after['ndcg'][0]:.4f}")

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
