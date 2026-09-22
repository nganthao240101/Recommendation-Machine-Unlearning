"""
CHẠY RECERASER GIỐNG CODE GỐC

Các bước:
1. Train WMF để lấy pretrained embeddings
2. Load pretrained embeddings vào RecEraser
3. Train RecEraser với pretrained embeddings
4. Unlearn
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
# WMF MODEL - Train để lấy pretrained embeddings
# ============================================================================

class WMF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, users, pos_items, neg_items, pos_weights, neg_weights):
        u_emb = self.user_embedding(users)
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        # Weighted BPR loss - giống code gốc
        pos_loss = -pos_weights * torch.log(torch.sigmoid(pos_scores) + 1e-10)
        neg_loss = -neg_weights * torch.log(1 - torch.sigmoid(neg_scores) + 1e-10)
        loss = (pos_loss + neg_loss).mean()

        return loss


def train_wmf(train_data, n_users, n_items, emb_dim, device, batch_size=512, lr=0.05, max_epochs=100):
    """Train WMF để lấy pretrained embeddings"""
    model = WMF(n_users, n_items, emb_dim).to(device)
    optimizer = Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    print(f"\n  Training WMF: emb_dim={emb_dim}, epochs={max_epochs}")

    for epoch in range(max_epochs):
        samples = []
        for user, items in train_data.items():
            for pos_item in items:
                neg_item = random.randint(0, n_items - 1)
                while neg_item in items:
                    neg_item = random.randint(0, n_items - 1)
                # Weight = 1 + log(1 + count) - sublinear weighting
                weight = 1.0
                samples.append((user, pos_item, neg_item, weight, weight))

        random.shuffle(samples)
        total_loss = 0
        n_batches = max(1, len(samples) // batch_size)

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(samples))
            batch = samples[start:end]

            users = torch.LongTensor([s[0] for s in batch]).to(device)
            pos_items = torch.LongTensor([s[1] for s in batch]).to(device)
            neg_items = torch.LongTensor([s[2] for s in batch]).to(device)
            pos_weights = torch.FloatTensor([s[3] for s in batch]).to(device)
            neg_weights = torch.FloatTensor([s[4] for s in batch]).to(device)

            optimizer.zero_grad()
            loss = model(users, pos_items, neg_items, pos_weights, neg_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/n_batches:.4f}")

    return model


# ============================================================================
# RECERASER MODEL - Giống code gốc
# ============================================================================

import math


class RecEraserBPR(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, num_local, agg_type='attention', attention_size=32):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.attention_size = attention_size
        self.num_local = num_local
        self.agg_type = agg_type
        self.Ks = [10, 20, 50]

        # Per-shard embeddings (shared)
        self.user_embedding = nn.Embedding(n_users, num_local * emb_dim)
        self.item_embedding = nn.Embedding(n_items, num_local * emb_dim)

        # Xavier uniform initialization - GIỐNG CODE GỐC
        for emb in (self.user_embedding, self.item_embedding):
            nn.init.xavier_uniform_(emb.weight)

        # Attention parameters
        self.WA = nn.Parameter(torch.empty(emb_dim, self.attention_size))
        self.BA = nn.Parameter(torch.zeros(self.attention_size))
        self.HA = nn.Parameter(torch.ones(self.attention_size, 1) * 0.1)

        self.WB = nn.Parameter(torch.empty(emb_dim, self.attention_size))
        self.BB = nn.Parameter(torch.zeros(self.attention_size))
        self.HB = nn.Parameter(torch.ones(self.attention_size, 1) * 0.1)

        # Truncated normal init - GIỐNG CODE GỐC
        std_w = math.sqrt(2.0 / (emb_dim + self.attention_size))
        nn.init.trunc_normal_(self.WA, mean=0.0, std=std_w, a=-2*std_w, b=2*std_w)
        nn.init.trunc_normal_(self.WB, mean=0.0, std=std_w, a=-2*std_w, b=2*std_w)

        # Transformation parameters - GIỐNG CODE GỐC (identity init)
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

    def local_loss(self, users, pos, neg, shard, decay=0.01):
        """Local BPR loss cho 1 shard - GIỐNG CODE GỐC"""
        u_emb = self._user_emb_for_shard(users, shard)
        pos_emb = self._item_emb_for_shard(pos, shard)
        neg_emb = self._item_emb_for_shard(neg, shard)

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        # BPR loss with softplus - GIỐNG CODE GỐC
        reg = (u_emb.pow(2).sum() + pos_emb.pow(2).sum() + neg_emb.pow(2).sum()) / users.size(0)
        diff = torch.clamp(pos_scores - neg_scores, -50.0, 50.0)
        mf = torch.mean(F.softplus(-diff))
        reg_loss = decay * reg

        return mf, reg_loss, mf + reg_loss

    def batch_ratings_full(self, users, items):
        """Full rating với attention aggregation - GIỐNG CODE GỐC"""
        u_emb = self._per_shard_user_emb(users)  # [B, num_local, D]
        i_emb = self._per_shard_item_emb(items)  # [B, num_local, D]

        # Transform each shard
        u_transformed = torch.einsum('bld,kdl->blk', u_emb, self.trans_W) + self.trans_B
        i_transformed = torch.einsum('bld,kdl->blk', i_emb, self.trans_W) + self.trans_B

        # Attention
        ui = u_transformed.unsqueeze(2) * i_transformed.unsqueeze(1)
        ui = ui.view(-1, self.num_local, self.emb_dim)

        e = torch.tanh(torch.einsum('bld,dl->bl', ui, self.WA) + self.BA)
        e = torch.einsum('bl,ld->bd', e, self.HA).squeeze(-1)
        alpha = F.softmax(e, dim=1)

        # Aggregate
        agg_u = (alpha.unsqueeze(-1) * u_transformed).sum(dim=1)
        agg_i = (alpha.unsqueeze(-1) * i_transformed).sum(dim=1)

        scores = (agg_u * agg_i).sum(dim=1)
        return scores

    def batch_ratings_local(self, users, items, shard):
        """Local rating cho 1 shard - GIỐNG CODE GỐC"""
        u_emb = self._user_emb_for_shard(users, shard)
        i_emb = self._item_emb_for_shard(items, shard)

        scores = (u_emb * i_emb).sum(dim=1)
        return scores

    def load_pretrained_embeddings(self, user_emb, item_emb):
        """Load pretrained WMF embeddings"""
        n_users, emb_dim = user_emb.shape
        n_items = item_emb.shape[0]

        user_emb_expanded = np.repeat(user_emb, self.num_local, axis=1)
        item_emb_expanded = np.repeat(item_emb, self.num_local, axis=1)

        self.user_embedding.weight.data = torch.FloatTensor(user_emb_expanded)
        self.item_embedding.weight.data = torch.FloatTensor(item_emb_expanded)

        print(f"    [RecEraser] Loaded pretrained embeddings: users={n_users}, items={n_items}, shards={self.num_local}")


import torch.nn.functional as F


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

        shard_counts = np.bincount(self.user_to_shard, minlength=self.n_shards)
        print(f"    [Partitioner] Shard sizes: {shard_counts}")

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
            rate_batch = rate_batch.cpu().numpy().copy()

            # Mask training items
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
    """Train local model cho 1 shard - GIỐNG CODE GỐC"""
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
    """Train aggregator - GIỐNG CODE GỐC"""
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

def run_receraser_like_original(dataset='ml-1m', emb_dim=64, n_shards=8,
                                batch_size=512, lr=0.05, attention_size=32,
                                max_epochs_local=500, max_epochs_agg=500,
                                unlearn_ratio=0.1, unlearn_mode='random',
                                unlearn_user_id=None, retrain_epochs=50,
                                agg_type='attention', wmf_epochs=100,
                                output_suffix=''):
    print(f"\n{'='*70}")
    print(f"RECERASER - GIỐNG CODE GỐC")
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
    elif unlearn_mode == 'fewest':
        user_interactions = [(u, len(items)) for u, items in train_data.items()]
        user_interactions.sort(key=lambda x: x[1])
        unlearn_users = set([u for u, _ in user_interactions[:n_unlearn]])
    elif unlearn_mode == 'most':
        user_interactions = [(u, len(items)) for u, items in train_data.items()]
        user_interactions.sort(key=lambda x: x[1], reverse=True)
        unlearn_users = set([u for u, _ in user_interactions[:n_unlearn]])
    else:
        unlearn_users = set(random.sample(all_users, n_unlearn))

    print(f"\nUnlearn: {len(unlearn_users)} users ({unlearn_ratio*100}%)")

    # Filter retained users for fair comparison
    test_data_retained = {u: items for u, items in test_data.items() if u not in unlearn_users}

    # =========================================================================
    # BƯỚC 1: TRAIN WMF ĐỂ LẤY PRETRAINED EMBEDDINGS
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 1: TRAIN WMF FOR PRETRAINED EMBEDDINGS")
    print(f"{'='*70}")

    wmf_output_path = os.path.join(DATA_DIR, dataset, 'wmf_embeddings.npz')

    if os.path.exists(wmf_output_path):
        print(f"  WMF embeddings already exist at {wmf_output_path}")
        print(f"  Loading...")
        wmf_data = np.load(wmf_output_path)
        user_emb_pretrained = wmf_data['user_embeddings']
        item_emb_pretrained = wmf_data['item_embeddings']
    else:
        print(f"  Training WMF ({wmf_epochs} epochs)...")
        t0 = time.time()
        wmf_model = train_wmf(train_data, n_users, n_items, emb_dim, device,
                               batch_size, lr, wmf_epochs)
        wmf_time = time.time() - t0
        print(f"  WMF training time: {wmf_time:.1f}s")

        # Save WMF embeddings
        user_emb_pretrained = wmf_model.user_embedding.weight.data.cpu().numpy()
        item_emb_pretrained = wmf_model.item_embedding.weight.data.cpu().numpy()
        np.savez(wmf_output_path, user_embeddings=user_emb_pretrained,
                 item_embeddings=item_emb_pretrained)
        print(f"  Saved WMF embeddings to {wmf_output_path}")

    # =========================================================================
    # BƯỚC 2: TRAIN RECERASER VỚI PRETRAINED EMBEDDINGS
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 2: TRAIN RECERASER WITH PRETRAINED EMBEDDINGS")
    print(f"{'='*70}")

    # Create partitioner
    partitioner = UserPartitioner(n_shards=n_shards, seed=42)
    partitioner.partition_users(train_data, n_users)
    partitioner.build_shard_data(train_data)

    # Create model
    model = RecEraserBPR(n_users, n_items, emb_dim, num_local=n_shards,
                          agg_type=agg_type, attention_size=attention_size).to(device)

    # Load pretrained embeddings
    print(f"  Loading WMF pretrained embeddings...")
    model.load_pretrained_embeddings(user_emb_pretrained, item_emb_pretrained)

    # Debug: Print embedding stats before training
    u_emb = model.user_embedding.weight.data
    i_emb = model.item_embedding.weight.data
    print(f"    [DEBUG] Before train - User emb: mean={u_emb.mean():.4f}, std={u_emb.std():.4f}")
    print(f"    [DEBUG] Before train - Item emb: mean={i_emb.mean():.4f}, std={i_emb.std():.4f}")

    # Phase 1: Local training
    print(f"\n  Phase 1: Local Training ({n_shards} shards, {max_epochs_local} epochs each)")
    t0 = time.time()
    for shard_id in range(n_shards):
        shard_data = partitioner.shard_data[shard_id]
        print(f"    Training shard {shard_id}...")
        train_local_model(model, shard_data, n_items, device, shard_id,
                          batch_size, lr, max_epochs_local, verbose=True)
    local_time = time.time() - t0
    print(f"  Local training time: {local_time:.1f}s")

    # Phase 2: Aggregator training
    print(f"\n  Phase 2: Aggregator Training ({max_epochs_agg} epochs)")
    t0 = time.time()
    train_aggregator(model, train_data, n_users, n_items, device,
                     batch_size, lr, max_epochs_agg, verbose=True)
    agg_time = time.time() - t0
    print(f"  Aggregator training time: {agg_time:.1f}s")

    train_time = local_time + agg_time

    # Evaluate BEFORE unlearn
    print(f"\n  Evaluating BEFORE unlearn...")
    results_before = evaluate_model(model, train_data, test_data_retained, n_users, n_items, device)
    print(f"    Before - R@10: {results_before['recall'][0]:.4f}, NDCG@10: {results_before['ndcg'][0]:.4f}")

    # =========================================================================
    # BƯỚC 3: UNLEARN
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 3: UNLEARN")
    print(f"{'='*70}")

    affected_shards = partitioner.get_affected_shards(unlearn_users)
    print(f"  Affected shards: {affected_shards}")

    t0 = time.time()
    for shard_id in affected_shards:
        filtered_data = partitioner.filter_shard_data(shard_id, unlearn_users)
        print(f"  Retraining shard {shard_id} ({len(filtered_data)} users)...")
        train_local_model(model, filtered_data, n_items, device, shard_id,
                          batch_size, lr, retrain_epochs, verbose=True)

        # Also retrain aggregator
        print(f"  Retraining aggregator...")
        train_aggregator(model, filtered_data, n_users, n_items, device,
                         batch_size, lr, retrain_epochs, verbose=True)

    unlearn_time = time.time() - t0
    print(f"  Unlearn time: {unlearn_time:.1f}s")

    # Evaluate AFTER unlearn
    print(f"\n  Evaluating AFTER unlearn...")
    results_after = evaluate_model(model, train_data, test_data_retained, n_users, n_items, device)
    print(f"    After - R@10: {results_after['recall'][0]:.4f}, NDCG@10: {results_after['ndcg'][0]:.4f}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results = {
        'method': f'RecEraser_{agg_type}_with_WMF',
        'dataset': dataset,
        'hyperparameters': {
            'emb_dim': emb_dim,
            'n_shards': n_shards,
            'batch_size': batch_size,
            'lr': lr,
            'attention_size': attention_size,
            'max_epochs_local': max_epochs_local,
            'max_epochs_agg': max_epochs_agg,
            'wmf_epochs': wmf_epochs,
            'retrain_epochs': retrain_epochs
        },
        'unlearn_ratio': unlearn_ratio,
        'unlearn_mode': unlearn_mode,
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

    output_path = f'results_receraser_like_original{output_suffix}.json'
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
    parser.add_argument('--max_epochs_local', type=int, default=500)
    parser.add_argument('--max_epochs_agg', type=int, default=500)
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)
    parser.add_argument('--unlearn_mode', type=str, default='random',
                        choices=['random', 'fewest', 'most', 'single'])
    parser.add_argument('--unlearn_user_id', type=int, default=None)
    parser.add_argument('--retrain_epochs', type=int, default=50)
    parser.add_argument('--agg_type', type=str, default='attention', choices=['attention', 'mean'])
    parser.add_argument('--wmf_epochs', type=int, default=100)
    parser.add_argument('--output_suffix', type=str, default='')

    args = parser.parse_args()

    run_receraser_like_original(
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
        wmf_epochs=args.wmf_epochs,
        output_suffix=args.output_suffix
    )
