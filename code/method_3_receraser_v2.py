"""
Method 3: RecEraser V2 - With WMF Pretrained Embeddings

3 cách chia shard:
1. InP (Interaction-based Partition): Dựa trên similarity của user-item embeddings
2. UBP (User-based Partition): Dựa trên user embeddings
3. Random: Random partition (như bài báo gốc)

Cách dùng:
  python method_3_receraser_v2.py --partition_type 1  # InP
  python method_3_receraser_v2.py --partition_type 2  # UBP
  python method_3_receraser_v2.py --partition_type 3  # Random
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
import heapq

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)


# ============================================================================
# DATA LOADER
# ============================================================================

class SimpleDataLoader:
    def __init__(self, data_dir):
        train_file = os.path.join(data_dir, 'train.txt')
        test_file = os.path.join(data_dir, 'test.txt')

        self.n_users, self.n_items = 0, 0
        self.train_items = {}
        self.test_set = {}

        with open(train_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                uid = int(parts[0])
                items = [int(i) for i in parts[1:] if i]
                self.train_items[uid] = items
                self.n_users = max(self.n_users, uid + 1)
                self.n_items = max(self.n_items, max(items) + 1 if items else 0)

        with open(test_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                uid = int(parts[0])
                items = [int(i) for i in parts[1:] if i]
                self.test_set[uid] = items

        print(f"  Loaded: {self.n_users} users, {self.n_items} items")


def load_data(dataset='ml-1m'):
    data_path = os.environ.get('RECUNLEARN_DATA_PATH', None)
    if data_path:
        dataset_name = os.environ.get('RECUNLEARN_DATASET', dataset)
        data_dir = os.path.join(data_path, dataset_name)
    else:
        base_dir = os.path.dirname(PROJ)
        data_dir = os.path.join(base_dir, 'data', dataset)
    return SimpleDataLoader(data_dir)


# ============================================================================
# LOAD PRETRAINED EMBEDDINGS
# ============================================================================

def load_or_train_wmf(dataset='ml-1m', emb_dim=64, max_epochs_wmf=100, batch_size=512, lr=0.05):
    """Load pretrained embeddings từ WMF trained trước đó"""
    # Load from pretrain_wmf.py output path
    pretrained_dir = os.path.join(PROJ, '..', 'data', dataset)
    emb_path = os.path.join(pretrained_dir, f'wmf_embeddings.npz')

    if os.path.exists(emb_path):
        print(f"  Loading pretrained embeddings from: {emb_path}")
        data = np.load(emb_path)
        user_emb = data['user_embeddings']
        item_emb = data['item_embeddings']
        print(f"  Loaded: user_emb={user_emb.shape}, item_emb={item_emb.shape}")
        return user_emb, item_emb
    else:
        print(f"  Pretrained embeddings not found at {emb_path}")
        print(f"  Training WMF from scratch...")
        # Import WMF model
        from train_wmf_pretrain import train_wmf
        train_wmf(dataset=dataset, emb_dim=emb_dim, max_epochs=max_epochs_wmf, batch_size=batch_size, lr=lr)
        # Try loading again
        if os.path.exists(emb_path):
            data = np.load(emb_path)
            user_emb = data['user_embeddings']
            item_emb = data['item_embeddings']
            return user_emb, item_emb
        else:
            raise FileNotFoundError(f"Failed to create pretrained embeddings at {emb_path}")


def load_or_train_receraser(dataset='ml-1m', emb_dim=64, n_shards=8, partition_type=3,
                           max_epochs_wmf=100, max_epochs_local=50, batch_size=512, lr=0.01,
                           lr_finetune=0.001):
    """Load RecEraser model từ file, hoặc train nếu chưa có"""
    partition_names = {1: 'InP', 2: 'UBP', 3: 'Random'}
    partition_name = partition_names.get(partition_type, 'Random')

    checkpoint_dir = os.path.join(PROJ, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'receraser_{dataset}_d{emb_dim}_k{n_shards}_{partition_name}.pt')

    if os.path.exists(checkpoint_path):
        print(f"  Loading RecEraser model from: {checkpoint_path}")
        # Load embeddings first
        user_emb, item_emb = load_or_train_wmf(dataset, emb_dim, max_epochs_wmf, batch_size, lr)
        return user_emb, item_emb, checkpoint_path
    else:
        print(f"  RecEraser model not found, will train from scratch")
        return None, None, checkpoint_path


def save_model(model, path):
    """Save model to file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  Saved model to: {path}")


def load_model(model, path, device):
    """Load model from file"""
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"  Loaded model from: {path}")


# ============================================================================
# DATA PARTITIONER - 3 CÁCH CHIA SHARD
# ============================================================================

class DataPartitioner:
    def __init__(self, n_shards=8, partition_type=1, seed=42):
        """
        partition_type:
          1: InP (Interaction-based Partition)
          2: UBP (User-based Partition)
          3: Random
        """
        self.n_shards = n_shards
        self.partition_type = partition_type
        self.seed = seed
        self.user_to_shard = None
        self.shard_data = None

    def partition_users(self, train_data, n_users, user_emb=None, item_emb=None):
        """Chia users vào các shards"""
        random.seed(self.seed)
        np.random.seed(self.seed)

        if self.partition_type == 3:
            # Random partition
            return self._partition_random(n_users)
        elif self.partition_type == 2:
            # UBP: User-based partition (dựa trên user embeddings)
            return self._partition_ubp(n_users, user_emb)
        elif self.partition_type == 1:
            # InP: Interaction-based partition (dựa trên user-item embeddings)
            return self._partition_inp(train_data, n_users, user_emb, item_emb)
        else:
            raise ValueError(f"Unknown partition_type: {self.partition_type}")

    def _partition_random(self, n_users):
        """Random partition (như bài báo gốc)"""
        self.user_to_shard = np.zeros(n_users, dtype=np.int32)
        for uid in range(n_users):
            self.user_to_shard[uid] = uid % self.n_shards
        print(f"    [Partition] Random: {self.n_shards} shards")
        return self.user_to_shard

    def _partition_ubp(self, n_users, user_emb):
        """UBP: User-based partition - dựa trên similarity của user embeddings"""
        print(f"    [Partition] UBP: {self.n_shards} shards (using user embeddings)")

        # K-means clustering trên user embeddings
        user_emb_valid = user_emb[:n_users]

        # Random init centroids
        centroids_idx = random.sample(range(n_users), self.n_shards)
        centroids = user_emb_valid[centroids_idx]

        # K-means iterations - increase to 50 for better convergence
        for iteration in range(50):
            # Assign users to nearest centroid
            distances = np.zeros((n_users, self.n_shards))
            for k in range(self.n_shards):
                distances[:, k] = np.linalg.norm(user_emb_valid - centroids[k], axis=1)

            new_assignments = np.argmin(distances, axis=1)

            # Check convergence
            if iteration > 0 and np.array_equal(new_assignments, prev_assignments):
                print(f"    [UBP] Converged at iteration {iteration}")
                break
            prev_assignments = new_assignments.copy()

            # Update centroids
            for k in range(self.n_shards):
                mask = new_assignments == k
                if mask.sum() > 0:
                    centroids[k] = user_emb_valid[mask].mean(axis=0)

        self.user_to_shard = new_assignments

        # Print statistics
        shard_counts = np.bincount(self.user_to_shard, minlength=self.n_shards)
        print(f"    [UBP] Shard sizes: min={shard_counts.min()}, max={shard_counts.max()}")

        return self.user_to_shard

    def _partition_inp(self, train_data, n_users, user_emb, item_emb):
        """InP: Interaction-based partition - dựa trên user-item interaction similarity"""
        print(f"    [Partition] InP: {self.n_shards} shards (using user-item embeddings)")

        # Tạo user representation từ user và item embeddings
        user_emb_valid = user_emb[:n_users]

        # Với mỗi user, lấy trung bình item embeddings của các items user đã tương tác
        user_repr = np.zeros((n_users, user_emb.shape[1]))
        for uid, items in train_data.items():
            if uid < n_users and len(items) > 0:
                valid_items = [i for i in items if i < item_emb.shape[0]]
                if valid_items:
                    user_repr[uid] = item_emb[valid_items].mean(axis=0)

        # K-means clustering
        user_repr_valid = user_repr[:n_users]

        # Random init centroids
        centroids_idx = random.sample(range(n_users), self.n_shards)
        centroids = user_repr_valid[centroids_idx]

        # K-means iterations - increase to 50 for better convergence
        for iteration in range(50):
            distances = np.zeros((n_users, self.n_shards))
            for k in range(self.n_shards):
                distances[:, k] = np.linalg.norm(user_repr_valid - centroids[k], axis=1)

            new_assignments = np.argmin(distances, axis=1)

            # Check convergence
            if iteration > 0 and np.array_equal(new_assignments, prev_assignments):
                print(f"    [InP] Converged at iteration {iteration}")
                break
            prev_assignments = new_assignments.copy()

            for k in range(self.n_shards):
                mask = new_assignments == k
                if mask.sum() > 0:
                    centroids[k] = user_repr_valid[mask].mean(axis=0)

        self.user_to_shard = new_assignments

        shard_counts = np.bincount(self.user_to_shard, minlength=self.n_shards)
        print(f"    [InP] Shard sizes: min={shard_counts.min()}, max={shard_counts.max()}")

        return self.user_to_shard

    def build_shard_data(self, train_data):
        self.shard_data = [{} for _ in range(self.n_shards)]
        for user_id, items in train_data.items():
            shard_id = self.user_to_shard[user_id]
            self.shard_data[shard_id][user_id] = items.copy()

        for sid in range(self.n_shards):
            n_users = len(self.shard_data[sid])
            n_interactions = sum(len(items) for items in self.shard_data[sid].values())
            print(f"    [Shards] Shard {sid}: {n_users} users, {n_interactions} interactions")

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
# RECERASER MODEL
# ============================================================================

class RecEraserModel(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, num_local=8, use_attention=True):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.num_local = num_local
        self.use_attention = use_attention
        self.attention_size = 32

        self.user_embedding = nn.Embedding(n_users, num_local * emb_dim)
        self.item_embedding = nn.Embedding(n_items, num_local * emb_dim)

        # Attention parameters (like original code)
        self.WA = nn.Parameter(torch.empty(emb_dim, self.attention_size))
        self.BA = nn.Parameter(torch.zeros(self.attention_size))
        self.HA = nn.Parameter(torch.ones(self.attention_size, 1) * 0.1)

        self.WB = nn.Parameter(torch.empty(emb_dim, self.attention_size))
        self.BB = nn.Parameter(torch.zeros(self.attention_size))
        self.HB = nn.Parameter(torch.ones(self.attention_size, 1) * 0.1)

        # Transformation parameters
        self.trans_W = nn.Parameter(torch.empty(num_local, emb_dim, emb_dim))
        self.trans_B = nn.Parameter(torch.zeros(num_local, emb_dim))
        for k in range(num_local):
            self.trans_W.data[k] = torch.eye(emb_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def load_pretrained_embeddings(self, user_emb, item_emb, device='cpu'):
        """Load pretrained WMF embeddings"""
        n_users, emb_dim = user_emb.shape
        n_items = item_emb.shape[0]

        # Expand: (n, emb_dim) -> (n, num_local * emb_dim)
        user_emb_expanded = np.repeat(user_emb, self.num_local, axis=1)
        item_emb_expanded = np.repeat(item_emb, self.num_local, axis=1)

        self.user_embedding.weight.data = torch.FloatTensor(user_emb_expanded).to(device)
        self.item_embedding.weight.data = torch.FloatTensor(item_emb_expanded).to(device)

        print(f"    [RecEraser] Loaded pretrained embeddings to {device}")

    def _get_shard_emb(self, emb, shard):
        return emb.view(-1, self.num_local, self.emb_dim)[:, shard, :]

    def _attention_aggregate(self, embs):
        """Attention aggregation - combine all shard embeddings"""
        # embs: [B, num_local, D]
        # Score: H^T . ReLU(emb @ W + B)
        hidden = torch.einsum('bkd,dc->bkc', embs, self.WA) + self.BA  # [B, K, A]
        hidden = torch.relu(hidden)
        score = torch.einsum('bkc,ca->bka', hidden, self.HA)  # [B, K, 1]
        attn = torch.softmax(score, dim=1)
        agg = (attn * embs).sum(dim=1)  # [B, D]
        return agg, attn

    def forward(self, users, pos_items, neg_items, shard, use_aggregation=False):
        if users.size(0) == 0:
            return torch.tensor(0.0, device=users.device, requires_grad=True)

        if use_aggregation:
            # Use attention aggregation
            u_es = self.user_embedding(users).view(-1, self.num_local, self.emb_dim)
            pos_es = self.item_embedding(pos_items).view(-1, self.num_local, self.emb_dim)
            neg_es = self.item_embedding(neg_items).view(-1, self.num_local, self.emb_dim)

            # Apply transformation
            u_es = torch.einsum('bkd,kde->bke', u_es, self.trans_W) + self.trans_B
            pos_es = torch.einsum('bkd,kde->bke', pos_es, self.trans_W) + self.trans_B
            neg_es = torch.einsum('bkd,kde->bke', neg_es, self.trans_W) + self.trans_B

            # Aggregate
            u_emb, _ = self._attention_aggregate(u_es)
            pos_emb, _ = self._attention_aggregate(pos_es)
            neg_emb, _ = self._attention_aggregate(neg_es)
        else:
            # Use hard-routing (single shard)
            u_emb = self._get_shard_emb(self.user_embedding(users), shard)
            pos_emb = self._get_shard_emb(self.item_embedding(pos_items), shard)
            neg_emb = self._get_shard_emb(self.item_embedding(neg_items), shard)

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        # Use softplus like original (more stable)
        diff = torch.clamp(pos_scores - neg_scores, -50.0, 50.0)
        loss = torch.mean(torch.nn.functional.softplus(-diff))

        # Regularization (like original)
        reg = (u_emb.pow(2).sum() + pos_emb.pow(2).sum() + neg_emb.pow(2).sum()) / users.size(0)
        reg_loss = 0.01 * reg  # L2 regularization
        return loss + reg_loss

    @torch.no_grad()
    def predict(self, users, items, shard, use_aggregation=False):
        if use_aggregation:
            # Use attention aggregation
            u_es = self.user_embedding(users).view(-1, self.num_local, self.emb_dim)
            i_es = self.item_embedding(items).view(-1, self.num_local, self.emb_dim)

            u_es = torch.einsum('bkd,kde->bke', u_es, self.trans_W) + self.trans_B
            i_es = torch.einsum('bkd,kde->bke', i_es, self.trans_W) + self.trans_B

            u_emb, _ = self._attention_aggregate(u_es)
            i_emb, _ = self._attention_aggregate(i_es)
        else:
            # Use hard-routing (single shard)
            u_emb = self._get_shard_emb(self.user_embedding(users), shard)
            i_emb = self._get_shard_emb(self.item_embedding(items), shard)
        return (u_emb * i_emb).sum(dim=1)


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, partitioner, shard_data, n_items, device, batch_size=512, lr=0.05, max_epochs=100):
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    samples = []
    for user, items in shard_data.items():
        for pos_item in items:
            neg_item = random.randint(0, n_items - 1)
            while neg_item in items:
                neg_item = random.randint(0, n_items - 1)
            samples.append((user, pos_item, neg_item))

    if not samples:
        return

    for epoch in range(max_epochs):
        random.shuffle(samples)
        total_loss = 0
        n_batches = max(1, len(samples) // batch_size)

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(samples))
            batch = samples[start:end]

            if not batch:
                continue

            # Get unique shards in this batch
            shard_ids = set()
            for s in batch:
                user_id = s[0]
                if user_id < len(partitioner.user_to_shard):
                    shard_ids.add(partitioner.user_to_shard[user_id])

            # Train only the shards that have users in this batch
            for shard_id in shard_ids:
                shard_samples = [(s[0], s[1], s[2]) for s in batch
                                if s[0] < len(partitioner.user_to_shard) and
                                partitioner.user_to_shard[s[0]] == shard_id]
                if not shard_samples:
                    continue

                users = torch.LongTensor([s[0] for s in shard_samples]).to(device)
                pos_items = torch.LongTensor([s[1] for s in shard_samples]).to(device)
                neg_items = torch.LongTensor([s[2] for s in shard_samples]).to(device)

                optimizer.zero_grad()
                loss = model(users, pos_items, neg_items, shard_id)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/n_batches:.4f}")


def train_aggregator(model, train_data, n_users, n_items, device, batch_size=512, lr=0.05, max_epochs=50):
    """Train attention aggregator (Phase 2)"""
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    # Create samples from full training data
    samples = []
    for user, items in train_data.items():
        for pos_item in items:
            neg_item = random.randint(0, n_items - 1)
            while neg_item in items:
                neg_item = random.randint(0, n_items - 1)
            samples.append((user, pos_item, neg_item))

    if not samples:
        return

    for epoch in range(max_epochs):
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
            loss = model(users, pos_items, neg_items, shard=0, use_aggregation=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Agg Epoch {epoch+1}: loss={total_loss/n_batches:.4f}")


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, partitioner, train_data, test_data, n_users, n_items, device):
    model.eval()
    rec_log, ndcg_log = [], []

    # Check if model uses attention aggregation
    use_agg = hasattr(model, 'use_attention') and model.use_attention

    with torch.no_grad():
        for user in range(n_users):
            if user not in test_data or not test_data[user]:
                continue

            shard_id = partitioner.user_to_shard[user]
            user_t = torch.LongTensor([user]).to(device)

            scores = []
            for i in range(0, n_items, 256):
                batch_items = torch.LongTensor(list(range(i, min(i+256, n_items)))).to(device)
                score = model.predict(user_t, batch_items, shard_id, use_aggregation=use_agg).cpu().numpy()
                scores.extend(score.tolist())

            scores = np.array(scores)
            train_items = set(train_data.get(user, []))
            for item in train_items:
                scores[item] = -np.inf

            rank_list = heapq.nlargest(10, range(len(scores)), key=scores.__getitem__)
            item_pos = test_data.get(user, [])
            item_set = set(item_pos)

            hit_num = len(set(rank_list) & item_set)
            rec = hit_num / len(item_pos) if len(item_pos) > 0 else 0
            dcg = sum(1.0/np.log2(i+2) for i, item in enumerate(rank_list) if item in item_set)
            idcg = sum(1.0/np.log2(i+2) for i in range(min(len(item_pos), 10)))
            ndcg = dcg/idcg if idcg > 0 else 0

            rec_log.append(rec)
            ndcg_log.append(ndcg)

    model.train()
    return np.mean(rec_log), np.mean(ndcg_log)


# ============================================================================
# MAIN
# ============================================================================

def run_receraser_v2(dataset='ml-1m', emb_dim=64, n_shards=8, partition_type=1,
                    max_epochs_wmf=100, max_epochs_local=50, unlearn_ratio=0.1,
                    unlearn_mode='random',
                    batch_size=512, lr=0.01, lr_finetune=0.001,
                    use_attention=True, output_suffix=''):
    """
    RecEraser với WMF pretrained embeddings

    partition_type:
      1: InP (Interaction-based Partition)
      2: UBP (User-based Partition)
      3: Random
    """
    partition_names = {1: 'InP', 2: 'UBP', 3: 'Random'}
    partition_name = partition_names.get(partition_type, 'Unknown')
    attention_str = 'attn' if use_attention else 'mean'

    print(f"\n{'='*70}")
    print(f"RECERASER V2 - {partition_name} PARTITION")
    print(f"{'='*70}")
    print(f"Dataset: {dataset}")
    print(f"Partition type: {partition_type} ({partition_name})")
    print(f"Embedding dim: {emb_dim}")
    print(f"N shards: {n_shards}")
    print(f"WMF epochs: {max_epochs_wmf}")
    print(f"Local epochs: {max_epochs_local}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    data = load_data(dataset)
    n_users, n_items = data.n_users, data.n_items
    train_data, test_data = data.train_items, data.test_set

    # Select unlearn users
    random.seed(42)
    all_users = list(train_data.keys())
    n_unlearn = int(len(all_users) * unlearn_ratio)

    if unlearn_mode == 'most':
        user_interactions = [(u, len(train_data.get(u, []))) for u in all_users]
        user_interactions.sort(key=lambda x: x[1], reverse=True)
        unlearn_users = set([u for u, _ in user_interactions[:n_unlearn]])
        print(f"\nUnlearn mode: MOST interactions ({n_unlearn} users)")
        print(f"  Max: user {user_interactions[0][0]} ({user_interactions[0][1]} interactions)")
    elif unlearn_mode == 'fewest':
        user_interactions = [(u, len(train_data.get(u, []))) for u in all_users]
        user_interactions.sort(key=lambda x: x[1])
        unlearn_users = set([u for u, _ in user_interactions[:n_unlearn]])
        print(f"\nUnlearn mode: FEWEST interactions ({n_unlearn} users)")
        print(f"  Min: user {user_interactions[0][0]} ({user_interactions[0][1]} interactions)")
    else:
        unlearn_users = set(random.sample(all_users, n_unlearn))
        print(f"\nUnlearn mode: RANDOM ({n_unlearn} users)")

    print(f"Unlearn users: {len(unlearn_users)}")

    # Retained test set
    test_data_retained = {u: items for u, items in test_data.items() if u not in unlearn_users}
    print(f"Test users (retained): {len(test_data_retained)}")

    # =========================================================================
    # STEP 1: Train WMF để lấy pretrained embeddings
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 1: TRAIN WMF FOR PRETRAINED EMBEDDINGS")
    print(f"{'='*70}")

    # Load pretrained embeddings (hoặc train nếu chưa có)
    user_emb, item_emb = load_or_train_wmf(dataset, emb_dim, max_epochs_wmf, batch_size, lr)

    # =========================================================================
    # STEP 2: Partition users dựa trên embeddings
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 2: PARTITION USERS")
    print(f"{'='*70}")

    partitioner = DataPartitioner(n_shards=n_shards, partition_type=partition_type, seed=42)
    partitioner.partition_users(train_data, n_users, user_emb, item_emb)
    partitioner.build_shard_data(train_data)

    # =========================================================================
    # STEP 3: Train RecEraser với pretrained embeddings
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 3: TRAIN RECERASER")
    print(f"{'='*70}")

    partition_names = {1: 'InP', 2: 'UBP', 3: 'Random'}
    partition_name = partition_names.get(partition_type, 'Random')
    attention_str = 'attn' if use_attention else 'mean'
    checkpoint_dir = os.path.join(PROJ, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, f'receraser_{dataset}_d{emb_dim}_k{n_shards}_{partition_name}_{attention_str}.pt')

    model = RecEraserModel(n_users, n_items, emb_dim, num_local=n_shards, use_attention=use_attention).to(device)

    # Check if model already exists
    if os.path.exists(checkpoint_path):
        print(f"  Loading model from checkpoint: {checkpoint_path}")
        load_model(model, checkpoint_path, device)
    else:
        print(f"  Training RecEraser on all shards (first time)...")
        model.load_pretrained_embeddings(user_emb, item_emb, device)
        for shard_id in range(n_shards):
            if partitioner.shard_data[shard_id]:
                print(f"  Training shard {shard_id}...")
                train_model(model, partitioner, partitioner.shard_data[shard_id], n_items, device,
                           batch_size=batch_size, lr=lr_finetune, max_epochs=max_epochs_local)

        # Phase 2: Train aggregator (like original code)
        if use_attention:
            print(f"  Training aggregator (attention)...")
            train_aggregator(model, train_data, n_users, n_items, device,
                           batch_size=batch_size, lr=lr_finetune, max_epochs=max_epochs_local)

        # Save model after training
        # save_model(model, checkpoint_path)  # Disabled due to disk space

    # =========================================================================
    # STEP 4: Evaluate BEFORE unlearn
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 4: EVALUATE BEFORE UNLEARN")
    print(f"{'='*70}")

    # BEFORE: Evaluate on FULL test set (all users)
    # BEFORE: Evaluate on RETAINED test set (same as AFTER - fair comparison)
    recall_before, ndcg_before = evaluate_model(model, partitioner, train_data, test_data_retained, n_users, n_items, device)
    print(f"  Before (RETAINED) - R@10: {recall_before:.4f}, NDCG@10: {ndcg_before:.4f}")

    # =========================================================================
    # STEP 5: Unlearn (retrain affected shards)
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 5: UNLEARN")
    print(f"{'='*70}")

    affected_shards = partitioner.get_affected_shards(unlearn_users)
    print(f"  Affected shards: {affected_shards}")

    # Measure only retrain time
    t0 = time.time()
    for shard_id in affected_shards:
        filtered_data = partitioner.filter_shard_data(shard_id, unlearn_users)
        print(f"  Retraining shard {shard_id}...")
        train_model(model, partitioner, filtered_data, n_items, device,
                   batch_size=batch_size, lr=lr_finetune, max_epochs=max_epochs_local)

    unlearn_time = time.time() - t0

    # Save model after unlearn for next time
    # save_model(model, checkpoint_path)  # Disabled due to disk space
    print(f"  Saved model after unlearn")

    # =========================================================================
    # STEP 6: Evaluate AFTER unlearn
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 6: EVALUATE AFTER UNLEARN")
    print(f"{'='*70}")

    recall_after, ndcg_after = evaluate_model(model, partitioner, train_data, test_data_retained, n_users, n_items, device)
    print(f"  After (RETAINED) - R@10: {recall_after:.4f}, NDCG@10: {ndcg_after:.4f}")
    print(f"  Unlearn time: {unlearn_time:.2f}s")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results = {
        'method': f'RecEraser_{partition_name}',
        'partition_type': partition_type,
        'partition_name': partition_name,
        'dataset': dataset,
        'hyperparameters': {
            'emb_dim': emb_dim,
            'n_shards': n_shards,
            'max_epochs_wmf': max_epochs_wmf,
            'max_epochs_local': max_epochs_local,
            'batch_size': batch_size,
            'lr': lr
        },
        'unlearn_ratio': unlearn_ratio,
        'n_unlearn': n_unlearn,
        'before': {
            'recall@10': recall_before,
            'ndcg@10': ndcg_before
        },
        'after': {
            'recall@10': recall_after,
            'ndcg@10': ndcg_after
        },
        'change_percent': (recall_after - recall_before) / recall_before * 100 if recall_before > 0 else 0,
        'unlearn_time': unlearn_time
    }

    output_path = f'results_receraser_{partition_name.lower()}{output_suffix}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--n_shards', type=int, default=8)
    parser.add_argument('--partition_type', type=int, default=1, choices=[1, 2, 3],
                       help='1=InP, 2=UBP, 3=Random')
    parser.add_argument('--max_epochs_wmf', type=int, default=100)
    parser.add_argument('--max_epochs_local', type=int, default=50)
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)
    parser.add_argument('--unlearn_mode', type=str, default='random',
                       choices=['random', 'fewest', 'most'],
                       help='random: ngau nhien, fewest: it interaction nhat, most: nhieu interaction nhat')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_finetune', type=float, default=0.001)
    parser.add_argument('--use_attention', type=int, default=1, help='1: use attention aggregation, 0: use mean aggregation')
    parser.add_argument('--output_suffix', type=str, default='')

    args = parser.parse_args()

    run_receraser_v2(
        dataset=args.dataset,
        emb_dim=args.emb_dim,
        n_shards=args.n_shards,
        partition_type=args.partition_type,
        max_epochs_wmf=args.max_epochs_wmf,
        max_epochs_local=args.max_epochs_local,
        unlearn_ratio=args.unlearn_ratio,
        unlearn_mode=args.unlearn_mode,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_finetune=args.lr_finetune,
        use_attention=bool(args.use_attention),
        output_suffix=args.output_suffix
    )
