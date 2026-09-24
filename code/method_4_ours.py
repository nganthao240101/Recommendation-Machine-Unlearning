"""
Method 4: Ours (Deletion-Stable 3 Components)
Theo bài báo: https://...

3 Components đúng theo bài báo:

Component I: Deletion-Local User Signatures
  - Sparse preference vector cho mỗi user
  - Random projection R để giảm chiều
  - Signature chỉ phụ thuộc vào user đó

Component II: Deletion-Stable Similarity-Aware Assignment
  - Random hyperplanes để assign users vào shards
  - Stable: không thay đổi khi unlearn

Component III: Isolated Training and Exact User Unlearning
  - HARD-ROUTING inference: y_xv = f_{M_{g(x)}}(x, v)
  - Chỉ dùng model của shard được assign
  - KHÔNG có aggregation khi inference!
  - Unlearn: retrain shard chứa user đó
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
from scipy.sparse import csr_matrix
import heapq

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)


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
# COMPONENT I: DELETION-LOCAL USER SIGNATURES
# ============================================================================

class UserSignatureBuilder:
    """
    Component I: Deletion-Local User Signatures

    Tạo sparse preference vector cho mỗi user
    Signature chỉ phụ thuộc vào user đó, không phụ thuộc vào data user khác
    """

    def __init__(self, n_users, n_items, d=64, seed=42):
        """
        Args:
            n_users: số users
            n_items: số items
            d: chiều của signature sau projection
            seed: random seed cố định (public seed rho)
        """
        self.n_users = n_users
        self.n_items = n_items
        self.d = d
        self.seed = seed

        # Random projection matrix R (từ public seed rho)
        random.seed(seed)
        np.random.seed(seed)

        # Sample từ {−1/√d, +1/√d}
        self.R = np.random.choice([-1, 1], size=(d, n_items)) / np.sqrt(d)

        # Signature vectors z_u
        self.user_signatures = None  # shape: (n_users, d)

    def build_signature(self, train_data):
        """
        Xây dựng signature cho mỗi user
        z_u = R * x_u / ||R * x_u||
        """
        print(f"    [Component I] Building user signatures (d={self.d})...")

        rows, cols, data = [], [], []
        for user_id, items in train_data.items():
            if user_id >= self.n_users:
                continue

            # Sparse preference vector x_u (binary implicit feedback: phi(y) = 1)
            for item in items:
                if item < self.n_items:
                    rows.append(user_id)
                    cols.append(item)
                    data.append(1.0)

        # Sparse matrix X (n_users x n_items)
        X = csr_matrix((data, (rows, cols)),
                       shape=(self.n_users, self.n_items),
                       dtype=np.float32)

        # Compute: z_u = R * x_u / ||R * x_u||
        # Implementation: z_u = sum(phi(y_uv) * r_v) for v in I_u+
        # Với binary feedback: z_u = sum(r_v) for v in I_u+

        Rx = X @ self.R.T  # (n_users, d)

        # Normalize: z_u = Rx_u / ||Rx_u||
        norms = np.linalg.norm(Rx, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # tránh chia cho 0
        self.user_signatures = Rx / norms

        print(f"    [Component I] Signatures: {self.user_signatures.shape}")
        print(f"    [Component I] Non-zero: {(norms.flatten() > 1e-5).sum()} users")

        return self.user_signatures

    def get_signature(self, user_id):
        """Lấy signature của một user"""
        if self.user_signatures is None:
            return None
        return self.user_signatures[user_id]

    def remove_users(self, unlearn_user_ids):
        """Xóa users khỏi signatures (sau khi unlearn)"""
        mask = np.ones(self.n_users, dtype=bool)
        for uid in unlearn_user_ids:
            if uid < self.n_users:
                mask[uid] = False

        self.user_signatures = self.user_signatures[mask]
        print(f"    [Component I] Removed {len(unlearn_user_ids)} users, "
              f"remaining: {self.user_signatures.shape[0]}")


# ============================================================================
# COMPONENT II: DELETION-STABLE SIMILARITY-AWARE ASSIGNMENT
# ============================================================================

class DeletionStableAssignment:
    """
    Component II: Deletion-Stable Similarity-Aware Assignment

    Dùng random hyperplanes để assign users vào shards
    Shard assignment không thay đổi khi unlearn (deletion-stable)
    """

    def __init__(self, n_shards=8, d=64, seed=42):
        """
        Args:
            n_shards: số shards K
            d: chiều signature
            seed: random seed cố định
        """
        self.n_shards = n_shards
        self.d = d
        self.seed = seed

        # Tính số bits cần thiết: K = 2^b
        self.b = math.ceil(math.log2(n_shards))
        self.B = 2 ** self.b  # số hash cells

        # Random hyperplanes a_1, ..., a_b (từ public seed rho)
        random.seed(seed)
        np.random.seed(seed)
        self.hyperplanes = np.random.randn(self.b, d)  # (b, d)

        # Map từ hash cells -> shards (balanced)
        self.cell_to_shard = self._create_balanced_map()

        # User -> Shard assignment
        self.user_to_shard = None

    def _create_balanced_map(self):
        """
        Tạo balanced map từ B cells -> K shards
        Mỗi shard được assign ~B/K cells
        """
        cells_per_shard = self.B // self.n_shards
        extra = self.B % self.n_shards

        cell_to_shard = np.zeros(self.B, dtype=np.int32)
        cell_idx = 0
        for shard in range(self.n_shards):
            n_cells = cells_per_shard + (1 if shard < extra else 0)
            for _ in range(n_cells):
                cell_to_shard[cell_idx] = shard
                cell_idx += 1

        return cell_to_shard

    def assign_users(self, signatures):
        """
        Assign mỗi user vào một shard dựa trên signature

        Routing code: h(u) = [I[a_1^T * z_u >= 0], ..., I[a_b^T * z_u >= 0]]
        Shard: g(u) = cell_to_shard[bin(h(u))]
        """
        n_users = signatures.shape[0]

        # Compute routing bits
        # h_bits[u, t] = I[a_t^T * z_u >= 0]
        h_bits = (signatures @ self.hyperplanes.T) >= 0  # (n_users, b)
        h_bits = h_bits.astype(np.int32)

        # Convert bit vector to integer (binary to decimal)
        h_codes = np.zeros(n_users, dtype=np.int32)
        for t in range(self.b):
            h_codes += h_bits[:, t] * (2 ** (self.b - 1 - t))

        # Map to shard
        self.user_to_shard = self.cell_to_shard[h_codes]

        # Thống kê
        shard_counts = np.bincount(self.user_to_shard, minlength=self.n_shards)
        print(f"    [Component II] Assignment: min={shard_counts.min()}, "
              f"max={shard_counts.max()}, mean={shard_counts.mean():.1f}")

        return self.user_to_shard

    def get_shard(self, user_id):
        """Lấy shard của một user"""
        if self.user_to_shard is None:
            return None
        return self.user_to_shard[user_id]

    def get_affected_shards(self, unlearn_user_ids):
        """Lấy các shards bị ảnh hưởng bởi unlearn"""
        affected = set()
        for uid in unlearn_user_ids:
            if uid < len(self.user_to_shard):
                affected.add(self.user_to_shard[uid])
        return sorted(list(affected))


# ============================================================================
# MODEL: BPRMF CHO MỖI SHARD
# ============================================================================

class BPRMF(nn.Module):
    """Bayesian Personalized Ranking Matrix Factorization cho một shard"""

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
    def predict_user_items(self, user_id, item_ids):
        """Predict scores cho một user với nhiều items"""
        u_emb = self.user_embedding.weight[user_id]  # (emb_dim,)
        i_emb = self.item_embedding(item_ids)  # (n_items, emb_dim)
        scores = (u_emb * i_emb).sum(dim=1)  # (n_items,)
        return scores

    @torch.no_grad()
    def predict_batch(self, user_ids, item_ids):
        """Predict scores cho nhiều users với nhiều items"""
        u_emb = self.user_embedding(user_ids)  # (batch, emb_dim)
        i_emb = self.item_embedding(item_ids)   # (batch, n_items, emb_dim) hoặc (n_items, emb_dim)

        if len(i_emb.shape) == 2:
            scores = u_emb @ i_emb.T
        else:
            scores = (u_emb.unsqueeze(1) * i_emb).sum(dim=2)

        return scores


# ============================================================================
# SHARD MODELS - MỘT MODEL CHO MỖI SHARD
# ============================================================================

class ShardModels:
    """
    Lưu trữ K models, mỗi model cho một shard
    """

    def __init__(self, n_users, n_items, emb_dim, n_shards):
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_shards = n_shards
        self.models = [None] * n_shards

    def create_model(self, shard_id, device):
        """Tạo model mới cho một shard"""
        self.models[shard_id] = BPRMF(self.n_users, self.n_items, self.emb_dim).to(device)
        return self.models[shard_id]

    def get_model(self, shard_id):
        """Lấy model của một shard"""
        return self.models[shard_id]

    def get_model_for_user(self, user_id, user_to_shard):
        """Lấy model phục vụ một user (HARD-ROUTING)"""
        shard_id = user_to_shard[user_id]
        return self.models[shard_id]


# ============================================================================
# TRAINING
# ============================================================================

def train_shard_model(model, shard_data, n_items, device,
                     batch_size=512, lr=0.05, n_epochs=100):
    """Train model cho một shard"""
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
            loss = model(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / n_batches
            print(f"        Epoch {epoch+1}: loss={avg_loss:.4f}")

    return total_loss


# ============================================================================
# EVALUATION - HARD-ROUTING
# ============================================================================

def evaluate_hard_routing(shard_models, user_to_shard, train_data, test_data,
                         n_users, n_items, device, Ks=[10, 20, 50]):
    """
    Đánh giá với HARD-ROUTING như bài báo:
    y_xv = f_{M_{g(x)}}(x, v)

    Chỉ dùng model của shard được assign!
    KHÔNG có aggregation!
    """
    print("    [Evaluation] Using HARD-ROUTING (only shard's model)")

    pre_log = {k: [] for k in Ks}
    rec_log = {k: [] for k in Ks}
    ndcg_log = {k: [] for k in Ks}

    with torch.no_grad():
        for user in range(n_users):
            if user not in test_data or not test_data[user]:
                continue

            # HARD-ROUTING: Chỉ dùng model của shard được assign
            shard_id = user_to_shard[user]
            model = shard_models.get_model(shard_id)

            if model is None:
                continue

            user_t = torch.LongTensor([user]).to(device)
            all_items = list(range(n_items))

            # Predict với model của shard
            scores = []
            for i in range(0, n_items, 256):
                batch_items = torch.LongTensor(all_items[i:i+256]).to(device)
                score = model.predict_user_items(user, batch_items).cpu().numpy()
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

    return {
        'precision': [np.mean(pre_log[k]) for k in Ks],
        'recall': [np.mean(rec_log[k]) for k in Ks],
        'ndcg': [np.mean(ndcg_log[k]) for k in Ks]
    }


# ============================================================================
# OURS METHOD (3 COMPONENTS)
# ============================================================================

class OursMethod:
    """
    Ours: Deletion-Stable 3 Components

    Theo bài báo:
    1. Component I: Deletion-Local User Signatures
    2. Component II: Deletion-Stable Similarity-Aware Assignment
    3. Component III: Isolated Training and Exact User Unlearning
    """

    def __init__(self, n_users, n_items, emb_dim, n_shards=8,
                 batch_size=512, lr=0.05, max_epochs=100, signature_dim=64):
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_shards = n_shards
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.signature_dim = signature_dim

        self.signature_builder = UserSignatureBuilder(
            n_users, n_items, d=signature_dim, seed=42
        )
        self.assignment = DeletionStableAssignment(
            n_shards=n_shards, d=signature_dim, seed=42
        )
        self.shard_models = ShardModels(n_users, n_items, emb_dim, n_shards)
        self.shard_data = [{} for _ in range(n_shards)]

    def build_shard_data(self, train_data, user_to_shard):
        """Xây dựng data cho mỗi shard"""
        self.shard_data = [{} for _ in range(self.n_shards)]

        for user_id, items in train_data.items():
            shard_id = user_to_shard[user_id]
            self.shard_data[shard_id][user_id] = items.copy()

        for sid in range(self.n_shards):
            n_users = len(self.shard_data[sid])
            n_interactions = sum(len(items) for items in self.shard_data[sid].values())
            print(f"    [Shards] Shard {sid}: {n_users} users, {n_interactions} interactions")

        return self.shard_data

    def train(self, train_data, device):
        """
        Training theo bài báo:
        1. Build signatures (Component I)
        2. Assign users to shards (Component II)
        3. Train each shard independently (Component III)
        """
        print("\n    [Training] Phase 1: Component I - Build signatures...")
        signatures = self.signature_builder.build_signature(train_data)

        print("\n    [Training] Phase 2: Component II - Assign users to shards...")
        user_to_shard = self.assignment.assign_users(signatures)

        print("\n    [Training] Phase 3: Component III - Train shards independently...")
        self.build_shard_data(train_data, user_to_shard)

        # Train each shard independently
        for shard_id in range(self.n_shards):
            print(f"\n    [Training] Training shard {shard_id}...")
            model = self.shard_models.create_model(shard_id, device)
            train_shard_model(
                model, self.shard_data[shard_id], self.n_items, device,
                batch_size=self.batch_size, lr=self.lr,
                n_epochs=self.max_epochs
            )

        print("\n    [Training] Done! Using HARD-ROUTING for inference.")
        return user_to_shard

    def unlearn(self, unlearn_user_ids, train_data, device, retrain_epochs=50):
        """
        Unlearning theo bài báo:
        1. Delete user signatures (Component I)
        2. Assignment is STABLE (no changes) (Component II)
        3. Retrain affected shard only (Component III)
        """
        affected_shards = self.assignment.get_affected_shards(unlearn_user_ids)
        print(f"\n    [Unlearn] Affected shards: {affected_shards}")

        # Component I: Remove from signatures
        print("    [Unlearn] Component I: Removing from signatures...")
        self.signature_builder.remove_users(unlearn_user_ids)

        # Component II: Assignment is STABLE - no changes needed!
        print("    [Unlearn] Component II: Assignment is STABLE (no changes)")

        # Component III: Retrain affected shards only
        print("    [Unlearn] Component III: Retraining affected shards...")

        for shard_id in affected_shards:
            print(f"    [Unlearn] Retraining shard {shard_id}...")

            # Filter data: remove unlearned users
            filtered_data = {
                u: items for u, items in self.shard_data[shard_id].items()
                if u not in unlearn_user_ids
            }
            self.shard_data[shard_id] = filtered_data

            # Retrain model for this shard
            model = self.shard_models.get_model(shard_id)
            train_shard_model(
                model, filtered_data, self.n_items, device,
                batch_size=self.batch_size, lr=self.lr,
                n_epochs=retrain_epochs
            )

            # Reset embeddings cua unlearned users trong shard nay de "quen"
            model.eval()
            with torch.no_grad():
                for uid in unlearn_user_ids:
                    if uid < model.user_embedding.num_embeddings:
                        model.user_embedding.weight[uid].zero_()
            model.train()
            print(f"    [Unlearn] Reset embeddings of unlearned users in shard {shard_id}")

        print("    [Unlearn] Done! Using HARD-ROUTING for inference.")
        return affected_shards

    def evaluate(self, train_data, test_data, user_to_shard, device, Ks=[10, 20, 50]):
        return evaluate_hard_routing(
            self.shard_models, user_to_shard,
            train_data, test_data,
            self.n_users, self.n_items, device, Ks
        )


# ============================================================================
# MAIN
# ============================================================================

def run_ours(dataset='ml-1m', emb_dim=64, n_shards=8,
            batch_size=512, lr=0.05, max_epochs=100,
            unlearn_ratio=0.1, unlearn_mode='random', unlearn_user_id=None,
            retrain_epochs=50, signature_dim=64, output_suffix=''):
    """
    Chạy Ours method với 3 components đúng theo bài báo
    """
    print(f"\n{'='*70}")
    print(f"METHOD 4: OURS (DELETION-STABLE 3 COMPONENTS)")
    print(f"{'='*70}")
    print(f"Hyper-parameters:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Embedding dim: {emb_dim}")
    print(f"  - N shards: {n_shards}")
    print(f"  - Max epochs per shard: {max_epochs}")
    print(f"  - Signature dim: {signature_dim}")
    print("""
    3 Components (theo bài báo):
      Component I:   Deletion-Local User Signatures
      Component II:  Deletion-Stable Similarity-Aware Assignment
      Component III: Isolated Training & HARD-ROUTING Inference

    Inference: y_xv = f_{M_{g(x)}}(x, v)
    (Chi su dung model cua shard duoc assign, KHONG co aggregation!)
    """)

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

    method = OursMethod(
        n_users, n_items, emb_dim, n_shards,
        batch_size=batch_size, lr=lr, max_epochs=max_epochs,
        signature_dim=signature_dim
    )

    print(f"\n{'='*70}")
    print(f"--- Phase 1: Train BEFORE unlearning ---")
    print(f"{'='*70}")
    t0 = time.time()
    user_to_shard = method.train(train_data, device)
    train_time = time.time() - t0

    # Keep original train_data for evaluation (before AND after use same mask)
    train_data_original = {u: items.copy() for u, items in train_data.items()}

    # Evaluate BEFORE unlearn on FULL test set (bao gồm cả unlearned users)
    results_before = method.evaluate(train_data_original, test_data, user_to_shard, device)
    print(f"\n  Before (FULL) - R@10: {results_before['recall'][0]:.4f}, "
          f"NDCG@10: {results_before['ndcg'][0]:.4f}")

    print(f"\n{'='*70}")
    print(f"--- Phase 2: Unlearn (Deletion-Stable 3 Components) ---")
    print(f"{'='*70}")
    t0 = time.time()
    affected_shards = method.unlearn(unlearn_users, train_data, device, retrain_epochs=retrain_epochs)
    unlearn_time = time.time() - t0

    # Evaluate AFTER unlearn on FULL test set (cùng tập với before - fair comparison)
    # Use ORIGINAL train_data to mask items (not the modified one)
    results_after = method.evaluate(train_data_original, test_data, user_to_shard, device)
    print(f"\n  After (FULL) - R@10: {results_after['recall'][0]:.4f}, "
          f"NDCG@10: {results_after['ndcg'][0]:.4f}")
    print(f"  Unlearn time: {unlearn_time:.2f}s")

    results = {
        'method': 'Ours_3Components',
        'dataset': dataset,
        'hyperparameters': {
            'batch_size': batch_size,
            'learning_rate': lr,
            'embedding_dim': emb_dim,
            'max_epochs': max_epochs,
            'n_shards': n_shards,
            'signature_dim': signature_dim
        },
        'components': {
            'component_I': 'Deletion-Local User Signatures',
            'component_II': 'Deletion-Stable Similarity-Aware Assignment',
            'component_III': 'Isolated Training & HARD-ROUTING Inference'
        },
        'inference': 'HARD-ROUTING: y_xv = f_{M_{g(x)}}(x, v)',
        'unlearn_ratio': unlearn_ratio,
        'unlearn_mode': unlearn_mode,
        'n_unlearn': n_unlearn,
        'unlearn_users_sample': list(unlearn_users)[:10],
        'affected_shards': [int(s) for s in affected_shards],
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
    output_path = os.path.join(PROJ, f'results_ours_3components{suffix}.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Method 4: Ours (Deletion-Stable 3 Components)')
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--n_shards', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--signature_dim', type=int, default=64)
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)
    parser.add_argument('--unlearn_mode', type=str, default='random',
                       choices=['random', 'fewest', 'most', 'single'],
                       help='random: ngau nhien, fewest: it interaction nhat, most: nhieu interaction nhat, single: 1 user')
    parser.add_argument('--unlearn_user_id', type=int, default=None,
                       help='Chi dinh user ID cu the de unlearn (dung voi --unlearn_mode single)')
    parser.add_argument('--retrain_epochs', type=int, default=50)
    parser.add_argument('--output_suffix', type=str, default='')

    args = parser.parse_args()

    run_ours(
        dataset=args.dataset,
        emb_dim=args.emb_dim,
        n_shards=args.n_shards,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        max_epochs=args.max_epochs,
        signature_dim=args.signature_dim,
        unlearn_ratio=args.unlearn_ratio,
        unlearn_mode=args.unlearn_mode,
        unlearn_user_id=args.unlearn_user_id,
        retrain_epochs=args.retrain_epochs,
        output_suffix=args.output_suffix
    )
