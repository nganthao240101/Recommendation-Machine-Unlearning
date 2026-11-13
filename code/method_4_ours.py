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
import fast_train
import torch.nn.functional as F

from scipy.sparse import csr_matrix
import heapq

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

from metrics_fmt import fmt_metrics as _fmt


# ============================================================================
# CHECKPOINT HELPERS
# ============================================================================

def save_models(models, path):
    """Save list of models to file"""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    torch.save([m.state_dict() for m in models], path)


def load_models(models, path, device):
    """Load list of models from file"""
    state_dicts = torch.load(path, map_location=device)
    for i, state_dict in enumerate(state_dicts):
        models[i].load_state_dict(state_dict)


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
    """Train model cho một shard.

    Returns (final_loss, elapsed_seconds, n_samples) so a caller can
    report per-shard cost and check that the sum of the per-shard times
    equals the reported unlearn_time (spec S5: a shard with the same
    wall time as the global retrain would mean the loop is iterating
    over the whole dataset).
    """
    t0 = time.time()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    if fast_train.use_gpu_engine():
        samples = fast_train.draw_samples(shard_data, n_items)
        if not samples[0]:
            return 0.0, time.time() - t0, 0

        def step(u, p, n):
            optimizer.zero_grad()
            loss = model(u, p, n)
            loss.backward()
            optimizer.step()
            return loss
        total, _ = fast_train.train_epochs(step, samples, device, batch_size,
                                           n_epochs, log_every=20)
        fast_train.sync(device)
        return total, time.time() - t0, len(samples[0])

    samples = []
    for user, items in shard_data.items():
        for pos_item in items:
            neg_item = random.randint(0, n_items - 1)
            while neg_item in items:
                neg_item = random.randint(0, n_items - 1)
            samples.append((user, pos_item, neg_item))

    if not samples:
        return 0.0, time.time() - t0, 0

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

    return total_loss, time.time() - t0, len(samples)


# ============================================================================
# EVALUATION - MEAN AGGREGATION
# ============================================================================

def evaluate_mean_aggregation(shard_models, train_data, test_data,
                             n_users, n_items, device, Ks=[10, 20, 50]):
    """
    Đánh giá với MEAN AGGREGATION:
    y_xv = (1/K) * sum(f_{M_i}(x, v))
    """
    print("    [Evaluation] Using MEAN AGGREGATION (all shards)")

    pre_log = {k: [] for k in Ks}
    rec_log = {k: [] for k in Ks}
    ndcg_log = {k: [] for k in Ks}

    n_shards = len(shard_models.models)

    with torch.no_grad():
        for user in range(n_users):
            if user not in test_data or not test_data[user]:
                continue

            user_t = torch.LongTensor([user]).to(device)
            all_items = list(range(n_items))

            scores_sum = np.zeros(n_items)
            valid_shards = 0
            for shard_id in range(n_shards):
                model = shard_models.get_model(shard_id)
                if model is None:
                    continue
                valid_shards += 1
                scores = []
                for i in range(0, n_items, 256):
                    batch_items = torch.LongTensor(all_items[i:i+256]).to(device)
                    s = model.predict_user_items(user, batch_items).cpu().numpy()
                    scores.extend(s.tolist())
                scores_sum += np.array(scores)

            # Avoid division by zero
            if valid_shards > 0:
                scores = scores_sum / valid_shards
            else:
                scores = scores_sum

            train_items = set(train_data.get(user, []))
            for item in train_items:
                scores[item] = -np.inf

            rank_list = heapq.nlargest(max(Ks), range(len(scores)), key=scores.__getitem__)
            item_pos = test_data.get(user, [])
            item_set = set(item_pos)

            hit_num = len(set(rank_list) & item_set)
            rec = hit_num / len(item_pos) if len(item_pos) > 0 else 0
            dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(rank_list) if item in item_set)
            idcg = sum(1.0 / np.log2(i + 2.0) for i in range(len(item_pos)))
            ndcg = dcg / idcg if idcg > 0 else 0

            for k in Ks:
                rank_k = heapq.nlargest(k, range(len(scores)), key=scores.__getitem__)
                hit_num_k = len(set(rank_k) & item_set)
                rec_k = hit_num_k / len(item_pos) if len(item_pos) > 0 else 0
                pre_k = hit_num_k / k
                rec_log[k].append(rec_k)
                pre_log[k].append(pre_k)

            rec_log[max(Ks)].append(rec)
            ndcg_log[max(Ks)].append(ndcg)

    return {
        'precision': [np.nanmean(pre_log[k]) if pre_log[k] else 0 for k in Ks],
        'recall': [np.nanmean(rec_log[k]) if rec_log[k] else 0 for k in Ks],
        'ndcg': [np.nanmean(ndcg_log[k]) if ndcg_log[k] else 0 for k in Ks]
    }


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

    if fast_train.use_gpu_engine():
        u2s = torch.as_tensor(np.asarray(user_to_shard), dtype=torch.long, device=device)

        def score_block(ut):
            sc = torch.empty((len(ut), n_items), device=device)
            sh = u2s[ut]
            for s in torch.unique(sh).tolist():
                rows = (sh == s).nonzero(as_tuple=True)[0]
                m = shard_models.get_model(s)
                sc[rows] = m.user_embedding.weight[ut[rows]] @ m.item_embedding.weight.T
            return sc
        users = [u for u in range(n_users) if test_data.get(u)
                 and shard_models.get_model(int(user_to_shard[u])) is not None]
        return fast_train.evaluate_topk(score_block, users, train_data, test_data,
                                        n_items, device, Ks, per_list=True)

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

                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(item_pos), k)))
                ndcg = dcg / idcg if idcg > 0 else 0

                pre_log[k].append(pre)
                rec_log[k].append(rec)
                ndcg_log[k].append(ndcg)

    return {
        'precision': [np.nanmean(pre_log[k]) if pre_log[k] else 0 for k in Ks],
        'recall': [np.nanmean(rec_log[k]) if rec_log[k] else 0 for k in Ks],
        'ndcg': [np.nanmean(ndcg_log[k]) if ndcg_log[k] else 0 for k in Ks]
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

    def train(self, train_data, device, user_to_shard=None):
        """
        Training theo bài báo:
        1. Build signatures (Component I)
        2. Assign users to shards (Component II)
        3. Train each shard independently (Component III)

        user_to_shard (tuỳ chọn): FROZEN partition (vd InP-kept). Nếu có, BỎ qua
        Component I/II (signature) và dùng mapping này trực tiếp -> Ours unlearn
        stable 0% nhưng partition = InP (utility cao hơn signature).
        """
        if user_to_shard is not None:
            print("\n    [Training] FROZEN partition mode (Component I/II skipped).")
            print(f"    [Training] user_to_shard: min=0 max={int(np.max(user_to_shard))}, "
                  f"len={len(user_to_shard)}")
            # Lưu vào assignment để get_affected_shards() hoạt động lúc unlearn
            self.assignment.user_to_shard = user_to_shard
        else:
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
        # (skip if frozen partition mode - no signatures built, assignment is fixed)
        if self.signature_builder.user_signatures is not None:
            print("    [Unlearn] Component I: Removing from signatures...")
            self.signature_builder.remove_users(unlearn_user_ids)
        else:
            print("    [Unlearn] Component I: SKIPPED (frozen partition, no signatures)")

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

            # Retrain model for this shard (from scratch - theo bài báo)
            # Bài báo: "retrain only Mj from scratch using Sj−"
            model = self.shard_models.get_model(shard_id)
            train_shard_model(
                model, filtered_data, self.n_items, device,
                batch_size=self.batch_size, lr=self.lr,
                n_epochs=retrain_epochs
            )

            # KHÔNG reset embeddings - giống bài báo gốc
            # Model vẫn giữ embeddings nhưng đánh giá chỉ trên retained users

        print("    [Unlearn] Done! Using HARD-ROUTING for inference.")
        return affected_shards

    def evaluate(self, train_data, test_data, user_to_shard, device, Ks=[10, 20, 50], use_mean=False):
        if use_mean:
            return evaluate_mean_aggregation(
                self.shard_models,
                train_data, test_data,
                self.n_users, self.n_items, device, Ks
            )
        return evaluate_hard_routing(
            self.shard_models, user_to_shard,
            train_data, test_data,
            self.n_users, self.n_items, device, Ks
        )


# ============================================================================
# MAIN
# ============================================================================

def train_bpr_full_embeddings(n_users, n_items, emb_dim, train_data, device,
                              batch_size=512, lr=0.05, n_epochs=100):
    """Train BPRMF (BPR + L2 reg 0.01 + clamp±50 - model trong method_1) trên FULL data,
    trả (user_emb, item_emb) ndarray (n x D) - nguồn embedding LƯA CHỌN cho partition
    (so với WMF-npz: khác = L2 reg 0.01 + clamp, còn lại cùng BPR, cùng full data)."""
    model = BPRMF(n_users, n_items, emb_dim).to(device)
    print(f"  [embed=bprfull] training FULL-data BPRMF: emb_dim={emb_dim}, epochs={n_epochs}")
    train_shard_model(model, train_data, n_items, device,
                      batch_size=batch_size, lr=lr, n_epochs=n_epochs)
    user_emb = model.user_embedding.weight.data.cpu().numpy()
    item_emb = model.item_embedding.weight.data.cpu().numpy()
    print(f"  [embed=bprfull] embeddings: user={user_emb.shape}, item={item_emb.shape}")
    return user_emb, item_emb


def build_frozen_partition(dataset, emb_dim, n_shards, train_data, n_users,
                           partition_type=1, max_epochs_wmf=100, batch_size=512, lr=0.05,
                           n_items=0, embed_source='wmf', persist=False):
    """Build Rec's partition (InP=1 user+item WMF k-means / UBP=2 user-only k-means)
    ONCE on full data, return FROZEN mapping (Ours unlearn KHÔNG re-cluster -> stable 0%).

    Dùng lại DataPartitioner + load_or_train_wmf của method_3.
    Result = user_to_shard (int32, len n_users).

    persist=True: lưu mapping ra data/<dataset>/ours_partition_<type>_<src>_k<K>.npz
    để dùng lại cho các lần xóa tiếp theo (multi-request / restart process).
    Mặc định False = in-memory (đủ cho benchmark 1 lần xóa).
    """
    from method_3_receraser_v2 import DataPartitioner, load_or_train_wmf
    if partition_type == 3:
        # Random/modulo: KHÔNG cần embedding (partition thuần user_id % n_shards)
        user_emb, item_emb = None, None
    elif embed_source == 'bprfull':
        # Train BPRMF (BPR + L2 reg 0.01 + clamp) trên FULL data -> embedding BPR
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        user_emb, item_emb = train_bpr_full_embeddings(
            n_users, n_items, emb_dim, train_data, device,
            batch_size=batch_size, lr=lr, n_epochs=max_epochs_wmf)
    else:
        # Mặc định: WMF-npz (load, hoặc train nếu chưa có)
        user_emb, item_emb = load_or_train_wmf(dataset, emb_dim, max_epochs_wmf, batch_size, lr)
    part = DataPartitioner(n_shards=n_shards, partition_type=partition_type, seed=42)
    user_to_shard = part.partition_users(train_data, n_users, user_emb, item_emb)
    counts = np.bincount(user_to_shard, minlength=n_shards)
    name = {1: 'InP', 2: 'UBP', 3: 'Random'}.get(partition_type, f'type{partition_type}')
    print(f"  [frozen-{name}] partition built: {n_shards} shards, sizes min/max = "
          f"{int(counts.min())}/{int(counts.max())}")
    if persist:
        p = _partition_cache_path(dataset, partition_type, embed_source, n_shards)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.savez_compressed(p, user_to_shard=user_to_shard,
                            partition_type=partition_type, embed_source=embed_source,
                            n_shards=n_shards, n_users=n_users)
        print(f"  [frozen-{name}] partition SAVED: {p}")
    return user_to_shard


def _partition_cache_path(dataset, partition_type, embed_source, n_shards):
    """Đường dẫn file lưu FROZEN partition của Ours (dùng khi --persist_partition).

    Tên file KHÔNG chứa unlearn_ratio/unlearn_mode - ĐÚNG, vì partition luôn
    được build trên FULL data (trước khi xóa) và giữ nguyên qua mọi lần xóa.
    Nhờ vậy sweep ratio/mode dùng CHUNG đúng 1 partition -> phép so sánh
    cô lập đúng yếu tố "ratio" (không lẫn yếu tố "partition").

    Có 'v1' trong tên để bump version khi đổi logic partition (tránh dùng
    nhầm cache cũ).
    """
    base_dir = os.path.dirname(PROJ)
    name = {1: 'inp', 2: 'ubp', 3: 'random'}.get(partition_type, f'type{partition_type}')
    return os.path.join(base_dir, 'data', dataset,
                        f'ours_partition_v1_{name}_{embed_source}_k{n_shards}.npz')


# Giữ tên cũ (InP) cho tương thích
def build_frozen_inp_partition(dataset, emb_dim, n_shards, train_data, n_users,
                                max_epochs_wmf=100, batch_size=512, lr=0.05):
    return build_frozen_partition(dataset, emb_dim, n_shards, train_data, n_users,
                                  partition_type=1, max_epochs_wmf=max_epochs_wmf,
                                  batch_size=batch_size, lr=lr)


def run_ours(dataset='ml-1m', emb_dim=64, n_shards=8,
            batch_size=512, lr=0.05, max_epochs=100,
            unlearn_ratio=0.1, unlearn_mode='random', unlearn_user_id=None,
            retrain_epochs=50, signature_dim=64, use_mean=False, output_suffix='',
            partition='signature', max_epochs_wmf=100, embed_source='wmf',
            persist_partition=False, seed=42):
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
    print(f"  - Use mean aggregation: {use_mean}")
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

    import torch as _torch
    random.seed(seed)
    np.random.seed(seed)
    _torch.manual_seed(seed)
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

    # Checkpoint path
    checkpoint_dir = 'checkpoints'
    checkpoint_path = os.path.join(checkpoint_dir, f'ours_{dataset}_d{emb_dim}_k{n_shards}_sig{signature_dim}.pt')

    print(f"\n{'='*70}")
    print(f"--- Phase 1: Train BEFORE unlearning ---")
    print(f"{'='*70}")
    t0 = time.time()

    # Check if checkpoint exists (disabled due to corruption issues)
    if False and os.path.exists(checkpoint_path):
        print(f"  Loading models from checkpoint: {checkpoint_path}")
        # Need to rebuild partitioner and models
        method.signature_builder.build_signature(train_data)
        method.assignment.assign_users(method.signature_builder.user_signatures)
        method.build_shard_data(train_data, method.assignment.user_to_shard)
        # Create models first
        method.shard_models.models = []
        for shard_id in range(n_shards):
            model = BPRMF(n_users, n_items, emb_dim).to(device)
            method.shard_models.models.append(model)
        load_models(method.shard_models.models, checkpoint_path, device)
        user_to_shard = method.assignment.user_to_shard
    else:
        _FROZEN = {'inpfrozen': 1, 'ubpfrozen': 2, 'randomfrozen': 3}
        if partition in _FROZEN:
            _pname = {1: 'InP', 2: 'UBP', 3: 'Random'}[_FROZEN[partition]]
            _kind = '(modulo, no WMF)' if _FROZEN[partition] == 3 else '(WMF k-means)'
            _cache_p = _partition_cache_path(dataset, _FROZEN[partition], embed_source, n_shards)
            _loaded = False
            if persist_partition and os.path.exists(_cache_p):
                try:
                    _z = np.load(_cache_p)
                    _cached = _z['user_to_shard']
                    if len(_cached) == n_users:
                        frozen = _cached.astype(np.int32)
                        print(f"\n  [frozen-{_pname}] LOADED saved partition: {_cache_p}")
                        _loaded = True
                    else:
                        print(f"  [frozen-{_pname}] cache mismatch (len {len(_cached)} != {n_users}) -> rebuild")
                except Exception as e:
                    print(f"  [frozen-{_pname}] cache unreadable ({e}) -> rebuild")
            if not _loaded:
                print(f"\n  [frozen-{_pname}] building FROZEN {_pname} {_kind} partition "
                      f"(Ours unlearn sẽ KHÔNG re-cluster -> stable 0% + utility {_pname})")
                frozen = build_frozen_partition(
                    dataset, emb_dim, n_shards, train_data, n_users,
                    partition_type=_FROZEN[partition],
                    max_epochs_wmf=max_epochs_wmf, batch_size=batch_size, lr=lr,
                    n_items=n_items, embed_source=embed_source,
                    persist=persist_partition)
            user_to_shard = method.train(train_data, device, user_to_shard=frozen)
        else:
            user_to_shard = method.train(train_data, device)
        # Skip saving checkpoint due to disk space
        # save_models(method.shard_models.models, checkpoint_path)
        print(f"  Skipped saving checkpoint due to disk space")

    train_time = time.time() - t0
    print(f"  Train time (before unlearn): {train_time:.2f}s")

    # Filter out unlearned users from test set
    test_data_retained = {u: items for u, items in test_data.items() if u not in unlearn_users}

    # BEFORE: Evaluate on FULL test set (all users)
    results_before = method.evaluate(train_data, test_data, user_to_shard, device, use_mean=use_mean)
    print("\n  Before (FULL) - " + _fmt(results_before))

    print(f"\n{'='*70}")
    print(f"--- Phase 2: Unlearn (Deletion-Stable 3 Components) ---")
    print(f"{'='*70}")
    t0 = time.time()
    affected_shards = method.unlearn(unlearn_users, train_data, device, retrain_epochs=retrain_epochs)
    unlearn_time = time.time() - t0

    # Save model after unlearn for next time
    # Skip saving checkpoint due to disk space
    # save_models(method.shard_models.models, checkpoint_path)

    # Evaluate AFTER unlearn on RETAINED users only (cùng tập với before - fair comparison)
    # AFTER: Evaluate on RETAINED test set (train on RETAINED data)
    train_data_retained = {u: items for u, items in train_data.items() if u not in unlearn_users}
    results_after = method.evaluate(train_data_retained, test_data_retained, user_to_shard, device, use_mean=use_mean)
    print("\n  After (RETAINED, retrain) - " + _fmt(results_after))
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
    parser.add_argument('--use_mean', type=int, default=0, help='1: use mean aggregation, 0: use hard-routing')
    parser.add_argument('--partition', type=str, default='signature',
                       choices=['signature', 'inpfrozen', 'ubpfrozen', 'randomfrozen'],
                       help='signature: Ours base (hash). inpfrozen: Rec InP (user+item WMF k-means) FROZEN. '
                            'ubpfrozen: Rec UBP (user-only WMF k-means) FROZEN. '
                            'randomfrozen: Random/modulo (no WMF) FROZEN. '
                            'All frozen -> Ours 0% deletion-stable + correct scope + that partition\'s utility')
    parser.add_argument('--max_epochs_wmf', type=int, default=100, help='WMF/BPR-embed epochs (dùng khi --partition inpfrozen/ubpfrozen; epochs để train BPRMF nếu --embed_source bprfull)')
    parser.add_argument('--embed_source', type=str, default='wmf', choices=['wmf', 'bprfull'],
                       help='Nguồn embedding cho partition InP/UBP: wmf (load wmf_embeddings.npz - BPR-trọng-1, không reg) '
                            'hoặc bprfull (train BPRMF + L2 reg 0.01 + clamp trên FULL data - model method_1). '
                            'Chỉ dùng khi --partition inpfrozen/ubpfrozen.')
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)
    parser.add_argument('--unlearn_mode', type=str, default='random',
                       choices=['random', 'fewest', 'most', 'single'],
                       help='random: ngau nhien, fewest: it interaction nhat, most: nhieu interaction nhat, single: 1 user')
    parser.add_argument('--unlearn_user_id', type=int, default=None,
                       help='Chi dinh user ID cu the de unlearn (dung voi --unlearn_mode single)')
    parser.add_argument('--retrain_epochs', type=int, default=50)
    parser.add_argument('--persist_partition', type=int, default=0, choices=[0, 1],
                        help='1 = lưu FROZEN partition ra data/<dataset>/ours_partition_*.npz và '
                             'DÙNG LẠI nếu đã có (cho multi-request / restart process). '
                             '0 = in-memory (đủ cho benchmark 1 lần xóa).')
    parser.add_argument('--seed', type=int, default=42,
                    help='seed cho model-init (partition LSH vẫn dùng seed=42 công khai, frozen)')
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
        use_mean=bool(args.use_mean),
        unlearn_ratio=args.unlearn_ratio,
        unlearn_mode=args.unlearn_mode,
        unlearn_user_id=args.unlearn_user_id,
        retrain_epochs=args.retrain_epochs,
        output_suffix=args.output_suffix,
        partition=args.partition,
        max_epochs_wmf=args.max_epochs_wmf,
        embed_source=args.embed_source,
        persist_partition=bool(args.persist_partition),
        seed=args.seed
    )
