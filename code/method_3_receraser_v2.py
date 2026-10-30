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
import pickle
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


def data_dir_for(dataset='ml-1m'):
    """Trở về đường dẫn data/<dataset> (dùng cho cache C/C_U/C_I + wmf_embeddings)."""
    data_path = os.environ.get('RECUNLEARN_DATA_PATH', None)
    if data_path:
        dataset_name = os.environ.get('RECUNLEARN_DATASET', dataset)
        return os.path.join(data_path, dataset_name)
    return os.path.join(os.path.dirname(PROJ), 'data', dataset)


def load_data(dataset='ml-1m'):
    return SimpleDataLoader(data_dir_for(dataset))


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
    def __init__(self, n_shards=8, partition_type=1, seed=42, T=10, cache_path=None):
        """
        partition_type:
          1: InP (Interaction-based Partition, RecEraser paper — user+item emb, balanced)
          2: UBP (User-based Partition, RecEraser paper — user-only emb, balanced)
          3: Random (modulo, always balanced)
        T: số iteration k-means (InP và UBP; không dùng cho Random)
        """
        self.n_shards = n_shards
        self.partition_type = partition_type
        self.seed = seed
        self.T = T
        self.user_to_shard = None
        self.shard_data = None

    def partition_users(self, train_data, n_users, user_emb=None, item_emb=None):
        """Chia users vào các shards (RecEraser paper: UBP/InP balanced + Random modulo)"""
        random.seed(self.seed)
        np.random.seed(self.seed)

        if self.partition_type == 3:
            return self._partition_random(n_users)
        elif self.partition_type == 2:
            return self._partition_ubp_paper(train_data, n_users, user_emb)
        elif self.partition_type == 1:
            return self._partition_inp_paper(train_data, n_users, user_emb, item_emb)
        else:
            raise ValueError(f"Unknown partition_type: {self.partition_type}")

    # ------------------------------------------------------------------
    # CACHE (giữ giống bộ utility cũ: C / C_U / C_I .pk, định danh theo type + k)
    # ------------------------------------------------------------------
    @staticmethod
    def _cache_path(data_dir, partition_type, n_shards, which):
        """which in {'','U','I'} -> cùng naming bộ utility cũ (vd: C_U_type-1_num-8.pk):
           ''->C_type-1_num-8.pk, U->C_U_type-1_num-8.pk, I->C_I_type-1_num-8.pk"""
        suffix = f'C_{which}' if which else 'C'
        return os.path.join(data_dir, f'{suffix}_type-{partition_type}_num-{n_shards}.pk')

    def save_cache(self, data_dir, train_data):
        """Lưu C / C_U / C_I như utility gốc (pickle, cùng định danh)."""
        C, users, items = self._shard_cui(train_data)
        for which, obj in [('', C), ('U', users), ('I', items)]:
            with open(self._cache_path(data_dir, self.partition_type, self.n_shards, which), 'wb') as f:
                pickle.dump(obj, f)
        print(f"    [Cache] saved C/C_U/C_I type-{self.partition_type} num-{self.n_shards} -> {data_dir}")

    def load_cache(self, data_dir, train_data):
        """Đọc cache. True nếu cache hợp lệ (đúng số user), không thì False."""
        try:
            with open(self._cache_path(data_dir, self.partition_type, self.n_shards, 'U'), 'rb') as f:
                users = pickle.load(f)
            with open(self._cache_path(data_dir, self.partition_type, self.n_shards, ''), 'rb') as f:
                C = pickle.load(f)
            # tái dựng user_to_shard từ C (để không phụ thuộc định dạng C cụ thể)
            self.user_to_shard = np.zeros(max(train_data.keys()) + 1 if train_data else 0, dtype=np.int32)
            for k, shard in enumerate(C):
                for u in (shard if isinstance(shard, dict) else []):
                    if u in train_data and u < len(self.user_to_shard):
                        self.user_to_shard[u] = k
            print(f"    [Cache] loaded partition type-{self.partition_type} num-{self.n_shards} from {data_dir}")
            return True
        except Exception:
            return False

    def _shard_cui(self, train_data):
        """Trả về (C, users, items) cùng định dạng utility gốc:
          C: list[shard] của dict {user: [items]}
          users: list[shard] của list user ids
          items: list[shard] của list item ids (dùnh)
        """
        C = [{} for _ in range(self.n_shards)]
        for u, items in train_data.items():
            k = int(self.user_to_shard[u])
            C[k][u] = list(items)
        users = [list(shard.keys()) for shard in C]
        items = []
        for shard in C:
            s = set()
            for it in shard.values():
                s.update(it)
            items.append(list(s))
        return C, users, items

    def _partition_random(self, n_users):
        """Random partition (modulo, luôn cân bằng)"""
        self.user_to_shard = np.zeros(n_users, dtype=np.int32)
        for uid in range(n_users):
            self.user_to_shard[uid] = uid % self.n_shards
        print(f"    [Random] {self.n_shards} shards (modulo, balanced)")
        return self.user_to_shard

    @staticmethod
    def _e_score2(a, b):
        """E_score2 = squared L2 (độ "sai" giữa 2 vector embedding), như code gốc."""
        return float(np.sum(np.power(np.asarray(a) - np.asarray(b), 2)))

    def _partition_ubp_paper(self, train_data, n_users, user_emb):
        """UBP - copy 1:1 `data_partition_2` (RecEraser, code/utility/data_partition.py).

        Khác duy nhất so với code tác giả:
          - `uidW[i]` (tác giả: dict {id: vector} đọc từ .pk) -> ở đây WMF là ndarray
            (N x D), nên `uidW[i]` = 1 hàng của matrix. Cùng nghĩa, chỉ khác kiểu container.
          - guard: shard rỗng -> giữ centroid cũ (code tác giả `np.mean([])` = NaN -> crash).
        Còn lại y hệt: `C`/`C_num`/`fl`, `Scores` global-sorted, `max_data = 1.2*N/k`.
        """
        print(f"    [UBP] {self.n_shards} shards (user embeddings, T={self.T})")
        uidW = np.asarray(user_emb)              # ndarray N x D (thay cho dict {id: emb})
        k = self.n_shards
        T = self.T

        # 'data = train_items'  (giữ user nằm trong range embedding để re-partition on RETAINED)
        data = {u: items for u, items in train_data.items() if u < len(uidW)}
        if not data:
            self.user_to_shard = np.zeros(n_users, dtype=np.int32)
            return self.user_to_shard

        # Randomly select k centroids
        max_data = 1.2 * len(data) / k
        centroids = random.sample(list(data.keys()), min(k, len(data)))

        # centro emb
        centroembs = []
        for i in range(len(centroids)):
            temp_u = uidW[centroids[i]]
            centroembs.append(temp_u)

        for _ in range(T):
            C = [{} for i in range(k)]
            C_num = [0 for i in range(k)]
            Scores = {}
            for i in list(data.keys()):
                for j in range(len(centroembs)):
                    score_u = self._e_score2(uidW[i], centroembs[j])
                    Scores[i, j] = -score_u

            Scores = sorted(Scores.items(), key=lambda x: x[1], reverse=True)

            fl = set()
            for i in range(len(Scores)):
                if Scores[i][0][0] not in fl:
                    if C_num[Scores[i][0][1]] < max_data:
                        C[Scores[i][0][1]][Scores[i][0][0]] = data[Scores[i][0][0]]
                        fl.add(Scores[i][0][0])
                        C_num[Scores[i][0][1]] += 1

            centroembs_next = []
            for i in range(len(centroembs)):
                temp_u = []
                for u in list(C[i].keys()):
                    temp_u.append(uidW[u])
                if len(temp_u) == 0:
                    centroembs_next.append(centroembs[i])      # guard shard rỗng
                else:
                    centroembs_next.append(np.mean(temp_u))
            centroembs = centroembs_next

        # tái dựng user_to_shard từ C
        assign = np.zeros(n_users, dtype=np.int32)
        for j in range(k):
            for u in C[j].keys():
                assign[u] = j
        self.user_to_shard = assign.astype(np.int32)
        counts = np.bincount(self.user_to_shard, minlength=k)
        print(f"    [UBP] shard sizes min/max = {counts.min()}/{counts.max()}")
        return self.user_to_shard

    def _partition_inp_paper(self, train_data, n_users, user_emb, item_emb):
        """InP - copy 1:1 `data_partition_1` (RecEraser, code/utility/data_partition.py).

        Mỗi INTERACTION (u,i) được gán shard theo -E_score2(u)*E_score2(i);
        user thuộc shard của interaction đặt nó (fl set -> mỗi user 1 lần).
        cap `max_data = 1.2*N/k`. Khác duy nhất: uidW/iidW dict (tác giả) -> ndarray (ở đây).
        """
        print(f"    [InP] {self.n_shards} shards (user+item emb, T={self.T})")
        uidW = np.asarray(user_emb)
        iidW = np.asarray(item_emb)
        k = self.n_shards
        T = self.T

        # 'data = [[i, j] for i in train_items for j in train_items[i]]'
        data = []
        for i in train_data:
            for j in train_data[i]:
                data.append([i, j])
        if not data:
            return self._partition_random(n_users)

        # Randomly select k centroids
        max_data = 1.2 * len(data) / k
        centroids = random.sample(data, min(k, len(data)))

        # centro emb
        centroembs = []
        for i in range(len(centroids)):
            temp_u = uidW[centroids[i][0]]
            temp_i = iidW[centroids[i][1]]
            centroembs.append([temp_u, temp_i])

        for _ in range(T):
            C = [{} for i in range(k)]
            C_num = [0 for i in range(k)]
            Scores = {}
            for i in range(len(data)):
                for j in range(len(centroembs)):
                    score_u = self._e_score2(uidW[data[i][0]], centroembs[j][0])
                    score_i = self._e_score2(iidW[data[i][1]], centroembs[j][1])
                    Scores[i, j] = -score_u * score_i

            Scores = sorted(Scores.items(), key=lambda x: x[1], reverse=True)

            fl = set()
            for i in range(len(Scores)):
                if data[Scores[i][0][0]][0] not in fl:
                    if C_num[Scores[i][0][1]] < max_data:
                        if data[Scores[i][0][0]][0] not in C[Scores[i][0][1]]:
                            C[Scores[i][0][1]][data[Scores[i][0][0]][0]] = [data[Scores[i][0][1]][1]]
                        else:
                            C[Scores[i][0][1]][data[Scores[i][0][0]][0]].append(data[Scores[i][0][1]][1])
                        fl.add(Scores[i][0][0])
                        C_num[Scores[i][0][1]] += 1

            centroembs_next = []
            for i in range(len(centroembs)):
                temp_u, temp_i = [], []
                for j in list(C[i].keys()):
                    for l in C[i][j]:
                        temp_u.append(uidW[j])
                        temp_i.append(iidW[l])
                if len(temp_u) == 0:
                    centroembs_next.append(centroembs[i])      # guard shard rỗng
                else:
                    centroembs_next.append([np.mean(temp_u), np.mean(temp_i)])
            centroembs = centroembs_next

        # tái dựng user_to_shard từ C
        assign = np.zeros(n_users, dtype=np.int32)
        for j in range(k):
            for u in C[j].keys():
                assign[u] = j
        self.user_to_shard = assign.astype(np.int32)
        counts = np.bincount(self.user_to_shard, minlength=k)
        print(f"    [InP] shard sizes min/max = {counts.min()}/{counts.max()}")
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


def train_aggregator(model, train_data, n_users, n_items, device, batch_size=512, lr=0.005, max_epochs=50):
    """Train attention aggregator (Phase 2).

    Chống NaN (đặc biệt trên dataset to/sparse như yelp):
      - Adagrad `initial_accumulator_value=1.0` (chuẩn) thay vì 1e-8
        (1e-8 -> update đầu cực lớn -> embedding "bay" -> reg/loss NaN).
      - lr nhỏ (0.005), clip gradient, và SKIP batch NaN (không để NaN lan
        sang optimizer / parameters).
      - Giữ nguyên cơ chế ATTENTION (không phải mean).
    """
    # initial_accumulator_value=1.0: bước Adagrad đầu "đúng mức" (tránh NaN do update đầu)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1.0)
    n_nan_skipped = 0

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

            # SKIP batch NaN (bỏ update, không để NaN lan)
            if torch.isnan(loss) or torch.isinf(loss):
                n_nan_skipped += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # chặn gradient "bay"
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Agg Epoch {epoch+1}: loss={total_loss/max(1,n_batches):.4f}"
                  + (f"  (skipped {n_nan_skipped} NaN batches)" if n_nan_skipped else ""))


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
    # STEP 1+2: Partition users (cache C/C_U/C_I như utility cũ)
    #   Nếu có cache -> LOAD (không cần WMF, không re-cluster).
    #   Nếu chưa -> train WMF (nếu cần), partition, rồi SAVE cache.
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 1+2: PARTITION USERS (with C/C_U/C_I cache)")
    print(f"{'='*70}")

    cache_dir = data_dir_for(dataset)
    partitioner = DataPartitioner(n_shards=n_shards, partition_type=partition_type, seed=42)

    if partitioner.load_cache(cache_dir, train_data):
        print("  Using cached partition (skipping WMF + re-clustering)")
        partitioner.build_shard_data(train_data)
        user_emb, item_emb = None, None   # không cần nữa
    else:
        # Chưa có cache: cần WMF (nếu partition_type != 3)
        if partition_type == 3:
            user_emb, item_emb = None, None
            print("  [Random] không cần WMF")
        elif max_epochs_wmf == 0:
            user_emb, item_emb = None, None
            print("  Skipping WMF (max_epochs_wmf=0); partition sẽ run không có embedding")
        else:
            user_emb, item_emb = load_or_train_wmf(dataset, emb_dim, max_epochs_wmf, batch_size, lr)

        partitioner.partition_users(train_data, n_users, user_emb, item_emb)
        partitioner.save_cache(cache_dir, train_data)
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

    # Check if model already exists (disabled due to corruption issues)
    if False and os.path.exists(checkpoint_path):
        print(f"  Loading model from checkpoint: {checkpoint_path}")
        load_model(model, checkpoint_path, device)
    else:
        print(f"  Training RecEraser on all shards (first time)...")
        if user_emb is not None and item_emb is not None:
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
    recall_before, ndcg_before = evaluate_model(model, partitioner, train_data, test_data, n_users, n_items, device)
    print(f"  Before (FULL) - R@10: {recall_before:.4f}, NDCG@10: {ndcg_before:.4f}")

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

    # AFTER: Train on RETAINED data, evaluate on RETAINED test
    train_data_retained = {u: items for u, items in train_data.items() if u not in unlearn_users}
    recall_after, ndcg_after = evaluate_model(model, partitioner, train_data_retained, test_data_retained, n_users, n_items, device)
    print(f"  After (RETAINED, retrain) - R@10: {recall_after:.4f}, NDCG@10: {ndcg_after:.4f}")
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
