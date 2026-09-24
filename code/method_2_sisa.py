"""
Method 2: SISA (Sharded Isolated Slicing and Aggregation)

Hyper-parameters theo bài báo:
- Batch size: 512
- Learning rate: 0.05
- Embedding size: 64
- Max epochs: 1000
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
sys.path.insert(0, PROJ)


# ============================================================================
# CUSTOM DATA LOADER
# ============================================================================

class SimpleDataLoader:
    """Simple data loader không dùng utility.parser."""

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
    """Load data."""
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
# MODEL: BPRMF
# ============================================================================

class BPRMF(nn.Module):
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


class WMF(nn.Module):
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
# DATA PARTITIONER
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
        print(f"    [SISA] Shard sizes: min={shard_counts.min()}, max={shard_counts.max()}")

        return self.user_to_shard

    def build_shard_data(self, train_data):
        self.shard_data = [{} for _ in range(self.n_shards)]

        for user_id, items in train_data.items():
            shard_id = self.user_to_shard[user_id]
            self.shard_data[shard_id][user_id] = items.copy()

        for sid in range(self.n_shards):
            n_users = len(self.shard_data[sid])
            n_interactions = sum(len(items) for items in self.shard_data[sid].values())
            print(f"    [SISA] Shard {sid}: {n_users} users, {n_interactions} interactions")

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


def evaluate_sisa(models, train_data, test_data, n_users, n_items, device, user_to_shard, Ks=[10, 20, 50]):
    """Evaluate SISA - chỉ dùng model của shard mà user được assign (KHÔNG average)."""
    pre_log = {k: [] for k in Ks}
    rec_log = {k: [] for k in Ks}
    ndcg_log = {k: [] for k in Ks}

    for user in range(n_users):
        if user not in test_data or not test_data[user]:
            continue

        # Chỉ dùng model của shard mà user được assign
        shard_id = user_to_shard[user]
        model = models[shard_id]

        with torch.no_grad():
            user_t = torch.LongTensor([user]).to(device)
            all_items = list(range(n_items))
            scores = []
            for i in range(0, n_items, 256):
                batch_items = torch.LongTensor(all_items[i:i+256]).to(device)
                s = model.predict(user_t, batch_items).cpu().numpy()
                scores.extend(s.tolist())
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

    return {
        'precision': [np.mean(pre_log[k]) for k in Ks],
        'recall': [np.mean(rec_log[k]) for k in Ks],
        'ndcg': [np.mean(ndcg_log[k]) for k in Ks]
    }


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_data, n_users, n_items, device,
                batch_size=512, lr=0.05, max_epochs=1000, verbose=True):
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
        total_loss = 0

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch = samples[start_idx:end_idx]

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

        if verbose and (epoch + 1) % 20 == 0:
            avg_loss = total_loss / n_batches
            print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}")

    return model, max_epochs


# ============================================================================
# SISA METHOD
# ============================================================================

class SISAMethod:
    def __init__(self, model_class, n_users, n_items, emb_dim, n_shards=8,
                 batch_size=512, lr=0.05, max_epochs=1000):
        self.model_class = model_class
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_shards = n_shards
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.models = None
        self.partitioner = DataPartitioner(n_shards)

    def train(self, train_data, device):
        print("    [SISA] Partitioning users...")
        self.partitioner.partition_users(train_data, self.n_users)

        print("    [SISA] Building shard data...")
        self.partitioner.build_shard_data(train_data)

        print("    [SISA] Training each shard...")
        self.models = []

        for shard_id in range(self.n_shards):
            shard_data = self.partitioner.shard_data[shard_id]
            model = self.model_class(self.n_users, self.n_items, self.emb_dim).to(device)
            print(f"    [SISA] Training shard {shard_id}...")
            model, epochs = train_model(model, shard_data, self.n_users, self.n_items, device,
                                       batch_size=self.batch_size, lr=self.lr,
                                       max_epochs=self.max_epochs)
            self.models.append(model)

        return self.models

    def unlearn(self, unlearn_user_ids, train_data, device, retrain_epochs=50):
        affected_shards = self.partitioner.get_affected_shards(unlearn_user_ids)
        print(f"    [SISA] Affected shards: {affected_shards}")

        for shard_id in affected_shards:
            filtered_data = self.partitioner.filter_shard_data(shard_id, set(unlearn_user_ids))
            print(f"    [SISA] Retraining shard {shard_id}...")
            train_model(self.models[shard_id], filtered_data, self.n_users,
                       self.n_items, device, batch_size=self.batch_size, lr=self.lr,
                       max_epochs=retrain_epochs, verbose=True)

            # Reset embeddings cua unlearned users trong shard nay de "quen"
            unlearn_ids = list(unlearn_user_ids)
            for uid in unlearn_ids:
                shard_user = uid  # SISA dung round-robin
                if shard_user < self.models[shard_id].user_embedding.num_embeddings:
                    # Reset ve zero - model se khong con nho gi ve user nay
                    with torch.no_grad():
                        self.models[shard_id].user_embedding.weight[shard_user].zero_()
            print(f"    [SISA] Reset embeddings of unlearned users in shard {shard_id}")

        return self.models, affected_shards

    def evaluate(self, train_data, test_data, device, Ks=[10, 20, 50]):
        return evaluate_sisa(self.models, train_data, test_data,
                           self.n_users, self.n_items, device,
                           self.partitioner.user_to_shard, Ks)


# ============================================================================
# MAIN
# ============================================================================

def run_sisa(model_name='BPRMF', dataset='ml-1m', emb_dim=64, n_shards=8,
           batch_size=512, lr=0.05, max_epochs=1000,
           unlearn_ratio=0.1, unlearn_mode='random', unlearn_user_id=None,
           retrain_epochs=50, output_suffix=''):
    print(f"\n{'='*60}")
    print(f"METHOD 2: SISA")
    print(f"{'='*60}")
    print(f"Hyper-parameters:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Embedding dim: {emb_dim}")
    print(f"  - Max epochs: {max_epochs}")
    print(f"  - N shards: {n_shards}")

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

    model_classes = {'BPRMF': BPRMF, 'WMF': WMF}
    model_class = model_classes.get(model_name, BPRMF)

    method = SISAMethod(model_class, n_users, n_items, emb_dim, n_shards,
                       batch_size=batch_size, lr=lr, max_epochs=max_epochs)

    print(f"\n--- Phase 1: Train BEFORE unlearning ---")
    t0 = time.time()
    method.train(train_data, device)
    train_time = time.time() - t0

    # Keep original train_data for evaluation (before AND after use same mask)
    train_data_original = {u: items.copy() for u, items in train_data.items()}

    # Evaluate BEFORE unlearn on FULL test set (bao gồm cả unlearned users)
    results_before = method.evaluate(train_data_original, test_data, device)
    print(f"  Before (FULL) - R@10: {results_before['recall'][0]:.4f}, NDCG@10: {results_before['ndcg'][0]:.4f}")

    print(f"\n--- Phase 2: Unlearn (retrain affected shards only) ---")
    t0 = time.time()
    method.unlearn(unlearn_users, train_data, device, retrain_epochs=retrain_epochs)
    unlearn_time = time.time() - t0

    # Evaluate AFTER unlearn on FULL test set (cùng tập với before - fair comparison)
    # Use ORIGINAL train_data to mask items (not the modified one)
    results_after = method.evaluate(train_data_original, test_data, device)
    print(f"  After (FULL) - R@10: {results_after['recall'][0]:.4f}, NDCG@10: {results_after['ndcg'][0]:.4f}")
    print(f"  Unlearn time: {unlearn_time:.2f}s")

    results = {
        'method': 'SISA',
        'model': model_name,
        'dataset': dataset,
        'hyperparameters': {
            'batch_size': batch_size,
            'learning_rate': lr,
            'embedding_dim': emb_dim,
            'max_epochs': max_epochs,
            'n_shards': n_shards
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
    output_path = os.path.join(PROJ, f'results_sisa_{model_name.lower()}{suffix}.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Method 2: SISA')
    parser.add_argument('--model_name', type=str, default='BPRMF', choices=['BPRMF', 'WMF'])
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--n_shards', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)
    parser.add_argument('--unlearn_mode', type=str, default='random',
                       choices=['random', 'fewest', 'most', 'single'],
                       help='random: ngau nhien, fewest: it interaction nhat, most: nhieu interaction nhat, single: 1 user')
    parser.add_argument('--unlearn_user_id', type=int, default=None,
                       help='Chi dinh user ID cu the de unlearn (dung voi --unlearn_mode single)')
    parser.add_argument('--retrain_epochs', type=int, default=50)
    parser.add_argument('--output_suffix', type=str, default='')

    args = parser.parse_args()

    run_sisa(
        model_name=args.model_name,
        dataset=args.dataset,
        emb_dim=args.emb_dim,
        n_shards=args.n_shards,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        max_epochs=args.max_epochs,
        unlearn_ratio=args.unlearn_ratio,
        unlearn_mode=args.unlearn_mode,
        unlearn_user_id=args.unlearn_user_id,
        retrain_epochs=args.retrain_epochs,
        output_suffix=args.output_suffix
    )
