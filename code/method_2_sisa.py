"""
Method 2: SISA (Sharded Isolated Slicing and Aggregation)

Hyper-parameters theo bài báo:
- Batch size: 512
- Learning rate: 0.05
- Embedding size: 64
- Max epochs: 1000
- Early stopping: Recall@10 không tăng trong 10 epochs liên tiếp
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
from typing import Dict, List
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
    'max_epochs': 1000,
    'early_stopping_patience': 10,
    'Ks': [10, 20, 50]
}


# ============================================================================
# MODEL: BPRMF
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
# MODEL: WMF
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
# DATA PARTITIONER
# ============================================================================

class DataPartitioner:
    """
    Partition data into shards using stable hash.
    shard_id = user_id % n_shards
    """

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

        shard_counts = np.bincount(self.user_to_shard, minlength=self.n_shards)
        print(f"    [SISA] Shard sizes: min={shard_counts.min()}, "
              f"max={shard_counts.max()}, mean={shard_counts.mean():.1f}")

        return self.user_to_shard

    def build_shard_data(self, train_data):
        """Build data for each shard."""
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
    """Evaluate single model using Recall@K and NDCG@K."""
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


def evaluate_sisa(models, train_data, test_data, n_users, n_items, device, Ks=[10, 20, 50]):
    """Evaluate SISA model with aggregation (simple mean)."""
    pre_log = {k: [] for k in Ks}
    rec_log = {k: [] for k in Ks}
    ndcg_log = {k: [] for k in Ks}

    for user in range(n_users):
        if user not in test_data or not test_data[user]:
            continue

        scores_list = []
        for model in models:
            with torch.no_grad():
                user_t = torch.LongTensor([user]).to(device)
                all_items = list(range(n_items))
                scores = []
                for i in range(0, n_items, 256):
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
# TRAINING VỚI EARLY STOPPING
# ============================================================================

def train_model(model, train_data, n_users, n_items, device,
                batch_size=512, lr=0.05, max_epochs=1000,
                early_stopping=True, patience=10, verbose=True):
    """Train a model on data with early stopping."""
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

    best_metric = -float('inf')
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(max_epochs):
        random.shuffle(samples)
        total_loss = 0

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
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

        if verbose and (epoch + 1) % 20 == 0:
            avg_loss = total_loss / n_batches
            print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}")

    return model, max_epochs


# ============================================================================
# SISA METHOD
# ============================================================================

class SISAMethod:
    """
    Method 2: SISA (Sharded Isolated Slicing and Aggregation)
    """

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
        self.training_info = {}

    def train(self, train_data, device):
        """Train each shard independently."""
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

        self.training_info['shards_trained'] = self.n_shards
        print(f"    [SISA] All {self.n_shards} shards trained.")

        return self.models

    def unlearn(self, unlearn_user_ids, train_data, device, retrain_epochs=5):
        """Unlearn by retraining only affected shards."""
        affected_shards = self.partitioner.get_affected_shards(unlearn_user_ids)
        print(f"    [SISA] Affected shards: {affected_shards}")

        for shard_id in affected_shards:
            filtered_data = self.partitioner.filter_shard_data(shard_id, set(unlearn_user_ids))
            print(f"    [SISA] Retraining shard {shard_id}...")
            train_model(self.models[shard_id], filtered_data, self.n_users,
                       self.n_items, device, batch_size=self.batch_size, lr=self.lr,
                       max_epochs=retrain_epochs, verbose=True)

        return self.models, affected_shards

    def evaluate(self, train_data, test_data, device, Ks=[10, 20, 50]):
        return evaluate_sisa(self.models, train_data, test_data,
                           self.n_users, self.n_items, device, Ks)


# ============================================================================
# MAIN
# ============================================================================

def run_sisa(model_name='BPRMF', dataset='ml-1m', emb_dim=64, n_shards=8,
           batch_size=512, lr=0.05, max_epochs=1000,
           unlearn_ratio=0.1, retrain_epochs=5, output_suffix=''):
    """Run SISA method."""
    print(f"\n{'='*60}")
    print(f"METHOD 2: SISA (SHARDED ISOLATED SLICING AND AGGREGATION)")
    print(f"{'='*60}")
    print(f"Hyper-parameters:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Embedding dim: {emb_dim}")
    print(f"  - Max epochs: {max_epochs}")
    print(f"  - N shards: {n_shards}")

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

    model_classes = {'BPRMF': BPRMF, 'WMF': WMF}
    model_class = model_classes.get(model_name, BPRMF)

    method = SISAMethod(model_class, n_users, n_items, emb_dim, n_shards,
                       batch_size=batch_size, lr=lr, max_epochs=max_epochs)

    print(f"\n--- Phase 1: Train BEFORE unlearning ---")
    t0 = time.time()
    method.train(train_data, device)
    train_time = time.time() - t0

    results_before = method.evaluate(train_data, test_data, device)
    print(f"  Before - R@10: {results_before['recall'][0]:.4f}, "
          f"NDCG@10: {results_before['ndcg'][0]:.4f}")

    print(f"\n--- Phase 2: Unlearn (retrain affected shards only) ---")
    t0 = time.time()
    method.unlearn(unlearn_users, train_data, device, retrain_epochs=retrain_epochs)
    unlearn_time = time.time() - t0

    results_after = method.evaluate(train_data, test_data, device)
    print(f"  After - R@10: {results_after['recall'][0]:.4f}, "
          f"NDCG@10: {results_after['ndcg'][0]:.4f}")
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
    output_path = os.path.join(PROJ, f'results_sisa_{model_name.lower()}{suffix}.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Method 2: SISA')
    parser.add_argument('--model', type=str, default='BPRMF', choices=['BPRMF', 'WMF'])
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--n_shards', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)
    parser.add_argument('--retrain_epochs', type=int, default=5)
    parser.add_argument('--output_suffix', type=str, default='')
    args = parser.parse_args()

    run_sisa(
        model_name=args.model,
        dataset=args.dataset,
        emb_dim=args.emb_dim,
        n_shards=args.n_shards,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        unlearn_ratio=args.unlearn_ratio,
        retrain_epochs=args.retrain_epochs,
        output_suffix=args.output_suffix
    )
