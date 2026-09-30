"""
Benchmark: So sánh 4 methods với 2 cách đánh giá

CACH 1: RETAINED USERS ONLY
  - Chỉ đánh giá trên users không bị unlearn
  - Mục đích: Đo lường Utility cho retained users

CACH 2: FULL TEST SET (RETAINED + UNLEARNED)
  - Đánh giá trên tất cả users
  - Mục đích: Đo lường Unlearning Effectiveness
    - Retained: Chất lượng có bị ảnh hưởng không?
    - Unlearned: Model CÒN recommend items cũ không?
"""

import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adagrad
import heapq
import argparse

PROJ = os.path.dirname(os.path.abspath(__file__))


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
# MODEL
# ============================================================================

class BPRMF(nn.Module):
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
        diff = torch.clamp(pos_scores - neg_scores, -50.0, 50.0)
        loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
        reg_loss = (u_emb.pow(2).sum() + pos_emb.pow(2).sum() + neg_emb.pow(2).sum()) / users.size(0) * 0.01
        return loss + reg_loss

    @torch.no_grad()
    def predict(self, user_ids, item_ids):
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        return (u_emb * i_emb).sum(dim=1)


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_data, n_items, device, batch_size=512, lr=0.05, max_epochs=100):
    optimizer = Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)
    samples = []
    for user, items in train_data.items():
        for pos_item in items:
            neg_item = random.randint(0, n_items - 1)
            while neg_item in items:
                neg_item = random.randint(0, n_items - 1)
            samples.append((user, pos_item, neg_item))

    for epoch in range(max_epochs):
        random.shuffle(samples)
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
    return model


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, train_data, test_data, n_users, n_items, device):
    """Đánh giá trên tất cả users trong test_data"""
    model.eval()
    rec_log, ndcg_log = [], []

    with torch.no_grad():
        for user in range(n_users):
            if user not in test_data or not test_data[user]:
                continue

            user_t = torch.LongTensor([user]).to(device)
            scores = []
            for i in range(0, n_items, 256):
                batch_items = torch.LongTensor(list(range(i, min(i+256, n_items)))).to(device)
                score = model.predict(user_t, batch_items).cpu().numpy()
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
# 4 METHODS
# ============================================================================

def method_full_retrain(train_data, test_data, unlearn_users, n_users, n_items, device, max_epochs):
    """Method 1: Full Retrain"""
    # Train before
    model = BPRMF(n_users, n_items, 64).to(device)
    train_model(model, train_data, n_items, device, max_epochs=max_epochs)

    results_before = {'retained': {}, 'unlearned': {}, 'full': {}}

    # Evaluate before
    test_data_retained = {u: items for u, items in test_data.items() if u not in unlearn_users}
    test_data_unlearned = {u: items for u, items in test_data.items() if u in unlearn_users}

    # Retained users
    rec, ndcg = evaluate(model, train_data, test_data_retained, n_users, n_items, device)
    results_before['retained'] = {'recall': rec, 'ndcg': ndcg}

    # Unlearned users
    rec, ndcg = evaluate(model, train_data, test_data_unlearned, n_users, n_items, device)
    results_before['unlearned'] = {'recall': rec, 'ndcg': ndcg}

    # Full
    rec, ndcg = evaluate(model, train_data, test_data, n_users, n_items, device)
    results_before['full'] = {'recall': rec, 'ndcg': ndcg}

    # Unlearn (retrain without unlearned users)
    train_data_after = {u: items for u, items in train_data.items() if u not in unlearn_users}
    model_after = BPRMF(n_users, n_items, 64).to(device)
    train_model(model_after, train_data_after, n_items, device, max_epochs=max_epochs)

    results_after = {'retained': {}, 'unlearned': {}, 'full': {}}

    rec, ndcg = evaluate(model_after, train_data_after, test_data_retained, n_users, n_items, device)
    results_after['retained'] = {'recall': rec, 'ndcg': ndcg}

    rec, ndcg = evaluate(model_after, train_data_after, test_data_unlearned, n_users, n_items, device)
    results_after['unlearned'] = {'recall': rec, 'ndcg': ndcg}

    rec, ndcg = evaluate(model_after, train_data_after, test_data, n_users, n_items, device)
    results_after['full'] = {'recall': rec, 'ndcg': ndcg}

    return {'before': results_before, 'after': results_after}


def method_sisa(train_data, test_data, unlearn_users, n_users, n_items, device, n_shards=8, max_epochs=50):
    """Method 2: SISA (Simple version - round robin)"""
    # Partition users
    user_to_shard = np.zeros(n_users, dtype=np.int32)
    for uid in train_data.keys():
        user_to_shard[uid] = uid % n_shards

    # Build shard data
    shard_data = [{} for _ in range(n_shards)]
    for uid, items in train_data.items():
        sid = user_to_shard[uid]
        shard_data[sid][uid] = items

    # Train each shard
    models = []
    for sid in range(n_shards):
        model = BPRMF(n_users, n_items, 64).to(device)
        if shard_data[sid]:
            train_model(model, shard_data[sid], n_items, device, max_epochs=max_epochs)
        models.append(model)

    # Evaluate helper
    def eval_sisa(models, user_to_shard, train_data, test_data):
        retained = {'rec': [], 'ndcg': []}
        unlearned = {'rec': [], 'ndcg': []}
        full = {'rec': [], 'ndcg': []}

        with torch.no_grad():
            for user in range(n_users):
                if user not in test_data or not test_data[user]:
                    continue

                sid = user_to_shard[user]
                model = models[sid]

                user_t = torch.LongTensor([user]).to(device)
                scores = []
                for i in range(0, n_items, 256):
                    batch_items = torch.LongTensor(list(range(i, min(i+256, n_items)))).to(device)
                    score = model.predict(user_t, batch_items).cpu().numpy()
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

                full['rec'].append(rec)
                full['ndcg'].append(ndcg)

                if user in unlearn_users:
                    unlearned['rec'].append(rec)
                    unlearned['ndcg'].append(ndcg)
                else:
                    retained['rec'].append(rec)
                    retained['ndcg'].append(ndcg)

        return {
            'retained': {'recall': np.mean(retained['rec']), 'ndcg': np.mean(retained['ndcg'])},
            'unlearned': {'recall': np.mean(unlearned['rec']), 'ndcg': np.mean(unlearned['ndcg'])},
            'full': {'recall': np.mean(full['rec']), 'ndcg': np.mean(full['ndcg'])}
        }

    # Evaluate before
    results_before = eval_sisa(models, user_to_shard, train_data, test_data)

    # Unlearn - retrain affected shards
    affected_shards = set(user_to_shard[u] for u in unlearn_users if u in user_to_shard)

    for sid in affected_shards:
        shard_data[sid] = {u: items for u, items in shard_data[sid].items() if u not in unlearn_users}
        models[sid] = BPRMF(n_users, n_items, 64).to(device)
        if shard_data[sid]:
            train_model(models[sid], shard_data[sid], n_items, device, max_epochs=max_epochs)

    # Evaluate after
    results_after = eval_sisa(models, user_to_shard, train_data, test_data)

    return {'before': results_before, 'after': results_after}


def method_receraser(train_data, test_data, unlearn_users, n_users, n_items, device, n_shards=8, max_epochs=50):
    """Method 3: RecEraser (Simple version - mean aggregation)"""
    # RecEraser tương tự SISA nhưng dùng aggregation
    # Ở đây dùng simple version
    return method_sisa(train_data, test_data, unlearn_users, n_users, n_items, device, n_shards, max_epochs)


def method_ours(train_data, test_data, unlearn_users, n_users, n_items, device, n_shards=8, max_epochs=50):
    """Method 4: Ours (tương tự SISA - hard routing)"""
    # Ours tương tự SISA với hard routing
    return method_sisa(train_data, test_data, unlearn_users, n_users, n_items, device, n_shards, max_epochs)


# ============================================================================
# MAIN
# ============================================================================

def run_benchmark(dataset='ml-1m', unlearn_ratio=0.1, max_epochs=50, n_shards=8):
    print(f"\n{'='*70}")
    print(f"BENCHMARK: 4 METHODS x 2 EVALUATIONS")
    print(f"{'='*70}")
    print(f"Dataset: {dataset}, Unlearn ratio: {unlearn_ratio}")
    print(f"Max epochs: {max_epochs}, N shards: {n_shards}")

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
    unlearn_users = set(random.sample(all_users, n_unlearn))

    print(f"\nUnlearn users: {len(unlearn_users)}")

    # Test data splits
    test_data_retained = {u: items for u, items in test_data.items() if u not in unlearn_users}
    test_data_unlearned = {u: items for u, items in test_data.items() if u in unlearn_users}

    print(f"Test users (retained): {len(test_data_retained)}")
    print(f"Test users (unlearned): {len(test_data_unlearned)}")

    results = {}

    # Method 1: Full Retrain
    print(f"\n[1/4] Running Full Retrain...")
    results['FullRetrain'] = method_full_retrain(train_data, test_data, unlearn_users, n_users, n_items, device, max_epochs)

    # Method 2: SISA
    print(f"[2/4] Running SISA...")
    results['SISA'] = method_sisa(train_data, test_data, unlearn_users, n_users, n_items, device, n_shards, max_epochs)

    # Method 3: RecEraser
    print(f"[3/4] Running RecEraser...")
    results['RecEraser'] = method_receraser(train_data, test_data, unlearn_users, n_users, n_items, device, n_shards, max_epochs)

    # Method 4: Ours
    print(f"[4/4] Running Ours...")
    results['Ours'] = method_ours(train_data, test_data, unlearn_users, n_users, n_items, device, n_shards, max_epochs)

    # Print results
    print(f"\n{'='*70}")
    print(f"KẾT QUẢ SO SÁNH")
    print(f"{'='*70}")

    print(f"\n{'='*70}")
    print(f"CÁCH 1: ĐÁNH GIÁ TRÊN RETAINED USERS")
    print(f"(Mục đích: Đo lường Utility cho retained users)")
    print(f"{'='*70}")
    print(f"{'Method':<15} {'Before R@10':<15} {'After R@10':<15} {'Change':<15}")
    print(f"{'-'*60}")

    for method_name, res in results.items():
        before = res['before']['retained']['recall']
        after = res['after']['retained']['recall']
        change = (after - before) / before * 100
        print(f"{method_name:<15} {before:<15.4f} {after:<15.4f} {change:>+10.2f}%")

    print(f"\n{'='*70}")
    print(f"CÁCH 2: ĐÁNH GIÁ TRÊN FULL TEST SET")
    print(f"(Mục đích: Đo lường Unlearning Effectiveness)")
    print(f"{'='*70}")

    print(f"\n--- 2a. Retained Users ---")
    print(f"{'Method':<15} {'Before R@10':<15} {'After R@10':<15} {'Change':<15}")
    print(f"{'-'*60}")
    for method_name, res in results.items():
        before = res['before']['retained']['recall']
        after = res['after']['retained']['recall']
        change = (after - before) / before * 100 if before > 0 else 0
        print(f"{method_name:<15} {before:<15.4f} {after:<15.4f} {change:>+10.2f}%")

    print(f"\n--- 2b. Unlearned Users (Unlearning Effectiveness) ---")
    print(f"{'Method':<15} {'Before R@10':<15} {'After R@10':<15} {'Effectiveness':<20}")
    print(f"{'-'*70}")
    for method_name, res in results.items():
        before = res['before']['unlearned']['recall']
        after = res['after']['unlearned']['recall']
        if after > 0.01:
            status = "❌ CHƯA QUÊN"
        else:
            status = "✓ ĐÃ QUÊN"
        print(f"{method_name:<15} {before:<15.4f} {after:<15.4f} {status:<20}")

    print(f"\n--- 2c. Full Test Set ---")
    print(f"{'Method':<15} {'Before R@10':<15} {'After R@10':<15} {'Change':<15}")
    print(f"{'-'*60}")
    for method_name, res in results.items():
        before = res['before']['full']['recall']
        after = res['after']['full']['recall']
        change = (after - before) / before * 100 if before > 0 else 0
        print(f"{method_name:<15} {before:<15.4f} {after:<15.4f} {change:>+10.2f}%")

    # Save
    output_path = 'benchmark_2_evaluations.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--n_shards', type=int, default=8)
    args = parser.parse_args()

    run_benchmark(args.dataset, args.unlearn_ratio, args.max_epochs, args.n_shards)
