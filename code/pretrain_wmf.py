"""
Train WMF (Weighted Matrix Factorization) để lấy pretrained embeddings
Dùng cho RecEraser và Ours
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
    data_path = os.environ.get('RECUNLEARN_DATA_PATH', None)
    if data_path:
        dataset_name = os.environ.get('RECUNLEARN_DATASET', dataset)
        data_dir = os.path.join(data_path, dataset_name)
    else:
        base_dir = os.path.dirname(PROJ)
        data_dir = os.path.join(base_dir, 'data', dataset)
    return SimpleDataLoader(data_dir, batch_size)


# ============================================================================
# WMF MODEL
# ============================================================================

class WMF(nn.Module):
    """Weighted Matrix Factorization"""
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

        # Weighted BPR loss
        pos_loss = -pos_weights * torch.log(torch.sigmoid(pos_scores) + 1e-10)
        neg_loss = -neg_weights * torch.log(1 - torch.sigmoid(neg_scores) + 1e-10)
        loss = (pos_loss + neg_loss).mean()

        return loss


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

            u_emb = model.user_embedding.weight[user]
            all_items = list(range(n_items))
            scores = []

            for i in range(0, n_items, 256):
                batch_items = torch.LongTensor(all_items[i:i+256]).to(device)
                i_emb = model.item_embedding(batch_items)
                score = (u_emb * i_emb).sum(dim=1).cpu().numpy()
                scores.extend(score.tolist())

            scores = np.array(scores)
            train_items = set(train_data.get(user, []))
            for item in train_items:
                scores[item] = -np.inf

            rank_list = heapq.nlargest(max(Ks), range(len(scores)), key=scores.__getitem__)
            item_pos = test_data.get(user, [])
            item_set = set(item_pos)

            for k in Ks:
                hit_list = rank_list[:k]
                hit_num = len(set(hit_list) & item_set)
                rec = hit_num / len(item_pos) if len(item_pos) > 0 else 0

                dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(hit_list) if item in item_set)
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(item_pos), k)))
                ndcg = dcg / idcg if idcg > 0 else 0

                rec_log[k].append(rec)
                ndcg_log[k].append(ndcg)

    model.train()
    return {
        'recall': [np.mean(rec_log[k]) for k in Ks],
        'ndcg': [np.mean(ndcg_log[k]) for k in Ks]
    }


# ============================================================================
# TRAIN WMF
# ============================================================================

def train_wmf(train_data, n_users, n_items, emb_dim, device, batch_size=512, lr=0.05, max_epochs=100):
    """Train WMF model"""
    model = WMF(n_users, n_items, emb_dim).to(device)
    optimizer = Adagrad(model.parameters(), lr=lr)

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
            metrics = evaluate_model(model, train_data, train_data, n_users, n_items, device)
            print(f"    Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, recall@10={metrics['recall'][0]:.4f}")

    return model


# ============================================================================
# SAVE / LOAD EMBEDDINGS
# ============================================================================

def save_embeddings(model, path):
    """Save embeddings to file"""
    user_emb = model.user_embedding.weight.data.cpu().numpy()
    item_emb = model.item_embedding.weight.data.cpu().numpy()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, user_embeddings=user_emb, item_embeddings=item_emb)
    print(f"  Saved embeddings to {path}")


def load_embeddings(path, device='cpu'):
    """Load embeddings from file"""
    data = np.load(path)
    return data['user_embeddings'], data['item_embeddings']


# ============================================================================
# MAIN
# ============================================================================

def run_pretrain_wmf(dataset='ml-1m', emb_dim=64, batch_size=512, lr=0.05, max_epochs=100, output_path=None):
    print(f"\n{'='*60}")
    print(f"PRETRAIN WMF FOR RECERASER")
    print(f"{'='*60}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    print(f"\nLoading data...")
    data = load_data(dataset=dataset, batch_size=batch_size)
    n_users, n_items = data.n_users, data.n_items
    train_data, test_data = data.train_items, data.test_set

    # Train WMF
    t0 = time.time()
    model = train_wmf(train_data, n_users, n_items, emb_dim, device, batch_size, lr, max_epochs)
    train_time = time.time() - t0

    # Evaluate
    results = evaluate_model(model, train_data, test_data, n_users, n_items, device)
    print(f"\n  WMF Results: R@10={results['recall'][0]:.4f}, NDCG@10={results['ndcg'][0]:.4f}")
    print(f"  Training time: {train_time:.2f}s")

    # Save embeddings
    if output_path is None:
        output_path = os.path.join(PROJ, f'../data/{dataset}/wmf_embeddings.npz')

    save_embeddings(model, output_path)

    # Print embedding stats
    user_emb = model.user_embedding.weight.data
    item_emb = model.item_embedding.weight.data
    print(f"\n  Embedding stats:")
    print(f"    User emb: mean={user_emb.mean():.4f}, std={user_emb.std():.4f}, norm={user_emb.norm():.2f}")
    print(f"    Item emb: mean={item_emb.mean():.4f}, std={item_emb.std():.4f}, norm={item_emb.norm():.2f}")

    return model, results, train_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain WMF embeddings')
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--output_path', type=str, default=None)

    args = parser.parse_args()

    run_pretrain_wmf(
        dataset=args.dataset,
        emb_dim=args.emb_dim,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        max_epochs=args.max_epochs,
        output_path=args.output_path
    )
