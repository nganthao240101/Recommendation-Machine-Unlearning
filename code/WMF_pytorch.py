"""
WMF (Weighted Matrix Factorization) in PyTorch.
Train to get pretrained embeddings for RecEraser.
"""
import os
import sys
import time
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adagrad

# Setup paths
PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

os.environ['RECUNLEARN_DATA_PATH'] = os.path.join(os.path.dirname(PROJ), 'data/')
os.environ['RECUNLEARN_DATASET'] = 'ml-1m'

from utility.parser import parse_args
from utility.load_data import Data
import heapq

# Override args
args = parse_args()
args.data_path = os.path.join(os.path.dirname(PROJ), 'data/ml-1m')


class WMF(nn.Module):
    """Weighted Matrix Factorization."""

    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        # Initialize
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, users, pos_items, neg_items):
        """BPR-like loss."""
        u_emb = self.user_embedding(users)
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        diff = pos_scores - neg_scores
        loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

        # L2 regularization
        reg_loss = (u_emb.pow(2).sum() + pos_emb.pow(2).sum() +
                    neg_emb.pow(2).sum()) / users.size(0) * 0.01

        return loss + reg_loss


def get_train_samples(train_items, n_items):
    """Get all (user, pos_item, neg_item) triplets."""
    samples = []
    for user, items in train_items.items():
        for pos_item in items:
            # Random negative item
            neg_item = random.randint(0, n_items - 1)
            while neg_item in items:
                neg_item = random.randint(0, n_items - 1)
            samples.append((user, pos_item, neg_item))
    return samples


def evaluate(model, train_items, test_set, n_users, n_items, device, Ks=[10, 20, 50]):
    """Evaluate model."""
    model.eval()
    all_users = list(range(n_users))

    pre_loger = {k: [] for k in Ks}
    rec_loger = {k: [] for k in Ks}
    ndcg_loger = {k: [] for k in Ks}

    with torch.no_grad():
        for user in all_users:
            users_t = torch.LongTensor([user] * 100).to(device)
            items_t = torch.LongTensor(list(range(100))).to(device)

            # Get all item scores
            scores = []
            batch_size = 256
            for i in range(0, n_items, batch_size):
                batch_items = torch.LongTensor(list(range(i, min(i + batch_size, n_items)))).to(device)
                u_emb = model.user_embedding(torch.LongTensor([user]).to(device))
                i_emb = model.item_embedding(batch_items)
                score = (u_emb * i_emb).sum(dim=1).cpu().numpy()
                scores.extend(score.tolist())

            scores = np.array(scores)

            # Mask training items
            train_items_user = train_items.get(user, [])
            for item in train_items_user:
                scores[item] = -np.inf

            # Get top-K
            rank_list = heapq.nlargest(max(Ks), range(len(scores)), key=scores.__getitem__)

            item_pos = test_set.get(user, [])
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

                pre_loger[k].append(pre)
                rec_loger[k].append(rec)
                ndcg_loger[k].append(ndcg)

    model.train()

    return {
        'precision': [np.mean(pre_loger[k]) for k in Ks],
        'recall': [np.mean(rec_loger[k]) for k in Ks],
        'ndcg': [np.mean(ndcg_loger[k]) for k in Ks]
    }


def train():
    # Load data
    print("Loading data...")
    data_gen = Data(
        path=args.data_path,
        batch_size=512,
        part_type=1,
        part_num=1,
        part_T=5
    )

    n_users = data_gen.n_users
    n_items = data_gen.n_items
    train_items = data_gen.train_items
    test_set = data_gen.test_set

    print(f"n_users={n_users}, n_items={n_items}")
    print(f"n_interactions={sum(len(items) for items in train_items.values())}")

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create model
    model = WMF(n_users, n_items, args.embed_size).to(device)
    optimizer = Adagrad(model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)

    # Get training samples
    print("Preparing training samples...")
    samples = get_train_samples(train_items, n_items)
    print(f"Total samples: {len(samples)}")

    # Train
    n_epochs = args.epoch
    batch_size = args.batch_size
    n_batches = len(samples) // batch_size + 1

    print(f"\nTraining WMF: {n_epochs} epochs, {n_batches} batches/epoch")

    for epoch in range(n_epochs):
        t0 = time.time()
        random.shuffle(samples)

        loss_sum = 0
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(samples))

            batch_samples = samples[start:end]
            if not batch_samples:
                continue

            users = [s[0] for s in batch_samples]
            pos_items = [s[1] for s in batch_samples]
            neg_items = [s[2] for s in batch_samples]

            users_t = torch.LongTensor(users).to(device)
            pos_t = torch.LongTensor(pos_items).to(device)
            neg_t = torch.LongTensor(neg_items).to(device)

            loss = model(users_t, pos_t, neg_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        epoch_time = time.time() - t0
        print(f"Epoch {epoch}: loss={loss_sum / n_batches:.4f}, time={epoch_time:.1f}s")

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            ret = evaluate(model, train_items, test_set, n_users, n_items, device)
            print(f"  Recall@10: {ret['recall'][0]:.4f}, "
                  f"Recall@20: {ret['recall'][1]:.4f}, "
                  f"NDCG@10: {ret['ndcg'][0]:.4f}")

    # Get final embeddings
    print("\nExtracting embeddings...")
    model.eval()
    with torch.no_grad():
        user_emb = model.user_embedding.weight.cpu().numpy()
        item_emb = model.item_embedding.weight.cpu().numpy()

    print(f"User embeddings: {user_emb.shape}")
    print(f"Item embeddings: {item_emb.shape}")

    # Save embeddings
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save with timestamp
    user_file = os.path.join(args.data_path, f'user_pretrain_{timestamp}.pk')
    item_file = os.path.join(args.data_path, f'item_pretrain_{timestamp}.pk')

    with open(user_file, 'wb') as f:
        pickle.dump(user_emb, f)
    with open(item_file, 'wb') as f:
        pickle.dump(item_emb, f)

    # Also save as default
    with open(os.path.join(args.data_path, 'user_pretrain.pk'), 'wb') as f:
        pickle.dump(user_emb, f)
    with open(os.path.join(args.data_path, 'item_pretrain.pk'), 'wb') as f:
        pickle.dump(item_emb, f)

    print(f"\n[SAVED] Embeddings:")
    print(f"  {user_file}")
    print(f"  {item_file}")
    print(f"  {os.path.join(args.data_path, 'user_pretrain.pk')}")
    print(f"  {os.path.join(args.data_path, 'item_pretrain.pk')}")


if __name__ == '__main__':
    train()
