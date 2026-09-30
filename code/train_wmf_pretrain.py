"""
Script to train WMF and save pretrained embeddings

Cách dùng:
  python train_wmf_pretrain.py --dataset ml-1m --emb_dim 64 --max_epochs 100

Sau đó dùng embeddings này cho RecEraser V2:
  python method_3_receraser_v2.py --dataset ml-1m --emb_dim 64 --partition_type 1
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import random
import json

PROJ = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# DATA LOADER
# ============================================================================

class SimpleDataLoader:
    def __init__(self, data_dir):
        train_file = os.path.join(data_dir, 'train.txt')

        self.n_users, self.n_items = 0, 0
        self.train_items = {}

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
# WMF MODEL
# ============================================================================

class WMF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, users, pos_items, neg_items):
        if users.size(0) == 0:
            return torch.tensor(0.0, device=users.device)
        u_emb = self.user_embedding(users)
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        diff = torch.clamp(pos_scores - neg_scores, -50.0, 50.0)
        loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
        reg_loss = (u_emb.pow(2).sum() + pos_emb.pow(2).sum() + neg_emb.pow(2).sum()) / users.size(0) * 0.01
        return loss + reg_loss


# ============================================================================
# MAIN
# ============================================================================

def train_wmf(dataset='ml-1m', emb_dim=64, max_epochs=100, batch_size=512, lr=0.05):
    print(f"\n{'='*60}")
    print(f"Train WMF for Pretrained Embeddings")
    print(f"{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"Embedding dim: {emb_dim}")
    print(f"Max epochs: {max_epochs}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    data = load_data(dataset)
    n_users, n_items = data.n_users, data.n_items
    train_data = data.train_items

    # Create model
    model = WMF(n_users, n_items, emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Prepare training samples
    print(f"\nPreparing training samples...")
    samples = []
    for user, items in train_data.items():
        for pos_item in items:
            neg_item = random.randint(0, n_items - 1)
            while neg_item in items:
                neg_item = random.randint(0, n_items - 1)
            samples.append((user, pos_item, neg_item))

    print(f"Total samples: {len(samples)}")

    # Train
    print(f"\nTraining WMF...")
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
            loss = model(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{max_epochs}: loss={total_loss/n_batches:.4f}")

    # Get embeddings
    user_emb = model.user_embedding.weight.data.cpu().numpy()
    item_emb = model.item_embedding.weight.data.cpu().numpy()

    print(f"\nUser embeddings shape: {user_emb.shape}")
    print(f"Item embeddings shape: {item_emb.shape}")

    # Save embeddings
    output_dir = os.path.join(PROJ, 'pretrained_embeddings')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f'{dataset}_wmfdim{emb_dim}_ep{max_epochs}.npz')
    np.savez(output_path,
             user_embeddings=user_emb,
             item_embeddings=item_emb,
             n_users=n_users,
             n_items=n_items,
             emb_dim=emb_dim)

    print(f"\nSaved embeddings to: {output_path}")

    # Save metadata
    metadata = {
        'dataset': dataset,
        'emb_dim': emb_dim,
        'max_epochs': max_epochs,
        'n_users': n_users,
        'n_items': n_items,
        'user_emb_shape': list(user_emb.shape),
        'item_emb_shape': list(item_emb.shape)
    }

    metadata_path = os.path.join(output_dir, f'{dataset}_wmfdim{emb_dim}_ep{max_epochs}_meta.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to: {metadata_path}")

    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.05)

    args = parser.parse_args()

    train_wmf(
        dataset=args.dataset,
        emb_dim=args.emb_dim,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
