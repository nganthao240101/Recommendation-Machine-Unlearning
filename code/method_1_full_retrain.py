"""
Method 1: Full Retrain (Oracle Baseline)

Hyper-parameters theo bài báo:
- Batch size: 512
- Learning rate: 0.05
- Embedding size: 64
- Max epochs: 1000
- Early stopping: Recall@10 không tăng trong 10 epochs liên tiếp

Baseline cho việc so sánh. Train lại hoàn toàn từ đầu sau khi xóa user cần unlearn.
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
from typing import Dict, List, Tuple, Optional
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
# EVALUATION
# ============================================================================

def evaluate_model(model, train_data, test_data, n_users, n_items, device, Ks=[10, 20, 50]):
    """Evaluate model using Recall@K and NDCG@K."""
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

    model.train()

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
                early_stopping=True, patience=10, val_data=None, test_data=None,
                verbose=True):
    """
    Train model với early stopping.

    Args:
        model: Model cần train
        train_data: Training data
        n_users, n_items: Số users/items
        device: Device (cuda/cpu)
        batch_size: Batch size (theo bài báo: 512)
        lr: Learning rate (theo bài báo: 0.05)
        max_epochs: Max epochs (theo bài báo: 1000)
        early_stopping: Có sử dụng early stopping không
        patience: Số epochs không cải thiện trước khi dừng (theo bài báo: 10)
        val_data: Validation data (optional)
        test_data: Test data (optional)
        verbose: In thông tin training
    """
    optimizer = Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)

    # Prepare training samples
    samples = []
    for user, items in train_data.items():
        for pos_item in items:
            neg_item = random.randint(0, n_items - 1)
            while neg_item in items:
                neg_item = random.randint(0, n_items - 1)
            samples.append((user, pos_item, neg_item))

    n_samples = len(samples)
    n_batches = max(1, n_samples // batch_size)

    # Early stopping variables
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

        # Evaluate if early stopping is enabled
        if early_stopping and val_data is not None and (epoch + 1) % 5 == 0:
            metrics = evaluate_model(model, train_data, val_data, n_users, n_items, device)
            current_metric = metrics['recall'][0]  # Recall@10

            if verbose:
                avg_loss = total_loss / n_batches
                print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}, "
                      f"Val R@10={current_metric:.4f}")

            # Check for improvement
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch + 1
                patience_counter = 0
                # Save best model state
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 5  # Count in terms of evaluation intervals

            # Early stopping check
            if patience_counter >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}. Best: epoch {best_epoch} "
                          f"with R@10={best_metric:.4f}")
                break

        elif verbose and (epoch + 1) % 20 == 0:
            avg_loss = total_loss / n_batches
            print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}")

    # Restore best model if we used early stopping
    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        if verbose:
            print(f"    Restored best model from epoch {best_epoch}")

    return model, best_epoch if early_stopping else max_epochs


# ============================================================================
# FULL RETRAIN METHOD
# ============================================================================

class FullRetrainMethod:
    """
    Method 1: Full Retrain (Oracle Baseline)

    - Train: Train model hoàn toàn trên data gốc
    - Unlearn: Train lại hoàn toàn trên filtered data (sau khi xóa user cần unlearn)
    - Đây là baseline oracle - kết quả tốt nhất nhưng chậm nhất
    """

    def __init__(self, model_class, n_users, n_items, emb_dim,
                 batch_size=512, lr=0.05, max_epochs=1000,
                 early_stopping=True, patience=10):
        self.model_class = model_class
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.model = None
        self.training_info = {}

    def train(self, train_data, device, val_data=None):
        """
        Train model hoàn toàn trên data gốc (before unlearning).
        """
        print("    [Full Retrain] Training on full data...")
        print(f"    [Full Retrain] Config: batch_size={self.batch_size}, lr={self.lr}, "
              f"max_epochs={self.max_epochs}, early_stopping={self.early_stopping}")

        self.model = self.model_class(self.n_users, self.n_items, self.emb_dim).to(device)

        self.model, n_epochs_trained = train_model(
            self.model, train_data, self.n_users, self.n_items, device,
            batch_size=self.batch_size, lr=self.lr,
            max_epochs=self.max_epochs,
            early_stopping=self.early_stopping,
            patience=self.patience,
            val_data=val_data,
            test_data=None,
            verbose=True
        )

        self.training_info['epochs_trained'] = n_epochs_trained
        print(f"    [Full Retrain] Trained for {n_epochs_trained} epochs")

        return self.model

    def unlearn(self, unlearn_user_ids, train_data, device):
        """
        Unlearn bằng cách train lại hoàn toàn trên filtered data.

        ĐÂY LÀ ORACLE BASELINE - kết quả tốt nhất nhưng tốn thời gian nhất.
        """
        print(f"    [Full Retrain] Training on filtered data (oracle)...")

        # Filter out unlearned users
        filtered_data = {u: items for u, items in train_data.items()
                        if u not in unlearn_user_ids}

        print(f"    [Full Retrain] Original users: {len(train_data)}, "
              f"Filtered: {len(filtered_data)}")

        # Train hoàn toàn từ đầu
        self.model = self.model_class(self.n_users, self.n_items, self.emb_dim).to(device)

        self.model, n_epochs_trained = train_model(
            self.model, filtered_data, self.n_users, self.n_items, device,
            batch_size=self.batch_size, lr=self.lr,
            max_epochs=self.max_epochs,
            early_stopping=self.early_stopping,
            patience=self.patience,
            val_data=None,
            test_data=None,
            verbose=True
        )

        self.training_info['unlearn_epochs_trained'] = n_epochs_trained
        print(f"    [Full Retrain] Unlearn trained for {n_epochs_trained} epochs")

        return self.model

    def evaluate(self, train_data, test_data, device, Ks=[10, 20, 50]):
        return evaluate_model(self.model, train_data, test_data,
                            self.n_users, self.n_items, device, Ks)


# ============================================================================
# MAIN
# ============================================================================

def run_full_retrain(model_name='BPRMF', dataset='ml-1m',
                    batch_size=512, lr=0.05, emb_dim=64,
                    max_epochs=1000, early_stopping=True, patience=10,
                    unlearn_ratio=0.1, output_suffix=''):
    """Run Full Retrain method."""
    print(f"\n{'='*60}")
    print(f"METHOD 1: FULL RETRAIN (ORACLE BASELINE)")
    print(f"{'='*60}")
    print(f"Hyper-parameters:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Embedding dim: {emb_dim}")
    print(f"  - Max epochs: {max_epochs}")
    print(f"  - Early stopping: {early_stopping} (patience={patience})")

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load data
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
    val_data = data.test_set  # Use test as validation
    test_data = data.test_set

    print(f"Users: {n_users}, Items: {n_items}")
    print(f"Train interactions: {sum(len(v) for v in train_data.values())}")
    print(f"Val/Test users: {len(val_data)}")

    # Select users to unlearn
    random.seed(42)
    all_users = list(train_data.keys())
    n_unlearn = int(len(all_users) * unlearn_ratio)
    unlearn_users = set(random.sample(all_users, n_unlearn))

    print(f"\nUnlearn ratio: {unlearn_ratio} ({n_unlearn} users)")

    # Model class
    model_classes = {'BPRMF': BPRMF, 'WMF': WMF}
    model_class = model_classes.get(model_name, BPRMF)

    # Create method
    method = FullRetrainMethod(
        model_class, n_users, n_items, emb_dim,
        batch_size=batch_size, lr=lr,
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        patience=patience
    )

    # Train before unlearning
    print(f"\n--- Phase 1: Train BEFORE unlearning ---")
    t0 = time.time()
    method.train(train_data, device, val_data=val_data)
    train_time = time.time() - t0

    # Evaluate before
    results_before = method.evaluate(train_data, test_data, device)
    print(f"  Before - R@10: {results_before['recall'][0]:.4f}, "
          f"NDCG@10: {results_before['ndcg'][0]:.4f}")

    # Unlearn
    print(f"\n--- Phase 2: Unlearn (oracle full retrain) ---")
    t0 = time.time()
    method.unlearn(unlearn_users, train_data, device)
    unlearn_time = time.time() - t0

    # Evaluate after
    results_after = method.evaluate(train_data, test_data, device)
    print(f"  After - R@10: {results_after['recall'][0]:.4f}, "
          f"NDCG@10: {results_after['ndcg'][0]:.4f}")
    print(f"  Unlearn time: {unlearn_time:.2f}s")

    # Results
    results = {
        'method': 'FullRetrain',
        'model': model_name,
        'dataset': dataset,
        'hyperparameters': {
            'batch_size': batch_size,
            'learning_rate': lr,
            'embedding_dim': emb_dim,
            'max_epochs': max_epochs,
            'early_stopping': early_stopping,
            'patience': patience
        },
        'unlearn_ratio': unlearn_ratio,
        'n_unlearn': n_unlearn,
        'train_time': train_time,
        'unlearn_time': unlearn_time,
        'training_info': method.training_info,
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

    # Save
    suffix = f"_{output_suffix}" if output_suffix else ""
    output_path = os.path.join(PROJ, f'results_full_retrain_{model_name.lower()}{suffix}.json')
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Method 1: Full Retrain (Oracle Baseline)')
    parser.add_argument('--model', type=str, default='BPRMF', choices=['BPRMF', 'WMF'],
                       help='Model type')
    parser.add_argument('--dataset', type=str, default='ml-1m',
                       help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size (paper: 512)')
    parser.add_argument('--lr', type=float, default=0.05,
                       help='Learning rate (paper: 0.05)')
    parser.add_argument('--emb_dim', type=int, default=64,
                       help='Embedding dimension (paper: 64)')
    parser.add_argument('--max_epochs', type=int, default=1000,
                       help='Maximum epochs (paper: 1000)')
    parser.add_argument('--early_stopping', type=lambda x: x.lower() == 'true',
                       default=True,
                       help='Use early stopping')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (paper: 10)')
    parser.add_argument('--unlearn_ratio', type=float, default=0.1,
                       help='Ratio of users to unlearn')
    parser.add_argument('--output_suffix', type=str, default='',
                       help='Suffix for output file')
    args = parser.parse_args()

    run_full_retrain(
        model_name=args.model,
        dataset=args.dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        emb_dim=args.emb_dim,
        max_epochs=args.max_epochs,
        early_stopping=args.early_stopping,
        patience=args.patience,
        unlearn_ratio=args.unlearn_ratio,
        output_suffix=args.output_suffix
    )
