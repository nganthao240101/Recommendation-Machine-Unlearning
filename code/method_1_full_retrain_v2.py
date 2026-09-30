"""
Method 1: Full Retrain (Oracle) - V2
Với 2 cách đánh giá:
1. RETAINED USERS: Chất lượng cho users còn lại
2. FULL TEST SET: Kiểm tra unlearning effectiveness
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

def train_model(model, train_data, n_users, n_items, device, batch_size=512, lr=0.05, max_epochs=100, verbose=True):
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
            print(f"    Epoch {epoch+1}: loss={total_loss/n_batches:.4f}")
    return model


# ============================================================================
# CÁCH ĐÁNH GIÁ 1: RETAINED USERS ONLY
# ============================================================================

def evaluate_on_retained(model, train_data, test_data, unlearn_users, n_users, n_items, device, Ks=[10, 20, 50]):
    """
    CÁCH 1: Đánh giá trên RETAINED USERS ONLY

    Mục đích: Đo lường Utility cho retained users
    Câu hỏi: Chất lượng recommendation cho users còn lại có tốt không?
    """
    print("\n" + "="*70)
    print("CÁCH 1: ĐÁNH GIÁ TRÊN RETAINED USERS ONLY")
    print("="*70)
    print("Mục đích: Đo lường Utility cho retained users")
    print("Câu hỏi: Chất lượng recommendation cho users còn lại có tốt không?")

    # Lọc chỉ lấy retained users
    test_data_retained = {u: items for u, items in test_data.items() if u not in unlearn_users}

    print(f"\n[Info] Test users (tổng): {len(test_data)}")
    print(f"[Info] Test users (retained): {len(test_data_retained)}")
    print(f"[Info] Unlearned users: {len(unlearn_users)}")

    model.eval()
    pre_log, rec_log, ndcg_log = {k: [] for k in Ks}, {k: [] for k in Ks}, {k: [] for k in Ks}

    with torch.no_grad():
        for user in range(n_users):
            if user not in test_data_retained or not test_data_retained[user]:
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
            rank_list = heapq.nlargest(max(Ks), range(len(scores)), key=scores.__getitem__)
            item_pos = test_data_retained.get(user, [])
            item_set = set(item_pos)
            for k in Ks:
                hit_list = rank_list[:k]
                hit_num = len(set(hit_list) & item_set)
                rec = hit_num / len(item_pos) if len(item_pos) > 0 else 0
                dcg = sum(1.0/np.log2(i+2) for i, item in enumerate(hit_list) if item in item_set)
                idcg = sum(1.0/np.log2(i+2) for i in range(min(len(item_pos), k)))
                ndcg = dcg/idcg if idcg > 0 else 0
                rec_log[k].append(rec)
                ndcg_log[k].append(ndcg)

    model.train()
    results = {
        'recall': [np.mean(rec_log[k]) for k in Ks],
        'ndcg': [np.mean(ndcg_log[k]) for k in Ks]
    }

    print(f"\n[Kết quả] Recall@10: {results['recall'][0]:.4f}, NDCG@10: {results['ndcg'][0]:.4f}")
    return results


# ============================================================================
# CÁCH ĐÁNH GIÁ 2: FULL TEST SET
# ============================================================================

def evaluate_on_full(model, train_data, test_data, unlearn_users, n_users, n_items, device, Ks=[10, 20, 50]):
    """
    CÁCH 2: Đánh giá trên FULL TEST SET (RETAINED + UNLEARNED)

    Mục đích: Đo lường Unlearning Effectiveness
    Câu hỏi:
      a) Retained users: Chất lượng có bị ảnh hưởng không?
      b) Unlearned users: Model CÒN recommend items cũ của họ không?
    """
    print("\n" + "="*70)
    print("CÁCH 2: ĐÁNH GIÁ TRÊN FULL TEST SET (RETAINED + UNLEARNED)")
    print("="*70)
    print("Mục đích: Đo lường Unlearning Effectiveness")
    print("Câu hỏi:")
    print("  a) Retained users: Chất lượng có bị ảnh hưởng không?")
    print("  b) Unlearned users: Model CÒN recommend items cũ của họ không?")

    # Tách test data
    test_data_retained = {u: items for u, items in test_data.items() if u not in unlearn_users}
    test_data_unlearned = {u: items for u, items in test_data.items() if u in unlearn_users}

    print(f"\n[Info] Test users (tổng): {len(test_data)}")
    print(f"[Info] Test users (retained): {len(test_data_retained)}")
    print(f"[Info] Test users (unlearned): {len(test_data_unlearned)}")

    model.eval()
    results = {}

    # 2a. Đánh giá trên RETAINED users
    print(f"\n--- 2a. Kết quả trên RETAINED users ---")
    rec_log, ndcg_log = {k: [] for k in Ks}, {k: [] for k in Ks}
    with torch.no_grad():
        for user in test_data_retained:
            if not test_data_retained[user]:
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
            rank_list = heapq.nlargest(max(Ks), range(len(scores)), key=scores.__getitem__)
            item_pos = test_data_retained.get(user, [])
            item_set = set(item_pos)
            for k in Ks:
                hit_list = rank_list[:k]
                hit_num = len(set(hit_list) & item_set)
                rec = hit_num / len(item_pos) if len(item_pos) > 0 else 0
                dcg = sum(1.0/np.log2(i+2) for i, item in enumerate(hit_list) if item in item_set)
                idcg = sum(1.0/np.log2(i+2) for i in range(min(len(item_pos), k)))
                ndcg = dcg/idcg if idcg > 0 else 0
                rec_log[k].append(rec)
                ndcg_log[k].append(ndcg)

    results['retained'] = {
        'recall': [np.mean(rec_log[k]) for k in Ks],
        'ndcg': [np.mean(ndcg_log[k]) for k in Ks]
    }
    print(f"  Recall@10: {results['retained']['recall'][0]:.4f}, NDCG@10: {results['retained']['ndcg'][0]:.4f}")

    # 2b. Đánh giá trên UNLEARNED users
    print(f"\n--- 2b. Kết quả trên UNLEARNED users ---")
    print("  (Model có CÒN recommend đúng items cũ của unlearned users không?)")

    if len(test_data_unlearned) == 0:
        print("  [KHÔNG có unlearned users trong test set]")
        results['unlearned'] = {'recall': [0, 0, 0], 'ndcg': [0, 0, 0]}
    else:
        rec_log, ndcg_log = {k: [] for k in Ks}, {k: [] for k in Ks}
        with torch.no_grad():
            for user in test_data_unlearned:
                if not test_data_unlearned[user]:
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
                rank_list = heapq.nlargest(max(Ks), range(len(scores)), key=scores.__getitem__)
                item_pos = test_data_unlearned.get(user, [])
                item_set = set(item_pos)
                for k in Ks:
                    hit_list = rank_list[:k]
                    hit_num = len(set(hit_list) & item_set)
                    rec = hit_num / len(item_pos) if len(item_pos) > 0 else 0
                    dcg = sum(1.0/np.log2(i+2) for i, item in enumerate(hit_list) if item in item_set)
                    idcg = sum(1.0/np.log2(i+2) for i in range(min(len(item_pos), k)))
                    ndcg = dcg/idcg if idcg > 0 else 0
                    rec_log[k].append(rec)
                    ndcg_log[k].append(ndcg)

        results['unlearned'] = {
            'recall': [np.mean(rec_log[k]) for k in Ks],
            'ndcg': [np.mean(ndcg_log[k]) for k in Ks]
        }
        print(f"  Recall@10: {results['unlearned']['recall'][0]:.4f}, NDCG@10: {results['unlearned']['ndcg'][0]:.4f}")

        # PHÂN TÍCH UNLEARNING
        unlearn_recall = results['unlearned']['recall'][0]
        print(f"\n[PHÂN TÍCH UNLEARNING EFFECTIVENESS]")
        if unlearn_recall > 0.01:
            print(f"  ❌ CHƯA QUÊN: Model vẫn recommend cho unlearned users")
            print(f"     Unlearned Recall@10 = {unlearn_recall:.4f} (> 0.01)")
            print(f"     -> Model vẫn CÓ khả năng predict items của unlearned users")
        else:
            print(f"  ✓ ĐÃ QUÊN: Model không còn recommend cho unlearned users")
            print(f"     Unlearned Recall@10 = {unlearn_recall:.4f} (< 0.01)")
            print(f"     -> Model đã 'quên' items của unlearned users")

    # 2c. Tổng hợp trên FULL test set
    print(f"\n--- 2c. Kết quả trên FULL test set ---")
    rec_log, ndcg_log = {k: [] for k in Ks}, {k: [] for k in Ks}
    with torch.no_grad():
        for user in test_data:
            if not test_data[user]:
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
            rank_list = heapq.nlargest(max(Ks), range(len(scores)), key=scores.__getitem__)
            item_pos = test_data.get(user, [])
            item_set = set(item_pos)
            for k in Ks:
                hit_list = rank_list[:k]
                hit_num = len(set(hit_list) & item_set)
                rec = hit_num / len(item_pos) if len(item_pos) > 0 else 0
                dcg = sum(1.0/np.log2(i+2) for i, item in enumerate(hit_list) if item in item_set)
                idcg = sum(1.0/np.log2(i+2) for i in range(min(len(item_pos), k)))
                ndcg = dcg/idcg if idcg > 0 else 0
                rec_log[k].append(rec)
                ndcg_log[k].append(ndcg)

    results['full'] = {
        'recall': [np.mean(rec_log[k]) for k in Ks],
        'ndcg': [np.mean(ndcg_log[k]) for k in Ks]
    }
    print(f"  Recall@10: {results['full']['recall'][0]:.4f}, NDCG@10: {results['full']['ndcg'][0]:.4f}")

    model.train()
    return results


# ============================================================================
# MAIN
# ============================================================================

def run_full_retrain_v2(dataset='ml-1m', emb_dim=64, max_epochs=100,
                       unlearn_ratio=0.1, unlearn_mode='random',
                       unlearn_user_id=None, output_suffix=''):
    print(f"\n{'='*70}")
    print(f"METHOD 1: FULL RETRAIN V2 - 2 CÁCH ĐÁNH GIÁ")
    print(f"{'='*70}")

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

    if unlearn_mode == 'single' and unlearn_user_id is not None:
        unlearn_users = {unlearn_user_id}
    else:
        unlearn_users = set(random.sample(all_users, n_unlearn))

    print(f"\nUnlearn: {len(unlearn_users)} users ({unlearn_ratio*100}%)")

    # Train BEFORE
    print(f"\n{'='*70}")
    print("TRAINING BEFORE UNLEARN")
    print(f"{'='*70}")
    model_before = BPRMF(n_users, n_items, emb_dim).to(device)
    train_model(model_before, train_data, n_users, n_items, device, max_epochs=max_epochs)

    # ĐÁNH GIÁ BEFORE
    print(f"\n{'='*70}")
    print("EVALUATION BEFORE UNLEARN")
    print(f"{'='*70}")

    # Cách 1: Retained users
    results_before_retained = evaluate_on_retained(
        model_before, train_data, test_data, unlearn_users,
        n_users, n_items, device)

    # Cách 2: Full test set
    results_before_full = evaluate_on_full(
        model_before, train_data, test_data, unlearn_users,
        n_users, n_items, device)

    # UNLEARN (Full Retrain)
    print(f"\n{'='*70}")
    print("UNLEARN (FULL RETRAIN)")
    print(f"{'='*70}")

    # Lọc bỏ unlearned users khỏi train data
    train_data_after = {u: items for u, items in train_data.items() if u not in unlearn_users}

    t0 = time.time()
    model_after = BPRMF(n_users, n_items, emb_dim).to(device)
    train_model(model_after, train_data_after, n_users, n_items, device, max_epochs=max_epochs)
    unlearn_time = time.time() - t0

    # ĐÁNH GIÁ AFTER
    print(f"\n{'='*70}")
    print("EVALUATION AFTER UNLEARN")
    print(f"{'='*70}")

    # Cách 1: Retained users
    results_after_retained = evaluate_on_retained(
        model_after, train_data_after, test_data, unlearn_users,
        n_users, n_items, device)

    # Cách 2: Full test set
    results_after_full = evaluate_on_full(
        model_after, train_data_after, test_data, unlearn_users,
        n_users, n_items, device)

    # TỔNG HỢP
    print(f"\n{'='*70}")
    print("TỔNG HỢP KẾT QUẢ")
    print(f"{'='*70}")

    print(f"\n[1] ĐÁNH GIÁ TRÊN RETAINED USERS:")
    print(f"    Before: Recall@10 = {results_before_retained['recall'][0]:.4f}, NDCG@10 = {results_before_retained['ndcg'][0]:.4f}")
    print(f"    After:  Recall@10 = {results_after_retained['recall'][0]:.4f}, NDCG@10 = {results_after_retained['ndcg'][0]:.4f}")

    print(f"\n[2] ĐÁNH GIÁ TRÊN UNLEARNED USERS (Unlearning Effectiveness):")
    print(f"    Before: Recall@10 = {results_before_full['unlearned']['recall'][0]:.4f}")
    print(f"    After:  Recall@10 = {results_after_full['unlearned']['recall'][0]:.4f}")

    print(f"\n[3] ĐÁNH GIÁ TRÊN FULL TEST SET:")
    print(f"    Before: Recall@10 = {results_before_full['full']['recall'][0]:.4f}")
    print(f"    After:  Recall@10 = {results_after_full['full']['recall'][0]:.4f}")

    print(f"\n[4] UNLEARN TIME: {unlearn_time:.2f}s")

    # Save results
    results = {
        'method': 'FullRetrain_V2',
        'dataset': dataset,
        'unlearn_ratio': unlearn_ratio,
        'before': {
            'retained': results_before_retained,
            'full': results_before_full
        },
        'after': {
            'retained': results_after_retained,
            'full': results_after_full
        },
        'unlearn_time': unlearn_time
    }

    output_path = f'results_fullretrain_v2_{output_suffix}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)
    parser.add_argument('--unlearn_mode', type=str, default='random')
    parser.add_argument('--unlearn_user_id', type=int, default=None)
    parser.add_argument('--output_suffix', type=str, default='')
    args = parser.parse_args()

    run_full_retrain_v2(
        dataset=args.dataset,
        emb_dim=args.emb_dim,
        max_epochs=args.max_epochs,
        unlearn_ratio=args.unlearn_ratio,
        unlearn_mode=args.unlearn_mode,
        unlearn_user_id=args.unlearn_user_id,
        output_suffix=args.output_suffix
    )
