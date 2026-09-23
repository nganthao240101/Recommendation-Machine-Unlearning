"""
Test Unlearning Effectiveness

Các test cần thực hiện:
1. Privacy Leakage Test: Model có "nhớ" thông tin unlearned users không?
2. Prediction Change Test: Predictions của unlearned users có thay đổi không?
3. Retained Users Test: Retained users có bị ảnh hưởng không?
"""

import os
import sys
import json
import random
import numpy as np
import torch
import heapq

PROJ = os.path.dirname(os.path.abspath(__file__))


def load_data(dataset='ml-1m'):
    """Load data"""
    data_dir = os.path.join(os.path.dirname(PROJ), 'data', dataset)

    train_items = {}
    test_set = {}
    n_users, n_items = 0, 0

    with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split()
            uid = int(parts[0])
            items = [int(i) for i in parts[1:]]
            train_items[uid] = items
            n_users = max(n_users, uid + 1)
            n_items = max(n_items, max(items) + 1 if items else 0)

    with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split()
            uid = int(parts[0])
            items = [int(i) for i in parts[1:]]
            test_set[uid] = items

    return train_items, test_set, n_users, n_items


def compute_recall(model, train_data, test_data, n_users, n_items, users_to_test, K=10):
    """Compute recall@K cho một tập users cụ thể"""
    model.eval()
    recalls = []

    with torch.no_grad():
        for user in users_to_test:
            if user not in test_data or not test_data[user]:
                continue

            user_t = torch.LongTensor([user]).to(next(model.parameters()).device)
            all_items = list(range(n_items))

            # Get predictions
            scores = []
            for i in range(0, n_items, 256):
                batch_items = torch.LongTensor(all_items[i:i+256]).to(next(model.parameters()).device)
                s = model.predict(user_t, batch_items).cpu().numpy()
                scores.extend(s.tolist())
            scores = np.array(scores)

            # Mask training items
            for item in train_data.get(user, []):
                scores[item] = -np.inf

            # Top-K
            rank_list = heapq.nlargest(K, range(len(scores)), key=scores.__getitem__)
            hits = len(set(rank_list) & set(test_data[user]))
            recall = hits / len(test_data[user]) if test_data[user] else 0
            recalls.append(recall)

    model.train()
    return np.mean(recalls) if recalls else 0


def test_privacy_leakage(train_data_before, train_data_after, model_before, model_after,
                          unlearn_users, n_items, K=10):
    """
    Test xem model có "nhớ" thông tin unlearned users không

    Cách test:
    1. Lấy predictions của model_before cho unlearned users
    2. Lấy predictions của model_after cho unlearned users
    3. So sánh - nếu khác nhau nhiều → Model đã "quên"
    """
    print("\n" + "="*60)
    print("TEST 1: PRIVACY LEAKAGE")
    print("="*60)
    print("Kiểm tra: Model có 'quên' unlearned users không?")

    model_before.eval()
    model_after.eval()

    changes = []
    for user in list(unlearn_users)[:50]:  # Test 50 users
        if user not in train_data_before or not train_data_before[user]:
            continue

        # Get predictions BEFORE unlearn
        with torch.no_grad():
            user_t = torch.LongTensor([user]).to(next(model_before.parameters()).device)
            all_items = list(range(n_items))

            scores_before = []
            for i in range(0, n_items, 256):
                batch_items = torch.LongTensor(all_items[i:i+256]).to(next(model_before.parameters()).device)
                s = model_before.predict(user_t, batch_items).cpu().numpy()
                scores_before.extend(s.tolist())
            scores_before = np.array(scores_before)

        # Get predictions AFTER unlearn
        with torch.no_grad():
            user_t = torch.LongTensor([user]).to(next(model_after.parameters()).device)

            scores_after = []
            for i in range(0, n_items, 256):
                batch_items = torch.LongTensor(all_items[i:i+256]).to(next(model_after.parameters()).device)
                s = model_after.predict(user_t, batch_items).cpu().numpy()
                scores_after.extend(s.tolist())
            scores_after = np.array(scores_after)

        # Compare top-K items
        top_k_before = set(heapq.nlargest(K, range(len(scores_before)), key=scores_before.__getitem__))
        top_k_after = set(heapq.nlargest(K, range(len(scores_after)), key=scores_after.__getitem__))

        # Jaccard similarity
        if len(top_k_before | top_k_after) > 0:
            jaccard = len(top_k_before & top_k_after) / len(top_k_before | top_k_after)
            changes.append(1 - jaccard)  # Change = 1 - similarity

    model_before.train()
    model_after.train()

    avg_change = np.mean(changes) if changes else 0
    print(f"  Average top-{K} change: {avg_change*100:.1f}%")
    print(f"  (100% = hoàn toàn khác, 0% = hoàn toàn giống)")

    if avg_change > 0.5:
        print(f"  ✓ Model đã 'quên' unlearned users!")
    else:
        print(f"  ✗ Model vẫn 'nhớ' unlearned users!")

    return avg_change


def test_retained_users_stability(train_data, model_before, model_after,
                                  retained_users, n_items, K=10):
    """
    Test xem retained users có bị ảnh hưởng không
    """
    print("\n" + "="*60)
    print("TEST 2: RETAINED USERS STABILITY")
    print("="*60)
    print("Kiểm tra: Retained users có bị ảnh hưởng không?")

    recall_before = compute_recall(model_before, train_data,
                                   {u: train_data[u] for u in retained_users if u in train_data},
                                   len(retained_users), n_items, retained_users, K)

    recall_after = compute_recall(model_after, train_data,
                                  {u: train_data[u] for u in retained_users if u in train_data},
                                  len(retained_users), n_items, retained_users, K)

    print(f"  Recall@{K} (before): {recall_before:.4f}")
    print(f"  Recall@{K} (after):  {recall_after:.4f}")
    print(f"  Change: {(recall_after - recall_before)*100:+.2f}%")

    if abs(recall_after - recall_before) < 0.05:
        print(f"  ✓ Retained users ổn định!")
    else:
        print(f"  ⚠ Retained users có thay đổi!")

    return recall_before, recall_after


def test_unlearned_users_drop(train_data, model_after,
                              unlearn_users, n_items, K=10):
    """
    Test xem unlearned users có bị "quên" không

    Lý thuyết: Sau unlearn, model không còn train data của unlearned users
    → Recall của họ sẽ THẤP vì không có positive examples
    """
    print("\n" + "="*60)
    print("TEST 3: UNLEARNED USERS DROP")
    print("="*60)
    print("Kiểm tra: Unlearned users có bị 'quên' không?")

    # Lấy test data của unlearned users
    test_unlearned = {u: train_data[u] for u in unlearn_users if u in train_data}

    recall = compute_recall(model_after, train_data, test_unlearned,
                            len(unlearn_users), n_items, list(unlearn_users), K)

    print(f"  Recall@{K} của unlearned users (after unlearn): {recall:.4f}")

    # Baseline: random prediction
    random_recall = K / n_items
    print(f"  Random baseline: {random_recall:.4f}")

    if recall < random_recall * 2:
        print(f"  ✓ Unlearned users đã bị 'quên'!")
    else:
        print(f"  ✗ Unlearned users vẫn còn 'nhớ'!")

    return recall


def test_with_higher_ratio(unlearn_ratio=0.3):
    """Test với tỉ lệ unlearn cao hơn"""
    print("\n" + "="*60)
    print(f"TEST VỚI UNLEARN RATIO = {unlearn_ratio*100:.0f}%")
    print("="*60)


if __name__ == '__main__':
    print("="*60)
    print("UNLEARNING EFFECTIVENESS TESTS")
    print("="*60)

    # Load data
    print("\nLoading data...")
    train_data, test_data, n_users, n_items = load_data('ml-1m')
    print(f"  Users: {n_users}, Items: {n_items}")

    # Select unlearn users
    random.seed(42)
    all_users = list(train_data.keys())
    unlearn_ratio = 0.1
    n_unlearn = int(len(all_users) * unlearn_ratio)
    unlearn_users = set(random.sample(all_users, n_unlearn))
    retained_users = set(all_users) - unlearn_users

    print(f"\nUnlearn ratio: {unlearn_ratio*100:.0f}%")
    print(f"  Unlearned users: {len(unlearn_users)}")
    print(f"  Retained users: {len(retained_users)}")

    # Check if we have saved models from previous runs
    print("\n" + "="*60)
    print("ĐỂ CHẠY TEST NÀY, CẦN:")
    print("="*60)
    print("1. Load model BEFORE unlearn")
    print("2. Load model AFTER unlearn")
    print("3. So sánh predictions")
    print()
    print("Bạn cần chạy experiment với việc save model state.")
    print("Hoặc chạy script test trực tiếp trong quá trình training.")
