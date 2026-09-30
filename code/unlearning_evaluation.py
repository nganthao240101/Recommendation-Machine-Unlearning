"""
Evaluation Script for Unlearning Methods
Có 2 cách đánh giá:

1. RETAINED USERS ONLY:
   - Chỉ đánh giá trên users không bị unlearn
   - Đo lường: Utility cho retained users
   - Câu hỏi: Chất lượng recommendation cho users còn lại có tốt không?

2. FULL TEST SET (RETAINED + UNLEARNED):
   - Đánh giá trên tất cả users (cả retained và unlearned)
   - Đo lường: Unlearning effectiveness
   - Câu hỏi:
     - a) Retained users: Chất lượng có bị ảnh hưởng không?
     - b) Unlearned users: Model CÒN recommend items cũ của họ không?
       - Nếu CÓ = chưa "quên" thực sự (BAD)
       - Nếu KHÔNG = đã quên thực sự (GOOD)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
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
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model, train_data, test_data, n_users, n_items, device, Ks=[10, 20, 50]):
    """Đánh giá model trên một tập test"""
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
        'ndcg': [np.mean(ndcg_log[k]) for k in Ks],
        'n_users_evaluated': sum(1 for u in test_data.keys() if u in test_data and test_data[u])
    }


# ============================================================================
# CÁCH ĐÁNH GIÁ 1: RETAINED USERS ONLY
# ============================================================================

def evaluate_retained_users(model, train_data, test_data, unlearn_users,
                          n_users, n_items, device, Ks=[10, 20, 50]):
    """
    CÁCH 1: Đánh giá trên RETAINED USERS ONLY

    Mục đích: Đo lường Utility cho retained users
    Câu hỏi: Chất lượng recommendation cho users còn lại có tốt không?

    Lọc bỏ unlearned users khỏi test set
    """
    print("\n" + "="*70)
    print("CÁCH 1: ĐÁNH GIÁ TRÊN RETAINED USERS ONLY")
    print("="*70)
    print("Mục đích: Đo lường Utility cho retained users")
    print("Câu hỏi: Chất lượng recommendation cho users còn lại có tốt không?")
    print("-"*70)

    # Lọc chỉ lấy retained users
    test_data_retained = {u: items for u, items in test_data.items() if u not in unlearn_users}
    n_retained = len(test_data_retained)

    print(f"Test users: {len(test_data)} (trước)")
    print(f"Test users: {n_retained} (sau khi lọc unlearned users)")
    print(f"Unlearned users: {len(unlearn_users)}")

    results = evaluate_model(model, train_data, test_data_retained, n_users, n_items, device, Ks)

    print(f"\nKết quả trên RETAINED users:")
    print(f"  Recall@10: {results['recall'][0]:.4f}")
    print(f"  Recall@20: {results['recall'][1]:.4f}")
    print(f"  NDCG@10:  {results['ndcg'][0]:.4f}")

    return results


# ============================================================================
# CÁCH ĐÁNH GIÁ 2: FULL TEST SET (RETAINED + UNLEARNED)
# ============================================================================

def evaluate_full_test_set(model, train_data, test_data, unlearn_users,
                          n_users, n_items, device, Ks=[10, 20, 50]):
    """
    CÁCH 2: Đánh giá trên FULL TEST SET

    Mục đích: Đo lường Unlearning Effectiveness
    Câu hỏi:
      a) Retained users: Chất lượng có bị ảnh hưởng không?
      b) Unlearned users: Model CÒN recommend items cũ của họ không?

    Đánh giá trên tất cả users (cả retained và unlearned)
    """
    print("\n" + "="*70)
    print("CÁCH 2: ĐÁNH GIÁ TRÊN FULL TEST SET (RETAINED + UNLEARNED)")
    print("="*70)
    print("Mục đích: Đo lường Unlearning Effectiveness")
    print("Câu hỏi:")
    print("  a) Retained users: Chất lượng có bị ảnh hưởng không?")
    print("  b) Unlearned users: Model CÒN recommend items cũ của họ không?")
    print("-"*70)

    # Tách test data thành 2 phần
    test_data_retained = {u: items for u, items in test_data.items() if u not in unlearn_users}
    test_data_unlearned = {u: items for u, items in test_data.items() if u in unlearn_users}

    print(f"Test users (tổng): {len(test_data)}")
    print(f"Test users (retained): {len(test_data_retained)}")
    print(f"Test users (unlearned): {len(test_data_unlearned)}")

    # 2a. Đánh giá trên RETAINED users
    print(f"\n--- 2a. Kết quả trên RETAINED users ---")
    results_retained = evaluate_model(model, train_data, test_data_retained, n_users, n_items, device, Ks)
    print(f"  Recall@10: {results_retained['recall'][0]:.4f}")
    print(f"  Recall@20: {results_retained['recall'][1]:.4f}")
    print(f"  NDCG@10:  {results_retained['ndcg'][0]:.4f}")

    # 2b. Đánh giá trên UNLEARNED users
    print(f"\n--- 2b. Kết quả trên UNLEARNED users ---")
    print("  (Model có CÒN recommend đúng items cũ của unlearned users không?)")

    if len(test_data_unlearned) == 0:
        print("  [KHÔNG có unlearned users trong test set]")
        results_unlearned = {'recall': [0, 0, 0], 'ndcg': [0, 0, 0], 'n_users_evaluated': 0}
    else:
        results_unlearned = evaluate_model(model, train_data, test_data_unlearned, n_users, n_items, device, Ks)
        print(f"  Recall@10: {results_unlearned['recall'][0]:.4f}")
        print(f"  Recall@20: {results_unlearned['recall'][1]:.4f}")
        print(f"  NDCG@10:  {results_unlearned['ndcg'][0]:.4f}")

    # 2c. Tổng hợp trên FULL test set
    print(f"\n--- 2c. Kết quả trên FULL test set ---")
    results_full = evaluate_model(model, train_data, test_data, n_users, n_items, device, Ks)
    print(f"  Recall@10: {results_full['recall'][0]:.4f}")
    print(f"  Recall@20: {results_full['recall'][1]:.4f}")
    print(f"  NDCG@10:  {results_full['ndcg'][0]:.4f}")

    # PHÂN TÍCH UNLEARNING EFFECTIVENESS
    print("\n" + "="*70)
    print("PHÂN TÍCH UNLEARNING EFFECTIVENESS:")
    print("="*70)

    if results_unlearned['n_users_evaluated'] > 0:
        unlearn_recall = results_unlearned['recall'][0]
        if unlearn_recall > 0.01:  # Ngưỡng: nếu Recall@10 > 1%
            print(f"  ❌ CHƯA QUÊN: Model vẫn recommend cho unlearned users")
            print(f"     Unlearned Recall@10 = {unlearn_recall:.4f} (> 0.01)")
            print(f"     -> Model vẫn CÓ khả năng predict items của unlearned users")
        else:
            print(f"  ✓ ĐÃ QUÊN: Model không còn recommend cho unlearned users")
            print(f"     Unlearned Recall@10 = {unlearn_recall:.4f} (< 0.01)")
            print(f"     -> Model đã 'quên' items của unlearned users")
    else:
        print("  (Không có unlearned users trong test set)")

    return {
        'full': results_full,
        'retained': results_retained,
        'unlearned': results_unlearned
    }


# ============================================================================
# MAIN
# ============================================================================

def run_evaluation(model_path, dataset='ml-1m', unlearn_ratio=0.1, evaluation_mode='both'):
    """
    Chạy đánh giá với 2 cách

    Args:
        model_path: Đường dẫn đến model (sau khi unlearn)
        dataset: Dataset name
        unlearn_ratio: Tỷ lệ unlearn
        evaluation_mode: 'retained', 'full', hoặc 'both'
    """
    print(f"\n{'='*70}")
    print(f"EVALUATION FOR UNLEARNING")
    print(f"{'='*70}")
    print(f"Dataset: {dataset}")
    print(f"Unlearn ratio: {unlearn_ratio}")
    print(f"Evaluation mode: {evaluation_mode}")

    # Load data
    data = load_data(dataset)
    n_users, n_items = data.n_users, data.n_items
    train_data, test_data = data.train_items, data.test_set

    # Random select unlearn users (giống như khi train)
    import random
    random.seed(42)
    all_users = list(train_data.keys())
    n_unlearn = int(len(all_users) * unlearn_ratio)
    unlearn_users = set(random.sample(all_users, n_unlearn))

    # Load model (placeholder - cần implement theo từng method)
    # Đây là placeholder, cần implement cụ thể cho từng method
    print(f"\n[Lưu ý: Cần implement model loading cụ thể cho từng method]")
    print(f"Unlearn users: {len(unlearn_users)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Chạy đánh giá
    results = {}

    if evaluation_mode in ['retained', 'both']:
        # Cách 1: Retained users only
        # results['retained'] = evaluate_retained_users(...)
        print("\n[Chưa implement - cần thêm model loading]")

    if evaluation_mode in ['full', 'both']:
        # Cách 2: Full test set
        # results['full'] = evaluate_full_test_set(...)
        print("\n[Chưa implement - cần thêm model loading]")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation for Unlearning')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model')
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)
    parser.add_argument('--evaluation_mode', type=str, default='both',
                       choices=['retained', 'full', 'both'],
                       help='retained: đánh giá trên retained users only')
                               # 'full: đánh giá trên full test set')
                               # 'both: cả hai cách')

    args = parser.parse_args()

    run_evaluation(args.model_path, args.dataset, args.unlearn_ratio, args.evaluation_mode)
