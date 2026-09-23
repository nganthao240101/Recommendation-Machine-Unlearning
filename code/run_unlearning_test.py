"""
Test Unlearning Effectiveness với nhiều unlearn ratios

Chạy: python run_unlearning_test.py --method sisa --unlearn_ratio 0.3
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import torch
import heapq

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)


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


def evaluate_model(model, train_data, test_data, n_users, n_items, users_to_eval, device, Ks=[10, 20, 50]):
    """Evaluate model chỉ trên một tập users cụ thể"""
    model.eval()
    rec_log = {k: [] for k in Ks}

    with torch.no_grad():
        for user in users_to_eval:
            if user not in test_data or not test_data[user]:
                continue

            user_t = torch.LongTensor([user]).to(device)
            all_items = list(range(n_items))

            scores = []
            for i in range(0, n_items, 256):
                batch_items = torch.LongTensor(all_items[i:i+256]).to(device)
                s = model(user_t, batch_items).cpu().numpy()
                scores.extend(s.tolist())
            scores = np.array(scores)

            # Mask training items
            for item in train_data.get(user, []):
                scores[item] = -np.inf

            rank_list = heapq.nlargest(max(Ks), range(len(scores)), key=scores.__getitem__)
            item_set = set(test_data[user])

            for k in Ks:
                hit_list = rank_list[:k]
                hit_num = len(set(hit_list) & item_set)
                rec = hit_num / len(item_set) if item_set else 0
                rec_log[k].append(rec)

    model.train()
    return {k: np.mean(rec_log[k]) for k in Ks}


def test_unlearning(model_before, model_after, train_data_original, test_data,
                    unlearn_users, retained_users, n_items, device):
    """Test xem unlearning có hiệu quả không"""

    print("\n" + "="*60)
    print("UNLEARNING EFFECTIVENESS TEST")
    print("="*60)

    unlearn_list = list(unlearn_users)
    retained_list = list(retained_users)

    # Test 1: Unlearned users - Recall thay đổi như thế nào?
    print("\n[1] UNLEARNED USERS (should DECREASE)")
    recall_before_unlearned = evaluate_model(
        model_before, train_data_original, test_data,
        len(unlearn_list), n_items, unlearn_list, device
    )
    recall_after_unlearned = evaluate_model(
        model_after, train_data_original, test_data,
        len(unlearn_list), n_items, unlearn_list, device
    )

    print(f"    Before unlearn - R@10: {recall_before_unlearned[10]:.4f}")
    print(f"    After unlearn  - R@10: {recall_after_unlearned[10]:.4f}")
    change = (recall_after_unlearned[10] - recall_before_unlearned[10]) / recall_before_unlearned[10] * 100
    print(f"    Change: {change:+.1f}%")
    print(f"    → {'✓ DECREASED (model forgot!)' if change < -20 else '⚠ Still remembers!'}")

    # Test 2: Retained users - Có ổn định không?
    print("\n[2] RETAINED USERS (should be STABLE)")
    recall_before_retained = evaluate_model(
        model_before, train_data_original, test_data,
        len(retained_list), n_items, retained_list, device
    )
    recall_after_retained = evaluate_model(
        model_after, train_data_original, test_data,
        len(retained_list), n_items, retained_list, device
    )

    print(f"    Before unlearn - R@10: {recall_before_retained[10]:.4f}")
    print(f"    After unlearn  - R@10: {recall_after_retained[10]:.4f}")
    change = (recall_after_retained[10] - recall_before_retained[10]) / recall_before_retained[10] * 100
    print(f"    Change: {change:+.1f}%")
    print(f"    → {'✓ STABLE!' if abs(change) < 5 else '⚠ Changed!'}")

    # Test 3: Overall - Full test set
    print("\n[3] FULL TEST SET")
    all_users = list(unlearn_users | retained_users)
    recall_before_all = evaluate_model(
        model_before, train_data_original, test_data,
        len(all_users), n_items, all_users, device
    )
    recall_after_all = evaluate_model(
        model_after, train_data_original, test_data,
        len(all_users), n_items, all_users, device
    )

    print(f"    Before unlearn - R@10: {recall_before_all[10]:.4f}")
    print(f"    After unlearn  - R@10: {recall_after_all[10]:.4f}")
    retention = recall_after_all[10] / recall_before_all[10] * 100
    print(f"    Retention: {retention:.1f}%")

    return {
        'unlearned_change': change if 'change' in locals() else 0,
        'retained_change': change,
        'retention': retention
    }


def main():
    parser = argparse.ArgumentParser(description='Test Unlearning Effectiveness')
    parser.add_argument('--method', type=str, default='sisa', choices=['sisa', 'ours'])
    parser.add_argument('--unlearn_ratio', type=float, default=0.3)
    parser.add_argument('--dataset', type=str, default='ml-1m')
    args = parser.parse_args()

    print("="*60)
    print(f"UNLEARNING TEST - {args.method.upper()} - Ratio: {args.unlearn_ratio*100:.0f}%")
    print("="*60)

    # Load data
    print("\nLoading data...")
    train_data, test_data, n_users, n_items = load_data(args.dataset)
    print(f"  Users: {n_users}, Items: {n_items}")

    # Select unlearn users
    random.seed(42)
    all_users = list(train_data.keys())
    n_unlearn = int(len(all_users) * args.unlearn_ratio)
    unlearn_users = set(random.sample(all_users, n_unlearn))
    retained_users = set(all_users) - unlearn_users

    print(f"\nUnlearn ratio: {args.unlearn_ratio*100:.0f}%")
    print(f"  Unlearned users: {len(unlearn_users)}")
    print(f"  Retained users: {len(retained_users)}")

    # Giữ train_data gốc
    train_data_original = {u: items.copy() for u, items in train_data.items()}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if args.method == 'sisa':
        from method_2_sisa import SISAMethod, BPRMF

        print("\n--- Training SISA model (BEFORE unlearn) ---")
        method = SISAMethod(BPRMF, n_users, n_items, 64, 8, batch_size=512, lr=0.05, max_epochs=50)
        method.train(train_data, device)
        model_before = method.models[0]  # Lấy một shard làm đại diện

        # Unlearn
        print("\n--- Unlearning ---")
        method.unlearn(unlearn_users, train_data, device, retrain_epochs=20)
        model_after = method.models[0]

    else:  # ours
        from method_4_ours import OursMethod

        print("\n--- Training Ours model (BEFORE unlearn) ---")
        method = OursMethod(n_users, n_items, 64, 8, batch_size=512, lr=0.05, max_epochs=50)
        user_to_shard = method.train(train_data, device)
        model_before = method.models[0]

        # Unlearn
        print("\n--- Unlearning ---")
        method.unlearn(unlearn_users, train_data, device, retrain_epochs=20)
        model_after = method.models[0]

    # Test unlearning effectiveness
    results = test_unlearning(
        model_before, model_after,
        train_data_original, test_data,
        unlearn_users, retained_users, n_items, device
    )

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Unlearn ratio: {args.unlearn_ratio*100:.0f}%")
    print(f"Method: {args.method.upper()}")
    print(f"  - Unlearned users recall change: {results['unlearned_change']:+.1f}%")
    print(f"  - Retained users recall change: {results['retained_change']:+.1f}%")
    print(f"  - Overall retention: {results['retention']:.1f}%")


if __name__ == '__main__':
    main()
