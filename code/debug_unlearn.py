"""Debug: Kiểm tra xem unlearned users có trong test set không"""

import os
import sys
import random
import argparse

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

from utility.loader import load_data


def debug_unlearn(dataset='ml-1m', unlearn_ratio=0.1, unlearn_mode='random', unlearn_user_id=None):
    print(f"\n{'='*60}")
    print(f"DEBUG: UNLEARN USERS ANALYSIS")
    print(f"{'='*60}")
    
    # Load data
    data = load_data(dataset=dataset)
    train_data = data.train_items
    test_data = data.test_set
    
    n_users = data.n_users
    n_items = data.n_items
    
    print(f"\nData: {n_users} users, {n_items} items")
    print(f"Train users: {len(train_data)}")
    print(f"Test users: {len(test_data)}")
    
    # Select unlearn users
    random.seed(42)
    all_users = list(train_data.keys())
    n_unlearn = int(len(all_users) * unlearn_ratio)
    
    if unlearn_mode == 'single' and unlearn_user_id is not None:
        unlearn_users = {unlearn_user_id}
    elif unlearn_mode == 'fewest':
        user_interactions = [(u, len(items)) for u, items in train_data.items()]
        user_interactions.sort(key=lambda x: x[1])
        unlearn_users = set([u for u, _ in user_interactions[:n_unlearn]])
    elif unlearn_mode == 'most':
        user_interactions = [(u, len(items)) for u, items in train_data.items()]
        user_interactions.sort(key=lambda x: x[1], reverse=True)
        unlearn_users = set([u for u, _ in user_interactions[:n_unlearn]])
    else:
        unlearn_users = set(random.sample(all_users, n_unlearn))
    
    print(f"\nUnlearn: {len(unlearn_users)} users ({unlearn_ratio*100}%)")
    print(f"Unlearn mode: {unlearn_mode}")
    
    # Check how many unlearned users are in test set
    unlearned_in_test = 0
    retained_in_test = 0
    total_test_users = len(test_data)
    
    for user in test_data.keys():
        if user in unlearn_users:
            unlearned_in_test += 1
        else:
            retained_in_test += 1
    
    print(f"\n--- Test Set Analysis ---")
    print(f"Total test users: {total_test_users}")
    print(f"Unlearned users IN test set: {unlearned_in_test} ({100*unlearned_in_test/total_test_users:.1f}%)")
    print(f"Retained users IN test set: {retained_in_test} ({100*retained_in_test/total_test_users:.1f}%)")
    
    print(f"\n--- Key Insight ---")
    print(f"After unlearn:")
    print(f"  - {retained_in_test} users: train data CÒN → model predict TỐT")
    print(f"  - {unlearned_in_test} users: train data BỊ XÓA → model predict KÉM")
    print(f"  → Overall performance = weighted avg of 90% TỐT + 10% KÉM")
    
    if retained_in_test > unlearned_in_test:
        print(f"\n→ Vì {retained_in_test} > {unlearned_in_test}, average TĂNG!")
        print(f"  Đây là LÝ DO kết quả After tốt hơn Before!")
    
    return {
        'total_test_users': total_test_users,
        'unlearned_in_test': unlearned_in_test,
        'retained_in_test': retained_in_test
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)
    parser.add_argument('--unlearn_mode', type=str, default='random')
    parser.add_argument('--unlearn_user_id', type=int, default=None)
    args = parser.parse_args()
    
    debug_unlearn(args.dataset, args.unlearn_ratio, args.unlearn_mode, args.unlearn_user_id)
