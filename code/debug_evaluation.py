"""
Debug evaluation - Kiểm tra chi tiết before/after evaluation
"""

import os
import sys
import json
import random

PROJ = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(PROJ), 'data')


class SimpleDataLoader:
    def __init__(self, data_dir):
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


def debug_evaluation():
    print("="*70)
    print("DEBUG EVALUATION - CHI TIET BEFORE/AFTER")
    print("="*70)

    # Load data
    data_dir = os.path.join(DATA_DIR, 'ml-1m')
    data = SimpleDataLoader(data_dir)

    train_data = data.train_items
    test_data = data.test_set
    n_users = data.n_users
    n_items = data.n_items

    print(f"\nData: {n_users} users, {n_items} items")
    print(f"Train users: {len(train_data)}")
    print(f"Test users: {len(test_data)}")

    # Select unlearn users (same as in code)
    random.seed(42)
    all_users = list(train_data.keys())
    unlearn_ratio = 0.1
    n_unlearn = int(len(all_users) * unlearn_ratio)
    unlearn_users = set(random.sample(all_users, n_unlearn))

    print(f"\nUnlearn: {n_unlearn} users ({unlearn_ratio*100}%)")

    # Test set analysis
    all_test_users = set(test_data.keys())
    retained_users = all_test_users - unlearn_users

    print(f"\n--- TEST SET ANALYSIS ---")
    print(f"Total test users: {len(all_test_users)}")
    print(f"Unlearn users in test set: {len(unlearn_users & all_test_users)}")
    print(f"Retained users in test set: {len(retained_users)}")

    # Load results if exists
    methods = {
        'FullRetrain': 'results_full_retrain_bprmf_fixed.json',
        'SISA': 'results_sisa_bprmf_fixed.json',
        'Ours': 'results_ours_3components_fixed.json'
    }

    print("\n" + "="*70)
    print("ANALYSIS: BEFORE vs AFTER (ON RETAINED USERS)")
    print("="*70)

    for name, f in methods.items():
        if os.path.exists(f):
            with open(f) as fp:
                d = json.load(fp)

                before = d.get('before', {})
                after = d.get('after', {})

                print(f"\n--- {name} ---")
                print(f"Before: R@10 = {before.get('recall@10', 0):.4f}")
                print(f"After:  R@10 = {after.get('recall@10', 0):.4f}")

                change = after.get('recall@10', 0) - before.get('recall@10', 0)
                print(f"Change: {change:+.4f}")

                if change > 0:
                    print("  ⚠️ AFTER > BEFORE - CO THE CO VAN DE!")
                elif change < -0.01:
                    print("  ⚠️ AFTER < BEFORE - THAT BAI")
                else:
                    print("  ✅ After ≈ Before - ON DINH")

    # KEY INSIGHT
    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("""
Neu After > Before, co the co cac ly do:

1. OVERFITTING SAU KHI XOA DATA:
   - Model retrain tren it data hon
   - Model overfit vao retained users tot hon
   - Day la hien tuong binh thuong

2. CODE CO VAN DE:
   - Before danh gia tren tat ca users (ke ca unlearned)
   - After danh gia tren retained users
   - => So sanh khong chinh xac

3. UNLEARNED USERS VAN CON TRONG TEST SET:
   - Unlearned users van co test data
   - Model van co the predict tot cho users nay
   - Vi embeddings van con thong tin ve user do

DE KHAC PHUC:
- Chi danh gia tren RETAINED users CHO CA TRUOC VA SAU UNLEARN
- Hoac tang so luong unlearn (vd: 50%) de thay su khac biet
""")


def analyze_results_by_unlearn_ratio():
    """Phan tich ket qua khi unlearn voi cac ti le khac nhau"""
    print("\n" + "="*70)
    print("PHAN TICH: UNLEARN RATIO vs PERFORMANCE CHANGE")
    print("="*70)

    print("""
Neu unlearn 10%:
- Full Retrain: After < Before (mat thong tin)
- SISA/Ours: After > Before (overfit)

Neu unlearn 50%:
- Full Retrain: After << Before (mat nhieu thong tin)
- SISA/Ours: After < Before (ro rang hon)

De thay su khac biet ro rang, nen thu unlearn 50%:
""")

    print("\nLenh chay unlearn 50%:")
    print("""
# Unlearn 50% users de thay su khac biet ro rang hon
nohup python method_1_full_retrain.py --model_name BPRMF --max_epochs 20 --early_stopping False --unlearn_ratio 0.5 --output_suffix ratio50 > method1_ratio50.log 2>&1 &
nohup python method_2_sisa.py --model_name BPRMF --max_epochs 20 --unlearn_ratio 0.5 --output_suffix ratio50 > method2_ratio50.log 2>&1 &
nohup python method_3_receraser.py --max_epochs_local 20 --max_epochs_agg 20 --unlearn_ratio 0.5 --output_suffix ratio50 > method3_ratio50.log 2>&1 &
nohup python method_4_ours.py --max_epochs 20 --unlearn_ratio 0.5 --output_suffix ratio50 > method4_ratio50.log 2>&1 &
""")


if __name__ == '__main__':
    debug_evaluation()
    analyze_results_by_unlearn_ratio()
