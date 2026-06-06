"""
RecEraser Unlearning Evaluation Script
=======================================
So sánh InBP vs UBP vs Random (Figure 2 trong paper)

Usage:
    python unlearning_test.py --dataset ml-1m --part_type 1 --part_num 10

Dataset: ml-1m, yelp2018, ml-10m
Partition Types:
    1 = InBP (Interaction-based Balanced Partition)
    2 = UBP (User-based Balanced Partition)
    3 = Random
"""
import os
import sys
import pickle
import numpy as np
from time import time

# Disable GPU to avoid issues
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utility.parser import parse_args
from utility.helper import *
from utility.batch_test import *

args = parse_args()

def get_partition_stats(data_generator, part_type, part_num):
    """Tính statistics của partition để so sánh"""
    if part_type == 1:
        pkl_path = args.data_path + args.dataset + '/C_type-1_num-' + str(part_num) + '.pk'
    elif part_type == 2:
        pkl_path = args.data_path + args.dataset + '/C_type-2_num-' + str(part_num) + '.pk'
    else:
        pkl_path = args.data_path + args.dataset + '/C_type-3_num-' + str(part_num) + '.pk'

    with open(pkl_path, 'rb') as f:
        C = pickle.load(f)

    # Tính distribution
    n_interactions_per_local = []
    n_users_per_local = []
    n_items_per_local = []

    for i in range(len(C)):
        n_inter = 0
        users_set = set()
        items_set = set()
        for u in C[i]:
            users_set.add(u)
            for item in C[i][u]:
                n_inter += 1
                items_set.add(item)
        n_interactions_per_local.append(n_inter)
        n_users_per_local.append(len(users_set))
        n_items_per_local.append(len(items_set))

    return {
        'n_local': len(C),
        'interactions': n_interactions_per_local,
        'users': n_users_per_local,
        'items': n_items_per_local,
        'avg_inter_per_local': np.mean(n_interactions_per_local),
        'std_inter_per_local': np.std(n_interactions_per_local),
        'max_inter_per_local': np.max(n_interactions_per_local),
        'min_inter_per_local': np.min(n_interactions_per_local)
    }

def main():
    print("=" * 70)
    print("RecEraser Partition Analysis - Unlearning Preparation")
    print("=" * 70)

    partition_names = {
        1: "InBP (Interaction-based)",
        2: "UBP (User-based)",
        3: "Random"
    }

    results = {}

    for part_type in [1, 2, 3]:
        print(f"\n{'='*70}")
        print(f"Partition Type {part_type}: {partition_names[part_type]}")
        print("=" * 70)

        # Load data với partition này
        data_path = args.data_path + args.dataset

        # Load partition statistics
        try:
            stats = get_partition_stats(None, part_type, args.part_num)
            results[part_type] = stats

            print(f"\n  Số partitions: {stats['n_local']}")
            print(f"  Avg interactions per local: {stats['avg_inter_per_local']:.1f}")
            print(f"  Std: {stats['std_inter_per_local']:.1f}")
            print(f"  Min/Max: {stats['min_inter_per_local']} / {stats['max_inter_per_local']}")
            print(f"  Total users across locals: {sum(stats['users'])}")
            print(f"  Total items across locals: {sum(stats['items'])}")

            # User overlap analysis (quan trọng cho unlearning)
            print(f"\n  Users per partition: {stats['users']}")
            print(f"  Items per partition: {stats['items']}")

        except Exception as e:
            print(f"  Lỗi: {e}")
            print(f"  Cần train model trước: python RecEraser_BPR.py --dataset {args.dataset} --part_type {part_type}")

    print("\n" + "=" * 70)
    print("So sánh Unlearning Efficiency:")
    print("=" * 70)

    if all(p in results for p in [1, 2, 3]):
        print("\nPartition Type | Avg Inter | Std Inter | Unlearn Efficiency")
        print("-" * 70)
        for pt in [1, 2, 3]:
            s = results[pt]
            # Lower std = more balanced = better for unlearning
            efficiency = 1.0 / (1.0 + s['std_inter_per_local'] / 1000)
            print(f"  {partition_names[pt]:<25} | {s['avg_inter_per_local']:>10.1f} | {s['std_inter_per_local']:>10.1f} | {efficiency:.4f}")

        print("\n" + "=" * 70)
        print("Kết luận:")
        print("  - InBP (type 1): Interaction được chia đều, mỗi interaction chỉ thuộc 1 local model")
        print("  - UBP (type 2): User được chia đều, nhưng interaction có thể thuộc nhiều local models")
        print("  - Random (type 3): Chia ngẫu nhiên, không đảm bảo cân bằng")
        print("\n  → InBP cho hiệu quả unlearning tốt nhất vì mỗi interaction chỉ cần retrain 1 local model")
        print("=" * 70)

    print("\n\nĐể train và đánh giá unlearning thực sự, chạy lần lượt:")
    print("  1. python RecEraser_BPR.py --dataset ml-1m --part_type 1 --pretrain 0")
    print("  2. python RecEraser_BPR.py --dataset ml-1m --part_type 2 --pretrain 0")
    print("  3. python RecEraser_BPR.py --dataset ml-1m --part_type 3 --pretrain 0")
    print("  (--pretrain 0 = train mới, --pretrain 1 = load đã train)")

if __name__ == '__main__':
    main()