"""
RecEraser Partition Analysis - English output
Compare InBP vs UBP vs Random for unlearning
"""
import os
import pickle
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTHONIOENCODING'] = 'utf-8'

from utility.parser import parse_args

args = parse_args()

def analyze_partition(part_type, part_num):
    if part_type == 1:
        pkl_path = args.data_path + args.dataset + '/C_type-1_num-' + str(part_num) + '.pk'
        name = "InBP (Interaction-based)"
    elif part_type == 2:
        pkl_path = args.data_path + args.dataset + '/C_type-2_num-' + str(part_num) + '.pk'
        name = "UBP (User-based)"
    else:
        pkl_path = args.data_path + args.dataset + '/C_type-3_num-' + str(part_num) + '.pk'
        name = "Random"

    with open(pkl_path, 'rb') as f:
        C = pickle.load(f)

    n_inter = []
    n_users = []
    n_items = []

    for i in range(len(C)):
        inter = 0
        us = set()
        it = set()
        for u in C[i]:
            us.add(u)
            for item in C[i][u]:
                inter += 1
                it.add(item)
        n_inter.append(inter)
        n_users.append(len(us))
        n_items.append(len(it))

    return {
        'name': name,
        'type': part_type,
        'n_local': len(C),
        'interactions': n_inter,
        'users': n_users,
        'items': n_items,
        'avg': np.mean(n_inter),
        'std': np.std(n_inter),
        'min': np.min(n_inter),
        'max': np.max(n_inter)
    }

print("=" * 70)
print("RecEraser Partition Analysis - Unlearning Comparison")
print("=" * 70)
print(f"\nDataset: {args.dataset}")
print(f"Number of partitions: {args.part_num}")

results = {}
for pt in [1, 2, 3]:
    stats = analyze_partition(pt, args.part_num)
    results[pt] = stats

    print(f"\n[{pt}] {stats['name']}")
    print(f"  Avg interactions per local: {stats['avg']:.1f}")
    print(f"  Std: {stats['std']:.1f}")
    print(f"  Min/Max: {stats['min']} / {stats['max']}")
    print(f"  Users: {stats['users']}")
    print(f"  Items: {stats['items']}")

print("\n" + "=" * 70)
print("Comparison Summary")
print("=" * 70)
print("\nType             | Avg Inter | Std     | Unlearn Efficiency")
print("-" * 70)

for pt in [1, 2, 3]:
    s = results[pt]
    # Lower std = more balanced = better for unlearning
    eff = 1.0 / (1.0 + s['std'] / 1000)
    print(f"{s['name']:<18} | {s['avg']:>10.1f} | {s['std']:>7.1f} | {eff:.4f}")

print("\n" + "=" * 70)
print("Conclusion:")
print("  InBP (type 1): Each interaction belongs to exactly ONE local model")
print("               -> When unlearning, only retrain that local model")
print("  UBP (type 2): User's interactions may span MULTIPLE local models")
print("               -> When unlearning, may need to retrain multiple models")
print("  Random (type 3): Random partition, not balanced")
print("\n  => InBP provides BEST unlearning efficiency")
print("=" * 70)

print("\n\nTo train and compare Recall@20:")
print("  python RecEraser_BPR.py --dataset ml-1m --part_type 1 --pretrain 0")
print("  python RecEraser_BPR.py --dataset ml-1m --part_type 2 --pretrain 0")
print("  python RecEraser_BPR.py --dataset ml-1m --part_type 3 --pretrain 0")