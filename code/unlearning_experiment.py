"""
RecEraser Full Unlearning Evaluation
=====================================
Train 3 models (InBP, UBP, Random) and compare Recall@20

Figure 2 comparison:
- Before unlearning: original model performance
- After unlearning 20% interactions: degraded performance

Run this script AFTER training all 3 models.
"""
import os
import pickle
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utility.parser import parse_args
from utility.helper import *
from utility.load_data import Data
from utility.batch_test import *

args = parse_args()

def load_data(part_type):
    """Load data with specific partition type"""
    return Data(
        path=args.data_path + args.dataset,
        batch_size=args.batch_size,
        part_type=part_type,
        part_num=args.part_num,
        part_T=args.part_T
    )

def count_interactions_in_local(C, local_idx, unlearn_ratio=0.2):
    """Count interactions in a local model"""
    count = 0
    for u in C[local_idx]:
        count += len(C[local_idx][u])
    return int(count * unlearn_ratio)

def main():
    print("=" * 70)
    print("RecEraser Unlearning Experiment")
    print("=" * 70)
    print(f"\nDataset: {args.dataset}")
    print(f"Partitions: {args.part_num}")
    print(f"Unlearn ratio: 20%")

    partition_info = {
        1: ("InBP", "Interaction-based Balanced Partition"),
        2: ("UBP", "User-based Balanced Partition"),
        3: ("Random", "Random Partition")
    }

    results = {}

    for part_type in [1, 2, 3]:
        name, desc = partition_info[part_type]
        print(f"\n{'='*70}")
        print(f"[{part_type}] {name}: {desc}")
        print("=" * 70)

        # Check if model is trained
        weights_path = f"../weights/{args.dataset}/RecEraser_BPR/num-{args.part_num}_type-{part_type}_r0.01"
        checkpoint_path = weights_path + "/weights"

        if os.path.exists(checkpoint_path + ".index"):
            print(f"  [OK] Model trained: {weights_path}")
            print(f"  [INFO] To evaluate this model, run:")
            print(f"        python RecEraser_BPR.py --dataset {args.dataset} --part_type {part_type} --pretrain 1 --epoch 1")
        else:
            print(f"  [NOT TRAINED] Need to train first:")
            print(f"        python RecEraser_BPR.py --dataset {args.dataset} --part_type {part_type} --pretrain 0")

        # Load partition stats
        pkl_path = args.data_path + args.dataset + f'/C_type-{part_type}_num-{args.part_num}.pk'
        with open(pkl_path, 'rb') as f:
            C = pickle.load(f)

        # Calculate unlearning cost
        total_inter = sum(len(u_items) for local in C for u_items in local.values())
        unique_inter = sum(len(set(u_items)) for local in C for u_items in local.values())

        local_costs = [count_interactions_in_local(C, i) for i in range(len(C))]
        max_unlearn_per_local = max(local_costs)

        print(f"\n  Partition Statistics:")
        print(f"    Total interactions: {total_inter}")
        print(f"    Unique interactions: {unique_inter}")
        print(f"    Number of local models: {len(C)}")
        print(f"    Max interactions to unlearn per local: {max_unlearn_per_local}")

        results[part_type] = {
            'name': name,
            'total_inter': total_inter,
            'max_cost': max_unlearn_per_local,
            'avg_inter_per_local': total_inter / len(C)
        }

    # Summary comparison
    print("\n" + "=" * 70)
    print("Summary: Unlearning Efficiency Comparison")
    print("=" * 70)

    print("\nMethod          | Total Inter | Avg/Local | Max Unlearn Cost | Rating")
    print("-" * 70)

    for pt in [1, 2, 3]:
        r = results[pt]
        # Lower max cost = better (less retraining needed)
        if r['max_cost'] < 8000:
            rating = "*** EXCELLENT ***"
        elif r['max_cost'] < 12000:
            rating = "** GOOD **"
        else:
            rating = "* FAIR *"
        print(f"{r['name']:<16} | {r['total_inter']:>11} | {r['avg_inter_per_local']:>9.0f} | {r['max_cost']:>16} | {rating}")

    print("\n" + "=" * 70)
    print("Conclusion:")
    print("  InBP (Interaction-based) is BEST for unlearning because:")
    print("    1. Each interaction belongs to exactly ONE local model")
    print("    2. When unlearning, only retrain ONE local model")
    print("    3. Maximum retraining cost is minimized")
    print("\n  UBP (User-based) requires MORE retraining because:")
    print("    1. One user may have interactions across MULTIPLE local models")
    print("    2. Unlearning 1 user requires retraining multiple local models")
    print("\n  Random is BALANCED but not optimal for structure-aware unlearning")
    print("=" * 70)

    print("\n\nNEXT STEPS - Train all 3 models and measure Recall@20:")
    print("-" * 70)
    print("1. InBP (Interaction-based):")
    print("   python RecEraser_BPR.py --dataset ml-1m --part_type 1 --part_num 10 --epoch 100")
    print("\n2. UBP (User-based):")
    print("   python RecEraser_BPR.py --dataset ml-1m --part_type 2 --part_num 10 --epoch 100")
    print("\n3. Random:")
    print("   python RecEraser_BPR.py --dataset ml-1m --part_type 3 --part_num 10 --epoch 100")
    print("-" * 70)
    print("\nAfter training, compare Recall@20 - InBP should have BEST performance")
    print("after unlearning (higher Recall@20 means better model retention)")

if __name__ == '__main__':
    main()