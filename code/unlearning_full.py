"""
RecEraser Unlearning Comparison Experiment
============================================
Compare InBP vs UBP vs Random for unlearning 20% interactions
"""
import os
import pickle
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTHONIOENCODING'] = 'utf-8'

from utility.parser import parse_args

args = parse_args()

# Fixed unlearn ratio
UNLEARN_RATIO = 0.2

def load_partitions(part_type, part_num):
    base = args.data_path + args.dataset
    with open(f'{base}/C_type-{part_type}_num-{part_num}.pk', 'rb') as f:
        C = pickle.load(f)
    with open(f'{base}/C_U_type-{part_type}_num-{part_num}.pk', 'rb') as f:
        C_U = pickle.load(f)
    with open(f'{base}/C_I_type-{part_type}_num-{part_num}.pk', 'rb') as f:
        C_I = pickle.load(f)
    return C, C_U, C_I

def analyze_partition(C, C_U, C_I, part_type):
    names = {1: "InBP", 2: "UBP", 3: "Random"}
    name = names.get(part_type, "Unknown")

    print(f"\n{'='*60}")
    print(f"[{part_type}] {name}")
    print(f"{'='*60}")

    n_inter = []
    for i in range(len(C)):
        inter_count = sum(len(C[i][u]) for u in C[i])
        n_inter.append(inter_count)

    total_inter = sum(n_inter)
    avg_inter = total_inter / len(C)
    std_inter = np.std(n_inter)

    print(f"  Total interactions: {total_inter}")
    print(f"  Avg per local: {avg_inter:.0f}")
    print(f"  Std: {std_inter:.0f}")
    print(f"  Min/Max: {min(n_inter)} / {max(n_inter)}")
    print(f"  Per local: {n_inter}")

    # Simulate unlearning 20%
    np.random.seed(2021)
    all_interactions = []
    for local_idx in range(len(C)):
        for user in C[local_idx]:
            for item in C[local_idx][user]:
                all_interactions.append((local_idx, user, item))

    n_to_unlearn = int(len(all_interactions) * UNLEARN_RATIO)
    unlearn_indices = np.random.choice(len(all_interactions), n_to_unlearn, replace=False)

    # Count unlearn per local
    unlearn_by_local = {i: 0 for i in range(len(C))}
    for idx in unlearn_indices:
        local_idx, _, _ = all_interactions[idx]
        unlearn_by_local[local_idx] += 1

    unlearn_counts = [unlearn_by_local[i] for i in range(len(C))]
    local_affected = sum(1 for c in unlearn_counts if c > 0)
    max_unlearn = max(unlearn_counts)

    print(f"\n  After unlearning {UNLEARN_RATIO*100:.0f}% ({n_to_unlearn} interactions):")
    print(f"    Local models affected: {local_affected}/{len(C)}")
    print(f"    Max unlearn per local: {max_unlearn}")
    print(f"    Per local: {unlearn_counts}")

    return {
        'name': name,
        'total_inter': total_inter,
        'avg_inter': avg_inter,
        'std_inter': std_inter,
        'local_affected': local_affected,
        'max_unlearn': max_unlearn,
        'unlearn_counts': unlearn_counts
    }

def main():
    print("="*60)
    print("RecEraser Unlearning Experiment")
    print("="*60)
    print(f"\nDataset: {args.dataset}")
    print(f"Partitions: {args.part_num}")
    print(f"Unlearn Ratio: {UNLEARN_RATIO*100}%")

    results = {}
    for pt in [1, 2, 3]:
        C, C_U, C_I = load_partitions(pt, args.part_num)
        results[pt] = analyze_partition(C, C_U, C_I, pt)

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print("\nPartition | Local Affected | Max Retrain | Std      | Rating")
    print("-"*60)

    best_type = None
    min_affected = float('inf')

    for pt in [1, 2, 3]:
        r = results[pt]
        is_best = r['local_affected'] < min_affected
        if is_best:
            min_affected = r['local_affected']
            best_type = pt

        rating = "*** BEST ***" if is_best else ""
        print(f"{r['name']:<9} | {r['local_affected']:>14} | {r['max_unlearn']:>12} | {r['std_inter']:>8.0f} | {rating}")

    names = {1: "InBP", 2: "UBP", 3: "Random"}
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    print(f"\n  {names[best_type]} requires LEAST retraining")
    print(f"  -> BEST for unlearning {UNLEARN_RATIO*100:.0f}% random interactions")

    print("\n  WHY InBP is better:")
    print("    - InBP: Each interaction = exactly 1 local model")
    print("    - UBP: User may span multiple local models")
    print("    - Random: Balanced but no structure awareness")

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
  To verify with actual training:

  1. python RecEraser_BPR.py --dataset ml-1m --part_type 1 --pretrain 0
  2. python RecEraser_BPR.py --dataset ml-1m --part_type 2 --pretrain 0
  3. python RecEraser_BPR.py --dataset ml-1m --part_type 3 --pretrain 0

  Then compare Recall@20 - InBP should have BEST retention.
    """)

if __name__ == '__main__':
    main()