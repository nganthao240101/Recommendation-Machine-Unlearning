"""
Inspect the partition files for a given (part_type, part_num) and print
summary statistics.  Useful for understanding how the paper-style
K-means partitioning has sliced the training data.
"""
import os
import sys
import pickle
import numpy as np


PROJ = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(PROJ, '..', 'data', 'ml-1m')


METHOD_NAMES = {
    1: 'InBP (interaction-based)',
    2: 'UBP (user-based)',
    3: 'Random',
    4: 'IBP (item-based)',
}


def inspect(part_type, part_num, regs='0.01'):
    name = METHOD_NAMES.get(part_type, 'unknown')
    print(f'\n=== {name} (part_type={part_type}, part_num={part_num}) ===')

    # Load partition files
    paths = {
        'C':   os.path.join(DATA, f'C_type-{part_type}_num-{part_num}.pk'),
        'C_U': os.path.join(DATA, f'C_U_type-{part_type}_num-{part_num}.pk'),
        'C_I': os.path.join(DATA, f'C_I_type-{part_type}_num-{part_num}.pk'),
    }
    missing = [p for p, path in paths.items() if not os.path.isfile(path)]
    if missing:
        print(f'  Missing partition files: {missing}')
        print(f'  Train once with: python RecEraser_BPR.py --part_type '
              f'{part_type} --part_num {part_num}')
        return

    with open(paths['C'], 'rb') as f:
        C = pickle.load(f)
    with open(paths['C_U'], 'rb') as f:
        C_U = pickle.load(f)
    with open(paths['C_I'], 'rb') as f:
        C_I = pickle.load(f)

    print(f'  Total shards: {len(C)}')
    n_users = sum(len(shard) for shard in C)
    n_int = sum(len(items) for shard in C for items in shard.values())
    print(f'  Total users in partitions: {n_users}')
    print(f'  Total interactions in partitions: {n_int}')

    print(f'\n  Per-shard statistics:')
    print(f'  {"shard":>6s}  {"users":>7s}  {"items":>7s}  {"interact":>9s}  {"avg/user":>9s}')
    for i, shard in enumerate(C):
        n_u = len(shard)
        all_items = set()
        n_int_shard = 0
        for items in shard.values():
            all_items.update(items)
            n_int_shard += len(items)
        avg = n_int_shard / n_u if n_u else 0
        print(f'  {i:>6d}  {n_u:>7d}  {len(all_items):>7d}  '
              f'{n_int_shard:>9d}  {avg:>9.1f}')

    # Overlap statistics
    print(f'\n  Cross-shard overlap:')
    user_sets = [set(shard.keys()) for shard in C]
    item_sets = [set(i for items in shard.values() for i in items)
                 for shard in C]
    user_overlaps = []
    item_overlaps = []
    for i in range(len(C)):
        for j in range(i+1, len(C)):
            u_overlap = len(user_sets[i] & user_sets[j])
            i_overlap = len(item_sets[i] & item_sets[j])
            user_overlaps.append(u_overlap)
            item_overlaps.append(i_overlap)
    print(f'    avg user overlap (pairwise): '
          f'{np.mean(user_overlaps):.1f}')
    print(f'    avg item overlap (pairwise): '
          f'{np.mean(item_overlaps):.1f}')

    # Histogram of shard sizes
    sizes = [sum(len(items) for items in shard.values()) for shard in C]
    print(f'\n  Shard interaction count: '
          f'min={min(sizes)}, max={max(sizes)}, '
          f'mean={np.mean(sizes):.0f}, std={np.std(sizes):.0f}')


def main():
    ap = __import__('argparse').ArgumentParser()
    ap.add_argument('--part_type', type=int, required=True)
    ap.add_argument('--part_num', type=int, default=5)
    cli = ap.parse_args()
    inspect(cli.part_type, cli.part_num)


if __name__ == '__main__':
    main()
