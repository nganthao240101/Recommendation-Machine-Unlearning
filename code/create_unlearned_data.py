"""
Create unlearned training data by removing a percentage of interactions.

Usage:
  python create_unlearned_data.py --ratio 0.05 --unlearn_type interaction
"""
import os
import sys
import random
import argparse

def create_unlearned_data(input_file, output_file, ratio, unlearn_type='interaction'):
    """Remove ratio% of interactions from training data."""
    # Read original data
    user_items = {}
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            uid = int(parts[0])
            items = [int(x) for x in parts[1:] if x]
            user_items[uid] = items

    # Create unlearned data
    if unlearn_type == 'interaction':
        # Randomly remove ratio% of interactions
        all_interactions = []
        for uid, items in user_items.items():
            for item in items:
                all_interactions.append((uid, item))

        random.seed(42)
        n_remove = int(len(all_interactions) * ratio)
        to_remove = set(random.sample(all_interactions, n_remove))

        unlearned_data = {}
        for uid, items in user_items.items():
            remaining = [item for item in items if (uid, item) not in to_remove]
            if remaining:
                unlearned_data[uid] = remaining

    elif unlearn_type == 'user':
        # Remove ratio% of users completely
        all_users = list(user_items.keys())
        random.seed(42)
        n_remove = int(len(all_users) * ratio)
        to_remove = set(random.sample(all_users, n_remove))

        unlearned_data = {uid: items for uid, items in user_items.items()
                         if uid not in to_remove}

    elif unlearn_type == 'item':
        # Remove ratio% of items completely
        all_items = set()
        for items in user_items.values():
            all_items.update(items)
        all_items = list(all_items)
        random.seed(42)
        n_remove = int(len(all_items) * ratio)
        to_remove = set(random.sample(all_items, n_remove))

        unlearned_data = {}
        for uid, items in user_items.items():
            remaining = [item for item in items if item not in to_remove]
            if remaining:
                unlearned_data[uid] = remaining

    # Write output
    with open(output_file, 'w') as f:
        for uid in sorted(unlearned_data.keys()):
            items = unlearned_data[uid]
            f.write(f"{uid} {' '.join(map(str, items))}\n")

    print(f"Created {output_file}")
    print(f"  Original users: {len(user_items)}")
    print(f"  Unlearned users: {len(unlearned_data)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratio', type=float, default=0.05)
    ap.add_argument('--unlearn_type', type=str, default='interaction',
                    choices=['interaction', 'user', 'item'])
    ap.add_argument('--data_path', type=str,
                    default='data/ml-1m')
    cli = ap.parse_args()

    input_file = os.path.join(cli.data_path, 'train.txt')
    output_file = os.path.join(cli.data_path,
        f'train_unlearned_{cli.unlearn_type}_r{int(cli.ratio*100):02d}.txt')

    create_unlearned_data(input_file, output_file, cli.ratio, cli.unlearn_type)

if __name__ == '__main__':
    main()
