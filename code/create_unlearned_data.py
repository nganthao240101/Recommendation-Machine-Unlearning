"""
Create unlearned training data by removing a percentage of interactions/users/items.

Usage:
  # Unlearn interaction (random)
  python create_unlearned_data.py --ratio 0.05 --unlearn_type interaction

  # Unlearn item (random)
  python create_unlearned_data.py --ratio 0.05 --unlearn_type item

  # Unlearn user (random)
  python create_unlearned_data.py --ratio 0.05 --unlearn_type user_random

  # Unlearn user with MANY interactions (top ratio%)
  python create_unlearned_data.py --ratio 0.05 --unlearn_type user_high

  # Unlearn user with FEW interactions (bottom ratio%)
  python create_unlearned_data.py --ratio 0.05 --unlearn_type user_low
"""
import os
import sys
import random
import argparse

def create_unlearned_data(input_file, output_file, ratio, unlearn_type='interaction'):
    """Remove data based on unlearn_type."""
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

    print(f"Original data: {len(user_items)} users")

    # Create unlearned data based on type
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
        print(f"  Removed {n_remove} interactions ({ratio*100}%)")

    elif unlearn_type == 'item':
        # Randomly remove ratio% of items completely
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
        print(f"  Removed {n_remove} items ({ratio*100}%)")

    elif unlearn_type == 'user_random':
        # Randomly remove ratio% of users completely
        all_users = list(user_items.keys())
        random.seed(42)
        n_remove = int(len(all_users) * ratio)
        to_remove = set(random.sample(all_users, n_remove))

        unlearned_data = {uid: items for uid, items in user_items.items()
                         if uid not in to_remove}
        print(f"  Removed {n_remove} random users ({ratio*100}%)")

    elif unlearn_type == 'user_high':
        # Remove users with the MOST interactions (top ratio%)
        # Sort users by number of interactions (descending)
        user_interaction_counts = [(uid, len(items)) for uid, items in user_items.items()]
        user_interaction_counts.sort(key=lambda x: x[1], reverse=True)  # Most interactions first

        n_remove = int(len(user_interaction_counts) * ratio)
        to_remove = set([uid for uid, _ in user_interaction_counts[:n_remove]])

        unlearned_data = {uid: items for uid, items in user_items.items()
                         if uid not in to_remove}

        # Show stats
        removed_counts = [count for uid, count in user_interaction_counts[:n_remove]]
        if removed_counts:
            print(f"  Removed {n_remove} users with HIGH interactions ({ratio*100}%)")
            print(f"    Interactions range: {min(removed_counts)} - {max(removed_counts)}")
            print(f"    Mean interactions: {sum(removed_counts)/len(removed_counts):.1f}")

    elif unlearn_type == 'user_low':
        # Remove users with the FEWEST interactions (bottom ratio%)
        # Sort users by number of interactions (ascending)
        user_interaction_counts = [(uid, len(items)) for uid, items in user_items.items()]
        user_interaction_counts.sort(key=lambda x: x[1])  # Fewest interactions first

        n_remove = int(len(user_interaction_counts) * ratio)
        to_remove = set([uid for uid, _ in user_interaction_counts[:n_remove]])

        unlearned_data = {uid: items for uid, items in user_items.items()
                         if uid not in to_remove}

        # Show stats
        removed_counts = [count for uid, count in user_interaction_counts[:n_remove]]
        if removed_counts:
            print(f"  Removed {n_remove} users with LOW interactions ({ratio*100}%)")
            print(f"    Interactions range: {min(removed_counts)} - {max(removed_counts)}")
            print(f"    Mean interactions: {sum(removed_counts)/len(removed_counts):.1f}")

    else:
        raise ValueError(f"Unknown unlearn_type: {unlearn_type}")

    # Write output
    with open(output_file, 'w') as f:
        for uid in sorted(unlearned_data.keys()):
            items = unlearned_data[uid]
            f.write(f"{uid} {' '.join(map(str, items))}\n")

    print(f"  Final users: {len(unlearned_data)}")
    print(f"Saved to: {output_file}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratio', type=float, default=0.05)
    ap.add_argument('--unlearn_type', type=str, default='interaction',
                    choices=['interaction', 'item', 'user_random', 'user_high', 'user_low'])
    ap.add_argument('--data_path', type=str, default='data/ml-1m')
    cli = ap.parse_args()

    input_file = os.path.join(cli.data_path, 'train.txt')
    output_file = os.path.join(cli.data_path,
        f'train_unlearned_{cli.unlearn_type}_r{int(cli.ratio*100):02d}.txt')

    create_unlearned_data(input_file, output_file, cli.ratio, cli.unlearn_type)

if __name__ == '__main__':
    main()
