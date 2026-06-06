"""
Unlearn experiment: compare partition methods and aggregation types
3 dimensions:
- 2 Aggregation: mean vs attention
- 4 Partition: InBP (1), UBP (2), IBP (4), Random (3)
- 3 Unlearn ratios: 10%, 20%, 50%
"""
import os
import sys
import random
import pickle
import subprocess
import time

random.seed(42)

def create_unlearn_data(data_path, dataset, unlearn_type, unlearn_ratio):
    """
    Create unlearn data by removing random interactions or users
    Returns list of (user, item) pairs to unlearn, or list of user_ids to unlearn
    """
    train_file = os.path.join(data_path, dataset, 'train.txt')

    with open(train_file) as f:
        all_interactions = []
        user_items = {}
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) < 2:
                continue
            uid = int(parts[0])
            items = [int(i) for i in parts[1:]]
            user_items[uid] = items
            for i in items:
                all_interactions.append((uid, i))

    print(f"Total interactions: {len(all_interactions)}")
    print(f"Total users: {len(user_items)}")

    if unlearn_type == 'interaction':
        # Unlearn random interactions
        n_unlearn = int(len(all_interactions) * unlearn_ratio)
        unlearned = random.sample(all_interactions, n_unlearn)
        print(f"Unlearning {n_unlearn} interactions ({unlearn_ratio*100}%)")
        return unlearned, 'interaction'
    else:
        # Unlearn random users (and all their interactions)
        n_unlearn = int(len(user_items) * unlearn_ratio)
        unlearned_users = random.sample(list(user_items.keys()), n_unlearn)
        unlearned = []
        for u in unlearned_users:
            for i in user_items[u]:
                unlearned.append((u, i))
        print(f"Unlearning {n_unlearn} users ({unlearn_ratio*100}%) = {len(unlearned)} interactions")
        return unlearned, 'user'


def create_unlearned_train_file(data_path, dataset, unlearned, unlearn_type):
    """Create a new train.txt with unlearned data removed"""
    train_file = os.path.join(data_path, dataset, 'train.txt')
    output_file = os.path.join(data_path, dataset, 'train_unlearned.txt')

    unlearned_set = set()
    unlearned_users = set()
    for u, i in unlearned:
        unlearned_set.add((u, i))
        unlearned_users.add(u)

    kept_interactions = 0
    with open(train_file) as fin, open(output_file, 'w') as fout:
        for line in fin:
            parts = line.strip().split(' ')
            if len(parts) < 2:
                continue
            uid = int(parts[0])
            items = [int(i) for i in parts[1:]]

            if unlearn_type == 'user' and uid in unlearned_users:
                # Skip this user entirely
                continue

            # Filter items
            new_items = [i for i in items if (uid, i) not in unlearned_set]
            if new_items:
                fout.write(f"{uid} {' '.join(map(str, new_items))}\n")
                kept_interactions += len(new_items)

    print(f"Kept {kept_interactions} interactions after unlearning")
    return output_file


def run_experiment(part_type, agg_type, unlearn_type, unlearn_ratio):
    """Run a single experiment"""
    data_path = '../data/'
    dataset = 'ml-1m'

    # Create unlearned data
    print(f"\n{'='*60}")
    print(f"Running: part_type={part_type}, agg={agg_type}, unlearn={unlearn_type}@{unlearn_ratio*100}%")
    print('='*60)

    unlearned, _ = create_unlearn_data(data_path, dataset, unlearn_type, unlearn_ratio)
    unlearned_train = create_unlearned_train_file(data_path, dataset, unlearned, unlearn_type)

    # Backup original train.txt
    train_file = os.path.join(data_path, dataset, 'train.txt')
    backup_file = os.path.join(data_path, dataset, 'train_backup.txt')
    if not os.path.exists(backup_file):
        os.rename(train_file, backup_file)
    os.rename(unlearned_train, train_file)

    # Delete cached partition files to force re-partitioning
    for f in os.listdir(os.path.join(data_path, dataset)):
        if f.startswith('C_type-') and f.endswith('.pk'):
            os.remove(os.path.join(data_path, dataset, f))

    # Run training
    cmd = [
        'python', 'RecEraser_BPR.py',
        '--dataset', dataset,
        '--part_type', str(part_type),
        '--part_num', '10',
        '--epoch', '50',
        '--lr', '0.05',
        '--regs', '[0.01]',
        '--batch_size', '256',
        '--agg_type', agg_type,
        '--unlearn_ratio', str(unlearn_ratio),
        '--unlearn_type', unlearn_type
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    # Restore original train.txt
    if os.path.exists(train_file):
        os.remove(train_file)
    if os.path.exists(backup_file):
        os.rename(backup_file, train_file)

    # Parse recall from output
    output = result.stdout + result.stderr
    recall_lines = [l for l in output.split('\n') if 'recall=' in l and 'Epoch' in l]

    if recall_lines:
        # Get the best recall
        best_recall = 0
        for line in recall_lines:
            try:
                # Extract recall value
                import re
                match = re.search(r'recall=\[([0-9.]+)', line)
                if match:
                    recall = float(match.group(1))
                    if recall > best_recall:
                        best_recall = recall
            except:
                pass

        print(f"Best Recall@20: {best_recall:.4f}")
        return best_recall
    else:
        print("No recall results found")
        return 0.0


def main():
    results = {}

    # Test combinations
    part_types = [1, 2, 3, 4]  # InBP, UBP, Random, IBP
    part_names = {1: 'InBP', 2: 'UBP', 3: 'Random', 4: 'IBP'}
    agg_types = ['mean', 'attention']
    unlearn_ratios = [0.1, 0.2, 0.5]

    for agg_type in agg_types:
        for part_type in part_types:
            for ratio in unlearn_ratios:
                key = f"{part_names[part_type]}_{agg_type}_{ratio}"
                recall = run_experiment(part_type, agg_type, 'interaction', ratio)
                results[key] = recall

                # Save intermediate results
                with open('../results/unlearn_results.pkl', 'wb') as f:
                    pickle.dump(results, f)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for key, recall in sorted(results.items()):
        print(f"{key:40s}: Recall@20 = {recall:.4f}")


if __name__ == '__main__':
    main()