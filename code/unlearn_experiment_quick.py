"""
Quick test of unlearn experiment with BPR
Reduced hyperparameters for fast testing
"""
import os
import random
import pickle
import subprocess
import re
import time

random.seed(42)

def create_unlearn_data(data_path, dataset, unlearn_type, unlearn_ratio):
    """Create unlearn data"""
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

    if unlearn_type == 'interaction':
        n_unlearn = int(len(all_interactions) * unlearn_ratio)
        unlearned = random.sample(all_interactions, n_unlearn)
    else:
        n_unlearn = int(len(user_items) * unlearn_ratio)
        unlearned_users = random.sample(list(user_items.keys()), n_unlearn)
        unlearned = []
        for u in unlearned_users:
            for i in user_items[u]:
                unlearned.append((u, i))

    return unlearned, unlearn_type


def create_unlearned_train_file(data_path, dataset, unlearned, unlearn_type):
    """Create a new train.txt with unlearned data removed"""
    train_file = os.path.join(data_path, dataset, 'train.txt')
    output_file = os.path.join(data_path, dataset, 'train_unlearned.txt')

    unlearned_set = set()
    unlearned_users = set()
    for u, i in unlearned:
        unlearned_set.add((u, i))
        unlearned_users.add(u)

    with open(train_file) as fin, open(output_file, 'w') as fout:
        for line in fin:
            parts = line.strip().split(' ')
            if len(parts) < 2:
                continue
            uid = int(parts[0])
            items = [int(i) for i in parts[1:]]

            if unlearn_type == 'user' and uid in unlearned_users:
                continue

            new_items = [i for i in items if (uid, i) not in unlearned_set]
            if new_items:
                fout.write(f"{uid} {' '.join(map(str, new_items))}\n")

    return output_file


def run_quick_experiment(part_type, agg_type, unlearn_ratio, num_partitions=10, num_epochs=20):
    """Run a single quick experiment"""
    data_path = '../data/'
    dataset = 'ml-1m'

    print(f"\n[Running] part={part_type}, agg={agg_type}, unlearn={unlearn_ratio*100:.0f}%")

    unlearned, _ = create_unlearn_data(data_path, dataset, 'interaction', unlearn_ratio)
    unlearned_train = create_unlearned_train_file(data_path, dataset, unlearned, 'interaction')

    # Backup and replace train.txt
    train_file = os.path.join(data_path, dataset, 'train.txt')
    backup_file = os.path.join(data_path, dataset, 'train_backup.txt')
    if not os.path.exists(backup_file):
        os.rename(train_file, backup_file)
    os.rename(unlearned_train, train_file)

    # Delete cached partitions
    for f in os.listdir(os.path.join(data_path, dataset)):
        if f.startswith('C_type-') and f.endswith('.pk'):
            os.remove(os.path.join(data_path, dataset, f))

    # Run training
    cmd = [
        'python', 'RecEraser_BPR.py',
        '--dataset', dataset,
        '--part_type', str(part_type),
        '--part_num', str(num_partitions),
        '--epoch', str(num_epochs),
        '--lr', '0.05',
        '--regs', '[0.01]',
        '--batch_size', '256',
        '--agg_type', agg_type,
        '--unlearn_ratio', str(unlearn_ratio),
        '--unlearn_type', 'interaction'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        # Restore original train.txt
        if os.path.exists(train_file):
            os.remove(train_file)
        if os.path.exists(backup_file):
            os.rename(backup_file, train_file)

        # Parse recall
        output = result.stdout + result.stderr
        # Find agg results - pattern: "Epoch X [...] train==[...], recall=[X, Y, Z]"
        agg_pattern = r'Epoch \d+ \[.*?\]: train==\[.*?\], recall=\[([0-9.]+),'
        matches = re.findall(agg_pattern, output)

        if matches:
            best_recall = max(float(m) for m in matches)
            print(f"  [OK] Best Recall@20 = {best_recall:.4f}")
            return best_recall
        else:
            print(f"  [FAIL] No recall found")
            return 0.0

    except subprocess.TimeoutExpired:
        if os.path.exists(train_file):
            os.remove(train_file)
        if os.path.exists(backup_file):
            os.rename(backup_file, train_file)
        print(f"  [TIMEOUT]")
        return 0.0
    except Exception as e:
        if os.path.exists(train_file):
            os.remove(train_file)
        if os.path.exists(backup_file):
            os.rename(backup_file, train_file)
        print(f"  [ERROR] {e}")
        return 0.0


def main():
    results = {}

    # Test parameters (small for quick test)
    part_types = [1, 2, 3, 4]  # InBP, UBP, Random, IBP
    part_names = {1: 'InBP', 2: 'UBP', 3: 'Random', 4: 'IBP'}
    agg_types = ['mean', 'attention']
    unlearn_ratios = [0.1, 0.2, 0.5]
    num_partitions = 10
    num_epochs = 20  # Reduced for quick test

    total = len(agg_types) * len(part_types) * len(unlearn_ratios)
    count = 0

    for agg_type in agg_types:
        for part_type in part_types:
            for ratio in unlearn_ratios:
                count += 1
                key = f"{part_names[part_type]}_{agg_type}_{int(ratio*100)}pct"
                print(f"\n[{count}/{total}] {key}")

                recall = run_quick_experiment(
                    part_type, agg_type, ratio,
                    num_partitions=num_partitions,
                    num_epochs=num_epochs
                )
                results[key] = recall

                # Save intermediate results
                with open('../results/unlearn_quick_results.pkl', 'wb') as f:
                    pickle.dump(results, f)

    # Print final results
    print("\n" + "="*70)
    print("FINAL RESULTS - Quick Test (BPR, ml-1m)")
    print("="*70)
    print(f"{'Config':<35} {'Recall@20':<15}")
    print("-"*70)
    for key, recall in sorted(results.items()):
        print(f"{key:<35} {recall:<15.4f}")

    # Save as text
    with open('../results/unlearn_quick_results.txt', 'w') as f:
        f.write("Unlearn Experiment Results - BPR on ml-1m\n")
        f.write("="*70 + "\n")
        f.write(f"Config: {num_partitions} partitions, {num_epochs} epochs\n\n")
        for key, recall in sorted(results.items()):
            f.write(f"{key:<35} Recall@20 = {recall:.4f}\n")


if __name__ == '__main__':
    main()