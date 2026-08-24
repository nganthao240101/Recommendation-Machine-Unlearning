"""
Random User Unlearning - Run multiple times with different random users.

Usage:
  python unlearn_random_users.py --agg_type attention --ratio 5 --part_type 1 --runs 5
"""
import os
import sys
import json
import random
import argparse
import numpy as np

# Add project path
PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

project_root = os.path.dirname(PROJ)
sys.path.insert(0, os.path.join(project_root, 'code'))

from online_unlearn_pytorch import run_one_scenario

METHOD_INFO = {1: 'InP', 2: 'UBP', 3: 'Random', 4: 'IBP'}


def load_train(path):
    """Load train.txt -> dict {uid: [items...]}."""
    data = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            uid = int(parts[0])
            items = [int(x) for x in parts[1:] if x]
            if items:
                data[uid] = items
    return data


def create_random_unlearn_file(input_file, output_file, n_users, seed):
    """Create unlearn file with randomly selected n_users."""
    random.seed(seed)

    # Load data
    user_items = load_train(input_file)
    all_users = list(user_items.keys())

    # Randomly select n_users to unlearn
    n_to_unlearn = min(n_users, len(all_users))
    unlearned_users = set(random.sample(all_users, n_to_unlearn))

    # Create unlearned data
    unlearned_data = {uid: items for uid, items in user_items.items()
                     if uid not in unlearned_users}

    # Save
    with open(output_file, 'w') as f:
        for uid in sorted(unlearned_data.keys()):
            items = unlearned_data[uid]
            f.write(f"{uid} {' '.join(map(str, items))}\n")

    print(f"  Created {output_file} with {n_to_unlearn} unlearned users")
    return unlearned_users


def run_random_unlearn(agg_type, ratio, part_type, part_num, n_runs, seed_start=42):
    """Run unlearn multiple times with different random users."""
    print(f"\n{'='*60}")
    print(f"RANDOM USER UNLEARN")
    print(f"  agg_type: {agg_type}")
    print(f"  ratio: {ratio}% ({ratio/100:.2f})")
    print(f"  partition: {METHOD_INFO[part_type]}")
    print(f"  runs: {n_runs}")
    print(f"{'='*60}")

    data_path = os.path.join(project_root, 'data/ml-1m')
    results_dir = os.path.join(project_root, 'results')

    os.makedirs(results_dir, exist_ok=True)

    # Load original data to get total users
    original_data = load_train(os.path.join(data_path, 'train.txt'))
    total_users = len(original_data)
    n_users_to_unlearn = int(total_users * ratio / 100)

    print(f"\nTotal users: {total_users}")
    print(f"Users to unlearn: {n_users_to_unlearn} ({ratio}%)")

    all_results = []

    for run_idx in range(n_runs):
        seed = seed_start + run_idx
        print(f"\n--- Run {run_idx + 1}/{n_runs} (seed={seed}) ---")

        # Create temporary unlearn file with random users
        temp_file = os.path.join(data_path, f'temp_random_user_r{ratio:02d}_run{run_idx}.txt')
        unlearned_users = create_random_unlearn_file(
            os.path.join(data_path, 'train.txt'),
            temp_file,
            n_users_to_unlearn,
            seed
        )

        # Run unlearn by modifying get_unlearn_entities to use temp file
        try:
            result = run_one_scenario(
                part_num=part_num,
                part_type=part_type,
                agg_type=agg_type,
                unlearn_type=f'random_user_run{run_idx}',
                ratio=ratio / 100,
                regs='0.01'
            )

            if result:
                result['run_idx'] = run_idx + 1
                result['seed'] = seed
                result['n_unlearned_users'] = len(unlearned_users)
                all_results.append(result)
                print(f"  Run {run_idx + 1}: Recall@20 = {result['baseline']['recall20']:.4f} -> {result['online_unlearn']['recall20']:.4f}")

        except Exception as e:
            print(f"  Error in run {run_idx + 1}: {e}")

        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

    if not all_results:
        print("\nNo results collected!")
        return

    # Calculate average
    print(f"\n{'='*60}")
    print(f"AVERAGE RESULTS ({n_runs} runs)")
    print(f"{'='*60}")

    baseline_recalls = [r['baseline']['recall20'] for r in all_results]
    after_recalls = [r['online_unlearn']['recall20'] for r in all_results]
    changes = [(r['online_unlearn']['recall20'] - r['baseline']['recall20']) / r['baseline']['recall20'] * 100 for r in all_results]

    avg_baseline_recall20 = np.mean(baseline_recalls)
    avg_after_recall20 = np.mean(after_recalls)
    avg_change = np.mean(changes)

    print(f"  Baseline Recall@20: {avg_baseline_recall20:.4f} +/- {np.std(baseline_recalls):.4f}")
    print(f"  After Unlearn Recall@20: {avg_after_recall20:.4f} +/- {np.std(after_recalls):.4f}")
    print(f"  Average Change: {avg_change:+.2f}%")

    # Save results
    output_file = os.path.join(results_dir,
        f'unlearn_random_users_p{part_num}_t{part_type}_{agg_type}_r{ratio:02d}_runs{n_runs}.json')

    output_data = {
        'config': {
            'agg_type': agg_type,
            'ratio': ratio,
            'part_type': METHOD_INFO[part_type],
            'n_runs': n_runs,
            'n_users_unlearned': n_users_to_unlearn
        },
        'average': {
            'baseline_recall20_mean': float(avg_baseline_recall20),
            'baseline_recall20_std': float(np.std(baseline_recalls)),
            'after_recall20_mean': float(avg_after_recall20),
            'after_recall20_std': float(np.std(after_recalls)),
            'change_percent_mean': float(avg_change),
            'change_percent_std': float(np.std(changes))
        },
        'individual_runs': all_results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to: {output_file}")
    return output_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agg_type', type=str, default='attention',
                    choices=['attention', 'mean'])
    ap.add_argument('--ratio', type=int, default=5,
                    help='Percentage of users to unlearn (default: 5)')
    ap.add_argument('--part_type', type=int, default=1,
                    help='Partition type: 1=InP, 2=UBP, 3=Random, 4=IBP')
    ap.add_argument('--n_runs', type=int, default=5,
                    help='Number of runs with different random users (default: 5)')
    cli = ap.parse_args()

    run_random_unlearn(
        agg_type=cli.agg_type,
        ratio=cli.ratio,
        part_type=cli.part_type,
        part_num=10,
        n_runs=cli.n_runs,
        seed_start=42
    )


if __name__ == '__main__':
    main()
