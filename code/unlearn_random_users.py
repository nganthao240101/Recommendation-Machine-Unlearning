"""
Random User Unlearning - Run multiple times with different random users.

Usage:
  python unlearn_random_users.py --agg_type attention --ratio 5 --part_type 1 --n_runs 5
"""
import os
import sys
import json
import random
import argparse
import numpy as np
import pickle
import torch
from torch.optim import Adagrad
from time import time

# Project paths
PROJ = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(PROJ)
sys.path.insert(0, PROJ)

# Set data path
os.environ['RECUNLEARN_DATA_PATH'] = os.path.join(project_root, 'data/')
os.environ['RECUNLEARN_DATASET'] = 'ml-1m'

from RecEraser_BPR_pytorch import RecEraserBPR, test_torch

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


def find_affected_shards(C, unlearn_type, unlearned_uids, unlearned_iids):
    """Return list of shard indices containing unlearned entities."""
    affected = []
    for i, shard in enumerate(C):
        shard_users = set(shard.keys())
        shard_items = set()
        for items in shard.values():
            shard_items.update(items)

        if unlearn_type == 'user':
            hit = bool(shard_users & unlearned_uids)
        elif unlearn_type == 'item':
            hit = bool(shard_items & unlearned_iids)
        else:
            hit = False
            for u, items in shard.items():
                if u in unlearned_uids:
                    hit = True
                    break
                for item_id in items:
                    if item_id in unlearned_iids:
                        hit = True
                        break
                if hit:
                    break
        if hit:
            affected.append(i)
    return affected


def filter_shard_data(shard, unlearn_type, unlearned_uids, unlearned_iids):
    """Return new shard dict with unlearned entities removed."""
    if unlearn_type == 'user':
        return {u: items for u, items in shard.items() if u not in unlearned_uids}
    elif unlearn_type == 'item':
        return {u: [i for i in items if i not in unlearned_iids]
                for u, items in shard.items()}
    else:
        result = {}
        for u, items in shard.items():
            if u in unlearned_uids:
                continue
            filtered = [i for i in items if i not in unlearned_iids]
            if filtered:
                result[u] = filtered
        return result


def retrain_shard(model, shard_id, shard_data, lr, batch_size, device, n_epochs=1):
    """Retrain ONE shard on filtered data."""
    if not shard_data:
        return 0.0

    union_items = set()
    for items in shard_data.values():
        union_items.update(items)
    union_items = list(union_items)
    user_list = list(shard_data.keys())

    if not user_list or not union_items:
        return 0.0

    optimizer = Adagrad(model.parameters(), lr=lr, initial_accumulator_value=1e-8)
    rnd = random.Random(42)

    loss_acc = 0.0
    for epoch in range(n_epochs):
        n_batch = max(1, len(user_list) // batch_size)
        for _ in range(n_batch):
            users = [rnd.choice(user_list) for _ in range(batch_size)]
            pos_items = []
            neg_items = []
            for u in users:
                if not shard_data.get(u):
                    continue
                pos = rnd.choice(shard_data[u])
                neg = rnd.choice(union_items)
                while neg in shard_data.get(u, []):
                    neg = rnd.choice(union_items)
                pos_items.append(pos)
                neg_items.append(neg)

            if not pos_items:
                continue

            users_t = torch.LongTensor(users[:len(pos_items)]).to(device)
            pos_t = torch.LongTensor(pos_items).to(device)
            neg_t = torch.LongTensor(neg_items).to(device)

            optimizer.zero_grad()
            mf, reg, total = model.local_loss(users_t, pos_t, neg_t, shard_id)
            total.backward()
            optimizer.step()
            loss_acc += total.item()

    return loss_acc


def run_one_unlearn(part_num, part_type, agg_type, ratio, data_path, n_runs=5, seed_start=42):
    """Run unlearn multiple times with different random users."""
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load partition data
    C_path = os.path.join(data_path, f'C_type-{part_type}_num-{part_num}.pk')
    with open(C_path, 'rb') as f:
        C = pickle.load(f)

    # Load weights
    lr = 0.05
    for ep in [100, 1000, 5, 3]:
        weights_path = os.path.join(
            project_root, 'weights', 'ml-1m', 'RecEraser_BPR',
            f'p{part_num}-t{part_type}-e{ep}-lr{lr}-agg-{agg_type}',
            'weights.pt'
        )
        if os.path.exists(weights_path):
            break

    if not os.path.exists(weights_path):
        print(f"  [SKIP] no checkpoint at {weights_path}")
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RecEraserBPR(
        n_users=6040,
        n_items=3706,
        emb_dim=64,
        num_local=part_num,
        regs=[0.01],
        lr=lr,
    ).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"  loaded pretrained from {weights_path}")

    # Get test users
    from utility.load_data import Data
    data_gen = Data(
        path=data_path,
        batch_size=512,
        part_type=part_type,
        part_num=part_num,
        part_T=5
    )
    users_to_test = list(data_gen.test_set.keys())

    all_results = []

    for run_idx in range(n_runs):
        seed = seed_start + run_idx
        print(f"\n  --- Run {run_idx + 1}/{n_runs} (seed={seed}) ---")

        # Create random unlearned users
        temp_file = os.path.join(data_path, f'temp_random_r{ratio:02d}_run{run_idx}.txt')
        unlearned_users = create_random_unlearn_file(
            os.path.join(data_path, 'train.txt'),
            temp_file,
            int(6040 * ratio / 100),  # 5% of 6040 users
            seed
        )

        # Get unlearn entities
        base_data = load_train(os.path.join(data_path, 'train.txt'))
        target_data = load_train(temp_file)

        unlearned_uids = set()
        for uid in base_data:
            if uid not in target_data:
                unlearned_uids.add(uid)

        print(f"  unlearn: {len(unlearned_uids)} users")

        # Find affected shards
        affected = find_affected_shards(C, 'user', unlearned_uids, set())
        print(f"  affected shards: {affected}")

        # Evaluate baseline
        baseline = test_torch(model, users_to_test, device=device)
        print(f"  baseline Recall@20={baseline['recall'][1]:.4f}")

        # Retrain affected shards
        t0 = time()
        for sid in affected:
            shard_data = filter_shard_data(C[sid], 'user', unlearned_uids, set())
            loss = retrain_shard(model, sid, shard_data, lr, 512, device, n_epochs=1)
            print(f"  shard {sid} retrain done (loss={loss:.4f})")

        retrain_time = time() - t0

        # Evaluate after
        after = test_torch(model, users_to_test, device=device)
        print(f"  after Recall@20={after['recall'][1]:.4f} (change: {(after['recall'][1] - baseline['recall'][1])/baseline['recall'][1]*100:+.1f}%)")

        result = {
            'run_idx': run_idx + 1,
            'seed': seed,
            'n_unlearned_users': len(unlearned_users),
            'baseline': {
                'recall20': float(baseline['recall'][1]),
            },
            'after': {
                'recall20': float(after['recall'][1]),
            },
            'change_pct': float((after['recall'][1] - baseline['recall'][1]) / baseline['recall'][1] * 100),
            'retrain_time_s': float(retrain_time),
            'affected_shards': affected,
        }
        all_results.append(result)

        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return all_results


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

    print(f"\n{'='*60}")
    print(f"RANDOM USER UNLEARN")
    print(f"  agg_type: {cli.agg_type}")
    print(f"  ratio: {cli.ratio}%")
    print(f"  partition: {METHOD_INFO[cli.part_type]}")
    print(f"  runs: {cli.n_runs}")
    print(f"{'='*60}")

    data_path = os.path.join(project_root, 'data/ml-1m')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    all_results = run_one_unlearn(
        part_num=10,
        part_type=cli.part_type,
        agg_type=cli.agg_type,
        ratio=cli.ratio,
        data_path=data_path,
        n_runs=cli.n_runs,
        seed_start=42
    )

    if not all_results:
        print("No results!")
        return

    # Calculate statistics
    baseline_recalls = [r['baseline']['recall20'] for r in all_results]
    after_recalls = [r['after']['recall20'] for r in all_results]
    changes = [r['change_pct'] for r in all_results]

    print(f"\n{'='*60}")
    print(f"AVERAGE RESULTS ({cli.n_runs} runs)")
    print(f"{'='*60}")
    print(f"  Baseline Recall@20: {np.mean(baseline_recalls):.4f} +/- {np.std(baseline_recalls):.4f}")
    print(f"  After Recall@20:     {np.mean(after_recalls):.4f} +/- {np.std(after_recalls):.4f}")
    print(f"  Avg Change:          {np.mean(changes):+.2f}% +/- {np.std(changes):.2f}%")

    # Save
    output_file = os.path.join(results_dir,
        f'random_user_unlearn_p{10}_t{cli.part_type}_{cli.agg_type}_r{cli.ratio:02d}_runs{cli.n_runs}.json')

    output_data = {
        'config': {
            'agg_type': cli.agg_type,
            'ratio': cli.ratio,
            'part_type': METHOD_INFO[cli.part_type],
            'n_runs': cli.n_runs,
        },
        'average': {
            'baseline_recall20_mean': float(np.mean(baseline_recalls)),
            'baseline_recall20_std': float(np.std(baseline_recalls)),
            'after_recall20_mean': float(np.mean(after_recalls)),
            'after_recall20_std': float(np.std(after_recalls)),
            'change_pct_mean': float(np.mean(changes)),
            'change_pct_std': float(np.std(changes)),
        },
        'runs': all_results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to: {output_file}")


if __name__ == '__main__':
    main()
