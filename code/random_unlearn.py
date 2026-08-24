"""
Random Unlearning (Interaction, Item, User) - Run multiple times with different random selections.

Usage:
  # Unlearn random INTERACTION
  python random_unlearn.py --unlearn_type interaction --ratio 5 --part_type 1 --n_runs 5

  # Unlearn random ITEM
  python random_unlearn.py --unlearn_type item --ratio 5 --part_type 1 --n_runs 5

  # Unlearn random USER
  python random_unlearn.py --unlearn_type user --ratio 5 --part_type 1 --n_runs 5
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

# Parse arguments FIRST, before any other imports
ap = argparse.ArgumentParser(description='Random Unlearning')
ap.add_argument('--ratio', type=int, default=5, help='Percentage to unlearn')
ap.add_argument('--part_type', type=int, default=1, help='Partition type: 1=InP, 2=UBP, 3=Random, 4=IBP')
ap.add_argument('--n_runs', type=int, default=5, help='Number of runs')
ap.add_argument('--unlearn_type', type=str, default='interaction',
                choices=['interaction', 'item', 'user'])
cli = ap.parse_args()

# BLOCK TensorFlow imports
sys.argv = ['random_unlearn.py']

# Project paths
PROJ = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(PROJ)
sys.path.insert(0, PROJ)

# Set data path
os.environ['RECUNLEARN_DATA_PATH'] = os.path.join(project_root, 'data/')
os.environ['RECUNLEARN_DATASET'] = 'ml-1m'

# Import
from RecEraser_BPR_pytorch import RecEraserBPR, test_torch
from utility.load_data import Data

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


def create_random_unlearn_file(input_file, output_file, unlearn_type, ratio, seed):
    """Create unlearn file based on type. Returns (unlearned_uids, unlearned_iids, unlearned_data)."""
    random.seed(seed)
    user_items = load_train(input_file)

    unlearned_uids = set()
    unlearned_iids = set()

    if unlearn_type == 'user':
        # Unlearn random users
        all_users = list(user_items.keys())
        n_to_unlearn = int(len(all_users) * ratio / 100)
        unlearned_users = set(random.sample(all_users, n_to_unlearn))
        unlearned_data = {uid: items for uid, items in user_items.items() if uid not in unlearned_users}
        unlearned_uids = unlearned_users

    elif unlearn_type == 'item':
        # Unlearn random items
        all_items = set()
        for items in user_items.values():
            all_items.update(items)
        all_items = list(all_items)
        n_to_unlearn = int(len(all_items) * ratio / 100)
        unlearned_items = set(random.sample(all_items, n_to_unlearn))
        unlearned_data = {}
        for uid, items in user_items.items():
            remaining = [i for i in items if i not in unlearned_items]
            if remaining:
                unlearned_data[uid] = remaining
        unlearned_iids = unlearned_items

    elif unlearn_type == 'interaction':
        # Unlearn random interactions
        all_interactions = []
        for uid, items in user_items.items():
            for item in items:
                all_interactions.append((uid, item))
        n_to_unlearn = int(len(all_interactions) * ratio / 100)
        to_unlearn = set(random.sample(all_interactions, n_to_unlearn))
        unlearned_data = {}
        for uid, items in user_items.items():
            remaining = [i for i in items if (uid, i) not in to_unlearn]
            if remaining:
                unlearned_data[uid] = remaining
            else:
                unlearned_uids.add(uid)  # User lost all items

    return unlearned_uids, unlearned_iids, unlearned_data


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
        else:  # interaction
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
        return {u: [i for i in items if i not in unlearned_iids] for u, items in shard.items()}
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
            pos_items, neg_items = [], []
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


def run_random_unlearn(unlearn_type, part_type, ratio, n_runs, seed_start=42):
    """Run unlearn multiple times."""
    part_num = 10
    lr = 0.05

    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    data_path = os.path.join(project_root, 'data/ml-1m')

    print(f"\n  partition: {METHOD_INFO[part_type]} (type={part_type})")

    # Load partition data
    C_path = os.path.join(data_path, f'C_type-{part_type}_num-{part_num}.pk')
    with open(C_path, 'rb') as f:
        C = pickle.load(f)

    # Load weights
    for ep in [100, 1000, 5, 3]:
        weights_path = os.path.join(
            project_root, 'weights', 'ml-1m', 'RecEraser_BPR',
            f'p{part_num}-t{part_type}-e{ep}-lr{lr}-agg-attention',
            'weights.pt'
        )
        if os.path.exists(weights_path):
            break

    if not os.path.exists(weights_path):
        print(f"  [SKIP] no checkpoint at {weights_path}")
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RecEraserBPR(
        n_users=6040, n_items=3706, emb_dim=64,
        num_local=part_num, regs=[0.01], lr=lr,
    ).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"  loaded from {weights_path}")

    # Get test users
    data_gen = Data(path=data_path, batch_size=512, part_type=part_type, part_num=part_num, part_T=5)
    users_to_test = list(data_gen.test_set.keys())

    all_results = []

    for run_idx in range(n_runs):
        seed = seed_start + run_idx
        print(f"\n  --- Run {run_idx + 1}/{n_runs} (seed={seed}) ---")

        # RELOAD MODEL FROM CHECKPOINT for each run
        model.load_state_dict(checkpoint['state_dict'])

        # Create random unlearned data
        temp_file = os.path.join(data_path, f'temp_{unlearn_type}_r{ratio:02d}_run{run_idx}.txt')
        unlearned_uids, unlearned_iids, unlearned_data = create_random_unlearn_file(
            os.path.join(data_path, 'train.txt'), temp_file, unlearn_type, ratio, seed
        )

        # Find affected shards
        affected = find_affected_shards(C, unlearn_type, unlearned_uids, unlearned_iids)
        print(f"  affected shards: {affected}")

        # Evaluate baseline
        baseline = test_torch(model, users_to_test, device=device)
        print(f"  baseline Recall@20={baseline['recall'][1]:.4f}")

        # Retrain affected shards
        t0 = time()
        for sid in affected:
            shard_data = filter_shard_data(C[sid], unlearn_type, unlearned_uids, unlearned_iids)
            loss = retrain_shard(model, sid, shard_data, lr, 512, device, n_epochs=1)
            print(f"  shard {sid} retrain done (loss={loss:.4f})")

        retrain_time = time() - t0

        # Evaluate after
        after = test_torch(model, users_to_test, device=device)
        change = (after['recall'][1] - baseline['recall'][1]) / baseline['recall'][1] * 100
        print(f"  after Recall@20={after['recall'][1]:.4f} (change: {change:+.1f}%)")

        result = {
            'run_idx': run_idx + 1,
            'seed': seed,
            'baseline_recall20': float(baseline['recall'][1]),
            'after_recall20': float(after['recall'][1]),
            'change_pct': float(change),
            'retrain_time_s': float(retrain_time),
            'affected_shards': affected,
        }
        all_results.append(result)

        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return all_results


def main():
    print(f"\n{'='*60}")
    print(f"RANDOM UNLEARN ({cli.unlearn_type.upper()})")
    print(f"  ratio: {cli.ratio}%")
    print(f"  partition: {METHOD_INFO[cli.part_type]}")
    print(f"  runs: {cli.n_runs}")
    print(f"{'='*60}")

    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    all_results = run_random_unlearn(
        unlearn_type=cli.unlearn_type,
        part_type=cli.part_type,
        ratio=cli.ratio,
        n_runs=cli.n_runs,
        seed_start=42
    )

    if not all_results:
        print("No results!")
        return

    # Calculate statistics
    baseline_recalls = [r['baseline_recall20'] for r in all_results]
    after_recalls = [r['after_recall20'] for r in all_results]
    changes = [r['change_pct'] for r in all_results]

    print(f"\n{'='*60}")
    print(f"AVERAGE RESULTS ({cli.n_runs} runs)")
    print(f"{'='*60}")
    print(f"  Baseline Recall@20: {np.mean(baseline_recalls):.4f} +/- {np.std(baseline_recalls):.4f}")
    print(f"  After Recall@20:     {np.mean(after_recalls):.4f} +/- {np.std(after_recalls):.4f}")
    print(f"  Avg Change:           {np.mean(changes):+.2f}% +/- {np.std(changes):.2f}%")

    # Save
    output_file = os.path.join(results_dir,
        f'random_{cli.unlearn_type}_p10_t{cli.part_type}_r{cli.ratio:02d}_runs{cli.n_runs}.json')

    output_data = {
        'config': {
            'unlearn_type': cli.unlearn_type,
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
