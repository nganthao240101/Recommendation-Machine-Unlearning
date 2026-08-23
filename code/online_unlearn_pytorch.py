"""
Online Recommendation Unlearning with PyTorch (paper RecEraser Section 4.2).

This script unlearns interactions using the SHARDED strategy.
"""
import os
import sys
import json
import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adagrad
from time import time

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

os.environ['RECUNLEARN_DATA_PATH'] = os.path.join(os.path.dirname(PROJ), 'data/')

from utility.helper import early_stopping
from utility.load_data import Data
from evaluator.python.evaluate_foldout import eval_score_matrix_foldout
from RecEraser_BPR_pytorch import RecEraserBPR, test_torch

RESULTS = os.path.join(PROJ, '..', 'results')

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


def get_unlearn_entities(unlearn_type, ratio, data_path):
    """Return (unlearned_uids, unlearned_iids)."""
    base = os.path.join(data_path, 'train.txt')
    possible_names = [
        f'train_unlearned_{unlearn_type}_r{int(ratio*100):02d}.txt',
        f'train_unlearned_r{int(ratio*100):02d}.txt',
        'train_unlearned.txt'
    ]
    target = None
    for name in possible_names:
        candidate = os.path.join(data_path, name)
        if os.path.exists(candidate):
            target = candidate
            break
    if target is None:
        print(f"   [ERROR] No unlearned file found. Tried: {possible_names}")
        return set(), set()

    base_data = load_train(base)
    target_data = load_train(target)

    unlearned_uids = set()
    unlearned_iids = set()

    for uid in base_data:
        if uid not in target_data:
            unlearned_uids.add(uid)
        else:
            orig_items = set(base_data[uid])
            new_items = set(target_data.get(uid, []))
            if orig_items - new_items:
                unlearned_iids.update(orig_items - new_items)

    return unlearned_uids, unlearned_iids


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


def retrain_shard(model, shard_id, shard_data, args, device, n_epochs=1):
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

    optimizer = Adagrad(model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)
    rnd = random.Random(42)

    loss_acc = 0.0
    for epoch in range(n_epochs):
        n_batch = max(1, len(user_list) // args.batch_size)
        for _ in range(n_batch):
            users = [rnd.choice(user_list) for _ in range(args.batch_size)]
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


def run_one_scenario(part_num, part_type, agg_type, unlearn_type, ratio, regs='0.01'):
    """Run one unlearning scenario. Returns metrics dict."""
    print(f'\n=== num={part_num}, type={part_type} ({METHOD_INFO[part_type]}), '
          f'agg={agg_type}, unlearn={unlearn_type} r={ratio} ===', flush=True)

    sys.argv = [
        '', '--dataset', 'ml-1m',
        '--part_type', str(part_type),
        '--part_num', str(part_num),
        '--agg_type', agg_type,
        '--regs', regs,
        '--pretrain', '1'
    ]

    project_root = os.path.dirname(PROJ)
    data_path = os.path.join(project_root, 'data/ml-1m/')

    from utility.parser import parse_args
    args = parse_args()
    args.data_path = data_path

    data_generator = Data(
        path=data_path,
        batch_size=args.batch_size,
        part_type=args.part_type,
        part_num=args.part_num,
        part_T=getattr(args, 'part_T', 5)
    )

    for ep in [100, 1000, 5, 3]:
        weights_path = os.path.join(
            project_root, 'weights', 'ml-1m', 'RecEraser_BPR',
            f'p{part_num}-t{part_type}-e{ep}-lr{args.lr}-agg-{agg_type}',
            'weights.pt'
        )
        if os.path.exists(weights_path):
            print(f'   found weights at: {weights_path}')
            break

    if not os.path.exists(weights_path):
        print(f'   [SKIP] no checkpoint at {weights_path}')
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RecEraserBPR(
        n_users=data_generator.n_users,
        n_items=data_generator.n_items,
        emb_dim=args.embed_size,
        num_local=part_num,
        regs=[float(regs)],
        lr=args.lr,
    ).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f'   loaded pretrained from {weights_path}')

    C_path = os.path.join(project_root, 'data', 'ml-1m', f'C_type-{part_type}_num-{part_num}.pk')
    if not os.path.exists(C_path):
        print(f'   [SKIP] no partition file at {C_path}')
        return None

    with open(C_path, 'rb') as f:
        C = pickle.load(f)

    print('   evaluating baseline...', flush=True)
    users_to_test = list(data_generator.test_set.keys())
    baseline = test_torch(model, users_to_test, device=device)
    print(f'   baseline Recall@20={baseline["recall"][1]:.4f}')

    u_unlearn, i_unlearn = get_unlearn_entities(unlearn_type, ratio, data_path)
    print(f'   unlearn: {len(u_unlearn)} users, {len(i_unlearn)} items')

    affected = find_affected_shards(C, unlearn_type, u_unlearn, i_unlearn)
    print(f'   affected shards: {affected}')

    t0 = time()
    for sid in affected:
        print(f'   retrain shard {sid}...', flush=True)
        new_shard = filter_shard_data(C[sid], unlearn_type, u_unlearn, i_unlearn)
        loss = retrain_shard(model, sid, new_shard, args, device, n_epochs=1)
        print(f'   shard {sid} retrain done (loss={loss:.4f})', flush=True)

    retrain_time = time() - t0
    print(f'   retrain time: {retrain_time:.1f}s')

    print('   evaluating after unlearn...', flush=True)
    online = test_torch(model, users_to_test, device=device)

    return {
        'baseline': {
            'recall10': float(baseline['recall'][0]),
            'recall20': float(baseline['recall'][1]),
            'recall50': float(baseline['recall'][2]),
            'ndcg10': float(baseline['ndcg'][0]),
            'ndcg20': float(baseline['ndcg'][1]),
            'ndcg50': float(baseline['ndcg'][2]),
        },
        'online_unlearn': {
            'recall10': float(online['recall'][0]),
            'recall20': float(online['recall'][1]),
            'recall50': float(online['recall'][2]),
            'ndcg10': float(online['ndcg'][0]),
            'ndcg20': float(online['ndcg'][1]),
            'ndcg50': float(online['ndcg'][2]),
        },
        'n_affected_shards': len(affected),
        'affected_shards': affected,
        'retrain_time_s': retrain_time,
        'partition': METHOD_INFO[part_type],
        'agg': agg_type,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part_type', type=int, default=None)
    ap.add_argument('--unlearn_type', type=str, default='interaction')
    ap.add_argument('--unlearn_ratio', type=float, default=0.1)
    ap.add_argument('--agg_type', type=str, default='attention',
                    choices=['attention', 'mean'])
    ap.add_argument('--regs', default='0.01')
    ap.add_argument('--out', default=None)
    cli = ap.parse_args()

    ptypes = [cli.part_type] if cli.part_type else [1, 2, 3, 4]
    utypes = [cli.unlearn_type]

    part_num = 10

    for pt in ptypes:
        for ut in utypes:
            r = run_one_scenario(part_num, pt, cli.agg_type, ut, cli.unlearn_ratio, cli.regs)
            if r is None:
                continue

            # Save to file with partition in name
            out_filename = f'online_unlearn_p{part_num}_t{pt}_{cli.agg_type}_{ut}_r{int(cli.unlearn_ratio*100):02d}.json'
            out_path = os.path.join(RESULTS, out_filename)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            with open(out_path, 'w') as f:
                json.dump({f'num{part_num}_{METHOD_INFO[pt]}_{cli.agg_type}_{ut}_r{int(cli.unlearn_ratio*100):02d}': r}, f, indent=2)
            print(f'[OK] Saved: {out_path}')


if __name__ == '__main__':
    main()
