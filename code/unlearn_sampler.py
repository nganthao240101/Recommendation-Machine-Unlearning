"""
Sample unlearned training sets for 3 unlearn types × 3 ratios.

Input:  data/ml-1m/train.txt  (original training set)
Output: data/ml-1m/train_unlearned_<type>_<ratio>.txt
        + unlearn_log.json listing what was removed

Unlearn types (paper RecEraser protocol, simplified):
  - interaction: remove X% of (user, item) pairs at random
  - user:        remove X% of users entirely
  - item:        remove X% of items entirely

Usage:
  python unlearn_sampler.py            # sample all 3 types x 3 ratios
  python unlearn_sampler.py --type interaction --ratio 0.1
"""
import os
import json
import random
import argparse


def load_train(path):
    """Load train.txt -> dict {uid: [item,...]}."""
    data = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(' ')
            if not parts or parts == ['']:
                continue
            uid = int(parts[0])
            items = [int(x) for x in parts[1:] if x]
            if items:
                data[uid] = items
    return data


def save_train(data, path):
    with open(path, 'w') as f:
        for uid, items in sorted(data.items()):
            f.write(f'{uid} ' + ' '.join(str(i) for i in items) + '\n')


def sample_unlearn_interaction(data, ratio, seed=42):
    """Remove `ratio` of (uid, item) pairs at random; keep user, drop items."""
    rnd = random.Random(seed)
    pairs = [(u, i) for u, items in data.items() for i in items]
    n_remove = int(len(pairs) * ratio)
    removed = set(rnd.sample(pairs, n_remove))
    new_data = {u: [i for i in items if (u, i) not in removed]
                for u, items in data.items()}
    new_data = {u: items for u, items in new_data.items() if items}
    return new_data, sorted(removed)


def sample_unlearn_user(data, ratio, seed=42):
    """Remove `ratio` of users entirely."""
    rnd = random.Random(seed)
    uids = list(data.keys())
    n_remove = int(len(uids) * ratio)
    removed = rnd.sample(uids, n_remove)
    new_data = {u: items for u, items in data.items() if u not in removed}
    return new_data, removed


def sample_unlearn_item(data, ratio, seed=42):
    """Remove `ratio` of items entirely from all users."""
    rnd = random.Random(seed)
    all_items = set()
    for items in data.values():
        all_items.update(items)
    all_items = list(all_items)
    n_remove = int(len(all_items) * ratio)
    removed = set(rnd.sample(all_items, n_remove))
    new_data = {u: [i for i in items if i not in removed]
                for u, items in data.items()}
    new_data = {u: items for u, items in new_data.items() if items}
    return new_data, sorted(removed)


SAMPLERS = {
    'interaction': sample_unlearn_interaction,
    'user':        sample_unlearn_user,
    'item':        sample_unlearn_item,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', default='../data/ml-1m')
    ap.add_argument('--types', nargs='+', default=['interaction', 'user', 'item'])
    ap.add_argument('--ratios', nargs='+', type=float, default=[0.05, 0.1, 0.2])
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    train_path = os.path.join(args.data_path, 'train.txt')
    print(f'Loading {train_path}...')
    data = load_train(train_path)
    print(f'  {len(data)} users, '
          f'{sum(len(v) for v in data.values())} interactions')

    log = {'original': {'users': len(data),
                        'interactions': sum(len(v) for v in data.values())},
           'samples': {}}

    for utype in args.types:
        for ratio in args.ratios:
            tag = f'{utype}_r{int(ratio * 100):02d}'
            out_path = os.path.join(args.data_path, f'train_unlearned_{tag}.txt')
            new_data, removed = SAMPLERS[utype](data, ratio, seed=args.seed)
            save_train(new_data, out_path)
            log['samples'][tag] = {
                'type': utype,
                'ratio': ratio,
                'users': len(new_data),
                'interactions': sum(len(v) for v in new_data.values()),
                'removed_count': len(removed),
                'removed_sample': removed[:10],   # first 10 only
            }
            print(f'  [{tag:25s}] {len(new_data):4d} users  '
                  f'{sum(len(v) for v in new_data.values()):7d} interactions  '
                  f'-> {out_path}')

    log_path = os.path.join(args.data_path, 'unlearn_log.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f'\n[OK] Saved log: {log_path}')


if __name__ == '__main__':
    main()
