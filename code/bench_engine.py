"""
Which engine is faster on this machine's GPU? Times the same work with the
original per-batch loops and with the GPU-resident engine (fast_train.py):
training one BPRMF on a shard-sized slice of the dataset, then scoring every
test user.

  python code/bench_engine.py --dataset ml-1m        # about 1-2 minutes
  python code/bench_engine.py --dataset yelp2018 --users 4000

The original loops are the default; the GPU engine is opt-in until its
RecEraser arm matches them (see fast_train.py):
  bash code/run_blocks.sh start                            # the original loops (default)
  RECUNLEARN_ENGINE=gpu bash code/run_blocks.sh start      # the GPU engine
"""

import argparse
import contextlib
import io
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fast_train                                   # noqa: E402
from method_4_ours import BPRMF, train_shard_model  # noqa: E402
from exp_partition_quality import load_data         # noqa: E402
import method_4_ours                                # noqa: E402


class One:
    def __init__(self, m):
        self.models = [m]

    def get_model(self, s):
        return self.models[0]


def run(engine, train, test, n_users, n_items, epochs, device):
    fast_train.ENGINE = engine
    random.seed(0)
    torch.manual_seed(0)
    model = BPRMF(n_users, n_items, 64).to(device)
    with contextlib.redirect_stdout(io.StringIO()):
        _, t_train, n = train_shard_model(model, train, n_items, device, 512, 0.05, epochs)
        fast_train.sync(device)
        t0 = time.time()
        res = method_4_ours.evaluate_hard_routing(
            One(model), np.zeros(n_users, dtype=int), train, test, n_users, n_items,
            device, [10, 20, 50])
        fast_train.sync(device)
    return t_train, time.time() - t0, n, res['recall'][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ml-1m')
    ap.add_argument('--users', type=int, default=1500,
                    help='users in the slice (about one shard of ml-1m at K=4-8)')
    ap.add_argument('--epochs', type=int, default=5)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_root = os.path.join(os.path.dirname(HERE), 'data')
    train_all, test_all, n_users, n_items = load_data(args.dataset, data_root)
    users = sorted(train_all)[:args.users]
    train = {u: train_all[u] for u in users}
    test = {u: test_all[u] for u in users if test_all.get(u)}
    print(f'{args.dataset}: {len(users)} users, {sum(map(len, train.values()))} '
          f'interactions, {args.epochs} epochs, device={device}')
    out = {}
    for eng in ('python', 'gpu'):
        out[eng] = run(eng, train, test, n_users, n_items, args.epochs, device)
        t_tr, t_ev, n, r10 = out[eng]
        print(f'  {eng:<7} train {t_tr:7.1f}s ({t_tr / args.epochs:.2f}s/epoch)   '
              f'eval {t_ev:6.1f}s   R@10={r10:.4f}')
    sp_tr = out['python'][0] / max(out['gpu'][0], 1e-9)
    sp_ev = out['python'][1] / max(out['gpu'][1], 1e-9)
    print(f'  speed-up of the GPU engine: train x{sp_tr:.1f}, eval x{sp_ev:.1f}')
    print('  (the GPU engine stays opt-in until its RecEraser arm matches the original loops)')


if __name__ == '__main__':
    main()
