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
    print(f'  {engine:<7} train {t_train:7.1f}s ({t_train / epochs:.2f}s/epoch)', end='',
          flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        t0 = time.time()
        res = method_4_ours.evaluate_hard_routing(
            One(model), np.zeros(n_users, dtype=int), train, test, n_users, n_items,
            device, [10, 20, 50])
        fast_train.sync(device)
    t_ev = time.time() - t0
    print(f'   eval {t_ev:6.1f}s for {len(test)} users   R@10={res["recall"][0]:.4f}',
          flush=True)
    return t_train, t_ev, n, res['recall'][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ml-1m')
    ap.add_argument('--users', type=int, default=600,
                    help='users in the slice (about one shard of ml-1m at K=4-8)')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--eval_users', type=int, default=100,
                    help='test users scored (the original loop takes seconds per user '
                         'on a busy GPU)')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_root = os.path.join(os.path.dirname(HERE), 'data')
    train_all, test_all, n_users, n_items = load_data(args.dataset, data_root)
    users = sorted(train_all)[:args.users]
    train = {u: train_all[u] for u in users}
    test = {u: test_all[u] for u in users[:args.eval_users] if test_all.get(u)}
    print(f'{args.dataset}: {len(users)} users, {sum(map(len, train.values()))} '
          f'interactions, {args.epochs} epochs, device={device}')
    out = {}
    # the GPU engine first: its numbers come in seconds, while the original
    # loop waits on a busy GPU after every batch and can take minutes
    for eng in ('gpu', 'python'):
        out[eng] = run(eng, train, test, n_users, n_items, args.epochs, device)
    sp_tr = out['python'][0] / max(out['gpu'][0], 1e-9)
    sp_ev = out['python'][1] / max(out['gpu'][1], 1e-9)
    print(f'  speed-up of the GPU engine: train x{sp_tr:.1f}, eval x{sp_ev:.1f}')
    print('  (the GPU engine stays opt-in until its RecEraser arm matches the original loops)')


if __name__ == '__main__':
    main()
