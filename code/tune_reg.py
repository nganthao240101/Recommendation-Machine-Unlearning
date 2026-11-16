"""
Choose the BPR L2 weight (reg) and the learning rate on the VALIDATION split.

The shards stop improving on validation after ~5 epochs at the authors'
reg 0.01 / lr 0.05: they overfit before they learn more than popularity. A
stronger L2 or a smaller step may let them learn longer. This script tries
a grid and picks ONE setting for every method, on validation only:

  for every (reg, lr): train the K shards of SAFER A (PROPOSED) and of
  Random (RANDOM_PERM) with early stopping (Recall@20 every --eval_every
  epochs, --patience evaluations), keep each shard's best validation
  Recall@20 and its epoch;
  score of a method = validation Recall@20 over all its validation users
  (each user answered by its own shard: hard routing, as in the results);
  score of a setting = mean of the two methods' scores (no method favoured).

RECOMMENDED is the setting with the best score. The test set is not read.

  python code/tune_reg.py --dataset yelp2018 --regs "0.01 0.03 0.1 0.3" --lrs "0.05 0.02"
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

import early_stop      # noqa: E402
import fast_train      # noqa: E402
import method_4_ours   # noqa: E402
import method_3_receraser_v2   # noqa: E402
import wmf_backbone    # noqa: E402
from exp_partition_quality import (load_data, set_all_seeds, build_shard_data,   # noqa: E402
                                   partition_proposed)
from partition_variants import random_perm as pv_random_perm   # noqa: E402
from method_4_ours import ShardModels, train_shard_model        # noqa: E402

LABEL = {'PROPOSED': 'SAFER A', 'RANDOM_PERM': 'Random'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='yelp2018')
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--regs', default='0.01 0.03 0.1 0.3')
    ap.add_argument('--lrs', default='0.05 0.02')
    ap.add_argument('--arms', default='PROPOSED,RANDOM_PERM')
    ap.add_argument('--n_shards', type=int, default=8)
    ap.add_argument('--emb_dim', type=int, default=64)
    ap.add_argument('--sig_dim', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--val_seed', type=int, default=2024)
    ap.add_argument('--max_epochs', type=int, default=1000)
    ap.add_argument('--eval_every', type=int, default=1,
                    help='evaluate every N epochs (1: the peak can be very early)')
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    data_root = args.data_root or os.path.join(os.path.dirname(HERE), 'data')
    wmf_backbone.configure('bprmf', None, 0.7, 256, dataset=args.dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train, test, n_users, n_items = load_data(args.dataset, data_root)
    del test                                    # never read here
    fit, val = early_stop.split_validation(train, args.val_ratio, args.val_seed)
    early_stop.configure(val, every=args.eval_every, patience=args.patience,
                         max_epochs=args.max_epochs)
    K = args.n_shards
    arms = [a for a in args.arms.split(',') if a]
    umaps = {}
    for arm in arms:
        set_all_seeds(args.seed)
        umaps[arm] = (partition_proposed(fit, n_users, n_items, K, args.sig_dim, 42)
                      if arm == 'PROPOSED' else pv_random_perm(n_users, K, args.seed))
    shards = {a: build_shard_data(fit, umaps[a], K) for a in arms}
    print(f"{args.dataset}: {n_users} users, {n_items} items, device={device}, "
          f"engine={fast_train.engine_name()}")
    print(f"validation only: {early_stop.describe_train()}")

    grid = [(float(r), float(l)) for r in args.regs.split() for l in args.lrs.split()]
    rows = []
    t_all = time.time()
    for reg, lr in grid:
        method_4_ours.BPR_REG = method_3_receraser_v2.BPR_REG = reg
        res = {}
        t0 = time.time()
        for arm in arms:
            num, den, eps = 0.0, 0, []
            for sid in range(K):
                set_all_seeds(args.seed)
                sm = ShardModels(n_users, n_items, args.emb_dim, K)
                model = sm.create_model(sid, device)
                es = early_stop.model_stopper(model, list(shards[arm][sid]),
                                              shards[arm][sid], n_items, device)
                train_shard_model(model, shards[arm][sid], n_items, device,
                                  batch_size=args.batch_size, lr=lr,
                                  n_epochs=args.max_epochs, epoch_end=es)
                es.finish()
                n_val = len(es.users)
                if n_val and es.best >= 0:
                    num += es.best * n_val
                    den += n_val
                eps.append(es.best_epoch)
            res[arm] = (num / max(den, 1), eps)
        score = float(np.mean([res[a][0] for a in arms]))
        rows.append((reg, lr, score, res))
        print(f"  reg={reg:<6g} lr={lr:<6g} "
              + "  ".join(f"{LABEL.get(a, a)} val R@20={res[a][0]:.4f} "
                          f"(best epochs {min(res[a][1])}-{max(res[a][1])})" for a in arms)
              + f"  -> score {score:.4f}   [{time.time() - t0:.0f}s]")

    rows.sort(key=lambda r: -r[2])
    print(f"\nVALIDATION Recall@20 (hard routing over all validation users), best first")
    print(f"  {'reg':>8}{'lr':>8}" + ''.join(f"{LABEL.get(a, a):>12}" for a in arms)
          + f"{'mean':>10}{'epochs':>14}")
    for reg, lr, score, res in rows:
        ep = [e for a in arms for e in res[a][1]]
        print(f"  {reg:>8g}{lr:>8g}" + ''.join(f"{res[a][0]:>12.4f}" for a in arms)
              + f"{score:>10.4f}{f'{min(ep)}-{max(ep)}':>14}")
    reg, lr, score, _ = rows[0]
    base = [r for r in rows if r[0] == 0.01 and r[1] == 0.05]
    gain = (f", {100 * (score / base[0][2] - 1):+.1f}% vs reg 0.01 / lr 0.05"
            if base and base[0][2] > 0 else '')
    print(f"\nRECOMMENDED reg={reg:g} lr={lr:g}  (validation score {score:.4f}{gain})")
    print(f"total {(time.time() - t_all) / 60:.1f} min")


if __name__ == '__main__':
    main()
