"""
Choose the early-stopping patience on the VALIDATION split (no test data).

Trains, with a very long patience so that the whole validation curve is
seen, the models whose epochs early stopping would pick:

  * the full model (Retrain, GLOBAL),
  * every shard of SAFER A (PROPOSED) and of Random (RANDOM_PERM), K shards,

each evaluated on its validation users every --eval_every epochs (Recall@20,
as the authors), up to --max_epochs. It then replays every curve under
candidate patiences and reports, per kind of model:

  stop epoch      where training would stop
  best epoch      the epoch whose weights would be kept
  val loss %      how far that is below the best Recall@20 of the whole curve
  cost            epochs trained, relative to the longest patience

and recommends the smallest patience whose validation loss stays within
--tol percent (mean) for every kind of model.

  python code/tune_patience.py --dataset yelp2018
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
import wmf_backbone    # noqa: E402
from exp_partition_quality import (load_data, set_all_seeds, build_shard_data,   # noqa: E402
                                   partition_proposed, _train_bprmf)
from partition_variants import random_perm as pv_random_perm   # noqa: E402
from method_4_ours import ShardModels, train_shard_model        # noqa: E402

CANDIDATES = (2, 3, 5, 8, 10, 15, 20, 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ml-1m')
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--n_shards', type=int, default=8)
    ap.add_argument('--emb_dim', type=int, default=64)
    ap.add_argument('--sig_dim', type=int, default=64)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--val_seed', type=int, default=2024)
    ap.add_argument('--max_epochs', type=int, default=1000)
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--cap', type=int, default=40,
                    help='stop a curve after this many evaluations without '
                         'improvement (must exceed every candidate)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tol', type=float, default=0.5,
                    help='accepted mean validation loss, percent of the best')
    ap.add_argument('--kinds', default='GLOBAL,PROPOSED,RANDOM_PERM')
    ap.add_argument('--backbone', default='bprmf', choices=['bprmf', 'wmf'])
    args = ap.parse_args()

    data_root = args.data_root or os.path.join(os.path.dirname(HERE), 'data')
    wmf_backbone.configure(args.backbone, None, 0.7, 256, dataset=args.dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train, test, n_users, n_items = load_data(args.dataset, data_root)
    fit, val = early_stop.split_validation(train, args.val_ratio, args.val_seed)
    early_stop.configure(val, every=args.eval_every, patience=args.cap,
                         max_epochs=args.max_epochs)
    print(f"{args.dataset}: {n_users} users, {n_items} items, device={device}, "
          f"engine={fast_train.engine_name()}")
    print(f"early stopping probe: {early_stop.describe()}")
    curves = {}                     # kind -> [history, ...]
    t_all = time.time()

    for kind in [k for k in args.kinds.split(',') if k]:
        curves[kind] = []
        if kind == 'GLOBAL':
            t0 = time.time()
            holder = {}

            def stopper(m):
                holder['es'] = early_stop.model_stopper(m, list(fit), fit, n_items, device)
                return holder['es']
            _train_bprmf(fit, n_users, n_items, args.emb_dim, args.max_epochs, args.lr,
                         args.batch_size, device, args.seed, stopper=stopper)
            es = holder['es']
            curves[kind].append(es.history)
            print(f"  [GLOBAL] best epoch {es.best_epoch} (val R@20 {es.best:.4f}), "
                  f"stopped at {es.last_epoch}, {time.time() - t0:.0f}s")
            continue
        set_all_seeds(args.seed)
        if kind == 'PROPOSED':
            umap = partition_proposed(fit, n_users, n_items, args.n_shards, args.sig_dim, 42)
        elif kind == 'RANDOM_PERM':
            umap = pv_random_perm(n_users, args.n_shards, args.seed)
        else:
            raise ValueError(kind)
        shard_data = build_shard_data(fit, umap, args.n_shards)
        sm = ShardModels(n_users, n_items, args.emb_dim, args.n_shards)
        for sid in range(args.n_shards):
            t0 = time.time()
            model = sm.create_model(sid, device)
            es = early_stop.model_stopper(model, list(shard_data[sid]), shard_data[sid],
                                          n_items, device)
            train_shard_model(model, shard_data[sid], n_items, device,
                              batch_size=args.batch_size, lr=args.lr,
                              n_epochs=args.max_epochs, epoch_end=es)
            es.finish()
            curves[kind].append(es.history)
            print(f"  [{kind} s{sid}] best epoch {es.best_epoch} (val R@20 {es.best:.4f}), "
                  f"stopped at {es.last_epoch}, {time.time() - t0:.0f}s")

    # ---- replay every curve under each candidate patience
    print(f"\nPATIENCE (evaluations of {args.eval_every} epochs without a better "
          f"validation Recall@20); val loss = how far the kept weights are below "
          f"the best of the whole curve")
    long_cost = {k: sum(h[-1][0] for h in v if h) for k, v in curves.items()}
    ok = {}
    for kind, hs in curves.items():
        hs = [h for h in hs if h]
        if not hs:
            continue
        print(f"\n  {kind} ({len(hs)} model{'s' if len(hs) > 1 else ''})")
        print(f"  {'patience':>9}{'= epochs':>10}{'stop ep':>10}{'best ep':>10}"
              f"{'val loss %':>12}{'max loss %':>12}{'cost':>8}")
        for p in CANDIDATES:
            if p > args.cap:
                continue
            stops, bests, losses = [], [], []
            for h in hs:
                top = max(r for _, r in h)
                st, be, br = early_stop.simulate(h, p)
                stops.append(st)
                bests.append(be)
                losses.append((top - br) / top * 100 if top > 0 else 0.0)
            cost = sum(stops) / max(long_cost[kind], 1)
            print(f"  {p:>9}{p * args.eval_every:>10}{np.mean(stops):>10.0f}"
                  f"{np.mean(bests):>10.0f}{np.mean(losses):>12.2f}{max(losses):>12.2f}"
                  f"{cost:>8.2f}")
            if np.mean(losses) <= args.tol and kind not in ok:
                ok[kind] = p
    need = max(ok.values()) if ok and len(ok) == len([k for k in curves if curves[k]]) else None
    print()
    for kind in curves:
        print(f"  {kind:<12} smallest patience within {args.tol}% val loss: "
              f"{ok.get(kind, 'none of the candidates')}")
    if need:
        print(f"\nRECOMMENDED patience = {need} evaluations "
              f"(= {need * args.eval_every} epochs without improvement), "
              f"the smallest that keeps every kind of model within {args.tol}%")
    else:
        print('\nno candidate keeps every kind within the tolerance; '
              'use the longest one or raise --tol')
    print(f"total {(time.time() - t_all) / 60:.1f} min")


if __name__ == '__main__':
    main()
