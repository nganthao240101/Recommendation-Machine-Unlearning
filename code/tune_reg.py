"""
Choose the BPR L2 weight (reg), the learning rate and the embedding size on
the VALIDATION split.

For every (reg, lr, dim) and every method in --arms: train the K shards of the
method with early stopping (Recall@20 every --eval_every epochs, --patience
evaluations), keep each shard's best validation Recall@20 and its epoch;
score of a method = validation Recall@20 over all its validation users (each
user answered by its own shard: hard routing, as in the results; SISA_FULL:
R slices trained incrementally, the K models averaged, as in the results).

Printed:
  the full table;
  BEST <arm> ...   each method at its own best setting (per-method tuning);
  RECOMMENDED ...  ONE setting for every method: the best mean of the
                   methods' scores (no method favoured).
The test set is not read.

Every finished (method, setting) is appended to a JSON-lines cache
(logs/tunegrid_<dataset>.jsonl), so a stopped run resumes and a later, wider
grid reuses the cells already trained.

  python code/tune_reg.py --dataset yelp2018 --arms SAFER_P,SAFER_PI,RANDOM_PERM \
      --regs "0.001 0.003 0.01 0.03" --lrs "0.05 0.02 0.01" --dims "64"
"""

import argparse
import json
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
                                   partition_proposed, partition_proposed_balanced,
                                   partition_pca_tree, partition_random_embedding)
from partition_variants import random_perm as pv_random_perm   # noqa: E402
from partition_variants import build_signature as pv_build_signature   # noqa: E402
from method_4_ours import ShardModels, train_shard_model        # noqa: E402

LABEL = {'PROPOSED': 'SAFER A', 'PROPOSED_BAL': 'SAFER B', 'SAFER_P': 'SAFER-P',
         'SAFER_PI': 'SAFER-PI', 'RANDOM_PERM': 'Random', 'RANDOM_EMB': 'RandomEmb',
         'SISA_FULL': 'SISA'}
LEGACY_SEED = 42           # the partition seed of exp_partition_quality.py


def build_umap(arm, fit, n_users, n_items, K, sig_dim, seed):
    """The user->shard map of an arm, built as in exp_partition_quality.py
    (on the fit part of train: the validation items are held out)."""
    if arm == 'PROPOSED':
        return partition_proposed(fit, n_users, n_items, K, sig_dim, LEGACY_SEED)
    if arm == 'PROPOSED_BAL':
        return partition_proposed_balanced(fit, n_users, n_items, K, sig_dim, LEGACY_SEED)
    if arm in ('SAFER_P', 'SAFER_PI'):
        sig = pv_build_signature(fit, n_users, n_items, d=sig_dim, seed=LEGACY_SEED,
                                 weighting='binary' if arm == 'SAFER_P' else 'idf',
                                 verbose=False)
        return partition_pca_tree(fit, n_users, n_items, K, sig)
    if arm == 'RANDOM_EMB':
        return partition_random_embedding(fit, n_users, n_items, K, sig_dim, seed)
    if arm in ('RANDOM_PERM', 'SISA_FULL'):
        return pv_random_perm(n_users, K, seed)
    raise SystemExit(f'unknown arm {arm}')


def sisa_cell(args, shard_data, fit, val, n_users, n_items, K, dim, lr, device):
    """SISA as in the experiment (sisa_full.py): R slices per shard, every
    stage trained incrementally with early stopping on its own users'
    validation items; the K final models are AVERAGED (mean aggregation) and
    the aggregate is scored on every validation user.
    -> (validation Recall@20, last-stage best epoch of every shard)"""
    import sisa_full
    sm = ShardModels(n_users, n_items, dim, K)
    eps = []
    for sid in range(K):
        set_all_seeds(args.seed)
        m = sm.create_model(sid, device)
        opt = torch.optim.Adagrad(m.parameters(), lr=lr, initial_accumulator_value=1e-8)
        sl = sisa_full.slices_of(shard_data[sid], args.sisa_slices, args.seed, sid)
        best = 0
        for r in range(args.sisa_slices):
            keep = set(u for x in sl[:r + 1] for u in x)
            data = {u: shard_data[sid][u] for u in keep if u in shard_data[sid]}
            _, best, _, _ = sisa_full._stage(m, opt, data, n_items, device,
                                             args.batch_size, lr, 0,
                                             args.sisa_slices, True)
        eps.append(best)
    live = [sm.get_model(s) for s in range(K) if sm.get_model(s) is not None]

    def score_block(ut):
        sc = torch.zeros((len(ut), n_items), device=device)
        for m in live:
            sc += m.user_embedding.weight[ut] @ m.item_embedding.weight.T
        return sc / K
    users = sorted(u for u in val if val[u])
    with torch.no_grad():
        r20 = fast_train.evaluate_topk(score_block, users, fit, val, n_items, device,
                                       Ks=(early_stop.CFG['k'],))['recall'][0]
    return float(r20), eps


def cell_key(args, arm, reg, lr, dim):
    return '|'.join(str(x) for x in (args.dataset, arm, f'{reg:g}', f'{lr:g}', dim,
                                     args.n_shards, args.sig_dim, args.batch_size,
                                     args.val_ratio, args.val_seed, args.max_epochs,
                                     args.eval_every, args.patience, args.seed)
                   + ((f'R{args.sisa_slices}',) if arm == 'SISA_FULL' else ()))


def load_cache(path):
    done = {}
    if path and os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            try:
                r = json.loads(line)
                done[r['key']] = r
            except (ValueError, KeyError):
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='yelp2018')
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--regs', default='0.01 0.03 0.1 0.3')
    ap.add_argument('--lrs', default='0.05 0.02')
    ap.add_argument('--dims', default=None,
                    help='embedding sizes to try (default: --emb_dim only)')
    ap.add_argument('--arms', default='PROPOSED,RANDOM_PERM',
                    help='any of PROPOSED, PROPOSED_BAL, SAFER_P, SAFER_PI, '
                         'RANDOM_PERM, RANDOM_EMB, SISA_FULL')
    ap.add_argument('--sisa_slices', type=int, default=5,
                    help='R, the slices per shard of SISA_FULL')
    ap.add_argument('--cache', default=None,
                    help='JSON-lines file of finished cells (default '
                         'logs/tunegrid_<dataset>.jsonl)')
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
        umaps[arm] = build_umap(arm, fit, n_users, n_items, K, args.sig_dim, args.seed)
    shards = {a: build_shard_data(fit, umaps[a], K) for a in arms}
    print(f"{args.dataset}: {n_users} users, {n_items} items, device={device}, "
          f"engine={fast_train.engine_name()}")
    print(f"validation only: {early_stop.describe_train()}")
    cache = args.cache or os.path.join(os.path.dirname(HERE), 'logs',
                                       f'tunegrid_{args.dataset}.jsonl')
    os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
    done = load_cache(cache)
    print(f"cache: {cache} ({len(done)} finished cells)")

    dims = [int(d) for d in (args.dims or str(args.emb_dim)).split()]
    grid = [(float(r), float(l), d) for d in dims
            for r in args.regs.split() for l in args.lrs.split()]
    rows = []
    t_all = time.time()
    for reg, lr, dim in grid:
        method_4_ours.BPR_REG = method_3_receraser_v2.BPR_REG = reg
        res = {}
        t0 = time.time()
        for arm in arms:
            key = cell_key(args, arm, reg, lr, dim)
            if key in done:
                res[arm] = (done[key]['val_r20'], done[key]['epochs'])
                continue
            if arm == 'SISA_FULL':
                res[arm] = sisa_cell(args, shards[arm], fit, val, n_users, n_items,
                                     K, dim, lr, device)
                with open(cache, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'key': key, 'arm': arm, 'reg': reg, 'lr': lr,
                                        'dim': dim, 'val_r20': res[arm][0],
                                        'epochs': res[arm][1]}) + '\n')
                continue
            num, den, eps = 0.0, 0, []
            for sid in range(K):
                set_all_seeds(args.seed)
                sm = ShardModels(n_users, n_items, dim, K)
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
            with open(cache, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'key': key, 'arm': arm, 'reg': reg, 'lr': lr,
                                    'dim': dim, 'val_r20': res[arm][0],
                                    'epochs': eps}) + '\n')
        score = float(np.mean([res[a][0] for a in arms]))
        rows.append((reg, lr, dim, score, res))
        print(f"  reg={reg:<6g} lr={lr:<6g} dim={dim:<4d} "
              + "  ".join(f"{LABEL.get(a, a)} val R@20={res[a][0]:.4f} "
                          f"(best epochs {min(res[a][1])}-{max(res[a][1])})" for a in arms)
              + f"  -> score {score:.4f}   [{time.time() - t0:.0f}s]")

    rows.sort(key=lambda r: -r[3])
    print(f"\nVALIDATION Recall@20 (hard routing over all validation users), best first")
    print(f"  {'reg':>8}{'lr':>8}{'dim':>6}" + ''.join(f"{LABEL.get(a, a):>12}" for a in arms)
          + f"{'mean':>10}{'epochs':>14}")
    for reg, lr, dim, score, res in rows:
        ep = [e for a in arms for e in res[a][1]]
        print(f"  {reg:>8g}{lr:>8g}{dim:>6d}" + ''.join(f"{res[a][0]:>12.4f}" for a in arms)
              + f"{score:>10.4f}{f'{min(ep)}-{max(ep)}':>14}")
    # every method at its own best setting (the usual per-method tuning)
    print(f"\nBEST SETTING OF EACH METHOD (validation)")
    for a in arms:
        reg, lr, dim, _, res = max(rows, key=lambda r: r[4][a][0])
        edge = []
        if len(args.regs.split()) > 1 and f'{reg:g}' in (args.regs.split()[0], args.regs.split()[-1]):
            edge.append('reg')
        if len(args.lrs.split()) > 1 and f'{lr:g}' in (args.lrs.split()[0], args.lrs.split()[-1]):
            edge.append('lr')
        if len(dims) > 1 and dim in (dims[0], dims[-1]):
            edge.append('dim')
        note = f"   (at the grid edge: {', '.join(edge)})" if edge else ''
        print(f"  BEST {a:<12} reg={reg:g} lr={lr:g} dim={dim}  "
              f"val R@20={res[a][0]:.4f}{note}")
    reg, lr, dim, score, _ = rows[0]
    print(f"\nRECOMMENDED reg={reg:g} lr={lr:g} dim={dim}  (one setting for all the "
          f"methods above: mean validation R@20 {score:.4f})")
    print(f"total {(time.time() - t_all) / 60:.1f} min")


if __name__ == '__main__':
    main()
