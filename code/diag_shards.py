"""
Why does one partition score lower than another? Per-shard test quality of
the SAVED models (checkpoints/ from an early-stopping one-model run), no
training.

For every arm and every shard: users, interactions, the epoch early stopping
kept, and Recall@20 / NDCG@20 on the test items of the shard's users for

  model      the shard's trained model (hard routing, as in the results)
  shard-pop  recommending the items most popular INSIDE the shard
  global-pop recommending the items most popular in the whole training set

so one sees whether the models are better than popularity, and whether a
partition loses on its small shards, its large shards, or everywhere.

  python code/diag_shards.py --dataset yelp2018 --arms PROPOSED,RANDOM_PERM
"""

import argparse
import glob
import os
import sys
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import early_stop    # noqa: E402
import fast_train    # noqa: E402
from exp_partition_quality import (load_data, build_shard_data, partition_proposed,   # noqa: E402
                                   partition_proposed_balanced,
                                   partition_random_embedding)
from partition_variants import random_perm as pv_random_perm, random_modulo   # noqa: E402


def find_ckpt(model_dir, dataset, K, arm, want_es=True):
    best = None
    for p in glob.glob(os.path.join(model_dir, f'{dataset}_K{K}_{arm}_*.pt')):
        try:
            blob = torch.load(p, map_location='cpu', weights_only=False)
        except Exception:
            continue
        meta = blob.get('meta', {})
        if meta.get('arm') != arm or (want_es and float(meta.get('val_ratio') or 0) <= 0):
            continue
        if best is None or os.path.getmtime(p) > os.path.getmtime(best[0]):
            best = (p, blob)
    return best


def partition(arm, fit, n_users, n_items, K, sig_dim, seed):
    if arm == 'PROPOSED':
        return partition_proposed(fit, n_users, n_items, K, sig_dim, 42)
    if arm == 'PROPOSED_BAL':
        return partition_proposed_balanced(fit, n_users, n_items, K, sig_dim, 42)
    if arm == 'RANDOM_PERM':
        return pv_random_perm(n_users, K, seed)
    if arm == 'SISA':
        return random_modulo(n_users, K, seed)
    if arm == 'RANDOM_EMB':
        return partition_random_embedding(fit, n_users, n_items, K, sig_dim, seed,
                                          router='hyperplane')
    raise ValueError(arm)


def pop_scores(data, n_items, device):
    c = Counter(i for its in data.values() for i in its)
    v = torch.zeros(n_items, device=device)
    for i, n in c.items():
        v[i] = n
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='yelp2018')
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--arms', default='PROPOSED,RANDOM_PERM')
    ap.add_argument('--n_shards', type=int, default=8)
    ap.add_argument('--sig_dim', type=int, default=64)
    ap.add_argument('--emb_dim', type=int, default=64)
    ap.add_argument('--model_dir', default=os.path.join(os.path.dirname(HERE), 'checkpoints'))
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_root = args.data_root or os.path.join(os.path.dirname(HERE), 'data')
    train, test, n_users, n_items = load_data(args.dataset, data_root)
    K = args.n_shards
    from method_4_ours import BPRMF
    summary = {}
    for arm in [a for a in args.arms.split(',') if a]:
        found = find_ckpt(args.model_dir, args.dataset, K, arm)
        if not found:
            print(f'\n{arm}: no saved early-stopping model in {args.model_dir}')
            continue
        path, blob = found
        meta, cache = blob['meta'], blob['cache']
        fit, val = early_stop.split_validation(train, meta['val_ratio'], meta['val_seed'])
        early_stop.CFG['val'] = val
        excl = early_stop.with_val(fit)
        umap = partition(arm, fit, n_users, n_items, K, args.sig_dim, meta['train_seed'])
        shards = build_shard_data(fit, umap, K)
        gpop = pop_scores(fit, n_items, device)
        epochs = cache.get('shard_epochs', {})
        print(f'\n=== {arm}  ({os.path.basename(path)})')
        print(f"  {'shard':>5}{'users':>8}{'inter':>9}{'epoch':>7}"
              f"{'model R@20':>12}{'N@20':>8}{'shard-pop':>11}{'global-pop':>12}")
        tot = {'users': 0, 'model': 0.0, 'nd': 0.0, 'spop': 0.0, 'gpop': 0.0}
        for sid in range(K):
            users = [u for u in shards[sid] if test.get(u)]
            if not users:
                continue
            m = BPRMF(n_users, n_items, args.emb_dim).to(device)
            m.load_state_dict(cache['states'][sid])
            W_u, W_i = m.user_embedding.weight.detach(), m.item_embedding.weight.detach()
            r = fast_train.evaluate_topk(lambda ut: W_u[ut] @ W_i.T, users, excl, test,
                                         n_items, device, Ks=(20,), per_list=True)
            sp = pop_scores(shards[sid], n_items, device)
            rs = fast_train.evaluate_topk(lambda ut: sp.expand(len(ut), -1), users, excl, test,
                                          n_items, device, Ks=(20,), per_list=True)
            rg = fast_train.evaluate_topk(lambda ut: gpop.expand(len(ut), -1), users, excl, test,
                                          n_items, device, Ks=(20,), per_list=True)
            n_int = sum(len(v) for v in shards[sid].values())
            print(f"  {sid:>5}{len(users):>8}{n_int:>9}{epochs.get(sid, '-'):>7}"
                  f"{r['recall'][0]:>12.4f}{r['ndcg'][0]:>8.4f}{rs['recall'][0]:>11.4f}"
                  f"{rg['recall'][0]:>12.4f}")
            w = len(users)
            tot['users'] += w
            tot['model'] += w * r['recall'][0]
            tot['nd'] += w * r['ndcg'][0]
            tot['spop'] += w * rs['recall'][0]
            tot['gpop'] += w * rg['recall'][0]
        n = max(tot['users'], 1)
        summary[arm] = {k: tot[k] / n for k in ('model', 'nd', 'spop', 'gpop')}
        s = summary[arm]
        print(f"  {'all':>5}{tot['users']:>8}{'':>9}{'':>7}{s['model']:>12.4f}{s['nd']:>8.4f}"
              f"{s['spop']:>11.4f}{s['gpop']:>12.4f}   (user-weighted, before any deletion)")
    if len(summary) > 1:
        print('\nSUMMARY (test Recall@20, saved models before deletion)')
        for arm, s in summary.items():
            print(f"  {arm:<14} model {s['model']:.4f}   shard-pop {s['spop']:.4f}   "
                  f"global-pop {s['gpop']:.4f}   model/global-pop {s['model'] / max(s['gpop'], 1e-12):.2f}x")


if __name__ == '__main__':
    main()
