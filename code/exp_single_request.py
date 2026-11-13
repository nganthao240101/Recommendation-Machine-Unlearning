"""
Cost of ONE deletion request (spec N2).

The batch experiments delete 1% of the users at once and report the total
time. This script measures the thing a deployed system actually answers: how
long does it take to remove ONE user from the model?

For each of N randomly chosen users (the same N for every arm, so the arms are
comparable), starting from the ORIGINAL trained model:

  * delete that one user,
  * retrain only the shard that held the user (the whole model for Retrain),
  * and for RecEraser also the aggregation layer,
  * measure the wall time of that retrain.

The reported number is the mean +- std over the N requests, per arm.

Run:
  python code/exp_single_request.py --dataset ml-1m --n_shards 8 \
      --n_users 20 --seeds 0,1,2,3,4 \
      --arms PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,RECERASER,GLOBAL
"""

import os
import sys

# Write every line to the log as it is printed. Piped through tee, Python
# otherwise buffers about 8 KB, so a log could sit hours behind the run and
# a progress check would read a live run as stuck.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass
import time
import random
import argparse
import statistics as st

import numpy as np
import torch

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

from exp_partition_quality import (          # noqa: E402
    load_data, pick_deletion_users, build_shard_data, set_all_seeds,
    partition_proposed, partition_proposed_balanced, partition_random_embedding,
    random_perm_balanced, _train_bprmf,
)
from partition_variants import (              # noqa: E402
    random_perm as pv_random_perm, random_modulo as pv_random_modulo,
)
from method_4_ours import ShardModels, train_shard_model, BPRMF  # noqa: E402
import gpu_probe  # noqa: E402


def build_partition(arm, train_items, n_users, n_items, n_shards, args, seed,
                    data_root):
    """The partition of each arm, built exactly as the main experiment does."""
    if arm == 'PROPOSED':
        return partition_proposed(train_items, n_users, n_items, n_shards,
                                  args.sig_dim, args.legacy_seed,
                                  router='hyperplane',
                                  sig_weighting=args.sig_weighting)
    if arm == 'PROPOSED_BAL':
        return partition_proposed_balanced(train_items, n_users, n_items,
                                           n_shards, args.sig_dim,
                                           args.legacy_seed)
    if arm == 'RANDOM_EMB':
        return partition_random_embedding(train_items, n_users, n_items,
                                          n_shards, args.sig_dim, seed,
                                          router=args.router)
    if arm == 'RANDOM_PERM':
        return pv_random_perm(n_users, n_shards, seed)
    if arm == 'RANDOM_BAL':
        return random_perm_balanced(train_items, n_users, n_shards, seed)
    if arm in ('SISA', 'RANDOM_MODULO'):
        return pv_random_modulo(n_users, n_shards, seed)
    if arm == 'RECERASER':
        from exp_receraser_arm import receraser_ubp_partition
        return receraser_ubp_partition(train_items, n_users, n_items,
                                       n_shards, seed, data_root=data_root,
                                       kind=args.receraser_partition)
    raise ValueError(f'unknown arm {arm}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ml-1m')
    ap.add_argument('--emb_dim', type=int, default=64)
    ap.add_argument('--sig_dim', type=int, default=64)
    ap.add_argument('--n_shards', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--retrain_epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--n_users', type=int, default=20,
                    help='How many single deletion requests to time.')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--arms', default='PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,'
                                      'RANDOM_EMB,RECERASER,GLOBAL')
    ap.add_argument('--global_requests', type=int, default=3,
                    help='How many of the requests to time for GLOBAL. Each one '
                         'retrains the whole model on all the remaining data, '
                         'so it costs a full training; the time hardly varies '
                         'between users, so a few requests measure it.')
    ap.add_argument('--legacy_seed', type=int, default=42)
    ap.add_argument('--router', default='hyperplane')
    ap.add_argument('--sig_weighting', default='binary')
    ap.add_argument('--receraser_partition', default='ubp',
                    choices=['ubp', 'inp', 'random'])
    ap.add_argument('--receraser_init', default='random',
                    choices=['random', 'wmf'],
                    help='random (the original RecEraser, as the other arms) '
                         'or wmf (warm start, needs emb_dim == 64)')
    ap.add_argument('--agg_epochs', type=int, default=50)
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--skip_runs', default='',
                    help='ARM:seed,ARM:seed,... already finished in an earlier, '
                         'interrupted run of this log; they are not run again. '
                         'run_blocks.sh passes them when it resumes a log.')
    args = ap.parse_args()

    data_root = args.data_root or os.path.join(os.path.dirname(PROJ), 'data')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_items, test_data, n_users, n_items = load_data(args.dataset, data_root)
    print(f"  {args.dataset}: {n_users} users, {n_items} items, device={device}")
    # From here on data_root only locates the dataset's own files (the WMF
    # embeddings of the RECERASER partition), which live in data/<dataset>/,
    # not in the parent data/ that load_data takes.
    data_root = os.path.join(data_root, args.dataset)
    emb = os.path.join(data_root, 'wmf_embeddings.npz')
    print(f"  WMF embeddings: {emb} ({'found' if os.path.exists(emb) else 'MISSING'})")

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    skip = {(a, int(s)) for a, s in (x.split(':') for x in args.skip_runs.split(',') if x)}
    if skip:
        print(f"  resuming: {len(skip)} run(s) already done earlier, not run again")
    arms = [a.strip() for a in args.arms.split(',') if a.strip()]

    # The same set of users is deleted for every arm and every seed, so the
    # requests are directly comparable across arms.
    rng = np.random.RandomState(12345)
    candidates = sorted(train_items.keys())
    victims = sorted(rng.choice(candidates, size=min(args.n_users, len(candidates)),
                                 replace=False).tolist())
    print(f"  {len(victims)} single-deletion requests, the same for every arm: "
          f"{victims[:10]}{'...' if len(victims) > 10 else ''}")

    results = []

    for seed in seeds:
        print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")
        for arm in arms:
            if (arm, seed) in skip:
                continue
            print(f"\n--- ARM {arm} (seed {seed}) ---")
            bg0 = gpu_probe.background_load()
            set_all_seeds(seed)

            if arm == 'RECERASER':
                times = _time_receraser(args, train_items, n_users, n_items,
                                        victims, seed, data_root, device)
            elif arm == 'GLOBAL':
                times = _time_global(args, train_items, n_users, n_items,
                                     victims[:max(1, args.global_requests)],
                                     seed, device)
            else:
                umap = build_partition(arm, train_items, n_users, n_items,
                                       args.n_shards, args, seed, data_root)
                times = _time_sharded(args, umap, train_items, n_users, n_items,
                                      victims, seed, device)

            mean = st.mean(times)
            sd = st.stdev(times) if len(times) > 1 else 0.0
            print(f"  [{arm}] single-request unlearn time: "
                  f"{mean:.2f}+-{sd:.2f}s over {len(times)} requests")
            bg1 = gpu_probe.background_load()
            shared = any(v is not None and v >= gpu_probe.THRESHOLD for v in (bg0, bg1))
            print(f"       bg-load: before={gpu_probe.fmt(bg0)} after={gpu_probe.fmt(bg1)}"
                  + ("  [SHARED GPU - other work was running, times may be inflated]"
                     if shared else ""))
            results.append({'arm': arm, 'seed': seed, 'n_requests': len(times),
                            'single_time_mean': mean, 'single_time_std': sd,
                            'single_time_min': min(times),
                            'single_time_max': max(times),
                            'times': [f'{t:.4f}' for t in times]})

    _print_summary(results, arms, seeds)
    return results


def _time_global(args, train_items, n_users, n_items, victims, seed, device):
    """Time one deletion at a time for the full retrain.

    Without shards, removing a user means training a fresh model on all the
    remaining data, so each request costs one full training. The same
    trainer as the GLOBAL arm of the main experiment, and as many epochs as
    the shard retrains get (--retrain_epochs).
    """
    times = []
    for v in victims:
        kept = {u: it for u, it in train_items.items() if u != v}
        _, elapsed, _, _ = _train_bprmf(kept, n_users, n_items, args.emb_dim,
                                        args.retrain_epochs, args.lr,
                                        args.batch_size, device, seed)
        times.append(elapsed)
    return times


def _time_sharded(args, umap, train_items, n_users, n_items, victims, seed,
                  device):
    """Time one deletion at a time for a sharded arm.

    The original model is trained ONCE. For each victim the model is copied,
    the victim is removed and the affected shard retrained, then the copy is
    discarded. Copying keeps every request independent and identical to the
    first one, which is what "one request, starting from the original model"
    means.
    """
    shard_data = build_shard_data(train_items, umap, args.n_shards)
    models = ShardModels(n_users, n_items, args.emb_dim, args.n_shards)
    for sid in range(args.n_shards):
        m = models.create_model(sid, device)
        train_shard_model(m, shard_data[sid], n_items, device,
                          batch_size=args.batch_size, lr=args.lr,
                          n_epochs=args.epochs)

    times = []
    for victim in victims:
        sid = int(umap[victim])
        # a fresh copy of the affected shard, so the retrain starts from the
        # ORIGINAL weights and the request does not affect the next one
        model = models.create_model(sid, device)
        model.load_state_dict(models.get_model(sid).state_dict())
        kept = {u: it for u, it in shard_data[sid].items() if u != victim}
        t0 = time.time()
        train_shard_model(model, kept, n_items, device,
                          batch_size=args.batch_size, lr=args.lr,
                          n_epochs=args.retrain_epochs)
        times.append(time.time() - t0)
        models.models[sid] = models.get_model(sid)   # restore the original
    return times


def _time_receraser(args, train_items, n_users, n_items, victims, seed,
                    data_root, device):
    """Time one deletion at a time for RecEraser.

    RecEraser has to retrain the affected shard AND the aggregation layer,
    because the attention weights depend on every user's embedding.
    """
    from exp_receraser_arm import RecEraserArm, receraser_ubp_partition

    emb_path = os.path.join(data_root, 'wmf_embeddings.npz')
    emb = np.load(emb_path)
    user_emb = emb['user_embeddings'][:n_users]
    item_emb = emb['item_embeddings']

    umap = receraser_ubp_partition(train_items, n_users, n_items,
                                   args.n_shards, seed, data_root=data_root,
                                   kind=args.receraser_partition)
    shard_data = build_shard_data(train_items, umap, args.n_shards)

    arm = RecEraserArm(n_users, n_items, args.emb_dim, args.n_shards,
                        device, seed)
    if args.receraser_init == 'wmf':
        arm.load_pretrained(user_emb, item_emb)
    print(f'    [RecEraser] init={args.receraser_init} '
          f'partition={args.receraser_partition} (emb_dim={args.emb_dim}, '
          f'K={args.n_shards})')
    for sid in range(args.n_shards):
        arm.train_shard(sid, shard_data[sid], n_items,
                         batch_size=args.batch_size, lr=args.lr,
                         n_epochs=args.epochs)
    arm.train_aggregator(train_items, n_users, n_items,
                         batch_size=args.batch_size, n_epochs=args.agg_epochs)

    # snapshot of the original state
    orig_state = {k: v.detach().clone()
                  for k, v in arm.model.state_dict().items()}

    times = []
    for victim in victims:
        sid = int(umap[victim])
        arm.model.load_state_dict(orig_state)
        kept = {u: it for u, it in train_items.items() if u != victim}
        kept_shard = {u: it for u, it in shard_data[sid].items() if u != victim}
        t0 = time.time()
        # the shard slice is rebuilt from scratch, so the victim's embedding
        # is genuinely gone
        d = args.emb_dim
        fresh_u = torch.nn.Embedding(n_users, d).weight.data.to(device)
        fresh_i = torch.nn.Embedding(n_items, d).weight.data.to(device)
        torch.nn.init.xavier_uniform_(fresh_u)
        torch.nn.init.xavier_uniform_(fresh_i)
        arm.set_shard_parameters(sid, fresh_u, fresh_i)
        arm.train_shard(sid, kept_shard, n_items, batch_size=args.batch_size,
                        lr=args.lr, n_epochs=args.retrain_epochs)
        # and the aggregator has to follow, since it reads every user's vector
        arm.train_aggregator(kept, n_users, n_items,
                             batch_size=args.batch_size,
                             n_epochs=args.agg_epochs)
        times.append(time.time() - t0)
    arm.model.load_state_dict(orig_state)
    return times


def _print_summary(results, arms, seeds):
    print()
    print('=' * 78)
    print('CONSOLE SUMMARY - single deletion request (seconds)')
    print('=' * 78)
    print(f'\n{"arm":<18}{"mean":>14}{"std":>14}{"min":>14}{"max":>14}{"n":>6}')
    print('-' * 78)
    for arm in arms:
        rs = [r for r in results if r['arm'] == arm]
        if not rs:
            continue
        vals = [r['single_time_mean'] for r in rs]
        print(f'{arm:<18}'
              f'{st.mean(vals):>13.2f}+-{st.stdev(vals) if len(vals) > 1 else 0.0:>13.2f}'
              f'{min(r["single_time_min"] for r in rs):>14.2f}'
              f'{max(r["single_time_max"] for r in rs):>14.2f}'
              f'{rs[0]["n_requests"]:>6}')

    # relative to the retrain baseline, which is the reference point of the
    # "why shard at all if you can retrain" question
    base = [r for r in results if r['arm'] == 'GLOBAL']
    if base:
        b = st.mean([r['single_time_mean'] for r in base])
        print()
        print(f'  ratio to Retrain (GLOBAL), mean={b:.2f}s:')
        for arm in arms:
            rs = [r for r in results if r['arm'] == arm]
            if not rs:
                continue
            v = st.mean([r['single_time_mean'] for r in rs])
            print(f'    {arm:<18}{v:>10.2f}s   {v / b:.3f}x')


if __name__ == '__main__':
    main()
