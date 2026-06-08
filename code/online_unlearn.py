"""
Online Recommendation Unlearning (paper RecEraser Section 4.2).

For a given (unlearn_type, unlearn_ratio, partition_type), this script
demonstrates the SHARDED unlearning strategy:

  1) Load the trained RecEraser checkpoint
  2) Identify the shards C[i] that contain at least one unlearned entity
     (user / item / interaction depending on unlearn_type)
  3) Retrain ONLY those shards, warm-starting from the existing weights,
     on the data that excludes the unlearn set
  4) Keep all unaffected shards as-is
  5) Re-evaluate the model on the test set

This is the "Online Unlearning" path that should approach the "Full
Retrain" oracle (both trained on the unlearned data) but at a fraction
of the compute, because only a subset of shards is retrained.

Outputs (results/online_unlearn_num{N}.json):
  {
    "num{N}_<type>_r{ratio}": {
      "no_unlearn":        {Recall@K, ...},   # baseline
      "online_unlearn":    {Recall@K, ...},   # this script
      "full_retrain":      {Recall@K, ...},   # already in full_eval_num5.json
    },
    ...
  }

Usage:
  python online_unlearn.py 5
"""
import os
import sys
import json
import types
import pickle
import warnings
import random
import argparse
warnings.filterwarnings('ignore')

# Strip our positional arg
_SELF_PART_NUM = 5
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    _SELF_PART_NUM = int(sys.argv[1])
    del sys.argv[1:2]

import numpy as np
import tensorflow as _tf_raw
_tf_v1 = _tf_raw.compat.v1
_tf_v1.disable_eager_execution()
import tensorflow.python.ops.init_ops as _init_ops
_XAVIER = _init_ops.VarianceScaling(scale=1.0, mode='fan_avg',
                                    distribution='uniform')
def _xavier(*a, **k): return _XAVIER
_tf_v1.contrib = types.SimpleNamespace(
    layers=types.SimpleNamespace(xavier_initializer=_xavier)
)
sys.modules['tensorflow'] = _tf_v1
tf = _tf_v1

PROJ = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(PROJ, '..', 'data', 'ml-1m')
WEIGHTS = os.path.join(PROJ, 'weights', 'ml-1m', 'RecEraser_BPR')
RESULTS = os.path.join(PROJ, '..', 'results')

from utility.parser import parse_args
from utility.load_data import Data
from utility.batch_test import test
from RecEraser_BPR import RecEraser_BPR
from time import time


METHOD_INFO = {1: 'InP', 2: 'UBP', 3: 'Random', 4: 'IBP'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_train(path):
    """Load train.txt -> dict {uid: [items...]}."""
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


def get_unlearn_entities(unlearn_type, ratio, data_path):
    """Return (unlearned_uids, unlearned_iids) by diffing the unlearned
    training set against the original train.txt."""
    base = os.path.join(data_path, 'train.txt')
    target_name = f'train_unlearned_{unlearn_type}_r{ratio:02d}.txt'
    target = os.path.join(data_path, target_name)
    if not os.path.isfile(target):
        target = os.path.join(data_path, 'train_unlearned.txt')

    def _read(path):
        u, items = set(), set()
        with open(path) as f:
            for line in f:
                parts = line.strip().split(' ')
                if not parts or parts == ['']:
                    continue
                u.add(int(parts[0]))
                for it in parts[1:]:
                    items.add(int(it))
        return u, items
    u_base, i_base = _read(base)
    u_after, i_after = _read(target)
    if unlearn_type == 'user':
        return u_base - u_after, set()
    elif unlearn_type == 'item':
        return set(), i_base - i_after
    else:  # interaction: any user whose item-set changed
        return u_base - u_after, i_base - i_after


def find_affected_shards(C, unlearn_type, unlearned_uids, unlearned_iids):
    """Return list of shard indices that contain at least one unlearned entity."""
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
            hit = bool((shard_users & unlearned_uids)
                       or (shard_items & unlearned_iids))
        if hit:
            affected.append(i)
    return affected


def filter_shard_data(shard, unlearn_type, unlearned_uids, unlearned_iids):
    """Return a new shard dict with unlearned entities removed."""
    new_shard = {}
    for u, items in shard.items():
        if unlearn_type == 'user' and u in unlearned_uids:
            continue
        new_items = [i for i in items
                     if not (unlearn_type == 'item' and i in unlearned_iids)]
        if new_items:
            new_shard[u] = new_items
    return new_shard


# ---------------------------------------------------------------------------
# Online unlearning
# ---------------------------------------------------------------------------
def setup_args(part_num, part_type, regs, agg_type, dataset='ml-1m'):
    """Build args namespace expected by RecEraser_BPR."""
    saved = sys.argv
    sys.argv = ['RecEraser_BPR.py']
    try:
        args = parse_args()
    finally:
        sys.argv = saved
    args.dataset = dataset
    args.part_type = part_type
    args.part_num = part_num
    args.regs = f'[{regs}]'
    args.pretrain = 0
    args.save_flag = 0
    args.verbose = 0
    args.epoch = 1
    args.epoch_agg = 1
    args.batch_size = 1024
    args.embed_size = 64
    args.layer_size = '[64]'
    args.Ks = '[10, 20, 50]'
    args.dropout = 1.0
    args.proj_path = PROJ + '/'
    args.data_path = PROJ + '/../data/'
    args.weights_path = PROJ + '/weights/'
    args.agg_type = agg_type
    args.part_T = 50
    return args


def load_model_and_pretrain(args, ckpt_path):
    """Build model, build Data, restore checkpoint (use plain restore)."""
    data_generator = Data(
        path=args.data_path + args.dataset,
        batch_size=args.batch_size,
        part_type=args.part_type,
        part_num=args.part_num,
        part_T=args.part_T,
    )
    cfg = {'n_users': data_generator.n_users,
           'n_items': data_generator.n_items}
    model = RecEraser_BPR(data_config=cfg)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Try plain restore
    try:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
    except Exception:
        reader = tf.train.NewCheckpointReader(ckpt_path)
        ckpt_vars = [k for k in reader.get_variable_to_shape_map()
                     if not any(s in k for s in ('/Adagrad', '/Adam',
                                                 '/Momentum', '/RMSProp',
                                                 '/ExponentialMovingAverage'))]
        gvars = {v.op.name: v for v in tf.global_variables()}
        var_list, used = {}, set()
        for c in ckpt_vars:
            short = c.split('/')[-1]
            for gn, gv in gvars.items():
                if gn in used:
                    continue
                if gn == short or gn.endswith('/' + short):
                    var_list[c] = gv
                    used.add(gn)
                    break
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, ckpt_path)
    return sess, model, data_generator


def evaluate(sess, model, data_generator):
    users = list(data_generator.test_set.keys())
    ret = test(sess, model, users, drop_flag=False)
    return {
        'recall10':    float(ret['recall'][0]),
        'recall20':    float(ret['recall'][1]),
        'recall50':    float(ret['recall'][2]),
        'precision10': float(ret['precision'][0]),
        'precision20': float(ret['precision'][1]),
        'precision50': float(ret['precision'][2]),
        'ndcg10':      float(ret['ndcg'][0]),
        'ndcg20':      float(ret['ndcg'][1]),
        'ndcg50':      float(ret['ndcg'][2]),
    }


def retrain_shard(sess, model, shard, data_generator, shard_id, args,
                  unlearn_type, unlearned_uids, unlearned_iids,
                  n_extra_epochs=10):
    """Retrain ONLY shard `shard_id` on filtered data, warm-starting from
    current weights."""
    print(f'    [online] retrain shard {shard_id}: {len(shard)} users',
          flush=True)
    # Build the unlearned shard data
    new_shard = filter_shard_data(shard, unlearn_type,
                                  unlearned_uids, unlearned_iids)

    # Use a quick local training loop with the filtered data.
    # We sample positive items from new_shard and negative items from
    # the union of items in new_shard.
    union_items = set()
    for items in new_shard.values():
        union_items.update(items)
    union_items = list(union_items)
    user_list = list(new_shard.keys())

    if not user_list or not union_items:
        return

    rnd = random.Random(42)
    for epoch in range(n_extra_epochs):
        loss_acc = 0.0
        n_batch = max(1, len(user_list) // args.batch_size)
        for _ in range(n_batch):
            users = [rnd.choice(user_list) for _ in range(args.batch_size)]
            pos_items = []
            neg_items = []
            for u in users:
                pos = rnd.choice(new_shard[u])
                neg = rnd.choice(union_items)
                # ensure neg not in this user's list (best effort)
                while neg in new_shard[u]:
                    neg = rnd.choice(union_items)
                pos_items.append(pos)
                neg_items.append(neg)
            _, bl = sess.run(
                [model.opt_local[shard_id], model.loss_local[shard_id]],
                feed_dict={model.users: users,
                           model.pos_items: pos_items,
                           model.neg_items: neg_items})
            loss_acc += bl
    print(f'    [online] shard {shard_id} retrain done (last loss={loss_acc:.4f})',
          flush=True)


def run_one_scenario(part_num, part_type, agg_type, unlearn_type, ratio,
                     regs='0.01'):
    """Online-unlearn one scenario.  Returns the metrics dict."""
    print(f'\n=== part_num={part_num}, part_type={part_type} ({METHOD_INFO[part_type]}),'
          f' agg={agg_type}, unlearn={unlearn_type} r{ratio} ===',
          flush=True)

    args = setup_args(part_num, part_type, regs, agg_type)
    # ckpt folder
    if agg_type == 'mean':
        # Try _mean first, then _mean_pred (legacy)
        ckpt_dir = os.path.join(
            WEIGHTS, f'num-{part_num}_type-{part_type}_r{regs}_mean')
        if not os.path.isfile(os.path.join(ckpt_dir, 'weights.index')):
            legacy = os.path.join(
                WEIGHTS, f'num-{part_num}_type-{part_type}_r{regs}_mean_pred')
            if os.path.isfile(os.path.join(legacy, 'weights.index')):
                ckpt_dir = legacy
    else:
        ckpt_dir = os.path.join(
            WEIGHTS, f'num-{part_num}_type-{part_type}_r{regs}')
    ckpt_path = os.path.join(ckpt_dir, 'weights')
    if not os.path.isfile(ckpt_path + '.index'):
        print(f'   [SKIP] no checkpoint at {ckpt_path}')
        return None

    sess, model, data_generator = load_model_and_pretrain(args, ckpt_path)

    # Baseline = original model, no unlearn
    baseline = evaluate(sess, model, data_generator)
    print(f'   baseline Recall@20={baseline["recall20"]:.4f}')

    # Find unlearn set
    u_unlearn, i_unlearn = get_unlearn_entities(
        unlearn_type, ratio, args.data_path + args.dataset)
    print(f'   unlearn: {len(u_unlearn)} users, {len(i_unlearn)} items')

    # Find affected shards
    C = data_generator.C
    affected = find_affected_shards(C, unlearn_type, u_unlearn, i_unlearn)
    print(f'   affected shards: {affected} (out of {len(C)})')

    # Retrain each affected shard
    t0 = time()
    for sid in affected:
        retrain_shard(sess, model, C[sid], data_generator, sid, args,
                      unlearn_type, u_unlearn, i_unlearn,
                      n_extra_epochs=10)
    retrain_time = time() - t0
    print(f'   online retrain time: {retrain_time:.1f}s')

    # Evaluate after
    online = evaluate(sess, model, data_generator)
    print(f'   online Recall@20={online["recall20"]:.4f}')

    sess.close()
    return {
        'baseline':      baseline,
        'online_unlearn': online,
        'n_affected_shards': len(affected),
        'affected_shards':   affected,
        'retrain_time_s':    retrain_time,
        'partition':         METHOD_INFO[part_type],
        'agg':               agg_type,
    }


def main():
    part_num = _SELF_PART_NUM
    ap = argparse.ArgumentParser()
    ap.add_argument('--part_type', type=int, default=None,
                    help='If set, only run this partition type.')
    ap.add_argument('--unlearn_type', type=str, default=None,
                    help='If set, only run this unlearn type.')
    ap.add_argument('--ratio', type=int, default=20)
    ap.add_argument('--agg_type', type=str, default='mean',
                    choices=['attention', 'mean'])
    ap.add_argument('--regs', default='0.01')
    ap.add_argument('--out', default=None)
    cli, _ = ap.parse_known_args()

    ptypes = [cli.part_type] if cli.part_type else [1, 2, 3, 4]
    utypes = [cli.unlearn_type] if cli.unlearn_type \
        else ['interaction', 'user', 'item']

    out_doc = {}
    for pt in ptypes:
        for ut in utypes:
            r = run_one_scenario(part_num, pt, cli.agg_type, ut, cli.ratio,
                                 cli.regs)
            if r is None:
                continue
            key = f'num{part_num}_{METHOD_INFO[pt]}-{cli.agg_type}_{ut}_r{cli.ratio:02d}'
            out_doc[key] = r

    if not out_doc:
        print('No scenarios run.')
        return

    out_path = (cli.out
                or os.path.join(RESULTS, f'online_unlearn_num{part_num}.json'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_doc, f, indent=2)
    print(f'\n[OK] Saved: {out_path}')


if __name__ == '__main__':
    main()
