"""
Run a full evaluation across (unlearn_type x unlearn_ratio x partition x agg).

Workflow per cell:
  1) load the saved RecEraser_BPR checkpoint for the given part_type
  2) re-load the test set (always the same) and the unlearned train set
     (so that C, C_U, C_I partitions match the unlearned data)
  3) measure Recall/precision/NDCG @10/20/50 with both aggregations
  4) append to a single JSON keyed as `num{part_num}_{unlearn_tag}`

If a checkpoint for the unlearned variant does not exist, the cell is
skipped (the entry is not written).

Usage:
  python unlearn_metrics.py 5           # for part_num=5
"""
import os
import sys
import json
import glob
import types
import warnings
warnings.filterwarnings('ignore')

# Strip our positional arg before parse_args() runs in batch_test.
_SELF_PART_NUM = 10
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
sys.path.insert(0, PROJ)

from utility.parser import parse_args
from utility.load_data import Data
from utility.batch_test import test
from RecEraser_BPR import RecEraser_BPR
from time import time


METHOD_INFO = {1: 'InP', 2: 'UBP', 3: 'Random', 4: 'IBP'}
METHOD_COLOR = {
    1: '#A23B72', 2: '#2E86AB', 3: '#888888', 4: '#F18F01',
}
AGG_TYPES = ['attention', 'mean']


def evaluate_one(part_type, part_num, agg_type, regs='0.01',
                 unlearn_tag=None):
    """Eval the saved checkpoint with the requested aggregation."""
    saved_argv = sys.argv
    sys.argv = ['RecEraser_BPR.py']
    try:
        args = parse_args()
    finally:
        sys.argv = saved_argv

    args.dataset = 'ml-1m'
    args.part_type = part_type
    args.part_num = part_num
    args.regs = f'[{regs}]'
    args.agg_type = agg_type
    args.pretrain = 1
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

    name = METHOD_INFO[part_type]
    suffix = f' [{unlearn_tag}]' if unlearn_tag else ''
    print(f'\n>>> {name:6s} (part_num={part_num}, agg={agg_type}){suffix}',
          flush=True)
    t0 = time()

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

    ckpt_root = os.path.join(
        PROJ, 'weights', args.dataset, 'RecEraser_BPR',
        f'num-{part_num}_type-{part_type}_r{regs}'
    )
    if unlearn_tag:
        ckpt_root = os.path.join(ckpt_root, unlearn_tag)
    weights_path = os.path.join(ckpt_root, 'weights')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    try:
        saver = tf.train.Saver()
        saver.restore(sess, weights_path)
    except Exception:
        reader = tf.train.NewCheckpointReader(weights_path)
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
        try:
            saver2 = tf.train.Saver(var_list=var_list)
            saver2.restore(sess, weights_path)
        except Exception as e:
            print(f'   [FAIL] restore: {e}', flush=True)
            sess.close()
            return None

    users_to_test = list(data_generator.test_set.keys())
    ret = test(sess, model, users_to_test, drop_flag=False)
    sess.close()

    out = {
        'recall10':    float(ret['recall'][0]),
        'recall20':    float(ret['recall'][1]),
        'recall50':    float(ret['recall'][2]),
        'precision10': float(ret['precision'][0]),
        'precision20': float(ret['precision'][1]),
        'precision50': float(ret['precision'][2]),
        'ndcg10':      float(ret['ndcg'][0]),
        'ndcg20':      float(ret['ndcg'][1]),
        'ndcg50':      float(ret['ndcg'][2]),
        'partition':   name,
        'agg':         agg_type,
        'color':       METHOD_COLOR[part_type],
    }
    print(f'   Recall@20={out["recall20"]:.4f}  '
          f'NDCG@20={out["ndcg20"]:.4f}  ({time()-t0:.0f}s)', flush=True)
    return out


def find_checkpoints(part_num, regs='0.01'):
    """Walk the weights dir; return list of (part_type, unlearn_tag_or_None)."""
    root = os.path.join(PROJ, 'weights', 'ml-1m', 'RecEraser_BPR')
    out = []
    if not os.path.isdir(root):
        return out
    for entry in sorted(os.listdir(root)):
        if not entry.startswith(f'num-{part_num}_type-'):
            continue
        try:
            pt = int(entry.split('_type-')[1].split('_')[0].split('-')[0])
        except Exception:
            continue
        full = os.path.join(root, entry)
        if os.path.isfile(os.path.join(full, 'weights.index')):
            out.append((pt, None))  # baseline checkpoint
        # look for nested unlearned variants
        for sub in sorted(os.listdir(full)):
            sub_full = os.path.join(full, sub)
            if (sub.startswith('interaction_') or
                    sub.startswith('user_') or
                    sub.startswith('item_')):
                if os.path.isfile(os.path.join(sub_full, 'weights.index')):
                    out.append((pt, sub))
    return out


def main():
    part_num = _SELF_PART_NUM
    ckpts = find_checkpoints(part_num)
    print(f'Found {len(ckpts)} checkpoint entries for part_num={part_num}:')
    for pt, tag in ckpts:
        name = METHOD_INFO[pt]
        print(f'  type={pt} ({name}) unlearn_tag={tag}')
    if not ckpts:
        return

    # Group by unlearn_tag so we emit one JSON block per scenario.
    by_tag = {}
    for pt, tag in ckpts:
        by_tag.setdefault(tag or 'baseline', []).append(pt)

    out_doc = {}
    for tag, pts in by_tag.items():
        print(f'\n=== Scenario: {tag} ===')
        results = {}
        for pt in sorted(pts):
            name = METHOD_INFO[pt]
            for agg in AGG_TYPES:
                r = evaluate_one(pt, part_num, agg, unlearn_tag=tag)
                if r is not None:
                    results[f'{name}-{agg}'] = r
        if results:
            key = f'num{part_num}_{tag}'
            out_doc[key] = results

    if not out_doc:
        print('No results collected.')
        return

    os.makedirs(os.path.join(PROJ, '..', 'results'), exist_ok=True)
    out_path = os.path.join(PROJ, '..', 'results',
                            f'unlearn_eval_num{part_num}.json')
    with open(out_path, 'w') as f:
        json.dump(out_doc, f, indent=2)
    print(f'\n[OK] Saved: {out_path}')


if __name__ == '__main__':
    main()
