"""
Visualize results by evaluating trained checkpoints.

Loads the saved TensorFlow checkpoints for each (part_num, part_type) combination
under code/weights/ml-1m/RecEraser_BPR/, runs evaluation on the test set,
saves the numbers to results/eval_metrics.json, and produces the comparison charts.

Usage:
  python viz_from_ckpt.py           # default: part_num=10
  python viz_from_ckpt.py 5         # evaluate num-5 checkpoints
"""
import os
import sys
import json
import types
import warnings
warnings.filterwarnings('ignore')

# Strip our own positional argument BEFORE any parser runs.
# utility/batch_test.py calls parse_args() at import time, so we have to
# clean sys.argv first.
_SELF_PART_NUM = 10
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    _SELF_PART_NUM = int(sys.argv[1])
    del sys.argv[1:2]

import numpy as np

# Make the entire script behave like TF 1.x. The model code uses tf.placeholder,
# tf.Session, tf.global_variables_initializer, tf.truncated_normal, tf.div,
# tf.contrib.layers.xavier_initializer — all of which are removed in TF 2.x.
# We replace `tensorflow` and `tensorflow.compat.v1` in sys.modules with the TF1
# compat module so that `import tensorflow as tf` inside RecEraser_BPR.py picks
# up the compat.v1 module, not TF2.
import tensorflow as _tf_raw
_tf_v1 = _tf_raw.compat.v1
_tf_v1.disable_eager_execution()

# Build a Xavier/Glorot initializer without going through keras (which has a
# re-entrant import problem in some TF versions). Use the raw uniform xavier
# formula: U(-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out))).
import math as _math
import tensorflow.python.ops.init_ops as _init_ops
_XAVIER_FACTORY = _init_ops.VarianceScaling(
    scale=1.0, mode='fan_avg', distribution='uniform'
)
# Old TF1 signature passed the args via the layer; we accept and ignore them.
def _xavier_initializer(*args, **kwargs):
    return _XAVIER_FACTORY
_tf_v1.contrib = types.SimpleNamespace(
    layers=types.SimpleNamespace(xavier_initializer=_xavier_initializer)
)
# Hide the TF2-only name 'tensorflow' from the import system so that
# `import tensorflow as tf` resolves to our compat module.
sys.modules['tensorflow'] = _tf_v1
# Also expose `tf` as a name in our own script
tf = _tf_v1
# Expose a handful of TF2 module names that batch_test / load_data expect
import tensorflow as _tf_full
sys.modules.setdefault('tf_slim', types.SimpleNamespace())

# Project paths
PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

from utility.parser import parse_args
from utility.load_data import Data
from utility.batch_test import test
from RecEraser_BPR import RecEraser_BPR
from time import time


# Map part_type -> method name + color
METHOD_INFO = {
    1: {'name': 'InBP',  'color': '#A23B72'},
    2: {'name': 'UBP',   'color': '#2E86AB'},
    3: {'name': 'Random','color': '#888888'},
    4: {'name': 'IBP',   'color': '#F18F01'},
}


def evaluate_one(part_type, part_num, regs='0.01'):
    """Load one checkpoint, evaluate on test set, return metric dict."""
    # Re-parse args with empty argv so our positional arg doesn't trigger it
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
    args.agg_type = 'attention'
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

    print(f'\n>>> Evaluating part_type={part_type} (part_num={part_num})')
    t0 = time()

    data_generator = Data(
        path=args.data_path + args.dataset,
        batch_size=args.batch_size,
        part_type=args.part_type,
        part_num=args.part_num,
        part_T=args.part_T,
    )

    config = {
        'n_users': data_generator.n_users,
        'n_items': data_generator.n_items,
    }

    model = RecEraser_BPR(data_config=config)

    # Try restoring
    ckpt_root = os.path.join(
        PROJ, 'weights', args.dataset, 'RecEraser_BPR',
        f'num-{part_num}_type-{part_type}_r{regs}'
    )
    weights_path = os.path.join(ckpt_root, 'weights')

    config_sess = tf.compat.v1.ConfigProto()
    config_sess.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config_sess)
    sess.run(tf.compat.v1.global_variables_initializer())

    # Inspect checkpoint variables once
    if os.environ.get('DUMP_CKPT') == '1':
        try:
            reader = tf.compat.v1.train.NewCheckpointReader(weights_path)
            print('   [DUMP] Checkpoint vars:')
            for k in sorted(reader.get_variable_to_shape_map()):
                print(f'     {k}  {reader.get_variable_to_shape_map()[k]}')
            print('   [DUMP] Graph vars:')
            for v in tf.global_variables():
                print(f'     {v.op.name}  {v.shape}')
        except Exception as e:
            print(f'   [DUMP] Failed: {e}')

    # Try plain restore first, then build a name-aware var_list.
    restored = False
    try:
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, weights_path)
        print(f'   [OK] Restored from {weights_path}')
        restored = True
    except Exception as e1:
        reader = tf.compat.v1.train.NewCheckpointReader(weights_path)
        ckpt_var_to_shape = reader.get_variable_to_shape_map()
        # Filter out optimizer slots (e.g. Adagrad accumulators) — they are not
        # in our model graph.
        ckpt_vars = [k for k in ckpt_var_to_shape
                     if not (k.endswith('/Adagrad') or '/Adam' in k or
                             '/ExponentialMovingAverage' in k or
                             '/Momentum' in k or '/RMSProp' in k)]
        graph_var_names = {v.op.name: v for v in tf.global_variables()}
        var_list = {}
        used_graph = set()
        for ckpt_name in ckpt_vars:
            short = ckpt_name.split('/')[-1]  # drop leading scope
            match = None
            for gname, gvar in graph_var_names.items():
                if gname in used_graph:
                    continue
                if gname == short or gname.endswith('/' + short):
                    match = gvar
                    break
            if match is not None:
                var_list[ckpt_name] = match
                used_graph.add(match.op.name)
        try:
            saver2 = tf.compat.v1.train.Saver(var_list=var_list)
            saver2.restore(sess, weights_path)
            print(f'   [OK] Restored (mapped, {len(var_list)} vars) from {weights_path}')
            restored = True
        except Exception as e2:
            print(f'   [FAIL] plain={e1!r}\n          mapped={e2!r}')
            sess.close()
            return None

    users_to_test = list(data_generator.test_set.keys())
    ret = test(sess, model, users_to_test, drop_flag=False)
    sess.close()

    # ret['recall'] is array of length len(Ks) sorted by Ks
    Ks = [10, 20, 50]
    out = {
        'recall10': float(ret['recall'][0]),
        'recall20': float(ret['recall'][1]),
        'recall50': float(ret['recall'][2]),
        'precision10': float(ret['precision'][0]),
        'precision20': float(ret['precision'][1]),
        'precision50': float(ret['precision'][2]),
        'ndcg10': float(ret['ndcg'][0]),
        'ndcg20': float(ret['ndcg'][1]),
        'ndcg50': float(ret['ndcg'][2]),
    }
    dt = time() - t0
    print(f'   Recall@20 = {out["recall20"]:.4f}   ({dt:.1f}s)')
    return out


def main():
    part_num = _SELF_PART_NUM

    # Detect which (part_type, part_num, regs) checkpoints are available
    ckpt_root = os.path.join(PROJ, 'weights', 'ml-1m', 'RecEraser_BPR')
    available = []
    for d in os.listdir(ckpt_root):
        # format: num-{n}_type-{t}_r{regs}
        try:
            parts = d.split('_')
            n = int(parts[0].split('-')[1])
            t = int(parts[1].split('-')[1])
            regs = parts[2].lstrip('r')
            if n == part_num and t in METHOD_INFO:
                available.append((t, regs))
        except Exception:
            continue
    available.sort()

    if not available:
        print(f'No checkpoints found for part_num={part_num} in {ckpt_root}')
        return

    print(f'Found {len(available)} checkpoints for part_num={part_num}:')
    for t, r in available:
        print(f'  type={t} ({METHOD_INFO[t]["name"]}) regs={r}')

    # Evaluate each
    results = {}
    for part_type, regs in available:
        name = METHOD_INFO[part_type]['name']
        m = evaluate_one(part_type, part_num, regs)
        if m is not None:
            m['color'] = METHOD_INFO[part_type]['color']
            results[name] = m

    if not results:
        print('No successful evaluations.')
        return

    # Save metrics to JSON
    os.makedirs(os.path.join(PROJ, '..', 'results'), exist_ok=True)
    out_json = os.path.join(PROJ, '..', 'results', f'eval_metrics_num{part_num}.json')
    with open(out_json, 'w') as f:
        json.dump({f'num{part_num}': results}, f, indent=2)
    print(f'\n[OK] Saved metrics: {out_json}')

    # Now visualize
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    method_list = [m for m in ['Random', 'InBP', 'IBP', 'UBP'] if m in results]
    k = ['K=10', 'K=20', 'K=50']
    x = np.arange(len(k))
    w = 0.2

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f'RecEraser BPR on ml-1m: Partition Methods Comparison (part_num={part_num})',
        fontsize=14, fontweight='bold'
    )

    def grouped_bars(ax, key_prefix, ylabel, title, ymax):
        for i, method in enumerate(method_list):
            vals = [results[method][f'{key_prefix}{k_[-2:]}'] for k_ in k]
            bars = ax.bar(x + i*w, vals, w, label=method,
                          color=results[method]['color'], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                        f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
        ax.set_xlabel('Top-K')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x + w*1.5)
        ax.set_xticklabels(k)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, ymax)

    grouped_bars(axes[0, 0], 'recall',     'Recall',    'Recall@K',    0.42)
    grouped_bars(axes[0, 1], 'precision',  'Precision', 'Precision@K', 0.32)
    grouped_bars(axes[0, 2], 'ndcg',       'NDCG',      'NDCG@K',      0.38)

    # Ranking
    ax4 = axes[1, 0]
    sorted_m = sorted(method_list, key=lambda mm: -results[mm]['recall20'])
    sorted_r = [results[mm]['recall20'] for mm in sorted_m]
    bars = ax4.barh(sorted_m, sorted_r,
                    color=[results[mm]['color'] for mm in sorted_m], alpha=0.85)
    for bar, v in zip(bars, sorted_r):
        ax4.text(v + 0.003, bar.get_y() + bar.get_height()/2,
                 f'{v:.4f}', va='center', fontweight='bold')
    ax4.set_xlabel('Recall@20')
    ax4.set_title('Recall@20 Ranking')
    ax4.grid(axis='x', alpha=0.3)
    ax4.set_xlim(0, max(0.25, max(sorted_r) * 1.15))

    # Improvement vs Random
    ax5 = axes[1, 1]
    if 'Random' in results:
        base = results['Random']['recall20']
        improvements = [(results[mm]['recall20'] - base) / base * 100
                        for mm in method_list]
    else:
        base = min(results[mm]['recall20'] for mm in method_list)
        improvements = [(results[mm]['recall20'] - base) / base * 100
                        for mm in method_list]
    colors_imp = ['#666666' if i == 0 else results[mm]['color']
                  for i, mm in enumerate(method_list)]
    bars = ax5.bar(method_list, improvements, color=colors_imp, alpha=0.85)
    for bar, v in zip(bars, improvements):
        lbl = f'{v:+.1f}%' if v != 0 else 'baseline'
        ax5.text(bar.get_x() + bar.get_width()/2, v + (0.3 if v >= 0 else -0.6),
                 lbl, ha='center', fontweight='bold')
    ax5.set_ylabel('Improvement vs Random (%)')
    ax5.set_title('Recall@20 Improvement vs Random')
    ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax5.grid(axis='y', alpha=0.3)

    # Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    summary = '+' + '=' * 60 + '+\n'
    summary += f'| RECALL@20: PARTITION METHODS (part_num={part_num}) |\n'
    summary += '+' + '=' * 60 + '+\n'
    summary += '| Rank | Method   | Recall@20 | vs Random  |\n'
    summary += '+' + '=' * 60 + '+\n'
    for i, mm in enumerate(method_list):
        rank = sorted_m.index(mm) + 1
        rank_str = f'#{rank}'
        summary += f'| {rank_str:<5} | {mm:<8} | {results[mm]["recall20"]:.4f}   | {improvements[i]:+7.1f}%  |\n'
    summary += '+' + '=' * 60 + '+\n'
    ax6.text(0.0, 0.5, summary, fontsize=10, fontfamily='monospace',
             verticalalignment='center', transform=ax6.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    out_png1 = os.path.join(PROJ, '..', 'results',
                            f'partition_comparison_num{part_num}.png')
    plt.savefig(out_png1, dpi=120, bbox_inches='tight')
    print(f'[OK] Saved: {out_png1}')
    plt.close()

    # Simple bar
    fig2, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sorted_m, sorted_r,
                  color=[results[mm]['color'] for mm in sorted_m],
                  alpha=0.9, edgecolor='black', linewidth=1.5)
    for bar, v in zip(bars, sorted_r):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall@20', fontsize=12)
    ax.set_title(f'RecEraser BPR on ml-1m: Recall@20 by Partition Method (part_num={part_num})',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(0.25, max(sorted_r) * 1.15))
    out_png2 = os.path.join(PROJ, '..', 'results',
                            f'recall20_comparison_num{part_num}.png')
    plt.savefig(out_png2, dpi=120, bbox_inches='tight')
    print(f'[OK] Saved: {out_png2}')
    plt.close()


if __name__ == '__main__':
    main()
