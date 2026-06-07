"""
Visualize 4 partition methods on a single model (RecEraser_BPR, part_num=5)
with the same hyperparameters.

Layout (matches the user-provided mockup):
  rows: 9 metrics (Recall/precision/NDCG @10/20/50)
  cols: Retrain, InP-Mean, InP-Att, UBP-Mean, UBP-Att, IBP-Mean, IBP-Att,
        Random-Mean, Random-Att
  cells: absolute value + heatmap background + bold per row

Inputs:
  - results/retrain_bpr_num5.json      (Retrain BPR on unlearned set)
  - results/full_eval_num5.json        (from run_full_eval.py 5)

Outputs:
  - results/table_4partitions_num5.png
  - results/table_4partitions_num5_focused.png
"""
import os
import sys
import json
import glob
import types
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


PROJ = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PROJ, '..', 'results')


# ---------------------------------------------------------------------------
# 1) Train Retrain BPR baseline on the unlearned training set
# ---------------------------------------------------------------------------
def train_retrain_bpr(part_num=5, regs='0.01', epoch=30, batch_size=1024,
                      embed_size=64):
    """Train a fresh BPR model on train_unlearned.txt (post-unlearn set).

    This is the "Retrain" baseline in the RecEraser paper: the same BPR
    architecture, but retrained from scratch on whatever data the user
    still has after the unlearning request. RecEraser checkpoints are also
    trained on train_unlearned.txt, so this gives an apples-to-apples
    upper-bound for what an "ideal" unlearning method could achieve.
    """
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

    sys.path.insert(0, PROJ)
    # BPRMF uses the same parser+helper as RecEraser.
    from utility.parser import parse_args
    from utility.load_data import Data as _Data
    from utility.batch_test import test
    from BPRMF import BPRMF
    from time import time

    # Monkey-patch: Retrain baseline must use the post-unlearn training set
    # (train_unlearned.txt) so the comparison with RecEraser is fair.
    # In this repo, train_unlearned.txt lives in the SAME folder as
    # train.txt (data/ml-1m/) — so we keep the path but override the
    # train_file at open time by wrapping open() inside the Data class.
    import builtins as _builtins
    _real_open = _builtins.open

    class Data(_Data):
        def __init__(self, path, batch_size, part_type, part_num, part_T):
            # Wrap built-in open so any read of "<path>/train.txt" inside
            # the parent constructor becomes "<path>/train_unlearned.txt".
            # Test file is left untouched.
            def _patched_open(file, *a, **kw):
                if isinstance(file, str) and file.endswith('/train.txt'):
                    file = file[:-len('/train.txt')] + '/train_unlearned.txt'
                return _real_open(file, *a, **kw)
            _builtins.open = _patched_open
            try:
                super().__init__(path, batch_size, part_type, part_num, part_T)
            finally:
                _builtins.open = _real_open

    saved = sys.argv
    sys.argv = ['BPRMF.py']
    try:
        args = parse_args()
    finally:
        sys.argv = saved

    args.dataset = 'ml-1m'
    args.regs = f'[{regs}]'
    args.pretrain = 0
    args.save_flag = 0
    args.verbose = 0
    args.epoch = epoch
    args.epoch_agg = 0
    args.batch_size = batch_size
    args.embed_size = embed_size
    args.layer_size = '[64]'
    args.Ks = '[10, 20, 50]'
    args.dropout = 1.0
    args.proj_path = PROJ + '/'
    args.data_path = PROJ + '/../data/'
    args.weights_path = PROJ + '/weights/'
    args.part_type = 0
    args.part_num = part_num
    args.part_T = 50
    args.agg_type = 'attention'

    print(f'>>> Retrain BPR on ml-1m (epoch={epoch})')
    t0 = time()
    data_generator = Data(
        path=args.data_path + args.dataset,
        batch_size=args.batch_size,
        part_type=0,
        part_num=part_num,
        part_T=args.part_T,
    )
    cfg = {'n_users': data_generator.n_users,
           'n_items': data_generator.n_items}
    model = BPRMF(data_config=cfg)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    cur_best, stopping = 0., 0
    for ep in range(args.epoch):
        n_batch = data_generator.n_train // args.batch_size + 1
        loss = mf = reg = 0.
        for _ in range(n_batch):
            u, pi, ni = data_generator.sample()
            _, l, ml, rl = sess.run(
                [model.opt, model.loss, model.mf_loss, model.reg_loss],
                feed_dict={model.users: u, model.pos_items: pi,
                           model.neg_items: ni,
                           model.dropout_keep_prob: args.dropout})
            loss += l
            mf += ml
            reg += rl
        if (ep + 1) % 1 != 0:
            continue
        users = list(data_generator.test_set.keys())
        ret = test(sess, model, users, drop_flag=False)
        if args.verbose:
            print(f'   retrain ep {ep:2d} loss={loss:.4f} '
                  f'recall20={ret["recall"][1]:.4f} ndcg20={ret["ndcg"][1]:.4f}')
        cur_best, stopping, stop = (
            max(cur_best, ret['recall'][1]),
            0 if ret['recall'][1] > cur_best else stopping + 1,
            stopping + 1 >= 10)
        if stop:
            break

    users = list(data_generator.test_set.keys())
    ret = test(sess, model, users, drop_flag=False)
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
    }
    sess.close()
    print(f'   retrain done  recall20={out["recall20"]:.4f} '
          f'ndcg20={out["ndcg20"]:.4f}  ({time()-t0:.0f}s)')
    return out


# ---------------------------------------------------------------------------
# 2) Render the heatmap table
# ---------------------------------------------------------------------------
METRICS = [
    ('recall10',    'Recall@10'),
    ('recall20',    'Recall@20'),
    ('recall50',    'Recall@50'),
    ('ndcg10',      'NDCG@10'),
    ('ndcg20',      'NDCG@20'),
    ('ndcg50',      'NDCG@50'),
    ('precision10', 'Precision@10'),
    ('precision20', 'Precision@20'),
    ('precision50', 'Precision@50'),
]

# Display column order (matches the screenshot).
COL_ORDER = [
    ('Retrain',  None,       '#555555'),
    ('InP',      'mean',     '#A23B72'),
    ('InP',      'attention','#A23B72'),
    ('UBP',      'mean',     '#2E86AB'),
    ('UBP',      'attention','#2E86AB'),
    ('IBP',      'mean',     '#F18F01'),
    ('IBP',      'attention','#F18F01'),
    ('Random',   'mean',     '#888888'),
    ('Random',   'attention','#888888'),
]

# Optional Mean-Paper column, only shown if results/mean_paper_num{N}.json
# is provided.  Inserted right after Retrain.
MEAN_PAPER_COL = ('Mean-Paper', None, '#000000')

PARTITION_BG = {
    'Retrain': '#EEEEEE',
    'InP':    '#A23B72',
    'UBP':    '#2E86AB',
    'IBP':    '#F18F01',
    'Random': '#888888',
}


def make_matrix(retrain, full, mean_paper=None, use_mean_paper=True):
    """rows = metrics, cols = COL_ORDER (+ optional Mean-Paper) entries."""
    n_rows = len(METRICS)
    if use_mean_paper and mean_paper:
        cols = [MEAN_PAPER_COL] + COL_ORDER
    else:
        cols = COL_ORDER
    n_cols = len(cols)
    matrix = np.full((n_rows, n_cols), np.nan)
    for j, (part, agg, _color) in enumerate(cols):
        if part == 'Retrain':
            src = retrain
        elif part == 'Mean-Paper':
            src = mean_paper
        else:
            key = f'{part}-{agg}'
            src = (full or {}).get(key)
        for i, (mkey, _name) in enumerate(METRICS):
            if src is not None and mkey in src:
                matrix[i, j] = src[mkey]
    return matrix


def draw_table(ax, matrix, title, show_partition_bands=True):
    n_rows, n_cols = matrix.shape
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.axis('off')

    # Column group bands (one per partition, width = 2 for InP/UBP/IBP/Random,
    # width = 1 for Retrain)
    if show_partition_bands:
        x = 0
        for part, agg, _ in COL_ORDER:
            w = 1
            ax.add_patch(mpatches.Rectangle(
                (x, -0.6), w, n_rows + 0.6,
                facecolor=PARTITION_BG[part], alpha=0.08,
                edgecolor='none', zorder=0))
            # Group header (partition name) at top
            ax.text(x + w / 2, -0.45, part,
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold',
                    color=PARTITION_BG[part])
            x += w

    # Sub-headers (aggregation tag for partition columns; nothing for Retrain)
    for j, (part, agg, _c) in enumerate(COL_ORDER):
        if part == 'Retrain':
            label = '—'
        else:
            label = (agg or '')[:3]
        ax.text(j + 0.5, -0.05, label, ha='center', va='top',
                fontsize=8, color='#444')

    # Heatmap cells
    finite = matrix[~np.isnan(matrix)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    if vmax == vmin:
        vmax = vmin + 1e-9
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i in range(n_rows):
        ax.text(-0.05, i + 0.5, METRICS[i][1],
                ha='right', va='center',
                fontsize=10, fontweight='bold')
        row_vals = matrix[i]
        row_max = np.nanmax(row_vals) if np.any(~np.isnan(row_vals)) else None
        for j in range(n_cols):
            v = matrix[i, j]
            if np.isnan(v):
                ax.add_patch(mpatches.Rectangle(
                    (j, i), 1, 1, facecolor='#EEEEEE',
                    edgecolor='#CCCCCC', hatch='//', zorder=1))
                ax.text(j + 0.5, i + 0.5, 'N/A',
                        ha='center', va='center',
                        fontsize=8, color='#999')
            else:
                color = cmap(norm(v))
                ax.add_patch(mpatches.Rectangle(
                    (j, i), 1, 1, facecolor=color,
                    edgecolor='white', zorder=1))
                is_best = (row_max is not None and v == row_max)
                ax.text(j + 0.5, i + 0.5, f'{v:.4f}',
                        ha='center', va='center', fontsize=9,
                        fontweight='bold' if is_best else 'normal',
                        color='black')

    ax.set_title(title, fontsize=12, fontweight='bold', pad=60)


def main():
    part_num = 5
    os.makedirs(RESULTS, exist_ok=True)

    # 1) Retrain BPR (skip if JSON already exists)
    retrain_json = os.path.join(RESULTS, 'retrain_bpr_num5.json')
    if os.path.exists(retrain_json):
        with open(retrain_json) as f:
            retrain = json.load(f)
        print(f'>>> Loaded cached retrain metrics: {retrain_json}')
    else:
        retrain = train_retrain_bpr(part_num=part_num, epoch=30)
        with open(retrain_json, 'w') as f:
            json.dump(retrain, f, indent=2)
        print(f'   [OK] Saved retrain metrics to {retrain_json}')

    # 2) 4 partitions x 2 aggregations (load existing JSON if present, else
    #    run the eval on the saved checkpoints)
    full_json = os.path.join(RESULTS, f'full_eval_num{part_num}.json')
    if os.path.exists(full_json):
        with open(full_json) as f:
            full_doc = json.load(f)
        full = full_doc.get(f'num{part_num}', full_doc)
        print(f'>>> Loaded cached partition metrics: {full_json}')
    else:
        print('No full_eval_num5.json found. Run:')
        print('   python run_full_eval.py 5')
        print('then re-run this script.')
        full = None

    # 2b) Optional Mean-Paper column — needs separate paper-faithful
    #     checkpoints (RecEraser_BPR_MeanPaper.py).  If absent, the column
    #     is dropped and the figure is the standard 9-col layout.
    mean_paper_json = os.path.join(RESULTS, f'mean_paper_num{part_num}.json')
    mean_paper = None
    use_mean_paper = False
    if os.path.exists(mean_paper_json):
        with open(mean_paper_json) as f:
            mean_paper = json.load(f)
        if mean_paper:
            use_mean_paper = True
            print(f'>>> Loaded Mean-Paper metrics: {mean_paper_json}')

    # 3) Build matrix and draw
    matrix = make_matrix(retrain, full, mean_paper=mean_paper,
                         use_mean_paper=use_mean_paper)
    n_rows, n_cols = matrix.shape
    fig, ax = plt.subplots(figsize=(15, 1.6 + n_rows * 0.55))
    draw_table(ax, matrix,
               title=f'RecEraser BPR on ml-1m (part_num={part_num}): '
                     f'Retrain vs 4 partition methods × 2 aggregations')
    fig.text(0.5, 0.01,
             'Green = higher.  Bold = best in row (N/A if checkpoint missing).',
             ha='center', fontsize=9, style='italic', color='#555')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out1 = os.path.join(RESULTS, f'table_4partitions_num{part_num}.png')
    plt.savefig(out1, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'[OK] Saved: {out1}')

    # 4) Focused version: only Recall/NDCG @20 + @50
    focused_metrics = [
        ('recall20', 'Recall@20'),
        ('recall50', 'Recall@50'),
        ('ndcg20',   'NDCG@20'),
        ('ndcg50',   'NDCG@50'),
    ]
    fmatrix = np.full((len(focused_metrics), n_cols), np.nan)
    for i, (mk, _name) in enumerate(focused_metrics):
        for j, (part, agg, _c) in enumerate(COL_ORDER):
            if part == 'Retrain':
                src = retrain
            else:
                key = f'{part}-{agg}'
                src = (full or {}).get(key)
            if src is not None and mk in src:
                fmatrix[i, j] = src[mk]

    fig2, ax2 = plt.subplots(figsize=(15, 0.9 + len(focused_metrics) * 0.7))
    draw_table(ax2, fmatrix,
               title=f'RecEraser BPR on ml-1m (part_num={part_num}): '
                     f'focused on Recall/NDCG @20 & @50')
    fig2.text(0.5, 0.01,
              'Focused view: top-line metrics only. Green = higher.',
              ha='center', fontsize=9, style='italic', color='#555')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out2 = os.path.join(RESULTS, f'table_4partitions_num{part_num}_focused.png')
    plt.savefig(out2, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'[OK] Saved: {out2}')


if __name__ == '__main__':
    main()
