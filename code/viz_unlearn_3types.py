"""
Visualize results for 3 unlearn types x N ratios x 4 partition methods.

For each (unlearn_type, ratio) we draw a small heatmap table:
  rows: 9 metrics (Recall/precision/NDCG @10/20/50)
  cols: 8 method-aggregation columns (InP-mean/att, UBP-mean/att, IBP-mean/att,
        Random-mean/att)

Inputs:
  results/unlearn_eval_num5.json
    = {
        "num5_baseline":          {"InP-attention": {...}, ...},
        "num5_interaction_r05":   {...},
        "num5_interaction_r10":   {...},
        "num5_interaction_r20":   {...},
        "num5_user_r05":          {...},
        "num5_user_r10":          {...},
        "num5_user_r20":          {...},
        "num5_item_r05":          {...},
        "num5_item_r10":          {...},
        "num5_item_r20":          {...},
      }

Output:
  results/table_unlearn_3types_num5.png    (1 figure with all blocks)

Usage:
  python viz_unlearn_3types.py 5
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

PARTITIONS = ['InP', 'UBP', 'IBP', 'Random']
AGG_TYPES = ['mean', 'attention']
COL_ORDER = []
for p in PARTITIONS:
    for a in AGG_TYPES:
        COL_ORDER.append(f'{p}-{a}')

PARTITION_COLOR = {
    'InP':    '#A23B72',
    'UBP':    '#2E86AB',
    'IBP':    '#F18F01',
    'Random': '#888888',
}


def make_matrix(block, retrain=None):
    """block = {Method-agg: {recall10:..., ...}}. Return [n_metrics x n_cols]."""
    n_rows, n_cols = len(METRICS), len(COL_ORDER)
    M = np.full((n_rows, n_cols), np.nan)
    for j, col in enumerate(COL_ORDER):
        src = block.get(col)
        if src is None:
            continue
        for i, (k, _name) in enumerate(METRICS):
            if k in src:
                M[i, j] = src[k]
    return M


def draw_table(ax, M, title):
    n_rows, n_cols = M.shape
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.axis('off')

    # Column group bands
    for i, p in enumerate(PARTITIONS):
        ax.add_patch(mpatches.Rectangle(
            (i * 2, 0), 2, n_rows,
            facecolor=PARTITION_COLOR[p], alpha=0.10,
            edgecolor='none', zorder=0))
        ax.text(i * 2 + 1, -0.5, p, ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=PARTITION_COLOR[p])

    for j, col in enumerate(COL_ORDER):
        ax.text(j + 0.5, -0.05, col.split('-')[1][:3],
                ha='center', va='top', fontsize=8, color='#444444')

    finite = M[~np.isnan(M)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    if vmax == vmin:
        vmax = vmin + 1e-9
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i in range(n_rows):
        ax.text(-0.05, i + 0.5, METRICS[i][1],
                ha='right', va='center', fontsize=10, fontweight='bold')
        row_max = np.nanmax(M[i]) if np.any(~np.isnan(M[i])) else None
        for j in range(n_cols):
            v = M[i, j]
            if np.isnan(v):
                ax.add_patch(mpatches.Rectangle(
                    (j, i), 1, 1, facecolor='#EEEEEE',
                    edgecolor='#CCCCCC', hatch='//', zorder=1))
                ax.text(j + 0.5, i + 0.5, 'N/A',
                        ha='center', va='center', fontsize=8, color='#999')
            else:
                color = cmap(norm(v))
                ax.add_patch(mpatches.Rectangle(
                    (j, i), 1, 1, facecolor=color,
                    edgecolor='white', zorder=1))
                is_best = (row_max is not None and v == row_max)
                ax.text(j + 0.5, i + 0.5, f'{v:.4f}',
                        ha='center', va='center', fontsize=9,
                        fontweight='bold' if is_best else 'normal')

    ax.set_title(title, fontsize=12, fontweight='bold', pad=40)


def order_scenarios(doc):
    """Return a list of (key, title) in display order."""
    by_type = {'baseline': [], 'interaction': [], 'user': [], 'item': []}
    for key in doc.keys():
        # e.g. "num5_interaction_r10" or "num5_baseline"
        parts = key.split('_')
        if 'baseline' in parts:
            by_type['baseline'].append((key, 'baseline (no unlearn)'))
        else:
            # find ratio token
            for tok in parts:
                if tok.startswith('r') and tok[1:].isdigit():
                    ratio = int(tok[1:])
                    break
            else:
                ratio = 0
            for ut in ('interaction', 'user', 'item'):
                if ut in parts:
                    by_type[ut].append((key, f'{ut} unlearn — {ratio}%'))
                    break
    ordered = []
    ordered += sorted(by_type['baseline'],
                      key=lambda x: x[0])
    for ut in ('interaction', 'user', 'item'):
        # sort by ratio
        ordered += sorted(by_type[ut],
                          key=lambda x: int(x[0].split('_r')[1]))
    return ordered


def main():
    part_num = 5
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        part_num = int(sys.argv[1])
    json_path = os.path.join(RESULTS, f'unlearn_eval_num{part_num}.json')
    if not os.path.exists(json_path):
        print(f'Missing {json_path}. Run:')
        print(f'   python unlearn_metrics.py {part_num}')
        # Synthesize a demo doc so the figure is still useful
        doc = {}
    else:
        with open(json_path) as f:
            doc = json.load(f)
        print(f'Loaded {json_path} ({len(doc)} scenarios)')

    scenarios = order_scenarios(doc)
    if not scenarios:
        print('No scenarios to render.')
        return

    n = len(scenarios)
    fig, axes = plt.subplots(n, 1,
                             figsize=(14, 2.0 + n * 5.0),
                             squeeze=False)
    fig.suptitle(
        f'RecEraser BPR on ml-1m (part_num={part_num}): '
        f'general utility across 3 unlearn types × ratios × 4 partitions',
        fontsize=13, fontweight='bold', y=0.995)

    for ax_row, (key, title) in zip(axes, scenarios):
        M = make_matrix(doc.get(key, {}))
        if M is None or np.all(np.isnan(M)):
            ax_row[0].axis('off')
            continue
        draw_table(ax_row[0], M, title)

    fig.text(0.5, 0.01,
             'Green = higher.  Bold = best in row.  / = N/A '
             '(no checkpoint for this scenario).',
             ha='center', fontsize=9, style='italic', color='#555')
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = os.path.join(RESULTS, f'table_unlearn_3types_num{part_num}.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'[OK] Saved: {out}')


if __name__ == '__main__':
    main()
