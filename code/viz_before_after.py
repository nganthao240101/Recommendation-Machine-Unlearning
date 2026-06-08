"""
Before/After unlearn comparison table.

For each (unlearn_type, ratio, partition, aggregation) we display the
metric values BEFORE (from baseline checkpoint) and AFTER unlearning, in
the same cell.  Color encodes the AFTER value, with delta in parentheses.

Layout per panel = one (unlearn_type, ratio) scenario:
  rows: 9 metrics (Recall/precision/NDCG @10/20/50)
  cols: 8 method-aggregations (InP-mean/att, UBP-mean/att, IBP-mean/att,
        Random-mean/att)

Inputs:
  results/unlearn_eval_num5.json
    must contain keys like:
       num5_baseline
       num5_interaction_r05 / r10 / r20
       num5_user_r05 / r10 / r20
       num5_item_r05 / r10 / r20

Output:
  results/before_after_unlearn_num5.png

Usage:
  python viz_before_after.py 5
"""
import os
import sys
import json
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
COL_ORDER = [f'{p}-{a}' for p in PARTITIONS for a in AGG_TYPES]
PARTITION_COLOR = {
    'InP': '#A23B72', 'UBP': '#2E86AB',
    'IBP': '#F18F01', 'Random': '#888888',
}


def make_diff_block(baseline, after, focus='recall20'):
    """Return list of (baseline_val, after_val, delta_pct) per column."""
    out = []
    for col in COL_ORDER:
        b = baseline.get(col, {}).get(focus)
        a = after.get(col, {}).get(focus)
        if b is None or a is None or b == 0:
            out.append((b, a, None))
        else:
            out.append((b, a, (a - b) / b * 100))
    return out


def draw_panel(ax, baseline, after, title, focus_metric='recall20'):
    """Draw a 9-row x 8-col table.  Each cell shows before / after / delta."""
    n_rows, n_cols = len(METRICS), len(COL_ORDER)
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

    # Color map based on AFTER value across this panel
    all_after = []
    for col in COL_ORDER:
        v = after.get(col, {}).get(focus_metric)
        if v is not None:
            all_after.append(v)
    if not all_after:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='#999999')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=40)
        return
    vmin, vmax = min(all_after), max(all_after)
    if vmax == vmin:
        vmax = vmin + 1e-9
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i, (mkey, mname) in enumerate(METRICS):
        ax.text(-0.05, i + 0.5, mname,
                ha='right', va='center', fontsize=9, fontweight='bold')
        # gather row max for bolding
        row_after = [after.get(c, {}).get(mkey) for c in COL_ORDER]
        row_max = max((v for v in row_after if v is not None), default=None)
        for j, col in enumerate(COL_ORDER):
            b = baseline.get(col, {}).get(mkey)
            a = after.get(col, {}).get(mkey)
            if a is None and b is None:
                ax.add_patch(mpatches.Rectangle(
                    (j, i), 1, 1, facecolor='#EEEEEE',
                    edgecolor='#CCCCCC', hatch='//', zorder=1))
                ax.text(j + 0.5, i + 0.5, 'N/A',
                        ha='center', va='center', fontsize=7, color='#999999')
                continue
            # Color by AFTER value
            color = cmap(norm(a)) if a is not None else '#EEEEEE'
            ax.add_patch(mpatches.Rectangle(
                (j, i), 1, 1, facecolor=color,
                edgecolor='white', zorder=1))
            is_best = (a is not None and row_max is not None
                       and a == row_max)
            if b is not None and a is not None and b != 0:
                delta_pct = (a - b) / b * 100
                sign = '+' if delta_pct >= 0 else ''
                cell_text = f'{a:.4f}\n({sign}{delta_pct:.1f}%)'
            else:
                cell_text = f'{a:.4f}' if a is not None else 'N/A'
            ax.text(j + 0.5, i + 0.5, cell_text,
                    ha='center', va='center', fontsize=6.5,
                    fontweight='bold' if is_best else 'normal')

    ax.set_title(title, fontsize=11, fontweight='bold', pad=40)


def order_scenarios(doc):
    """Return list of (key, title) in display order."""
    groups = {'baseline': [], 'interaction': [], 'user': [], 'item': []}
    for key in doc.keys():
        parts = key.split('_')
        if 'baseline' in parts:
            groups['baseline'].append((key, 'baseline (no unlearn)'))
        else:
            ratio_tok = next((t for t in parts
                              if t.startswith('r') and t[1:].isdigit()), None)
            ratio = int(ratio_tok[1:]) if ratio_tok else 0
            for ut in ('interaction', 'user', 'item'):
                if ut in parts:
                    groups[ut].append(
                        (key, f'{ut} unlearn — {ratio}%'))
                    break
    ordered = sorted(groups['baseline'], key=lambda x: x[0])
    for ut in ('interaction', 'user', 'item'):
        ordered += sorted(groups[ut], key=lambda x: x[0])
    return ordered


def main():
    part_num = 5
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        part_num = int(sys.argv[1])
    json_path = os.path.join(RESULTS, f'unlearn_eval_num{part_num}.json')
    if not os.path.exists(json_path):
        print(f'Missing {json_path}. Run:')
        print(f'   python unlearn_metrics.py {part_num}')
        return
    with open(json_path) as f:
        doc = json.load(f)
    print(f'Loaded {json_path} ({len(doc)} scenarios)')

    baseline_key = next((k for k in doc if k.endswith('_baseline')), None)
    if baseline_key is None:
        print('No baseline scenario in JSON.')
        return
    baseline = doc[baseline_key]
    print(f'Baseline: {baseline_key}')

    scenarios = order_scenarios(doc)
    after_keys = [k for k, _ in scenarios if k != baseline_key]
    if not after_keys:
        print('No after-unlearn scenarios to compare.')
        return

    n = len(after_keys)
    fig, axes = plt.subplots(n, 1,
                             figsize=(14, 1.6 + n * 5.0),
                             squeeze=False)
    fig.suptitle(
        f'RecEraser BPR on ml-1m (part_num={part_num}): '
        f'Before vs After unlearn\n'
        f'(cell: after value with delta% vs baseline in parentheses; '
        f'color encodes after value, bold = best in row)',
        fontsize=12, fontweight='bold', y=0.995)

    for ax_row, (key, _title) in zip(axes, after_keys):
        title = key.replace(f'num{part_num}_', '').replace('_', ' ')
        draw_panel(ax_row[0], baseline, doc[key], title,
                   focus_metric='recall20')

    fig.text(0.5, 0.01,
             'Green = higher after-unlearn.  Bold = best after-value in row.  '
             '/ = N/A.',
             ha='center', fontsize=9, style='italic', color='#555555')
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = os.path.join(RESULTS, f'before_after_unlearn_num{part_num}.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'[OK] Saved: {out}')


if __name__ == '__main__':
    main()
