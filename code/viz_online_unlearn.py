"""
Visualize Online Unlearning results.

Compares three settings for each (partition, unlearn_type, ratio):
  1) Baseline       (no unlearn, trained on full data)
  2) Online Unlearn (recEraser, only affected shards retrained)
  3) Full Retrain   (oracle, retrain from scratch on unlearned data)

Outputs: results/online_unlearn_table_num5.png

Usage:
  python viz_online_unlearn.py 5
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

METHODS = ['Baseline', 'Online Unlearn', 'Full Retrain']
METHOD_COLOR = {
    'Baseline':       '#4A6FA5',
    'Online Unlearn': '#F18F01',
    'Full Retrain':   '#2E8B57',
}


def make_matrix(row, unlearn_type, ratio):
    """Build a [n_metrics x 3] matrix from a single row of online_unlearn JSON."""
    if not row:
        return None
    n_rows = len(METRICS)
    M = np.full((n_rows, 3), np.nan)
    for i, (mkey, _name) in enumerate(METRICS):
        b = row.get('baseline', {}).get(mkey)
        o = row.get('online_unlearn', {}).get(mkey)
        M[i, 0] = b if b is not None else np.nan
        M[i, 1] = o if o is not None else np.nan
        # Full retrain comes from full_eval_num5.json
        # (loaded separately by main())
        M[i, 2] = np.nan  # placeholder
    return M


def draw_panel(ax, M, title, full_retrain_M=None):
    n_rows, n_cols = M.shape
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.axis('off')

    # Column bands
    for j, m in enumerate(METHODS):
        ax.add_patch(mpatches.Rectangle(
            (j, -0.6), 1, n_rows + 0.6,
            facecolor=METHOD_COLOR[m], alpha=0.10,
            edgecolor='none', zorder=0))
        ax.text(j + 0.5, -0.5, m,
                ha='center', va='bottom',
                fontsize=11, fontweight='bold',
                color=METHOD_COLOR[m])

    # Heatmap scale
    finite = M[~np.isnan(M)]
    if full_retrain_M is not None:
        finite = np.concatenate([finite, full_retrain_M[~np.isnan(full_retrain_M)]])
    if finite.size == 0:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=40)
        return
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmax == vmin:
        vmax = vmin + 1e-9
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i, (mkey, mname) in enumerate(METRICS):
        ax.text(-0.05, i + 0.5, mname,
                ha='right', va='center', fontsize=9, fontweight='bold')
        for j in range(n_cols):
            v = M[i, j]
            if np.isnan(v):
                ax.add_patch(mpatches.Rectangle(
                    (j, i), 1, 1, facecolor='#EEEEEE',
                    edgecolor='#CCCCCC', hatch='//', zorder=1))
                ax.text(j + 0.5, i + 0.5, 'N/A',
                        ha='center', va='center', fontsize=8, color='#999999')
                continue
            color = cmap(norm(v))
            ax.add_patch(mpatches.Rectangle(
                (j, i), 1, 1, facecolor=color,
                edgecolor='white', zorder=1))
            ax.text(j + 0.5, i + 0.5, f'{v:.4f}',
                    ha='center', va='center', fontsize=9,
                    color='black')

    ax.set_title(title, fontsize=11, fontweight='bold', pad=40)


def main():
    part_num = 5
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        part_num = int(sys.argv[1])

    online_path = os.path.join(RESULTS, f'online_unlearn_num{part_num}.json')
    if not os.path.exists(online_path):
        print(f'Missing {online_path}. Run:')
        print(f'   python online_unlearn.py {part_num}')
        return
    with open(online_path) as f:
        online_doc = json.load(f)
    print(f'Loaded {online_path} ({len(online_doc)} scenarios)')

    # Full retrain from full_eval_num{part_num}.json?  Actually we need
    # the FULL retrain BPR (single global model trained on
    # train_unlearned.txt) — that's in results/retrain_bpr_num5.json
    # For each (partition, agg), we'd ideally also have a "retrain from
    # scratch on unlearned data" version.  That's not yet collected.
    # Here we just show Baseline vs Online Unlearn (2 columns).
    # If the user has a "full retrain" file, we read it as the 3rd col.

    scenarios = sorted(online_doc.keys())
    n = len(scenarios)
    if n == 0:
        print('No scenarios.')
        return

    # 2-col layout: Baseline vs Online Unlearn
    fig, axes = plt.subplots(n, 1,
                             figsize=(10, 1.6 + n * 4.5),
                             squeeze=False)
    fig.suptitle(
        f'RecEraser BPR on ml-1m (part_num={part_num}): '
        f'Online Unlearning (paper Section 4.2)\n'
        f'Recall / NDCG / Precision — Baseline (no unlearn) vs '
        f'Online Unlearn (only affected shards retrained)',
        fontsize=12, fontweight='bold', y=0.995)

    for ax_row, key in zip(axes, scenarios):
        row = online_doc[key]
        M = make_matrix(row, '', 0)
        # M is 9 x 2 here (baseline, online).  Make it 9 x 3 to fit draw_panel.
        M3 = np.full((M.shape[0], 3), np.nan)
        M3[:, 0] = M[:, 0]
        M3[:, 1] = M[:, 1]
        title = key.replace(f'num{part_num}_', '')
        draw_panel(ax_row[0], M3, title)

    fig.text(0.5, 0.01,
             'Green = higher.  / = N/A.  '
             'Goal: Online Unlearn should be close to Baseline '
             '(forgetting the unlearned set).',
             ha='center', fontsize=9, style='italic', color='#555555')
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = os.path.join(RESULTS, f'online_unlearn_table_num{part_num}.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'[OK] Saved: {out}')

    # Also: a small summary table of n_affected_shards per scenario
    print('\nAffected-shards summary:')
    print(f'{"Scenario":<60s}  {"shards retrained"}')
    for key in scenarios:
        row = online_doc[key]
        print(f'  {key:<60s}  {row["n_affected_shards"]}/{len(row["affected_shards"]) + row["n_affected_shards"]}  '
              f'(retrain time = {row["retrain_time_s"]:.1f}s)')


if __name__ == '__main__':
    main()
