"""
Build a clean heatmap-style comparison table from the JSON output of run_full_eval.py.

Layout:
- rows: metrics (Recall@10, Recall@20, Recall@50, NDCG@10, NDCG@20, NDCG@50,
        Precision@10, Precision@20, Precision@50)
- cols: methods (Retrain / InP-Mean / InP-Att / UBP-Mean / UBP-Att /
                  IBP-Mean / IBP-Att / Random-Mean / Random-Att)

For each (metric, method) cell, we display the absolute value plus a heatmap
background. A second block is built for the percent-vs-Retrain view.
"""
import os
import sys
import json
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_latest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


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

# Partitions in the order we want columns
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


def collect(json_data, part_num_key):
    """Extract values from JSON into matrix [metrics x methods]."""
    if not json_data or part_num_key not in json_data:
        return None, None
    block = json_data[part_num_key]
    rows = []
    for key, _ in METRICS:
        row = []
        for col in COL_ORDER:
            if col in block:
                row.append(block[col][key])
            else:
                row.append(None)
        rows.append(row)
    return np.array([[v if v is not None else np.nan for v in r] for r in rows]), block


def draw_table(ax, matrix, block, title):
    """Render a heatmap-style table on the given axis."""
    n_rows, n_cols = matrix.shape
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.axis('off')

    # Column group backgrounds (4 partition groups of 2)
    for i, p in enumerate(PARTITIONS):
        c = PARTITION_COLOR[p]
        rect = mpatches.Rectangle(
            (i * 2, 0), 2, n_rows,
            facecolor=c, alpha=0.10, edgecolor='none', zorder=0
        )
        ax.add_patch(rect)
        # Header for partition
        ax.text(
            i * 2 + 1, -0.5, p, ha='center', va='bottom',
            fontsize=11, fontweight='bold', color=c
        )

    # Column headers (agg type)
    for j, col in enumerate(COL_ORDER):
        agg = col.split('-')[1]
        ax.text(j + 0.5, -0.05, agg[:3],
                ha='center', va='top', fontsize=8, color='#444444')

    # Cells with heatmap background
    # Compute vmin/vmax for colormap
    vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
    if vmax == vmin:
        vmax = vmin + 1e-9
    cmap = plt.cm.RdYlGn  # red->yellow->green
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i in range(n_rows):
        metric_name = METRICS[i][1]
        ax.text(-0.05, i + 0.5, metric_name,
                ha='right', va='center', fontsize=10, fontweight='bold')
        for j in range(n_cols):
            v = matrix[i, j]
            if np.isnan(v):
                ax.add_patch(mpatches.Rectangle(
                    (j, i), 1, 1, facecolor='#EEEEEE',
                    edgecolor='#CCCCCC', hatch='//', zorder=1
                ))
                ax.text(j + 0.5, i + 0.5, 'N/A',
                        ha='center', va='center', fontsize=8, color='#999')
            else:
                color = cmap(norm(v))
                ax.add_patch(mpatches.Rectangle(
                    (j, i), 1, 1, facecolor=color,
                    edgecolor='white', zorder=1
                ))
                # bold the best in this row
                is_best = (v == np.nanmax(matrix[i]))
                ax.text(j + 0.5, i + 0.5, f'{v:.4f}',
                        ha='center', va='center',
                        fontsize=9,
                        fontweight='bold' if is_best else 'normal',
                        color='black')

    ax.set_title(title, fontsize=12, fontweight='bold', pad=40)


def main():
    # Find all available JSONs
    json_files = sorted(glob.glob('../results/full_eval_num*.json'))
    if not json_files:
        print('No full_eval_num*.json found in ../results/')
        return

    # Use the most recent
    latest = max(json_files, key=os.path.getmtime)
    print(f'Using {latest}')
    with open(latest) as f:
        data = json.load(f)

    # Render for each part_num present
    n_blocks = len(data)
    fig, axes = plt.subplots(n_blocks, 1,
                             figsize=(14, 2.0 + n_blocks * 5.0),
                             squeeze=False)
    fig.suptitle(
        'RecEraser BPR on ml-1m: Partition × Aggregation Comparison (from checkpoints)',
        fontsize=14, fontweight='bold', y=0.995
    )

    for ax_row, (key, block) in zip(axes, data.items()):
        matrix, _ = collect(data, key)
        if matrix is None:
            ax_row[0].axis('off')
            continue
        title = f'part_num = {key.replace("num", "")}  (4 partitions × 2 aggregations)'
        draw_table(ax_row[0], matrix, block, title)

    # Colorbar legend
    fig.text(0.5, 0.01,
             'Green = higher, Red = lower. Bold = best per row. / = N/A (no checkpoint).',
             ha='center', fontsize=9, style='italic', color='#555')

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    out = os.path.join('..', 'results', 'comparison_heatmap_table.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f'[OK] Saved: {out}')
    plt.close()

    # Also produce a side-by-side: num=10 attention-only and mean-only
    if 'num10' in data:
        # Build a focused view with just best-per-partition + agg
        fig2, ax = plt.subplots(figsize=(13, 5))
        block = data['num10']
        focused_metrics = [
            ('recall20', 'Recall@20'),
            ('recall50', 'Recall@50'),
            ('ndcg20', 'NDCG@20'),
            ('ndcg50', 'NDCG@50'),
        ]
        n_rows = len(focused_metrics)
        # Columns: 8 (4 partitions × 2 aggs) + 1 for delta
        cols = []
        for p in PARTITIONS:
            for a in AGG_TYPES:
                cols.append((p, a))
        n_cols = len(cols)

        matrix = np.zeros((n_rows, n_cols))
        for i, (mk, _) in enumerate(focused_metrics):
            for j, (p, a) in enumerate(cols):
                key = f'{p}-{a}'
                matrix[i, j] = block.get(key, {}).get(mk, np.nan)

        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        ax.invert_yaxis()
        ax.axis('off')

        vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        for i, (_, mname) in enumerate(focused_metrics):
            ax.text(-0.1, i + 0.5, mname, ha='right', va='center',
                    fontsize=11, fontweight='bold')
            for j, (p, a) in enumerate(cols):
                # Partition group bg
                grp_i = PARTITIONS.index(p)
                ax.add_patch(mpatches.Rectangle(
                    (j, i), 1, 1,
                    facecolor=PARTITION_COLOR[p], alpha=0.08,
                    edgecolor='none', zorder=0
                ))
                v = matrix[i, j]
                if np.isnan(v):
                    ax.add_patch(mpatches.Rectangle(
                        (j, i), 1, 1, facecolor='#EEEEEE',
                        edgecolor='#CCCCCC', hatch='//', zorder=1
                    ))
                    ax.text(j + 0.5, i + 0.5, 'N/A',
                            ha='center', va='center', fontsize=9, color='#999')
                else:
                    color = cmap(norm(v))
                    ax.add_patch(mpatches.Rectangle(
                        (j, i), 1, 1, facecolor=color,
                        edgecolor='white', zorder=1
                    ))
                    is_best = (v == np.nanmax(matrix[i]))
                    ax.text(j + 0.5, i + 0.5, f'{v:.4f}',
                            ha='center', va='center', fontsize=10,
                            fontweight='bold' if is_best else 'normal')

        # Column headers
        for j, (p, a) in enumerate(cols):
            ax.text(j + 0.5, -0.4, a[:3].upper(),
                    ha='center', va='top', fontsize=9, fontweight='bold')
            ax.text(j + 0.5, -0.05, '┐',
                    ha='center', va='top', fontsize=6, color=PARTITION_COLOR[p])
        for i2, p in enumerate(PARTITIONS):
            ax.text(i2 * 2 + 1, -0.85, p, ha='center', va='top',
                    fontsize=11, fontweight='bold', color=PARTITION_COLOR[p])
        ax.set_title(
            'ml-1m (part_num=10): Recall & NDCG — Partition × Aggregation',
            fontsize=13, fontweight='bold', pad=50
        )
        plt.tight_layout()
        out2 = os.path.join('..', 'results', 'comparison_focused_heatmap.png')
        plt.savefig(out2, dpi=130, bbox_inches='tight')
        print(f'[OK] Saved: {out2}')
        plt.close()


if __name__ == '__main__':
    main()
