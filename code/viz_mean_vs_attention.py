"""
Direct Mean vs Attention comparison.

Reads results/full_eval_num{N}.json and draws a bar chart per metric
showing the 2 aggregations side-by-side for each partition.

Output: results/mean_vs_attention_num{N}.png

Usage:
  python viz_mean_vs_attention.py 5
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
PARTITION_COLOR = {
    'InP':    '#A23B72',
    'UBP':    '#2E86AB',
    'IBP':    '#F18F01',
    'Random': '#888888',
}


def main():
    part_num = 5
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        part_num = int(sys.argv[1])

    json_path = os.path.join(RESULTS, f'full_eval_num{part_num}.json')
    if not os.path.exists(json_path):
        print(f'Missing {json_path}. Run: python run_full_eval.py {part_num}')
        return
    with open(json_path) as f:
        doc = json.load(f).get(f'num{part_num}', json.load(open(json_path)))

    # 3 subplots: Recall, NDCG, Precision (one bar chart per K)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    metric_groups = [
        ('Recall@K',   [(f'recall{k}', f'Recall@{k}') for k in (10, 20, 50)]),
        ('NDCG@K',     [(f'ndcg{k}',   f'NDCG@{k}')   for k in (10, 20, 50)]),
        ('Precision@K',[(f'precision{k}', f'Precision@{k}') for k in (10, 20, 50)]),
    ]

    n_part = len(PARTITIONS)
    x = np.arange(3)  # K=10, 20, 50
    width = 0.18  # 4 partitions × 2 aggregations = 8 bars per K
    # Bars will be grouped per (partition, agg)

    for ax, (group_name, metric_list) in zip(axes, metric_groups):
        # Each K has 8 bars: InP-mean, InP-att, UBP-mean, UBP-att, ...
        for p_idx, part in enumerate(PARTITIONS):
            for a_idx, agg in enumerate(['mean', 'attention']):
                vals = []
                for mkey, _ in metric_list:
                    v = doc.get(f'{part}-{agg}', {}).get(mkey)
                    vals.append(v if v is not None else np.nan)
                offset = (p_idx * 2 + a_idx - n_part) * width
                bars = ax.bar(x + offset, vals, width,
                              color=PARTITION_COLOR[part],
                              alpha=0.5 if agg == 'mean' else 1.0,
                              edgecolor='black' if agg == 'mean' else 'none',
                              hatch='//' if agg == 'mean' else None,
                              label=f'{part}-{agg}' if metric_list[0][1] == 'Recall@10' else None)
                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                                f'{v:.3f}', ha='center', fontsize=6,
                                rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([f'K={k}' for _, k in zip(*[range(3), (10, 20, 50)])])
        ax.set_ylabel(group_name)
        ax.set_title(f'{group_name} — Mean (hatched) vs Attention (solid)',
                     fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    # Single legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc='lower center', ncol=8, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f'RecEraser BPR on ml-1m (part_num={part_num}): '
        f'Mean (avg prediction) vs Attention',
        fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = os.path.join(RESULTS, f'mean_vs_attention_num{part_num}.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'[OK] Saved: {out}')

    # Also: print numerical comparison
    print(f'\n=== Mean vs Attention (num={part_num}) ===')
    print(f'{"Metric":<15s} | ', end='')
    for part in PARTITIONS:
        print(f'{"Mean":>9s} {"Att":>9s} {"Δ%":>7s} | ', end='')
    print()
    print('-' * (15 + 1 + (9 + 9 + 7 + 3) * len(PARTITIONS)))
    for mkey, mname in METRICS:
        print(f'{mname:<15s} | ', end='')
        for part in PARTITIONS:
            v_mean = doc.get(f'{part}-mean', {}).get(mkey)
            v_att  = doc.get(f'{part}-attention', {}).get(mkey)
            if v_mean is None or v_att is None:
                print(f'{"N/A":>9s} {"N/A":>9s} {"N/A":>7s} | ', end='')
                continue
            delta = (v_att - v_mean) / v_mean * 100 if v_mean > 0 else 0
            sign = '+' if delta >= 0 else ''
            print(f'{v_mean:>9.4f} {v_att:>9.4f} {sign}{delta:>6.2f}% | ', end='')
        print()


if __name__ == '__main__':
    main()
