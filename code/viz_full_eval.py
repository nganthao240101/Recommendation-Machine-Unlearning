"""
Visualize full_eval results.
Usage: python viz_full_eval.py [path_to_json]
"""
import json
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METHOD_INFO = {
    'InP': '#A23B72',
    'UBP': '#2E86AB',
    'Random': '#888888',
    'IBP': '#F18F01',
}
AGG_STYLE = {
    'mean': {'hatch': '//', 'alpha': 0.55},
    'attention': {'hatch': None, 'alpha': 0.95},
}

def load_data(path):
    with open(path) as f:
        return json.load(f)

def extract_metrics(data, part_num=5):
    """Extract metrics from full_eval JSON. Returns dict: (partition, agg) -> metrics"""
    key = f'num{part_num}'
    if key not in data:
        # Try num5
        key = list(data.keys())[0]

    results = {}
    items = data[key]
    if isinstance(items, list):
        for item in items:
            for subk, subv in item.items():
                if isinstance(subv, dict) and 'recall20' in subv:
                    results[subk] = subv
    elif isinstance(items, dict):
        for subk, subv in items.items():
            if isinstance(subv, dict) and 'recall20' in subv:
                results[subk] = subv
    return results

def make_figure(json_path, out_path=None):
    data = load_data(json_path)
    if out_path is None:
        out_path = json_path.replace('.json', '_viz.png')

    metrics_dict = extract_metrics(data)
    if not metrics_dict:
        print("No metrics found in JSON")
        return

    # Sort by partition then agg
    def sort_key(k):
        # k like "InP-attention", "UBP-mean"
        partition, agg = k.rsplit('-', 1)
        order = {'InP': 0, 'UBP': 1, 'Random': 2, 'IBP': 3}
        return (order.get(partition, 99), 0 if agg == 'attention' else 1)

    sorted_keys = sorted(metrics_dict.keys(), key=sort_key)
    print(f"Found {len(sorted_keys)} combinations: {sorted_keys}")

    # 3 subplots: Recall, Precision, NDCG at K=20
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Full Evaluation Results (Unlearn: num={json_path.split("num")[-1].split(".")[0]})',
                 fontsize=13, fontweight='bold')

    for ax_idx, k_short in enumerate(['recall20', 'precision20', 'ndcg20']):
        ax = axes[ax_idx]
        vals = []
        colors = []
        hatches = []
        labels = []

        for key in sorted_keys:
            m = metrics_dict[key]
            partition, agg = key.rsplit('-', 1)
            vals.append(m[k_short])
            colors.append(METHOD_INFO[partition])
            hatches.append(AGG_STYLE[agg]['hatch'])
            alpha = AGG_STYLE[agg]['alpha']
            labels.append(f'{partition}-{agg[:3]}')

        bars = ax.bar(range(len(vals)), vals, color=colors,
                      edgecolor='black', alpha=0.9)
        for bar, h in zip(bars, hatches):
            if h:
                bar.set_hatch(h)

        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, f'{v:.4f}',
                   ha='center', fontsize=9, fontweight='bold')

        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(f'{k_short.upper()}')
        ax.set_title(f'{k_short.upper()} (Unlearn vs Baseline)')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(vals) * 1.2)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = []
    for pt, color in METHOD_INFO.items():
        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=pt))
    legend_elements.append(Patch(facecolor='white', edgecolor='black', hatch='//', label='mean'))
    legend_elements.append(Patch(facecolor='white', edgecolor='black', label='attention'))
    fig.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'[OK] Saved: {out_path}')
    plt.close()

    # Print table
    print('\n=== Table ===')
    print(f'{"Method":<22} {"R@10":<8} {"R@20":<8} {"R@50":<8} {"P@20":<8} {"N@20":<8}')
    for key in sorted_keys:
        m = metrics_dict[key]
        print(f'{key:<22} '
              f'{m["recall10"]:<8.4f} '
              f'{m["recall20"]:<8.4f} '
              f'{m["recall50"]:<8.4f} '
              f'{m["precision20"]:<8.4f} '
              f'{m["ndcg20"]:<8.4f}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python viz_full_eval.py <path_to_json>")
        sys.exit(1)
    make_figure(sys.argv[1])