"""
Visualize unlearn results from online_unlearn output JSON.
Compares baseline (no unlearn) vs online unlearn for each partition + aggregation.

Usage:
  python viz_unlearn_results.py results/online_unlearn_num5.json
"""
import json
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METHOD_INFO = {
    1: 'InP',   # Interaction-based Partition
    2: 'UBP',   # User-based Partition
    3: 'Random',
    4: 'IBP',   # Item-based Partition
}
METHOD_COLOR = {
    'InP': '#A23B72',
    'UBP': '#2E86AB',
    'Random': '#888888',
    'IBP': '#F18F01',
}
AGG_STYLE = {
    'mean': {'hatch': '//', 'alpha': 0.55, 'edgecolor': 'black'},
    'attention': {'hatch': None, 'alpha': 0.95, 'edgecolor': 'black'},
}

def load_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def make_figure(json_path, out_path=None):
    data = load_data(json_path)
    if out_path is None:
        out_path = json_path.replace('.json', '_viz.png')

    # Discover unlearn types present
    unlearn_types = set()
    partitions = set()
    for key in data.keys():
        # key format: "<Method>-<agg>" e.g. "InP-attention", "UBP-mean"
        # But could also have unlearn tag nested
        for ut in ['interaction', 'user', 'item']:
            for pt in METHOD_INFO.values():
                for agg in ['mean', 'attention']:
                    tag = f"{pt}-{agg}_{ut}"
                    if tag in data:
                        unlearn_types.add(ut)
                        partitions.add(pt)

    unlearn_types = sorted(unlearn_types)
    partitions = sorted(partitions, key=lambda p: list(METHOD_INFO.values()).index(p))
    aggs = ['mean', 'attention']

    if not unlearn_types:
        print("No unlearn data found in JSON")
        return

    print(f"Unlearn types: {unlearn_types}")
    print(f"Partitions: {partitions}")

    # For each unlearn type + ratio, create a comparison figure
    # Find all ratios
    ratios_seen = set()
    for key in data.keys():
        for ut in unlearn_types:
            if f'_{ut}_' in key:
                # Extract ratio from key like "InP-attention_interaction_r10"
                parts = key.split('_r')
                if len(parts) > 1:
                    ratio_str = parts[-1]
                    try:
                        ratios_seen.add(int(ratio_str))
                    except ValueError:
                        pass

    ratios = sorted(ratios_seen) if ratios_seen else [10]
    print(f"Ratios: {ratios}")

    for ut in unlearn_types:
        for ratio in ratios:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            ratio_str = f'r{ratio:02d}'
            fig.suptitle(f'Unlearn {ut} ({ratio_str}): Baseline vs Online Unlearn',
                         fontsize=13, fontweight='bold')

            for ax_idx, k in enumerate(['recall20', 'precision20', 'ndcg20']):
                ax = axes[ax_idx]
                method_names = []
                deltas = []  # (unlearn - baseline) / baseline * 100
                colors = []

                for pt in partitions:
                    for agg in aggs:
                        key_base = f"{pt}-{agg}_{ut}_{ratio_str}"
                        if key_base in data:
                            bl = data[key_base].get('baseline', {})
                            un = data[key_base].get('online_unlearn', {})
                            if k in bl and k in un and bl[k] > 0:
                                delta = (un[k] - bl[k]) / bl[k] * 100
                                method_names.append(f"{pt}-{agg[:3]}")
                                deltas.append(delta)
                                colors.append(METHOD_COLOR[pt])

                if not deltas:
                    ax.set_title(f'{k.upper()} (no data)')
                    continue

                # Sort by delta (best to worst, less degradation = better)
                sorted_idx = sorted(range(len(deltas)), key=lambda i: deltas[i])
                method_names = [method_names[i] for i in sorted_idx]
                deltas = [deltas[i] for i in sorted_idx]
                colors = [colors[i] for i in sorted_idx]

                # Build bar styles (hatch for mean)
                hatches = ['//' if 'mean' in m else None for m in method_names]

                bars = ax.bar(range(len(deltas)), deltas, color=colors,
                             edgecolor='black', alpha=0.85)
                for bar, h in zip(bars, hatches):
                    if h:
                        bar.set_hatch(h)

                for i, v in enumerate(deltas):
                    label = f'{v:+.1f}%'
                    color = 'green' if v >= 0 else 'red'
                    ax.text(i, v + (0.5 if v >= 0 else -1.0), label,
                           ha='center', fontsize=9, fontweight='bold', color=color)

                ax.set_xticks(range(len(method_names)))
                ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel(f'{k.upper()} delta (%)')
                ax.set_title(f'{k.upper()} (Unlearn vs Baseline)')
                ax.axhline(y=0, color='black', linewidth=0.8)
                ax.grid(axis='y', alpha=0.3)
                # Auto-scale
                margin = max(abs(min(deltas)), abs(max(deltas)), 5) * 0.2
                ax.set_ylim(min(deltas) - margin, max(deltas) + margin)

            plt.tight_layout()
            out_name = out_path.replace('.png', f'_{ut}_{ratio_str}.png')
            plt.savefig(out_name, dpi=120, bbox_inches='tight')
            print(f'[OK] Saved: {out_name}')
            plt.close()


def make_table(json_path, out_path=None):
    """Make a markdown table of all unlearn results"""
    data = load_data(json_path)
    if out_path is None:
        out_path = json_path.replace('.json', '_table.md')

    lines = []
    lines.append('# Unlearn Results Summary')
    lines.append('')
    lines.append('| Partition | Agg | Unlearn Type | Ratio | K | Baseline | Online Unlearn | Delta (%) |')
    lines.append('|-----------|-----|--------------|-------|---|----------|----------------|-----------|')

    for key, val in sorted(data.items()):
        if 'baseline' in val and 'online_unlearn' in val:
            bl = val['baseline']
            un = val['online_unlearn']
            for k in ['recall10', 'recall20', 'recall50', 'precision20', 'ndcg20']:
                if k in bl and k in un and bl[k] > 0:
                    delta = (un[k] - bl[k]) / bl[k] * 100
                    lines.append(f"| {key} | | | | {k} | {bl[k]:.4f} | {un[k]:.4f} | {delta:+.2f}% |")

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"[OK] Saved: {out_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python viz_unlearn_results.py <path_to_json>")
        sys.exit(1)
    json_path = sys.argv[1]
    make_figure(json_path)
    make_table(json_path)