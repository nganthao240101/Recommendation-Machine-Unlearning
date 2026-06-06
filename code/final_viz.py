import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

results = {
 'Random': [0.18195, 0.31794, 0.21459, 0.27471],
 'UBP': [0.21844, 0.37638, 0.24331, 0.31114],
 'InBP': [0.20243, 0.35447, 0.22949, 0.29232],
 'IBP': [0.0, 0.0, 0.0, 0.0],
}
metrics = ['Recall@20', 'Recall@50', 'Precision@20', 'NDCG@20']
colors = {'Random': '#888888', 'UBP': '#2E86AB', 'InBP': '#A23B72', 'IBP': '#F18F01'}
methods = list(results.keys())

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('RecEraser BPR on ml-1m: Partition Methods Comparison', fontsize=14, fontweight='bold')

for idx, metric in enumerate(metrics):
 ax = axes[idx // 2, idx % 2]
 vals = [results[m][idx] if results[m][idx] else 0 for m in methods]
 bars = ax.bar(methods, vals, color=[colors[m] for m in methods], alpha=0.8)
 ax.set_ylabel(metric)
 ax.set_title(metric + ' by Partition Method')
 ax.grid(axis='y', alpha=0.3)
 for bar, val in zip(bars, vals):
 if val:
 ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.4f}', ha='center', fontweight='bold')
 else:
 ax.text(bar.get_x() + bar.get_width()/2, 0.05, 'TBD', ha='center', fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('../results/partition_comparison_final.png', dpi=150, bbox_inches='tight')
print('Saved')
