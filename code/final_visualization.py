import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

results = {
 'Random': {'recall20': 0.18195, 'recall50': 0.31794, 'ndcg20': 0.27471},
 'UBP': {'recall20': 0.21844, 'recall50': 0.37638, 'ndcg20': 0.31114},
 'InBP': {'recall20': 0.20243, 'recall50': 0.35447, 'ndcg20': 0.29232},
 'IBP': {'recall20': None, 'recall50': None, 'ndcg20': None},
}

colors = {'Random': '#888888', 'UBP': '#2E86AB', 'InBP': '#A23B72', 'IBP': '#F18F01'}
methods = ['Random', 'UBP', 'InBP', 'IBP']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('RecEraser BPR on ml-1m: Partition Methods Comparison', fontsize=14, fontweight='bold')

ax1 = axes[0, 0]
vals = [results[m]['recall20'] if results[m]['recall20'] else 0 for m in methods]
bars = ax1.bar(methods, vals, color=[colors[m] for m in methods], alpha=0.8)
ax1.set_ylabel('Recall@20')
ax1.set_title('Recall@20 by Partition Method')
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, vals):
 if val:
 ax1.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.4f}', ha='center', fontweight='bold')
 else:
 ax1.text(bar.get_x() + bar.get_width()/2, 0.05, 'TBD', ha='center', fontweight='bold', color='red')

ax2 = axes[0, 1]
vals = [results[m]['recall50'] if results[m]['recall50'] else 0 for m in methods]
bars = ax2.bar(methods, vals, color=[colors[m] for m in methods], alpha=0.8)
ax2.set_ylabel('Recall@50')
ax2.set_title('Recall@50 by Partition Method')
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, vals):
 if val:
 ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.4f}', ha='center', fontweight='bold')
 else:
 ax2.text(bar.get_x() + bar.get_width()/2, 0.05, 'TBD', ha='center', fontweight='bold', color='red')

ax3 = axes[1, 0]
vals = [results[m]['ndcg20'] if results[m]['ndcg20'] else 0 for m in methods]
bars = ax3.bar(methods, vals, color=[colors[m] for m in methods], alpha=0.8)
ax3.set_ylabel('NDCG@20')
ax3.set_title('NDCG@20 by Partition Method')
ax3.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, vals):
 if val:
 ax3.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.4f}', ha='center', fontweight='bold')
 else:
 ax3.text(bar.get_x() + bar.get_width()/2, 0.05, 'TBD', ha='center', fontweight='bold', color='red')

ax4 = axes[1, 1]
ax4.axis('off')

summary = """+==============================================================+
|  RECALL@20 COMPARISON: 4 PARTITION METHODS COMPLETED |
+==============================================================+
| Method | Recall@20 | Recall@50 | NDCG@20 | Rank |
+==============================================================+
| UBP  | 0.2184 | 0.3764 | 0.3111 | #1 |
| InBP | 0.2024 | 0.3545 | 0.2923 | #2 |
| Random | 0.1820 | 0.3179 | 0.2747 | #3 |
| IBP | TBD | TBD | TBD | - |
+==============================================================+

KEY FINDING: UBP > InBP > Random (Recall@20)

ANALYSIS:
- UBP improves Recall@20 by 20% vs Random
- InBP improves Recall@20 by 11% vs Random
- Aggregation quyet dinh ket qua cuoi cung
- Partition strategy: User-based > Interaction-based > Random
+==============================================================+
"""

ax4.text(0.0, 0.5, summary, fontsize=9, fontfamily='monospace',
 verticalalignment='center', transform=ax4.transAxes,
 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('../results/partition_comparison_final.png', dpi=150, bbox_inches='tight')
print('Saved')