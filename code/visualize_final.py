import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

results = {
'Random': {'recall10': 0.11359, 'recall20': 0.18195, 'recall50': 0.31794, 'precision10': 0.25421, 'precision20': 0.21459, 'precision50': 0.16168, 'ndcg10': 0.28371, 'ndcg20': 0.27471, 'ndcg50': 0.29634, 'color': '#888888'},
'InBP': {'recall10': 0.12372, 'recall20': 0.20243, 'recall50': 0.35447, 'precision10': 0.26682, 'precision20': 0.22949, 'precision50': 0.17440, 'ndcg10': 0.29601, 'ndcg20': 0.29232, 'ndcg50': 0.32083, 'color': '#A23B72'},
'UBP': {'recall10': 0.13541, 'recall20': 0.21844, 'recall50': 0.37638, 'precision10': 0.28240, 'precision20': 0.24331, 'precision50': 0.18402, 'ndcg10': 0.31306, 'ndcg20': 0.31114, 'ndcg50': 0.34155, 'color': '#2E86AB'}
}

methods = ['Random', 'InBP', 'UBP']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('RecEraser BPR on ml-1m: Partition Methods Comparison', fontsize=14, fontweight='bold')
k_values = ['K=10', 'K=20', 'K=50']
x = np.arange(len(k_values))
width = 0.25

ax1 = axes[0, 0]
for i, method in enumerate(methods):
     vals = [results[method]['recall10'], results[method]['recall20'], results[method]['recall50']]
     bars = ax1.bar(x + i*width, vals, width, label=method, color=results[method]['color'], alpha=0.85)
     for bar, v in zip(bars, vals):
     ax1.text(bar.get_x() + bar.get_width()/2, v + 0.005, '{:.3f}'.format(v), ha='center', fontsize=8, fontweight='bold')
ax1.set_xlabel('Top-K')
ax1.set_ylabel('Recall')
ax1.set_title('Recall@K Comparison')
ax1.set_xticks(x + width)
ax1.set_xticklabels(k_values)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 0.42)

ax2 = axes[0, 1]
for i, method in enumerate(methods):
     vals = [results[method]['precision10'], results[method]['precision20'], results[method]['precision50']]
     bars = ax2.bar(x + i*width, vals, width, label=method, color=results[method]['color'], alpha=0.85)
     for bar, v in zip(bars, vals):
     ax2.text(bar.get_x() + bar.get_width()/2, v + 0.005, '{:.3f}'.format(v), ha='center', fontsize=8, fontweight='bold')
ax2.set_xlabel('Top-K')
ax2.set_ylabel('Precision')
ax2.set_title('Precision@K Comparison')
ax2.set_xticks(x + width)
ax2.set_xticklabels(k_values)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 0.32)

ax3 = axes[0, 2]
for i, method in enumerate(methods):
     vals = [results[method]['ndcg10'], results[method]['ndcg20'], results[method]['ndcg50']]
     bars = ax3.bar(x + i*width, vals, width, label=method, color=results[method]['color'], alpha=0.85)
     for bar, v in zip(bars, vals):
     ax3.text(bar.get_x() + bar.get_width()/2, v + 0.005, '{:.3f}'.format(v), ha='center', fontsize=8, fontweight='bold')
ax3.set_xlabel('Top-K')
ax3.set_ylabel('NDCG')
ax3.set_title('NDCG@K Comparison')
ax3.set_xticks(x + width)
ax3.set_xticklabels(k_values)
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, 0.38)

ax4 = axes[1, 0]
sorted_m = sorted(methods, key=lambda m: -results[m]['recall20'])
sorted_r = [results[m]['recall20'] for m in sorted_m]
bars = ax4.barh(sorted_m, sorted_r, color=[results[m]['color'] for m in sorted_m], alpha=0.85)
for bar, v in zip(bars, sorted_r):
     ax4.text(v + 0.003, bar.get_y() + bar.get_height()/2, '{:.4f}'.format(v), va='center', fontweight='bold')
ax4.set_xlabel('Recall@20')
ax4.set_title('Recall@20 Ranking')
ax4.grid(axis='x', alpha=0.3)
ax4.set_xlim(0, 0.25)

ax5 = axes[1, 1]
random_baseline = results['Random']['recall20']
improvements = [(results[m]['recall20'] - random_baseline) / random_baseline * 100 for m in methods]
colors_imp = ['#666666' if i == 0 else results[methods[i]]['color'] for i in range(len(methods))]
bars = ax5.bar(methods, improvements, color=colors_imp, alpha=0.85)
for bar, v in zip(bars, improvements):
     lbl = '{:+.1f}%'.format(v) if v != 0 else 'baseline'
     ax5.text(bar.get_x() + bar.get_width()/2, v + 0.3, lbl, ha='center', fontweight='bold')
ax5.set_ylabel('Improvement vs Random (%)')
ax5.set_title('Recall@20 Improvement vs Random')
ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax5.grid(axis='y', alpha=0.3)
ax5.set_ylim(-2, 25)

ax6 = axes[1, 2]
ax6.axis('off')
summary = '+============================================================+\n| RECALL@20: PARTITION METHODS COMPARISON |\n+============================================================+\n| Rank | Method | Recall@20 | vs Random |\n+============================================================+\n| #1 | UBP | 0.2184 | +20.1% (best) |\n| #2 | InBP | 0.2024 | +11.2% |\n| #3 | Random | 0.1820 | baseline |\n+============================================================+\n\nCONCLUSION:\n- UBP (User-based) is the BEST partition\n- UBP > InBP by 1.6% in Recall@20\n- Random provides worst baseline\n- AGGREGATION DECIDES FINAL PERFORMANCE\n+============================================================+'
ax6.text(0.0, 0.5, summary, fontsize=9, fontfamily='monospace', verticalalignment='center', transform=ax6.transAxes, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('../results/partition_comparison_4methods.png', dpi=120, bbox_inches='tight')
print('[OK] Saved: ../results/partition_comparison_4methods.png')
plt.close()

fig2, ax = plt.subplots(figsize=(10, 6))
sorted_m = sorted(methods, key=lambda m: -results[m]['recall20'])
sorted_r = [results[m]['recall20'] for m in sorted_m]
sorted_c = [results[m]['color'] for m in sorted_m]
bars = ax.bar(sorted_m, sorted_r, color=sorted_c, alpha=0.9, edgecolor='black', linewidth=1.5)
for bar, v in zip(bars, sorted_r):
     ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, '{:.4f}'.format(v), ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('Recall@20', fontsize=12)
ax.set_title('RecEraser BPR on ml-1m: Recall@20 by Partition Method', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 0.25)
plt.savefig('../results/recall20_comparison.png', dpi=120, bbox_inches='tight')
print('[OK] Saved: ../results/recall20_comparison.png')
plt.close()
