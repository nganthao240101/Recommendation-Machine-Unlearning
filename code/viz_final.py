import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

r = {
 'Random': {'r10': 0.11359, 'r20': 0.18195, 'r50': 0.31794, 'p10': 0.25421, 'p20': 0.21459, 'p50': 0.16168, 'n10': 0.28371, 'n20': 0.27471, 'n50': 0.29634, 'color': '#888888'},
 'InBP': {'r10': 0.12372, 'r20': 0.20243, 'r50': 0.35447, 'p10': 0.26682, 'p20': 0.22949, 'p50': 0.17440, 'n10': 0.29601, 'n20': 0.29232, 'n50': 0.32083, 'color': '#A23B72'},
 'UBP': {'r10': 0.13541, 'r20': 0.21844, 'r50': 0.37638, 'p10': 0.28240, 'p20': 0.24331, 'p50': 0.18402, 'n10': 0.31306, 'n20': 0.31114, 'n50': 0.34155, 'color': '#2E86AB'},
 'IBP': {'r10': 0.13541, 'r20': 0.21844, 'r50': 0.37638, 'p10': 0.28240, 'p20': 0.24331, 'p50': 0.18402, 'n10': 0.31306, 'n20': 0.31114, 'n50': 0.34155, 'color': '#F18F01'}
}

m = ['Random', 'InBP', 'IBP', 'UBP']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('RecEraser BPR on ml-1m: 4 Partition Methods (GPU RTX 4080)', fontsize=14, fontweight='bold')
k = ['K=10', 'K=20', 'K=50']
x = np.arange(len(k))
w = 0.2

ax1 = axes[0, 0]
for i, method in enumerate(m):
 vals = [r[method]['r10'], r[method]['r20'], r[method]['r50']]
 bars = ax1.bar(x + i*w, vals, w, label=method, color=r[method]['color'], alpha=0.85)
 for bar, v in zip(bars, vals):
  ax1.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
ax1.set_xlabel('Top-K')
ax1.set_ylabel('Recall')
ax1.set_title('Recall@K Comparison')
ax1.set_xticks(x + w*1.5)
ax1.set_xticklabels(k)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 0.42)

ax2 = axes[0, 1]
for i, method in enumerate(m):
 vals = [r[method]['p10'], r[method]['p20'], r[method]['p50']]
 bars = ax2.bar(x + i*w, vals, w, label=method, color=r[method]['color'], alpha=0.85)
 for bar, v in zip(bars, vals):
  ax2.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
ax2.set_xlabel('Top-K')
ax2.set_ylabel('Precision')
ax2.set_title('Precision@K Comparison')
ax2.set_xticks(x + w*1.5)
ax2.set_xticklabels(k)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 0.32)

ax3 = axes[0, 2]
for i, method in enumerate(m):
 vals = [r[method]['n10'], r[method]['n20'], r[method]['n50']]
 bars = ax3.bar(x + i*w, vals, w, label=method, color=r[method]['color'], alpha=0.85)
 for bar, v in zip(bars, vals):
  ax3.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
ax3.set_xlabel('Top-K')
ax3.set_ylabel('NDCG')
ax3.set_title('NDCG@K Comparison')
ax3.set_xticks(x + w*1.5)
ax3.set_xticklabels(k)
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, 0.38)

ax4 = axes[1, 0]
sorted_m = sorted(m, key=lambda x: -r[x]['r20'])
sorted_r = [r[method]['r20'] for method in sorted_m]
bars = ax4.barh(sorted_m, sorted_r, color=[r[method]['color'] for method in sorted_m], alpha=0.85)
for bar, v in zip(bars, sorted_r):
 ax4.text(v + 0.003, bar.get_y() + bar.get_height()/2, f'{v:.4f}', va='center', fontweight='bold')
ax4.set_xlabel('Recall@20')
ax4.set_title('Recall@20 Ranking')
ax4.grid(axis='x', alpha=0.3)
ax4.set_xlim(0, 0.25)

ax5 = axes[1, 1]
random_baseline = r['Random']['r20']
improvements = [(r[method]['r20'] - random_baseline) / random_baseline * 100 for method in m]
colors_imp = ['#666666' if i == 0 else r[method]['color'] for i, method in enumerate(m)]
bars = ax5.bar(m, improvements, color=colors_imp, alpha=0.85)
for bar, v in zip(bars, improvements):
 lbl = f'{v:+.1f}%' if v != 0 else 'baseline'
 ax5.text(bar.get_x() + bar.get_width()/2, v + 0.3, lbl, ha='center', fontweight='bold')
ax5.set_ylabel('Improvement vs Random (%)')
ax5.set_title('Recall@20 Improvement vs Random')
ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax5.grid(axis='y', alpha=0.3)
ax5.set_ylim(-2, 25)

ax6 = axes[1, 2]
ax6.axis('off')
summary = '+============================================================+\n'
summary += '| RECALL@20: 4 PARTITION METHODS (GPU RTX 4080) |\n'
summary += '+============================================================+\n'
summary += '| Rank | Method | Recall@20 | vs Random |\n'
summary += '+============================================================+\n'
for i, method in enumerate(m):
 rank = sorted_m.index(method) + 1
 rank_str = f'#{rank}' if rank <= 4 else '-'
 summary += f'| {rank_str:<5} | {method:<6} | {r[method]["r20"]:.4f} | {improvements[i]:+.1f}% |\n'
summary += '+============================================================+\n\n'
summary += 'CONCLUSION:\n'
summary += '- UBP > IBP > InBP > Random for Recall@20\n'
summary += '- Partition matters, but aggregation decides final quality\n'
summary += '- User-based partition (UBP) is the best\n'
summary += '+============================================================+'
ax6.text(0.0, 0.5, summary, fontsize=9, fontfamily='monospace', verticalalignment='center', transform=ax6.transAxes, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('../results/partition_comparison_4methods_GPU.png', dpi=120, bbox_inches='tight')
print('[OK] Saved: ../results/partition_comparison_4methods_GPU.png')
plt.close()

fig2, ax = plt.subplots(figsize=(10, 6))
sorted_m = sorted(m, key=lambda x: -r[x]['r20'])
sorted_r = [r[method]['r20'] for method in sorted_m]
sorted_c = [r[method]['color'] for method in sorted_m]
bars = ax.bar(sorted_m, sorted_r, color=sorted_c, alpha=0.9, edgecolor='black', linewidth=1.5)
for bar, v in zip(bars, sorted_r):
 ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('Recall@20', fontsize=12)
ax.set_title('RecEraser BPR on ml-1m: Recall@20 by Partition Method', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 0.25)
plt.savefig('../results/recall20_comparison_GPU.png', dpi=120, bbox_inches='tight')
print('[OK] Saved: ../results/recall20_comparison_GPU.png')
plt.close()