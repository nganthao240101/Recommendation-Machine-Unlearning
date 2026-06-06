"""
Visualize results from 4 partition methods
"""
import matplotlib.pyplot as plt
import numpy as np

# Kết quả từ các lần chạy
# InBP (num=3, epoch=10+)
inbp_results = {
    'agg_recall': [0.10856, 0.11089, 0.11316, 0.11294, 0.11416, 0.11565, 0.11528, 0.11624, 0.11702],
    'agg_recall20': [0.18016, 0.18545, 0.18642, 0.18727, 0.18790, 0.19063, 0.18990, 0.19123, 0.19265],
    'agg_ndcg20': [0.26289, 0.26930, 0.27014, 0.27252, 0.27294, 0.27691, 0.27598, 0.27702, 0.27919]
}

# UBP (num=10, epoch=100) - từ lần chạy trước
ubp_results = {
    'agg_recall20': 0.2184,
    'agg_precision20': 0.24331,
    'agg_ndcg20': 0.31114
}

# Random (num=10, epoch=100) - từ lần chạy trước
random_results = {
    'agg_recall20': 0.1976,
    'agg_precision20': 0.22764,
    'agg_ndcg20': 0.29104
}

# Local models recall@20 (InBP, num=3)
inbp_local = [0.08980, 0.09706, 0.08540]
inbp_local_avg = np.mean(inbp_local)

# UBP local models
ubp_local = [0.05271, 0.04887, 0.02804, 0.03781, 0.03723, 0.01972, 0.02024, 0.02567, 0.01281, 0.01323]
ubp_local_avg = np.mean(ubp_local)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. InBP Training Curves
ax1 = axes[0, 0]
epochs = range(len(inbp_results['agg_recall20']))
ax1.plot(epochs, inbp_results['agg_recall20'], 'b-o', label='Recall@20', linewidth=2, markersize=8)
ax1.plot(epochs, inbp_results['agg_ndcg20'], 'r-s', label='NDCG@20', linewidth=2, markersize=8)
ax1.set_xlabel('Aggregation Epoch')
ax1.set_ylabel('Score')
ax1.set_title('InBP (Interaction-based) - Training Progress')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Aggregation Comparison (4 methods)
ax2 = axes[0, 1]
methods = ['InBP', 'UBP', 'Random', 'IBP']
recall_20 = [inbp_results['agg_recall20'][-1], ubp_results['agg_recall20'], random_results['agg_recall20'], 0.0]
ndcg_20 = [inbp_results['agg_ndcg20'][-1], ubp_results['agg_ndcg20'], random_results['agg_ndcg20'], 0.0]

x = np.arange(len(methods))
width = 0.35
ax2.bar(x - width/2, recall_20, width, label='Recall@20', color='steelblue', alpha=0.8)
ax2.bar(x + width/2, ndcg_20, width, label='NDCG@20', color='coral', alpha=0.8)
ax2.set_xlabel('Partition Method')
ax2.set_ylabel('Score')
ax2.set_title('Aggregation Model Performance Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Local Models Performance
ax3 = axes[1, 0]
local_data = [inbp_local, ubp_local]
bp = ax3.boxplot(local_data, labels=['InBP (n=3)', 'UBP (n=10)'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['steelblue', 'coral']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax3.set_ylabel('Recall@20')
ax3.set_title('Distribution of Local Models Performance')
ax3.grid(axis='y', alpha=0.3)

# 4. Summary Table
ax4 = axes[1, 1]
ax4.axis('off')
summary = """
+============================================================+
|    SUMMARY: 4 Partition Methods Comparison                 |
+============================================================+
| Method   | Agg Recall@20 | Local Avg  | Notes            |
+============================================================+
| InBP     |   0.1927      |  0.0908    | Best in test     |
| UBP      |   0.2184      |  0.0296    | Stable           |
| Random   |   0.1976      |  -         | Baseline         |
| IBP      |   TBD         |  TBD       | Item-based       |
+============================================================+
| KEY INSIGHT:                                               |
| - Aggregation determines final performance                |
| - InBP gets best Recall@50 (0.3385)                        |
| - UBP has highest Recall@20 (0.2184)                       |
+============================================================+
"""
ax4.text(0.05, 0.5, summary, fontsize=9, fontfamily='monospace',
         verticalalignment='center', transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('partition_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: partition_comparison.png")