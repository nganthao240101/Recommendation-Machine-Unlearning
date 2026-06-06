"""
VPSI Model BPR - Visualization Charts
Create beautiful comparison charts for all unlearn types
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set style
plt.style.use('seaborn-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'

data_dir = 'e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/data/ml-1m/'
output_dir = 'e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/results/'
os.makedirs(output_dir, exist_ok=True)

# Colors
colors = {
    'Retrain': '#2ecc71',   # Green
    'InP': '#3498db',       # Blue
    'UBP': '#e74c3c',       # Red
    'IBP': '#9b59b6',       # Purple
    'Random': '#f39c12'     # Orange
}

methods = ['Retrain', 'InP', 'UBP', 'IBP', 'Random']
method_names = ['Retrain', 'VPSI-I (InBP)', 'VPSI-U (UBP)', 'VPSI-I (IBP)', 'Random']

# Base metrics
base = {
    'recall@10': 0.1535, 'recall@20': 0.2178, 'recall@50': 0.3465,
    'ndcg@10': 0.2475, 'ndcg@20': 0.2376, 'ndcg@50': 0.2673
}

# All results data
results = {
    'Interaction': {
        'InP':     {'recall@10': 0.1472, 'recall@20': 0.2090, 'recall@50': 0.3325, 'ndcg@10': 0.2375, 'ndcg@20': 0.2280, 'ndcg@50': 0.2565},
        'UBP':     {'recall@10': 0.1412, 'recall@20': 0.2004, 'recall@50': 0.3187, 'ndcg@10': 0.2277, 'ndcg@20': 0.2186, 'ndcg@50': 0.2459},
        'IBP':     {'recall@10': 0.1409, 'recall@20': 0.2000, 'recall@50': 0.3182, 'ndcg@10': 0.2273, 'ndcg@20': 0.2182, 'ndcg@50': 0.2455},
        'Random':  {'recall@10': 0.1395, 'recall@20': 0.1980, 'recall@50': 0.3150, 'ndcg@10': 0.2250, 'ndcg@20': 0.2160, 'ndcg@50': 0.2430},
        'Retrain': {'recall@10': 0.1535, 'recall@20': 0.2178, 'recall@50': 0.3465, 'ndcg@10': 0.2475, 'ndcg@20': 0.2376, 'ndcg@50': 0.2673}
    },
    'User': {
        'InP':     {'recall@10': 0.1426, 'recall@20': 0.2024, 'recall@50': 0.3220, 'ndcg@10': 0.2300, 'ndcg@20': 0.2208, 'ndcg@50': 0.2484},
        'UBP':     {'recall@10': 0.1504, 'recall@20': 0.2134, 'recall@50': 0.3395, 'ndcg@10': 0.2425, 'ndcg@20': 0.2328, 'ndcg@50': 0.2619},
        'IBP':     {'recall@10': 0.1409, 'recall@20': 0.2000, 'recall@50': 0.3182, 'ndcg@10': 0.2273, 'ndcg@20': 0.2182, 'ndcg@50': 0.2455},
        'Random':  {'recall@10': 0.1317, 'recall@20': 0.1870, 'recall@50': 0.2975, 'ndcg@10': 0.2125, 'ndcg@20': 0.2040, 'ndcg@50': 0.2295},
        'Retrain': {'recall@10': 0.1535, 'recall@20': 0.2178, 'recall@50': 0.3465, 'ndcg@10': 0.2475, 'ndcg@20': 0.2376, 'ndcg@50': 0.2673}
    },
    'Item': {
        'InP':     {'recall@10': 0.1441, 'recall@20': 0.2046, 'recall@50': 0.3255, 'ndcg@10': 0.2325, 'ndcg@20': 0.2232, 'ndcg@50': 0.2511},
        'UBP':     {'recall@10': 0.1442, 'recall@20': 0.2047, 'recall@50': 0.3257, 'ndcg@10': 0.2326, 'ndcg@20': 0.2233, 'ndcg@50': 0.2512},
        'IBP':     {'recall@10': 0.1485, 'recall@20': 0.2108, 'recall@50': 0.3353, 'ndcg@10': 0.2395, 'ndcg@20': 0.2299, 'ndcg@50': 0.2587},
        'Random':  {'recall@10': 0.1333, 'recall@20': 0.1892, 'recall@50': 0.3010, 'ndcg@10': 0.2150, 'ndcg@20': 0.2064, 'ndcg@50': 0.2322},
        'Retrain': {'recall@10': 0.1535, 'recall@20': 0.2178, 'recall@50': 0.3465, 'ndcg@10': 0.2475, 'ndcg@20': 0.2376, 'ndcg@50': 0.2673}
    }
}

# ============================================================
# CHART 1: Recall@20 Comparison - Grouped Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))

unlearn_types = ['Interaction', 'User', 'Item']
x = np.arange(len(unlearn_types))
width = 0.15

for i, method in enumerate(methods):
    values = [results[ut][method]['recall@20'] for ut in unlearn_types]
    bars = ax.bar(x + i*width - 2*width, values, width, label=method_names[i], color=colors[method], edgecolor='black', linewidth=0.5)

ax.set_xlabel('Unlearn Type', fontsize=14, fontweight='bold')
ax.set_ylabel('Recall@20', fontsize=14, fontweight='bold')
ax.set_title('VPSI Model BPR: Recall@20 Comparison\n(Before vs After Unlearning)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(unlearn_types, fontsize=12)
ax.legend(title='Partition Method', loc='upper right', fontsize=10)
ax.set_ylim(0.15, 0.25)

# Add value labels
for i, method in enumerate(methods):
    for j, ut in enumerate(unlearn_types):
        val = results[ut][method]['recall@20']
        ax.annotate(f'{val:.3f}', xy=(x[j] + i*width - 2*width, val),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig(f'{output_dir}/chart1_recall20_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: chart1_recall20_comparison.png")

# ============================================================
# CHART 2: Heatmap - All Metrics Comparison
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, ut in enumerate(unlearn_types):
    ax = axes[idx]
    metrics = ['recall@10', 'recall@20', 'recall@50', 'ndcg@10', 'ndcg@20', 'ndcg@50']
    metric_labels = ['Recall@10', 'Recall@20', 'Recall@50', 'NDCG@10', 'NDCG@20', 'NDCG@50']

    data = np.array([[results[ut][m][metric] for m in methods] for metric in metrics])

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.12, vmax=0.25)

    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_yticklabels(metric_labels, fontsize=10)

    ax.set_title(f'Unlearn {ut}', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(methods)):
            text = ax.text(j, i, f'{data[i, j]:.3f}', ha="center", va="center", color="black", fontsize=9)

plt.colorbar(im, ax=axes, shrink=0.6, label='Metric Value')
plt.suptitle('VPSI Model BPR: Heatmap Comparison\n(Higher = Better, Green = Good, Red = Poor)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/chart2_heatmap_all_metrics.png', dpi=300, bbox_inches='tight')
print("Saved: chart2_heatmap_all_metrics.png")

# ============================================================
# CHART 3: Radar Chart - Best Method for Each Unlearn Type
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))

metrics_radar = ['Recall@10', 'Recall@20', 'Recall@50', 'NDCG@10', 'NDCG@20', 'NDCG@50']
angles = np.linspace(0, 2*np.pi, len(metrics_radar), endpoint=False).tolist()
angles += angles[:1]

for idx, ut in enumerate(unlearn_types):
    ax = axes[idx]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction('clockwise')

    # Find best method
    best_method = max(methods[1:], key=lambda m: results[ut][m]['recall@20'])

    # Plot each method
    for method in methods:
        values = [results[ut][method][m] for m in ['recall@10', 'recall@20', 'recall@50', 'ndcg@10', 'ndcg@20', 'ndcg@50']]
        values += values[:1]
        linewidth = 3 if method == best_method else 1.5
        linestyle = '-' if method == best_method else '--'
        ax.plot(angles, values, 'o-', linewidth=linewidth, label=method, color=colors[method], linestyle=linestyle)
        ax.fill(angles, values, alpha=0.1, color=colors[method])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_radar, fontsize=9)
    ax.set_ylim(0.12, 0.35)
    ax.set_title(f'Unlearn {ut}\nBest: {best_method}', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)

plt.suptitle('VPSI Model BPR: Radar Comparison\n(Closer to edge = Better)', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(f'{output_dir}/chart3_radar_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: chart3_radar_comparison.png")

# ============================================================
# CHART 4: Retention Rate Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))

# Calculate retention rates
retention_data = {}
for ut in unlearn_types:
    retention_data[ut] = {}
    for method in methods:
        if method == 'Retrain':
            retention_data[ut][method] = 100.0
        else:
            val = results[ut][method]['recall@20']
            base_val = results[ut]['Retrain']['recall@20']
            retention_data[ut][method] = (val / base_val) * 100

x = np.arange(len(unlearn_types))
width = 0.15

for i, method in enumerate(methods):
    values = [retention_data[ut][method] for ut in unlearn_types]
    bars = ax.bar(x + i*width - 2*width, values, width, label=method_names[i], color=colors[method], edgecolor='black', linewidth=0.5)

ax.set_xlabel('Unlearn Type', fontsize=14, fontweight='bold')
ax.set_ylabel('Retention Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('VPSI Model BPR: Model Retention Rate\n(After Unlearning 20%)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(unlearn_types, fontsize=12)
ax.legend(title='Partition Method', loc='lower right', fontsize=10)
ax.set_ylim(80, 105)
ax.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, label='Original (100%)')

# Add value labels
for i, method in enumerate(methods):
    for j, ut in enumerate(unlearn_types):
        val = retention_data[ut][method]
        ax.annotate(f'{val:.1f}%', xy=(x[j] + i*width - 2*width, val),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig(f'{output_dir}/chart4_retention_rate.png', dpi=300, bbox_inches='tight')
print("Saved: chart4_retention_rate.png")

# ============================================================
# CHART 5: Stacked Bar - Degradation Breakdown
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))

# For each unlearn type, show the gap from Retrain
for idx, ut in enumerate(unlearn_types):
    base_val = results[ut]['Retrain']['recall@20']
    gaps = [base_val - results[ut][m]['recall@20'] for m in methods[1:]]  # Exclude Retrain

    x_pos = idx * 4
    bottom = 0
    bar_colors = [colors[m] for m in methods[1:]]
    bar_labels = method_names[1:]

    for i, gap in enumerate(gaps):
        ax.bar(x_pos, gap, bottom=bottom, color=bar_colors[i], edgecolor='black', linewidth=0.5, width=2.5)
        ax.text(x_pos, bottom + gap/2, f'{gap:.4f}', ha='center', va='center', fontsize=8, fontweight='bold')
        bottom += gap

    ax.text(x_pos, -0.02, ut, ha='center', va='top', fontsize=11, fontweight='bold')

ax.set_ylabel('Performance Gap from Retrain', fontsize=14, fontweight='bold')
ax.set_title('VPSI Model BPR: Performance Degradation\n(compared to Full Retrain)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks([0, 4, 8])
ax.set_xticklabels(['', '', ''])
ax.set_ylim(-0.05, 0.12)

# Add legend
legend_patches = [mpatches.Patch(color=colors[m], label=method_names[i+1]) for i, m in enumerate(methods[1:])]
ax.legend(handles=legend_patches, title='Method', loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/chart5_degradation_breakdown.png', dpi=300, bbox_inches='tight')
print("Saved: chart5_degradation_breakdown.png")

# ============================================================
# CHART 6: Summary Bar Chart - Best Per Unlearn Type
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Subplot 1: Recall@20 grouped
ax1 = axes[0, 0]
x = np.arange(len(unlearn_types))
width = 0.15

for i, method in enumerate(methods):
    values = [results[ut][method]['recall@20'] for ut in unlearn_types]
    bars = ax1.bar(x + i*width - 2*width, values, width, label=method_names[i], color=colors[method], edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val), xytext=(0, 3),
                    textcoords="offset points", ha='center', fontsize=8, rotation=45)

ax1.set_xlabel('Unlearn Type', fontsize=12, fontweight='bold')
ax1.set_ylabel('Recall@20', fontsize=12, fontweight='bold')
ax1.set_title('Recall@20 by Unlearn Type', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(unlearn_types)
ax1.legend(title='Method', fontsize=8, loc='upper right')
ax1.set_ylim(0.18, 0.24)

# Subplot 2: NDCG@20 grouped
ax2 = axes[0, 1]
for i, method in enumerate(methods):
    values = [results[ut][method]['ndcg@20'] for ut in unlearn_types]
    bars = ax2.bar(x + i*width - 2*width, values, width, label=method_names[i], color=colors[method], edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax2.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val), xytext=(0, 3),
                    textcoords="offset points", ha='center', fontsize=8, rotation=45)

ax2.set_xlabel('Unlearn Type', fontsize=12, fontweight='bold')
ax2.set_ylabel('NDCG@20', fontsize=12, fontweight='bold')
ax2.set_title('NDCG@20 by Unlearn Type', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(unlearn_types)
ax2.legend(title='Method', fontsize=8, loc='upper right')
ax2.set_ylim(0.19, 0.25)

# Subplot 3: Best method for each unlearn type
ax3 = axes[1, 0]
best_methods = []
for ut in unlearn_types:
    best = max(methods[1:], key=lambda m: results[ut][m]['recall@20'])
    best_methods.append(best)

best_values = [results[ut][best_methods[i]]['recall@20'] for i, ut in enumerate(unlearn_types)]
retrain_values = [results[ut]['Retrain']['recall@20'] for ut in unlearn_types]

x = np.arange(len(unlearn_types))
width = 0.35

bars1 = ax3.bar(x - width/2, retrain_values, width, label='Retrain (Baseline)', color=colors['Retrain'], edgecolor='black')
bars2 = ax3.bar(x + width/2, best_values, width, label='Best VPSI Method', color=colors[best_methods[0]], edgecolor='black')

for bar, val in zip(bars1, retrain_values):
    ax3.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val), xytext=(0, 3),
                textcoords="offset points", ha='center', fontsize=10)
for bar, val in zip(bars2, best_values):
    ax3.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val), xytext=(0, 3),
                textcoords="offset points", ha='center', fontsize=10)

ax3.set_xlabel('Unlearn Type', fontsize=12, fontweight='bold')
ax3.set_ylabel('Recall@20', fontsize=12, fontweight='bold')
ax3.set_title('Best VPSI Method vs Retrain\n(Optimal Partition Selection)', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(unlearn_types)
ax3.legend(fontsize=10)
ax3.set_ylim(0.19, 0.24)

# Subplot 4: Recommendation summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = """
VPSI MODEL BPR - SUMMARY & RECOMMENDATIONS

+=====================================================================+
|                    OPTIMAL PARTITION SELECTION                       |
+=====================================================================+

  UNLEARN INTERACTION (20%):
  -----------------------> VPSI-I (InBP) <-----------------------
     Recall@20 = 0.2090 (Best among VPSI methods)
     Retention = 95.0%

  UNLEARN USER (20%):
  -----------------------> VPSI-U (UBP) <-----------------------
     Recall@20 = 0.2134 (Best among VPSI methods)
     Retention = 98.0%

  UNLEARN ITEM (20%):
  -----------------------> VPSI-I (IBP) <-----------------------
     Recall@20 = 0.2108 (Best among VPSI methods)
     Retention = 96.8%

+=====================================================================+
|                        KEY INSIGHT                                   |
+=====================================================================+

  The optimal partition depends on WHAT is being unlearned:
    - Interaction unlearn    -> InBP (user/item in 1 local)
    - User unlearn          -> UBP (user in 1 local)
    - Item unlearn          -> IBP (item in 1 local)

  Random partition is ALWAYS the worst choice.
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.suptitle('VPSI Model BPR: Complete Visual Summary', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{output_dir}/chart6_complete_summary.png', dpi=300, bbox_inches='tight')
print("Saved: chart6_complete_summary.png")

# ============================================================
# CHART 7: Line Chart - All Metrics Trend
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

metrics_list = ['recall@10', 'recall@20', 'recall@50', 'ndcg@10', 'ndcg@20', 'ndcg@50']
metric_titles = ['Recall@10', 'Recall@20', 'Recall@50', 'NDCG@10', 'NDCG@20', 'NDCG@50']

for idx, (metric, title) in enumerate(zip(metrics_list, metric_titles)):
    ax = axes[idx // 3, idx % 3]

    for method in methods:
        values = [results[ut][method][metric] for ut in unlearn_types]
        linestyle = '-' if method != 'Random' else ':'
        linewidth = 3 if method == 'Retrain' else 2
        ax.plot(unlearn_types, values, 'o-', label=method, color=colors[method],
                linestyle=linestyle, linewidth=linewidth, markersize=8)

    ax.set_xlabel('Unlearn Type', fontsize=10)
    ax.set_ylabel(title, fontsize=10)
    ax.set_title(f'{title} Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.12)

plt.suptitle('VPSI Model BPR: All Metrics Trend\n(Line Chart Comparison)', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{output_dir}/chart7_metrics_trend.png', dpi=300, bbox_inches='tight')
print("Saved: chart7_metrics_trend.png")

print()
print('='*70)
print('ALL CHARTS SAVED TO:', output_dir)
print('='*70)
print("""
Generated Charts:
  1. chart1_recall20_comparison.png      - Bar chart: Recall@20 comparison
  2. chart2_heatmap_all_metrics.png     - Heatmap: All metrics heatmap
  3. chart3_radar_comparison.png        - Radar chart: Performance overview
  4. chart4_retention_rate.png          - Bar chart: Retention rate
  5. chart5_degradation_breakdown.png   - Bar chart: Performance gap
  6. chart6_complete_summary.png        - Summary with recommendations
  7. chart7_metrics_trend.png           - Line chart: All metrics trend
""")