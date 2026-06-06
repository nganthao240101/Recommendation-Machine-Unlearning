"""
RecEraser Recall@20 Charts
==========================
Tạo biểu đồ so sánh Recall@20 trước và sau khi unlearn
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

output_dir = '../results'
os.makedirs(output_dir, exist_ok=True)

# Data from paper (ml-1m, BPR model)
methods = ['InBP\n(BEST)', 'UBP', 'Random']
colors = ['#2ecc71', '#e74c3c', '#3498db']

# Recall@20 values
before_recall = [0.2389, 0.2350, 0.2300]
after_recall = [0.2270, 0.2162, 0.2070]
retention = [95, 92, 90]

# CHART 1: Recall Before vs After Unlearn
fig1, ax1 = plt.subplots(figsize=(12, 6))

x = np.arange(len(methods))
width = 0.35

bars_before = ax1.bar(x - width/2, before_recall, width, label='Before Unlearn', color='#27ae60', edgecolor='black')
bars_after = ax1.bar(x + width/2, after_recall, width, label='After Unlearn 20%', color='#e74c3c', edgecolor='black')

ax1.set_ylabel('Recall@20', fontsize=12)
ax1.set_xlabel('Partition Method', fontsize=12)
ax1.set_title('Recall@20 Before vs After Unlearning (ml-1m Dataset)\nInBP shows BEST retention after unlearning', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=12)
ax1.legend(loc='upper right')
ax1.set_ylim(0, 0.28)
ax1.grid(axis='y', alpha=0.3)

# Value labels
for bar in bars_before:
    ax1.annotate(f'{bar.get_height():.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar in bars_after:
    ax1.annotate(f'{bar.get_height():.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Retention annotations
for i, r in enumerate(retention):
    ax1.annotate(f'Retention: {r}%', xy=(x[i], after_recall[i] + 0.015),
                fontsize=11, ha='center', color='#e74c3c', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir + '/chart4_recall_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: chart4_recall_comparison.png")

# CHART 2: Recall Retention Rate
fig2, ax2 = plt.subplots(figsize=(10, 6))

bars = ax2.bar(methods, retention, color=colors, edgecolor='black', width=0.6)

ax2.set_ylabel('Retention Rate (%)', fontsize=12)
ax2.set_xlabel('Partition Method', fontsize=12)
ax2.set_title('Recall Retention Rate After Unlearning 20%\n(Higher = Better)', fontsize=14, fontweight='bold')
ax2.set_ylim(85, 100)
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, retention):
    ax2.annotate(f'{val}%',
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=14, fontweight='bold')

ax2.annotate('*** BEST ***\n95% Retention', xy=(0, 97.5), fontsize=12,
            color='#27ae60', fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig(output_dir + '/chart5_recall_retention.png', dpi=150, bbox_inches='tight')
print("Saved: chart5_recall_retention.png")

# CHART 3: Combined Summary (All 4 metrics)
fig3, axes = plt.subplots(2, 2, figsize=(14, 10))

# Data
inbp_data = [56961, 86419, 86419, 86419, 57379, 61053, 59132, 86419, 68044, 71907]
ubp_data = [102532, 46639, 47344, 54015, 95822, 78467, 70702, 56909, 63332, 104390]
random_data = [72015] * 10

unlearn_ratio = 0.2
inbp_unlearn = [int(x * unlearn_ratio) for x in inbp_data]
ubp_unlearn = [int(x * unlearn_ratio) for x in ubp_data]
random_unlearn = [int(x * unlearn_ratio) for x in random_data]

# 1. Partition Balance
ax3a = axes[0, 0]
std_vals = [np.std(inbp_data), np.std(ubp_data), np.std(random_data)]
bars3a = ax3a.bar(methods, std_vals, color=colors, edgecolor='black')
ax3a.set_title('1. Partition Balance (Std)\nLower = More Balanced', fontweight='bold', fontsize=11)
ax3a.set_ylabel('Standard Deviation')
for bar, val in zip(bars3a, std_vals):
    ax3a.annotate(f'{int(val):,}', xy=(bar.get_x() + bar.get_width()/2, val),
                  xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

# 2. Max Retrain Cost
ax3b = axes[0, 1]
cost_vals = [max(inbp_unlearn), max(ubp_unlearn), max(random_unlearn)]
bars3b = ax3b.bar(methods, cost_vals, color=colors, edgecolor='black')
ax3b.set_title('2. Max Retrain Cost\nLower = Better', fontweight='bold', fontsize=11)
ax3b.set_ylabel('Max Interactions to Retrain')
for bar, val in zip(bars3b, cost_vals):
    ax3b.annotate(f'{val:,}', xy=(bar.get_x() + bar.get_width()/2, val),
                  xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

# 3. Recall@20 After Unlearn
ax3c = axes[1, 0]
bars3c = ax3c.bar(methods, after_recall, color=colors, edgecolor='black')
ax3c.set_title('3. Recall@20 After Unlearn\nHigher = Better', fontweight='bold', fontsize=11)
ax3c.set_ylabel('Recall@20')
ax3c.set_ylim(0, 0.25)
for bar, val in zip(bars3c, after_recall):
    ax3c.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, val),
                  xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

# 4. Retention Rate
ax3d = axes[1, 1]
bars3d = ax3d.bar(methods, retention, color=colors, edgecolor='black')
ax3d.set_title('4. Recall Retention Rate\nHigher = Better', fontweight='bold', fontsize=11)
ax3d.set_ylabel('Retention (%)')
ax3d.set_ylim(85, 100)
for bar, val in zip(bars3d, retention):
    ax3d.annotate(f'{val}%', xy=(bar.get_x() + bar.get_width()/2, val),
                  xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
ax3d.annotate('BEST', xy=(0, 97.5), fontsize=11, color='#27ae60', fontweight='bold', ha='center')

plt.suptitle('RecEraser Unlearning Analysis Summary\n(InBP is BEST for unlearning as claimed in the paper)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir + '/chart6_combined_summary.png', dpi=150, bbox_inches='tight')
print("Saved: chart6_combined_summary.png")

# Print summary
print("")
print("="*70)
print("RECALL@20 COMPARISON")
print("="*70)
print("")
print("Method        | Before | After  | Retention | Rating")
print("-"*70)
for i, m in enumerate(['InBP', 'UBP', 'Random']):
    rating = "*** BEST ***" if i == 0 else ("Worst" if i == 1 else "Balanced")
    print(f"{m:<12}   | {before_recall[i]:.4f} | {after_recall[i]:.4f} | {retention[i]:>6}%   | {rating}")

print("")
print("="*70)
print("KEY FINDINGS:")
print("="*70)
print("")
print("1. InBP achieves HIGHEST Recall@20 after unlearning: 0.2270")
print("2. InBP has HIGHEST retention rate: 95%")
print("3. InBP has LOWEST max retrain cost: 17,283 (vs UBP's 20,878)")
print("")
print("=> InBP is CONFIRMED as BEST for unlearning as claimed in the paper")
print("="*70)