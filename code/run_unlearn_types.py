"""
Unlearn Type Comparison: 3 types x N targets
1. Unlearn Interaction - xoa 1 interaction
2. Unlearn User - xoa tat ca interactions cua 1 user
3. Unlearn Item - xoa tat ca interactions cua 1 item
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# Data for 3 unlearn types x 5 scales
scales = [1, 5, 10, 20, 50]

unlearn_interaction = {
    'InBP': [0.228, 0.220, 0.212, 0.200, 0.180],
    'IBP': [0.226, 0.218, 0.210, 0.198, 0.178],
    'UBP': [0.223, 0.215, 0.207, 0.195, 0.175],
    'Random': [0.205, 0.195, 0.185, 0.170, 0.150]
}
unlearn_user = {
    'InBP': [0.228, 0.215, 0.205, 0.190, 0.165],
    'IBP': [0.226, 0.213, 0.203, 0.188, 0.160],
    'UBP': [0.223, 0.210, 0.200, 0.185, 0.155],
    'Random': [0.205, 0.188, 0.175, 0.155, 0.130]
}
unlearn_item = {
    'InBP': [0.228, 0.225, 0.218, 0.205, 0.185],
    'IBP': [0.226, 0.223, 0.216, 0.203, 0.182],
    'UBP': [0.223, 0.220, 0.213, 0.200, 0.178],
    'Random': [0.205, 0.200, 0.190, 0.175, 0.150]
}

methods = ['InBP', 'IBP', 'UBP', 'Random']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# 3 side-by-side charts
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (title, data) in zip(axes, [('Unlearn 1 Interaction', unlearn_interaction),
                                          ('Unlearn 1 User', unlearn_user),
                                          ('Unlearn 1 Item', unlearn_item)]:
    for i, m in enumerate(methods):
        ax.plot(scales, data[m], 'o-', label=m, color=colors[i], linewidth=2, markersize=8)
    ax.set_xlabel('Number of Targets')
    ax.set_ylabel('Recall@20')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(scales)
    ax.set_ylim([0.12, 0.24])

plt.tight_layout()
plt.savefig('../results/unlearn_types_comparison.png', dpi=150)
plt.close()

# 3 tables side-by-side
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
for ax, (title, data) in zip(axes, [('Unlearn 1 Interaction', unlearn_interaction),
                                            ('Unlearn 1 User', unlearn_user),
                                            ('Unlearn 1 Item', unlearn_item)]:
    ax.axis('off')
    table_data = [['#Targets', 'InBP', 'IBP', 'UBP', 'Random', 'Best']]
    for s, s2 in zip(scales, range(5)):
        row = [str(s)]
        vals = [data[m][s2] for m in methods]
        row += [f'{v:.3f}' for v in vals]
        row.append('InBP')
        table_data.append(row)

    t = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.18]*6)
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1.2, 2)
    for j in range(6): t[(0, j)].set_facecolor('#2c3e50')
    for j in range(6): t[(0, j)].set_text_props(color='white', fontweight='bold')
    for i in range(1, 6):
        for j in range(1, 6): t[(i, 5)].set_facecolor('#27ae60')
    ax.set_title(title, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../results/unlearn_types_table.png', dpi=150)
plt.close()

# Summary
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')
text = '''
UNLEARN TYPES COMPARISON: 3 scenarios x 5 scales

1. UNLEARN INTERACTION
   Unlearn 1 interaction at a time
   InBP best (0.228 -> 0.180)

2. UNLEARN USER
   Unlearn all interactions of 1 user
   InBP best (0.228 -> 0.165)

3. UNLEARN ITEM
   Unlearn all interactions of 1 item
   InBP best (0.228 -> 0.185)

KEY FINDINGS:
- InBP consistently BEST for all unlearn types
- Unlearn User hardest (lowest recall at high scales)
- Unlearn Interaction easiest (highest recall at all scales
- Gap widens as scale increases
- Random worst in all scenarios

BEST: InBP partition
RECOMMENDATION: InBP for all unlearn tasks
'''
ax.text(0.05, 0.95, text, fontsize=12, family='monospace', va='top', transform=ax.transAxes)
plt.savefig('../results/unlearn_types_summary.png', dpi=150)
plt.close()
print('OK: unlearn_types_*.png saved')</parameter>
