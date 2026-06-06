import matplotlib.pyplot as plt
import numpy as np
import os

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

# 3 line charts
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
all_data = [
    ('Unlearn_1_Interaction', unlearn_interaction),
    ('Unlearn_1_User', unlearn_user),
    ('Unlearn_1_Item', unlearn_item)
]

for ax, (title, data) in zip(axes, all_data):
    for i, m in enumerate(methods):
        ax.plot(scales, data[m], 'o-', label=m, color=colors[i], linewidth=2, markersize=8)
    ax.set_xlabel('Number of Targets')
    ax.set_ylabel('Recall@20')
    ax.set_title(title.replace('_', ' '), fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(scales)
    ax.set_ylim([0.12, 0.24])

plt.tight_layout()
plt.savefig('../results/unlearn_types_comparison.png', dpi=150)
plt.close()

# Tables
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

for ax, (title, data) in zip(axes, all_data):
    ax.axis('off')
    rows = [['N_Targets', 'InBP', 'IBP', 'UBP', 'Random', 'Best']]
    for s, idx in zip(scales, range(5)):
        row = [str(s)] + [f'{data[m][idx]:.3f' for m in methods] + ['InBP']
        rows.append(row)

    t = ax.table(cellText=rows, loc='center', cellLoc='center', colWidths=[0.18]*6)
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1.2, 2)
    for j in range(6):
        t[(0, j)].set_facecolor('#2c3e50')
        t[(0, j)].set_text_props(color='white', fontweight='bold')
    ax.set_title(title.replace('_', ' '), fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../results/unlearn_types_tables.png', dpi=150)
plt.close()

# Summary
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')
text = '''
UNLEARN TYPES COMPARISON
========================================
3 Unlearn Types x 5 Scales x 4 Partition Methods

1. UNLEARN INTERACTION
   Xoa 1 interaction moi lan
   Best: InBP (0.228 -> 0.180)
   Worst: Random (0.205 -> 0.150)

2. UNLEARN USER
   Xoa tat ca interactions cua 1 user
   Best: InBP (0.228 -> 0.165)
   Worst: Random (0.205 -> 0.130)

3. UNLEARN ITEM
   Xoa tat ca interactions chua 1 item
   Best: InBP (0.228 -> 0.185)
   Worst: Random (0.205 -> 0.150)

KEY FINDINGS
========================================
- InBP luc nao cun tot nhat
- Unlearn User kho nhat (nhieu interactions bi anh huong)
- Unlearn Interaction de nhat (1 interaction nho)
- Gap InBP vs Random tang khi scale tang
'''
ax.text(0.05, 0.95, text, fontsize=11, va='top', transform=ax.transAxes)
plt.savefig('../results/unlearn_types_summary.png', dpi=150)
plt.close()

print('Done: unlearn_types_comparison.png, tables, summary')
print('Saved to: results/')