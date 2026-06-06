"""
RQ2: Visual Charts - Bar Charts, Heatmaps, Comparison
3 Unlearn Types x 2 Aggregation x 4 Partition
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# Data
results = {
    'Unlearn\nInteraction': {
        'InBP': {'mean': 0.208, 'attn': 0.225},
        'IBP': {'mean': 0.206, 'attn': 0.223},
        'UBP': {'mean': 0.204, 'attn': 0.220},
        'Random': {'mean': 0.188, 'attn': 0.202}
    },
    'Unlearn\nUser': {
        'InBP': {'mean': 0.215, 'attn': 0.228},
        'IBP': {'mean': 0.213, 'attn': 0.226},
        'UBP': {'mean': 0.210, 'attn': 0.223},
        'Random': {'mean': 0.195, 'attn': 0.208}
    },
    'Unlearn\nItem': {
        'InBP': {'mean': 0.205, 'attn': 0.222},
        'IBP': {'mean': 0.203, 'attn': 0.220},
        'UBP': {'mean': 0.200, 'attn': 0.218},
        'Random': {'mean': 0.185, 'attn': 0.198}
    }
}

def create_big_visual():
    """Create big readable visualizations"""
    fig = plt.figure(figsize=(20, 16))

    # ========== PLOT 1: Grouped Bar Chart ==========
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(4)
    w = 0.35
    unlearn_types = list(results.keys())
    colors = {'mean': '#3498db', 'attn': '#e74c3c'}

    # Create grouped bar chart
    all_means = []
    all_atts = []
    labels = []
    for ut in unlearn_types:
        for m in ['InBP', 'IBP', 'UBP', 'Random']:
            all_means.append(results[ut][m]['mean'])
            all_atts.append(results[ut][m]['attn'])
            labels.append('{} - {}'.format(ut.replace('\n', ' '), m))

    # Grouped bar
    positions_mean = np.arange(12)
    positions_attn = positions_mean + w
    ax1.bar(positions_mean, all_means, w, label='Mean Aggregation', color='#3498db', alpha=0.8)
    ax1.bar(positions_attn, all_atts, w, label='Attention Aggregation', color='#e74c3c', alpha=0.8)

    ax1.set_ylabel('Recall@20', fontsize=14)
    ax1.set_title('ALL Results: Mean vs Attention Aggregation\n(Attention consistently better)', fontsize=16, fontweight='bold')
    ax1.set_xticks(positions_mean + w/2)
    ax1.set_xticklabels(['Int-InBP', 'Int-IBP', 'Int-UBP', 'Int-Rand',
                      'User-InBP', 'User-IBP', 'User-UBP', 'User-Rand',
                      'Item-InBP', 'Item-IBP', 'Item-UBP', 'Item-Rand'],
                     rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0.17, 0.24])
    ax1.axhline(y=0.22, color='green', linestyle='--', alpha=0.5, label='Reference')

    # ========== PLOT 2: Heatmap ==========
    ax2 = fig.add_subplot(2, 2, 2)
    methods = ['InBP', 'IBP', 'UBP', 'Random']
    agg_types = ['Mean', 'Attention']
    unlearn_list = ['Interaction', 'User', 'Item']

    data = np.zeros((3, 4, 2))
    for i, ut in enumerate(results.keys()):
        for j, m in enumerate(methods):
            data[i, j, 0] = results[ut][m]['mean']
            data[i, j, 1] = results[ut][m]['attn']

    # Average across unlearn types
    avg_data = data.mean(axis=0)

    im = ax2.imshow(avg_data, cmap='RdYlGn', vmin=0.18, vmax=0.23)
    for i in range(4):
        for j in range(2):
            text = ax2.text(j, i, f'{avg_data[i, j]:.3f}',
                          ha='center', va='center', fontsize=14, fontweight='bold')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Mean', 'Attention'], fontsize=12)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(methods, fontsize=12)
    ax2.set_title('Heatmap: Partition vs Aggregation\n(Average across 3 Unlearn Types)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Recall@20')

    # ========== PLOT 3: Line Chart ==========
    ax3 = fig.add_subplot(2, 2, 3)
    for m in methods:
        means = [results[ut][m]['mean'] for ut in results.keys()]
        attns = [results[ut][m]['attn'] for ut in results.keys()]
        ax3.plot([1, 2, 3], means, 'o-', label=f'{m} Mean', linewidth=2, markersize=10)
        ax3.plot([1, 2, 3], attns, 's--', label=f'{m} Attn', linewidth=2, markersize=10)
    ax3.set_xticks([1, 2, 3])
    ax3.set_xticklabels(['Interaction', 'User', 'Item'])
    ax3.set_xlabel('Unlearn Type')
    ax3.set_ylabel('Recall@20')
    ax3.set_title('Trend: All Combinations', fontsize=14, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ========== PLOT 4: Best Combination Highlight ==========
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    text = '''
    ==================================================================
                         BEST COMBINATIONS RANKING
    ==================================================================

    RANK  METHOD              AGG       RECALL
    ----  ------------------  ---------  -------
     #1   InBP + Attention    Attention  0.228
     #2   IBP + Attention    Attention  0.226
     #3   UBP + Attention    Attention  0.223
     #4   Random + Attention  Attention  0.208

    ==================================================================
    KEY INSIGHT:
    - Attention aggregation consistently beats Mean aggregation
    - InBP partition gives best results across all scenarios
    - Unlearn User has highest recall (most redundancy)
    - Unlearn Item has lowest recall (least redundancy)
    ==================================================================
    '''
    ax4.text(0.05, 0.95, text, fontsize=12, fontfamily='monospace',
             verticalalignment='top', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('../results/RQ2_big_visual.png', dpi=150, bbox_inches='tight')
    print('[OK] Big visual saved')
    plt.close()

def create_comparison_charts():
    """3 side-by-side bar charts for each unlearn type"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    unlearn_list = ['Unlearn\nInteraction', 'Unlearn\nUser', 'Unlearn\nItem']
    methods = ['InBP', 'IBP', 'UBP', 'Random']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    for idx, ut in enumerate(results.keys()):
        ax = axes[idx]
        x = np.arange(4)
        w = 0.35

        means = [results[ut][m]['mean'] for m in methods]
        attns = [results[ut][m]['attn'] for m in methods]

        bars1 = ax.bar(x - w/2, means, w, label='Mean', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + w/2, attns, w, label='Attention', color='#e74c3c', alpha=0.8)

        ax.set_ylabel('Recall@20', fontsize=14)
        ax.set_title(ut.replace('\n', ' '), fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0.17, 0.24])

        # Value labels
        for bar, val in zip(bars1, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                   f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
        for bar, val in zip(bars2, attns):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                   f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../results/RQ2_sidebyside.png', dpi=150, bbox_inches='tight')
    print('[OK] Side-by-side saved')
    plt.close()

def create_ranking_chart():
    """Visual ranking of all combinations"""
    fig, ax = plt.subplots(figsize=(14, 10))

    # All 12 combinations ranked
    rankings = [
        ('InBP + Attention', 0.228, '#2E86AB'),
        ('IBP + Attention', 0.226, '#A23B72'),
        ('UBP + Attention', 0.223, '#F18F01'),
        ('InBP + Mean', 0.215, '#2E86AB'),
        ('IBP + Mean', 0.213, '#A23B72'),
        ('UBP + Mean', 0.210, '#F18F01'),
        ('Random + Attention', 0.208, '#95a5a6'),
        ('User + Mean', 0.208, '#3498db'),
        ('Item + Attention', 0.205, '#27ae60'),
        ('Int + Attention', 0.225, '#3498db'),
        ('User + Attention', 0.228, '#e74c3c'),
        ('Int + Mean', 0.208, '#3498db'),
    ]

    # Sort by recall
    rankings = sorted(rankings, key=lambda x: x[1], reverse=True)

    y_pos = np.arange(len(rankings))
    recalls = [r[1] for r in rankings]
    colors_bar = [r[2] for r in rankings]
    labels = [r[0] for r in rankings]

    bars = ax.barh(y_pos, recalls, color=colors_bar, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Recall@20', fontsize=14)
    ax.set_title('RQ2: Full Ranking - All 12 Combinations\n(2 Aggregation x 4 Partition x 3 Unlearn Types)',
                fontsize=14, fontweight='bold')
    ax.set_xlim([0.18, 0.24])

    # Value labels
    for bar, val in zip(bars, recalls):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=11, fontweight='bold')

    # Legend
    ax.legend([plt.Rectangle((0,0), 1, 1, fc='#2E86AB'),
               plt.Rectangle((0,0), 1, 1, fc='#A23B72'),
               plt.Rectangle((0,0), 1, 1, fc='#F18F01'),
               plt.Rectangle((0,0), 1, 1, fc='#95a5a6')],
              ['InBP', 'IBP', 'UBP', 'Random'],
              loc='lower right', fontsize=11)

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/RQ2_ranking.png', dpi=150, bbox_inches='tight')
    print('[OK] Ranking chart saved')
    plt.close()

if __name__ == '__main__':
    os.makedirs('../results', exist_ok=True)
    create_big_visual()
    create_comparison_charts()
    create_ranking_chart()
    print('\nALL VISUALS SAVED TO ../results/')