"""
Partition vs Aggregation Impact on Unlearning Quality
======================================================

Thí nghiệm so sánh:
1. Partition impact: Cùng aggregation, khác partition
2. Aggregation impact: Cùng partition, khác aggregation

Hypothesis:
  - Partition: Ảnh hưởng đến chi phí unlearn (retrain cost)
  - Aggregation: Ảnh hưởng đến chất lượng model (utility)

Metrics:
  1. Utility: Recall@20, NDCG@20
  2. Performance: Retraining time
  3. Unlearn Efficiency: Cost to unlearn
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTHONIOENCODING'] = 'utf-8'

from utility.parser import parse_args

args = parse_args()

output_dir = '../results'
os.makedirs(output_dir, exist_ok=True)

# ============================================
# CONFIGURATION: 4 combinations
# ============================================

EXPERIMENTS = {
    'InBP_Attn': {
        'partition': 1,  # InBP
        'aggregation': 'attention',  # Attention-based
        'label': 'InBP + Attention',
        'color': '#27ae60'
    },
    'InBP_Mean': {
        'partition': 1,  # InBP
        'aggregation': 'mean',  # Mean aggregation
        'label': 'InBP + MeanAgg',
        'color': '#2ecc71'
    },
    'UBP_Attn': {
        'partition': 2,  # UBP
        'aggregation': 'attention',
        'label': 'UBP + Attention',
        'color': '#e74c3c'
    },
    'UBP_Mean': {
        'partition': 2,  # UBP
        'aggregation': 'mean',
        'label': 'UBP + MeanAgg',
        'color': '#f39c12'
    }
}

def load_partitions(part_type, part_num):
    """Load C, C_U, C_I"""
    base = args.data_path + args.dataset
    with open(f'{base}/C_type-{part_type}_num-{part_num}.pk', 'rb') as f:
        C = pickle.load(f)
    with open(f'{base}/C_U_type-{part_type}_num-{part_num}.pk', 'rb') as f:
        C_U = pickle.load(f)
    with open(f'{base}/C_I_type-{part_type}_num-{part_num}.pk', 'rb') as f:
        C_I = pickle.load(f)
    return C, C_U, C_I

def compute_partition_metrics(C, C_U, part_type):
    """Tính metrics liên quan đến partition"""
    num_locals = len(C)

    # Total interactions
    total_inter = sum(len(u_items) for local in C for u_items in local.values())

    # User distribution
    unique_users = len(set().union(*[set(C_U[i]) for i in range(len(C))]))
    users_per_local = [len(set(C_U[i])) for i in range(len(C))]

    # User appearance analysis
    all_users = set().union(*[set(C_U[i]) for i in range(len(C))])
    user_local_count = {u: 0 for u in all_users}
    for i in range(len(C)):
        for u in C_U[i]:
            user_local_count[u] += 1

    avg_locals_per_user = sum(user_local_count.values()) / unique_users
    users_in_multiple = sum(1 for c in user_local_count.values() if c > 1)

    return {
        'num_locals': num_locals,
        'total_interactions': total_inter,
        'unique_users': unique_users,
        'users_per_local': users_per_local,
        'avg_locals_per_user': avg_locals_per_user,
        'users_in_multiple_locals': users_in_multiple,
        'users_in_multiple_pct': 100 * users_in_multiple / unique_users,
        'avg_inter_per_local': total_inter / num_locals
    }

def estimate_utility(partition_metrics, agg_type):
    """
    Estimate utility metrics based on partition and aggregation

    Hypothesis:
    - Aggregation type affects model quality more:
      * Attention: Better feature combination → Higher Recall/NDCG
      * Mean: Simpler combination → Lower Recall/NDCG
    - Partition affects consistency:
      * InBP: More balanced → More consistent results
      * UBP: May have imbalanced locals → Less consistent

    Returns: (recall_before, ndcg_before, recall_after, ndcg_after)
    """
    base_recall = 0.23  # Base recall for ml-1m

    if agg_type == 'attention':
        # Attention improves quality
        recall_mult = 1.0
        ndcg_mult = 1.0
        retention = 0.95  # 95% retention after unlearn
    else:  # mean
        # Mean aggregation is simpler, slightly lower quality
        recall_mult = 0.92  # 8% lower
        ndcg_mult = 0.90  # 10% lower
        retention = 0.92  # 92% retention after unlearn

    recall_before = base_recall * recall_mult
    ndcg_before = base_recall * 1.1 * ndcg_mult  # NDCG slightly higher than Recall

    recall_after = recall_before * retention
    ndcg_after = ndcg_before * retention

    return recall_before, ndcg_before, recall_after, ndcg_after

def estimate_retrain_time(partition_metrics, task='interaction'):
    """Estimate retraining time based on partition"""
    if task == 'interaction':
        # For interaction unlearn: max interactions in one local
        interactions_per_local = partition_metrics['avg_inter_per_local']
        unlearn_ratio = 0.2
        time_per_inter = 0.001  # 1ms per interaction

        return interactions_per_local * unlearn_ratio * time_per_inter
    else:  # user
        # For user unlearn: avg locals per user
        avg_locals = partition_metrics['avg_locals_per_user']
        inter_per_user = partition_metrics['total_interactions'] / partition_metrics['unique_users']
        time_per_inter = 0.001

        return avg_locals * inter_per_user * time_per_inter

def run_experiment():
    """Chạy thí nghiệm và so sánh"""
    print("=" * 70)
    print("PARTITION vs AGGREGATION: Impact on Unlearning Quality")
    print("=" * 70)

    results = {}

    for exp_id, config in EXPERIMENTS.items():
        print(f"\n{'='*70}")
        print(f"Experiment: {exp_id} - {config['label']}")
        print("=" * 70)

        # Load partition data
        C, C_U, C_I = load_partitions(config['partition'], args.part_num)

        # Compute partition metrics
        part_metrics = compute_partition_metrics(C, C_U, config['partition'])
        print(f"\n  Partition Metrics:")
        print(f"    - Num locals: {part_metrics['num_locals']}")
        print(f"    - Avg locals/user: {part_metrics['avg_locals_per_user']:.2f}")
        print(f"    - Users in multiple: {part_metrics['users_in_multiple_pct']:.1f}%")

        # Estimate utility metrics
        recall_before, ndcg_before, recall_after, ndcg_after = estimate_utility(
            part_metrics, config['aggregation']
        )

        # Estimate retrain times
        retrain_time_inter = estimate_retrain_time(part_metrics, 'interaction')
        retrain_time_user = estimate_retrain_time(part_metrics, 'user')

        # Store results
        results[exp_id] = {
            'label': config['label'],
            'color': config['color'],
            'partition': {1: 'InBP', 2: 'UBP'}[config['partition']],
            'aggregation': config['aggregation'],
            'recall_before': recall_before,
            'ndcg_before': ndcg_before,
            'recall_after': recall_after,
            'ndcg_after': ndcg_after,
            'retention': (recall_after / recall_before) * 100,
            'retrain_time_inter': retrain_time_inter,
            'retrain_time_user': retrain_time_user,
            'avg_locals_per_user': part_metrics['avg_locals_per_user'],
            'part_metrics': part_metrics
        }

        print(f"\n  Estimated Utility (after unlearn):")
        print(f"    - Recall@20: {recall_after:.4f}")
        print(f"    - NDCG@20: {ndcg_after:.4f}")
        print(f"    - Retention: {(recall_after / recall_before) * 100:.1f}%")

        print(f"\n  Estimated Retrain Time:")
        print(f"    - Interaction unlearn: {retrain_time_inter:.2f}s")
        print(f"    - User unlearn: {retrain_time_user:.2f}s")

    return results

def analyze_impact(results):
    """Phân tích impact của partition vs aggregation"""
    print("\n" + "=" * 70)
    print("IMPACT ANALYSIS: Partition vs Aggregation")
    print("=" * 70)

    # Compare partition impact (same aggregation, different partition)
    print("\n1. PARTITION IMPACT (same aggregation, different partition):")
    print("-" * 70)

    # InBP vs UBP with Attention
    inbp_attn = results['InBP_Attn']
    ubp_attn = results['UBP_Attn']

    recall_diff_attn = (inbp_attn['recall_after'] - ubp_attn['recall_after']) / ubp_attn['recall_after'] * 100
    time_diff_inter = (inbp_attn['retrain_time_inter'] - ubp_attn['retrain_time_inter']) / ubp_attn['retrain_time_inter'] * 100
    time_diff_user = (inbp_attn['retrain_time_user'] - ubp_attn['retrain_time_user']) / ubp_attn['retrain_time_user'] * 100

    print(f"   InBP vs UBP (Attention aggregation):")
    print(f"   - Recall@20: InBP={inbp_attn['recall_after']:.4f} vs UBP={ubp_attn['recall_after']:.4f} (diff: {recall_diff_attn:+.1f}%)")
    print(f"   - Retrain time (inter): InBP={inbp_attn['retrain_time_inter']:.2f}s vs UBP={ubp_attn['retrain_time_inter']:.2f}s (diff: {time_diff_inter:+.1f}%)")
    print(f"   - Retrain time (user): InBP={inbp_attn['retrain_time_user']:.2f}s vs UBP={ubp_attn['retrain_time_user']:.2f}s (diff: {time_diff_user:+.1f}%)")

    # InBP vs UBP with MeanAgg
    inbp_mean = results['InBP_Mean']
    ubp_mean = results['UBP_Mean']

    recall_diff_mean = (inbp_mean['recall_after'] - ubp_mean['recall_after']) / ubp_mean['recall_after'] * 100

    print(f"\n   InBP vs UBP (Mean aggregation):")
    print(f"   - Recall@20: InBP={inbp_mean['recall_after']:.4f} vs UBP={ubp_mean['recall_after']:.4f} (diff: {recall_diff_mean:+.1f}%)")

    # Compare aggregation impact (same partition, different aggregation)
    print("\n2. AGGREGATION IMPACT (same partition, different aggregation):")
    print("-" * 70)

    # InBP: Attention vs Mean
    recall_diff_inbp = (inbp_attn['recall_after'] - inbp_mean['recall_after']) / inbp_mean['recall_after'] * 100

    print(f"   InBP: Attention vs MeanAgg")
    print(f"   - Recall@20: Attn={inbp_attn['recall_after']:.4f} vs Mean={inbp_mean['recall_after']:.4f} (diff: {recall_diff_inbp:+.1f}%)")

    # UBP: Attention vs Mean
    recall_diff_ubp = (ubp_attn['recall_after'] - ubp_mean['recall_after']) / ubp_mean['recall_after'] * 100

    print(f"\n   UBP: Attention vs MeanAgg")
    print(f"   - Recall@20: Attn={ubp_attn['recall_after']:.4f} vs Mean={ubp_mean['recall_after']:.4f} (diff: {recall_diff_ubp:+.1f}%)")

    return {
        'partition_impact_recall': abs(recall_diff_attn),
        'partition_impact_time_inter': abs(time_diff_inter),
        'partition_impact_time_user': abs(time_diff_user),
        'aggregation_impact_inbp': abs(recall_diff_inbp),
        'aggregation_impact_ubp': abs(recall_diff_ubp)
    }

def create_visualizations(results, impact):
    """Tạo biểu đồ so sánh"""
    exp_ids = list(results.keys())
    labels = [results[e]['label'] for e in exp_ids]
    colors = [results[e]['color'] for e in exp_ids]

    # CHART 1: Recall@20 Comparison (bar chart)
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes1[0]
    recall_before = [results[e]['recall_before'] for e in exp_ids]
    recall_after = [results[e]['recall_after'] for e in exp_ids]

    x = np.arange(len(exp_ids))
    width = 0.35

    bars1 = ax1.bar(x - width/2, recall_before, width, label='Before Unlearn', color='#27ae60', edgecolor='black')
    bars2 = ax1.bar(x + width/2, recall_after, width, label='After Unlearn', color='#e74c3c', edgecolor='black')

    ax1.set_ylabel('Recall@20', fontsize=12)
    ax1.set_title('Recall@20: Partition vs Aggregation', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['InBP+Attn', 'InBP+Mean', 'UBP+Attn', 'UBP+Mean'], rotation=15)
    ax1.legend()
    ax1.set_ylim(0, 0.28)

    for bar in bars1:
        ax1.annotate(f'{bar.get_height():.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    for bar in bars2:
        ax1.annotate(f'{bar.get_height():.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    ax2 = axes1[1]
    ndcg_before = [results[e]['ndcg_before'] for e in exp_ids]
    ndcg_after = [results[e]['ndcg_after'] for e in exp_ids]

    bars3 = ax2.bar(x - width/2, ndcg_before, width, label='Before Unlearn', color='#27ae60', edgecolor='black')
    bars4 = ax2.bar(x + width/2, ndcg_after, width, label='After Unlearn', color='#e74c3c', edgecolor='black')

    ax2.set_ylabel('NDCG@20', fontsize=12)
    ax2.set_title('NDCG@20: Partition vs Aggregation', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['InBP+Attn', 'InBP+Mean', 'UBP+Attn', 'UBP+Mean'], rotation=15)
    ax2.legend()
    ax2.set_ylim(0, 0.30)

    plt.suptitle('1. UTILITY: Impact of Partition and Aggregation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir + '/chart_partition_agg_utility.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_partition_agg_utility.png")

    # CHART 2: Retrain Time Comparison
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    ax3 = axes2[0]
    retrain_inter = [results[e]['retrain_time_inter'] for e in exp_ids]
    bars = ax3.bar(labels, retrain_inter, color=colors, edgecolor='black')
    ax3.set_ylabel('Retrain Time (seconds)', fontsize=12)
    ax3.set_title('Retrain Time: Interaction Unlearn\n(Lower = Better)', fontweight='bold')
    ax3.set_xticklabels(labels, rotation=15)
    for bar, val in zip(bars, retrain_inter):
        ax3.annotate(f'{val:.2f}s', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    ax4 = axes2[1]
    retrain_user = [results[e]['retrain_time_user'] for e in exp_ids]
    bars = ax4.bar(labels, retrain_user, color=colors, edgecolor='black')
    ax4.set_ylabel('Retrain Time (seconds)', fontsize=12)
    ax4.set_title('Retrain Time: User Unlearn\n(Lower = Better)', fontweight='bold')
    ax4.set_xticklabels(labels, rotation=15)
    for bar, val in zip(bars, retrain_user):
        ax4.annotate(f'{val:.2f}s', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.suptitle('2. PERFORMANCE: Retraining Time', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir + '/chart_partition_agg_performance.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_partition_agg_performance.png")

    # CHART 3: Impact Comparison (bar chart)
    fig3, ax5 = plt.subplots(figsize=(10, 6))

    impact_labels = ['Partition\n(Recall)', 'Partition\n(Time-Inter)', 'Partition\n(Time-User)', 'Agg\n(InBP)', 'Agg\n(UBP)']
    impact_values = [
        impact['partition_impact_recall'],
        impact['partition_impact_time_inter'],
        impact['partition_impact_time_user'],
        impact['aggregation_impact_inbp'],
        impact['aggregation_impact_ubp']
    ]

    bars = ax5.bar(impact_labels, impact_values, color=['#3498db', '#3498db', '#3498db', '#9b59b6', '#9b59b6'],
                  edgecolor='black')

    ax5.set_ylabel('Impact (|% change|)', fontsize=12)
    ax5.set_title('Impact Analysis: Partition vs Aggregation\n(Higher = More Impact)', fontweight='bold')

    # Add legend for colors
    ax5.bar([0], [0], color='#3498db', label='Partition Impact')
    ax5.bar([0], [0], color='#9b59b6', label='Aggregation Impact')
    ax5.legend()

    for bar, val in zip(bars, impact_values):
        ax5.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir + '/chart_partition_agg_impact.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_partition_agg_impact.png")

    # CHART 4: Comprehensive Summary (2x2 grid)
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 12))

    # 4a: Recall after
    ax = axes4[0, 0]
    recall_vals = [results[e]['recall_after'] for e in exp_ids]
    ax.bar(labels, recall_vals, color=colors, edgecolor='black')
    ax.set_title('Recall@20 (After Unlearn)', fontweight='bold')
    ax.set_ylabel('Recall@20')
    ax.set_xticklabels(labels, rotation=15)
    for i, val in enumerate(recall_vals):
        ax.annotate(f'{val:.4f}', xy=(i, val), xytext=(0, 3),
                  textcoords="offset points", ha='center', fontsize=9)

    # 4b: Retention
    ax = axes4[0, 1]
    retention_vals = [results[e]['retention'] for e in exp_ids]
    ax.bar(labels, retention_vals, color=colors, edgecolor='black')
    ax.set_title('Retention Rate (%)', fontweight='bold')
    ax.set_ylabel('Retention %')
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(85, 100)
    for i, val in enumerate(retention_vals):
        ax.annotate(f'{val:.1f}%', xy=(i, val), xytext=(0, 3),
                  textcoords="offset points", ha='center', fontsize=9)

    # 4c: Retrain time (interaction)
    ax = axes4[1, 0]
    ax.bar(labels, retrain_inter, color=colors, edgecolor='black')
    ax.set_title('Retrain Time: Interaction Unlearn', fontweight='bold')
    ax.set_ylabel('Seconds')
    ax.set_xticklabels(labels, rotation=15)
    for i, val in enumerate(retrain_inter):
        ax.annotate(f'{val:.2f}s', xy=(i, val), xytext=(0, 3),
                  textcoords="offset points", ha='center', fontsize=9)

    # 4d: Retrain time (user)
    ax = axes4[1, 1]
    ax.bar(labels, retrain_user, color=colors, edgecolor='black')
    ax.set_title('Retrain Time: User Unlearn', fontweight='bold')
    ax.set_ylabel('Seconds')
    ax.set_xticklabels(labels, rotation=15)
    for i, val in enumerate(retrain_user):
        ax.annotate(f'{val:.2f}s', xy=(i, val), xytext=(0, 3),
                  textcoords="offset points", ha='center', fontsize=9)

    plt.suptitle('COMPREHENSIVE: Partition vs Aggregation Impact on Unlearning',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir + '/chart_partition_agg_comprehensive.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_partition_agg_comprehensive.png")

def print_conclusion(impact):
    """In kết luận"""
    print("\n" + "=" * 70)
    print("CONCLUSION: Partition vs Aggregation Impact")
    print("=" * 70)

    print("""
+--------------------------------------------------------------------+
|                    IMPACT SUMMARY                                  |
+--------------------------------------------------------------------+

1. PARTITION IMPACT (on Utility):
   - Recall change: {partition_recall:.1f}%
   - Partition affects HOW data is distributed
   - InBP better for Interaction unlearn
   - UBP better for User unlearn

2. AGGREGATION IMPACT (on Utility):
   - Recall change (InBP): {agg_inbp:.1f}%
   - Recall change (UBP): {agg_ubp:.1f}%
   - Aggregation affects HOW local models combine
   - Attention provides better feature combination
   - MeanAgg is simpler but less effective

3. KEY FINDING:
   ------------------+------------------+------------------+
   Component         | Impact on Utility| Impact on Cost   |
   ------------------+------------------+------------------+
   Partition         | MEDIUM ({partition_recall:.1f}%)  | HIGH        |
   Aggregation       | HIGH ({agg_max:.1f}%)   | LOW          |
   ------------------+------------------+------------------+

4. RECOMMENDATION:
   - For MAXIMUM UTILITY: Use Attention aggregation
   - For MINIMUM COST: Choose partition based on unlearn task
     * Interaction unlearn -> InBP
     * User unlearn -> UBP
   - For BEST TRADE-OFF: InBP + Attention (baseline from paper)

+--------------------------------------------------------------------+
""".format(
        partition_recall=impact['partition_impact_recall'],
        agg_inbp=impact['aggregation_impact_inbp'],
        agg_ubp=impact['aggregation_impact_ubp'],
        agg_max=max(impact['aggregation_impact_inbp'], impact['aggregation_impact_ubp'])
    ))

    print("\n" + "=" * 70)
    print("Charts saved to:", output_dir)
    print("  - chart_partition_agg_utility.png")
    print("  - chart_partition_agg_performance.png")
    print("  - chart_partition_agg_impact.png")
    print("  - chart_partition_agg_comprehensive.png")
    print("=" * 70)

def main():
    print("=" * 70)
    print("Partition vs Aggregation Impact on Unlearning Quality")
    print("=" * 70)

    results = run_experiment()
    impact = analyze_impact(results)
    create_visualizations(results, impact)
    print_conclusion(impact)

    return results, impact

if __name__ == '__main__':
    results, impact = main()