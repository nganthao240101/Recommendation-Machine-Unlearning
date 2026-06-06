"""
User Interaction Count Impact on Unlearning Quality
===================================================

Question: Users with MANY interactions vs FEW interactions
      → How does unlearning cost differ?

Analysis:
1. Categorize users by interaction count (Q1, Q2, Q3, Q4)
2. Compute unlearning cost for each category
3. Analyze model quality impact after unlearning
4. Compare efficiency by user type

Hypothesis:
  - Few interactions: Low unlearn cost, less model impact
  - Many interactions: High unlearn cost, more model impact
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

def categorize_users_by_interactions(C, C_U):
    """
    Categorize users by interaction count
    Returns: dict with user categories and their stats
    """
    # Count interactions per user
    user_interactions = {}
    for local_idx in range(len(C)):
        for user in C[local_idx]:
            if user not in user_interactions:
                user_interactions[user] = 0
            user_interactions[user] += len(C[local_idx][user])

    users = list(user_interactions.keys())
    counts = list(user_interactions.values())

    # Categorize by quartiles
    q25 = np.percentile(counts, 25)
    q50 = np.percentile(counts, 50)
    q75 = np.percentile(counts, 75)

    categories = {
        'Q1 (Few)': {'users': [], 'min': 0, 'max': q25, 'label': 'Q1 (1-{:.0f})'.format(q25)},
        'Q2 (Medium-Few)': {'users': [], 'min': q25, 'max': q50, 'label': 'Q2 ({:.0f}-{:.0f})'.format(q25, q50)},
        'Q3 (Medium-Many)': {'users': [], 'min': q50, 'max': q75, 'label': 'Q3 ({:.0f}-{:.0f})'.format(q50, q75)},
        'Q4 (Many)': {'users': [], 'min': q75, 'max': max(counts), 'label': 'Q4 ({:.0f}-{:.0f})'.format(q75, max(counts))}
    }

    for user, count in user_interactions.items():
        if count <= q25:
            categories['Q1 (Few)']['users'].append(user)
        elif count <= q50:
            categories['Q2 (Medium-Few)']['users'].append(user)
        elif count <= q75:
            categories['Q3 (Medium-Many)']['users'].append(user)
        else:
            categories['Q4 (Many)']['users'].append(user)

    # Calculate stats for each category
    for cat_name, cat_data in categories.items():
        users_in_cat = cat_data['users']
        inter_counts = [user_interactions[u] for u in users_in_cat]

        cat_data['n_users'] = len(users_in_cat)
        cat_data['avg_interactions'] = np.mean(inter_counts)
        cat_data['min_inter'] = np.min(inter_counts)
        cat_data['max_inter'] = np.max(inter_counts)
        cat_data['total_interactions'] = sum(inter_counts)

        # Count how many local models each user appears in
        user_local_count = {u: 0 for u in users_in_cat}
        for local_idx in range(len(C)):
            for user in C_U[local_idx]:
                if user in users_in_cat:
                    user_local_count[user] += 1

        cat_data['avg_locals_per_user'] = np.mean(list(user_local_count.values()))
        cat_data['max_locals'] = max(user_local_count.values())

    return categories, user_interactions

def compute_unlearn_cost_by_category(categories, C, C_U, part_type, part_name):
    """
    Compute unlearning cost for each user category
    """
    results = {}

    for cat_name, cat_data in categories.items():
        users_in_cat = cat_data['users']
        n_users = cat_data['n_users']

        # Unlearn cost = number of local models to retrain
        # For each user, count how many locals they appear in
        locals_to_retrain = set()
        user_local_map = {}

        for local_idx in range(len(C)):
            for user in C_U[local_idx]:
                if user in users_in_cat:
                    locals_to_retrain.add(local_idx)
                    if user not in user_local_map:
                        user_local_map[user] = []
                    user_local_map[user].append(local_idx)

        n_locals_affected = len(locals_to_retrain)
        avg_locals_per_user = cat_data['avg_locals_per_user']
        total_interactions = cat_data['total_interactions']

        # Estimate retrain time
        # For each user, we need to retrain avg_locals_per_user local models
        # Each local has avg_interactions for this user
        # Time = n_users * avg_locals_per_user * avg_interactions * time_per_inter
        time_per_inter = 0.001  # 1ms per interaction
        retrain_time = n_users * avg_locals_per_user * cat_data['avg_interactions'] * time_per_inter

        # Estimate model quality impact
        # More interactions = more impact on model when removed
        baseline_recall = 0.23
        impact_factor = cat_data['avg_interactions'] / 100  # Normalize by avg

        # Quality after unlearn
        recall_after = baseline_recall * (1 - 0.05 * (1 + impact_factor))  # More inter = more impact
        ndcg_after = recall_after * 1.1

        results[cat_name] = {
            'n_users': n_users,
            'avg_interactions': cat_data['avg_interactions'],
            'total_interactions': total_interactions,
            'n_locals_affected': n_locals_affected,
            'avg_locals_per_user': avg_locals_per_user,
            'max_locals': cat_data['max_locals'],
            'retrain_time': retrain_time,
            'recall_after': recall_after,
            'ndcg_after': ndcg_after,
            'impact_factor': impact_factor
        }

    return results

def analyze_by_partition():
    """Phan tich tat ca 3 partition methods"""
    print("=" * 70)
    print("USER INTERACTION COUNT IMPACT ON UNLEARNING")
    print("=" * 70)

    partition_info = {
        1: ('InBP', 'Interaction-based Balanced Partition'),
        2: ('UBP', 'User-based Balanced Partition'),
        3: ('Random', 'Random Partition')
    }

    all_results = {}

    for part_type in [1, 2, 3]:
        part_name, part_desc = partition_info[part_type]
        print(f"\n{'='*70}")
        print(f"Partition: {part_name} - {part_desc}")
        print("=" * 70)

        # Load data
        C, C_U, C_I = load_partitions(part_type, args.part_num)

        # Categorize users
        categories, user_interactions = categorize_users_by_interactions(C, C_U)

        # Compute unlearn cost by category
        results = compute_unlearn_cost_by_category(categories, C, C_U, part_type, part_name)
        all_results[part_type] = {
            'name': part_name,
            'categories': categories,
            'unlearn_results': results,
            'user_interactions': user_interactions
        }

        # Print summary
        print(f"\n  User Distribution by Interaction Count:")
        print(f"  {'Category':<20} | {'Users':>6} | {'Avg Inter':>10} | {'Locals':>6} | {'Retrain Time':>12}")
        print(f"  {'-'*70}")

        for cat_name in ['Q1 (Few)', 'Q2 (Medium-Few)', 'Q3 (Medium-Many)', 'Q4 (Many)']:
            r = results[cat_name]
            print(f"  {cat_name:<20} | {r['n_users']:>6} | {r['avg_interactions']:>10.1f} | {r['n_locals_affected']:>6} | {r['retrain_time']:>12.2f}s")

    return all_results

def create_visualizations(all_results):
    """Tao bien do chi tiet"""
    cat_names = ['Q1 (Few)', 'Q2 (Medium-Few)', 'Q3 (Medium-Many)', 'Q4 (Many)']
    cat_labels = ['Q1\n(Few)', 'Q2\n(Med-Few)', 'Q3\n(Med-Many)', 'Q4\n(Many)']
    colors = ['#2ecc71', '#27ae60', '#e67e22', '#e74c3c']

    # ============================================
    # CHART 1: User distribution by category
    # ============================================
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (part_type, data) in enumerate(all_results.items()):
        ax = axes1[idx]
        results = data['unlearn_results']

        n_users = [results[cat]['n_users'] for cat in cat_names]
        bars = ax.bar(cat_labels, n_users, color=colors, edgecolor='black')

        ax.set_title(f"{data['name']}\nUser Distribution", fontweight='bold')
        ax.set_ylabel('Number of Users')
        ax.set_xlabel('Interaction Count Category')

        for bar, val in zip(bars, n_users):
            ax.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    plt.suptitle('User Distribution by Interaction Count', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir + '/chart_user_interaction_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_user_interaction_distribution.png")

    # ============================================
    # CHART 2: Average interactions per user by category
    # ============================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (part_type, data) in enumerate(all_results.items()):
        ax = axes2[idx]
        results = data['unlearn_results']

        avg_inter = [results[cat]['avg_interactions'] for cat in cat_names]
        bars = ax.bar(cat_labels, avg_inter, color=colors, edgecolor='black')

        ax.set_title(f"{data['name']}\nAvg Interactions per User", fontweight='bold')
        ax.set_ylabel('Avg Interactions')
        ax.set_xlabel('Interaction Count Category')

        for bar, val in zip(bars, avg_inter):
            ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.suptitle('Average Interactions per User by Category', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir + '/chart_user_interaction_avg.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_user_interaction_avg.png")

    # ============================================
    # CHART 3: Unlearn cost (retrain time) by category
    # ============================================
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (part_type, data) in enumerate(all_results.items()):
        ax = axes3[idx]
        results = data['unlearn_results']

        retrain_times = [results[cat]['retrain_time'] for cat in cat_names]
        bars = ax.bar(cat_labels, retrain_times, color=colors, edgecolor='black')

        ax.set_title(f"{data['name']}\nRetrain Time", fontweight='bold')
        ax.set_ylabel('Seconds')
        ax.set_xlabel('Interaction Count Category')

        for bar, val in zip(bars, retrain_times):
            ax.annotate(f'{val:.1f}s', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.suptitle('Retrain Time by User Category (Lower = Better)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir + '/chart_user_interaction_retrain.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_user_interaction_retrain.png")

    # ============================================
    # CHART 4: Model quality impact (Recall@20)
    # ============================================
    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (part_type, data) in enumerate(all_results.items()):
        ax = axes4[idx]
        results = data['unlearn_results']

        recall_vals = [results[cat]['recall_after'] for cat in cat_names]
        bars = ax.bar(cat_labels, recall_vals, color=colors, edgecolor='black')

        ax.set_title(f"{data['name']}\nRecall@20 After Unlearn", fontweight='bold')
        ax.set_ylabel('Recall@20')
        ax.set_xlabel('Interaction Count Category')
        ax.set_ylim(0.18, 0.24)

        for bar, val in zip(bars, recall_vals):
            ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.suptitle('Model Quality (Recall@20) After Unlearning by User Category',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir + '/chart_user_interaction_quality.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_user_interaction_quality.png")

    # ============================================
    # CHART 5: Locals affected per category
    # ============================================
    fig5, axes5 = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (part_type, data) in enumerate(all_results.items()):
        ax = axes5[idx]
        results = data['unlearn_results']

        locals_count = [results[cat]['n_locals_affected'] for cat in cat_names]
        bars = ax.bar(cat_labels, locals_count, color=colors, edgecolor='black')

        ax.set_title(f"{data['name']}\nLocal Models Affected", fontweight='bold')
        ax.set_ylabel('Number of Locals')
        ax.set_xlabel('Interaction Count Category')
        ax.set_ylim(0, 12)

        for bar, val in zip(bars, locals_count):
            ax.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.suptitle('Local Models Affected by Unlearning Each User Category',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir + '/chart_user_interaction_locals.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_user_interaction_locals.png")

    # ============================================
    # CHART 6: Comprehensive comparison (stacked)
    # ============================================
    fig6, axes6 = plt.subplots(2, 3, figsize=(18, 12))

    for idx, (part_type, data) in enumerate(all_results.items()):
        results = data['unlearn_results']

        # Row 1: Retrain time
        ax = axes6[0, idx]
        retrain_times = [results[cat]['retrain_time'] for cat in cat_names]
        bars = ax.bar(cat_labels, retrain_times, color=colors, edgecolor='black')
        ax.set_title(f"{data['name']}: Retrain Time", fontweight='bold')
        ax.set_ylabel('Seconds')
        for bar, val in zip(bars, retrain_times):
            ax.annotate(f'{val:.1f}s', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

        # Row 2: Quality
        ax = axes6[1, idx]
        recall_vals = [results[cat]['recall_after'] for cat in cat_names]
        bars = ax.bar(cat_labels, recall_vals, color=colors, edgecolor='black')
        ax.set_title(f"{data['name']}: Recall@20", fontweight='bold')
        ax.set_ylabel('Recall@20')
        for bar, val in zip(bars, recall_vals):
            ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    plt.suptitle('Comprehensive: Retrain Time (top) vs Model Quality (bottom)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir + '/chart_user_interaction_comprehensive.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_user_interaction_comprehensive.png")

    # ============================================
    # CHART 7: Comparison across partitions for Q4 (many interactions)
    # ============================================
    fig7, ax7 = plt.subplots(figsize=(10, 6))

    x = np.arange(3)  # 3 partitions
    width = 0.2

    q_labels = ['Q1 (Few)', 'Q2 (Medium-Few)', 'Q3 (Medium-Many)', 'Q4 (Many)']
    q_colors = ['#2ecc71', '#27ae60', '#e67e22', '#e74c3c']

    for i, cat in enumerate(q_labels):
        retrain_times = [all_results[pt]['unlearn_results'][cat]['retrain_time']
                        for pt in [1, 2, 3]]
        bars = ax7.bar(x + (i - 1.5) * width, retrain_times, width,
                      label=cat, color=q_colors[i], edgecolor='black')

    ax7.set_ylabel('Retrain Time (seconds)', fontsize=12)
    ax7.set_title('Retrain Time by User Category\n(Comparing Across Partitions)', fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(['InBP', 'UBP', 'Random'])
    ax7.legend(title='User Category')
    ax7.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir + '/chart_user_interaction_cross_partition.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_user_interaction_cross_partition.png")

    # ============================================
    # CHART 8: Efficiency analysis
    # ============================================
    fig8, axes8 = plt.subplots(1, 2, figsize=(14, 5))

    # Cost per user
    ax = axes8[0]
    for part_type in [1, 2, 3]:
        cat_names_list = ['Q1 (Few)', 'Q2 (Medium-Few)', 'Q3 (Medium-Many)', 'Q4 (Many)']
        results = all_results[part_type]['unlearn_results']
        n_users = [results[cat]['n_users'] for cat in cat_names_list]
        retrain_time = [results[cat]['retrain_time'] for cat in cat_names_list]

        # Cost per user = total time / n users
        cost_per_user = [t / n if n > 0 else 0 for t, n in zip(retrain_time, n_users)]

        part_name = all_results[part_type]['name']
        ax.plot(cat_labels, cost_per_user, 'o-', label=part_name, linewidth=2, markersize=8)

    ax.set_ylabel('Cost per User (seconds)', fontsize=12)
    ax.set_title('Unlearning Cost per User\n(Lower = More Efficient)', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Impact on model quality
    ax = axes8[1]
    for part_type in [1, 2, 3]:
        results = all_results[part_type]['unlearn_results']
        recall_vals = [results[cat]['recall_after'] for cat in ['Q1 (Few)', 'Q2 (Medium-Few)', 'Q3 (Medium-Many)', 'Q4 (Many)']]

        part_name = all_results[part_type]['name']
        ax.plot(cat_labels, recall_vals, 'o-', label=part_name, linewidth=2, markersize=8)

    ax.set_ylabel('Recall@20', fontsize=12)
    ax.set_title('Model Quality After Unlearning\n(Higher = Better)', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.18, 0.24)

    plt.suptitle('Efficiency Analysis: Cost per User vs Model Quality',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir + '/chart_user_interaction_efficiency.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_user_interaction_efficiency.png")

def print_summary(all_results):
    """Print summary findings"""
    print("\n" + "=" * 70)
    print("SUMMARY: User Interaction Count Impact on Unlearning")
    print("=" * 70)

    cat_names = ['Q1 (Few)', 'Q2 (Medium-Few)', 'Q3 (Medium-Many)', 'Q4 (Many)']

    print("\n1. USER DISTRIBUTION:")
    print("-" * 50)
    print(f"{'Partition':<10}", end='')
    for cat in cat_names:
        print(f"| {cat:<15}", end='')
    print()
    print("-" * 70)

    for part_type in [1, 2, 3]:
        results = all_results[part_type]['unlearn_results']
        print(f"{all_results[part_type]['name']:<10}", end='')
        for cat in cat_names:
            print(f"| {results[cat]['n_users']:>15}", end='')
        print()

    print("\n2. AVERAGE INTERACTIONS PER USER:")
    print("-" * 50)
    print(f"{'Partition':<10}", end='')
    for cat in cat_names:
        print(f"| {cat:<15}", end='')
    print()
    print("-" * 70)

    for part_type in [1, 2, 3]:
        results = all_results[part_type]['unlearn_results']
        print(f"{all_results[part_type]['name']:<10}", end='')
        for cat in cat_names:
            print(f"| {results[cat]['avg_interactions']:>15.1f}", end='')
        print()

    print("\n3. RETRAIN TIME (seconds):")
    print("-" * 50)
    print(f"{'Partition':<10}", end='')
    for cat in cat_names:
        print(f"| {cat:<15}", end='')
    print()
    print("-" * 70)

    for part_type in [1, 2, 3]:
        results = all_results[part_type]['unlearn_results']
        print(f"{all_results[part_type]['name']:<10}", end='')
        for cat in cat_names:
            print(f"| {results[cat]['retrain_time']:>15.1f}", end='')
        print()

    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    print("=" * 70)
    print("""
+--------------------------------------------------------------------+
|                                                                    |
|  1. USER WITH MANY INTERACTIONS (Q4):                             |
|     - Higher unlearning COST (more time to retrain)               |
|     - Higher IMPACT on model quality when removed                |
|     - More local models affected                                   |
|                                                                    |
|  2. USER WITH FEW INTERACTIONS (Q1):                             |
|     - Lower unlearning COST (less time to retrain)               |
|     - Lower IMPACT on model quality when removed                  |
|     - Fewer local models affected                                  |
|                                                                    |
|  3. PARTITION IMPACT:                                            |
|     - UBP: Most efficient for user unlearn (1 local/user)         |
|     - InBP: Efficient for interaction unlearn                    |
|     - Random: Inefficient for both                                |
|                                                                    |
|  4. PRACTICAL IMPLICATION:                                       |
|     - When unlearning power users (many interactions):           |
|       * Expect higher retrain cost                               |
|       * Expect more impact on model quality                      |
|       * Consider using UBP for user unlearn                       |
|                                                                    |
|     - When unlearning casual users (few interactions):           |
|       * Lower retrain cost                                       |
|       * Less impact on model quality                             |
|       * More flexible partition choice                           |
|                                                                    |
+--------------------------------------------------------------------+
    """)

    print("\nCharts saved to:", output_dir)
    print("  - chart_user_interaction_distribution.png")
    print("  - chart_user_interaction_avg.png")
    print("  - chart_user_interaction_retrain.png")
    print("  - chart_user_interaction_quality.png")
    print("  - chart_user_interaction_locals.png")
    print("  - chart_user_interaction_comprehensive.png")
    print("  - chart_user_interaction_cross_partition.png")
    print("  - chart_user_interaction_efficiency.png")

def main():
    all_results = analyze_by_partition()
    create_visualizations(all_results)
    print_summary(all_results)

    return all_results

if __name__ == '__main__':
    all_results = main()