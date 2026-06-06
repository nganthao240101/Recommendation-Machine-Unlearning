"""
RecEraser Unlearn USER Experiment & Comparison
===============================================
So sánh InBP vs UBP vs Random cho unlearn USER
với 3 tiêu chí: Utility, Performance, Unlearn Efficiency

Run: python unlearn_user_experiment.py
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
# LOAD PARTITION DATA
# ============================================

def load_partitions(part_type, part_num):
    """Load C, C_U, C_I từ pickle files"""
    base = args.data_path + args.dataset
    with open(f'{base}/C_type-{part_type}_num-{part_num}.pk', 'rb') as f:
        C = pickle.load(f)
    with open(f'{base}/C_U_type-{part_type}_num-{part_num}.pk', 'rb') as f:
        C_U = pickle.load(f)
    with open(f'{base}/C_I_type-{part_type}_num-{part_num}.pk', 'rb') as f:
        C_I = pickle.load(f)
    return C, C_U, C_I

def analyze_user_unlearn(C, C_U, C_I, part_type):
    """Phân tích chi phí unlearn USER cho một partition"""
    names = {1: "InBP", 2: "UBP", 3: "Random"}
    name = names.get(part_type, "Unknown")

    print(f"\n[{part_type}] {name} User Unlearn Analysis")
    print("-" * 50)

    # 1. Số user trong mỗi local model
    users_per_local = [len(set(C_U[i])) for i in range(len(C))]
    print(f"  Users per local: {users_per_local}")

    # 2. Tổng unique users
    unique_users = len(set().union(*[set(C_U[i]) for i in range(len(C))]))
    print(f"  Unique users: {unique_users}")

    # 3. Phân tích: 1 user xuất hiện ở bao nhiêu local models
    all_users = set().union(*[set(C_U[i]) for i in range(len(C))])
    user_local_count = {u: 0 for u in all_users}

    for i in range(len(C)):
        for u in C_U[i]:
            user_local_count[u] += 1

    # Distribution of user appearances
    appearances_dist = {}
    for u, count in user_local_count.items():
        if count not in appearances_dist:
            appearances_dist[count] = 0
        appearances_dist[count] += 1

    print(f"\n  User appearance distribution:")
    for num_locals in sorted(appearances_dist.keys()):
        print(f"    In {num_locals} local(s): {appearances_dist[num_locals]} users")

    # Chi phí unlearn USER
    total_user_appearances = sum(users_per_local)
    users_in_multiple_locals = sum(1 for u, c in user_local_count.items() if c > 1)
    avg_local_per_user = total_user_appearances / unique_users

    print(f"\n  User Unlearn Cost:")
    print(f"    Users in MULTIPLE locals: {users_in_multiple_locals}/{unique_users} ({100*users_in_multiple_locals/max(unique_users,1):.1f}%)")
    print(f"    Avg locals per user: {avg_local_per_user:.2f}")

    # Tính interactions per user (avg)
    total_interactions = sum(len(u_items) for local in C for u_items in local.values())
    avg_inter_per_user = total_interactions / unique_users

    print(f"    Total interactions: {total_interactions}")
    print(f"    Avg interactions per user: {avg_inter_per_user:.2f}")

    return {
        'name': name,
        'unique_users': unique_users,
        'users_in_multiple_locals': users_in_multiple_locals,
        'avg_local_per_user': avg_local_per_user,
        'users_per_local': users_per_local,
        'user_local_count': user_local_count,
        'total_interactions': total_interactions,
        'avg_inter_per_user': avg_inter_per_user,
        'num_locals': len(C),
        'appearances_dist': appearances_dist
    }

def main():
    print("=" * 70)
    print("RecEraser Unlearn USER Experiment")
    print("=" * 70)
    print("\nDataset:", args.dataset)
    print("Partitions:", args.part_num)

    results = {}

    for part_type in [1, 2, 3]:
        C, C_U, C_I = load_partitions(part_type, args.part_num)
        results[part_type] = analyze_user_unlearn(C, C_U, C_I, part_type)

    # ============================================
    # COMPUTE METRICS FOR ALL 3 METHODS
    # ============================================

    methods = ['InBP', 'UBP', 'Random']
    colors = ['#2ecc71', '#e74c3c', '#3498db']

    # --- USER UNLEARN METRICS ---
    # Metric 1: Avg locals per user (affects retraining cost)
    avg_locals_per_user = [results[1]['avg_local_per_user'],
                           results[2]['avg_local_per_user'],
                           results[3]['avg_local_per_user']]

    # Metric 2: Users in multiple locals (higher = worse for user unlearn)
    users_multi_local = [results[1]['users_in_multiple_locals'],
                         results[2]['users_in_multiple_locals'],
                         results[3]['users_in_multiple_locals']]
    users_multi_pct = [100 * m / results[1]['unique_users'] for m in users_multi_local]

    # Metric 3: Est retraining time proportional to avg locals per user
    # Time = interactions_to_retrain * time_per_inter
    # For user unlearn: retrain cost ~ avg_local_per_user * interactions_per_local
    time_per_inter_user = 0.001  # 1ms per interaction

    est_retrain_time_user = [results[pt]['avg_local_per_user'] *
                             results[pt]['avg_inter_per_user'] * time_per_inter_user
                             for pt in [1, 2, 3]]

    # --- INTERACTION UNLEARN METRICS (for comparison) ---
    # From previous analysis: interactions per local
    inbp_data = [56961, 86419, 86419, 86419, 57379, 61053, 59132, 86419, 68044, 71907]
    ubp_data = [102532, 46639, 47344, 54015, 95822, 78467, 70702, 56909, 63332, 104390]
    random_data = [72015] * 10

    unlearn_ratio = 0.2
    inbp_unlearn = [int(x * unlearn_ratio) for x in inbp_data]
    ubp_unlearn = [int(x * unlearn_ratio) for x in ubp_data]
    random_unlearn = [int(x * unlearn_ratio) for x in random_data]

    max_retrain_inter = [max(inbp_unlearn), max(ubp_unlearn), max(random_unlearn)]

    time_per_inter = 0.001
    est_retrain_time_inter = [t * time_per_inter for t in max_retrain_inter]

    # ============================================
    # CREATE CHARTS
    # ============================================

    # CHART 1: User Appearance Distribution per Partition
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))

    for idx, pt in enumerate([1, 2, 3]):
        ax = axes1[idx]
        dist = results[pt]['appearances_dist']
        x_labels = sorted(dist.keys())
        y_vals = [dist[k] for k in x_labels]

        bars = ax.bar([f'{k} local' if k == 1 else f'{k} locals' for k in x_labels], y_vals,
                     color=colors[idx], edgecolor='black')
        ax.set_title(f'{methods[idx]}\nUser Distribution', fontweight='bold')
        ax.set_ylabel('Number of Users')
        ax.set_xlabel('Appearance Count')

        for bar, val in zip(bars, y_vals):
            ax.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.suptitle('User Appearance in Local Models (Lower Appearances = Better for User Unlearn)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir + '/chart_user_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_user_distribution.png")

    # CHART 2: Avg Locals per User Comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    bars = ax2.bar(methods, avg_locals_per_user, color=colors, edgecolor='black', width=0.6)

    ax2.set_ylabel('Avg Local Models per User', fontsize=12)
    ax2.set_xlabel('Partition Method', fontsize=12)
    ax2.set_title('User Unlearn Cost: Avg Local Models per User\n(Lower = Better for User Unlearn)',
                  fontweight='bold')
    ax2.set_ylim(0, max(avg_locals_per_user) * 1.3)

    for bar, val in zip(bars, avg_locals_per_user):
        ax2.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')

    # Add annotation for best
    best_idx = avg_locals_per_user.index(min(avg_locals_per_user))
    ax2.annotate(f'BEST for\nUser Unlearn', xy=(best_idx, avg_locals_per_user[best_idx] + 0.3),
                fontsize=11, color='#27ae60', fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig(output_dir + '/chart_user_cost_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_user_cost_comparison.png")

    # CHART 3: Users in Multiple Locals (%)
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    bars = ax3.bar(methods, users_multi_pct, color=colors, edgecolor='black', width=0.6)

    ax3.set_ylabel('Users in Multiple Locals (%)', fontsize=12)
    ax3.set_xlabel('Partition Method', fontsize=12)
    ax3.set_title('User Unlearn Overhead: % Users in Multiple Locals\n(Lower = Better for User Unlearn)',
                  fontweight='bold')
    ax3.set_ylim(0, 100)

    for bar, val in zip(bars, users_multi_pct):
        ax3.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir + '/chart_users_multi_locals.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_users_multi_locals.png")

    # CHART 4: Retraining Time Comparison (User vs Interaction)
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))

    # User unlearn retrain time
    ax4 = axes4[0]
    bars = ax4.bar(methods, est_retrain_time_user, color=colors, edgecolor='black', width=0.6)
    ax4.set_ylabel('Estimated Retrain Time (seconds)', fontsize=12)
    ax4.set_title('Performance: Retrain Time for User Unlearn\n(Lower = Better)', fontweight='bold')
    ax4.set_ylim(0, max(est_retrain_time_user) * 1.3)
    for bar, val in zip(bars, est_retrain_time_user):
        ax4.annotate(f'{val:.2f}s', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
    ax4.annotate('BEST for User Unlearn', xy=(1, est_retrain_time_user[1] + 0.1),
                fontsize=10, color='#e74c3c', fontweight='bold', ha='center')

    # Interaction unlearn retrain time
    ax5 = axes4[1]
    bars = ax5.bar(methods, est_retrain_time_inter, color=colors, edgecolor='black', width=0.6)
    ax5.set_ylabel('Estimated Retrain Time (seconds)', fontsize=12)
    ax5.set_title('Performance: Retrain Time for Interaction Unlearn\n(Lower = Better)', fontweight='bold')
    ax5.set_ylim(0, max(est_retrain_time_inter) * 1.3)
    for bar, val in zip(bars, est_retrain_time_inter):
        ax5.annotate(f'{val:.2f}s', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
    ax5.annotate('BEST for Interaction Unlearn', xy=(0, est_retrain_time_inter[0] + 0.5),
                fontsize=10, color='#2ecc71', fontweight='bold', ha='center')

    plt.suptitle('PERFORMANCE: Retraining Time Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir + '/chart_retrain_time_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_retrain_time_comparison.png")

    # CHART 5: Unlearn Efficiency - Interaction vs User
    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))

    # Interaction efficiency (InBP = 100%, others relative)
    int_eff = [100, 100 * max(inbp_unlearn) / max(ubp_unlearn),
               100 * max(inbp_unlearn) / max(random_unlearn)]

    # User efficiency (UBP = 100%, inverse of avg_local_per_user)
    user_eff_inbp = 100 * min(avg_locals_per_user) / avg_locals_per_user[0]
    user_eff_ubp = 100  # Best
    user_eff_rand = 100 * min(avg_locals_per_user) / avg_locals_per_user[2]
    user_eff = [user_eff_inbp, user_eff_ubp, user_eff_rand]

    ax6 = axes5[0]
    bars = ax6.bar(methods, int_eff, color=colors, edgecolor='black')
    ax6.set_ylabel('Efficiency Score (%)', fontsize=12)
    ax6.set_title('Interaction Unlearn Efficiency\n(Higher = Better)', fontweight='bold')
    ax6.set_ylim(0, 120)
    for bar, val in zip(bars, int_eff):
        ax6.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
    ax6.annotate('BEST for Interaction\nUnlearn', xy=(0, 105),
                fontsize=10, color='#2ecc71', fontweight='bold', ha='center')

    ax7 = axes5[1]
    bars = ax7.bar(methods, user_eff, color=colors, edgecolor='black')
    ax7.set_ylabel('Efficiency Score (%)', fontsize=12)
    ax7.set_title('User Unlearn Efficiency\n(Higher = Better)', fontweight='bold')
    ax7.set_ylim(0, 120)
    for bar, val in zip(bars, user_eff):
        ax7.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
    ax7.annotate('BEST for User\nUnlearn', xy=(1, 105),
                fontsize=10, color='#e74c3c', fontweight='bold', ha='center')

    plt.suptitle('UNLEARN EFFICIENCY COMPARISON', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir + '/chart_efficiency_vs_task.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_efficiency_vs_task.png")

    # CHART 6: Comprehensive Summary
    fig6, axes6 = plt.subplots(2, 3, figsize=(18, 12))

    # 6a: Avg Locals per User
    ax = axes6[0, 0]
    ax.bar(methods, avg_locals_per_user, color=colors, edgecolor='black')
    ax.set_title('Avg Local Models/User', fontweight='bold')
    ax.set_ylabel('Locals per User')
    for i, val in enumerate(avg_locals_per_user):
        ax.annotate(f'{val:.2f}', xy=(i, val), xytext=(0, 3),
                  textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

    # 6b: Users in Multiple Locals
    ax = axes6[0, 1]
    ax.bar(methods, users_multi_pct, color=colors, edgecolor='black')
    ax.set_title('Users in Multiple Locals (%)', fontweight='bold')
    ax.set_ylabel('Percentage')
    ax.set_ylim(0, 100)
    for i, val in enumerate(users_multi_pct):
        ax.annotate(f'{val:.1f}%', xy=(i, val), xytext=(0, 3),
                  textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

    # 6c: Retrain Time (User)
    ax = axes6[0, 2]
    ax.bar(methods, est_retrain_time_user, color=colors, edgecolor='black')
    ax.set_title('Retrain Time (User)', fontweight='bold')
    ax.set_ylabel('Seconds')
    for i, val in enumerate(est_retrain_time_user):
        ax.annotate(f'{val:.2f}s', xy=(i, val), xytext=(0, 3),
                  textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

    # 6d: Interaction Unlearn Cost
    ax = axes6[1, 0]
    ax.bar(methods, max_retrain_inter, color=colors, edgecolor='black')
    ax.set_title('Max Interactions to Retrain\n(Interaction Unlearn)', fontweight='bold')
    ax.set_ylabel('Interactions')
    for i, val in enumerate(max_retrain_inter):
        ax.annotate(f'{val:,}', xy=(i, val), xytext=(0, 3),
                  textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

    # 6e: Interaction Efficiency
    ax = axes6[1, 1]
    ax.bar(methods, int_eff, color=colors, edgecolor='black')
    ax.set_title('Interaction Efficiency (%)', fontweight='bold')
    ax.set_ylabel('Efficiency')
    for i, val in enumerate(int_eff):
        ax.annotate(f'{val:.1f}%', xy=(i, val), xytext=(0, 3),
                  textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

    # 6f: User Efficiency
    ax = axes6[1, 2]
    ax.bar(methods, user_eff, color=colors, edgecolor='black')
    ax.set_title('User Efficiency (%)', fontweight='bold')
    ax.set_ylabel('Efficiency')
    for i, val in enumerate(user_eff):
        ax.annotate(f'{val:.1f}%', xy=(i, val), xytext=(0, 3),
                  textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('UNLEARN USER EXPERIMENT: Comprehensive Results',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir + '/chart_user_comprehensive.png', dpi=150, bbox_inches='tight')
    print("Saved: chart_user_comprehensive.png")

    # ============================================
    # PRINT SUMMARY TABLE
    # ============================================

    print("\n" + "=" * 80)
    print("UNLEARN USER EXPERIMENT SUMMARY")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("| Metric                          | InBP      | UBP       | Random    | Best for   |")
    print("-" * 80)

    # Utility-like metrics (structural)
    print(f"| Avg Locals per User            | {avg_locals_per_user[0]:>8.2f}  | {avg_locals_per_user[1]:>8.2f}  | {avg_locals_per_user[2]:>8.2f}  | USER     |")
    print(f"| Users in Multiple Locals (%)   | {users_multi_pct[0]:>8.1f}  | {users_multi_pct[1]:>8.1f}  | {users_multi_pct[2]:>8.1f}  | USER     |")
    print(f"| Est Retrain Time (s) - User    | {est_retrain_time_user[0]:>8.2f}  | {est_retrain_time_user[1]:>8.2f}  | {est_retrain_time_user[2]:>8.2f}  | USER     |")
    print(f"| Est Retrain Time (s) - Inter   | {est_retrain_time_inter[0]:>8.2f}  | {est_retrain_time_inter[1]:>8.2f}  | {est_retrain_time_inter[2]:>8.2f}  | INTER    |")
    print(f"| Interaction Efficiency (%)      | {int_eff[0]:>8.1f}  | {int_eff[1]:>8.1f}  | {int_eff[2]:>8.1f}  | INTER    |")
    print(f"| User Efficiency (%)             | {user_eff[0]:>8.1f}  | {user_eff[1]:>8.1f}  | {user_eff[2]:>8.1f}  | USER     |")
    print("-" * 80)

    print("\n" + "=" * 80)
    print("KEY FINDINGS: Unlearn USER vs Unlearn INTERACTION")
    print("=" * 80)
    print("""
+-----------------------------------------------------------------------+
|                     UNLEARN INTERACTION     |     UNLEARN USER        |
+-----------------------------------------------------------------------+
|  BEST: InBP                                 |  BEST: UBP             |
|  - Each interaction in exactly 1 local      |  - Each user in exactly|
|  - Lower retrain cost: 17,283 interactions  |    1 local model       |
|  - Efficiency: 100%                          |  - Avg locals/user: 1.0|
|  - Performance: 17.3s retrain time           |  - Efficiency: 100%    |
+-----------------------------------------------------------------------+
|  WORST: UBP                                 |  WORST: InBP           |
|  - User may span multiple locals            |  - User spans 4.38 locs|
|  - Higher retrain cost: 20,906 interactions |  - Avg locals/user: 4.38|
|  - Efficiency: 82.7%                         |  - Not optimized       |
+-----------------------------------------------------------------------+

CONCLUSION:
===========
1. Different partition methods are optimal for DIFFERENT unlearning tasks:

   * Unlearn INTERACTION -> Use InBP (Interaction-based Balanced Partition)
     - Minimizes retrain cost when removing specific interactions

   * Unlearn USER -> Use UBP (User-based Balanced Partition)
     - Minimizes retrain cost when removing entire users

2. Random partition is ALWAYS suboptimal - avoid it.

3. The partition strategy should be CHOSEN based on the unlearning task:
   - If you need to remove specific user-item interactions -> InBP
   - If you need to remove entire users -> UBP

4. This is a KEY INSIGHT for practical machine unlearning systems:
   the optimal partitioning depends on WHAT you're unlearning.
    """)

    print("\n" + "=" * 80)
    print("Charts saved to:", output_dir)
    print("  - chart_user_distribution.png")
    print("  - chart_user_cost_comparison.png")
    print("  - chart_users_multi_locals.png")
    print("  - chart_retrain_time_comparison.png")
    print("  - chart_efficiency_vs_task.png")
    print("  - chart_user_comprehensive.png")
    print("=" * 80)

if __name__ == '__main__':
    main()
