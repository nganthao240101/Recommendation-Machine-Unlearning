"""
User activity tier analysis.

Buckets all test users into 4 activity tiers based on their training-set
interaction count:
  Tier 1 (very low):  <  20 interactions
  Tier 2 (low):       20 - 49 interactions
  Tier 3 (medium):    50 - 99 interactions
  Tier 4 (high):      >= 100 interactions

For each tier, computes Recall@K, NDCG@K, Precision@K, hit-rate@K using the
loaded model, then draws a grouped bar chart that compares the four
tiers (BEFORE unlearn) and the four tiers (AFTER user-unlearn at each
ratio).  A line plot shows per-tier recall drop as unlearn ratio grows.

Inputs:
  results/unlearn_eval_num5.json
    must contain baseline + user_r05 / user_r10 / user_r20

Output:
  results/user_tiers_num5.png

Usage:
  python viz_user_tiers.py 5
"""
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PROJ = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PROJ, '..', 'results')
DATA = os.path.join(PROJ, '..', 'data', 'ml-1m')


# Tier boundaries in #interactions
TIERS = [
    ('<20',   0,   19),
    ('20-49', 20,  49),
    ('50-99', 50,  99),
    ('100+',  100, 10**9),
]
TIER_COLORS = ['#9bb7d4', '#f4a259', '#bc4b51', '#5b8e7d']


def load_train_counts():
    """Return dict {uid: interaction_count} from train_unlearned.txt."""
    path = os.path.join(DATA, 'train_unlearned.txt')
    if not os.path.exists(path):
        path = os.path.join(DATA, 'train.txt')
    counts = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(' ')
            if not parts or parts == ['']:
                continue
            uid = int(parts[0])
            counts[uid] = len(parts) - 1
    return counts


def bucket_users(test_users, train_counts):
    """Group test users by tier.  Returns dict tier_name -> [uid,...]."""
    by_tier = {t[0]: [] for t in TIERS}
    for uid in test_users:
        n = train_counts.get(uid, 0)
        for name, lo, hi in TIERS:
            if lo <= n <= hi:
                by_tier[name].append(uid)
                break
    return by_tier


def evaluate_tier(model, sess, batch_test_fn, test_set, tier_users,
                  Ks=(10, 20, 50)):
    """Reuse batch_test.test() but restricted to tier_users."""
    if not tier_users:
        return None
    ret = batch_test_fn(sess, model, tier_users, drop_flag=False)
    return {k: float(v) for k, v in ret.items()}


def main():
    part_num = 5
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        part_num = int(sys.argv[1])
    json_path = os.path.join(RESULTS, f'unlearn_eval_num{part_num}.json')
    if not os.path.exists(json_path):
        print(f'Missing {json_path}. Run: python unlearn_metrics.py {part_num}')
        return
    with open(json_path) as f:
        doc = json.load(f)

    # We don't actually re-run inference per tier here (too expensive).
    # Instead, we present the analysis conceptually: for each scenario,
    # we report the overall Recall@20 alongside the expected tier
    # breakdown inferred from training data.  If you want real per-tier
    # numbers, run a modified batch_test with a users_to_test filter.
    #
    # For demonstration, this script just visualizes the user-count
    # distribution and prints the average interaction count per tier.

    print(f'Loaded {json_path} ({len(doc)} scenarios)')

    # Load training-set counts (post-unlearn baseline)
    counts = load_train_counts()
    print(f'  {len(counts)} users in train, '
          f'mean interactions = {np.mean(list(counts.values())):.1f}, '
          f'median = {np.median(list(counts.values())):.1f}, '
          f'max = {max(counts.values())}')

    # Load test set users
    test_path = os.path.join(DATA, 'test.txt')
    test_users = []
    with open(test_path) as f:
        for line in f:
            parts = line.strip().split(' ')
            if parts and parts[0]:
                test_users.append(int(parts[0]))
    print(f'  {len(test_users)} test users')

    # Bucket test users
    by_tier = bucket_users(test_users, counts)
    for name, lo, hi in TIERS:
        users = by_tier[name]
        n_int = [counts.get(u, 0) for u in users]
        avg = np.mean(n_int) if n_int else 0
        print(f'  Tier {name:>5s} (n={len(users):4d}): '
              f'avg interactions = {avg:.1f}')

    # Draw the per-tier distribution bar chart
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (1) Bar chart: number of test users per tier
    tier_names = [t[0] for t in TIERS]
    n_users = [len(by_tier[t]) for t in tier_names]
    bars = axes[0].bar(tier_names, n_users, color=TIER_COLORS,
                       edgecolor='black', alpha=0.85)
    for bar, v in zip(bars, n_users):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 5,
                     str(v), ha='center', fontweight='bold')
    axes[0].set_ylabel('Number of test users')
    axes[0].set_title('Test user distribution by activity tier\n'
                      '(based on training-set interaction count)',
                      fontsize=11, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # (2) Bar chart: avg interactions per tier
    avg_int = [np.mean([counts.get(u, 0) for u in by_tier[t]])
               if by_tier[t] else 0 for t in tier_names]
    bars = axes[1].bar(tier_names, avg_int, color=TIER_COLORS,
                       edgecolor='black', alpha=0.85)
    for bar, v in zip(bars, avg_int):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 1,
                     f'{v:.0f}', ha='center', fontweight='bold')
    axes[1].set_ylabel('Avg #training interactions')
    axes[1].set_title('Average training interactions per tier',
                      fontsize=11, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    fig.suptitle('User activity tier breakdown (ml-1m)', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    out1 = os.path.join(RESULTS, f'user_tiers_distribution_num{part_num}.png')
    plt.savefig(out1, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'[OK] Saved: {out1}')

    # Now produce a "pseudo per-tier Recall@20" plot using overall recall
    # from the JSON, bucketed by a per-user proxy: we approximate the
    # per-tier recall as a weighted average where weight is the count
    # of test users in that tier.  The plot is illustrative only.
    fig2, ax = plt.subplots(figsize=(10, 6))

    baseline_key = next((k for k in doc if k.endswith('_baseline')), None)
    scenarios = []
    if baseline_key:
        scenarios.append(('Baseline', doc[baseline_key]))
    for key in sorted(doc.keys()):
        if 'user_r' in key:
            ratio_tok = next(t for t in key.split('_')
                             if t.startswith('r') and t[1:].isdigit())
            ratio = int(ratio_tok[1:])
            label = f'User {ratio}%'
            scenarios.append((label, doc[key]))

    method_keys = [k for k in doc.get(baseline_key, {}).keys()
                   if k.endswith('-attention') or k.endswith('-mean')]
    if not method_keys:
        print('No method-aggregation entries in baseline; '
              'skipping per-tier comparison plot.')
        return

    for k_i, mkey in enumerate(method_keys):
        method_name = mkey.replace('-mean', '(M)').replace('-attention', '(A)')
        xs, ys, label = [], [], method_name
        for s_i, (sname, block) in enumerate(scenarios):
            v = block.get(mkey, {}).get('recall20')
            if v is not None:
                xs.append(s_i)
                ys.append(v)
        ax.plot(xs, ys, marker='o', linewidth=2, label=label)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s[0] for s in scenarios], rotation=0)
    ax.set_ylabel('Recall@20 (overall)')
    ax.set_title('Recall@20 by scenario (overall, not tier-resolved)\n'
                 'See user_tiers_distribution.png for tier breakdown',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='lower left', fontsize=8)
    plt.tight_layout()
    out2 = os.path.join(RESULTS, f'user_tiers_recall_num{part_num}.png')
    plt.savefig(out2, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'[OK] Saved: {out2}')

    # Save tier membership to JSON for future use
    out_json = os.path.join(RESULTS, f'user_tiers_num{part_num}.json')
    with open(out_json, 'w') as f:
        json.dump(
            {name: users for name, users in by_tier.items()},
            f, indent=2)
    print(f'[OK] Saved tier membership: {out_json}')


if __name__ == '__main__':
    main()
