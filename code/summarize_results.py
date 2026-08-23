"""
Summarize all unlearning results into a table.
"""
import os
import json
import glob

def summarize_results(results_dir='results'):
    files = glob.glob(os.path.join(results_dir, 'online_unlearn_num10*.json'))

    results = []
    for f in sorted(files):
        with open(f, 'r') as fp:
            data = json.load(fp)

        for key, val in data.items():
            # Parse key: num10_InP-attention_interaction_r05
            parts = key.split('_')
            partition = parts[1]  # InP
            unlearn_type = parts[3]  # interaction
            ratio_str = parts[4]  # r05

            baseline_recall = val['baseline']['recall20']
            after_recall = val['online_unlearn']['recall20']
            change = (after_recall - baseline_recall) / baseline_recall * 100
            retrain_time = val['retrain_time_s']
            n_shards = val['n_affected_shards']

            results.append({
                'partition': partition,
                'unlearn_type': unlearn_type,
                'ratio': int(ratio_str[1:]),  # r05 -> 5
                'baseline': baseline_recall,
                'after': after_recall,
                'change_pct': change,
                'retrain_time': retrain_time,
                'n_shards': n_shards
            })

    # Sort by partition, unlearn_type, ratio
    results.sort(key=lambda x: (x['partition'], x['unlearn_type'], x['ratio']))

    # Print table
    print("=" * 120)
    print(f"{'Partition':<10} {'Unlearn Type':<15} {'Ratio':<8} {'Baseline':<12} {'After':<12} {'Change%':<10} {'Time(s)':<10} {'Shards':<8}")
    print("=" * 120)

    for r in results:
        status = "OK" if abs(r['change_pct']) < 5 else "BAD"
        print(f"{r['partition']:<10} {r['unlearn_type']:<15} {r['ratio']:>4}%   "
              f"{r['baseline']:<12.4f} {r['after']:<12.4f} {r['change_pct']:>+8.1f}% [{status}]  "
              f"{r['retrain_time']:<10.2f} {r['n_shards']:<8}")

    print("=" * 120)

    # Summary by unlearn type
    print("\n" + "=" * 60)
    print("SUMMARY BY UNLEARN TYPE (avg change %)")
    print("=" * 60)

    summary = {}
    for r in results:
        ut = r['unlearn_type']
        if ut not in summary:
            summary[ut] = []
        summary[ut].append(r['change_pct'])

    for ut, changes in sorted(summary.items()):
        avg = sum(changes) / len(changes)
        status = "[GOOD]" if avg > -5 else "[BAD]"
        print(f"{ut:<20}: {avg:>+8.2f}% avg change  {status}")

    print("=" * 60)
    print(f"\nTotal results: {len(results)}")

if __name__ == '__main__':
    summarize_results()
