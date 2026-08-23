"""
Summarize all unlearning results into a table.
"""
import os
import json
import glob
import re

def summarize_results(results_dir='results'):
    # Try multiple possible locations
    possible_dirs = [
        results_dir,
        os.path.join('..', results_dir),
        os.path.join(os.path.dirname(__file__), '..', results_dir),
    ]

    results_dir = None
    for d in possible_dirs:
        if os.path.exists(d) and os.path.isdir(d):
            results_dir = d
            break

    if results_dir is None:
        print("ERROR: results directory not found")
        return

    files = glob.glob(os.path.join(results_dir, 'online_unlearn_num10*.json'))

    if not files:
        print(f"No results found in {results_dir}")
        return

    print(f"Found {len(files)} result files in {results_dir}")

    results = []
    for f in sorted(files):
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)

            for key, val in data.items():
                # Parse key formats:
                # num10_InP-attention_interaction_r05
                # num10_IBP-attention_interaction_r05
                # num10_UBP-attention_interaction_r05
                # num10_Random-attention_interaction_r05

                # Extract parts using regex
                match = re.match(r'num(\d+)_(.+?)-(.+?)_(.+?)_r(\d+)', key)
                if not match:
                    # Try alternative format
                    parts = key.split('_')
                    if len(parts) >= 5:
                        partition = parts[1].split('-')[0]  # InP-attention -> InP
                        unlearn_type = parts[3]
                        ratio_str = parts[4]
                    else:
                        print(f"Skipping malformed key: {key}")
                        continue
                else:
                    partition = match.group(2)  # InP, IBP, UBP, Random
                    agg = match.group(3)  # attention, mean
                    unlearn_type = match.group(4)
                    ratio_str = 'r' + match.group(5)

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
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not results:
        print("No valid results found")
        return

    # Sort by partition, unlearn_type, ratio
    results.sort(key=lambda x: (x['partition'], x['unlearn_type'], x['ratio']))

    # Print table
    print("=" * 130)
    print(f"{'Partition':<10} {'Unlearn Type':<15} {'Ratio':<8} {'Baseline':<12} {'After':<12} {'Change%':<12} {'Time(s)':<10} {'Shards':<8}")
    print("=" * 130)

    for r in results:
        status = "OK" if abs(r['change_pct']) < 5 else "BAD"
        print(f"{r['partition']:<10} {r['unlearn_type']:<15} {r['ratio']:>4}%   "
              f"{r['baseline']:<12.4f} {r['after']:<12.4f} {r['change_pct']:>+10.1f}%  {status}  "
              f"{r['retrain_time']:<10.2f} {r['n_shards']:<8}")

    print("=" * 130)

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

    # Summary by partition
    print("\n" + "=" * 60)
    print("SUMMARY BY PARTITION (avg change %)")
    print("=" * 60)

    summary_partition = {}
    for r in results:
        pt = r['partition']
        if pt not in summary_partition:
            summary_partition[pt] = []
        summary_partition[pt].append(r['change_pct'])

    for pt, changes in sorted(summary_partition.items()):
        avg = sum(changes) / len(changes)
        status = "[GOOD]" if avg > -5 else "[BAD]"
        print(f"{pt:<10}: {avg:>+8.2f}% avg change  {status}")

    print("=" * 60)
    print(f"\nTotal results: {len(results)}")

if __name__ == '__main__':
    summarize_results()
