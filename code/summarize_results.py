"""
Summarize all unlearning results into a table.
"""
import os
import json
import glob
import re

def summarize_results(results_dir='results'):
    possible_dirs = [
        results_dir,
        os.path.join('..', results_dir),
        os.path.join(os.path.dirname(__file__), '..', results_dir),
    ]

    results_dir_path = None
    for d in possible_dirs:
        if os.path.exists(d) and os.path.isdir(d):
            results_dir_path = d
            break

    if results_dir_path is None:
        print("ERROR: results directory not found")
        return

    files = glob.glob(os.path.join(results_dir_path, 'online_unlearn*.json'))
    files = list(set(files))

    if not files:
        print("No results found in", results_dir_path)
        return

    print("Found", len(files), "result files in", results_dir_path)

    results = []
    for f in sorted(files):
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)

            for key, val in data.items():
                partition = val.get('partition', 'Unknown')
                unlearn_type = val.get('unlearn_type', 'unknown')
                ratio = 5

                ratio_match = re.search(r'r(\d+)', key)
                if ratio_match:
                    ratio = int(ratio_match.group(1))

                baseline_recall = val['baseline']['recall20']
                after_recall = val['online_unlearn']['recall20']
                change = (after_recall - baseline_recall) / baseline_recall * 100
                retrain_time = val['retrain_time_s']
                n_shards = val['n_affected_shards']

                results.append({
                    'partition': partition,
                    'unlearn_type': unlearn_type,
                    'ratio': ratio,
                    'baseline': baseline_recall,
                    'after': after_recall,
                    'change_pct': change,
                    'retrain_time': retrain_time,
                    'n_shards': n_shards
                })
        except Exception as e:
            print("Error reading", f, ":", e)

    if not results:
        print("No valid results found")
        return

    results.sort(key=lambda x: (x['partition'], x['unlearn_type'], x['ratio']))

    print("=" * 130)
    print("{:<10} {:<15} {:<8} {:<12} {:<12} {:<12} {:<10} {:<8}".format(
        'Partition', 'Unlearn Type', 'Ratio', 'Baseline', 'After', 'Change%', 'Time(s)', 'Shards'))
    print("=" * 130)

    for r in results:
        status = "OK" if abs(r['change_pct']) < 5 else "BAD"
        print("{:<10} {:<15} {:>4}%   {:<12.4f} {:<12.4f} {:>+10.1f}%  {}  {:<10.2f} {:<8}".format(
            r['partition'], r['unlearn_type'], r['ratio'],
            r['baseline'], r['after'], r['change_pct'], status,
            r['retrain_time'], r['n_shards']))

    print("=" * 130)

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
        print("{:<20}: {:>+8.2f}% avg change  {}".format(ut, avg, status))

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
        print("{:<10}: {:>+8.2f}% avg change  {}".format(pt, avg, status))

    print("=" * 60)
    print("\nTotal results:", len(results))

if __name__ == '__main__':
    summarize_results()
