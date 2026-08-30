"""
Summarize all random unlearning results into a table.
"""
import os
import json
import glob
import numpy as np

def summarize_random_results(results_dir='results'):
    files = glob.glob(os.path.join(results_dir, 'random_*.json'))

    if not files:
        print("No random unlearn results found!")
        return

    print(f"Found {len(files)} result files\n")

    results = {
        'interaction': {},
        'item': {},
        'user': {}
    }

    for f in sorted(files):
        with open(f, 'r') as fp:
            data = json.load(fp)

        # Parse filename
        filename = os.path.basename(f)
        # Format: random_{type}_p10_t{part}_{agg}_r{ratio}_runs{n}.json

        parts = filename.replace('.json', '').split('_')
        unlearn_type = parts[1]  # interaction, item, user
        part_num = int(parts[2].replace('p', ''))  # 10
        part_type = parts[3].replace('t', '')  # 1, 2, 3, 4

        partition_map = {'1': 'InP', '2': 'UBP', '3': 'Random', '4': 'IBP'}
        partition_name = partition_map.get(part_type, part_type)

        if unlearn_type in results:
            key = f"{partition_name}"
            if key not in results[unlearn_type]:
                results[unlearn_type][key] = []

            # Extract average data
            avg = data.get('average', {})
            results[unlearn_type][key].append({
                'baseline_mean': avg.get('baseline_recall20_mean', 0),
                'baseline_std': avg.get('baseline_recall20_std', 0),
                'after_mean': avg.get('after_recall20_mean', 0),
                'after_std': avg.get('after_recall20_std', 0),
                'change_mean': avg.get('change_pct_mean', 0),
                'change_std': avg.get('change_pct_std', 0),
                'n_runs': data.get('config', {}).get('n_runs', 0)
            })

    # Print summary tables
    print("=" * 100)
    print("RANDOM UNLEARN RESULTS SUMMARY")
    print("=" * 100)

    partitions = ['InP', 'UBP', 'Random', 'IBP']

    for unlearn_type in ['interaction', 'item', 'user']:
        print(f"\n{'='*60}")
        print(f"UNLEARN TYPE: {unlearn_type.upper()}")
        print(f"{'='*60}")
        print(f"{'Partition':<12} {'Baseline':<15} {'After':<15} {'Change%':<15} {'Runs':<8}")
        print("-" * 60)

        for partition in partitions:
            if partition in results[unlearn_type]:
                runs = results[unlearn_type][partition]
                if runs:
                    # Average across all runs
                    avg_baseline = np.mean([r['baseline_mean'] for r in runs])
                    avg_after = np.mean([r['after_mean'] for r in runs])
                    avg_change = np.mean([r['change_mean'] for r in runs])
                    n_runs = runs[0]['n_runs']

                    status = "OK" if abs(avg_change) < 5 else "BAD"
                    print(f"{partition:<12} {avg_baseline:<15.4f} {avg_after:<15.4f} {avg_change:>+12.2f}%  [{status}]  n={n_runs}")
            else:
                print(f"{partition:<12} {'N/A':<15}")

        # Summary by partition type
        print("\n" + "-" * 60)
        for partition in partitions:
            if partition in results[unlearn_type]:
                runs = results[unlearn_type][partition]
                if runs:
                    avg_change = np.mean([r['change_mean'] for r in runs])
                    avg_baseline = np.mean([r['baseline_mean'] for r in runs])
                    print(f"  {partition}: Baseline={avg_baseline:.4f}, Change={avg_change:+.2f}%")

    # Overall summary
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY BY UNLEARN TYPE")
    print("=" * 100)

    print(f"\n{'Unlearn Type':<15} {'Best Partition':<15} {'Best Baseline':<15} {'Avg Change':<15}")
    print("-" * 60)

    for unlearn_type in ['interaction', 'item', 'user']:
        best_partition = None
        best_change = float('inf')
        best_baseline = 0

        for partition in partitions:
            if partition in results[unlearn_type]:
                runs = results[unlearn_type][partition]
                if runs:
                    avg_change = np.mean([r['change_mean'] for r in runs])
                    avg_baseline = np.mean([r['baseline_mean'] for r in runs])

                    if abs(avg_change) < abs(best_change):
                        best_change = avg_change
                        best_partition = partition
                        best_baseline = avg_baseline

        if best_partition:
            print(f"{unlearn_type:<15} {best_partition:<15} {best_baseline:<15.4f} {best_change:>+14.2f}%")

    print("\n" + "=" * 100)
    print("DETAILED RESULTS BY PARTITION")
    print("=" * 100)

    for partition in partitions:
        print(f"\n{'='*40}")
        print(f"PARTITION: {partition}")
        print(f"{'='*40}")

        for unlearn_type in ['interaction', 'item', 'user']:
            if partition in results[unlearn_type]:
                runs = results[unlearn_type][partition]
                if runs:
                    avg_baseline = np.mean([r['baseline_mean'] for r in runs])
                    avg_after = np.mean([r['after_mean'] for r in runs])
                    avg_change = np.mean([r['change_mean'] for r in runs])
                    std_change = np.mean([r['change_std'] for r in runs])

                    status = "OK" if abs(avg_change) < 5 else "BAD"
                    print(f"  {unlearn_type:<15}: {avg_baseline:.4f} -> {avg_after:.4f} ({avg_change:+.2f}% +/- {std_change:.2f}%) [{status}]")


if __name__ == '__main__':
    summarize_random_results()
