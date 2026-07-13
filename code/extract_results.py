"""
Extract results from training logs or re-run evaluation using the same method as training.
"""
import os
import sys
import subprocess
import re

# Run training and extract final results
def train_and_extract(part_type, agg_type, epochs=10):
    cmd = [
        'python', 'code/RecEraser_BPR.py',
        '--dataset', 'ml-1m',
        '--part_type', str(part_type),
        '--part_num', '10',
        '--agg_type', agg_type,
        '--epoch', str(epochs),
        '--data_path', './data/',
        '--save_flag', '0'  # Don't save weights for faster execution
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Extract final recall values
    # Pattern: recall=[0.123, 0.456, 0.789]
    recall_match = re.search(r'recall=\[([\d.]+),\s*([\d.]+),\s*([\d.]+)\]', output)
    ndcg_match = re.search(r'ndcg=\[([\d.]+),\s*([\d.]+),\s*([\d.]+)\]', output)

    if recall_match and ndcg_match:
        return {
            'recall': [float(recall_match.group(i)) for i in range(1, 4)],
            'ndcg': [float(ndcg_match.group(i)) for i in range(1, 4)]
        }
    return None

def main():
    print("=" * 70)
    print("Attention vs MEAN Comparison (Extracted from Training)")
    print("=" * 70)

    partitions = [
        (1, 'InP'),
        (2, 'UBP'),
        (3, 'Random'),
        (4, 'IBP')
    ]

    results = {}

    for part_type, part_name in partitions:
        results[part_name] = {}

        for agg_type in ['attention', 'mean']:
            print(f"\n>>> Training {part_name} {agg_type}...", flush=True)
            result = train_and_extract(part_type, agg_type, epochs=10)

            if result:
                results[part_name][agg_type] = result
                print(f"    Recall@20: {result['recall'][1]:.4f}, NDCG@20: {result['ndcg'][1]:.4f}")
            else:
                results[part_name][agg_type] = None
                print(f"    ERROR: Could not extract results")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (from Training Evaluation)")
    print("=" * 70)
    print(f"{'Partition':<12} {'Method':<12} {'R@10':<10} {'R@20':<10} {'R@50':<10} {'N@10':<10} {'N@20':<10} {'N@50':<10}")
    print("-" * 70)

    for part_name, _ in partitions:
        for agg_type in ['attention', 'mean']:
            r = results[part_name][agg_type]
            if r:
                method = 'Attention' if agg_type == 'attention' else 'MEAN'
                print(f"{part_name:<12} {method:<12} {r['recall'][0]:<10.4f} {r['recall'][1]:<10.4f} {r['recall'][2]:<10.4f} {r['ndcg'][0]:<10.4f} {r['ndcg'][1]:<10.4f} {r['ndcg'][2]:<10.4f}")

    print("-" * 70)
    print("\nWINNER COMPARISON (R@20):")
    print("-" * 70)
    for part_name, _ in partitions:
        attn = results[part_name]['attention']
        mean = results[part_name]['mean']
        if attn and mean:
            attn_r20 = attn['recall'][1]
            mean_r20 = mean['recall'][1]
            if attn_r20 > mean_r20:
                winner = f"Attention (+{((attn_r20-mean_r20)/mean_r20*100):.1f}%)"
            elif mean_r20 > attn_r20:
                winner = f"MEAN (+{((mean_r20-attn_r20)/attn_r20*100):.1f}%)"
            else:
                winner = "Tie"
            print(f"{part_name:<12} Attention={attn_r20:.4f}, MEAN={mean_r20:.4f} => {winner}")
    print("=" * 70)

if __name__ == '__main__':
    main()
