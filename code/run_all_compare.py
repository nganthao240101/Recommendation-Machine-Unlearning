#!/usr/bin/env python3
"""
Run all experiments and print comparison results.
Usage: python run_all_compare.py
"""
import os
import sys
import subprocess
import re

def run_experiment(part_type, agg_type, epochs=30):
    """Run training and extract final results."""
    print(f"\n>>> Training Part {part_type} {agg_type.upper()}...", flush=True)

    cmd = [
        'python', 'code/RecEraser_BPR.py',
        '--dataset', 'ml-1m',
        '--part_type', str(part_type),
        '--part_num', '10',
        '--agg_type', agg_type,
        '--epoch', str(epochs),
        '--data_path', './data/',
        '--save_flag', '1'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Print last line (final result)
    lines = output.strip().split('\n')
    for line in reversed(lines):
        if 'recall=' in line:
            print(f"    Final: {line}")
            break

    # Extract metrics
    recall_match = re.search(r'recall=\[([\d.]+),\s*([\d.]+),\s*([\d.]+)\]', output)
    ndcg_match = re.search(r'ndcg=\[([\d.]+),\s*([\d.]+),\s*([\d.]+)\]', output)

    if recall_match and ndcg_match:
        return {
            'recall': [float(recall_match.group(i)) for i in range(1, 4)],
            'ndcg': [float(ndcg_match.group(i)) for i in range(1, 4)]
        }
    return None

def main():
    print("=" * 80)
    print("TRAINING ALL EXPERIMENTS (30 epochs) - ATTENTION vs MEAN")
    print("=" * 80)

    partitions = [(1, 'InP'), (2, 'UBP'), (3, 'Random'), (4, 'IBP')]
    results = {}

    for part_type, part_name in partitions:
        results[part_name] = {}

        for agg_type in ['attention', 'mean']:
            result = run_experiment(part_type, agg_type, epochs=30)
            results[part_name][agg_type] = result

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE (Recall@20, NDCG@20)")
    print("=" * 80)
    print(f"{'Partition':<12} {'Attn R@20':<12} {'Mean R@20':<12} {'Attn N@20':<12} {'Mean N@20':<12} {'Winner':<12}")
    print("-" * 80)

    for part_name, _ in partitions:
        attn = results[part_name]['attention']
        mean = results[part_name]['mean']

        if attn and mean:
            attn_r20 = attn['recall'][1]
            mean_r20 = mean['recall'][1]
            attn_n20 = attn['ndcg'][1]
            mean_n20 = mean['ndcg'][1]

            if attn_r20 > mean_r20:
                winner = "Attention"
            elif mean_r20 > attn_r20:
                winner = "MEAN"
            else:
                winner = "Tie"

            print(f"{part_name:<12} {attn_r20:<12.4f} {mean_r20:<12.4f} {attn_n20:<12.4f} {mean_n20:<12.4f} {winner:<12}")
        else:
            print(f"{part_name:<12} N/A")

    print("=" * 80)
    print("\nWINNER ANALYSIS (Recall@20):")
    print("-" * 80)

    attn_wins = 0
    mean_wins = 0

    for part_name, _ in partitions:
        attn = results[part_name]['attention']
        mean = results[part_name]['mean']

        if attn and mean:
            attn_r20 = attn['recall'][1]
            mean_r20 = mean['recall'][1]
            diff = attn_r20 - mean_r20
            pct = (diff / mean_r20) * 100 if mean_r20 > 0 else 0

            if diff > 0:
                attn_wins += 1
                print(f"{part_name}: Attention wins by {diff:.4f} ({pct:+.1f}%)")
            elif diff < 0:
                mean_wins += 1
                print(f"{part_name}: MEAN wins by {-diff:.4f} ({-pct:.1f}%)")
            else:
                print(f"{part_name}: Tie")

    print("-" * 80)
    print(f"Total: Attention {attn_wins} - {mean_wins} MEAN")

    if attn_wins > mean_wins:
        print("=> CONCLUSION: Attention is BETTER overall")
    elif mean_wins > attn_wins:
        print("=> CONCLUSION: MEAN is BETTER overall")
    else:
        print("=> CONCLUSION: Equal performance")

    print("=" * 80)
    print("All weights saved in weights/ml-1m/RecEraser_BPR/")
    print("=" * 80)

if __name__ == '__main__':
    main()
