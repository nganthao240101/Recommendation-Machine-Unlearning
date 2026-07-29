#!/usr/bin/env python3
"""
Run Attention and MEAN for one partition and show comparison.
"""
import subprocess
import re
import sys

def run(agg_type, part_type, epochs=5):
    print(f"\n>>> Running {agg_type.upper()} for partition {part_type}...", flush=True)
    cmd = [
        'python', 'code/RecEraser_BPR_pytorch.py',
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

    # Print tail of output
    lines = output.strip().split('\n')
    for line in lines[-10:]:
        print(line)

    # Extract metrics from final aggregation block
    recall = re.search(r'recall@10:\s*([\d.]+)\s*\n\s*recall@20:\s*([\d.]+)\s*\n\s*recall@50:\s*([\d.]+)', output)
    ndcg = re.search(r'ndcg@10:\s*([\d.]+)\s*\n\s*ndcg@20:\s*([\d.]+)\s*\n\s*ndcg@50:\s*([\d.]+)', output)

    if recall and ndcg:
        return {
            'r10': float(recall.group(1)),
            'r20': float(recall.group(2)),
            'r50': float(recall.group(3)),
            'n10': float(ndcg.group(1)),
            'n20': float(ndcg.group(2)),
            'n50': float(ndcg.group(3)),
        }
    print("WARNING: could not parse results", flush=True)
    return None


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    part_type = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    print("=" * 70)
    print(f" COMPARISON: Attention vs MEAN (partition {part_type}, {epochs} epochs)")
    print("=" * 70)

    attn = run('attention', part_type, epochs)
    mean = run('mean', part_type, epochs)

    if not attn or not mean:
        print("Failed to extract results")
        return

    print("\n" + "=" * 70)
    print(" COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Metric':<12} {'Attention':<12} {'MEAN':<12} {'Winner':<12}")
    print("-" * 70)
    for key, label in [('r10','Recall@10'),('r20','Recall@20'),('r50','Recall@50'),
                       ('n10','NDCG@10'),('n20','NDCG@20'),('n50','NDCG@50')]:
        a, m = attn[key], mean[key]
        winner = 'Attention' if a > m else 'MEAN' if m > a else 'Tie'
        diff = (a - m) / m * 100 if m > 0 else 0
        print(f"{label:<12} {a:<12.4f} {m:<12.4f} {winner} ({diff:+.1f}%)")
    print("=" * 70)


if __name__ == '__main__':
    main()
