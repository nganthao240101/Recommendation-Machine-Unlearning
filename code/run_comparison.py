"""
Script chạy RecEraser_BPR với 4 phương pháp partition và so sánh kết quả
"""
import subprocess
import re
import os

def run_experiment(part_type, part_name):
    """Chạy experiment với partition type cụ thể"""
    print(f"\n{'='*60}")
    print(f"Running {part_name} (part_type={part_type})")
    print('='*60)

    cmd = [
        'python', 'RecEraser_BPR.py',
        '--dataset', 'ml-1m',
        '--part_type', str(part_type),
        '--part_num', '10',
        '--epoch', '100',
        '--lr', '0.05',
        '--regs', '[0.01]',
        '--batch_size', '256'
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd='e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/code'
    )

    # Extract recall values from output
    output = result.stdout + result.stderr

    # Find all recall values for local models
    recall_pattern = r'\[local_model \d+\] Epoch \d+.*?recall=\[([0-9.]+), ([0-9.]+)\]'
    recalls = re.findall(recall_pattern, output)

    # Get final epoch recall for each local model
    final_recalls = {}
    for i in range(10):
        pattern = rf'\[local_model {i}\] Epoch \d+.*?recall=\[([0-9.]+), ([0-9.]+)\]'
        matches = re.findall(pattern, output)
        if matches:
            # Get the last match (final epoch)
            last_match = matches[-1]
            final_recalls[i] = {'K@10': float(last_match[0]), 'K@20': float(last_match[1])}

    return final_recalls, output

def main():
    results = {}

    # 4 partition methods
    methods = [
        (1, 'InBP (Interaction-based Balanced Partition)'),
        (2, 'UBP (User-based Partition)'),
        (3, 'Random'),
    ]

    for part_type, part_name in methods:
        print(f"\n{'#'*60}")
        print(f"# Running {part_name}")
        print('#'*60)

        recalls, _ = run_experiment(part_type, part_name)
        results[part_name] = recalls

        # Print summary
        if recalls:
            avg_recall_10 = sum(r['K@10'] for r in recalls.values()) / len(recalls)
            avg_recall_20 = sum(r['K@20'] for r in recalls.values()) / len(recalls)
            print(f"\n{part_name} Summary:")
            print(f"  Avg Recall@10: {avg_recall_10:.5f}")
            print(f"  Avg Recall@20: {avg_recall_20:.5f}")

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE - RecEraser BPR on ml-1m")
    print("="*80)
    print(f"{'Method':<45} {'Avg Recall@10':<15} {'Avg Recall@20':<15}")
    print("-"*80)

    for method, recalls in results.items():
        if recalls:
            avg_10 = sum(r['K@10'] for r in recalls.values()) / len(recalls)
            avg_20 = sum(r['K@20'] for r in recalls.values()) / len(recalls)
            print(f"{method:<45} {avg_10:.5f}{'':<10} {avg_20:.5f}")

if __name__ == '__main__':
    main()