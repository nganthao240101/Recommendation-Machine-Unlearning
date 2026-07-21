#!/usr/bin/env python3
"""
Run all experiments and print comparison at the end.
"""
import subprocess
import re

def run(part_type, agg_type, epochs=30):
    print(f"\n{'='*70}")
    print(f">>> Part {part_type} - {agg_type.upper()} ({epochs} epochs)")
    print('='*70)

    result = subprocess.run([
        'python', 'code/RecEraser_BPR.py',
        '--dataset', 'ml-1m', '--part_type', str(part_type),
        '--part_num', '10', '--agg_type', agg_type,
        '--epoch', str(epochs), '--data_path', './data/', '--save_flag', '1'
    ], capture_output=True, text=True)

    output = result.stdout + result.stderr

    # Find final recall
    lines = output.strip().split('\n')
    final_line = ""
    for line in reversed(lines):
        if 'recall=[' in line:
            final_line = line
            break

    print(final_line)

    # Extract metrics
    match = re.search(r'recall=\[([\d.]+),\s*([\d.]+),\s*([\d.]+)\]', final_line)
    ndcg_match = re.search(r'ndcg=\[([\d.]+),\s*([\d.]+),\s*([\d.]+)\]', final_line)

    if match and ndcg_match:
        return {
            'r10': float(match.group(1)),
            'r20': float(match.group(2)),
            'r50': float(match.group(3)),
            'n20': float(ndcg_match.group(2))
        }
    return None

def main():
    print("\n" + "="*70)
    print(" ATTENTION vs MEAN COMPARISON (30 epochs)")
    print("="*70)

    partitions = [(1,'InP'), (2,'UBP'), (3,'Random'), (4,'IBP')]
    results = {}

    for part_type, part_name in partitions:
        print(f"\n{'='*70}")
        print(f" PARTITION: {part_name}")
        print('='*70)

        # Attention
        results[(part_name,'attn')] = run(part_type, 'attention', 30)

        # MEAN
        results[(part_name,'mean')] = run(part_type, 'mean', 30)

    # Comparison Table
    print("\n" + "="*70)
    print(" COMPARISON TABLE")
    print("="*70)
    print(f"{'Part':<10} {'Attn R@20':<12} {'Mean R@20':<12} {'Attn N@20':<12} {'Mean N@20':<12} {'Winner':<10}")
    print("-"*70)

    attn_wins = 0
    mean_wins = 0

    for part_name, _ in partitions:
        attn = results.get((part_name,'attn'))
        mean = results.get((part_name,'mean'))

        if attn and mean:
            a_r20 = attn['r20']
            m_r20 = mean['r20']
            a_n20 = attn['n20']
            m_n20 = mean['n20']

            if a_r20 > m_r20:
                winner = "ATTENTION"
                attn_wins += 1
            elif m_r20 > a_r20:
                winner = "MEAN"
                mean_wins += 1
            else:
                winner = "TIE"

            print(f"{part_name:<10} {a_r20:<12.4f} {m_r20:<12.4f} {a_n20:<12.4f} {m_n20:<12.4f} {winner:<10}")
        else:
            print(f"{part_name:<10} N/A")

    print("-"*70)
    print(f"SCORE: Attention {attn_wins} - {mean_wins} MEAN")

    if attn_wins > mean_wins:
        print("=> CONCLUSION: ATTENTION is BETTER")
    elif mean_wins > attn_wins:
        print("=> CONCLUSION: MEAN is BETTER")
    else:
        print("=> CONCLUSION: EQUAL")

    print("="*70)
    print(" All weights saved!")
    print("="*70)

if __name__ == '__main__':
    main()
