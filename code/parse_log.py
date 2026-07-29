#!/usr/bin/env python3
"""Parse results log and print comparison table."""
import re
import sys
import os
from pathlib import Path

# Find latest log file
logs = sorted(Path('.').glob('results_pytorch_e*.log'), key=os.path.getmtime, reverse=True)
if not logs:
    print("No log file found")
    sys.exit(1)

log_file = logs[0]
print(f"Reading: {log_file}\n")

text = log_file.read_text()

# Parse all [AGGREGATION FINAL] blocks
# Format:
#   =============================================
#   [AGGREGATION FINAL] attention
#     recall@10: 0.1234
#     recall@20: 0.2345
#     recall@50: 0.3456
#     ndcg@10:   0.0987
#     ndcg@20:   0.1543
#     ndcg@50:   0.2345
#   =============================================

pattern = re.compile(
    r'\[AGGREGATION FINAL\]\s*(\w+)\s*\n'
    r'\s*recall@10:\s*([\d.]+)\s*\n'
    r'\s*recall@20:\s*([\d.]+)\s*\n'
    r'\s*recall@50:\s*([\d.]+)\s*\n'
    r'\s*ndcg@10:\s*([\d.]+)\s*\n'
    r'\s*ndcg@20:\s*([\d.]+)\s*\n'
    r'\s*ndcg@50:\s*([\d.]+)',
    re.MULTILINE
)

results = []
for m in pattern.finditer(text):
    agg = m.group(1).lower()
    r10, r20, r50 = float(m.group(2)), float(m.group(3)), float(m.group(4))
    n10, n20, n50 = float(m.group(5)), float(m.group(6)), float(m.group(7))
    results.append({'agg': agg, 'r10': r10, 'r20': r20, 'r50': r50,
                    'n10': n10, 'n20': n20, 'n50': n50})

if not results:
    print("No results found in log (need AGGREGATION FINAL blocks)")
    sys.exit(1)

# Group by agg
attn = [r for r in results if r['agg'] == 'attention']
mean = [r for r in results if r['agg'] == 'mean']

PART_NAMES = ['InP', 'UBP', 'Random', 'IBP']

print("=" * 80)
print(f" RECALL@20 COMPARISON")
print("=" * 80)
print(f"{'Partition':<12} {'Attention':<12} {'MEAN':<12} {'Winner':<14} {'Diff %':<10}")
print("-" * 80)
for i, name in enumerate(PART_NAMES):
    if i < len(attn) and i < len(mean):
        a, m = attn[i]['r20'], mean[i]['r20']
        diff = (a - m) / m * 100 if m > 0 else 0
        winner = 'Attention' if a > m else 'MEAN' if m > a else 'Tie'
        print(f"{name:<12} {a:<12.4f} {m:<12.4f} {winner:<14} {diff:+.1f}%")

print()
print("=" * 80)
print(f" NDCG@20 COMPARISON")
print("=" * 80)
print(f"{'Partition':<12} {'Attention':<12} {'MEAN':<12} {'Winner':<14} {'Diff %':<10}")
print("-" * 80)
for i, name in enumerate(PART_NAMES):
    if i < len(attn) and i < len(mean):
        a, m = attn[i]['n20'], mean[i]['n20']
        diff = (a - m) / m * 100 if m > 0 else 0
        winner = 'Attention' if a > m else 'MEAN' if m > a else 'Tie'
        print(f"{name:<12} {a:<12.4f} {m:<12.4f} {winner:<14} {diff:+.1f}%")

print()
print("=" * 80)
print(" FULL METRICS")
print("=" * 80)
for i, name in enumerate(PART_NAMES):
    if i < len(attn):
        a = attn[i]
        print(f"{name} - Attention: R@10={a['r10']:.4f} R@20={a['r20']:.4f} R@50={a['r50']:.4f} "
              f"N@10={a['n10']:.4f} N@20={a['n20']:.4f} N@50={a['n50']:.4f}")
    if i < len(mean):
        m = mean[i]
        print(f"{name} - MEAN:      R@10={m['r10']:.4f} R@20={m['r20']:.4f} R@50={m['r50']:.4f} "
              f"N@10={m['n10']:.4f} N@20={m['n20']:.4f} N@50={m['n50']:.4f}")
    print()

# Score
attn_wins = sum(1 for i in range(min(len(attn), len(mean), len(PART_NAMES)))
                if attn[i]['r20'] > mean[i]['r20'])
mean_wins = sum(1 for i in range(min(len(attn), len(mean), len(PART_NAMES)))
                if mean[i]['r20'] > attn[i]['r20'])
print("=" * 80)
print(f" SCORE (Recall@20): Attention {attn_wins} - {mean_wins} MEAN")
if attn_wins > mean_wins:
    print(" => Attention is BETTER overall")
elif mean_wins > attn_wins:
    print(" => MEAN is BETTER overall")
else:
    print(" => TIE")
print("=" * 80)