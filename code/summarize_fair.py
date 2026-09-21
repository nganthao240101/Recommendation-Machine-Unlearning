"""
Summarize fair benchmark results
"""

import json
import os

files = {
    'FullRetrain': 'results_full_retrain_bprmf_fair150.json',
    'SISA': 'results_sisa_bprmf_fair150.json',
    'RecEraser': 'results_receraser_attention_fair150.json',
    'Ours': 'results_ours_3components_fair150.json'
}

print()
print('='*120)
print('FAIR BENCHMARK RESULTS (150 EPOCHS, 50% UNLEARN)')
print('='*120)
print()
print(f"{'Method':<20} {'Before R@10':<15} {'After R@10':<15} {'Change':<15} {'Train(s)':<12} {'Unlearn(s)':<12}")
print('-'*120)

results = {}
for name, f in files.items():
    try:
        if os.path.exists(f):
            with open(f) as fp:
                d = json.load(fp)
                before = d.get('before', {}).get('recall@10', 0)
                after = d.get('after', {}).get('recall@10', 0)
                change = after - before
                train_t = d.get('train_time', 0)
                unlearn_t = d.get('unlearn_time', 0)

                change_str = f'{change:+.4f}'
                if change > 0.005:
                    change_str += ' ↑↑'
                elif change > 0:
                    change_str += ' ↑'
                elif change < -0.01:
                    change_str += ' ↓↓'
                elif change < 0:
                    change_str += ' ↓'
                else:
                    change_str += ' ≈'

                print(f"{name:<20} {before:<15.4f} {after:<15.4f} {change_str:<15} {train_t:<12.1f} {unlearn_t:<12.1f}")

                results[name] = {
                    'before': before,
                    'after': after,
                    'change': change,
                    'train_time': train_t,
                    'unlearn_time': unlearn_t
                }
        else:
            print(f"{name:<20} Chua co ket qua")
    except Exception as e:
        print(f"{name:<20} Loi: {e}")

print('='*120)
print()
print('CHI TIET SAU UNLEARN:')
print('-'*120)
print(f"{'Method':<20} {'R@10':<10} {'R@20':<10} {'R@50':<10} {'N@10':<10} {'N@20':<10} {'N@50':<10}")
print('-'*120)

for name, f in files.items():
    try:
        if os.path.exists(f):
            with open(f) as fp:
                d = json.load(fp)
                after = d.get('after', {})
                print(f"{name:<20} {after.get('recall@10',0):.4f}     {after.get('recall@20',0):.4f}     {after.get('recall@50',0):.4f}     {after.get('ndcg@10',0):.4f}     {after.get('ndcg@20',0):.4f}     {after.get('ndcg@50',0):.4f}")
    except:
        pass

print('='*120)
print()
print('PHAN TICH:')
print('-'*120)
print()
print("1. PERFORMANCE TRUOC UNLEARN (Before R@10):")
for name, r in results.items():
    print(f"   {name}: {r['before']:.4f}")

print()
print("2. PERFORMANCE SAU UNLEARN (After R@10):")
for name, r in results.items():
    print(f"   {name}: {r['after']:.4f}")

print()
print("3. SU THAY DOI (Change):")
for name, r in results.items():
    print(f"   {name}: {r['change']:+.4f}")

print()
print("4. THOI GIAN:")
for name, r in results.items():
    print(f"   {name}: Train={r['train_time']:.1f}s, Unlearn={r['unlearn_time']:.1f}s")

print()
print('='*120)
print()
print('KET LUAN:')
print('-'*120)

# Find best and worst
if results:
    best_before = max(results.items(), key=lambda x: x[1]['before'])
    best_after = max(results.items(), key=lambda x: x[1]['after'])
    best_retention = max(results.items(), key=lambda x: x[1]['change'])

    print(f"  - Best Before: {best_before[0]} ({best_before[1]['before']:.4f})")
    print(f"  - Best After: {best_after[0]} ({best_after[1]['after']:.4f})")
    print(f"  - Best Retention: {best_retention[0]} ({best_retention[1]['change']:+.4f})")

print()
print('='*120)
