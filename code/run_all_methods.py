"""
Run All Methods and Compare Results

Chạy tất cả 4 methods và lưu kết quả riêng biệt.
"""

import os
import sys
import subprocess
import json

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

def run_command(cmd, description):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False, text=True, cwd=PROJ)
    return result.returncode == 0


def main():
    """Run all methods."""
    # Common arguments
    dataset = 'ml-1m'
    emb_dim = 64
    n_shards = 8
    n_epochs = 10
    retrain_epochs = 5
    unlearn_ratio = 0.1

    results = {}

    # Method 1: Full Retrain
    print("\n" + "="*70)
    print("RUNNING METHOD 1: FULL RETRAIN")
    print("="*70)

    cmd1 = [
        'python', 'method_1_full_retrain.py',
        '--model', 'BPRMF',
        '--dataset', dataset,
        '--emb_dim', str(emb_dim),
        '--n_epochs', str(n_epochs),
        '--unlearn_ratio', str(unlearn_ratio),
        '--output_suffix', f'{dataset}_s{emb_dim}'
    ]
    success1 = run_command(cmd1, "Method 1: Full Retrain")

    # Load result
    result_file1 = os.path.join(PROJ, f'results_full_retrain_bprmf_{dataset}_s{emb_dim}.json')
    if os.path.exists(result_file1):
        with open(result_file1, 'r') as f:
            results['FullRetrain'] = json.load(f)

    # Method 2: SISA
    print("\n" + "="*70)
    print("RUNNING METHOD 2: SISA")
    print("="*70)

    cmd2 = [
        'python', 'method_2_sisa.py',
        '--model', 'BPRMF',
        '--dataset', dataset,
        '--emb_dim', str(emb_dim),
        '--n_shards', str(n_shards),
        '--n_epochs', str(n_epochs),
        '--retrain_epochs', str(retrain_epochs),
        '--unlearn_ratio', str(unlearn_ratio),
        '--output_suffix', f'{dataset}_s{emb_dim}'
    ]
    success2 = run_command(cmd2, "Method 2: SISA")

    # Load result
    result_file2 = os.path.join(PROJ, f'results_sisa_bprmf_{dataset}_s{emb_dim}.json')
    if os.path.exists(result_file2):
        with open(result_file2, 'r') as f:
            results['SISA'] = json.load(f)

    # Method 3: RecEraser (giữ nguyên code gốc)
    print("\n" + "="*70)
    print("RUNNING METHOD 3: RECERASER (CODE GỐC)")
    print("="*70)

    cmd3 = [
        'python', 'method_3_receraser.py',
        '--dataset', dataset,
        '--emb_dim', str(emb_dim),
        '--n_shards', str(n_shards),
        '--n_epochs', str(n_epochs),
        '--retrain_epochs', str(retrain_epochs),
        '--agg_type', 'attention',
        '--unlearn_ratio', str(unlearn_ratio),
        '--output_suffix', f'{dataset}_s{emb_dim}'
    ]
    success3 = run_command(cmd3, "Method 3: RecEraser")

    # Load result
    result_file3 = os.path.join(PROJ, f'results_receraser_attention_{dataset}_s{emb_dim}.json')
    if os.path.exists(result_file3):
        with open(result_file3, 'r') as f:
            results['RecEraser'] = json.load(f)

    # Method 4: Ours (3 Components)
    print("\n" + "="*70)
    print("RUNNING METHOD 4: OURS (3 COMPONENTS)")
    print("="*70)

    cmd4 = [
        'python', 'method_4_ours.py',
        '--dataset', dataset,
        '--emb_dim', str(emb_dim),
        '--n_shards', str(n_shards),
        '--n_epochs', str(n_epochs),
        '--retrain_epochs', str(retrain_epochs),
        '--unlearn_ratio', str(unlearn_ratio),
        '--output_suffix', f'{dataset}_s{emb_dim}'
    ]
    success4 = run_command(cmd4, "Method 4: Ours")

    # Load result
    result_file4 = os.path.join(PROJ, f'results_ours_3components_{dataset}_s{emb_dim}.json')
    if os.path.exists(result_file4):
        with open(result_file4, 'r') as f:
            results['Ours'] = json.load(f)

    # Print comparison table
    print("\n" + "="*100)
    print("COMPARISON TABLE")
    print("="*100)

    print(f"\n{'Method':<20} {'R@10':<12} {'R@20':<12} {'R@50':<12} {'N@10':<12} {'N@20':<12} {'N@50':<12} {'Time(s)':<12}")
    print("-"*100)

    for method_name, result in results.items():
        after = result.get('after', {})
        train_time = result.get('train_time', 0)
        unlearn_time = result.get('unlearn_time', 0)
        total_time = train_time + unlearn_time

        r10 = after.get('recall@10', 0)
        r20 = after.get('recall@20', 0)
        r50 = after.get('recall@50', 0)
        n10 = after.get('ndcg@10', 0)
        n20 = after.get('ndcg@20', 0)
        n50 = after.get('ndcg@50', 0)

        print(f"{method_name:<20} {r10:<12.4f} {r20:<12.4f} {r50:<12.4f} "
              f"{n10:<12.4f} {n20:<12.4f} {n50:<12.4f} {total_time:<12.2f}")

    # Save combined results
    combined_output = os.path.join(PROJ, 'results_all_methods.json')
    with open(combined_output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nCombined results saved to: {combined_output}")

    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)

    if 'FullRetrain' in results:
        fr = results['FullRetrain']['after']
        print(f"\nFull Retrain (Oracle Baseline):")
        print(f"  Recall@10: {fr.get('recall@10', 0):.4f}")
        print(f"  NDCG@10:   {fr.get('ndcg@10', 0):.4f}")

    if 'RecEraser' in results and 'Ours' in results:
        rec = results['RecEraser']['after']
        ours = results['Ours']['after']

        r10_diff = ours.get('recall@10', 0) - rec.get('recall@10', 0)
        n10_diff = ours.get('ndcg@10', 0) - rec.get('ndcg@10', 0)

        print(f"\nRecEraser vs Ours:")
        print(f"  RecEraser R@10: {rec.get('recall@10', 0):.4f}, NDCG@10: {rec.get('ndcg@10', 0):.4f}")
        print(f"  Ours R@10:      {ours.get('recall@10', 0):.4f}, NDCG@10:   {ours.get('ndcg@10', 0):.4f}")
        print(f"  Diff R@10:      {r10_diff:+.4f}")
        print(f"  Diff NDCG@10:   {n10_diff:+.4f}")

        rec_time = results['RecEraser'].get('unlearn_time', 0)
        ours_time = results['Ours'].get('unlearn_time', 0)
        time_diff = rec_time - ours_time

        print(f"\n  RecEraser Unlearn Time: {rec_time:.2f}s")
        print(f"  Ours Unlearn Time:      {ours_time:.2f}s")
        print(f"  Time Diff:              {time_diff:+.2f}s")


if __name__ == '__main__':
    main()
