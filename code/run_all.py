#!/usr/bin/env python3
"""
Run All Methods - Hỗ trợ chạy ngầm với nohup

Cách dùng:
    # Chạy với nohup (chạy ngầm, không bị dừng khi mất SSH)
    nohup python run_all.py --epochs 50 --retrain_epochs 20 --dataset ml-1m > run_all.log 2>&1 &

    # Kiểm tra tiến trình
    ps aux | grep run_all

    # Xem log
    tail -f run_all.log

    # Xem kết quả JSON
    cat results_full_retrain_*.json
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

from method_1_full_retrain import run_full_retrain
from method_2_sisa import run_sisa
from method_3_receraser import run_receraser
from method_4_ours import run_ours


def run_all_methods(epochs=50, retrain_epochs=20, n_shards=8, dataset='ml-1m',
                   unlearn_ratio=0.1, emb_dim=64, batch_size=512, lr=0.05):
    """Chạy tất cả 4 methods với cùng config."""

    print("=" * 70)
    print("RUNNING ALL 4 METHODS")
    print("=" * 70)
    print(f"Config:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Retrain epochs: {retrain_epochs}")
    print(f"  - N shards: {n_shards}")
    print(f"  - Dataset: {dataset}")
    print(f"  - Unlearn ratio: {unlearn_ratio}")
    print(f"  - Emb dim: {emb_dim}")
    print("=" * 70)

    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{dataset}_s{emb_dim}"

    # Method 1: Full Retrain (Oracle Baseline)
    print("\n" + "=" * 70)
    print("METHOD 1: FULL RETRAIN (Oracle Baseline)")
    print("=" * 70)
    print("Description: Train lại toàn bộ model từ đầu trên filtered data")
    print("(sau khi xóa user cần unlearn)")
    print("=" * 70)

    t0 = time.time()
    try:
        results['FullRetrain'] = run_full_retrain(
            model_name='BPRMF',
            dataset=dataset,
            batch_size=batch_size,
            lr=lr,
            emb_dim=emb_dim,
            max_epochs=epochs,
            early_stopping=False,
            patience=10,
            unlearn_ratio=unlearn_ratio,
            output_suffix=suffix
        )
        results['FullRetrain']['time'] = time.time() - t0
        print(f"\nFull Retrain completed in {results['FullRetrain']['time']:.2f}s")
    except Exception as e:
        print(f"ERROR in Full Retrain: {e}")
        results['FullRetrain'] = {'error': str(e)}

    # Method 2: SISA
    print("\n" + "=" * 70)
    print("METHOD 2: SISA")
    print("=" * 70)
    print("Description: Sharded Isolated Slicing and Aggregation")
    print("Train per-shard models, aggregate với mean")
    print("=" * 70)

    t0 = time.time()
    try:
        results['SISA'] = run_sisa(
            model_name='BPRMF',
            dataset=dataset,
            emb_dim=emb_dim,
            n_shards=n_shards,
            batch_size=batch_size,
            lr=lr,
            max_epochs=epochs,
            unlearn_ratio=unlearn_ratio,
            retrain_epochs=retrain_epochs,
            output_suffix=suffix
        )
        results['SISA']['time'] = time.time() - t0
        print(f"\nSISA completed in {results['SISA']['time']:.2f}s")
    except Exception as e:
        print(f"ERROR in SISA: {e}")
        results['SISA'] = {'error': str(e)}

    # Method 3: RecEraser
    print("\n" + "=" * 70)
    print("METHOD 3: RECERASER")
    print("=" * 70)
    print("Description: Per-shard embeddings + Attention aggregation")
    print("Two-phase training: local -> aggregator")
    print("Unlearn: Retrain shards + aggregator (attention weights CHANGE)")
    print("=" * 70)

    t0 = time.time()
    try:
        results['RecEraser'] = run_receraser(
            dataset=dataset,
            emb_dim=emb_dim,
            n_shards=n_shards,
            batch_size=batch_size,
            lr=lr,
            attention_size=32,
            max_epochs_local=epochs,
            max_epochs_agg=epochs,
            agg_type='attention',
            unlearn_ratio=unlearn_ratio,
            retrain_epochs=retrain_epochs,
            output_suffix=suffix
        )
        results['RecEraser']['time'] = time.time() - t0
        print(f"\nRecEraser completed in {results['RecEraser']['time']:.2f}s")
    except Exception as e:
        print(f"ERROR in RecEraser: {e}")
        results['RecEraser'] = {'error': str(e)}

    # Method 4: Ours (3 Components)
    print("\n" + "=" * 70)
    print("METHOD 4: OURS (3 Components)")
    print("=" * 70)
    print("Description: Deletion-Stable 3 Components")
    print("  - Component 1: Deletion-Local Signatures")
    print("  - Component 2: Deletion-Stable Assignment")
    print("  - Component 3: Isolated Training (fixed equal weights)")
    print("Unlearn: Retrain shards ONLY, NO aggregator (weights STABLE)")
    print("=" * 70)

    t0 = time.time()
    try:
        results['Ours'] = run_ours(
            dataset=dataset,
            emb_dim=emb_dim,
            n_shards=n_shards,
            batch_size=batch_size,
            lr=lr,
            max_epochs=epochs,
            unlearn_ratio=unlearn_ratio,
            retrain_epochs=retrain_epochs,
            output_suffix=suffix
        )
        results['Ours']['time'] = time.time() - t0
        print(f"\nOurs completed in {results['Ours']['time']:.2f}s")
    except Exception as e:
        print(f"ERROR in Ours: {e}")
        results['Ours'] = {'error': str(e)}

    # Save combined results
    combined_file = os.path.join(PROJ, f'results_all_methods_{suffix}_{timestamp}.json')
    with open(combined_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<20} {'R@10':<12} {'NDCG@10':<12} {'Time(s)':<12}")
    print("-" * 70)

    for method_name, result in results.items():
        if 'error' not in result:
            after = result.get('after', {})
            r10 = after.get('recall@10', 0)
            n10 = after.get('ndcg@10', 0)
            t = result.get('time', 0)
            print(f"{method_name:<20} {r10:<12.4f} {n10:<12.4f} {t:<12.2f}")
        else:
            print(f"{method_name:<20} {'ERROR':<12}")

    print("\n" + "=" * 70)
    print(f"Results saved to: {combined_file}")
    print("=" * 70)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all 4 unlearning methods')

    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs (default: 50)')
    parser.add_argument('--retrain_epochs', type=int, default=20,
                       help='Retrain epochs after unlearn (default: 20)')
    parser.add_argument('--n_shards', type=int, default=8,
                       help='Number of shards (default: 8)')
    parser.add_argument('--dataset', type=str, default='ml-1m',
                       help='Dataset name (default: ml-1m)')
    parser.add_argument('--unlearn_ratio', type=float, default=0.1,
                       help='Ratio of users to unlearn (default: 0.1)')
    parser.add_argument('--emb_dim', type=int, default=64,
                       help='Embedding dimension (default: 64)')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.05,
                       help='Learning rate (default: 0.05)')

    args = parser.parse_args()

    run_all_methods(
        epochs=args.epochs,
        retrain_epochs=args.retrain_epochs,
        n_shards=args.n_shards,
        dataset=args.dataset,
        unlearn_ratio=args.unlearn_ratio,
        emb_dim=args.emb_dim,
        batch_size=args.batch_size,
        lr=args.lr
    )
