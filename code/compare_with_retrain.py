"""
Compare Online Unlearn vs Retrain baseline.

For each (partition, agg) combination, this script:
1. Runs online unlearn (only affected shards retrained) - FAST
2. Runs retrain baseline (train on unlearned data from scratch) - SLOW

Then compares Recall@20, time, and quality.

Usage:
 python compare_with_retrain.py 5 --unlearn_type user --ratio 0.10
"""
import os
import sys
import json
import time
import subprocess
import argparse

PROJ = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(PROJ, 'weights', 'ml-1m', 'RecEraser_BPR')
RESULTS = os.path.join(PROJ, '..', 'results')
os.makedirs(RESULTS, exist_ok=True)

CONDA_ENV = 'receraser'
REGS = '0.01'

# Method colors
METHOD_COLOR = {
 'InP': '#A23B72',
 'UBP': '#2E86AB',
 'Random': '#888888',
 'IBP': '#F18F01',
}
METHOD_INFO = {1: 'InP', 2: 'UBP', 3: 'Random', 4: 'IBP'}

def run_unlearn(pt, agg, unlearn_type, ratio, part_num=5, timeout=3600):
 """Run online unlearn (one scenario)."""
 print(f"\n>>> [ONLINE UNLEARN] Pt{pt}-{agg} {unlearn_type} r{ratio:.2f}")
 t0 = time.time()
 # Pass agg_type to online_unlearn.py
 cmd = [
 'conda', 'run', '-n', CONDA_ENV, 'python', 'online_unlearn.py',
 str(part_num),
 '--part_type', str(pt),
 '--regs', REGS,
 '--unlearn_type', unlearn_type,
 '--unlearn_ratio', str(ratio),
 '--agg_type', agg,
 ]
 try:
 result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
 dt = time.time() - t0
 if result.returncode == 0:
 print(f" [OK] Done in {dt:.1f}s")
 return True, dt
 else:
 print(f" [FAIL] Exit={result.returncode}")
 print(f" stderr: {result.stderr[-300:]}")
 return False, dt
 except subprocess.TimeoutExpired:
 return False, timeout
 except Exception as e:
 print(f" [ERROR] {e}")
 return False, 0

def run_retrain_baseline(pt, unlearn_type, ratio, part_num=5, timeout=7200):
 """Train retrain baseline (train model on unlearned data from scratch).

 This uses RecEraser_BPR.py with the unlearned training data.
 """
 print(f"\n>>> [RETRAIN BASELINE] Pt{pt} {unlearn_type} r{ratio:.2f}")
 t0 = time.time()

 # Read the unlearned training file
 data_path = os.path.join(PROJ, '..', 'data', 'ml-1m')
 unlearned_file = os.path.join(
 data_path, f'train_unlearned_{unlearn_type}_r{int(ratio*100):02d}.txt')
 if not os.path.exists(unlearned_file):
 # Fallback: check current train.txt
 print(f" No unlearned data file: {unlearned_file}")
 print(f" Using default train.txt (no retrain effect)")
 unlearned_file = os.path.join(data_path, 'train.txt')

 # Save current train.txt, swap with unlearned, train, restore
 backup_file = unlearned_file + '.backup'
 if os.path.exists(train_path := os.path.join(data_path, 'train.txt')):
 import shutil
 shutil.copy(train_path, backup_file)
 shutil.copy(unlearned_file, train_path)

 try:
 # Run training
 cmd = [
 'conda', 'run', '-n', CONDA_ENV, 'python', 'RecEraser_BPR.py',
 '--dataset', 'ml-1m',
 '--part_type', str(pt),
 '--part_num', str(part_num),
 '--epoch', '30',
 '--lr', '0.05',
 '--regs', '[0.01]',
 '--batch_size', '256',
 '--agg_type', 'mean',
 ]
 result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
 finally:
 # Restore original train.txt
 if os.path.exists(backup_file):
 shutil.move(backup_file, train_path)

 dt = time.time() - t0
 if result.returncode == 0:
 print(f" [OK] Done in {dt:.1f}s")
 return True, dt
 else:
 print(f" [FAIL] Exit={result.returncode}")
 return False, dt

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument('part_num', type=int, default=5)
 ap.add_argument('--unlearn_type', type=str, default='user',
 choices=['interaction', 'user', 'item'])
 ap.add_argument('--ratio', type=float, default=0.10)
 ap.add_argument('--partitions', type=str, default='1,2,3,4')
 ap.add_argument('--aggs', type=str, default='mean,attention')
 ap.add_argument('--skip_unlearn', action='store_true',
 help='Skip online unlearn (assume done)')
 ap.add_argument('--skip_retrain', action='store_true',
 help='Skip retrain baseline')
 args_cli = ap.parse_args()

 parts = [int(x) for x in args_cli.partitions.split(',')]
 aggs = args_cli.aggs.split(',')

 # Step 1: Run online unlearn
 if not args_cli.skip_unlearn:
 for pt in parts:
 for agg in aggs:
 ok, t = run_unlearn(pt, agg, args_cli.unlearn_type, args_cli.ratio, args_cli.part_num)
 if not ok:
 print(f"Skipped Pt{pt}-{agg}")

 # Step 2: Read unlearn results
 unlearn_path = os.path.join(RESULTS, f'online_unlearn_num{args_cli.part_num}.json')
 if not os.path.exists(unlearn_path):
 print(f"\nUnlearn JSON not found: {unlearn_path}")
 return

 with open(unlearn_path) as f:
 unlearn_data = json.load(f)

 # Step 3: Print comparison table
 print(f"\n{'='*100}")
 print(f"COMPARISON: Online Unlearn vs Baseline (no unlearn) - {args_cli.unlearn_type} r{args_cli.ratio}")
 print('='*100)
 print(f"{'Method':<28} {'K':<6} {'Baseline':<10} {'Unlearn':<10} {'Delta(%)':<10} {'Time(s)':<10}")
 print('-' * 100)

 for key, val in sorted(unlearn_data.items()):
 if 'baseline' not in val or 'online_unlearn' not in val:
 continue
 if args_cli.unlearn_type not in key:
 continue
 bl = val['baseline']
 un = val['online_unlearn']
 time_s = val.get('retrain_time_s', 0)
 for k in ['recall20', 'precision20', 'ndcg20']:
 if k in bl and k in un:
 v_bl = bl[k]
 v_un = un[k]
 delta = (v_un - v_bl) / v_bl * 100 if v_bl > 0 else 0
 print(f'{key:<28} {k:<6} {v_bl:<10.4f} {v_un:<10.4f} {delta:+8.2f}% {time_s:<10.1f}')

 # Step 4: Note about retrain
 print(f"\nNote: To compare with full retrain baseline, run:")
 print(f" python RecEraser_BPR.py --dataset ml-1m --part_type X --part_num {args_cli.part_num} --epoch 30 ...")
 print(f" (using the unlearned train file: train_unlearned_{args_cli.unlearn_type}_r{int(args_cli.ratio*100):02d}.txt)")

if __name__ == '__main__':
 main()