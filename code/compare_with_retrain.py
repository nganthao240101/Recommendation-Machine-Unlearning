"""
Compare Online Unlearn vs Retrain baseline (train on unlearned data).

Usage:
 python compare_with_retrain.py 5 --unlearn_type user --ratio 0.10
 python compare_with_retrain.py 5 --unlearn_type user --ratio 0.10 --partitions 2 --aggs attention
"""
import os
import sys
import json
import time
import subprocess
import argparse
import shutil

PROJ = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(PROJ, 'weights', 'ml-1m', 'RecEraser_BPR')
RESULTS = os.path.join(PROJ, '..', 'results')
DATA_DIR = os.path.join(PROJ, '..', 'data', 'ml-1m')
os.makedirs(RESULTS, exist_ok=True)

CONDA_ENV = 'receraser'
REGS = '0.01'

METHOD_INFO = {1: 'InP', 2: 'UBP', 3: 'Random', 4: 'IBP'}


def run_online_unlearn(pt, agg, unlearn_type, ratio, part_num=5, timeout=3600):
 """Run online unlearn (only affected shards retrained)."""
 print(f"\n>>> [ONLINE UNLEARN] Pt{pt}-{agg} {unlearn_type} r{ratio:.2f}")
 t0 = time.time()
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
 print(f" stderr: {result.stderr[-500:]}")
 return False, dt
 except subprocess.TimeoutExpired:
 print(f" [TIMEOUT] > {timeout}s")
 return False, timeout
 except Exception as e:
 print(f" [ERROR] {e}")
 return False, 0


def run_retrain_baseline(pt, agg, unlearn_type, ratio, part_num=5, timeout=7200):
 """Run retrain baseline: train model from scratch on unlearned data."""
 print(f"\n>>> [RETRAIN BASELINE] Pt{pt}-{agg} {unlearn_type} r{ratio:.2f}")
 t0 = time.time()

 # The unlearned training file
 unlearned_file = os.path.join(
 DATA_DIR, f'train_unlearned_{unlearn_type}_r{int(ratio*100):02d}.txt')
 if not os.path.exists(unlearned_file):
 print(f" No unlearned data file: {unlearned_file}")
 print(f" Run online_unlearn.py first to generate unlearned data")
 return False, 0

 # Backup current train.txt, swap with unlearned, train, restore
 train_path = os.path.join(DATA_DIR, 'train.txt')
 backup_file = os.path.join(DATA_DIR, 'train.txt.backup')
 if os.path.exists(train_path):
 shutil.copy(train_path, backup_file)
 shutil.copy(unlearned_file, train_path)

 try:
 # Run training on unlearned data
 cmd = [
 'conda', 'run', '-n', CONDA_ENV, 'python', 'RecEraser_BPR.py',
 '--dataset', 'ml-1m',
 '--part_type', str(pt),
 '--part_num', str(part_num),
 '--epoch', '30',
 '--lr', '0.05',
 '--regs', '[0.01]',
 '--batch_size', '256',
 '--agg_type', agg,
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
 print(f" stderr: {result.stderr[-500:]}")
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
 help='Skip retrain baseline (assume done)')
 args_cli = ap.parse_args()

 parts = [int(x) for x in args_cli.partitions.split(',')]
 aggs = args_cli.aggs.split(',')
 ratio_str = f"{int(args_cli.ratio*100):02d}"

 print(f"Partitions: {parts}")
 print(f"Aggregations: {aggs}")
 print(f"Unlearn type: {args_cli.unlearn_type}, Ratio: {args_cli.ratio}")

 # Step 1: Run online unlearn
 if not args_cli.skip_unlearn:
 print("\n" + "="*60)
 print("STEP 1: Online Unlearn")
 print("="*60)
 for pt in parts:
 for agg in aggs:
 run_online_unlearn(pt, agg, args_cli.unlearn_type,
 args_cli.ratio, args_cli.part_num)

 # Step 2: Run retrain baseline
 retrain_times = {}
 if not args_cli.skip_retrain:
 print("\n" + "="*60)
 print("STEP 2: Retrain Baseline")
 print("="*60)
 for pt in parts:
 for agg in aggs:
 key = f"Pt{pt}-{agg}"
 ok, t = run_retrain_baseline(pt, agg, args_cli.unlearn_type,
 args_cli.ratio, args_cli.part_num)
 if ok:
 retrain_times[key] = t

 # Step 3: Read unlearn results
 unlearn_path = os.path.join(RESULTS, f'online_unlearn_num{args_cli.part_num}.json')
 if not os.path.exists(unlearn_path):
 print(f"\nUnlearn JSON not found: {unlearn_path}")
 return

 with open(unlearn_path) as f:
 unlearn_data = json.load(f)

 # Step 4: Print comparison table
 print("\n" + "="*100)
 print(f"COMPARISON: Online Unlearn vs Retrain Baseline - {args_cli.unlearn_type} r{args_cli.ratio}")
 print("="*100)
 print(f"{'Method':<25} {'K':<8} {'Unlearn':<10} {'Retrain':<10} {'Diff':<10} {'UnlearnT(s)':<12} {'RetrainT(s)':<10}")
 print("-" * 100)

 for key, val in sorted(unlearn_data.items()):
 if 'baseline' not in val or 'online_unlearn' not in val:
 continue
 if args_cli.unlearn_type not in key:
 continue
 bl = val['baseline']
 un = val['online_unlearn']
 unlearn_t = val.get('retrain_time_s', 0)

 for k in ['recall20', 'precision20', 'ndcg20']:
 if k in bl and k in un:
 v_un = un[k]
 v_rt = bl[k] # baseline Recall@20 = upper bound
 diff = v_un - v_rt
 # Find retrain time
 pt_match = None
 for p in parts:
 if METHOD_INFO[p] in key:
 pt_match = p
 break
 retrain_t = 0
 if pt_match:
 for a in aggs:
 if a in key:
 rt_key = f"Pt{pt_match}-{a}"
 retrain_t = retrain_times.get(rt_key, 0)
 break

 print(f'{key:<25} {k:<8} {v_un:<10.4f} {v_rt:<10.4f} '
 f'{diff:+8.4f} {unlearn_t:<12.1f} {retrain_t:<10.1f}')

 print("\n" + "="*60)
 print("Note: 'Retrain' = baseline (no unlearn) - upper bound for quality.")
 print("If Online Unlearn ≈ Retrain in Recall@20, the method is good.")
 print("And time should be MUCH less than retrain time.")
 print("="*60)


if __name__ == '__main__':
 main()