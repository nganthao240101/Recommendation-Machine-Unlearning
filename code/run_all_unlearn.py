"""
Run unlearn for:
- 4 partition types (InP=1, UBP=2, Random=3, IBP=4)
- 2 aggregation types (mean, attention)
- 1 unlearn type (user-specified)
- 1 unlearn ratio (user-specified)

Total: 4 x 2 = 8 combinations

Usage:
  python run_all_unlearn.py --unlearn_type interaction --ratio 0.1
  python run_all_unlearn.py --unlearn_type user --ratio 0.2
  python run_all_unlearn.py --unlearn_type item --ratio 0.05
"""
import subprocess
import sys
import os
import time
import argparse

PROJ = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJ)

PART_TYPES = [
    (1, 'InP'),
    (2, 'UBP'),
    (3, 'Random'),
    (4, 'IBP'),
]

AGG_TYPES = ['mean', 'attention']
CONDA_ENV = 'receraser'
PART_NUM = 5
REGS = '0.001'  # Default regs = 0.001

def run_one(pt, agg, unlearn_type, ratio):
    """Run one unlearn scenario"""
    name = f"{pt[1]}_{unlearn_type}_r{int(ratio*100):02d}_{agg}"
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print('='*60)

    cmd = [
        'conda', 'run', '-n', CONDA_ENV, 'python', 'online_unlearn.py',
        str(PART_NUM),
        '--part_type', str(pt[0]),
        '--regs', REGS,
        '--unlearn_type', unlearn_type,
        '--unlearn_ratio', str(ratio),
        '--agg_type', agg,
    ]
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        dt = time.time() - t0
        if result.returncode == 0:
            print(f"[OK] {name} completed in {dt:.1f}s")
            return True
        else:
            print(f"[FAIL] {name} failed (exit={result.returncode})")
            # Print last 500 chars of stderr
            stderr = result.stderr[-500:] if result.stderr else ""
            stdout = result.stdout[-500:] if result.stdout else ""
            print(f"   stderr: {stderr}")
            print(f"   stdout: {stdout}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {name} exceeded 1 hour")
        return False
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unlearn_type', type=str, required=True,
                    choices=['interaction', 'user', 'item'])
    ap.add_argument('--ratio', type=float, required=True,
                    help='Unlearn ratio, e.g. 0.05, 0.10, 0.20')
    ap.add_argument('--regs', type=str, default=REGS,
                    help='Regularization (default: 0.001)')
    args_cli = ap.parse_args()

    print(f"Unlearn type: {args_cli.unlearn_type}")
    print(f"Unlearn ratio: {args_cli.ratio}")
    print(f"Regs: {args_cli.regs}")
    print(f"Partitions: {[pt[1] for pt in PART_TYPES]}")
    print(f"Aggregations: {AGG_TYPES}")
    print(f"Total combinations: {len(PART_TYPES) * len(AGG_TYPES)}")

    success = 0
    total = 0

    for pt in PART_TYPES:
        for agg in AGG_TYPES:
            total += 1
            if run_one(pt, agg, args_cli.unlearn_type, args_cli.ratio):
                success += 1

    print(f"\n{'='*60}")
    print(f"Summary: {success}/{total} successful")
    print('='*60)

if __name__ == '__main__':
    main()