"""
Fair Benchmark - So sanh cong bang giua cac methods

Setup cong bang:
1. Full Retrain: train du epochs de converge (150 epochs)
2. RecEraser: train du epochs de converge (150 epochs local + 150 agg)
3. SISA: train du epochs de converge (150 epochs)
4. Ours: train du epochs de converge (150 epochs)

Data:
- Unlearn 50% users
- So sanh Before/After tren RETAINED users
"""

import os
import sys
import subprocess
import time

PROJ = os.path.dirname(os.path.abspath(__file__))


def run_command(cmd, log_file):
    """Chay command va ghi log"""
    print(f"Running: {cmd}")
    print(f"Log file: {log_file}")
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    return proc


def main():
    print("="*80)
    print("FAIR BENCHMARK - SO SANH CONG BANG")
    print("="*80)
    print()
    print("Config:")
    print("  - Unlearn ratio: 50%")
    print("  - Epochs: 150 (du de converge)")
    print("  - Evaluate: chi tren RETAINED users sau unlearn")
    print()
    print("="*80)

    # Commands
    commands = {
        'FullRetrain': {
            'cmd': 'python method_1_full_retrain.py --model_name BPRMF --max_epochs 150 --early_stopping False --unlearn_ratio 0.5 --output_suffix fair150',
            'log': 'method1_fair.log'
        },
        'SISA': {
            'cmd': 'python method_2_sisa.py --model_name BPRMF --max_epochs 150 --unlearn_ratio 0.5 --output_suffix fair150',
            'log': 'method2_fair.log'
        },
        'RecEraser': {
            'cmd': 'python method_3_receraser.py --max_epochs_local 150 --max_epochs_agg 150 --unlearn_ratio 0.5 --output_suffix fair150',
            'log': 'method3_fair.log'
        },
        'Ours': {
            'cmd': 'python method_4_ours.py --max_epochs 150 --unlearn_ratio 0.5 --output_suffix fair150',
            'log': 'method4_fair.log'
        }
    }

    # Run all commands
    processes = {}
    for name, info in commands.items():
        print(f"\nStarting {name}...")
        processes[name] = run_command(info['cmd'], info['log'])

    print("\n" + "="*80)
    print("TAT CA METHODS DA BAT DAU!")
    print("="*80)
    print("\nTheo doi tien trinh:")
    print("  ps aux | grep method")
    print("\nXem log:")
    print("  tail -f method1_fair.log")
    print("\nTong hop ket qua khi xong:")
    print("  python summarize_fair.py")
    print("="*80)


if __name__ == '__main__':
    main()
