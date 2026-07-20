#!/usr/bin/env python3
"""
Train experiments and save results with timestamp.
Usage: python train_and_save.py [--epochs N]
"""
import os
import sys
import subprocess
import argparse
from datetime import datetime

def run_experiment(part_type, part_name, agg_type, epochs):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"results/{part_name}_{agg_type}_{timestamp}.log"

    print(f"\n>>> Training {part_name} {agg_type.upper()} ({epochs} epochs)...")
    print(f"    Log file: {log_file}")

    cmd = [
        'python', 'code/RecEraser_BPR.py',
        '--dataset', 'ml-1m',
        '--part_type', str(part_type),
        '--part_num', '10',
        '--agg_type', agg_type,
        '--epoch', str(epochs),
        '--data_path', './data/',
        '--save_flag', '0'
    ]

    # Run and save output
    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        f.write(result.stdout)
        print(result.stdout)

    # Extract final results
    lines = result.stdout.split('\n')
    for line in reversed(lines):
        if 'recall=[' in line:
            print(f"\n    Final result: {line}")
            return line, log_file

    return None, log_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--part_type', type=int, default=1, help='1=InP, 2=UBP, 3=Random, 4=IBP')
    parser.add_argument('--agg_type', type=str, default='both', choices=['attention', 'mean', 'both'])
    args = parser.parse_args()

    # Create results directory
    os.makedirs('results', exist_ok=True)

    part_names = {1: 'InP', 2: 'UBP', 3: 'Random', 4: 'IBP'}
    part_name = part_names.get(args.part_type, 'InP')

    print("=" * 60)
    print(f"TRAINING EXPERIMENTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Epochs: {args.epochs}, Partition: {part_name}")
    print("=" * 60)

    all_results = {}

    if args.agg_type in ['attention', 'both']:
        line, log_file = run_experiment(args.part_type, part_name, 'attention', args.epochs)
        all_results['attention'] = {'line': line, 'log': log_file}

    if args.agg_type in ['mean', 'both']:
        line, log_file = run_experiment(args.part_type, part_name, 'mean', args.epochs)
        all_results['mean'] = {'line': line, 'log': log_file}

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<15} {'Log File':<50}")
    print("-" * 60)
    for agg, data in all_results.items():
        print(f"{agg.capitalize():<15} {data['log']}")

    # Save summary
    summary_file = f"results/{part_name}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Training Summary - {datetime.now()}\n")
        f.write(f"Partition: {part_name}, Epochs: {args.epochs}\n")
        f.write("=" * 60 + "\n")
        for agg, data in all_results.items():
            f.write(f"\n{agg.upper()}:\n")
            f.write(f"Log: {data['log']}\n")
            f.write(f"Result: {data['line']}\n")

    print(f"\nSummary saved to: {summary_file}")
    print("=" * 60)

if __name__ == '__main__':
    main()
