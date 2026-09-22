"""
Chạy benchmark nhiều lần và tổng hợp kết quả
"""

import os
import subprocess
import json
import time

PROJ = os.path.dirname(os.path.abspath(__file__))


def run_method(cmd, log_file, name):
    """Chạy 1 method"""
    print(f"  Running {name}...")
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    return proc


def wait_for_files(result_files, timeout=3600):
    """Đợi cho tất cả result files xuất hiện"""
    print("\nĐợi kết quả...")
    start = time.time()
    while time.time() - start < timeout:
        all_exist = all(os.path.exists(f) for f in result_files)
        if all_exist:
            print("Tất cả kết quả đã có!")
            return True
        time.sleep(30)
        elapsed = int(time.time() - start)
        print(f"  Đã đợi {elapsed}s...")
    return False


def summarize_results(result_files):
    """Tổng hợp kết quả"""
    results = {}

    for name, f in result_files.items():
        try:
            with open(f) as fp:
                d = json.load(fp)
                results[name] = {
                    'before': d.get('before', {}).get('recall@10', 0),
                    'after': d.get('after', {}).get('recall@10', 0),
                    'train_time': d.get('train_time', 0),
                    'unlearn_time': d.get('unlearn_time', 0)
                }
        except Exception as e:
            print(f"Lỗi đọc {f}: {e}")

    return results


def print_summary(results, n_runs):
    """In bảng tổng hợp"""
    print()
    print('='*150)
    print(f'KẾT QUẢ TỔNG HỢP - {n_runs} LẦN CHẠY')
    print('='*150)
    print()

    # Header
    print(f"{'Method':<20} | {'Before (mean±std)':<20} | {'After (mean±std)':<20} | {'Retention %':<15} | {'Train(s)':<15} | {'Unlearn(s)':<15}")
    print('-'*150)

    for name, runs in results.items():
        if not runs:
            continue

        befores = [r['before'] for r in runs]
        afters = [r['after'] for r in runs]

        import numpy as np
        before_mean = np.mean(befores)
        before_std = np.std(befores)
        after_mean = np.mean(afters)
        after_std = np.std(afters)

        retention = (after_mean / before_mean * 100) if before_mean > 0 else 0

        train_times = [r['train_time'] for r in runs]
        unlearn_times = [r['unlearn_time'] for r in runs]

        print(f"{name:<20} | {before_mean:.4f}±{before_std:.4f}     | {after_mean:.4f}±{after_std:.4f}     | {retention:<15.2f} | {np.mean(train_times):<15.1f} | {np.mean(unlearn_times):<15.1f}")

    print('='*150)


def main():
    n_runs = 3

    print('='*80)
    print(f'BENCHMARK - CHẠY {n_runs} LẦN')
    print('='*80)

    for run in range(1, n_runs + 1):
        print(f"\n{'='*60}")
        print(f"LẦN {run}/{n_runs}")
        print(f"{'='*60}")

        # Clean old results
        for f in ['results_full_retrain_bprmf.json', 'results_sisa_bprmf.json',
                   'results_receraser_attention.json', 'results_ours_3components.json']:
            if os.path.exists(f):
                os.remove(f)

        # Run all methods
        processes = {}

        # Full Retrain
        cmd1 = f"python method_1_full_retrain.py --model_name BPRMF --max_epochs 50 --early_stopping False --unlearn_ratio 0.5 --output_suffix benchmark > method1_run{run}.log 2>&1"
        processes['FullRetrain'] = run_method(cmd1, f'method1_run{run}.log', 'FullRetrain')

        # SISA
        cmd2 = f"python method_2_sisa.py --model_name BPRMF --max_epochs 6 --unlearn_ratio 0.5 --output_suffix benchmark > method2_run{run}.log 2>&1"
        processes['SISA'] = run_method(cmd2, f'method2_run{run}.log', 'SISA')

        # RecEraser
        cmd3 = f"python method_3_receraser.py --max_epochs_local 6 --max_epochs_agg 6 --agg_type attention --unlearn_ratio 0.5 --output_suffix benchmark > method3_run{run}.log 2>&1"
        processes['RecEraser'] = run_method(cmd3, f'method3_run{run}.log', 'RecEraser')

        # Ours
        cmd4 = f"python method_4_ours.py --max_epochs 6 --unlearn_ratio 0.5 --output_suffix benchmark > method4_run{run}.log 2>&1"
        processes['Ours'] = run_method(cmd4, f'method4_run{run}.log', 'Ours')

        # Wait for completion
        for name, proc in processes.items():
            proc.wait()
            print(f"  {name} hoàn thành!")

        time.sleep(5)  # Đợi file được ghi xong

    # Tổng hợp kết quả
    print(f"\n{'='*60}")
    print("TỔNG HỢP KẾT QUẢ")
    print(f"{'='*60}")

    result_files_template = {
        'FullRetrain': 'results_full_retrain_bprmf_benchmark.json',
        'SISA': 'results_sisa_bprmf_benchmark.json',
        'RecEraser': 'results_receraser_attention_benchmark.json',
        'Ours': 'results_ours_3components_benchmark.json'
    }

    all_results = {name: [] for name in result_files_template}

    for run in range(1, n_runs + 1):
        for name, template in result_files_template.items():
            f = template
            try:
                with open(f) as fp:
                    d = json.load(fp)
                    all_results[name].append({
                        'before': d.get('before', {}).get('recall@10', 0),
                        'after': d.get('after', {}).get('recall@10', 0),
                        'train_time': d.get('train_time', 0),
                        'unlearn_time': d.get('unlearn_time', 0)
                    })
            except:
                pass

    print_summary(all_results, n_runs)


if __name__ == '__main__':
    main()
