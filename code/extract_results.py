"""
Extract results from RecEraser_BPR.py output log
Usage:
 1. Save log to file: python RecEraser_BPR.py ... > log.txt 2>&1
 2. Run: python extract_results.py log.txt
 3. Or auto-detect: python extract_results.py
"""
import sys
import os
import re
import json
import pickle
import glob

def find_latest_log():
 """Find most recent log file"""
 # Common log locations
 patterns = [
 '*.log',
 'output.log',
 'train_*.log',
 'results/*.log',
 ]
 candidates = []
 for p in patterns:
 candidates.extend(glob.glob(p))
 candidates.extend(glob.glob('../' + p))

 if not candidates:
 return None

 # Sort by modification time
 candidates.sort(key=os.path.getmtime, reverse=True)
 return candidates[0]

def parse_log(filepath):
 """Parse log file and extract Recall@20, Precision, NDCG"""

 with open(filepath, 'r') as f:
 content = f.read()

 results = {}
 current_method = None

 # Parse lines like: [local_model 4] Epoch 29 [10.6s + 43.6s]: train==[...], recall=[0.07596, 0.12610, 0.21061], precision=[0.19450, 0.16773, 0.07299], ndcg=[0.21061, 0.20355, 0.16296]
 pattern = r'recall=\[([\d.]+), ([\d.]+), ([\d.]+)\], precision=\[([\d.]+), ([\d.]+), ([\d.]+)\], ndcg=\[([\d.]+), ([\d.]+), ([\d.]+)\]'

 # Find all epoch evaluations
 matches = re.findall(pattern, content)

 if not matches:
 print('No epoch evaluation found in log')
 return {}

 # Use LAST match (final epoch)
 last = matches[-1]
 r10, r20, r50 = float(last[0]), float(last[1]), float(last[2])
 p10, p20, p50 = float(last[3]), float(last[4]), float(last[5])
 n10, n20, n50 = float(last[6]), float(last[7]), float(last[8])

 return {
 'recall10': r10, 'recall20': r20, 'recall50': r50,
 'precision10': p10, 'precision20': p20, 'precision50': p50,
 'ndcg10': n10, 'ndcg20': n20, 'ndcg50': n50
 }

def detect_method(content):
 """Try to detect which method was used"""
 if 'part_type 1' in content or 'InBP' in content:
 return 'InBP'
 elif 'part_type 2' in content or 'UBP' in content:
 return 'UBP'
 elif 'part_type 3' in content or 'Random' in content:
 return 'Random'
 elif 'part_type 4' in content or 'IBP' in content:
 return 'IBP'
 return 'Unknown'

def main():
 if len(sys.argv) > 1:
 log_file = sys.argv[1]
 else:
 log_file = find_latest_log()
 if not log_file:
 print('No log file found!')
 print('Usage: python extract_results.py [log_file]')
 return

 print(f'Reading: {log_file}')

 with open(log_file, 'r') as f:
 content = f.read()

 method = detect_method(content)
 print(f'Detected method: {method}')

 results = parse_log(log_file)

 if not results:
 print('Failed to extract results')
 return

 print(f'\nExtracted results:')
 for k, v in results.items():
 print(f' {k}: {v:.4f}')

 # Save to JSON
 output_file = f'../results/{method}_results.json'
 os.makedirs('../results', exist_ok=True)
 with open(output_file, 'w') as f:
 json.dump({method: results}, f, indent=2)
 print(f'\n[OK] Saved to: {output_file}')

 # Also save to pickle
 with open(output_file.replace('.json', '.pkl'), 'wb') as f:
 pickle.dump({method: results}, f)
 print(f'[OK] Saved to: {output_file.replace(".json", ".pkl")}')

if __name__ == '__main__':
 main()