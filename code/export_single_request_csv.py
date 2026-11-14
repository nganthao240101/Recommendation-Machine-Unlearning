"""
Export the CONSOLE SUMMARY of exp_single_request.py logs to CSV.

The summary block looks like

    arm                         mean           std           min           max     n
    ------------------------------------------------------------------------------
    PROPOSED                   12.34+-         1.23         10.00         15.00    20
    ...
      ratio to Retrain (GLOBAL), mean=123.45s:
        PROPOSED               12.34s   0.100x

and becomes one row per (log, arm):

    log, arm, time_mean_s, time_std_s, time_min_s, time_max_s, n_requests,
    ratio_to_global

Usage
  python code/export_single_request_csv.py logs/single_request_ml1m.out \
      --out results/csv/single_request.csv
"""

import argparse
import csv
import os
import re

MARK = 'CONSOLE SUMMARY'
ROW = re.compile(r'^(\S+)\s+([\d.]+)\+-\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$')
RATIO = re.compile(r'^\s+(\S+)\s+([\d.]+)s\s+([\d.]+)x\s*$')
COLS = ['log', 'arm', 'time_mean_s', 'time_std_s', 'time_min_s', 'time_max_s',
        'n_requests', 'ratio_to_global']


REQ = re.compile(r'^\s*request (\d+)/(\d+) \(user \d+\): ([\d.]+)s\s*$')
HDR = re.compile(r'---\s*ARM\s+([A-Z_]+)\s*\(seed\s+(\d+)\)')


def parse(path):
    """Rows per arm: mean / std / min / max seconds of one deletion request.

    Taken from the per-request lines when the log has them, so a log that was
    resumed - its earlier requests kept, the summary covering only the new
    runs - still counts every request. Logs from before those lines fall back
    on the CONSOLE SUMMARY table.
    """
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    if MARK not in text:
        return None
    req, cur = {}, None
    for ln in text.splitlines():
        m = HDR.search(ln)
        if m:
            cur = m.group(1)
            continue
        m = REQ.match(ln)
        if m and cur:
            req.setdefault(cur, []).append(float(m.group(3)))
    if req:
        base = req.get('GLOBAL')
        bmean = sum(base) / len(base) if base else None
        rows = []
        for arm, v in req.items():
            mean = sum(v) / len(v)
            sd = (sum((x - mean) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
            rows.append({
                'arm': arm, 'time_mean_s': f'{mean:.2f}', 'time_std_s': f'{sd:.2f}',
                'time_min_s': f'{min(v):.2f}', 'time_max_s': f'{max(v):.2f}',
                'n_requests': str(len(v)),
                'ratio_to_global': f'{mean / bmean:.3f}' if bmean else '',
            })
        return rows

    rows, ratios = {}, {}
    for ln in text[text.index(MARK):].splitlines():
        m = ROW.match(ln)
        if m:
            arm = m.group(1)
            rows[arm] = {
                'arm': arm,
                'time_mean_s': m.group(2), 'time_std_s': m.group(3),
                'time_min_s': m.group(4), 'time_max_s': m.group(5),
                'n_requests': m.group(6),
            }
            continue
        m = RATIO.match(ln)
        if m:
            ratios[m.group(1)] = m.group(3)
    for arm, r in rows.items():
        r['ratio_to_global'] = ratios.get(arm, '')
    return list(rows.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out_rows = []
    for path in args.logs:
        if not os.path.exists(path):
            print(f'[skip] {path} (not found)')
            continue
        rows = parse(path)
        if rows is None:
            print(f'[skip] {path} (no CONSOLE SUMMARY - still running or crashed)')
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        for r in rows:
            r['log'] = stem
            out_rows.append(r)
        print(f'[read] {path}  ({len(rows)} arms)')

    if not out_rows:
        print('nothing to write')
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(out_rows)
    print(f'[written] {args.out}  ({len(out_rows)} rows)')


if __name__ == '__main__':
    main()
