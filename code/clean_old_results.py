"""
Sort logs/ and results/ into the results of the current code and the old ones,
and remove the old ones.

A log counts as NEW when it was written by the code after commit a716452,
which is the last fix to what the experiments measure (SISA evaluated with
its own mean aggregation, the GLOBAL retrain measured separately, the
affected shards retrained from a fresh model, per-shard timing). Only that
code prints the "S5 rate check" line, so the line is the marker:

  exp_partition_quality log  NEW  if it has "S5 rate check", or if it only
                                  runs the RECERASER arm (no shard rate is
                                  printed for it) and has the parallel= cost
                             RUNNING if no run has finished yet (kept)
                             OLD  otherwise
  single_request log         NEW  (the experiment only exists in the new code)
  block_*.out, _preflight    kept: these are the runner's own logs
  anything else in logs/     OLD  (cmp*, m_*, pq*, ... from the earlier runs)
  results/*                  OLD  except results/csv/, which `run_blocks.sh csv`
                                  rebuilds from the logs that are kept

A file changed in the last 15 minutes is never touched, in case a job is
still writing it.

Usage
  python code/clean_old_results.py                # list only, removes nothing
  python code/clean_old_results.py --yes          # pack the OLD files into
                                                  #   ../old_results_<time>.tar.gz,
                                                  #   then remove them
  python code/clean_old_results.py --yes --no-backup   # remove without a backup
"""

import argparse
import os
import re
import tarfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, 'logs')
RESULTS = os.path.join(ROOT, 'results')

RUN_LINE = re.compile(r'^\s*\[([A-Z_]+)\]\s+R@10=', re.M)
ARM_HDR = re.compile(r'---\s*ARM\s+([A-Z_]+)\s*\(seed\s+\d+\)')
RECENT_S = 15 * 60


def classify_log(path):
    """-> (verdict, reason)"""
    name = os.path.basename(path)
    if name.startswith('block_') or name.startswith('_preflight'):
        return 'KEEP', 'runner log'
    try:
        text = open(path, encoding='utf-8', errors='replace').read()
    except OSError as e:
        return 'KEEP', f'cannot read ({e})'

    if name.startswith('single_request'):
        return 'NEW', 'single-request experiment'

    runs = RUN_LINE.findall(text)
    arms = set(ARM_HDR.findall(text)) | set(runs)
    if 'S5 rate check' in text:
        return 'NEW', f'{len(runs)} runs, current code'
    if arms and arms <= {'RECERASER'} and 'parallel=' in text:
        return 'NEW', f'{len(runs)} runs, RECERASER only'
    if not runs and ('--- ARM' in text or 'PREFLIGHT' in text or not text.strip()):
        return 'RUNNING', 'no finished run yet'
    if runs and 'parallel=' not in text:
        return 'OLD', 'before the timing fix (no parallel= cost)'
    if runs:
        return 'OLD', 'before a716452 (SISA hard-routed, GLOBAL retrain not measured)'
    return 'OLD', 'not an experiment log of the current runner'


def human(n):
    for u in ('B', 'K', 'M', 'G'):
        if n < 1024:
            return f'{n:.0f}{u}'
        n /= 1024
    return f'{n:.1f}T'


def size_of(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(d, f))
               for d, _, fs in os.walk(p) for f in fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--yes', action='store_true', help='actually remove the OLD files')
    ap.add_argument('--no-backup', action='store_true',
                    help='remove without packing them into a .tar.gz first')
    ap.add_argument('--is-new', metavar='LOG', default=None,
                    help='exit 0 if LOG was written by the current code, 1 '
                         'otherwise (used by run_blocks.sh to decide whether a '
                         'finished log can be skipped)')
    args = ap.parse_args()

    if args.is_new:
        v, _ = classify_log(args.is_new) if os.path.exists(args.is_new) else ('OLD', '')
        raise SystemExit(0 if v == 'NEW' else 1)

    now = time.time()
    rows = []   # (verdict, path, reason)

    if os.path.isdir(LOGS):
        for name in sorted(os.listdir(LOGS)):
            p = os.path.join(LOGS, name)
            if now - os.path.getmtime(p) < RECENT_S:
                rows.append(('KEEP', p, 'changed in the last 15 min'))
                continue
            if os.path.isdir(p):
                rows.append(('OLD', p, 'directory of earlier runs'))
                continue
            v, why = classify_log(p)
            rows.append((v, p, why))

    if os.path.isdir(RESULTS):
        for name in sorted(os.listdir(RESULTS)):
            p = os.path.join(RESULTS, name)
            if name == 'csv':
                rows.append(('KEEP', p, 'rebuilt by run_blocks.sh csv'))
            elif now - os.path.getmtime(p) < RECENT_S:
                rows.append(('KEEP', p, 'changed in the last 15 min'))
            else:
                rows.append(('OLD', p, 'result file of an earlier run'))

    order = {'NEW': 0, 'RUNNING': 1, 'KEEP': 2, 'OLD': 3}
    rows.sort(key=lambda r: (order[r[0]], r[1]))
    print(f'{"":8}{"file":<46}{"size":>7}  reason')
    print('-' * 100)
    for v, p, why in rows:
        rel = os.path.relpath(p, ROOT)
        print(f'{v:<8}{rel:<46}{human(size_of(p)):>7}  {why}')

    old = [p for v, p, _ in rows if v == 'OLD']
    total = sum(size_of(p) for p in old)
    print()
    print(f'NEW={sum(r[0] == "NEW" for r in rows)}  '
          f'RUNNING={sum(r[0] == "RUNNING" for r in rows)}  '
          f'KEEP={sum(r[0] == "KEEP" for r in rows)}  '
          f'OLD={len(old)} ({human(total)})')

    if not old:
        print('nothing to remove')
        return
    if not args.yes:
        print('\nlist only - nothing removed. Remove the OLD files with:')
        print('  python code/clean_old_results.py --yes')
        return

    if not args.no_backup:
        stamp = time.strftime('%Y%m%d_%H%M%S')
        tar_path = os.path.join(os.path.dirname(ROOT), f'old_results_{stamp}.tar.gz')
        with tarfile.open(tar_path, 'w:gz') as tar:
            for p in old:
                tar.add(p, arcname=os.path.relpath(p, ROOT))
        print(f'\n[backup] {tar_path}  ({human(os.path.getsize(tar_path))})')

    import shutil
    for p in old:
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)
    print(f'[removed] {len(old)} files/dirs ({human(total)})')


if __name__ == '__main__':
    main()
