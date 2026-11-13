"""
Sort logs/ and results/ into the results of the current code and the old ones,
and remove the old ones.

A log counts as NEW when it was written by the code after commit a716452,
which is the last fix to what the experiments measure (SISA evaluated with
its own mean aggregation, the GLOBAL retrain measured separately, the
affected shards retrained from a fresh model, per-shard timing). Only that
code prints the "S5 rate check" line, so the line is the marker:

  NEW      the log has "S5 rate check" (or, for a RECERASER-only run, the
           parallel= cost), or it is a finished single-request log
  RUNNING  a job is writing it right now. The runner starts every job as
           `... | tee logs/<name>.out`, so a live `tee` process on the file
           is the proof; without /proc (not Linux) a change in the last
           60 minutes stands in for it
  FAILED   unfinished and no job is writing it: the run crashed or was
           stopped. It is removed so the block runs it again
  BUSY     .nfsXXXX: an NFS placeholder for a file that was deleted while a
           process still had it open. Removing it only makes NFS create it
           again, so it is left alone; it goes away when that process exits
  OLD      a finished log from the code before a716452, and everything else
           in logs/ (cmp*, m_*, pq*, ...) and in results/ except results/csv/

Nothing changed in the last 15 minutes is removed.

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
LIVE_FALLBACK_S = 60 * 60

# the logs run_blocks.sh writes, in run order
# in the order run_blocks.sh all runs them
_DS = ('ml1m', 'yelp')
EXPERIMENT_LOGS = (
    [f'main_{d}' for d in _DS]
    + [f'mode_{m}_{d}' for d in _DS for m in ('most', 'fewest')]
    + [f'single_{d}' for d in _DS]
    + [f'k{k}_{d}' for d in _DS for k in (4, 16)]
    + [f'dim{x}_{d}' for d in _DS for x in (32, 128)]
    + [f'recrand_{d}' for d in _DS]
)
# the RECERASER arm prints this since it starts from a random model, as the
# original does; before that it was seeded with a scrambled WMF vector
REC_MARK = '[RecEraser] init='
# printed by the UBP partition since it follows the authors' balanced k-means
# (user-count cap); the earlier port balanced interaction counts instead
UBP_MARK = 'empty-shard restart'


def live_logs():
    """-> set of absolute log paths a running `tee` is writing, or None when
    /proc is not available (not Linux)."""
    if not os.path.isdir('/proc'):
        return None
    live = set()
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                argv = f.read().split(b'\0')
            cwd = os.readlink(f'/proc/{pid}/cwd')
        except OSError:
            continue
        argv = [a.decode('utf-8', 'replace') for a in argv if a]
        if not argv or os.path.basename(argv[0]) != 'tee':
            continue
        for a in argv[1:]:
            if not a.startswith('-'):
                live.add(os.path.realpath(os.path.join(cwd, a)))
    return live


_LIVE = None


def is_live(path):
    global _LIVE
    if _LIVE is None:
        found = live_logs()
        _LIVE = False if found is None else found
    if _LIVE is False:  # no /proc: fall back on how recently it changed
        return time.time() - os.path.getmtime(path) < LIVE_FALLBACK_S
    return os.path.realpath(path) in _LIVE


def first_line(text):
    for ln in text.splitlines():
        if ln.strip():
            return ln.strip()[:70]
    return '(empty)'


def classify_log(path):
    """-> (verdict, reason)"""
    name = os.path.basename(path)
    if name.startswith('.nfs'):
        return 'BUSY', 'NFS placeholder, a process still holds it open'
    if name.startswith('block_') or name.startswith('_preflight'):
        return 'KEEP', 'runner log'
    try:
        text = open(path, encoding='utf-8', errors='replace').read()
    except OSError as e:
        return 'KEEP', f'cannot read ({e})'

    runs = RUN_LINE.findall(text)
    arms = set(ARM_HDR.findall(text)) | set(runs)
    finished = 'CONSOLE SUMMARY' in text

    # ---- not finished: running, stopped, or an abandoned old log. This comes
    # first: a run that crashed half way must read as FAILED with its error,
    # whatever code wrote it, never as OLD.
    if not finished:
        if is_live(path):
            return 'RUNNING', f'a job is writing it ({len(runs)} runs so far)'
        if name[:-4] in EXPERIMENT_LOGS:
            return 'FAILED', f'stopped after {len(runs)} runs, block will rerun it: {last_error(text)}'
        return 'OLD', 'unfinished log of an earlier run, no job writing it'

    # ---- finished: which code wrote it
    rec = 'RECERASER' in arms or re.search(r'^RECERASER\s', text, re.M)
    if rec and REC_MARK not in text:
        return 'OLD', 'RECERASER seeded with the scrambled WMF vector (old code)'
    if rec and not name.startswith('recrand') and UBP_MARK not in text:
        return 'OLD', 'RECERASER with the earlier UBP port (interaction-balanced)'
    if name.startswith('single_'):
        return 'NEW', 'single-request experiment, finished'
    if 'S5 rate check' in text:
        return 'NEW', f'{len(runs)} runs, current code'
    if arms and arms <= {'RECERASER'} and 'parallel=' in text:
        return 'NEW', f'{len(runs)} runs, RECERASER only'
    if runs and 'parallel=' not in text:
        return 'OLD', 'before the timing fix (no parallel= cost)'
    if runs:
        return 'OLD', 'before a716452 (SISA hard-routed, GLOBAL retrain not measured)'
    return 'OLD', 'not an experiment log of the current runner'


def last_error(text):
    """The exception line of the last traceback, else the last line."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if 'Traceback' in text:
        for ln in reversed(lines):
            if re.match(r'^[A-Za-z_.]*(Error|Exception|Interrupt)', ln):
                return ln[:110]
    return (lines[-1] if lines else '(empty)')[:110]


def print_status():
    """The progress of the 12 experiment logs of run_blocks.sh."""
    counts = {}
    for n in EXPERIMENT_LOGS:
        p = os.path.join(LOGS, n + '.out')
        if not os.path.exists(p):
            v, why = 'MISSING', ''
        else:
            v, why = classify_log(p)
            text = open(p, encoding='utf-8', errors='replace').read()
            if v == 'NEW' and 'CONSOLE SUMMARY' in text:
                v = 'DONE'
            elif v == 'NEW':
                v = 'RUNNING' if is_live(p) else 'FAILED'
                why = (why + (', job alive' if v == 'RUNNING'
                              else ', stopped before the summary'))
            if v == 'RUNNING':
                hdr = ARM_HDR.findall(text)
                why += f' | last arm: {hdr[-1]}' if hdr else ''
        counts[v] = counts.get(v, 0) + 1
        print(f'  {n:<24}{v:<9}{why}')
    print()
    print('  ' + '  '.join(f'{k}={v}' for k, v in sorted(counts.items()))
          + f'  (tong {len(EXPERIMENT_LOGS)})')


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
    ap.add_argument('--status', action='store_true',
                    help='print the progress of the 12 experiment logs and exit')
    args = ap.parse_args()

    if args.status:
        print_status()
        return

    if args.is_new:
        v, _ = classify_log(args.is_new) if os.path.exists(args.is_new) else ('OLD', '')
        raise SystemExit(0 if v == 'NEW' else 1)

    now = time.time()
    rows = []   # (verdict, path, reason)

    if os.path.isdir(LOGS):
        for name in sorted(os.listdir(LOGS)):
            p = os.path.join(LOGS, name)
            if name.startswith('.nfs'):
                rows.append(('BUSY', p, 'NFS placeholder'))
                continue
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                continue    # vanished while listing
            if now - mtime < RECENT_S:
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

    order = {'NEW': 0, 'RUNNING': 1, 'KEEP': 2, 'BUSY': 3, 'FAILED': 4, 'OLD': 5}
    rows.sort(key=lambda r: (order[r[0]], r[1]))
    n_busy = sum(r[0] == 'BUSY' for r in rows)
    print(f'{"":8}{"file":<46}{"size":>7}  reason')
    print('-' * 100)
    for v, p, why in rows:
        if v == 'BUSY':
            continue    # summarised below, one line each would drown the list
        rel = os.path.relpath(p, ROOT)
        print(f'{v:<8}{rel:<46}{human(size_of(p)):>7}  {why}')
    if n_busy:
        print(f'BUSY    logs/.nfs*  ({n_busy} files)  NFS placeholders of deleted '
              f'files a process still holds; left alone')

    old = [p for v, p, _ in rows if v in ('OLD', 'FAILED')]
    total = sum(size_of(p) for p in old)
    print()
    print('  '.join(f'{k}={sum(r[0] == k for r in rows)}'
                    for k in ('NEW', 'RUNNING', 'KEEP', 'BUSY', 'FAILED', 'OLD'))
          + f'   -> to remove: {len(old)} ({human(total)})')
    if n_busy:
        print('  .nfs* holders:  lsof +D logs 2>/dev/null | grep .nfs   '
              '(or: fuser -v logs/.nfs*)')

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
