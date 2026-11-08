"""
Summarise the experiment logs: print the CONSOLE SUMMARY block of every run.

Usage
  python code/summarize_logs.py                 # all logs in logs/
  python code/summarize_logs.py logs/pq3_yelp.out
  python code/summarize_logs.py --grep yelp     # only logs whose name matches
  python code/summarize_logs.py --raw           # also print the PAIRED blocks only
"""

import os
import re
import argparse
import glob

PROJ = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PROJ)
LOG_DIR = os.path.join(ROOT, 'logs')

MARK = 'CONSOLE SUMMARY'


def extract(path, only_paired=False):
    """Return the summary text of one log (from MARK to end)."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    except OSError as e:
        return None, f'!! cannot read: {e}'

    if MARK not in text:
        # run still in progress or crashed before the summary
        tail = '\n'.join(text.strip().splitlines()[-3:])
        status = 'RUNNING/INCOMPLETE'
        if 'Traceback' in text:
            status = 'CRASHED'
        return None, f'--- {status}: no summary yet. Last lines:\n{tail}'

    block = text[text.index(MARK):]
    lines = [ln.rstrip() for ln in block.splitlines()]
    out = []
    for ln in lines:
        if only_paired and ln.startswith('PAIRED:'):
            out.append(ln)
            continue
        if only_paired and not out:
            continue
        if only_paired and ln.startswith('='):
            out.append(ln)
        elif only_paired:
            out.append(ln)
    return '\n'.join(lines), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', help='log files (default: all in logs/)')
    ap.add_argument('--grep', default=None, help='only logs whose name contains this')
    ap.add_argument('--raw', action='store_true',
                    help='print only the PAIRED comparison blocks')
    ap.add_argument('--out', default=None, help='also write the summary to this file')
    args = ap.parse_args()

    if args.files:
        paths = args.files
    else:
        paths = sorted(glob.glob(os.path.join(LOG_DIR, '*.out')))
    if args.grep:
        paths = [p for p in paths if args.grep in os.path.basename(p)]

    if not paths:
        print(f'no logs found in {LOG_DIR}')
        return

    chunks = []
    for p in paths:
        name = os.path.basename(p)
        print('\n' + '#' * 78)
        print(f'# {name}')
        print('#' * 78)
        body, err = extract(p, only_paired=args.raw)
        if err:
            print(err)
            chunks.append(f'### {name}\n{err}')
        else:
            print(body)
            chunks.append(f'### {name}\n{body}')

    # a compact cross-run table of the per-arm R@10
    print('\n' + '=' * 78)
    print('CROSS-RUN R@10 (per arm, mean+-std as printed)')
    print('=' * 78)
    for p in paths:
        name = os.path.basename(p)
        body, err = extract(p)
        if err:
            continue
        for ln in body.splitlines():
            m = re.match(r'^(PROPOSED\S*|RANDOM\w*|SISA|GLOBAL)\s+([\d.]+\+?-?[\d.]*)\s+'
                         r'([\d.]+\+?-?[\d.]*)\s+([\d.]+\+?-?[\d.]*)\s+([\d.]+\+?-?[\d.]*)\s*$',
                         ln.strip())
            if m:
                print(f'  {name:<38} {m.group(1):<16} '
                      f'R@10={m.group(2):<18} N@10={m.group(3):<18} '
                      f'R@20={m.group(4):<18} N@20={m.group(5)}')

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(chunks))
        print(f'\nwritten -> {args.out}')


if __name__ == '__main__':
    main()
