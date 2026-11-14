"""
Side by side: the same experiment (main, 1% random deletion) at 50 epochs
(main_<ds>) and at more epochs (main_<ds>_e100, main_<ds>_e200, ...).

For every arm and every number of epochs: R/NDCG@10/20/50 (mean+-std over the
deletions), the unlearn time, and the change against 50 epochs. Then, per
number of epochs, SAFER A and SAFER B against Random: the difference and
whether it is larger than the spread of the runs.

  python code/compare_epochs.py --dataset yelp
"""

import argparse
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_partition_logs import parse_log   # noqa: E402

LOGS = os.path.join(os.path.dirname(HERE), 'logs')
LABEL = {'PROPOSED': 'SAFER A', 'PROPOSED_BAL': 'SAFER B', 'SISA': 'SISA',
         'RANDOM_PERM': 'Random', 'RANDOM_EMB': 'RandomEmb',
         'RECERASER': 'RecEraser', 'GLOBAL': 'Retrain'}
ORDER = ['PROPOSED', 'PROPOSED_BAL', 'SISA', 'RANDOM_PERM', 'RANDOM_EMB',
         'RECERASER', 'GLOBAL']
UTIL = [('recall_10', 'R@10'), ('ndcg_10', 'N@10'), ('recall_20', 'R@20'),
        ('ndcg_20', 'N@20'), ('recall_50', 'R@50'), ('ndcg_50', 'N@50')]


def ms(v, p=4):
    v = [x for x in v if x is not None]
    if not v:
        return '-'
    sd = st.stdev(v) if len(v) > 1 else 0.0
    return f'{st.mean(v):.{p}f}+-{sd:.{p}f}'


def mean(v):
    v = [x for x in v if x is not None]
    return st.mean(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='yelp')
    args = ap.parse_args()

    runs = {}                                   # epochs -> records
    base = os.path.join(LOGS, f'main_{args.dataset}.out')
    if os.path.exists(base):
        runs[50] = parse_log(base)
    for f in sorted(os.listdir(LOGS)) if os.path.isdir(LOGS) else []:
        m = re.fullmatch(rf'main_{re.escape(args.dataset)}_e(\d+)\.out', f)
        if m:
            runs[int(m.group(1))] = parse_log(os.path.join(LOGS, f))
    runs = {e: r for e, r in runs.items() if r}
    if not runs:
        print(f'no main_{args.dataset} logs with finished runs yet')
        return
    epochs = sorted(runs)
    present = {r['arm'] for rs in runs.values() for r in rs}
    arms = [a for a in ORDER if a in present]

    print('=' * 118)
    print(f'EPOCHS: main_{args.dataset} (1% random deletion, K=8, dim=64)  -  '
          f'epochs {", ".join(map(str, epochs))}')
    print('=' * 118)
    for a in arms:
        print(f'\n  {LABEL.get(a, a)}')
        print(f'  {"epochs":>8}{"n":>4}' + ''.join(f'{lab:>16}' for _, lab in UTIL)
              + f'{"unlearn(s)":>12}{"dR@20 vs 50":>13}')
        b20 = mean([r.get('recall_20') for r in runs.get(50, []) if r['arm'] == a])
        for e in epochs:
            rs = [r for r in runs[e] if r['arm'] == a]
            if not rs:
                print(f'  {e:>8}{0:>4}   (not run yet)')
                continue
            r20 = mean([r.get('recall_20') for r in rs])
            ul = mean([r.get('unlearn_time') for r in rs])
            d = (f'{(r20 - b20) / b20 * 100:+.1f}%' if b20 and r20 is not None
                 and e != 50 else '')
            print(f'  {e:>8}{len(rs):>4}'
                  + ''.join(f'{ms([r.get(k) for r in rs]):>16}' for k, _ in UTIL)
                  + f'{(f"{ul:.1f}" if ul is not None else "-"):>12}{d:>13}')

    # SAFER against Random at each number of epochs
    print(f'\n  SAFER vs Random (R@20, NDCG@20): difference of the means; '
          f'"within noise" when it is smaller than the larger std')
    for e in epochs:
        rnd = [r for r in runs[e] if r['arm'] == 'RANDOM_PERM']
        for a in ('PROPOSED', 'PROPOSED_BAL'):
            rs = [r for r in runs[e] if r['arm'] == a]
            if not rs or not rnd:
                continue
            out = []
            for k, lab in (('recall_20', 'R@20'), ('ndcg_20', 'N@20')):
                x = [r[k] for r in rs if r.get(k) is not None]
                y = [r[k] for r in rnd if r.get(k) is not None]
                if not x or not y:
                    continue
                d = st.mean(x) - st.mean(y)
                sd = max(st.stdev(x) if len(x) > 1 else 0.0,
                         st.stdev(y) if len(y) > 1 else 0.0)
                verdict = ('within noise' if abs(d) <= sd or len(x) < 2 or len(y) < 2
                           else ('SAFER higher' if d > 0 else 'Random higher'))
                out.append(f'{lab} {d:+.4f} ({d / st.mean(y) * 100:+.1f}%, {verdict})')
            print(f'    {e:>4} epochs  {LABEL[a]:<8} - Random: ' + '   '.join(out)
                  + f'   [n={len(rs)} vs {len(rnd)}]')


if __name__ == '__main__':
    main()
