"""
Print the latest results in one compact view, from the CSVs that
`bash code/run_blocks.sh csv` writes to results/csv/.

  python code/show_latest_results.py
"""

import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(ROOT, 'results', 'csv')
GROUPS = ('main', 'modes', 'shards', 'dims', 'recpart')


def pm(row, key, prec=4):
    m, s = row.get(key + '_mean', ''), row.get(key + '_std', '')
    if m == '':
        return '-'
    return f'{float(m):.{prec}f}+-{float(s):.{prec}f}' if s != '' else f'{float(m):.{prec}f}'


def val(row, key, prec=2):
    v = row.get(key, '')
    return f'{float(v):.{prec}f}' if v not in ('', None) else '-'


def main():
    if not os.path.isdir(CSV_DIR):
        print(f'no {CSV_DIR} - run: bash code/run_blocks.sh csv')
        return

    for g in GROUPS:
        path = os.path.join(CSV_DIR, f'{g}_mean.csv')
        if not os.path.exists(path):
            continue
        rows = list(csv.DictReader(open(path, encoding='utf-8')))
        if not rows:
            continue
        print()
        print('=' * 118)
        print(f'{g.upper()}   ({os.path.relpath(path, ROOT)})')
        print('=' * 118)
        print(f'{"log":<22}{"arm":<15}{"n":>3}{"R@10":>17}{"NDCG@10":>17}'
              f'{"unlearn(s)":>15}{"parallel(s)":>14}{"instab%":>9}'
              f'{"dR@10 vs RP":>13}{"p":>8}')
        print('-' * 118)
        last = None
        for r in rows:
            if last is not None and r['log'] != last:
                print()
            last = r['log']
            print(f'{r["log"]:<22}{r["arm"]:<15}{r.get("n_seeds", ""):>3}'
                  f'{pm(r, "recall_10"):>17}{pm(r, "ndcg_10"):>17}'
                  f'{pm(r, "unlearn_time", 1):>15}{pm(r, "parallel_time", 1):>14}'
                  f'{val(r, "instability_label"):>9}'
                  f'{r.get("delta_R10_vs_RANDOM_PERM", "") or "-":>13}'
                  f'{r.get("delta_R10_p", "") or "-":>8}')

    sp = os.path.join(CSV_DIR, 'single.csv')
    if os.path.exists(sp):
        rows = list(csv.DictReader(open(sp, encoding='utf-8')))
        if rows:
            print()
            print('=' * 118)
            print(f'SINGLE REQUEST   ({os.path.relpath(sp, ROOT)})')
            print('=' * 118)
            print(f'{"log":<14}{"arm":<18}{"mean(s)":>12}{"std":>10}{"min":>10}{"max":>10}'
                  f'{"n":>5}{"x Retrain":>11}')
            print('-' * 118)
            for r in rows:
                print(f'{r["log"]:<14}{r["arm"]:<18}{r["time_mean_s"]:>12}{r["time_std_s"]:>10}'
                      f'{r["time_min_s"]:>10}{r["time_max_s"]:>10}'
                      f'{r["n_requests"]:>5}{r["ratio_to_global"] or "-":>11}')

    print()
    print('n < 5: the run has not finished all 5 seeds. '
          'Full tables: results/csv/<block>_tables.txt')


if __name__ == '__main__':
    main()
