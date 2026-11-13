"""
Print the results of the run_blocks.sh experiments straight from the logs.

The finished experiments come first, in the order they finished, each with
its full tables. The ones that are not finished follow, with a notice:
how far a running one has got, which ones have not started, and the last
line of the ones that stopped.

  python code/show_latest_results.py            # everything
  python code/show_latest_results.py --brief    # R@10 / NDCG@10 / time only
"""

import argparse
import os
import re
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clean_old_results as cor                       # noqa: E402
from summarize_partition_logs import parse_log, paired_test   # noqa: E402
from export_single_request_csv import parse as parse_single   # noqa: E402

ROOT = cor.ROOT
LOGS = cor.LOGS
BASELINE = 'RANDOM_PERM'
ARM_ORDER = ['PROPOSED', 'PROPOSED_BAL', 'SISA', 'RANDOM_PERM', 'RANDOM_EMB',
             'RECERASER', 'GLOBAL']
N_SEEDS = 5
SINGLE_DONE = re.compile(r'^\s*\[([A-Z_]+)\]\s+single-request unlearn time', re.M)


def describe(name):
    """log name -> (dataset, setting, number of arms)"""
    stem, ds = name.rsplit('_', 1)
    ds = {'ml1m': 'ml-1m', 'yelp': 'yelp2018'}.get(ds, ds)
    base = {'k': 8, 'dim': 64, 'mode': 'random'}
    n_arms = 7
    if stem == 'main':
        pass
    elif stem.startswith('mode_'):
        base['mode'] = stem[5:]
    elif stem == 'single':
        base['mode'] = 'single request (20 users)'
    elif stem.startswith('k'):
        base['k'] = int(stem[1:])
    elif stem.startswith('dim'):
        base['dim'] = int(stem[3:])
    elif stem.startswith('rec'):
        base['partition'] = {'recinp': 'InP', 'recrand': 'random'}[stem]
        n_arms = 1
    setting = '  '.join(f'{k}={v}' for k, v in base.items())
    return ds, setting, n_arms


def ms(vals, prec=4):
    vals = [v for v in vals if v is not None]
    if not vals:
        return '-'
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f'{st.mean(vals):.{prec}f}+-{sd:.{prec}f}'


def mean(vals, prec=2):
    vals = [v for v in vals if v is not None]
    return f'{st.mean(vals):.{prec}f}' if vals else '-'


def ago(t):
    s = int(time.time() - t)
    if s < 90:
        return f'{s}s'
    if s < 5400:
        return f'{s // 60}m'
    return f'{s // 3600}h{(s % 3600) // 60:02d}m'


def arms_of(recs):
    present = {r['arm'] for r in recs}
    return [a for a in ARM_ORDER if a in present] + sorted(present - set(ARM_ORDER))


def print_pq(recs, brief):
    arms = arms_of(recs)
    if brief:
        print(f'  {"arm":<14}{"n":>3}{"R@10":>17}{"NDCG@10":>17}{"unlearn(s)":>15}'
              f'{"dR@10 vs RP":>13}{"p":>8}')
        for a in arms:
            rs = [r for r in recs if r['arm'] == a]
            pr = paired_test(recs, a, BASELINE, 'recall_10') if a != BASELINE else None
            d = f'{pr["mean_delta"]:+.4f}' if pr else '-'
            p = f'{pr["p"]:.4f}' if pr and pr['p'] == pr['p'] else '-'
            print(f'  {a:<14}{len(rs):>3}{ms([r.get("recall_10") for r in rs]):>17}'
                  f'{ms([r.get("ndcg_10") for r in rs]):>17}'
                  f'{ms([r.get("unlearn_time") for r in rs], 1):>15}{d:>13}{p:>8}')
        return

    print(f'  -- utility (mean+-std over seeds)')
    print(f'  {"arm":<14}{"n":>3}{"R@10":>16}{"NDCG@10":>16}{"R@20":>16}'
          f'{"NDCG@20":>16}{"R@50":>16}{"NDCG@50":>16}')
    for a in arms:
        rs = [r for r in recs if r['arm'] == a]
        print(f'  {a:<14}{len(rs):>3}' + ''.join(
            f'{ms([r.get(k) for r in rs]):>16}'
            for k in ('recall_10', 'ndcg_10', 'recall_20', 'ndcg_20',
                      'recall_50', 'ndcg_50')))

    print(f'  -- partition quality / load / instability (mean)')
    print(f'  {"arm":<14}{"IntraSim":>10}{"NR@10":>8}{"NR@20":>8}{"NR@50":>8}'
          f'{"user_cv":>9}{"inter_cv":>9}{"max/mean":>9}{"instab%":>9}{"pair%":>8}')
    for a in arms:
        rs = [r for r in recs if r['arm'] == a]
        g = lambda k, p=4: mean([r.get(k) for r in rs], p)
        print(f'  {a:<14}{g("intra_similarity"):>10}{g("nr_10", 3):>8}'
              f'{g("nr_20", 3):>8}{g("nr_50", 3):>8}{g("user_load_cv", 3):>9}'
              f'{g("interaction_load_cv", 3):>9}{g("max_mean_ratio", 2):>9}'
              f'{g("instability_label"):>9}{g("instability_pairwise"):>8}')

    print(f'  -- cost (s, mean+-std) and affected shards')
    print(f'  {"arm":<14}{"train":>16}{"unlearn":>16}{"parallel":>16}{"shards":>12}')
    for a in arms:
        rs = [r for r in recs if r['arm'] == a]
        print(f'  {a:<14}{ms([r.get("train_time") for r in rs], 1):>16}'
              f'{ms([r.get("unlearn_time") for r in rs], 1):>16}'
              f'{ms([r.get("parallel_time") for r in rs], 1):>16}'
              f'{ms([r.get("affected_shards") for r in rs], 1):>12}')

    if BASELINE in arms and len(arms) > 1:
        print(f'  -- paired vs {BASELINE} (same seeds; * = p < 0.05)')
        for a in arms:
            if a == BASELINE:
                continue
            parts = []
            for k, lab in (('recall_10', 'R@10'), ('ndcg_10', 'NDCG@10'),
                           ('unlearn_time', 'unlearn')):
                pr = paired_test(recs, a, BASELINE, k)
                if not pr:
                    continue
                star = ' *' if pr['p'] == pr['p'] and pr['p'] < 0.05 else ''
                p = f'p={pr["p"]:.4f}' if pr['p'] == pr['p'] else 'p=n/a'
                parts.append(f'{lab} {pr["mean_delta"]:+.4f} ({p}{star})')
            print(f'  {a:<14}' + '   '.join(parts))


def print_single(rows):
    print(f'  {"arm":<14}{"mean(s)":>10}{"std":>9}{"min":>9}{"max":>9}{"n":>5}{"x Retrain":>11}')
    order = {a: i for i, a in enumerate(ARM_ORDER)}
    for r in sorted(rows, key=lambda r: order.get(r['arm'], 99)):
        print(f'  {r["arm"]:<14}{r["time_mean_s"]:>10}{r["time_std_s"]:>9}'
              f'{r["time_min_s"]:>9}{r["time_max_s"]:>9}{r["n_requests"]:>5}'
              f'{r["ratio_to_global"] or "-":>11}')


def progress(name, text, n_arms):
    """-> (done runs, total runs, current arm/seed)"""
    total = n_arms * N_SEEDS
    if name.startswith('single_'):
        done = len(SINGLE_DONE.findall(text))
    else:
        done = len(cor.RUN_LINE.findall(text))
    hdr = cor.ARM_HDR.findall(text)
    seeds = re.findall(r'---\s*ARM\s+[A-Z_]+\s*\(seed\s+(\d+)\)', text)
    cur = f'{hdr[-1]} seed {seeds[-1]}' if hdr else 'starting'
    return done, total, cur


SINGLE_LINE = re.compile(
    r'^\s*\[([A-Z_]+)\]\s+single-request unlearn time:\s+([\d.]+)\+-([\d.]+)s over (\d+)',
    re.M)


def started_at(name):
    """Epoch at which run_blocks.sh started this log, from the START banner in
    logs/block_*.out ("START <name>   (HH:MM:SS)"); None if not found."""
    best = None
    for f in sorted(os.listdir(LOGS)):
        if not (f.startswith('block_') and f.endswith('.out')):
            continue
        try:
            txt = open(os.path.join(LOGS, f), encoding='utf-8', errors='replace').read()
        except OSError:
            continue
        for hms in re.findall(r'START ' + re.escape(name) + r'\s+\((\d\d:\d\d:\d\d)\)', txt):
            mt = os.path.getmtime(os.path.join(LOGS, name + '.out'))
            day = time.strftime('%Y-%m-%d', time.localtime(mt))
            t = time.mktime(time.strptime(f'{day} {hms}', '%Y-%m-%d %H:%M:%S'))
            while t > mt + 60:          # the banner came on an earlier day
                t -= 86400
            best = t
    return best


def print_partial(name, path, text, n_arms):
    """Tables of the runs that have finished in a log still running."""
    done, total, cur = progress(name, text, n_arms)
    t0 = started_at(name)
    head = f'{done}/{total} runs done, now {cur}'
    if t0 and done:
        el = time.time() - t0
        per = el / done
        head += (f' | running {ago(t0)}, ~{per / 60:.0f} min/run, '
                 f'~{(total - done) * per / 3600:.1f} h left for this log')
    print()
    print('-' * 112)
    print(f'PARTIAL  {name}   {head}')
    print('-' * 112)
    if not done:
        print('  (no run finished yet)')
        return
    if name.startswith('single_'):
        rows = {}
        for arm, m, sd, n in SINGLE_LINE.findall(text):
            rows.setdefault(arm, []).append(float(m))
        print(f'  {"arm":<14}{"seeds":>6}{"mean(s)":>10}')
        for a in arms_of([{'arm': k} for k in rows]):
            print(f'  {a:<14}{len(rows[a]):>6}{st.mean(rows[a]):>10.2f}')
        return
    print_pq(parse_log(path), brief=True)
    print('  (n = seeds finished so far; the paired p-value needs 3 seeds)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--brief', action='store_true')
    args = ap.parse_args()

    done, waiting, partial = [], [], []
    for name in cor.EXPERIMENT_LOGS:
        p = os.path.join(LOGS, name + '.out')
        if not os.path.exists(p):
            waiting.append((name, 'PENDING', 'not started yet'))
            continue
        text = open(p, encoding='utf-8', errors='replace').read()
        v, why = cor.classify_log(p)
        finished = 'CONSOLE SUMMARY' in text
        ds, setting, n_arms = describe(name)
        if v == 'NEW' and finished:
            done.append((os.path.getmtime(p), name, p))
        elif v == 'OLD':
            waiting.append((name, 'OLD', why + ' - the block will run it again'))
        elif cor.is_live(p):
            d, t, cur = progress(name, text, n_arms)
            partial.append((name, p, text, n_arms))
            waiting.append((name, 'RUNNING',
                            f'{d}/{t} runs done, now {cur}, last write {ago(os.path.getmtime(p))} ago'))
        else:
            d, t, cur = progress(name, text, n_arms)
            last = cor.last_error(text)
            if d:
                partial.append((name, p, text, n_arms))
            waiting.append((name, 'FAILED',
                            f'stopped at {d}/{t} runs ({cur}): {last}'))

    done.sort()
    print(f'{len(done)}/{len(cor.EXPERIMENT_LOGS)} experiments finished'
          + (f', printed in the order they finished' if done else ''))
    for i, (t, name, p) in enumerate(done, 1):
        ds, setting, _ = describe(name)
        print()
        print('=' * 112)
        print(f'[{i}] {name}   {ds}   {setting}   finished '
              f'{time.strftime("%Y-%m-%d %H:%M", time.localtime(t))}')
        print('=' * 112)
        if name.startswith('single_'):
            rows = parse_single(p) or []
            print_single(rows)
        else:
            print_pq(parse_log(p), args.brief)

    if waiting:
        print()
        print('=' * 112)
        print(f'NOT FINISHED ({len(waiting)})')
        print('=' * 112)
        for name, v, why in waiting:
            print(f'  {name:<18}{v:<9}{why}')

    for name, p, text, n_arms in partial:
        print_partial(name, p, text, n_arms)


if __name__ == '__main__':
    main()
