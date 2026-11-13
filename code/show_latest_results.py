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
from summarize_partition_logs import parse_log   # noqa: E402
from export_single_request_csv import parse as parse_single   # noqa: E402

ROOT = cor.ROOT
LOGS = cor.LOGS
SHARED = 10          # % background GPU load above which a run counts as shared
# the names of Table 2 in the paper; the logs keep the internal arm names
LABEL = {'PROPOSED': 'SAFER A', 'PROPOSED_BAL': 'SAFER B', 'SISA': 'SISA',
         'RANDOM_PERM': 'Random', 'RANDOM_EMB': 'RandomEmb',
         'RECERASER': 'RecEraser', 'GLOBAL': 'Retrain'}


def label(arm):
    return LABEL.get(arm, arm)
UTIL = (('recall_10', 'R@10'), ('ndcg_10', 'N@10'), ('recall_20', 'R@20'), ('ndcg_20', 'N@20'), ('recall_50', 'R@50'), ('ndcg_50', 'N@50'))
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


def timed(rs, engine):
    """The runs whose seconds count: one engine only, since the GPU engine
    and the original loops run at different speeds."""
    return [r for r in rs if r.get('engine', 'python') == engine]


def print_pq(recs, brief):
    arms = arms_of(recs)
    engines = {r.get('engine', 'python') for r in recs}
    eng = 'gpu' if 'gpu' in engines else 'python'
    if brief:
        # means only, every cut-off; the full view adds the std, partition,
        # load and instability
        print(f'  {"arm":<14}{"n":>3}' + ''.join(f'{lab:>8}' for _, lab in UTIL)
              + f'{"unlearn(s)":>12}{"work(M)":>9}')
        for a in arms:
            rs = [r for r in recs if r['arm'] == a]
            wk = [r['unlearn_samples'] / 1e6 for r in rs if r.get('unlearn_samples') is not None]
            ul = [r.get('unlearn_time') for r in timed(rs, eng) if r.get('unlearn_time') is not None]
            print(f'  {label(a):<14}{len(rs):>3}'
                  + ''.join(f'{mean([r.get(k) for r in rs], 4):>8}' for k, _ in UTIL)
                  + f'{(f"{st.mean(ul):.1f}" if ul else "-"):>12}'
                  + f'{(f"{st.mean(wk):.2f}" if wk else "-"):>9}')
        if len(engines) > 1:
            print(f'  unlearn(s): {eng}-engine runs only; utility and work use every run')
        return

    print(f'  -- utility (mean+-std over seeds)')
    print(f'  {"arm":<14}{"n":>3}{"R@10":>16}{"NDCG@10":>16}{"R@20":>16}'
          f'{"NDCG@20":>16}{"R@50":>16}{"NDCG@50":>16}')
    for a in arms:
        rs = [r for r in recs if r['arm'] == a]
        print(f'  {label(a):<14}{len(rs):>3}' + ''.join(
            f'{ms([r.get(k) for r in rs]):>16}'
            for k in ('recall_10', 'ndcg_10', 'recall_20', 'ndcg_20',
                      'recall_50', 'ndcg_50')))

    print(f'  -- partition quality / load / instability (mean)')
    print(f'  {"arm":<14}{"IntraSim":>10}{"NR@10":>8}{"NR@20":>8}{"NR@50":>8}'
          f'{"user_cv":>9}{"inter_cv":>9}{"max/mean":>9}{"instab%":>9}{"pair%":>8}')
    for a in arms:
        rs = [r for r in recs if r['arm'] == a]
        g = lambda k, p=4: mean([r.get(k) for r in rs], p)
        print(f'  {label(a):<14}{g("intra_similarity"):>10}{g("nr_10", 3):>8}'
              f'{g("nr_20", 3):>8}{g("nr_50", 3):>8}{g("user_load_cv", 3):>9}'
              f'{g("interaction_load_cv", 3):>9}{g("max_mean_ratio", 2):>9}'
              f'{g("instability_label"):>9}{g("instability_pairwise"):>8}')

    print(f'  -- cost: seconds (mean+-std), affected shards, unlearn work, GPU background load')
    print(f'  {"arm":<14}{"train":>14}{"unlearn":>14}{"parallel":>14}{"shards":>11}'
          f'{"work (M samples)":>18}{"bg% max":>9}')
    shared_runs = 0
    for a in arms:
        rs = [r for r in recs if r['arm'] == a]
        work = [r['unlearn_samples'] / 1e6 for r in rs if r.get('unlearn_samples') is not None]
        bg = [max(r.get('bg_load_before') or 0, r.get('bg_load_after') or 0) for r in rs
              if r.get('bg_load_before') is not None or r.get('bg_load_after') is not None]
        n_sh = sum(v >= SHARED for v in bg)
        shared_runs += n_sh
        flag = f'{max(bg):.0f}' + ('!' if n_sh else '') if bg else '-'
        tr = timed(rs, eng)
        print(f'  {label(a):<14}{ms([r.get("train_time") for r in tr], 1):>14}'
              f'{ms([r.get("unlearn_time") for r in tr], 1):>14}'
              f'{ms([r.get("parallel_time") for r in tr], 1):>14}'
              f'{ms([r.get("affected_shards") for r in rs], 1):>11}'
              f'{ms(work, 2):>18}{flag:>9}')
    if shared_runs:
        print(f'  ! {shared_runs} run(s) measured while other work used the GPU '
              f'(>= {SHARED}%): their seconds may be inflated; the work column is not')
    if eng == 'gpu' and 'RECERASER' in arms:
        print('  RecEraser trains on the original loops even under the GPU engine, so its '
              'seconds are not comparable with the other arms; compare the work column')
    if len(engines) > 1:
        n_old = sum(1 for r in recs if r.get('engine', 'python') != eng)
        print(f'  seconds: {eng}-engine runs only; {n_old} run(s) of the original loops '
              f'are left out of them (utility and work use every run)')
    if any(r.get('work_derived') for r in recs):
        print('  work of the older runs is derived from their per-shard interaction counts x 50 epochs')


def print_single(rows):
    print(f'  {"arm":<14}{"mean(s)":>10}{"std":>9}{"min":>9}{"max":>9}{"n":>5}{"x Retrain":>11}')
    order = {a: i for i, a in enumerate(ARM_ORDER)}
    for r in sorted(rows, key=lambda r: order.get(r['arm'], 99)):
        print(f'  {label(r["arm"]):<14}{r["time_mean_s"]:>10}{r["time_std_s"]:>9}'
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


def print_partial(name, path, text, n_arms, brief=False):
    """Tables of the runs that have finished in a log still running."""
    done, total, cur = progress(name, text, n_arms)
    t0 = started_at(name)
    head = f'{done}/{total} runs done, now {cur}'
    # runs finished since this START: a resumed log also holds the runs kept
    # from before, which took none of this time
    new = done
    if '[resume]' in text:
        after = text[text.rindex('[resume]'):]
        new = (len(SINGLE_DONE.findall(after)) if name.startswith('single_')
               else len(cor.RUN_LINE.findall(after)))
        head += f' ({done - new} kept from before the resume)'
    if t0 and done and not new:
        head += f' | resumed {ago(t0)} ago, no new run finished yet'
    elif t0 and new:
        live = cor.is_live(path)
        # a stopped log ended at its last write, not now
        end = time.time() if live else os.path.getmtime(path)
        per = (end - t0) / new
        if live:
            head += (f' | running {ago(t0)}, ~{per / 60:.0f} min/run, '
                     f'~{(total - done) * per / 3600:.1f} h left for this log')
        else:
            head += f' | ran {(end - t0) / 3600:.1f} h, ~{per / 60:.0f} min/run before it stopped'
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
            print(f'  {label(a):<14}{len(rows[a]):>6}{st.mean(rows[a]):>10.2f}')
        return
    print_pq(parse_log(path), brief=brief)
    print('  (n = seeds finished so far)')


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
        print_partial(name, p, text, n_arms, args.brief)


if __name__ == '__main__':
    main()
