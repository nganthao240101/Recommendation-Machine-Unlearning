"""
Summarise exp_partition_quality.py logs into the tables the paper needs.

Reads the per-run lines the experiment prints and rebuilds, per (log, arm):
  - utility: Recall/NDCG @10/20/50  (mean +- std over seeds)
  - cost:    train_time, unlearn_time, affected_shards
  - partition quality: IntraSim, NR@10/20/50
  - load balance: user_cv, inter_cv, max/mean
and the paired PROPOSED-vs-baseline comparison with a paired t-test on the
per-seed differences.

Per-seed records are kept, so the pairing is exact: a seed contributes to the
comparison only if BOTH arms ran on that seed.

Usage:
  python code/summarize_partition_logs.py logs/ex1_ml1m_random.out
  python code/summarize_partition_logs.py logs/*.out --baseline RANDOM_PERM
  python code/summarize_partition_logs.py logs/ex1_ml1m_random.out --json out.json
"""

import os
import re
import json
import glob
import argparse
import statistics as st
from collections import defaultdict


KS = (10, 20, 50)

ARM_HDR = re.compile(r'---\s*ARM\s+([A-Z_]+)\s*\(seed\s+(\d+)\)')
ARM_LINE = re.compile(r'^\s*\[([A-Z_]+)\]\s+R@10=')

RE_UTIL = {
    'recall_10':  re.compile(r'R@10=([\d.]+)'),
    'ndcg_10':    re.compile(r'NDCG@10=([\d.]+)'),
    'recall_20':  re.compile(r'R@20=([\d.]+)'),
    'ndcg_20':    re.compile(r'NDCG@20=([\d.]+)'),
    'recall_50':  re.compile(r'R@50=([\d.]+)'),
    'ndcg_50':    re.compile(r'NDCG@50=([\d.]+)'),
}
RE_COST = {
    'train_time':      re.compile(r'train=([\d.]+)s'),
    'unlearn_time':    re.compile(r'unlearn=([\d.]+)s'),
    'affected_shards': re.compile(r'shards=(\d+)'),
}
RE_QUAL = {
    'intra_similarity': re.compile(r'IntraSim=([\d.]+)'),
    'nr_10':            re.compile(r'NR@10=([\d.]+)'),
    'nr_20':            re.compile(r'NR@20=([\d.]+)'),
    'nr_50':            re.compile(r'NR@50=([\d.]+)'),
}
RE_LOAD = {
    'user_load_cv':         re.compile(r'user_cv=([\d.]+)'),
    'interaction_load_cv':   re.compile(r'inter_cv=([\d.]+)'),
    'max_mean_ratio':       re.compile(r'max/mean=([\d.]+)'),
}


def parse_log(path):
    """-> list of per-run records."""
    recs = []
    with open(path, encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()

    cur_seed = None
    cur_arm = None
    pending = None          # the record the next 2 continuation lines belong to

    for ln in lines:
        m = ARM_HDR.search(ln)
        if m:
            cur_arm, cur_seed = m.group(1), int(m.group(2))
            continue

        m = ARM_LINE.match(ln)
        if m:
            arm = m.group(1)
            rec = {'arm': arm, 'seed': cur_seed if cur_seed is not None else -1}
            for k, rx in RE_UTIL.items():
                mm = rx.search(ln)
                if mm:
                    rec[k] = float(mm.group(1))
            for k, rx in RE_COST.items():
                mm = rx.search(ln)
                if mm:
                    rec[k] = float(mm.group(1))
            recs.append(rec)
            pending = rec
            continue

        if pending is not None and 'partition quality' in ln:
            for k, rx in RE_QUAL.items():
                mm = rx.search(ln)
                if mm:
                    pending[k] = float(mm.group(1))
            continue
        if pending is not None and 'load:' in ln:
            for k, rx in RE_LOAD.items():
                mm = rx.search(ln)
                if mm:
                    pending[k] = float(mm.group(1))
            pending = None
            continue
        pending = None

    return recs


def ms(vals, width=16, prec=4):
    vals = [v for v in vals if v is not None]
    if not vals:
        return ' ' * width
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return f'{m:.{prec}f}+-{s:.{prec}f}'.rjust(width)


def raw(vals, width=16, prec=4):
    vals = [v for v in vals if v is not None]
    if not vals:
        return ' ' * width
    return f'{st.mean(vals):.{prec}f}'.rjust(width)


def paired_test(recs, arm_a, arm_b, key):
    """Paired comparison on per-seed differences."""
    by = defaultdict(dict)
    for r in recs:
        if key in r:
            by[r['arm']][r['seed']] = r[key]
    if arm_a not in by or arm_b not in by:
        return None
    seeds = sorted(set(by[arm_a]) & set(by[arm_b]))
    if len(seeds) < 2:
        return None
    a = [by[arm_a][s] for s in seeds]
    b = [by[arm_b][s] for s in seeds]
    d = [x - y for x, y in zip(a, b)]
    md = st.mean(d)
    sd = st.stdev(d) if len(d) > 1 else 0.0
    rel = st.mean([(x - y) / y * 100 for x, y in zip(a, b)]) if all(y != 0 for y in b) else float('nan')
    p = float('nan')
    if len(d) >= 3:
        try:
            from scipy import stats as _st
            p = float(_st.ttest_1samp(d, 0.0).pvalue)
        except Exception:
            p = float('nan')
    return {'n': len(seeds), 'seeds': seeds, 'a': st.mean(a), 'b': st.mean(b),
            'mean_delta': md, 'sd_delta': sd, 'rel_gain': rel, 'p': p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--baseline', default='RANDOM_PERM')
    ap.add_argument('--proposed', default='PROPOSED')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    paths = []
    for p in args.logs:
        paths.extend(sorted(glob.glob(p)) if any(ch in p for ch in '*?[') else [p])

    all_out = {}
    for path in paths:
        if not os.path.exists(path):
            print(f'[skip] {path} (not found)')
            continue
        recs = parse_log(path)
        if not recs:
            print(f'[skip] {path} (no parsed runs)')
            continue

        arms = sorted({r['arm'] for r in recs})
        all_out[path] = recs

        print()
        print('=' * 116)
        print(f'{path}   ({len(recs)} runs)')
        print('=' * 116)

        # ---- utility ----
        print(f'\n{"arm":<16}{"n":>4}{"R@10":>17}{"NDCG@10":>17}{"R@20":>15}{"N@20":>15}'
              f'{"R@50":>15}{"N@50":>15}')
        print('-' * 116)
        for a in arms:
            rs = [r for r in recs if r['arm'] == a]
            print(f'{a:<16}{len(rs):>4}'
                  f'{ms([r.get("recall_10") for r in rs]):>17}'
                  f'{ms([r.get("ndcg_10") for r in rs]):>17}'
                  f'{ms([r.get("recall_20") for r in rs]):>15}'
                  f'{ms([r.get("ndcg_20") for r in rs]):>15}'
                  f'{ms([r.get("recall_50") for r in rs]):>15}'
                  f'{ms([r.get("ndcg_50") for r in rs]):>15}')

        # ---- partition quality + load ----
        print(f'\n{"arm":<16}{"IntraSim":>13}{"NR@10":>13}{"NR@20":>13}{"NR@50":>13}'
              f'{"user_cv":>12}{"inter_cv":>12}{"max/mean":>12}')
        print('-' * 116)
        for a in arms:
            rs = [r for r in recs if r['arm'] == a]
            print(f'{a:<16}'
                  f'{raw([r.get("intra_similarity") for r in rs], 13):>13}'
                  f'{raw([r.get("nr_10") for r in rs], 13):>13}'
                  f'{raw([r.get("nr_20") for r in rs], 13):>13}'
                  f'{raw([r.get("nr_50") for r in rs], 13):>13}'
                  f'{raw([r.get("user_load_cv") for r in rs], 12):>12}'
                  f'{raw([r.get("interaction_load_cv") for r in rs], 12):>12}'
                  f'{raw([r.get("max_mean_ratio") for r in rs], 12):>12}')

        # ---- cost ----
        print(f'\n{"arm":<16}{"train(s)":>17}{"unlearn(s)":>17}{"affected shards":>18}')
        print('-' * 116)
        for a in arms:
            rs = [r for r in recs if r['arm'] == a]
            print(f'{a:<16}'
                  f'{ms([r.get("train_time") for r in rs], 17, 1):>17}'
                  f'{ms([r.get("unlearn_time") for r in rs], 17, 1):>17}'
                  f'{ms([r.get("affected_shards") for r in rs], 18, 2):>18}')

        # ---- paired comparison ----
        print(f'\nPAIRED: {args.proposed} vs {args.baseline}')
        for key in ('recall_10', 'ndcg_10', 'recall_20', 'ndcg_20',
                    'recall_50', 'ndcg_50', 'unlearn_time', 'affected_shards'):
            pr = paired_test(recs, args.proposed, args.baseline, key)
            if not pr:
                continue
            ptag = '' if pr['p'] != pr['p'] else f"  p={pr['p']:.4f}" + (' *' if pr['p'] < 0.05 else '')
            print(f'  {key:<16}{args.proposed}={pr["a"]:.4f}  {args.baseline}={pr["b"]:.4f}  '
                  f'rel_gain={pr["rel_gain"]:+.2f}%  mean_delta={pr["mean_delta"]:+.4f} '
                  f'(sd={pr["sd_delta"]:.4f})  n={pr["n"]}{ptag}')

        # ---- every other arm vs the baseline, for context ----
        for a in arms:
            if a == args.baseline:
                continue
            pr = paired_test(recs, a, args.baseline, 'recall_10')
            if not pr:
                continue
            pn = paired_test(recs, a, args.baseline, 'ndcg_10')
            extra = ''
            if pn:
                extra = (f"   NDCG: {pn['a']:.4f} vs {pn['b']:.4f} "
                         f"delta={pn['mean_delta']:+.4f} "
                         f"p={pn['p']:.4f}" if pn['p'] == pn['p'] else '')
            print(f'  vs {args.baseline}: {a:<16} R@10 delta={pr["mean_delta"]:+.4f} '
                  f'(sd={pr["sd_delta"]:.4f}) n={pr["n"]}{extra}')

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(all_out, f, indent=2)
        print(f'\n[written] {args.json}')


if __name__ == '__main__':
    main()
