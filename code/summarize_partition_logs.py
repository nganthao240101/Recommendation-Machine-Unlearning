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
import csv
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
    # written by the newer runs; absent from the older logs, which then
    # simply have no value for it
    'parallel_time':   re.compile(r'parallel=([\d.]+)s'),
}
RE_WORK = {
    'unlearn_samples': re.compile(r'unlearn_samples=(\d+)'),
}
RE_BG = {
    'bg_load_before': re.compile(r'before=(\d+)%'),
    'bg_load_after':  re.compile(r'after=(\d+)%'),
}
RE_INSTABILITY = {
    'instability_label':   re.compile(r'label-matched=([\d.]+)%'),
    'instability_pairwise': re.compile(r'pairwise=([\d.]+)%'),
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

        if pending is not None and 'cost:' in ln:
            # the newer runs print the cost on its own line
            for k, rx in RE_COST.items():
                mm = rx.search(ln)
                if mm:
                    pending[k] = float(mm.group(1))
            continue
        if pending is not None and 'work:' in ln:
            for k, rx in RE_WORK.items():
                mm = rx.search(ln)
                if mm:
                    pending[k] = float(mm.group(1))
            continue
        if pending is not None and 'bg-load:' in ln:
            for k, rx in RE_BG.items():
                mm = rx.search(ln)
                if mm:
                    pending[k] = float(mm.group(1))
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
            continue
        if pending is not None and 'instability:' in ln:
            for k, rx in RE_INSTABILITY.items():
                mm = rx.search(ln)
                if mm:
                    pending[k] = float(mm.group(1))
            continue
        if pending is not None and 'per-shard retrain' in ln:
            # the per-shard breakdown: keep it so the sum of the shard times
            # can be checked against the reported unlearn time
            pending['per_shard_detail'] = ln.strip()
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


def _mean_std(vals, prec=4, star=False):
    vals = [v for v in vals if v is not None]
    if not vals:
        return '---'
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    out = f'{m:.{prec}f} \\pm {s:.{prec}f}'
    return out + ('$^*$' if star else '')


def latex_table(recs, log_label, title, proposed, baseline, prec=4):
    """One .tex table per log: utility, co-location/load, cost, then the
    paired deltas. Every cell is mean +/- std over the seeds that ran."""
    arms = sorted({r['arm'] for r in recs})
    out = []
    out.append(f'% ===== {log_label} =====')
    out.append(f'% {title}  ({len(recs)} runs, n_seeds = '
               f'{max((r["seed"] for r in recs if r["seed"] >= 0), default=-1)}+1)')
    out.append(r'\begin{table}[t]')
    out.append(r'  \centering')
    out.append(f'  \\caption{{{title}}}  % label: tab:{log_label.lower().replace(" ", "_")}')
    out.append(r'  \small')
    out.append(r'  \begin{tabular}{l c c c c c c c}')
    out.append(r'    \toprule')
    out.append(r'    arm & R@10 & NDCG@10 & R@20 & R@50 & IntraSim & NR@10 '
               + r' & max/mean \\')
    out.append(r'    \midrule')
    for a in arms:
        rs = [r for r in recs if r['arm'] == a]
        out.append(
            f'    {a} & '
            f'{_mean_std([r.get("recall_10") for r in rs])} & '
            f'{_mean_std([r.get("ndcg_10") for r in rs])} & '
            f'{_mean_std([r.get("recall_20") for r in rs])} & '
            f'{_mean_std([r.get("recall_50") for r in rs])} & '
            f'{_mean_std([r.get("intra_similarity") for r in rs], prec=3)} & '
            f'{_mean_std([r.get("nr_10") for r in rs], prec=3)} & '
            f'{_mean_std([r.get("max_mean_ratio") for r in rs], prec=3)} \\'
        )
    out.append(r'    \bottomrule')
    out.append(r'  \end{tabular}')
    out.append(r'\end{table}')

    # cost table
    out.append('')
    out.append(r'\begin{table}[t]')
    out.append(r'  \centering')
    out.append(f'  \\caption{{Unlearning cost for {title}}}  '
               r'  % affected shards is how many shard models are retrained')
    out.append(r'  \small')
    out.append(r'  \begin{tabular}{l c c c c}')
    out.append(r'    \toprule')
    out.append(r'    arm & train(s) & unlearn(s) & parallel(s) & affected shards \\')
    out.append(r'    \midrule')
    for a in arms:
        rs = [r for r in recs if r['arm'] == a]
        out.append(
            f'    {a} & '
            f'{_mean_std([r.get("train_time") for r in rs], prec=0)} & '
            f'{_mean_std([r.get("unlearn_time") for r in rs], prec=0)} & '
            f'{_mean_std([r.get("parallel_time") for r in rs], prec=0)} & '
            f'{_mean_std([r.get("affected_shards") for r in rs], prec=0)} \\'
        )
    out.append(r'    \bottomrule')
    out.append(r'  \end{tabular}')
    out.append(r'\end{table}')

    # instability table: the share of retained users whose shard changed after
    # the partition was re-run on the remaining data
    out.append('')
    out.append(r'\begin{table}[t]')
    out.append(r'  \centering')
    out.append(f'  \\caption{{Instability rate for {title}}}  '
               r'  % label-matched: shard labels aligned by Hungarian matching')
    out.append(r'  \small')
    out.append(r'  \begin{tabular}{l c c}')
    out.append(r'    \toprule')
    out.append(r'    arm & label-matched (\%) & pairwise (\%) \\')
    out.append(r'    \midrule')
    for a in arms:
        rs = [r for r in recs if r['arm'] == a]
        out.append(
            f'    {a} & '
            f'{_mean_std([r.get("instability_label") for r in rs], prec=2)} & '
            f'{_mean_std([r.get("instability_pairwise") for r in rs], prec=2)} \\'
        )
    out.append(r'    \bottomrule')
    out.append(r'  \end{tabular}')
    out.append(r'\end{table}')

    # paired deltas
    out.append('')
    out.append(f'% --- paired deltas vs {baseline} ({title}) ---')
    for key, lab in (('recall_10', 'R@10'), ('ndcg_10', 'NDCG@10')):
        pr = paired_test(recs, proposed, baseline, key)
        if not pr:
            continue
        pval = f'{pr["p"]:.4f}' if pr['p'] == pr['p'] else 'n/a'
        out.append(f'% {lab}: {proposed} - {baseline} = {pr["mean_delta"]:+.4f} '
                   f'(sd={pr["sd_delta"]:.4f}, n={pr["n"]}, p={pval})')
    return '\n'.join(out)


RAW_COLS = [
    'seed', 'arm',
    'recall_10', 'ndcg_10', 'recall_20', 'ndcg_20', 'recall_50', 'ndcg_50',
    'train_time', 'unlearn_time', 'parallel_time', 'affected_shards',
    'unlearn_samples', 'bg_load_before', 'bg_load_after',
    'intra_similarity', 'nr_10', 'nr_20', 'nr_50',
    'user_load_cv', 'interaction_load_cv', 'max_mean_ratio',
    'instability_label', 'instability_pairwise',
]


def write_csvs(all_out, base, proposed='PROPOSED', baseline='RANDOM_PERM'):
    """Write two CSVs next to each other, ready to paste into the paper:

    <base>_raw.csv   one row per (log, arm, seed) - the data behind the tables
    <base>_mean.csv  one row per (log, arm) with mean +- std, plus the paired
                     PROPOSED-vs-baseline delta and p-value on R@10 and NDCG@10

    A log that has fewer than 5 parsed seeds still produces a row; the
    'n' column makes that visible so an incomplete run is not mistaken for a
    finished one.
    """
    raw_rows, agg_rows = [], []
    for log, recs in all_out.items():
        short = os.path.splitext(os.path.basename(log))[0]
        for r in recs:
            row = {'log': short, 'seed': r.get('seed', ''), 'arm': r['arm']}
            for c in RAW_COLS[2:]:
                v = r.get(c)
                row[c] = f'{v:.6f}' if isinstance(v, float) else (v if v is not None else '')
            raw_rows.append(row)

        arms = sorted({r['arm'] for r in recs})
        n = max((r['seed'] for r in recs if isinstance(r.get('seed'), int)
                 and r['seed'] >= 0), default=-1) + 1
        for a in arms:
            rs = [r for r in recs if r['arm'] == a]
            nr = len(rs)
            row = {'log': short, 'arm': a, 'n_seeds': nr, 'n_runs': nr}
            # util + cost: mean +- std. The slice stops before the
            # co-location block; a new cost column (parallel_time) or a new
            # util column belongs here.
            cost_end = RAW_COLS.index('intra_similarity')
            for c in RAW_COLS[2:cost_end]:
                vals = [r[c] for r in rs if r.get(c) is not None]
                if vals:
                    m = st.mean(vals)
                    sd = st.stdev(vals) if len(vals) > 1 else 0.0
                    row[c + '_mean'] = f'{m:.4f}'
                    row[c + '_std'] = f'{sd:.4f}'
                else:
                    row[c + '_mean'] = ''
                    row[c + '_std'] = ''
            # co-location + load + instability: mean only
            for c in RAW_COLS[cost_end:]:
                vals = [r[c] for r in rs if r.get(c) is not None]
                row[c] = f'{st.mean(vals):.4f}' if vals else ''
            # paired delta vs the baseline
            for c, lab in (('recall_10', 'R10'), ('ndcg_10', 'N10')):
                if a == baseline:
                    continue
                pr = paired_test(recs, a, baseline, c)
                if pr:
                    row[f'delta_{lab}_vs_{baseline}'] = f'{pr["mean_delta"]:+.4f}'
                    row[f'delta_{lab}_sd'] = f'{pr["sd_delta"]:.4f}'
                    row[f'delta_{lab}_p'] = (f'{pr["p"]:.4f}'
                                              if pr['p'] == pr['p'] else 'n/a')
            agg_rows.append(row)

    raw_path = f'{base}_raw.csv'
    agg_path = f'{base}_mean.csv'
    with open(raw_path, 'w', newline='', encoding='utf-8') as f:
        # raw_rows are dicts: a plain csv.writer would write their keys, not
        # their values, so every row would repeat the header
        w = csv.DictWriter(f, fieldnames=['log'] + RAW_COLS, extrasaction='ignore')
        w.writeheader()
        w.writerows(raw_rows)
    with open(agg_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()) if agg_rows else [])
        w.writeheader()
        for row in agg_rows:
            # keep a stable column order and drop empty trailing keys
            w.writerow({k: row.get(k, '') for k in row})
    print(f'\n[written] {raw_path}  ({len(raw_rows)} rows)')
    print(f'[written] {agg_path}  ({len(agg_rows)} rows)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--baseline', default='RANDOM_PERM')
    ap.add_argument('--proposed', default='PROPOSED')
    ap.add_argument('--json', default=None)
    ap.add_argument('--csv', default=None,
                    help='write <base>_raw.csv and <base>_mean.csv (paper-ready)')
    ap.add_argument('--latex', default=None,
                    help='write a ready-to-paste .tex with one table per log')
    args = ap.parse_args()

    paths = []
    for p in args.logs:
        paths.extend(sorted(glob.glob(p)) if any(ch in p for ch in '*?[') else [p])

    all_out = {}
    lat = []
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

        # a title readable enough for the .tex caption, e.g.
        # "ex1_ml1m_serial" -> "ml-1m main (K=8, random deletion)"
        stem = os.path.splitext(os.path.basename(path))[0]
        lat.append(latex_table(recs, stem, f'{stem}',
                               args.proposed, args.baseline))

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

        # ---- instability ----
        # label-matched: the share of retained users whose shard changed after
        # the partition was re-run on the remaining data. pairwise: the share
        # of retained user pairs whose co-location flipped (label-invariant).
        print(f'\n{"arm":<16}{"instab label%":>16}{"instab pairwise%":>18}')
        print('-' * 116)
        for a in arms:
            rs = [r for r in recs if r['arm'] == a]
            print(f'{a:<16}'
                  f'{ms([r.get("instability_label") for r in rs], 16, 2):>16}'
                  f'{ms([r.get("instability_pairwise") for r in rs], 18, 2):>18}')

        # ---- cost ----
        print(f'\n{"arm":<16}{"train(s)":>17}{"unlearn(s)":>17}{"parallel(s)":>17}'
              f'{"affected shards":>18}')
        print('-' * 116)
        for a in arms:
            rs = [r for r in recs if r['arm'] == a]
            print(f'{a:<16}'
                  f'{ms([r.get("train_time") for r in rs], 17, 1):>17}'
                  f'{ms([r.get("unlearn_time") for r in rs], 17, 1):>17}'
                  f'{ms([r.get("parallel_time") for r in rs], 17, 1):>17}'
                  f'{ms([r.get("affected_shards") for r in rs], 18, 2):>18}')

        # ---- paired comparison ----
        print(f'\nPAIRED: {args.proposed} vs {args.baseline}')
        for key in ('recall_10', 'ndcg_10', 'recall_20', 'ndcg_20',
                    'recall_50', 'ndcg_50', 'unlearn_time', 'parallel_time',
                    'affected_shards', 'instability_label',
                    'instability_pairwise'):
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

    if args.csv:
        write_csvs(all_out, args.csv, proposed=args.proposed, baseline=args.baseline)


if __name__ == '__main__':
    main()
