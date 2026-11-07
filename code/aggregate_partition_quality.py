"""
Aggregate partition_quality_*.csv -> mean +- std, relative gain, paired tests.

Statistical procedure (per the experiment spec):
  Delta_s = Utility_PROPOSED,s - Utility_RANDOM,s      (paired, per seed)
  - report mean and std of Delta_s
  - paired t-test when the differences are approximately normal
  - Wilcoxon signed-rank test as the non-parametric alternative
  - report p-value AND effect size (not just "significant")
  - conclusion based on BOTH Recall and NDCG; if only one metric improves,
    report it as a mixed result

Never edits the raw CSV - reads it and writes a separate summary.
"""

import os
import sys
import json
import argparse
import numpy as np

PROJ = os.path.dirname(os.path.abspath(__file__))

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("[warn] scipy not available -> using a normal-approximation t-test")


def cohens_d_paired(d):
    """Paired Cohen's d = mean(d) / std(d, ddof=1)."""
    if len(d) < 2:
        return float('nan')
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / sd) if sd > 0 else float('nan')


def paired_tests(d, label):
    """Return dict of test results for the paired differences d."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    out = {'n': n, 'mean_diff': float(np.mean(d)) if n else float('nan'),
           'std_diff': float(np.std(d, ddof=1)) if n > 1 else float('nan')}

    if n < 2:
        out.update({'t_stat': float('nan'), 'p_ttest': float('nan'),
                    'p_wilcoxon': float('nan'), 'cohens_d': float('nan'),
                    'test_used': 'insufficient data'})
        return out

    # normality of the differences -> decides which test to lead with
    if HAVE_SCIPY and n >= 3:
        w_p = stats.shapiro(d).pvalue
    else:
        w_p = float('nan')
    out['shapiro_p'] = float(w_p)

    if HAVE_SCIPY:
        t_res = stats.ttest_rel([0] * n, d)   # one-sample t on the differences
        out['t_stat'] = float(t_res.statistic)
        out['p_ttest'] = float(t_res.pvalue)
        try:
            w_res = stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided')
            out['p_wilcoxon'] = float(w_res.pvalue)
        except Exception:
            out['p_wilcoxon'] = float('nan')
        out['test_used'] = ('paired t-test (differences ~ normal)' if (w_p == w_p and w_p >= 0.05)
                            else 'Wilcoxon signed-rank (non-normal differences)')
    else:
        # normal approximation for the mean of the differences
        se = np.std(d, ddof=1) / np.sqrt(n)
        t = np.mean(d) / se if se > 0 else float('nan')
        out['t_stat'] = float(t)
        out['p_ttest'] = float('nan')
        out['p_wilcoxon'] = float('nan')
        out['test_used'] = 'normal approximation (scipy missing)'

    out['cohens_d'] = cohens_d_paired(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--baseline', default='RANDOM_PERM')
    ap.add_argument('--proposed', default='PROPOSED')
    ap.add_argument('--out_json', default=None)
    ap.add_argument('--out_md', default=None)
    args = ap.parse_args()

    import csv
    with open(args.csv, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print('empty csv')
        return

    def num(r, k):
        v = r.get(k, '')
        try:
            return float(v)
        except (TypeError, ValueError):
            return float('nan')

    seeds = sorted({int(r['seed']) for r in rows})
    arms = sorted({r['arm'] for r in rows})
    metrics = ['recall_10', 'ndcg_10', 'recall_20', 'ndcg_20', 'recall_50', 'ndcg_50']

    # index: (arm, seed) -> row
    idx = {(r['arm'], int(r['seed'])): r for r in rows}

    print(f"CSV: {args.csv}")
    print(f"seeds: {seeds}")
    print(f"arms : {arms}\n")

    # ---------------- summary per arm ----------------
    print("=" * 78)
    print(f"PER-ARM SUMMARY (mean +- std over seeds)")
    print("=" * 78)
    summary = {}
    for arm in arms:
        arm_rows = [idx[(arm, s)] for s in seeds if (arm, s) in idx]
        if not arm_rows:
            continue
        summary[arm] = {}
        for m in metrics:
            vals = np.array([num(r, m) for r in arm_rows], dtype=float)
            vals = vals[~np.isnan(vals)]
            summary[arm][m] = {
                'mean': float(np.mean(vals)) if len(vals) else float('nan'),
                'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                'n': len(vals),
            }
        for m in ['intra_similarity', 'nr_10', 'nr_20', 'nr_50',
                  'user_load_cv', 'interaction_load_cv', 'max_mean_ratio',
                  'train_time', 'unlearn_time']:
            if m not in arm_rows[0]:
                continue
            vals = np.array([num(r, m) for r in arm_rows], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals):
                summary[arm][m] = {'mean': float(np.mean(vals)),
                                   'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                   'n': len(vals)}

    hdr = f"{'arm':<16}" + "".join(f"{m:>20}" for m in metrics[:4])
    print(hdr)
    for arm in summary:
        line = f"{arm:<16}"
        for m in metrics[:4]:
            d = summary[arm][m]
            line += f"{d['mean']:.4f}+-{d['std']:.4f}".rjust(20)
        print(line)

    print("\n" + "=" * 78)
    print("PARTITION-QUALITY METRICS (mean +- std)")
    print("=" * 78)
    pm = ['intra_similarity', 'nr_10', 'nr_20', 'nr_50',
          'user_load_cv', 'interaction_load_cv', 'max_mean_ratio']
    print(f"{'arm':<16}" + "".join(f"{m[:14]:>16}" for m in pm))
    for arm in summary:
        line = f"{arm:<16}"
        for m in pm:
            if m in summary[arm]:
                d = summary[arm][m]
                line += f"{d['mean']:.4f}+-{d['std']:.4f}".rjust(16)
            else:
                line += "".rjust(16)
        print(line)

    # ---------------- paired comparison PROPOSED vs BASELINE ----------------
    print("\n" + "=" * 78)
    print(f"PAIRED COMPARISON: {args.proposed} vs {args.baseline}")
    print("=" * 78)

    paired_seeds = [s for s in seeds
                    if (args.proposed, s) in idx and (args.baseline, s) in idx]
    print(f"paired seeds: {paired_seeds} (n={len(paired_seeds)})\n")

    comp = {'baseline': args.baseline, 'proposed': args.proposed,
            'paired_seeds': paired_seeds, 'tests': {}, 'relative_gain': {}}

    for m in metrics:
        prop = np.array([num(idx[(args.proposed, s)], m) for s in paired_seeds])
        base = np.array([num(idx[(args.baseline, s)], m) for s in paired_seeds])
        mask = ~(np.isnan(prop) | np.isnan(base))
        p, b = prop[mask], base[mask]
        if len(p) == 0:
            continue
        d = p - b
        t = paired_tests(d, m)
        rg = float(np.mean((p - b) / b) * 100) if np.all(b != 0) else float('nan')
        comp['tests'][m] = t
        comp['relative_gain'][m] = rg

        star = ''
        pv = t.get('p_wilcoxon', t.get('p_ttest', float('nan')))
        if pv == pv and pv < 0.05:
            star = ' *'
        print(f"  {m}")
        print(f"    {args.proposed}: {np.mean(p):.4f}   {args.baseline}: {np.mean(b):.4f}")
        print(f"    relative gain = {rg:+.2f}%   mean Delta = {t['mean_diff']:+.4f} "
              f"(std {t['std_diff']:.4f})")
        print(f"    test used     : {t['test_used']}")
        if t.get('p_ttest') == t.get('p_ttest'):
            print(f"    paired t-test : t={t['t_stat']:.3f}, p={t['p_ttest']:.4g}")
        if t.get('p_wilcoxon') == t.get('p_wilcoxon'):
            print(f"    wilcoxon      : p={t['p_wilcoxon']:.4g}")
        print(f"    effect size   : Cohen's d (paired) = {t['cohens_d']:.3f}")
        print()

    # ---------------- verdict ----------------
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    rec_sig = comp['tests'].get('recall_10', {}).get('p_wilcoxon', float('nan'))
    ndc_sig = comp['tests'].get('ndcg_10', {}).get('p_wilcoxon', float('nan'))
    if not (rec_sig == rec_sig) or not (ndc_sig == ndc_sig):
        rec_sig = comp['tests'].get('recall_10', {}).get('p_ttest', float('nan'))
        ndc_sig = comp['tests'].get('ndcg_10', {}).get('p_ttest', float('nan'))

    rec_pos = comp['relative_gain'].get('recall_10', 0) > 0
    ndc_pos = comp['relative_gain'].get('ndcg_10', 0) > 0
    rec_sig = rec_sig == rec_sig and rec_sig < 0.05
    ndc_sig = ndc_sig == ndc_sig and ndc_sig < 0.05

    if rec_pos and ndc_pos and rec_sig and ndc_sig:
        verdict = ("H1 SUPPORTED: the proposed partition improves utility over "
                   "random on BOTH Recall and NDCG, with statistical significance.")
    elif (rec_pos and rec_sig) != (ndc_pos and ndc_sig):
        verdict = ("MIXED RESULT: the proposed partition improves one metric "
                   "significantly but not the other - report as mixed, not as a win.")
    elif rec_pos or ndc_pos:
        verdict = ("NOT SUPPORTED (yet): the proposed partition does not beat "
                   "random significantly. H0 cannot be rejected.")
    else:
        verdict = ("H0 NOT REJECTED: the proposed partition is not better than "
                   "random on this dataset.")
    print(f"  {verdict}")
    print(f"  Recall@10: gain {comp['relative_gain'].get('recall_10', float('nan')):+.2f}%, "
          f"p={'<0.05' if rec_sig else '>=0.05'}")
    print(f"  NDCG@10  : gain {comp['relative_gain'].get('ndcg_10', float('nan')):+.2f}%, "
          f"p={'<0.05' if ndc_sig else '>=0.05'}")

    # mechanism check: does the partition actually co-locate similar users?
    if 'intra_similarity' in summary.get(args.proposed, {}) and \
       'intra_similarity' in summary.get(args.baseline, {}):
        pi = summary[args.proposed]['intra_similarity']['mean']
        bi = summary[args.baseline]['intra_similarity']['mean']
        print(f"\n  MECHANISM: intra-similarity {args.proposed}={pi:.4f} vs "
              f"{args.baseline}={bi:.4f} ({'HIGHER' if pi > bi else 'NOT higher'})")
        for q in ['nr_10', 'nr_20', 'nr_50']:
            if q in summary.get(args.proposed, {}) and q in summary.get(args.baseline, {}):
                pv = summary[args.proposed][q]['mean']
                bv = summary[args.baseline][q]['mean']
                print(f"    {q}: proposed={pv:.4f} baseline={bv:.4f} "
                      f"(random expectation ~1/K)")

    comp['verdict'] = verdict
    comp['summary'] = summary

    if args.out_json:
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(comp, f, indent=2, ensure_ascii=False)
        print(f"\nJSON  -> {args.out_json}")

    if args.out_md:
        with open(args.out_md, 'w', encoding='utf-8') as f:
            f.write(f"# Partition quality: {args.proposed} vs {args.baseline}\n\n")
            f.write(f"Dataset: {rows[0].get('dataset')}, K={rows[0].get('K')}, "
                    f"seeds={paired_seeds}\n\n")
            f.write("| Metric | " + args.proposed + " | " + args.baseline +
                    " | Relative gain | p (Wilcoxon) | Cohen's d |\n")
            f.write("|---|---|---|---|---|---|\n")
            for m in metrics:
                if m not in comp['tests']:
                    continue
                t = comp['tests'][m]
                p = summary[args.proposed][m]['mean']
                b = summary[args.baseline][m]['mean']
                pv = t.get('p_wilcoxon')
                pv_s = f"{pv:.4g}" if (pv == pv) else "n/a"
                f.write(f"| {m} | {p:.4f} | {b:.4f} | "
                        f"{comp['relative_gain'][m]:+.2f}% | {pv_s} | {t['cohens_d']:.3f} |\n")
            f.write(f"\n**Verdict:** {verdict}\n")
        print(f"MD    -> {args.out_md}")


if __name__ == '__main__':
    main()
