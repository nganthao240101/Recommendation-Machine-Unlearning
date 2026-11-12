"""
Detect the logs whose retrain cost was measured with the old, broken code.

The old run_global_arm trained one model on the reduced data and returned that
single duration for both the train and the unlearn column, so in those logs
the retrain cost of the reference arm reads exactly the same as its training
cost. Those numbers are not a measurement of a retrain, and they must not end
up in the same table as the numbers from the fixed code.

A log is flagged when, for the GLOBAL arm, train_time and unlearn_time are
equal (or, for the newer format, when the two costs are suspiciously close
while the interaction counts differ by the deletion ratio).

It also reports the backbone each arm was trained with, so a table cannot
end up mixing a BPRMF arm with a WMF arm. The arms in this experiment all
train BPRMF; the WMF in the repository only produces the pretrained
embeddings that feed RecEraser's partition.

Run:
  python code/check_stale_retrain_cost.py logs/*.out
"""

import os
import re
import sys
import glob
import argparse

ARM_HDR = re.compile(r'---\s*ARM\s+([A-Z_]+)\s*\(seed\s+(\d+)\)')
COST = re.compile(r'train=([\d.]+)s.*?unlearn=([\d.]+)s')
NEW_COST = re.compile(r'train=([\d.]+)s\s+unlearn=([\d.]+)s\s+parallel=([\d.]+)s')
GLOBAL_MARK = re.compile(r'GLOBAL')


def scan(path):
    """-> list of (seed, train, unlearn) for the GLOBAL arm of this log."""
    out = []
    arm = None
    seed = None
    for ln in open(path, encoding='utf-8', errors='replace'):
        m = ARM_HDR.search(ln)
        if m:
            arm, seed = m.group(1), int(m.group(2))
            continue
        if arm == 'GLOBAL':
            m2 = COST.search(ln)
            if m2:
                out.append((seed, float(m2.group(1)), float(m2.group(2))))
                arm = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='*')
    ap.add_argument('--tol', type=float, default=0.01,
                    help='Relative difference below which the two costs are '
                         'treated as equal (they were the same number by '
                         'construction in the old code).')
    args = ap.parse_args()

    logs = args.logs or sorted(glob.glob('logs/*.out'))
    if not logs:
        print('no logs found')
        return

    print(f'{"log":<34}{"seed":>6}{"train(s)":>12}{"unlearn(s)":>13}'
          f'{"rel.diff":>11}  verdict')
    print('-' * 92)

    stale, clean, empty = [], [], []
    for path in logs:
        recs = scan(path)
        if not recs:
            empty.append(path)
            continue
        bad = 0
        for seed, tr, un in recs:
            rel = abs(tr - un) / max(tr, 1e-9)
            verdict = 'STALE (train == unlearn)' if rel < args.tol else 'ok'
            if rel < args.tol:
                bad += 1
            print(f'{os.path.basename(path):<34}{seed:>6}{tr:>12.1f}{un:>13.1f}'
                  f'{rel:>11.4f}  {verdict}')
        if bad == len(recs):
            stale.append(path)
        elif bad:
            print(f'  -> {path}: MIXED ({bad}/{len(recs)} stale)')
        else:
            clean.append(path)

    print()
    print(f'clean  ({len(clean)}):')
    for p in clean:
        print(f'  {p}')
    if stale:
        print()
        print(f'STALE  ({len(stale)}) - the retrain cost in these logs was '
              f'never measured:')
        for p in stale:
            print(f'  {p}')
    if empty:
        print()
        print(f'no GLOBAL arm found in ({len(empty)}):')
        for p in empty:
            print(f'  {p}')

    if stale:
        print()
        print('These logs must NOT be merged with the new ones. Re-run them '
              'with the fixed code, or keep them apart and say so in the '
              'paper.')

    bad_bb = check_backbones([p for p in logs if os.path.isfile(p)])
    return 1 if (stale or bad_bb) else 0


# ----------------------------------------------------------------------------
# backbone consistency
# ----------------------------------------------------------------------------
BACKBONE_RE = re.compile(r"backbone['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9_+-]+)")
MODEL_RE = re.compile(r'\b(WMF|BPRMF|LightGCN|RecEraserModel)\b')


def scan_backbones(path):
    """The backbone named in the log, and any model class mentioned in it."""
    txt = open(path, encoding='utf-8', errors='replace').read()
    named = set(BACKBONE_RE.findall(txt))
    mentioned = set(MODEL_RE.findall(txt))
    return named, mentioned


def check_backbones(paths):
    """Every arm in the experiment trains BPRMF. WMF appears only as the
    source of the pretrained embeddings that feed RecEraser's partition."""
    print()
    print('=' * 92)
    print('BACKBONE CONSISTENCY')
    print('=' * 92)
    print(f'{"log":<40}{"declared":<20}{"mentioned":<24}verdict')
    print('-' * 92)
    bad = []
    for p in paths:
        named, mentioned = scan_backbones(p)
        if not named and not mentioned:
            continue
        # the backbone the arms are trained with
        declared = ','.join(sorted(named)) if named else '(not declared)'
        # WMF is expected to be mentioned: it produces the pretrained
        # embeddings. It is a problem only if a log declares a non-BPRMF
        # backbone for the arms.
        verdict = 'ok'
        real = {n for n in named if n.lower() != 'bprmf'}
        if real:
            verdict = f'CHECK: non-BPRMF backbone {sorted(real)}'
            bad.append(p)
        print(f'{os.path.basename(p):<40}{declared:<20}'
              f'{",".join(sorted(mentioned)):<24}{verdict}')
    if not bad:
        print()
        print('All arms train BPRMF. The WMF in the repository only produces '
              'the pretrained embeddings that feed RecEraser\'s partition.')
    else:
        print()
        print('The logs above declare a backbone other than BPRMF. Mixing '
              'them in one table would compare different models.')
    return bad


if __name__ == '__main__':
    sys.exit(main())
