"""
ONE setting for every dataset, chosen on validation.

Reads the cells tune_reg.py cached for each dataset (logs/tunegrid_<ds>.jsonl)
and scores every (reg, lr, dim) that has every arm on every dataset:

  score on a dataset  = mean validation Recall@20 of the arms (no arm favoured)
  relative score      = that score / the best score of any setting on the
                        same dataset (the datasets' Recall levels differ ~4x,
                        so raw means would let ml-1m decide alone)
  joint score         = mean of the relative scores over the datasets

Printed: the table, best first, and
  JOINT RECOMMENDED reg=.. lr=.. dim=..
The test set is not read (tune_reg.py never reads it).

  python code/pick_joint.py --datasets "ml-1m yelp2018" \
      --arms SAFER_P,SAFER_PI,PROPOSED,PROPOSED_BAL,RANDOM_PERM,SISA_FULL
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL = {'PROPOSED': 'SAFER A', 'PROPOSED_BAL': 'SAFER B', 'SAFER_P': 'SAFER-P',
         'SAFER_PI': 'SAFER-PI', 'RANDOM_PERM': 'Random', 'RANDOM_EMB': 'RandomEmb',
         'SISA_FULL': 'SISA'}


def load(path, arms, every, patience):
    """-> {(reg, lr, dim): {arm: val_r20}} of the cells run with this protocol"""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding='utf-8'):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        f = r['key'].split('|')
        # key: ds|arm|reg|lr|dim|K|sig|batch|val_ratio|val_seed|max_ep|every|patience|seed[|R5]
        if r['arm'] not in arms or int(f[11]) != every or int(f[12]) != patience:
            continue
        out.setdefault((float(r['reg']), float(r['lr']), int(r['dim'])), {})[r['arm']] = r['val_r20']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='ml-1m yelp2018')
    ap.add_argument('--arms', default='SAFER_P,SAFER_PI,PROPOSED,PROPOSED_BAL,RANDOM_PERM,SISA_FULL')
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--dims', default=None, help='only these embedding sizes')
    args = ap.parse_args()
    dss = args.datasets.split()
    arms = [a for a in args.arms.split(',') if a]
    dims = {int(d) for d in args.dims.split()} if args.dims else None
    logs = os.path.join(os.path.dirname(HERE), 'logs')
    cells = {ds: load(os.path.join(logs, f'tunegrid_{ds}.jsonl'), arms,
                      args.eval_every, args.patience) for ds in dss}
    full = {ds: {s: v for s, v in c.items() if all(a in v for a in arms)
                 and (dims is None or s[2] in dims)} for ds, c in cells.items()}
    common = sorted(set.intersection(*(set(full[ds]) for ds in dss)))
    if not common:
        print('no setting has every arm on every dataset yet')
        for ds in dss:
            print(f'  {ds}: {len(full[ds])} complete settings')
        return 1
    mean = {ds: {s: sum(full[ds][s][a] for a in arms) / len(arms) for s in full[ds]}
            for ds in dss}
    # relative to the best setting tried on that dataset (any setting, not only the common ones)
    best = {ds: max(mean[ds].values()) for ds in dss}
    rows = []
    for s in common:
        rel = [mean[ds][s] / best[ds] for ds in dss]
        rows.append((sum(rel) / len(rel), s, rel))
    rows.sort(key=lambda r: -r[0])

    print(f'ONE SETTING FOR {", ".join(dss)}  (validation Recall@20, mean of '
          f'{", ".join(LABEL.get(a, a) for a in arms)})')
    print(f"  {'reg':>7}{'lr':>8}{'dim':>5}" + ''.join(f'{ds:>12}{"rel":>7}' for ds in dss)
          + f"{'joint':>8}")
    for j, (reg, lr, dim), rel in rows:
        print(f'  {reg:>7g}{lr:>8g}{dim:>5d}'
              + ''.join(f'{mean[ds][(reg, lr, dim)]:>12.4f}{r:>7.3f}' for ds, r in zip(dss, rel))
              + f'{j:>8.4f}')
    print('\nEACH ARM AT THE JOINT SETTING vs ITS OWN BEST (per dataset)')
    j, top, _ = rows[0]
    for ds in dss:
        for a in arms:
            own = max(full[ds].values(), key=lambda v: v[a])[a]
            print(f'  {ds:<9} {LABEL.get(a, a):<9} {full[ds][top][a]:.4f}  '
                  f'(own best {own:.4f}, {100 * (full[ds][top][a] / own - 1):+.1f}%)')
    reg, lr, dim = top
    print(f'\nJOINT RECOMMENDED reg={reg:g} lr={lr:g} dim={dim}  (joint score {j:.4f})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
