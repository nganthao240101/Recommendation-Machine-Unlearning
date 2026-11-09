"""
EX1 driver - chay 5 lan tren CAC FILE METHOD DA CO (khong viet lai logic method).

Muc tich: so 4 method chinh trong CUNG DIEU KIEN, 5 seed, tren 2 dataset.
  Full Retrain  (method_1_full_retrain_v2.py)
  Ours          (method_4_ours.py)            --partition signature
  SISA          (method_2_sisa.py)
  RecEraser     (method_3_receraser_v2.py)    --partition_type 1=InP 2=UBP 3=Random

Moi method duoc goi bang subprocess voi CLI co san; driver chi:
  1. lap 5 seed (0..4) x moi (dataset, method)
  2. doc JSON ket qua tra ve
  3. cong mean +- std cho TRUOC/SAU unlearn, in ra console

Tieu chi chinh: Recall@10 / NDCG@10 (va @20/@50) tren RETAINED test.
Chay tuần tự 1 job 1 luc (tranh GPU contention -> so thoi gian khong nhiễu).

Usage:
  python code/run_ex1_five_seed.py                       # 2 dataset, 5 seed, full set
  python code/run_ex1_five_seed.py --datasets ml-1m
  python code/run_ex1_five_seed.py --seeds 0,1,2
  python code/run_ex1_five_seed.py --methods ours,sisa
  python code/run_ex1_five_seed.py --dry_run
"""

import os
import sys
import json
import glob
import time
import argparse
import statistics
import subprocess

PROJ = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PROJ)
PY = sys.executable

KS = (10, 20, 50)


# ----------------------------------------------------------------------------
# method -> (script, extra CLI args, json filename template)
# ----------------------------------------------------------------------------
def build_methods(args):
    """Return list of (label, script, extra_args, json_tmpl)."""
    out = []

    if 'full' in args.methods:
        out.append((
            'Full Retrain',
            'method_1_full_retrain_v2.py',
            ['--max_epochs', str(args.epochs),
             '--unlearn_ratio', str(args.unlearn_ratio),
             '--unlearn_mode', args.unlearn_mode],
            'results_fullretrain_v2_{sfx}.json',
        ))

    if 'ours' in args.methods:
        out.append((
            'Ours (signature)',
            'method_4_ours.py',
            ['--emb_dim', str(args.emb_dim),
             '--signature_dim', str(args.sig_dim),
             '--n_shards', str(args.n_shards),
             '--max_epochs', str(args.epochs),
             '--retrain_epochs', str(args.retrain_epochs),
             '--learning_rate', str(args.lr),
             '--batch_size', str(args.batch_size),
             '--partition', 'signature',
             '--use_mean', '0',
             '--unlearn_ratio', str(args.unlearn_ratio),
             '--unlearn_mode', args.unlearn_mode],
            'results_ours_3components_{sfx}.json',
        ))

    if 'sisa' in args.methods:
        out.append((
            'SISA',
            'method_2_sisa.py',
            ['--model_name', 'BPRMF',
             '--emb_dim', str(args.emb_dim),
             '--n_shards', str(args.n_shards),
             '--max_epochs', str(args.epochs),
             '--retrain_epochs', str(args.retrain_epochs),
             '--learning_rate', str(args.lr),
             '--batch_size', str(args.batch_size),
             '--unlearn_ratio', str(args.unlearn_ratio),
             '--unlearn_mode', args.unlearn_mode],
            'results_sisa_bprmf_{sfx}.json',
        ))

    if 'rec_inp' in args.methods:
        out.append((
            'RecEraser (InP)',
            'method_3_receraser_v2.py',
            ['--emb_dim', str(args.emb_dim),
             '--n_shards', str(args.n_shards),
             '--partition_type', '1',
             '--max_epochs_local', str(args.epochs),
             '--lr', str(args.lr),
             '--lr_finetune', str(args.lr),
             '--batch_size', str(args.batch_size),
             '--use_attention', '0',
             '--pretrain', '0',
             '--embed_source', 'wmf',
             '--unlearn_ratio', str(args.unlearn_ratio),
             '--unlearn_mode', args.unlearn_mode],
            'results_receraser_inp_{sfx}.json',
        ))

    if 'rec_ubp' in args.methods:
        out.append((
            'RecEraser (UBP)',
            'method_3_receraser_v2.py',
            ['--emb_dim', str(args.emb_dim),
             '--n_shards', str(args.n_shards),
             '--partition_type', '2',
             '--max_epochs_local', str(args.epochs),
             '--lr', str(args.lr),
             '--lr_finetune', str(args.lr),
             '--batch_size', str(args.batch_size),
             '--use_attention', '0',
             '--pretrain', '0',
             '--embed_source', 'wmf',
             '--unlearn_ratio', str(args.unlearn_ratio),
             '--unlearn_mode', args.unlearn_mode],
            'results_receraser_ubp_{sfx}.json',
        ))

    if 'rec_random' in args.methods:
        out.append((
            'RecEraser (Random)',
            'method_3_receraser_v2.py',
            ['--emb_dim', str(args.emb_dim),
             '--n_shards', str(args.n_shards),
             '--partition_type', '3',
             '--max_epochs_local', str(args.epochs),
             '--lr', str(args.lr),
             '--lr_finetune', str(args.lr),
             '--batch_size', str(args.batch_size),
             '--use_attention', '0',
             '--pretrain', '0',
             '--embed_source', 'wmf',
             '--unlearn_ratio', str(args.unlearn_ratio),
             '--unlearn_mode', args.unlearn_mode],
            'results_receraser_random_{sfx}.json',
        ))

    return out


# ----------------------------------------------------------------------------
# doc JSON -> dict {before:{}, after:{}, unlearn_time}
# ----------------------------------------------------------------------------
def read_result(path):
    """Chuan hoa JSON cua 4 method ve cung shape."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None

    def flat(x):
        """-> {'recall@10':..,'ndcg@10':..}"""
        out = {}
        if isinstance(x, dict):
            for k, v in x.items():
                if k.startswith('recall@') or k.startswith('ndcg@'):
                    out[k] = v
                elif isinstance(v, dict):
                    # Full v2: {'retained': {...}, 'full': {...}}
                    if 'recall' in v and 'ndcg' in v:
                        for i, kk in enumerate(KS):
                            out[f'recall@{kk}'] = v['recall'][i]
                            out[f'ndcg@{kk}'] = v['ndcg'][i]
                    else:
                        out.update(flat(v))
        return out

    before = flat(d.get('before', {}))
    after = flat(d.get('after', {}))
    if not after and 'retained' in d.get('after', {}):
        after = flat(d['after']['retained'])
    if not before and 'retained' in d.get('before', {}):
        before = flat(d['before']['retained'])

    ut = d.get('unlearn_time')
    return {'before': before, 'after': after, 'unlearn_time': ut}


def run_one(script, extra, dataset, seed, sfx, dry_run=False):
    """Chay 1 method file bang subprocess. Tra (ok, thong bao loi)."""
    cmd = [PY, os.path.join(PROJ, script),
           '--dataset', dataset, '--seed', str(seed),
           '--output_suffix', sfx] + extra
    if dry_run:
        print('  DRY:', ' '.join(cmd))
        return True, None

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding='utf-8', errors='replace')
    dt = time.time() - t0
    if proc.returncode != 0:
        tail = '\n'.join((proc.stdout or '').splitlines()[-6:])
        err = (proc.stderr or '').splitlines()[-6:]
        return False, (f'exit={proc.returncode}\n{tail}\n' + '\n'.join(err))
    return True, f'{dt:.1f}s'


def fmt(vals):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return '        n/a       '
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f'{m:.4f}+-{sd:.4f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='ml-1m,yelp2018')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--methods', default='full,ours,sisa,rec_inp,rec_ubp,rec_random')
    ap.add_argument('--emb_dim', type=int, default=64)
    ap.add_argument('--sig_dim', type=int, default=64)
    ap.add_argument('--n_shards', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--retrain_epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--unlearn_ratio', type=float, default=0.01)
    ap.add_argument('--unlearn_mode', default='random')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    args.methods = [m.strip() for m in args.methods.split(',') if m.strip()]

    methods = build_methods(args)
    if not methods:
        print('no methods selected')
        return

    print('=' * 78)
    print('EX1 - 4 method, 5 seed, cung dieu kien')
    print('=' * 78)
    print(f'datasets={datasets}  seeds={seeds}')
    print(f'K={args.n_shards}  emb={args.emb_dim}  epochs={args.epochs}  '
          f'lr={args.lr}  unlearn={args.unlearn_ratio} {args.unlearn_mode}')
    print(f'methods={[m[0] for m in methods]}')
    print()

    for ds in datasets:
        print('\n' + '#' * 78)
        print(f'# DATASET {ds}')
        print('#' * 78)

        for label, script, extra, tmpl in methods:
            print(f'\n--- {label} ({ds}) ---')
            per_seed = []
            for sd in seeds:
                sfx = f'{ds.replace("-", "_")}_{sd}'
                ok, msg = run_one(script, extra, ds, sd, sfx, args.dry_run)
                res = read_result(os.path.join(PROJ, tmpl.format(sfx=sfx))) if not args.dry_run else None
                if not ok:
                    print(f'  seed {sd}: FAILED  {msg}')
                    per_seed.append(None)
                    continue
                if res is None:
                    print(f'  seed {sd}: no JSON found ({tmpl.format(sfx=sfx)})')
                    per_seed.append(None)
                    continue
                r10 = res['after'].get('recall@10')
                n10 = res['after'].get('ndcg@10')
                b10 = res['before'].get('recall@10')
                ut = res['unlearn_time']
                print(f'  seed {sd}: before R@10={fmt([b10]).strip():<14} '
                      f'after R@10={fmt([r10]).strip():<14} '
                      f'NDCG@10={fmt([n10]).strip():<14} '
                      f'unlearn={(f"{ut:.1f}s" if ut else "n/a")}')
                per_seed.append(res)

            good = [r for r in per_seed if r]
            if not good:
                print(f'  {label}: KHONG CO KET QUA')
                continue
            print(f'  >> {label}: before R@10={fmt([r["before"].get("recall@10") for r in good])}'
                  f'  after R@10={fmt([r["after"].get("recall@10") for r in good])}'
                  f'  after NDCG@10={fmt([r["after"].get("ndcg@10") for r in good])}'
                  f'  unlearn={fmt([r["unlearn_time"] for r in good])}s'
                  f'  (n={len(good)}/{len(seeds)})')

        # ---- bang tong hop mean +- std ----
        print('\n' + '=' * 78)
        print(f'CONSOLE SUMMARY - {ds}  ({len(seeds)} seed)')
        print('=' * 78)
        header = (f'{"method":<22}{"before R@10":<20}{"after R@10":<20}'
                  f'{"after N@10":<20}{"after R@20":<16}{"unlearn(s)":<18}')
        print(header)
        for label, script, extra, tmpl in methods:
            rows = []
            for sd in seeds:
                sfx = f'{ds.replace("-", "_")}_{sd}'
                r = read_result(os.path.join(PROJ, tmpl.format(sfx=sfx)))
                if r:
                    rows.append(r)
            if not rows:
                print(f'{label:<22}  (no data)')
                continue
            print(f'{label:<22}'
                  f'{fmt([r["before"].get("recall@10") for r in rows]):<20}'
                  f'{fmt([r["after"].get("recall@10") for r in rows]):<20}'
                  f'{fmt([r["after"].get("ndcg@10") for r in rows]):<20}'
                  f'{fmt([r["after"].get("recall@20") for r in rows]):<16}'
                  f'{fmt([r["unlearn_time"] for r in rows]):<18}')

    print('\nDone.')


if __name__ == '__main__':
    main()
