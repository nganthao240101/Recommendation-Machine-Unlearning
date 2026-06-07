"""
Performance & unlearn-efficiency metrics for RecEraser experiments.

Two kinds of measurements:

1) Retrain time (performance)
   - Wall-clock time to: (a) train all num_local shard models + (b) train
     the aggregator.
   - Measured by launching RecEraser_BPR.py as a subprocess and timing
     the whole run.

2) Shards affected (unlearn efficiency)
   - For each (unlearn_type, ratio), count how many of the `num_local`
     partitions contain at least one unlearned entity.
   - C[i] is the per-shard data dict; user unlearn -> C[i] has any
     unlearned uid; item unlearn -> C[i] has any unlearned iid.
   - Loaded from data/ml-1m/C_<...>.pk partition files; the actual
     unlearned entity set is read from the corresponding
     train_unlearned_<type>_rXX.txt (or unlearn_log.json).
"""
import os
import sys
import json
import time
import pickle
import subprocess
from typing import Optional, Dict, List, Tuple


PROJ = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(PROJ, '..', 'data', 'ml-1m')
WEIGHTS = os.path.join(PROJ, 'weights', 'ml-1m', 'RecEraser_BPR')
RESULTS = os.path.join(PROJ, '..', 'results')


# ---------------------------------------------------------------------------
# 1) Retrain time
# ---------------------------------------------------------------------------
def measure_retrain_time(part_type: int,
                         part_num: int = 5,
                         regs: str = '0.01',
                         epoch: int = 30,
                         epoch_agg: int = 50,
                         extra_args: Optional[List[str]] = None
                         ) -> Optional[float]:
    """Run `RecEraser_BPR.py` in a subprocess and time it.  Returns seconds.

    Skips if a checkpoint for the requested (part_type, part_num, regs)
    already exists AND the user passes a `force=True` arg (or reruns).
    """
    ckpt_dir = os.path.join(
        WEIGHTS, f'num-{part_num}_type-{part_type}_r{regs}')
    if not os.path.isfile(os.path.join(ckpt_dir, 'weights.index')):
        # train it
        cmd = [sys.executable, 'RecEraser_BPR.py',
               '--dataset', 'ml-1m',
               '--part_type', str(part_type),
               '--part_num', str(part_num),
               '--epoch', str(epoch),
               '--epoch_agg', str(epoch_agg),
               '--regs', f'[{regs}]']
        if extra_args:
            cmd.extend(extra_args)
        print(f'  [perf] timing: {" ".join(cmd)}')
        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=PROJ, capture_output=True,
                               text=True, timeout=7200)
            dt = time.time() - t0
            if r.returncode != 0:
                print(f'   [perf] FAILED (rc={r.returncode})')
                print(r.stderr[-1000:])
                return None
            return dt
        except subprocess.TimeoutExpired:
            print('   [perf] timeout')
            return None
    else:
        print(f'  [perf] ckpt exists, no retrain: {ckpt_dir}')
        return 0.0


# ---------------------------------------------------------------------------
# 2) Shards affected
# ---------------------------------------------------------------------------
def load_partitions(part_type: int, part_num: int) -> List[dict]:
    """Load C = list of {uid: [iid,...]} for each shard."""
    pk = os.path.join(DATA, f'C_type-{part_type}_num-{part_num}.pk')
    if not os.path.isfile(pk):
        print(f'   [eff] partition file not found: {pk}')
        return []
    with open(pk, 'rb') as f:
        C = pickle.load(f)
    return C


def get_unlearned_entities(unlearn_type: str, ratio: int
                           ) -> Tuple[set, set]:
    """Return (unlearned_uids, unlearned_iids) for a given scenario.

    We don't store removed-entity info per (type, ratio) explicitly, so we
    infer it: the unlearned set is the difference between train.txt and
    train_unlearned_<type>_rXX.txt.
    """
    base = os.path.join(DATA, 'train.txt')
    target_name = f'train_unlearned_{unlearn_type}_r{ratio:02d}.txt'
    target = os.path.join(DATA, target_name)
    if not os.path.isfile(target):
        # fall back to the pre-baked train_unlearned.txt
        target = os.path.join(DATA, 'train_unlearned.txt')

    def _read_users_items(path):
        u, items = set(), set()
        with open(path) as f:
            for line in f:
                parts = line.strip().split(' ')
                if not parts or parts == ['']:
                    continue
                uid = int(parts[0])
                u.add(uid)
                for it in parts[1:]:
                    items.add(int(it))
        return u, items

    u_base, i_base = _read_users_items(base)
    u_after, i_after = _read_users_items(target)
    if unlearn_type == 'user':
        return (u_base - u_after), set()
    elif unlearn_type == 'item':
        return set(), (i_base - i_after)
    elif unlearn_type == 'interaction':
        # (we don't know which exact pairs were removed; we treat it as
        # 'any user with a different item set is partially affected')
        return u_base - u_after, i_base - i_after
    return set(), set()


def count_shards_affected(part_type: int, part_num: int,
                          unlearn_type: str, ratio: int
                          ) -> Tuple[int, int, List[bool]]:
    """For each shard C[i], decide if it contains any unlearned entity.

    Returns (n_affected, n_total, list_of_bools).
    """
    C = load_partitions(part_type, part_num)
    if not C:
        return 0, 0, []
    u_unlearn, i_unlearn = get_unlearned_entities(unlearn_type, ratio)

    affected = []
    for shard in C:
        shard_users = set(shard.keys())
        shard_items = set()
        for items in shard.values():
            shard_items.update(items)
        if unlearn_type == 'user':
            hit = bool(shard_users & u_unlearn)
        elif unlearn_type == 'item':
            hit = bool(shard_items & i_unlearn)
        else:  # interaction
            hit = bool((shard_users & u_unlearn) or
                       (shard_items & i_unlearn))
        affected.append(hit)
    return sum(affected), len(affected), affected


# ---------------------------------------------------------------------------
# Convenience: dump a full perf+eff JSON
# ---------------------------------------------------------------------------
def build_report(part_num: int = 5, regs: str = '0.01',
                 epoch: int = 30, epoch_agg: int = 50,
                 measure_perf: bool = False
                 ) -> dict:
    """Build results/perf_eff_num{N}.json with all measurements."""
    out = {'part_num': part_num, 'regs': regs,
           'epoch': epoch, 'epoch_agg': epoch_agg,
           'retrain_time_s': {}, 'shards_affected': {}}

    # Retrain time per partition (only if measure_perf)
    if measure_perf:
        for pt, name in [(1, 'InP'), (2, 'UBP'),
                         (3, 'Random'), (4, 'IBP')]:
            dt = measure_retrain_time(pt, part_num, regs, epoch, epoch_agg)
            out['retrain_time_s'][name] = dt

    # Shards affected per (partition, unlearn_type, ratio)
    for pt, name in [(1, 'InP'), (2, 'UBP'),
                     (3, 'Random'), (4, 'IBP')]:
        out['shards_affected'][name] = {}
        for ut in ('interaction', 'user', 'item'):
            out['shards_affected'][name][ut] = {}
            for ratio in (5, 10, 20):
                n_aff, n_tot, flags = count_shards_affected(
                    pt, part_num, ut, ratio)
                out['shards_affected'][name][ut][f'r{ratio:02d}'] = {
                    'affected': n_aff,
                    'total':    n_tot,
                    'ratio':    n_aff / n_tot if n_tot else 0.0,
                    'flags':    flags,
                }

    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, f'perf_eff_num{part_num}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[OK] Saved: {out_path}')
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--part_num', type=int, default=5)
    ap.add_argument('--regs', default='0.01')
    ap.add_argument('--epoch', type=int, default=30)
    ap.add_argument('--epoch_agg', type=int, default=50)
    ap.add_argument('--measure_perf', action='store_true',
                    help='Actually time the retrain (slow).')
    args = ap.parse_args()
    build_report(part_num=args.part_num, regs=args.regs,
                 epoch=args.epoch, epoch_agg=args.epoch_agg,
                 measure_perf=args.measure_perf)
