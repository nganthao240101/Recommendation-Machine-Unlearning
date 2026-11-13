"""
Run both experiment scripts end to end on a tiny synthetic dataset, with all
seven arms, before a long run starts.

A bug that only shows at one arm (a wrong path, a shape mismatch) otherwise
surfaces hours into a run, after the arms before it have used the GPU. This
takes a minute or two and stops the start when anything fails.

  python code/smoke_test.py            # exit 0 = every arm ran to the summary
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = 'PROPOSED,PROPOSED_BAL,SISA,RANDOM_PERM,RANDOM_EMB,RECERASER,GLOBAL'


def make_data(root, n_users=120, n_items=90, seed=0):
    """train/test files and WMF embeddings laid out as data/<ds>/ is."""
    rng = np.random.RandomState(seed)
    d = os.path.join(root, 'smoke')
    os.makedirs(d)
    with open(os.path.join(d, 'train.txt'), 'w') as tr, \
            open(os.path.join(d, 'test.txt'), 'w') as te:
        for u in range(n_users):
            its = rng.choice(n_items, rng.randint(8, 20), replace=False)
            tr.write(' '.join(map(str, [u, *its[:-3]])) + '\n')
            te.write(' '.join(map(str, [u, *its[-3:]])) + '\n')
    np.savez(os.path.join(d, 'wmf_embeddings.npz'),
             user_embeddings=rng.randn(n_users, 64).astype(np.float32),
             item_embeddings=rng.randn(n_items, 64).astype(np.float32))


def run(name, cmd, log_dir):
    log = os.path.join(log_dir, name + '.out')
    with open(log, 'w') as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    text = open(log, encoding='utf-8', errors='replace').read()
    ok = rc == 0 and 'CONSOLE SUMMARY' in text
    missing = [a for a in ARMS.split(',') if f'ARM {a} ' not in text]
    ok = ok and not missing
    print(f'  {"OK  " if ok else "FAIL"} {name}  (rc={rc})')
    if not ok:
        if missing:
            print(f'       arms that never ran: {", ".join(missing)}')
        tail = [ln for ln in text.splitlines() if ln.strip()][-12:]
        print('       ' + '\n       '.join(tail))
    return ok


def main():
    root = tempfile.mkdtemp(prefix='smoke_')
    make_data(root)
    py = sys.executable
    common = ['--dataset', 'smoke', '--data_root', root, '--n_shards', '4',
              '--emb_dim', '16', '--sig_dim', '16', '--epochs', '1',
              '--retrain_epochs', '1', '--batch_size', '64', '--seeds', '0',
              '--arms', ARMS, '--agg_epochs', '1']
    print(f'smoke test on a {120}-user synthetic set, all 7 arms ({root})')
    ok = run('partition_quality',
             [py, os.path.join(HERE, 'exp_partition_quality.py'), *common,
              '--unlearn_mode', 'random', '--unlearn_ratio', '0.05'], root)
    ok &= run('single_request',
              [py, os.path.join(HERE, 'exp_single_request.py'), *common,
               '--n_users', '2'], root)
    print('smoke test PASSED' if ok else 'smoke test FAILED - not starting')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
