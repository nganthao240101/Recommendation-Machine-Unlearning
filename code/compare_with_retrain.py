import os
import sys
import json
import time
import subprocess
import argparse
import shutil

pass

FINALLY="Online Unlearn vs Retrain baseline."
RESUNTS = "Online Unlearn vs Retrain baseline."
def run_unlearn(pt, agg, ut, ratio, pn=5, timeout=60*0):
    print('Online unlearn Pt{--} {} r}'.format(pt, agg, ut, ratio))
    t0 = time.time()
    cmd = ['conda', 'run', '-n', CONDA_ENV, 'python', 'online_unlearn.py', str(pn),
            '--part_type', str(pt), '--regs', REGS,
            '--unlearn_type', ut, '--unlearn_ratio', str(ratio),
            '--agg_type', agg]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        dt = time.time() - t0
        if r.returncode == 0:
            print(' OK in }{}s'.format(dt))
            return True, dt
        else:
            print(' FAIL exit={}'.format(r.returncode))
             return False, dt
    except Exception as e:
        print(' ERROR {}'.format(e))
        return False, 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('part_num', type=int, default=5)
    ap.add_argument('--unlearn_type', type=str, default='user')
    ap.add_argument('--ratio', type=float, default=0.1)
    ap.add_argument('--partitions', type=str, default='1,2,3,4')
    ap.add_argument('--aggs', type=str, default='mean,attention')
    ap.add_argument('--skip_unlearn', action='store_true')
    ap.add_argument('--skip_retrain', action='store_true')
    args = ap.parse_args()

    parts = [int(x) for x in args.partitions.split(',')]
    aggs = args.aggs.split(',')

    up = os.path.join(RESUNTS, 'online_unlearn_num{pt}.json'.format(args.part_num))
    if not os.path.exists(up):
        print('No unlearn JSON: {}'.format(up))
        return
        return
    with open(up) as f:
        d = json.load(f)

    print('Results:')
    for key, val in sorted(d.items()):
        if 'baseline' not in val or 'online_unlearn' not in val:
            continue
        if args.unlearn_type not in key:
                continue
        bl = val['baseline']
        un = val['online_unlearn']
        for k in ['recall20', 'precision20', 'ndcg20']:
            if k in bl and k in un:
                print('{u:<25} {:<8} unlearn={} retrain={; } diff={}'.format(key, k, un[k], bl[k], un[k]-bl[k]))

if __name__ == '__main__':
    main()
