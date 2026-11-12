"""
Same check as check_ndcg_evaluator.py, but on the REAL ml-1m test split.

The synthetic check said the sharded evaluator is fine, yet the experiment logs
show NDCG@20 identical to NDCG@10 for every sharded arm while the GLOBAL arm
(evaluated by a different routine) does not. That contradiction can only be
settled on the real data, where the test-list lengths and the number of
distinct test items actually matter.

Run:  python code/check_ndcg_realdata.py
"""

import os
import sys
import heapq

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KS = [10, 20, 50]
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_real(dataset='ml-1m'):
    d = os.path.join(PROJ, 'data', dataset)
    train_items, test_set = {}, {}
    n_users, n_items = 0, 0
    with open(os.path.join(d, 'train.txt')) as f:
        for line in f:
            p = line.strip().split(' ')
            if not p or not p[0]:
                continue
            u = int(p[0])
            its = [int(x) for x in p[1:] if x]
            if not its:
                continue
            train_items[u] = its
            n_users = max(n_users, u + 1)
            n_items = max(n_items, max(its) + 1)
    with open(os.path.join(d, 'test.txt')) as f:
        for line in f:
            p = line.strip().split(' ')
            if not p or not p[0]:
                continue
            u = int(p[0])
            its = [int(x) for x in p[1:] if x]
            if not its:
                continue
            test_set[u] = its
            n_users = max(n_users, u + 1)
            n_items = max(n_items, max(its) + 1)
    return train_items, test_set, n_users, n_items


class _FakeModel:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float64)

    def predict_user_items(self, user, items):
        import torch
        idx = items.cpu().numpy() if hasattr(items, 'cpu') else np.asarray(items)
        return torch.tensor(self.scores[idx])


class _FakeShardModels:
    def __init__(self, per_shard):
        self.per_shard = per_shard

    def get_model(self, sid):
        return _FakeModel(self.per_shard[sid])


def main():
    from method_4_ours import evaluate_hard_routing

    train_items, test_data, n_users, n_items = load_real('ml-1m')
    print(f'ml-1m: {n_users} users, {n_items} items, '
          f'{len(test_data)} users with test data')

    lens = [len(v) for v in test_data.values()]
    dups = [len(v) - len(set(v)) for v in test_data.values()]
    print(f'test list length : min={min(lens)} max={max(lens)} '
          f'mean={np.mean(lens):.2f}')
    print(f'duplicates       : users with any duplicate = '
          f'{sum(1 for d in dups if d > 0)} / {len(dups)}')
    over10 = sum(1 for l in lens if l > 10)
    print(f'users with >10 test items : {over10} '
          f'({100.0*over10/len(lens):.1f}%)')
    print()

    rng = np.random.RandomState(0)
    K = 8
    shard_of = {u: u % K for u in range(n_users)}
    per_shard = {s: rng.rand(n_items) for s in range(K)}
    fake = _FakeShardModels(per_shard)

    got = evaluate_hard_routing(fake, shard_of, train_items, test_data,
                                n_users, n_items, 'cpu', Ks=KS)

    print(f'{"K":>4}{"repo R":>12}{"repo NDCG":>12}{"independent NDCG":>20}'
          f'{"|dNDCG|":>12}')
    mine = {k: [] for k in KS}
    for u in range(n_users):
        if u not in test_data:
            continue
        model = fake.get_model(shard_of[u])
        s = model.scores.copy()
        for it in set(train_items.get(u, [])):
            s[it] = -np.inf
        rank_list = heapq.nlargest(max(KS), range(len(s)), key=s.__getitem__)
        item_set = set(test_data[u])
        for k in KS:
            hit = rank_list[:k]
            dcg = sum(1.0 / np.log2(i + 2) for i, x in enumerate(hit)
                      if x in item_set)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(item_set), k)))
            mine[k].append(dcg / idcg if idcg > 0 else 0.0)

    for i, k in enumerate(KS):
        rn, mn = got['ndcg'][i], float(np.mean(mine[k]))
        print(f'{k:>4}{got["recall"][i]:>12.6f}{rn:>12.6f}{mn:>20.6f}'
              f'{abs(rn-mn):>12.2e}')

    print()
    for i, k in enumerate(KS):
        print(f'repo  NDCG@{k} = {got["ndcg"][i]:.6f}')
    print()
    print('repo NDCG@10 == NDCG@20 ?',
          abs(got['ndcg'][0] - got['ndcg'][1]) < 1e-12)


if __name__ == '__main__':
    main()
