"""
Independent check of the sharded evaluator's NDCG.

Two things looked wrong in the logs and are worth confirming with code rather
than with a guess:

  1. NDCG@20 came out identical to NDCG@10 for every sharded arm, while the
     GLOBAL arm (which uses a different evaluation routine) did not.
  2. The two NDCG implementations in the repo differ in how they compute the
     ideal DCG: method_4_ours.evaluate_hard_routing uses len(item_pos), the
     length of the test LIST, while exp_partition_quality's global arm uses
     len(item_set), the number of DISTINCT test items. If the test list
     contains duplicates, the two disagree.

This script re-implements Recall and NDCG from scratch, computes them over a
ranking it builds itself, and compares against evaluate_hard_routing on the
same synthetic data. If the sharded NDCG@20 really equalled NDCG@10, the two
implementations could not agree here.

Run:  python code/check_ndcg_evaluator.py
"""

import os
import sys
import heapq

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KS = [10, 20, 50]


class _FakeModel:
    """Stands in for a BPRMF shard model with a fixed score vector."""

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float64)

    def predict_user_items(self, user, items):
        import torch
        idx = items.cpu().numpy() if hasattr(items, 'cpu') else np.asarray(items)
        return torch.tensor(self.scores[idx])


class _FakeShardModels:
    def __init__(self, per_shard_scores):
        self.per_shard_scores = per_shard_scores

    def get_model(self, sid):
        return _FakeModel(self.per_shard_scores[sid])


def independent_metrics(scores, train_items, test_items, Ks=KS):
    """Recall and NDCG computed from scratch, no shared code with the repo."""
    scores = np.array(scores, dtype=np.float64).copy()
    for it in set(train_items):
        scores[it] = -np.inf
    rank_list = heapq.nlargest(max(Ks), range(len(scores)), key=scores.__getitem__)
    item_set = set(test_items)
    out = {}
    for k in Ks:
        hit = rank_list[:k]
        hits = [x for x in hit if x in item_set]
        rec = len(hits) / len(item_set) if item_set else 0.0
        dcg = sum(1.0 / np.log2(i + 2) for i, x in enumerate(hit) if x in item_set)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(item_set), k)))
        out[k] = (rec, dcg / idcg if idcg > 0 else 0.0)
    return out


def main():
    from method_4_ours import evaluate_hard_routing

    rng = np.random.RandomState(0)
    n_items, n_users, K = 200, 30, 4
    shard_of = {u: u % K for u in range(n_users)}

    train, test = {}, {}
    for u in range(n_users):
        train[u] = list(rng.choice(n_items, 20, replace=False))
        test[u] = list(rng.choice(n_items, 12, replace=False))

    # one score vector per shard, so different users in a shard share a model
    per_shard = {s: rng.rand(n_items) for s in range(K)}
    fake = _FakeShardModels(per_shard)

    got = evaluate_hard_routing(fake, shard_of, train, test,
                                n_users, n_items, 'cpu', Ks=KS)

    mine = {k: {'rec': [], 'ndcg': []} for k in KS}
    for u in range(n_users):
        model = fake.get_model(shard_of[u])
        scores = model.scores.copy()
        s = np.array(scores)
        for it in set(train[u]):
            s[it] = -np.inf
        m = independent_metrics(s, [], test[u])
        for k in KS:
            mine[k]['rec'].append(m[k][0])
            mine[k]['ndcg'].append(m[k][1])

    print(f'{"K":>4}{"repo R":>10}{"mine R":>10}{"repo NDCG":>12}{"mine NDCG":>12}'
          f'{"|dR|":>10}{"|dNDCG|":>10}')
    ok = True
    for i, k in enumerate(KS):
        r_repo = got['recall'][i]
        n_repo = got['ndcg'][i]
        r_mine = float(np.mean(mine[k]['rec']))
        n_mine = float(np.mean(mine[k]['ndcg']))
        dr = abs(r_repo - r_mine)
        dn = abs(n_repo - n_mine)
        if dr > 1e-9 or dn > 1e-9:
            ok = False
        print(f'{k:>4}{r_repo:>10.6f}{r_mine:>10.6f}{n_repo:>12.6f}{n_mine:>12.6f}'
              f'{dr:>10.2e}{dn:>10.2e}')

    print()
    repo_n = {k: got['ndcg'][i] for i, k in enumerate(KS)}
    mine_n = {k: float(np.mean(mine[k]['ndcg'])) for k in KS}
    print('repo  NDCG@10 == NDCG@20 ?',
          abs(repo_n[10] - repo_n[20]) < 1e-12,
          f'({repo_n[10]:.6f} vs {repo_n[20]:.6f})')
    print('mine  NDCG@10 == NDCG@20 ?',
          abs(mine_n[10] - mine_n[20]) < 1e-12,
          f'({mine_n[10]:.6f} vs {mine_n[20]:.6f})')
    print()
    print('VERDICT:', 'evaluator matches an independent implementation'
          if ok else 'MISMATCH - the sharded evaluator disagrees with the '
                     'independent one')


if __name__ == '__main__':
    main()
