"""
SAFER-E: signature-weighted ensemble of the shard models at inference.

Hard routing answers a user with the one model of his shard. Here every
shard model answers, weighted by how close the user's signature is to the
shard's users:

  w_k(u)   = softmax_k( tau * cos(z_u, c_k) )      c_k = mean signature of
                                                   shard k's users (retained)
  e_{u,k}  = the trained user vector            if k is u's shard
           = mean of shard k's vectors of u's   otherwise (fold-in: shard k
             training items                     never trained on u's data)
  s_k(u,i) = e_{u,k} . v_{i,k}, standardised per user and shard
  score    = sum_k w_k(u) s_k(u,i)

Exact unlearning is kept: no shard other than u's own ever trains on u's
data; deleting u retrains u's shard, recomputes the centroids from the
retained users (no training) and u's items are no longer anyone's fold-in.
There is no aggregator to retrain (RecEraser has one).

weights='uniform' gives every shard 1/K (the control without the signature).
tau is chosen on the validation split before any deletion
(choose_tau); tau=0 is the uniform average, tau=inf the user's own shard only
(hard routing), so validation can also decide against the ensemble.
"""

import numpy as np
import torch

TAUS = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, float('inf'))   # inf: own shard only (hard routing)


def _tensors(shard_models, n_shards):
    U, V = [], []
    for k in range(n_shards):
        m = shard_models.get_model(k)
        U.append(m.user_embedding.weight.detach())
        V.append(m.item_embedding.weight.detach())
    return U, V


def _fold_in(fold_data, n_users, n_items, V, device):
    """Mean of each shard's item vectors over every user's training items."""
    rows, cols, vals = [], [], []
    for u, its in fold_data.items():
        its = [i for i in its if i < n_items]
        if u >= n_users or not its:
            continue
        w = 1.0 / len(its)
        rows.extend([u] * len(its))
        cols.extend(its)
        vals.extend([w] * len(its))
    A = torch.sparse_coo_tensor(
        torch.as_tensor([rows, cols], dtype=torch.long, device=device),
        torch.as_tensor(vals, dtype=torch.float32, device=device),
        (n_users, n_items)).coalesce()
    return [torch.sparse.mm(A, v) for v in V]


def _weights(sigs, user_to_shard, members, n_shards, kind, tau, device):
    """(n_users, K) weights; members = users whose signature forms the centroids."""
    n_users = sigs.shape[0]
    if tau == float('inf'):                  # hard routing: the user's own shard
        own = torch.as_tensor(np.asarray(user_to_shard), dtype=torch.long, device=device)
        return torch.nn.functional.one_hot(own, n_shards).float()
    if kind == 'uniform' or tau == 0:
        return torch.full((n_users, n_shards), 1.0 / n_shards, device=device)
    Z = torch.as_tensor(sigs, dtype=torch.float32, device=device)
    C = torch.zeros((n_shards, Z.shape[1]), device=device)
    idx = torch.as_tensor(sorted(members), dtype=torch.long, device=device)
    sh = torch.as_tensor(np.asarray(user_to_shard), dtype=torch.long, device=device)[idx]
    C.index_add_(0, sh, Z[idx])
    C = C / C.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Zn = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.softmax(tau * (Zn @ C.T), dim=1)


def make_scorer(shard_models, user_to_shard, fold_data, sigs, n_shards, n_users,
                n_items, device, kind='sig', tau=10.0):
    """score_block(user_tensor) for fast_train.evaluate_topk."""
    U, V = _tensors(shard_models, n_shards)
    F = _fold_in(fold_data, n_users, n_items, V, device)
    W = _weights(sigs, user_to_shard, list(fold_data), n_shards, kind, tau, device)
    own = torch.as_tensor(np.asarray(user_to_shard), dtype=torch.long, device=device)

    def score_block(ut):
        acc = torch.zeros((len(ut), n_items), device=device)
        o = own[ut]
        for k in range(n_shards):
            e = torch.where((o == k).unsqueeze(1), U[k][ut], F[k][ut])
            s = e @ V[k].T
            s = (s - s.mean(dim=1, keepdim=True)) / s.std(dim=1, keepdim=True).clamp_min(1e-8)
            acc += W[ut, k].unsqueeze(1) * s
        return acc
    return score_block


def evaluate(shard_models, user_to_shard, excl, test_data, fold_data, sigs, n_shards,
             n_users, n_items, device, kind='sig', tau=10.0, Ks=(10, 20, 50)):
    import fast_train
    with torch.no_grad():
        sb = make_scorer(shard_models, user_to_shard, fold_data, sigs, n_shards,
                         n_users, n_items, device, kind, tau)
        users = [u for u in range(n_users) if test_data.get(u)]
        return fast_train.evaluate_topk(sb, users, excl, test_data, n_items, device,
                                        Ks, per_list=True)


def choose_tau(shard_models, user_to_shard, fit, val, sigs, n_shards, n_users, n_items,
               device, kind='sig', taus=TAUS):
    """tau with the best validation Recall@20 (no test data). -> (tau, {tau: R@20})"""
    if kind == 'uniform':
        return 0.0, {}
    scores = {}
    for t in taus:
        r = evaluate(shard_models, user_to_shard, fit, val, fit, sigs, n_shards,
                     n_users, n_items, device, kind, t, Ks=(20,))
        scores[t] = r['recall'][0]
    best = max(scores, key=lambda t: scores[t])
    return best, scores
