"""
GPU-resident training and evaluation for the experiment arms.

The original loops built every batch from Python tuples, copied it to the GPU
and read the loss back after each step. With a 64-dim matrix factorisation
the GPU work per batch is microseconds, so one CPU core sat at 100% feeding
it and the GPU waited. This module keeps the same algorithm and changes only
where the work happens:

  training    - the (user, pos, neg) triples are drawn exactly as before (one
                negative per positive, rejected while it is a training item of
                that user, with Python's random), then moved to the GPU once;
                each epoch shuffles them with torch.randperm and walks the same
                floor(n / batch_size) batches; the loss is read back only when
                it is printed
  evaluation  - scores for a block of users at once (user embeddings x all
                item embeddings), the user's training items set to -inf, and
                torch.topk instead of a heapq per user; recall and NDCG are
                computed exactly as before

Batch size, epochs, learning rate, optimiser, loss and negative sampling are
unchanged. The random stream is not: the shuffles come from torch instead of
Python, so a seed gives a statistically equivalent run, not an identical one.

RECUNLEARN_ENGINE=python switches every caller back to the original loops,
which is how the two are compared.
"""

import os
import random

import numpy as np
import torch

ENGINE = os.environ.get('RECUNLEARN_ENGINE', 'gpu').lower()
EVAL_BLOCK = 1024           # users scored at once


def use_gpu_engine():
    return ENGINE != 'python'


def engine_name():
    return ('gpu (resident batches, torch.topk evaluation)'
            if use_gpu_engine() else 'python (original per-batch loops)')


# ----------------------------------------------------------------- training
def draw_samples(data, n_items):
    """One (user, pos, neg) per training interaction, as the original loops
    draw them: the negative is redrawn while it is one of the user's items."""
    us, ps, ns = [], [], []
    for user, items in data.items():
        for pos_item in items:
            neg_item = random.randint(0, n_items - 1)
            while neg_item in items:
                neg_item = random.randint(0, n_items - 1)
            us.append(user)
            ps.append(pos_item)
            ns.append(neg_item)
    return us, ps, ns


def train_epochs(step, samples, device, batch_size, n_epochs, log_every=None,
                 log_fmt='        Epoch {ep}: loss={loss:.4f}', epoch_end=None):
    """Run n_epochs over the samples on the GPU.

    step(users, pos, neg) -> loss tensor; it does the forward, backward and
    optimiser step, and may return None for a batch it skipped. epoch_end(ep)
    is called after each epoch. Returns (total loss of the last epoch, batches
    per epoch).
    """
    us, ps, ns = samples
    n = len(us)
    if n == 0:
        return 0.0, 0
    U = torch.as_tensor(us, dtype=torch.long, device=device)
    P = torch.as_tensor(ps, dtype=torch.long, device=device)
    N = torch.as_tensor(ns, dtype=torch.long, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(random.getrandbits(63))      # follows the run's seed
    nb = max(1, n // batch_size)
    total = None
    for ep in range(n_epochs):
        perm = torch.randperm(n, device=device, generator=gen)
        total = torch.zeros((), device=device)
        for b in range(nb):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            if idx.numel() == 0:
                continue
            loss = step(U[idx], P[idx], N[idx])
            if loss is not None:
                total += loss.detach()
        if log_every and (ep + 1) % log_every == 0:
            print(log_fmt.format(ep=ep + 1, loss=float(total) / nb))
        if epoch_end is not None:
            epoch_end(ep)
    return float(total), nb


def sync(device):
    """Wait for the GPU, so a wall-clock time covers the work it queued."""
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


# --------------------------------------------------------------- evaluation
def evaluate_topk(score_block, users, train_data, test_data, n_items, device,
                  Ks=(10, 20, 50), block=EVAL_BLOCK, per_list=False):
    """Recall@K and NDCG@K for `users` (those with test items).

    score_block(user_tensor) -> (len(users), n_items) scores on the device.
    The metrics are computed exactly as in the original per-user loops. Most
    of those divide by the number of DISTINCT test items; the hard-routing one
    divides by the length of the test list, which per_list=True reproduces
    (the two agree whenever a test list has no repeated item).
    """
    kmax = max(Ks)
    pre = {k: [] for k in Ks}
    rec = {k: [] for k in Ks}
    ndcg = {k: [] for k in Ks}
    discounts = 1.0 / np.log2(np.arange(2, kmax + 2))
    users = [u for u in users if test_data.get(u)]
    with torch.no_grad():
        for start in range(0, len(users), block):
            ub = users[start:start + block]
            ut = torch.as_tensor(ub, dtype=torch.long, device=device)
            scores = score_block(ut).float()
            rows, cols = [], []
            for r, u in enumerate(ub):
                its = set(train_data.get(u, []))
                rows.extend([r] * len(its))
                cols.extend(its)
            if rows:
                scores[torch.as_tensor(rows, device=device),
                       torch.as_tensor(cols, device=device)] = -float('inf')
            top = torch.topk(scores, min(kmax, n_items), dim=1).indices.cpu().numpy()
            for r, u in enumerate(ub):
                item_set = set(test_data[u])
                n_rel = len(test_data[u]) if per_list else len(item_set)
                hit = np.array([x in item_set for x in top[r]], dtype=float)
                for k in Ks:
                    h = hit[:k]
                    pre[k].append(h.sum() / k)
                    rec[k].append(h.sum() / n_rel)
                    dcg = float((h * discounts[:len(h)]).sum())
                    idcg = float(discounts[:min(n_rel, k)].sum())
                    ndcg[k].append(dcg / idcg if idcg > 0 else 0.0)
    return {
        'precision': [float(np.nanmean(pre[k])) if pre[k] else 0.0 for k in Ks],
        'recall': [float(np.nanmean(rec[k])) if rec[k] else 0.0 for k in Ks],
        'ndcg': [float(np.nanmean(ndcg[k])) if ndcg[k] else 0.0 for k in Ks],
    }
