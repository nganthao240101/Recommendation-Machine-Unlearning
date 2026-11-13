"""
WMF as a drop-in replacement for the BPRMF backbone of every arm.

Everything else in an experiment stays as it is (the partitions, the epochs,
the learning rate, RecEraser's aggregator, the evaluation); only the model
and its loss change, so the methods can be compared on a second backbone.

The model and loss follow the authors' WMF.py / RecEraser_WMF.py:

  loss = c0 * sum( (I^T I) * (U_b^T U_b) )                 every item, weight c0
       + sum over the batch's positives of (1 - c0) r^2 - 2 r,  r = u . i

  - non-sampling: the negatives are all items, weighted by c0 (negative_weight)
  - one batch = a set of users with all their items (256 users, readme.txt)
  - dropout on the batch's user vectors, keep probability 0.7 (args.dropout)
  - embeddings drawn from N(0, 0.01^2), Adagrad (initial accumulator 1e-8),
    no L2 term (lambda_bilinear = [0, 0] in WMF.py)
  - c0 = 0.2 on ml-1m, 0.05 on yelp2018 (readme.txt)

One departure, the same for every arm: RecEraser_WMF.py puts only the
shard's own items in a shard's negative term; here every shard model, like
the single WMF of WMF.py, sees every item, so all arms train one model.

A model scores u . i exactly as BPRMF does, so the evaluation is unchanged.
"""

import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CFG = {'backbone': 'bprmf', 'c0': 0.2, 'keep': 0.7, 'batch_users': 256}
C0_DEFAULT = {'ml-1m': 0.2, 'yelp2018': 0.05}


def configure(backbone='bprmf', c0=None, keep=0.7, batch_users=256, dataset=None):
    CFG['backbone'] = backbone.lower()
    CFG['c0'] = c0 if c0 is not None else C0_DEFAULT.get(dataset, 0.2)
    CFG['keep'] = keep
    CFG['batch_users'] = batch_users


def use_wmf():
    return CFG['backbone'] == 'wmf'


def describe():
    if not use_wmf():
        return 'bprmf (BPR loss, 1 sampled negative, L2 0.01)'
    return (f"wmf (non-sampling loss, c0={CFG['c0']}, dropout keep={CFG['keep']}, "
            f"{CFG['batch_users']} users per batch, init N(0,0.01))")


class WMF(nn.Module):
    """The WMF model: a user table and an item table, scored by u . i."""

    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
        self.n_users, self.n_items, self.emb_dim = n_users, n_items, emb_dim
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_embedding.weight, 0.0, 0.01)
        nn.init.normal_(self.item_embedding.weight, 0.0, 0.01)

    @torch.no_grad()
    def predict_user_items(self, user_id, item_ids):
        u = self.user_embedding.weight[user_id]
        return (u * self.item_embedding(item_ids)).sum(dim=-1)


def wmf_loss(u_e, items, rows, cols, c0):
    """u_e: (B, D) batch users (after dropout); items: (N, D) every item;
    rows/cols: the batch's positives as (row in the batch, item id)."""
    gram = (items.t() @ items) * (u_e.t() @ u_e)
    r = (u_e[rows] * items[cols]).sum(-1)
    return c0 * gram.sum() + ((1.0 - c0) * r * r - 2.0 * r).sum()


def train_wmf(user_emb, item_emb, params, data, device, lr, n_epochs,
              log_every=None, log_fmt='        Epoch {ep}: loss={loss:.4f}',
              epoch_end=None, optimizer=None, clip=None, skip_nonfinite=False):
    """Train with the WMF loss over `data` ({user: [items]}).

    user_emb(user_tensor) -> (B, D) and item_emb() -> (N, D) give the vectors
    being trained, so the same loop serves a plain model, a slice of the
    RecEraser table and RecEraser's aggregated vectors. Returns (loss of the
    last epoch, interactions in `data`) - the interaction count keeps the
    work column comparable with the BPRMF runs.
    """
    users = [u for u in sorted(data) if data[u]]
    if not users:
        return 0.0, 0
    lens = [len(data[u]) for u in users]
    U = torch.as_tensor(users, dtype=torch.long, device=device)
    indptr = torch.zeros(len(users) + 1, dtype=torch.long, device=device)
    indptr[1:] = torch.cumsum(torch.as_tensor(lens, device=device), 0)
    indices = torch.as_tensor([i for u in users for i in data[u]], dtype=torch.long,
                              device=device)
    opt = optimizer or torch.optim.Adagrad(params, lr=lr, initial_accumulator_value=1e-8)
    gen = torch.Generator(device=device)
    gen.manual_seed(random.getrandbits(63))
    bs, c0, keep = CFG['batch_users'], CFG['c0'], CFG['keep']
    n = len(users)
    total = torch.zeros((), device=device)
    for ep in range(n_epochs):
        perm = torch.randperm(n, device=device, generator=gen)
        total = torch.zeros((), device=device)
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            starts = indptr[idx]
            ln = indptr[idx + 1] - starts
            rows = torch.repeat_interleave(torch.arange(len(idx), device=device), ln)
            offs = (torch.arange(int(ln.sum()), device=device)
                    - torch.repeat_interleave(torch.cumsum(ln, 0) - ln, ln))
            cols = indices[torch.repeat_interleave(starts, ln) + offs]
            u_e = F.dropout(user_emb(U[idx]), p=1.0 - keep, training=True)
            loss = wmf_loss(u_e, item_emb(), rows, cols, c0)
            if skip_nonfinite and not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(clip, 1.0)
            opt.step()
            total += loss.detach()
        if log_every and (ep + 1) % log_every == 0:
            print(log_fmt.format(ep=ep + 1, loss=float(total) / max(1, n)))
        if epoch_end is not None:
            epoch_end(ep)
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()
    return float(total), int(indptr[-1])


def train_wmf_model(model, data, device, lr, n_epochs, log_every=20, epoch_end=None):
    """train_shard_model's job for a WMF model: (loss, seconds, interactions)."""
    t0 = time.time()
    loss, n = train_wmf(lambda u: model.user_embedding(u),
                        lambda: model.item_embedding.weight,
                        model.parameters(), data, device, lr, n_epochs,
                        log_every=log_every, epoch_end=epoch_end)
    return loss, time.time() - t0, n
