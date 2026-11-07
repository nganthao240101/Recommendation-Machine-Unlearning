"""
Partition variants for the PROPOSED-vs-RANDOM utility experiment.

Everything here is a PARTITION CONSTRUCTOR only - no training, no evaluation.
Each function returns user_to_shard (int32 array of length n_users).

Variants
  signature_hyperplane : Ours Component I+II as shipped
                          (binary signature -> cosine LSH via random hyperplanes)
  signature_minhash    : Ours Component I + MinHash/LSH banding router
                          (Jaccard-style LSH, better suited to binary data)
  random_perm          : random permutation of user ids -> K shards
  random_perm_bal      : random permutation, greedy-balanced by interaction load
  random_modulo        : user_id % K (sanity / matches earlier logs)

Signature weighting
  binary     : phi(y) = 1                      (as shipped)
  log_count  : phi(y) = 1 + log(count)         (magnitude-aware profile)
"""

import math
import random
import numpy as np
from scipy.sparse import csr_matrix


# ----------------------------------------------------------------------------
# Component I - user signature
# ----------------------------------------------------------------------------
def build_signature(train_data, n_users, n_items, d=64, seed=42,
                    weighting='binary', verbose=True):
    """z_u = R * phi(x_u) / ||R * phi(x_u)||  (R: random +-1 projection, d x n_items).

    weighting:
      'binary'    phi(y) = 1
      'log_count' phi(y) = 1 + log(count)   -> encodes preference strength
    """
    if weighting not in ('binary', 'log_count'):
        raise ValueError(f'unknown weighting {weighting}')

    rng = np.random.RandomState(seed)
    R = rng.choice([-1, 1], size=(d, n_items)) / math.sqrt(d)

    rows, cols, data = [], [], []
    for user_id, items in train_data.items():
        if user_id >= n_users:
            continue
        if weighting == 'binary':
            for item in items:
                if item < n_items:
                    rows.append(user_id)
                    cols.append(item)
                    data.append(1.0)
        else:  # log_count
            counts = {}
            for item in items:
                if item < n_items:
                    counts[item] = counts.get(item, 0) + 1
            for item, c in counts.items():
                rows.append(user_id)
                cols.append(item)
                data.append(1.0 + math.log(c))

    X = csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    Rx = X @ R.T
    norms = np.maximum(np.linalg.norm(Rx, axis=1, keepdims=True), 1e-10)
    sigs = Rx / norms

    if verbose:
        nz = int((norms.flatten() > 1e-5).sum())
        print(f"    [Signature] weighting={weighting} d={d} shape={sigs.shape} non-zero={nz}")
    return sigs.astype(np.float32)


# ----------------------------------------------------------------------------
# Component II - routers (both deletion-stable: assignment never changes)
# ----------------------------------------------------------------------------
def _balanced_cell_map(B, n_shards):
    """Map B hash cells -> K shards, roughly B/K cells per shard."""
    per = B // n_shards
    extra = B % n_shards
    cell_to_shard = np.zeros(B, dtype=np.int32)
    idx = 0
    for shard in range(n_shards):
        for _ in range(per + (1 if shard < extra else 0)):
            cell_to_shard[idx] = shard
            idx += 1
    return cell_to_shard


def route_hyperplane(signatures, n_shards, seed=42):
    """Cosine LSH via random hyperplanes (the shipped Ours Component II).

    h(u) = [I[a_1^T z_u >= 0], ..., I[a_b^T z_u >= 0]];  b = ceil(log2 K)
    """
    b = math.ceil(math.log2(n_shards))
    B = 2 ** b
    rng = np.random.RandomState(seed)
    hyperplanes = rng.randn(b, signatures.shape[1])

    bits = (signatures @ hyperplanes.T) >= 0
    codes = np.zeros(signatures.shape[0], dtype=np.int64)
    for t in range(b):
        codes += bits[:, t].astype(np.int64) * (2 ** (b - 1 - t))

    cell_to_shard = _balanced_cell_map(B, n_shards)
    umap = cell_to_shard[codes].astype(np.int32)
    _print_assign(umap, n_shards, f"hyperplane LSH (b={b}, B={B})")
    return umap


def route_minhash(train_data, n_users, n_items, n_shards, num_perm=64,
                  bands=8, seed=42, weighting='binary'):
    """MinHash LSH banding router (Jaccard-style), deletion-stable.

    For each user: m_p(u) = min_{i in I_u} h_p(i) over num_perm hash functions.
    The minhash vector is split into `bands` groups; each group's tuple is
    hashed into a bucket; the tuple of band buckets is the LSH key; keys are
    mapped to shards with a balanced cell map.

    MinHash is well matched to the BINARY user-item vectors that Ours builds:
    its similarity estimate is Jaccard-like, i.e. set-overlap, whereas cosine
    LSH on a sparse binary vector is dominated by its few non-zeros.
    """
    rng = np.random.RandomState(seed)
    # (num_perm, n_items) random coefficients -> minhash via min(a*i + b) % prime
    prime = (1 << 61) - 1
    a = rng.randint(1, prime - 1, size=num_perm, dtype=np.int64)
    bcoef = rng.randint(0, prime - 1, size=num_perm, dtype=np.int64)

    # per-user minhash, computed over the user's item list only
    minhash = np.full((n_users, num_perm), prime, dtype=np.int64)
    for user_id, items in train_data.items():
        if user_id >= n_users:
            continue
        if not items:
            continue
        it = np.asarray([i for i in items if i < n_items], dtype=np.int64)
        if len(it) == 0:
            continue
        h = (np.outer(a, it) + bcoef[:, None]) % prime   # (num_perm, |I_u|)
        minhash[user_id] = h.min(axis=1)

    # banding: split num_perm into `bands` groups -> LSH key per user
    per_band = num_perm // bands
    keys = np.zeros(n_users, dtype=np.int64)
    for band in range(bands):
        sl = slice(band * per_band, (band + 1) * per_band)
        part = minhash[:, sl]
        # hash the band tuple to a 64-bit key
        band_hash = np.zeros(n_users, dtype=np.int64)
        for j in range(part.shape[1]):
            band_hash = (band_hash * 1000003 + part[:, j]) % (1 << 61)
        keys = keys * 1000003 + band_hash

    # map distinct keys -> cells -> shards (balanced)
    uniq, inv = np.unique(keys, return_inverse=True)
    B = max(len(uniq), 1)
    cell_to_shard = _balanced_cell_map(B, n_shards)
    umap = cell_to_shard[inv].astype(np.int32)
    _print_assign(umap, n_shards,
                  f"minhash LSH (perm={num_perm}, bands={bands}, cells={B})")
    return umap


def _print_assign(umap, n_shards, label):
    counts = np.bincount(umap, minlength=n_shards)
    print(f"    [Router] {label}: min={counts.min()} max={counts.max()} "
          f"mean={counts.mean():.1f}")


# ----------------------------------------------------------------------------
# Random baselines
# ----------------------------------------------------------------------------
def random_perm(n_users, n_shards, seed):
    """Random permutation of user ids -> K shards, balanced by USER COUNT."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_users)
    umap = np.zeros(n_users, dtype=np.int32)
    umap[perm] = (np.arange(n_users) % n_shards).astype(np.int32)
    return umap


def random_perm_balanced(train_data, n_users, n_shards, seed):
    """Random assignment, greedy-balanced by INTERACTION count (LPT).

    This is the load-matched control: it removes the confound that a partition
    may win simply by creating one very large shard (spec section 7).
    """
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_users)
    loads = np.zeros(n_shards, dtype=np.int64)
    umap = np.zeros(n_users, dtype=np.int32)
    # heaviest users first (LPT) -> tightest balance
    order = sorted(order, key=lambda u: -len(train_data.get(u, [])))
    for u in order:
        sid = int(np.argmin(loads))
        umap[u] = sid
        loads[sid] += len(train_data.get(u, []))
    _print_assign(umap, n_shards, "random perm (interaction-balanced)")
    return umap


def random_modulo(n_users, n_shards, seed):
    """user_id % K (sanity check; matches earlier logs)."""
    return (np.arange(n_users) % n_shards).astype(np.int32)
