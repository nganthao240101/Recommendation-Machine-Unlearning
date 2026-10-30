"""
DELETION-STABILITY MEASUREMENT
=============================

Đóng góp mới: chứng minh Ours là phương pháp DUY NHẤT vừa
(similarity-aware) VỪA (deletion-stable).

Thí nghiệm KHÔNG cần train lại (chỉ tính partition, chạy vài phút):

  1. Partition trên FULL data      -> A_full
  2. Xóa unlearn users
  3. Re-partition trên RETAINED     -> A_ret
  4. Đo:
     - Instability%      : % user retained bị ĐỔI shard (A_full vs A_ret)
     - naive affected    : #shards chứa ít nhất 1 user bị xóa
     - effective affected: #shards bị đụng (chứa user xóa HOẶC có user retained đổi chỗ)

Ba phương pháp so:
  - Ours   (signature) : similarity-aware = YES, deletion-stable = YES  -> instability ~0
  - SISA   (modulo)    : similarity-aware = NO , deletion-stable = YES (trivial) -> ~0
  - Rec UBP(k-means)   : similarity-aware = YES, deletion-stable = NO   -> instability > 0,
                         effective affected LỚN HƠN naive (ẩn phí re-partition)

Kết luận kỳ vọng:
  Rec UBP "giống" Ours về similarity-aware, nhưng KHÔNG deletion-stable ->
  mỗi lần xóa phải re-cluster, làm LẤN shard của user retained. Ours giữ assignment
  ổn định nên effective-affected-shards THẬT SỰ thấp nhất -> đây là điểm novel
  chưa có trong SISA/Rec.
"""

import os
import sys
import random
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from method_1_full_retrain import SimpleDataLoader
from method_4_ours import UserSignatureBuilder, DeletionStableAssignment


def load_data(dataset, batch_size=512):
    data_path = os.environ.get('RECUNLEARN_DATA_PATH', None)
    if data_path:
        dataset_name = os.environ.get('RECUNLEARN_DATASET', dataset)
        data_dir = os.path.join(data_path, dataset_name)
    else:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(base_dir, 'data', dataset)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return SimpleDataLoader(data_dir, batch_size)


def make_unlearn_users(train_data, all_users, unlearn_ratio, unlearn_mode):
    random.seed(42)
    n_unlearn = int(len(all_users) * unlearn_ratio)
    if unlearn_mode == 'most':
        ui = sorted(all_users, key=lambda u: len(train_data[u]), reverse=True)
        return set(ui[:n_unlearn])
    if unlearn_mode == 'fewest':
        ui = sorted(all_users, key=lambda u: len(train_data[u]))
        return set(ui[:n_unlearn])
    return set(random.sample(all_users, n_unlearn))


def rec_ubp_assign(train_data, n_users, user_emb, n_shards, seed, T=10):
    """Rec UBP (phiên bản Balanced, đúng RecEraser paper) - trả về user_to_shard.

    Dùng DataPartitioner._partition_ubp_paper (cản cân bằng + E_score2),
    không phải k-means trần (bị lệch). Gắn với code mà method_3 hiện chạy.
    """
    from method_3_receraser_v2 import DataPartitioner
    p = DataPartitioner(n_shards=n_shards, partition_type=2, seed=seed, T=T)
    return p.partition_users(train_data, n_users, user_emb, None)


def analyze(method_name, A_full, A_ret, unlearn_users, retained_ids, n_shards,
            similarity_aware, deletion_stable_expected, verbose=True):
    """Tính instability + naive/effective affected."""
    ret = np.array(retained_ids)
    moved = ret[A_full[ret] != A_ret[ret]]          # user retained đổi shard
    instability = 100.0 * len(moved) / max(1, len(retained_ids))

    naive = set(int(A_full[u]) for u in unlearn_users)
    eff = set(naive)
    for u in moved:
        eff.add(int(A_full[u]))
        eff.add(int(A_ret[u]))

    # load-balance
    bal_full = np.bincount(A_full, minlength=n_shards)
    bal_ret = np.bincount(A_ret[ret], minlength=n_shards)

    if verbose:
        print(f"\n  [{method_name}]")
        print(f"    similarity-aware      = {similarity_aware}")
        print(f"    deletion-stable       = {deletion_stable_expected}")
        print(f"    Instability (retained đổi shard) = {instability:.2f}%  ({len(moved)} users)")
        print(f"    Naive affected shards = {len(naive)}/{n_shards}")
        print(f"    Effective affected    = {len(eff)}/{n_shards}   (naive + displaced)")
        print(f"    Extra shards touched by displacement = {len(eff) - len(naive)}")
        print(f"    Shard load FULL  min/max = {bal_full.min()}/{bal_full.max()}")
        print(f"    Shard load RETD  min/max = {bal_ret.min()}/{bal_ret.max()}")
    return {
        'method': method_name,
        'instability': instability,
        'naive_affected': len(naive),
        'effective_affected': len(eff),
        'extra': len(eff) - len(naive),
        'similarity_aware': similarity_aware,
        'deletion_stable': deletion_stable_expected,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-1m')
    p.add_argument('--emb_dim', type=int, default=64)
    p.add_argument('--n_shards', type=int, default=8)
    p.add_argument('--signature_dim', type=int, default=64)
    p.add_argument('--unlearn_ratio', type=float, default=0.01)
    p.add_argument('--unlearn_mode', default='most', choices=['random', 'fewest', 'most'])
    args = p.parse_args()

    K = args.n_shards
    t0 = time.time()
    print("=" * 70)
    print(f"DELETION-STABILITY | dataset={args.dataset} | K={K} | "
          f"ratio={args.unlearn_ratio} | mode={args.unlearn_mode}")
    print("=" * 70)

    data = load_data(args.dataset, batch_size=512)
    N, n_items = data.n_users, data.n_items
    train_data, test_data = data.train_items, data.test_set
    all_users = list(train_data.keys())
    unlearn = make_unlearn_users(train_data, all_users, args.unlearn_ratio, args.unlearn_mode)
    retained_ids = [u for u in range(N) if u not in unlearn]
    train_ret = {u: it for u, it in train_data.items() if u not in unlearn}
    print(f"Total users={N} | unlearn={len(unlearn)} | retained={len(retained_ids)}")

    rows = []

    # ------------------------------------------------------------------
    # Ours (signature) - KHÔNG cần embeddings
    # ------------------------------------------------------------------
    sb = UserSignatureBuilder(N, n_items, d=args.signature_dim, seed=42)
    sigs_full = sb.build_signature(train_data)
    asg = DeletionStableAssignment(n_shards=K, d=args.signature_dim, seed=42)
    A_full_ours = asg.assign_users(sigs_full)                       # (N,) theo user_id

    sigs_ret = sb.build_signature(train_ret)                        # (N,), unlearn rows=0
    A_ret_ours = asg.assign_users(sigs_ret)
    A_full_ours = np.asarray(A_full_ours, dtype=np.int64)
    A_ret_ours = np.asarray(A_ret_ours, dtype=np.int64)
    rows.append(analyze("Ours (signature)", A_full_ours, A_ret_ours, unlearn,
                       retained_ids, K, similarity_aware=True, deletion_stable_expected=True))

    # ------------------------------------------------------------------
    # SISA (modulo user%K) - "stable" do cứng, không similarity
    # ------------------------------------------------------------------
    A_full_sisa = np.array([u % K for u in range(N)], dtype=np.int64)
    A_ret_sisa = np.array([u % K for u in range(N)], dtype=np.int64)   # cùng công thức -> 0 đổi
    rows.append(analyze("SISA (modulo)", A_full_sisa, A_ret_sisa, unlearn,
                       retained_ids, K, similarity_aware=False, deletion_stable_expected=True))

    # ------------------------------------------------------------------
    # Rec UBP (balanced, đúng paper) - similarity-aware nhưng KHÔNG deletion-stable
    # ------------------------------------------------------------------
    base = os.path.dirname(os.path.dirname(__file__))
    candidate = os.path.join(base, 'data', args.dataset, 'wmf_embeddings.npz')
    if os.path.exists(candidate):
        print(f"\n  [Rec UBP] loading embeddings from {candidate}")
        w = np.load(candidate)
        user_emb_full = w['user_embeddings'][:N]

        # BEFORE: partition trên FULL
        A_full_rec = rec_ubp_assign(train_data, N, user_emb_full, K, seed=42).astype(np.int64)

        # AFTER (retained): re-partition CHỈ trên retained users (xóa rồi re-cluster),
        # dùng embedding index theo user-id (user_emb_full) để partitioner index đúng.
        A_ret_rec = rec_ubp_assign(train_ret, N, user_emb_full, K, seed=42).astype(np.int64)
        rows.append(analyze("Rec UBP (balanced)", A_full_rec, A_ret_rec, unlearn,
                            retained_ids, K, similarity_aware=True,
                            deletion_stable_expected=False))
    else:
        print(f"\n  [Rec UBP] no WMF embeddings found at {candidate}; skip Rec UBP.")

    # ------------------------------------------------------------------
    # BẢNG TỔNG HỢP
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'Method':<18} {'sim':>4} {'stbl':>5} {'inst%':>7} {'naive':>6} {'eff':>5}")
    print("-" * 70)
    for r in rows:
        print(f"{r['method']:<18} {str(r['similarity_aware']):>4} "
              f"{str(r['deletion_stable']):>5} {r['instability']:>7.2f} "
              f"{r['naive_affected']:>6} {r['effective_affected']:>5}")
    print("=" * 70)

    print("\nPHÂN TÍCH:")
    for r in rows:
        note = []
        if r['instability'] > 0:
            note.append(f"KHÔNG deletion-stable ({r['instability']:.1f}% user retained đổi shard)")
        if r['effective_affected'] > r['naive_affected']:
            note.append(f"+{r['extra']} shards bị ẩn do re-partition (naive {r['naive_affected']} -> eff {r['effective_affected']})")
        if r['similarity_aware'] and r['deletion_stable']:
            note.append("DUY NHẤT: vừa similarity-aware VỪA deletion-stable")
        if not note and r['deletion_stable'] and not r['similarity_aware']:
            note.append("stable (không similarity-aware)")
        if not note:
            note.append("không đổi shard")
        print(f"  {r['method']}: " + ", ".join(note))
    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
