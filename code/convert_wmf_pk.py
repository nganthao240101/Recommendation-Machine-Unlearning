"""
CONVERT user/item_pretrain.pk  ->  wmf_embeddings.npz
=====================================================

method_3_receraser_v2.py (Rec UBP), benchmark_fair_compare.py (C6) và
measure_stability.py đọc file `data/<dataset>/wmf_embeddings.npz` với
nghiệp vụ:
    data = np.load(path)
    user_emb = data['user_embeddings']
    item_emb = data['item_embeddings']

Nhưng nhiều bộ data (ml-1m, yelp2018...) chỉ có `user_pretrain.pk` /
`item_pretrain.pk` (pickle). Script này chuyển 2 file .pk đó thành .npz
đúng định dạng, để các method Rec UBP / C6 chạy được BẢM (không cần
train lại WMF, không crash path).

Cách dùng (chạy 1 lần cho 1 dataset):
    python code/convert_wmf_pk.py --dataset yelp2018
    python code/convert_wmf_pk.py --dataset ml-1m
    python code/convert_wmf_pk.py --dataset ml-1m --emb_dim 64 --check
"""

import os
import sys
import pickle
import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-1m')
    p.add_argument('--data_root', default=None,
                   help='thư mục cha chứa data/<dataset>/ (mặc định: <repo>/data)')
    p.add_argument('--emb_dim', type=int, default=64,
                   help='chỉ để kiểm tra shape (không ép buộc)')
    p.add_argument('--check', action='store_true',
                   help='chỉ kiểm tra shape so với n_users/n_items từ train.txt')
    args = p.parse_args()

    if args.data_root is None:
        args.data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    ddir = os.path.join(args.data_root, args.dataset)

    upk = os.path.join(ddir, 'user_pretrain.pk')
    ipk = os.path.join(ddir, 'item_pretrain.pk')
    out = os.path.join(ddir, 'wmf_embeddings.npz')

    if not (os.path.exists(upk) and os.path.exists(ipk)):
        print(f"[LỖI] Không tìm thấy .pk tại:\n  {upk}\n  {ipk}")
        print("       -> dataset này dùng WMF khác (hãy chạy pretrain_wmf.py).")
        sys.exit(1)

    with open(upk, 'rb') as f:
        user_emb = pickle.load(f)
    with open(ipk, 'rb') as f:
        item_emb = pickle.load(f)

    user_emb = np.asarray(user_emb, dtype=np.float32)
    item_emb = np.asarray(item_emb, dtype=np.float32)
    print(f"[read] user_emb {user_emb.shape}  item_emb {item_emb.shape}  (dtype float32)")

    # (tùy chọn) đối chiếu với train.txt
    train = os.path.join(ddir, 'train.txt')
    if args.check and os.path.exists(train):
        mxu = mxi = 0
        for line in open(train):
            parts = line.split()
            if not parts:
                continue
            mxu = max(mxu, int(parts[0]))
            mxi = max(mxi, max(int(x) for x in parts[1:]))
        nu, ni = mxu + 1, mxi + 1
        ok_u = user_emb.shape[0] >= nu
        ok_i = item_emb.shape[0] >= ni
        print(f"[check] n_users={nu} n_items={ni} | user ok={ok_u} item ok={ok_i}")
        if not (ok_u and ok_i):
            print("       => embedding NHỎ hơn n_users/n_items! Cần WMF đúng kích cỡ.")
            sys.exit(1)

    if args.check:
        print("check-only, không ghi file.")
        return

    if os.path.exists(out):
        print(f"[skip] {out} đã tồn tại.")
    else:
        np.savez(out, user_embeddings=user_emb, item_embeddings=item_emb,
                 n_users=user_emb.shape[0], n_items=item_emb.shape[0],
                 emb_dim=user_emb.shape[1] if user_emb.ndim > 1 else 0)
        print(f"[OK] Đã tạo {out}")
        print("     -> Rec UBP / C6 / measure_stability giờ có thể chạy (kể cả inline).")


if __name__ == '__main__':
    main()
