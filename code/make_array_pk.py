"""
ĐỔ `user_pretrain.pk` / `item_pretrain.pk` từ DICT {id: vector} -> ARRAY (n, dim).
================================================================================

Lý do: code GỐC tác giả `code/RecEraser_BPR_pytorch.py` (--pretrain 1) đọc
    uidW = pickle.load(open('user_pretrain.pk'))
    user_emb_expanded = np.concatenate([uidW]*part_num, axis=1)
Nó GIẢ ĐỊNH `.pk` là mảng 2-D (n, dim). Nhưng `.pk` của bạn là DICT
{user_id: vector} (do bộ utility TF gốc sinh ra) -> `np.concatenate([dict]*K)`
CRASH.

Script này:
  - đọc `.pk` dict, "đổ" thành mảng (n, dim) (sắp theo key -> index).
  - BACKUP file gốc lên `user_pretrain.pk.bak_dict` (an toàn, chạy lại được).
  - ghi mảng array trả lại `user_pretrain.pk` (đúng định dạng code gốc cần).

KHÔNG sửa code tác giả. Chỉ chuẩn bị DATA.

Cách dùng (chạy 1 lần mỗi dataset):
    python code/make_array_pk.py --dataset ml-1m
    python code/make_array_pk.py --dataset yelp2018
    python code/make_array_pk.py --dataset ml-1m --dry-run   # chỉ xem, không ghi

Sau đó `RecEraser_BPR_pytorch.py --pretrain 1` chạy được (đọc `.pk` array).
"""

import os
import sys
import pickle
import argparse
import shutil

import numpy as np


def _as_2d_array(obj, n_rows=None):
    """Chuyển obj (dict {id:vec} HOẶC (n,dim) array) -> mảng float32 (n, dim).

    dict: sắp key tăng dần -> dòng. Giả định id = 0..n-1 (0-indexed, đúng
    bộ dataset này). Trả về (n, dim).
    """
    if isinstance(obj, dict):
        keys = sorted(int(k) for k in obj.keys())
        arr = np.asarray([obj[k] for k in keys], dtype=np.float32)
        return arr
    # đã là mảng
    arr = np.asarray(obj, dtype=np.float32)
    return arr


def convert_one(pk_path, backup_path):
    with open(pk_path, 'rb') as f:
        obj = pickle.load(f)

    if isinstance(obj, np.ndarray):
        print(f"    {os.path.basename(pk_path)}: ĐÃ là array {obj.shape} -> skip (không ghi)")
        return 'skip'
    if isinstance(obj, dict):
        arr = _as_2d_array(obj)
        # backup dict gốc (an toàn)
        if not os.path.exists(backup_path):
            shutil.copy2(pk_path, backup_path)
            print(f"    {os.path.basename(pk_path)}: dict -> array {arr.shape}  "
                  f"(backup: {os.path.basename(backup_path)})")
        else:
            print(f"    {os.path.basename(pk_path)}: dict -> array {arr.shape}  "
                  f"(backup đã có: {os.path.basename(backup_path)})")
        # viết array trở lại đúng tên .pk gốc (định dạng code tác giả cần)
        with open(pk_path, 'wb') as f:
            pickle.dump(arr, f)
        return 'converted'
    raise TypeError(f"{pk_path}: kiểu dữ liệu không biết: {type(obj)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-1m')
    p.add_argument('--data_root', default=None)
    p.add_argument('--dry_run', action='store_true',
                   help='chỉ kiểm tra, không ghi file')
    args = p.parse_args()

    if args.data_root is None:
        args.data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    ddir = os.path.join(args.data_root, args.dataset)
    if not os.path.isdir(ddir):
        print(f"[LỖI] không có thư mục {ddir}")
        sys.exit(1)

    print(f"[make_array_pk] {ddir}")
    upk = os.path.join(ddir, 'user_pretrain.pk')
    ipk = os.path.join(ddir, 'item_pretrain.pk')
    for name, pk in (('user', upk), ('item', ipk)):
        if not os.path.exists(pk):
            print(f"    {name}: {os.path.basename(pk)} KHÔNG CÓ -> skip")
            continue
        if args.dry_run:
            with open(pk, 'rb') as f:
                obj = pickle.load(f)
            kind = 'dict' if isinstance(obj, dict) else ('array' if isinstance(obj, np.ndarray) else type(obj).__name__)
            print(f"    {name}: {os.path.basename(pk)} hiện là {kind}"
                  + (f" shape~{np.asarray([obj[k] for k in list(obj)[:1]]).shape[1:] if kind=='dict' else ''}" if kind=='dict' else ''))
            continue
        bak = pk + '.bak_dict'
        convert_one(pk, bak)

    if args.dry_run:
        print("dry-run, không ghi file.")
    else:
        print("Xong. Giờ 'RecEraser_BPR_pytorch.py --pretrain 1' chạy được.")


if __name__ == '__main__':
    main()
