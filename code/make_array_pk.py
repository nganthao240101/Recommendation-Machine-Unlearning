"""
ĐỔ `user_pretrain.pk` / `item_pretrain.pk` từ DICT {id: vector} -> ARRAY (n, dim).
================================================================================

Lý do: code GỐC tác giả `code/RecEraser_BPR_pytorch.py` (--pretrain 1) đọc
    uidW = pickle.load(open('user_pretrain.pk'))
    user_emb_expanded = np.concatenate([uidW]*part_num, axis=1)
Nó GIẢ ĐỊNH `.pk` là mảng 2-D (n, dim) trong đó **dòng `i` = entity id `i`**.
Nhưng `.pk` của bạn là DICT {id: vector} (do bộ utility TF gốc sinh ra)
-> `np.concatenate([dict]*K)` crash, HOẶC (nếu đã từng "đổ" ngây thơ)
dòng index lệch (vd yelp "index 31668 out of bounds for size 31668").

Số dòng phải đúng = `max_id + 1` (0-indexed), GIỐNG hệt cách
`utility/load_data.py` tính n_users/n_items (line 53-67). Script này:

  - đọc train.txt/test.txt -> n_users, n_items (max_id+1, 0-indexed).
  - lấy dict gốc từ `.bak_dict` (nếu có) hoặc `.pk` (nếu vẫn dict).
  - build array (n, dim): **dòng id = vector của id, thiếu -> zero-fill**.
  - BACKUP dict gốc lên `<name>.pk.bak_dict` (nếu chưa có), rồi ghi array
    trở lại `<name>.pk` (đúng định dạng code tác giả cần).

KHÔNG sửa code tác giả. Chỉ chuẩn bị DATA.

Cách dùng (chạy 1 lần mỗi dataset):
    python code/make_array_pk.py --dataset ml-1m
    python code/make_array_pk.py --dataset yelp2018
    python code/make_array_pk.py --dataset ml-1m --dry-run   # chỉ xem

Sau đó `RecEraser_BPR_pytorch.py --pretrain 1` chạy được (đọc `.pk` array).
"""

import os
import sys
import pickle
import argparse

import numpy as np


# ---------------------------------------------------------------------------
# n_users / n_items - GIỐNG utility/load_data.py (max_id + 1, 0-indexed)
# ---------------------------------------------------------------------------
def read_n_entities(ddir):
    n_users = 0
    n_items = 0
    for fname, in_test in (('train.txt', False), ('test.txt', True)):
        fp = os.path.join(ddir, fname)
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            for line in f:
                line = line.strip('\n')
                if not line:
                    continue
                parts = line.split(' ')
                try:
                    uid = int(parts[0])
                    items = [int(i) for i in parts[1:] if i]
                except (ValueError, IndexError):
                    continue
                n_users = max(n_users, uid + 1)
                if items:
                    n_items = max(n_items, max(items) + 1)
    return n_users, n_items


def _dict_source(pk_path, bak_path):
    """Chọn dict "chính chủ": ưu tiên `.bak_dict` (đã được backup),
    nếu không, đọc `.pk` (nếu vẫn còn dict). Trả dict hoặc None."""
    if os.path.exists(bak_path):
        with open(bak_path, 'rb') as f:
            o = pickle.load(f)
        if isinstance(o, dict):
            return o
    with open(pk_path, 'rb') as f:
        o = pickle.load(f)
    if isinstance(o, dict):
        return o
    return None


def build_array(entity_dict, n_entities, dim, which):
    """entity_dict: {id: vector}. Trả array (n_entities, dim),
    dòng id = vector của id, id không có trong dict -> zero."""
    arr = np.zeros((n_entities, dim), dtype=np.float32)
    if not entity_dict:
        print(f"    {which}: dict RỖNG -> array zero-fill (n={n_entities}, dim={dim})")
        return arr
    filled = 0
    for uid, vec in entity_dict.items():
        i = int(uid)
        if 0 <= i < n_entities:
            arr[i] = np.asarray(vec, dtype=np.float32)
            filled += 1
    # dim: lấy từ vector đầu (nếu dim chưa biết trước)
    print(f"    {which}: {filled}/{len(entity_dict)} id khớp [0,{n_entities}) -> "
          f"array {arr.shape}")
    return arr


def _detect_dim(d):
    if not d:
        return None
    first = next(iter(d.values()))
    return int(np.asarray(first).shape[0])


def convert_one(pk_path, bak_path, ddir, n_users, n_items, dry_run):
    which = 'user' if 'user' in os.path.basename(pk_path) else 'item'
    target_n = n_users if which == 'user' else n_items

    d = _dict_source(pk_path, bak_path)
    # xem file .pk hiện tại
    with open(pk_path, 'rb') as f:
        cur = pickle.load(f)

    if d is None:
        # không có dict (đã là array thật, hoặc mất)
        if isinstance(cur, np.ndarray):
            ok = cur.shape[0] == target_n
            print(f"    {which}: .pk ĐÃ là array {cur.shape} (n cần {target_n})"
                  + ("" if ok else "  ⚠️ SAI số dòng! (chưa sửa được, cần dict)"))
            return 'already' if ok else 'badsize'
        print(f"    {which}: không có dict, .pk không phải array -> KHÔNG xử lý được")
        return 'unknown'

    dim = _detect_dim(d)
    if dim is None:
        dim = cur.shape[1] if isinstance(cur, np.ndarray) and cur.ndim == 2 else 64

    arr = build_array(d, target_n, dim, which)

    if dry_run:
        print(f"    [dry-run] {which}: sẽ ghi array {arr.shape} vào .pk")
        return 'dry'

    # backup dict gốc (nếu dict đang nằm trong .pk và chưa có backup)
    if isinstance(cur, dict) and not os.path.exists(bak_path):
        import shutil
        shutil.copy2(pk_path, bak_path)
        print(f"    {which}: backup dict gốc -> {os.path.basename(bak_path)}")
    with open(pk_path, 'wb') as f:
        pickle.dump(arr, f)
    return 'converted'


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

    n_users, n_items = read_n_entities(ddir)
    print(f"[make_array_pk] {ddir}  (n_users={n_users}, n_items={n_items})")

    upk = os.path.join(ddir, 'user_pretrain.pk')
    ipk = os.path.join(ddir, 'item_pretrain.pk')
    for name, pk in (('user', upk), ('item', ipk)):
        if not os.path.exists(pk):
            print(f"    {name}: {os.path.basename(pk)} KHÔNG CÓ -> skip")
            continue
        convert_one(pk, pk + '.bak_dict', ddir, n_users, n_items, args.dry_run)

    if args.dry_run:
        print("dry-run, không ghi file.")
    else:
        print("Xong. Giờ 'RecEraser_BPR_pytorch.py --pretrain 1' chạy được "
              "(.pk là array đúng n_entities).")


if __name__ == '__main__':
    main()
