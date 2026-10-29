"""
FAIR COMPARISON: 4 methods + 1 control, cùng nền tảng (cùng K, cùng epochs)
=============================================================================

Trả lời câu hỏi: "Rec cao hơn Full Retrain có công bằng không?"

Mỗi config dùng CÙNG K model, CÙNG max_epochs, đánh giá AFTER trên RETAINED
(train trên RETAINED data). Chênh lệch giữa các config tách được 2 nguồn:

  ENSEMBLE effect : dùng nhiều model (dù chỉ khác số model)
  SHARDING effect : mỗi model chỉ thấy 1/K data (vụn data)

Config:
  C1 Full Single    : 1 BPRMF, full data                 (Full Retrain)
  C2 Full Ensemble  : K BPRMF, mỗi model FULL data, MEAN (CONTROL)
  C3 SISA           : K BPRMF, user%K, 1/K data, MEAN    (SISA thật)
  C4 Ours           : K BPRMF, signature assign, hard     (Ours thật)
  C5 Rec (attn)     : K shards, random, attention         (Rec thật, no WMF)

Kết luận:
  - C2 - C1  = ensemble effect (nhiều model có giúp không, khi data không vụn)
  - C3 - C2  = sharding effect (vụn data có HẠI không -> thường <= 0)
  Nếu C3 < C2 => lợi thế "utility cao" của các method shard đến từ ENSEMBLE,
  không phải từ unlearn/sharding.
"""

import os
import sys
import time
import random
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from method_1_full_retrain import BPRMF, SimpleDataLoader, evaluate_model, train_model
from method_2_sisa import DataPartitioner as SisaPartitioner, evaluate_sisa, \
    BPRMF as SISA_BPRMF, train_model as sisa_train_model
from method_4_ours import UserSignatureBuilder, DeletionStableAssignment, BPRMF as OursBPRMF, \
    evaluate_hard_routing, ShardModels
from method_3_receraser_v2 import RecEraserModel, DataPartitioner as RecPartitioner, \
    train_model as rec_train_model, train_aggregator as rec_train_agg, evaluate_model as rec_eval


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
    elif unlearn_mode == 'fewest':
        ui = sorted(all_users, key=lambda u: len(train_data[u]))
        return set(ui[:n_unlearn])
    return set(random.sample(all_users, n_unlearn))


def evaluate_mean_ensemble(models, train_data, test_data, n_users, n_items, device, Ks=[10, 20, 50]):
    """K BPRMF đơn, kết hợp bằng MEAN (dùng chung cho C1 và C2)."""
    import heapq
    pre_log = {k: [] for k in Ks}
    rec_log = {k: [] for k in Ks}
    ndcg_log = {k: [] for k in Ks}
    with torch.no_grad():
        for user in range(n_users):
            if user not in test_data or not test_data[user]:
                continue
            user_t = torch.LongTensor([user]).to(device)
            all_items = list(range(n_items))
            scores_sum = np.zeros(n_items)
            for m in models:
                m.eval()
                scores = []
                for i in range(0, n_items, 256):
                    bi = torch.LongTensor(all_items[i:i+256]).to(device)
                    s = m.predict(user_t, bi).cpu().numpy()
                    scores.extend(s.tolist())
                scores_sum += np.array(scores)
            scores = scores_sum / len(models)
            train_items = set(train_data.get(user, []))
            for it in train_items:
                scores[it] = -np.inf
            rank_list = heapq.nlargest(max(Ks), range(len(scores)), key=scores.__getitem__)
            item_set = set(test_data.get(user, []))
            for k in Ks:
                hit = rank_list[:k]
                hit_num = len(set(hit) & item_set)
                rec = hit_num / len(item_set) if len(item_set) else 0
                pre = hit_num / k
                dcg = sum(1.0 / np.log2(i + 2.0) for i, it in enumerate(hit) if it in item_set)
                idcg = sum(1.0 / np.log2(i + 2.0) for i in range(min(len(item_set), k)))
                ndcg = dcg / idcg if idcg else 0
                pre_log[k].append(pre)
                rec_log[k].append(rec)
                ndcg_log[k].append(ndcg)
    return {
        'precision': [np.nanmean(pre_log[k]) for k in Ks],
        'recall': [np.nanmean(rec_log[k]) for k in Ks],
        'ndcg': [np.nanmean(ndcg_log[k]) for k in Ks]
    }


def train_full_single(n_users, n_items, emb_dim, train, test, device, max_epochs, seed, batch_size=512, lr=0.05):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = BPRMF(n_users, n_items, emb_dim).to(device)
    train_model(m, train, test, n_users, n_items, device,
                batch_size=batch_size, lr=lr, max_epochs=max_epochs,
                early_stopping=False, patience=10, verbose=False)
    return m


def train_full_ensemble(n_users, n_items, emb_dim, train, test, device, K, max_epochs, batch_size=512, lr=0.05):
    """K model, mỗi model train trên FULL data (không shard), seed khác nhau."""
    models = []
    for sid in range(K):
        m = train_full_single(n_users, n_items, emb_dim, train, test, device,
                              max_epochs, seed=sid, batch_size=batch_size, lr=lr)
        models.append(m)
    return models


def train_sisa(n_users, n_items, emb_dim, train, test, device, K, max_epochs, batch_size=512, lr=0.05):
    """SISA thật: user%K partition, BPRMF mỗi shard, mean eval."""
    part = SisaPartitioner(n_shards=K)
    part.partition_users(train, n_users)
    part.build_shard_data(train)
    models = []
    for sid in range(K):
        m = SISA_BPRMF(n_users, n_items, emb_dim).to(device)
        sisa_train_model(m, part.shard_data[sid], n_users, n_items, device,
                         batch_size=batch_size, lr=lr, max_epochs=max_epochs, verbose=False)
        models.append(m)
    return models, part


def train_ours(n_users, n_items, emb_dim, train, test, device, K, max_epochs,
               signature_dim=64, batch_size=512, lr=0.05):
    """Ours thật: signature assign, BPRMF mỗi shard, hard-routing eval."""
    sb = UserSignatureBuilder(n_users, n_items, d=signature_dim, seed=42)
    sigs = sb.build_signature(train)
    asg = DeletionStableAssignment(n_shards=K, d=signature_dim, seed=42)
    user_to_shard = asg.assign_users(sigs)

    from method_4_ours import train_shard_model
    shard_models = ShardModels(n_users, n_items, emb_dim, K)
    for sid in range(K):
        sd = {u: it for u, it in train.items() if user_to_shard[u] == sid}
        m = OursBPRMF(n_users, n_items, emb_dim).to(device)
        shard_models.models[sid] = m
        train_shard_model(m, sd, n_items, device, batch_size=batch_size, lr=lr,
                         n_epochs=max_epochs)
    return shard_models, user_to_shard


def train_rec(n_users, n_items, emb_dim, train, test, device, K, max_epochs,
              batch_size=512, lr=0.05, use_attention=True):
    """Rec thật (no WMF): random partition, RecEraserModel attention/mean."""
    part = RecPartitioner(n_shards=K, partition_type=3, seed=42)
    part.partition_users(train, n_users, None, None)
    part.build_shard_data(train)
    model = RecEraserModel(n_users, n_items, emb_dim, num_local=K, use_attention=use_attention).to(device)
    for sid in range(K):
        if part.shard_data[sid]:
            rec_train_model(model, part, part.shard_data[sid], n_items, device,
                           batch_size=batch_size, lr=lr, max_epochs=max_epochs)
    if use_attention:
        rec_train_agg(model, train, n_users, n_items, device,
                      batch_size=batch_size, lr=lr, max_epochs=max(5, max_epochs // 2))
    return model, part


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-1m')
    p.add_argument('--emb_dim', type=int, default=64)
    p.add_argument('--n_shards', type=int, default=8)
    p.add_argument('--max_epochs', type=int, default=50)
    p.add_argument('--unlearn_ratio', type=float, default=0.01)
    p.add_argument('--unlearn_mode', default='random', choices=['random', 'fewest', 'most'])
    p.add_argument('--signature_dim', type=int, default=64)
    p.add_argument('--skip_rec', action='store_true', help='skip Rec (attention) if slow')
    args = p.parse_args()

    K = args.n_shards
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"FAIR COMPARISON | K={K} | emb_dim={args.emb_dim} | epochs={args.max_epochs} | {device}")

    data = load_data(args.dataset, batch_size=512)
    n_users, n_items = data.n_users, data.n_items
    train_data, test_data = data.train_items, data.test_set
    all_users = list(train_data.keys())
    unlearn_users = make_unlearn_users(train_data, all_users, args.unlearn_ratio, args.unlearn_mode)
    print(f"Unlearn {len(unlearn_users)} users (mode={args.unlearn_mode})")

    # AFTER: train + eval trên RETAINED
    train_r = {u: it for u, it in train_data.items() if u not in unlearn_users}
    test_r = {u: it for u, it in test_data.items() if u not in unlearn_users}
    print(f"Retained: {len(train_r)} users | Retained test: {len(test_r)}")

    results = {}

    # C1 Full Single
    print("\n=== C1: Full Single (1 model, full data) ===")
    t0 = time.time()
    c1m = [train_full_single(n_users, n_items, args.emb_dim, train_r, test_r, device, args.max_epochs, seed=0)]
    c1_time = time.time() - t0
    c1 = evaluate_mean_ensemble(c1m, train_r, test_r, n_users, n_items, device)
    results['C1 Full Single'] = (c1['recall'][0], c1_time, 1)
    print(f"  C1 R@10={c1['recall'][0]:.4f}  time={c1_time:.0f}s")

    # C2 Full Ensemble
    print(f"\n=== C2: Full Ensemble K={K} (mỗi model FULL data, MEAN) ===")
    t0 = time.time()
    c2m = train_full_ensemble(n_users, n_items, args.emb_dim, train_r, test_r, device, K, args.max_epochs)
    c2_time = time.time() - t0
    c2 = evaluate_mean_ensemble(c2m, train_r, test_r, n_users, n_items, device)
    results['C2 Full Ensemble'] = (c2['recall'][0], c2_time, K)
    print(f"  C2 R@10={c2['recall'][0]:.4f}  time={c2_time:.0f}s")

    # C3 SISA
    print(f"\n=== C3: SISA (K model, user%K, MEAN) ===")
    t0 = time.time()
    c3m, c3part = train_sisa(n_users, n_items, args.emb_dim, train_r, test_r, device, K, args.max_epochs)
    c3_time = time.time() - t0
    c3 = evaluate_sisa(c3m, train_r, test_r, n_users, n_items, device, c3part.user_to_shard, Ks=[10, 20, 50])
    results['C3 SISA'] = (c3['recall'][0], c3_time, K)
    print(f"  C3 R@10={c3['recall'][0]:.4f}  time={c3_time:.0f}s")

    # C4 Ours
    print(f"\n=== C4: Ours (K model, signature, HARD-ROUTING) ===")
    t0 = time.time()
    c4m, c4map = train_ours(n_users, n_items, args.emb_dim, train_r, test_r, device, K, args.max_epochs,
                            signature_dim=args.signature_dim)
    c4_time = time.time() - t0
    c4 = evaluate_hard_routing(c4m, c4map, train_r, test_r, n_users, n_items, device, Ks=[10, 20, 50])
    results['C4 Ours (hard)'] = (c4['recall'][0], c4_time, K)
    print(f"  C4 R@10={c4['recall'][0]:.4f}  time={c4_time:.0f}s")

    # ------------------------------------------------------------------
    # C6: Rec SAME-CAPACITY (UBP/k-means partition + K standalone BPRMF + MEAN)
    #   -> loại bỏ attention, WMF-in-model, và 8× tham số.
    #   -> chỉ còn khác biệt "partition" (UBP) so với SISA(user%K)/Ours(signature).
    #   -> capacity = K BPRMF 1-dim (cùng lớp với C3/C4), công bằng.
    #   (UBP cần user-embedding cho k-means -> dùng WMF pretrain CHỈ để partition)
    # ------------------------------------------------------------------
    print(f"\n=== C6: Rec same-capacity (UBP partition, K BPRMF, MEAN) ===")
    wmf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'data', args.dataset, 'wmf_embeddings.npz')
    if os.path.exists(wmf_path):
        w = np.load(wmf_path)
        user_emb, item_emb = w['user_embeddings'], w['item_embeddings']
        rpart = RecPartitioner(n_shards=K, partition_type=2, seed=42)
        rpart.partition_users(train_r, n_users, user_emb, item_emb)
        rpart.build_shard_data(train_r)
        t0 = time.time()
        c6m = [train_full_single(n_users, n_items, args.emb_dim, rpart.shard_data[sid], test_r,
                                 device, args.max_epochs, seed=sid) for sid in range(K)]
        c6_time = time.time() - t0
        c6 = evaluate_mean_ensemble(c6m, train_r, test_r, n_users, n_items, device)
        results['C6 Rec (same-cap)'] = (c6['recall'][0], c6_time, K)
        print(f"  C6 R@10={c6['recall'][0]:.4f}  time={c6_time:.0f}s")
    else:
        print(f"  [skip C6 - không có WMF embeddings tại {wmf_path}; "
              f"chạy pretrain_wmf.py trước]")

    # C5 Rec
    if not args.skip_rec:
        print(f"\n=== C5: Rec (K shards, random, ATTENTION, no WMF) ===")
        t0 = time.time()
        c5m, c5part = train_rec(n_users, n_items, args.emb_dim, train_r, test_r, device, K, args.max_epochs,
                                use_attention=True)
        c5_time = time.time() - t0
        c5 = rec_eval(c5m, c5part, train_r, test_r, n_users, n_items, device)
        c5_recall = c5[0] if isinstance(c5, tuple) else c5['recall'][0]
        results['C5 Rec (attn)'] = (c5_recall, c5_time, K)
        print(f"  C5 R@10={c5_recall:.4f}  time={c5_time:.0f}s")

    # BẢNG TỔNG HỢP
    print("\n" + "=" * 68)
    print(f"{'CONFIG':<20} {'R@10':>8} {'Time(s)':>8} {'#model':>7}")
    print("-" * 68)
    for name, (r, t, nm) in results.items():
        print(f"{name:<20} {r:>8.4f} {t:>8.0f} {nm:>7}")
    print("=" * 68)

    # HIỆU ỨNG
    r1 = results['C1 Full Single'][0]
    r2 = results['C2 Full Ensemble'][0]
    r3 = results['C3 SISA'][0]
    print("\nPHÂN TÍCH HIỆU ỨNG:")
    print(f"  ENSEMBLE effect  (C2 - C1) = {r2 - r1:+.4f}")
    print(f"  SHARDING effect  (C3 - C2) = {r3 - r2:+.4f}")
    r5 = results.get('C5 Rec (attn)', (None,))[0]
    r4 = results['C4 Ours (hard)'][0]
    if r5 is not None:
        print(f"  Rec attn vs SISA mean (C5 - C3) = {r5 - r3:+.4f}  (attention vs mean)")
    print(f"  Ours hard vs SISA mean (C4 - C3) = {r4 - r3:+.4f}  (hard-routing vs mean)")

    # So sánh cùng capacity (4 method thật: Full / SISA / Rec / Ours)
    r6 = results.get('C6 Rec (same-cap)', (None,))[0]
    print("\nPHÂN TÍCH CÙNG CAPACITY (K BPRMF, không attention/WMF-in-model):")
    print(f"  Full Single (1 model)   : {r1:.4f}")
    if r6 is not None:
        print(f"  Rec UBP same-capacity : {r6:.4f}   (so Full {r6 - r1:+.4f})")
    print(f"  SISA (user%K, MEAN)     : {r3:.4f}   (so Full {r3 - r1:+.4f})")
    print(f"  Ours (signature, HARD)  : {r4:.4f}   (so Full {r4 - r1:+.4f})")
    if r6 is not None:
        print(f"  Rec vs SISA           : {r6 - r3:+.4f}   (UBP vs modulo)")
        print(f"  Ours vs Rec same-cap  : {r4 - r6:+.4f}")
    print("-" * 60)
    print("  => So sánh này đã loại bỏ 8× tham số + attention + WMF-in-model của Rec;")
    print("     còn lại đúng hiệu ứng PARTITION (UBP vs modulo vs signature).")
    print("=" * 68)
    if r3 < r2:
        print("  KẾT LUẬN: Shard data (C3) < Full-ensemble (C2) => lợi thế utility đến từ")
        print("            ENSEMBLE (nhiều model), KHÔNG phải từ cơ chế sharding/unlearn.")
    else:
        print("  KẾT LUẬN: Shard data giúp thêm => sharding có đóng góp thật.")


if __name__ == '__main__':
    main()
