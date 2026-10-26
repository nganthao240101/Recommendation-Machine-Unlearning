"""
FAIR COMPARISON: Full Retrain vs RecEraser-style ensemble
============================================================

Mục tiêu: tách bạch 2 nguồn của việc "RecEraser đạt recall cao hơn Full Retrain"
          để so sánh CÔNG BẰNG (cùng model, cùng epochs, cùng ngân sách compute).

Ba cấu hình (cùng emb_dim, cùng max_epochs, cùng kiểu aggregation = MEAN):
  C1. Full Single      : 1  BPRMF train trên TOÀN BỘ data
  C2. Full Ensemble    : K BPRMF, mỗi model train trên TOÀN BỘ data, kết hợp MEAN
  C3. Shard Ensemble   : K BPRMF, mỗi model train trên shard 1/K của nó, kết hợp MEAN

Compute: C2 và C3 cùng train K model (cùng ngân sách). C1 chỉ train 1 model.

So sánh (đánh giá AFTER trên RETAINED test, train trên RETAINED data):
  Ensemble effect  = Recall(C2) - Recall(C1)   (liệu dùng nhiều model có giúp?)
  Sharding effect  = Recall(C3) - Recall(C2)   (liệu shard data có giúp thêm? thường <= 0)

Nếu Sharding effect ~ 0 hoặc âm => lợi thế utility của RecEraser đến từ ENSEMBLE
(nhiều model + nhiều compute), KHÔNG phải từ cơ chế unlearn/sharding.
"""

import os
import sys
import time
import random
import heapq
import argparse

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from method_1_full_retrain import BPRMF, SimpleDataLoader, evaluate_model, train_model


def load_data(dataset, batch_size=512):
    import os as _os
    data_path = _os.environ.get('RECUNLEARN_DATA_PATH', None)
    if data_path:
        dataset_name = _os.environ.get('RECUNLEARN_DATASET', dataset)
        data_dir = _os.path.join(data_path, dataset_name)
    else:
        base_dir = _os.path.dirname(_os.path.dirname(__file__))
        data_dir = _os.path.join(base_dir, 'data', dataset)
    if not _os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return SimpleDataLoader(data_dir, batch_size)


# ============================================================================
# MEAN-AGGREGATE EVALUATION CHO NHIỀU BPRMF
# ============================================================================

def evaluate_mean_ensemble(models, train_data, test_data, n_users, n_items, device,
                           Ks=[10, 20, 50]):
    """Kết hợp K model bằng MEAN scores (cùng RecEraser dùng mean aggregation)."""
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
                    batch_items = torch.LongTensor(all_items[i:i+256]).to(device)
                    s = m.predict(user_t, batch_items).cpu().numpy()
                    scores.extend(s.tolist())
                scores_sum += np.array(scores)
            scores = scores_sum / len(models)

            train_items = set(train_data.get(user, []))
            for item in train_items:
                scores[item] = -np.inf

            rank_list = heapq.nlargest(max(Ks), range(len(scores)), key=scores.__getitem__)
            item_pos = test_data.get(user, [])
            item_set = set(item_pos)

            for k in Ks:
                hit_list = rank_list[:k]
                hit_num = len(set(hit_list) & item_set)
                rec = hit_num / len(item_pos) if len(item_pos) > 0 else 0
                pre = hit_num / k if k > 0 else 0
                dcg = 0.0
                for i, item in enumerate(hit_list):
                    if item in item_set:
                        dcg += 1.0 / np.log2(i + 2.0)
                idcg = sum(1.0 / np.log2(i + 2.0) for i in range(min(len(item_pos), k)))
                ndcg = dcg / idcg if idcg > 0 else 0
                pre_log[k].append(pre)
                rec_log[k].append(rec)
                ndcg_log[k].append(ndcg)

    return {
        'precision': [np.nanmean(pre_log[k]) for k in Ks],
        'recall': [np.nanmean(rec_log[k]) for k in Ks],
        'ndcg': [np.nanmean(ndcg_log[k]) for k in Ks]
    }


def train_bprmf(train_data, test_data, n_users, n_items, emb_dim, device,
                batch_size=512, lr=0.05, max_epochs=50, seed=0, verbose=False):
    """Train 1 BPRMF. Trả về model đã train (FIXED epochs, không early stopping)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = BPRMF(n_users, n_items, emb_dim).to(device)
    train_model(model, train_data, test_data, n_users, n_items, device,
                batch_size=batch_size, lr=lr, max_epochs=max_epochs,
                early_stopping=False, patience=10, verbose=verbose)
    return model


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--n_shards', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--unlearn_ratio', type=float, default=0.01)
    parser.add_argument('--unlearn_mode', type=str, default='random',
                        choices=['random', 'fewest', 'most'])
    args = parser.parse_args()

    K = args.n_shards
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"FAIR COMPARISON | K={K} | emb_dim={args.emb_dim} | epochs={args.max_epochs} | {device}")

    data = load_data(args.dataset, batch_size=512)
    n_users, n_items = data.n_users, data.n_items
    train_data, test_data = data.train_items, data.test_set

    # Chọn users cần unlearn
    random.seed(42)
    all_users = list(train_data.keys())
    n_unlearn = int(len(all_users) * args.unlearn_ratio)
    if args.unlearn_mode == 'most':
        ui = sorted(all_users, key=lambda u: len(train_data[u]), reverse=True)
        unlearn_users = set(ui[:n_unlearn])
    elif args.unlearn_mode == 'fewest':
        ui = sorted(all_users, key=lambda u: len(train_data[u]))
        unlearn_users = set(ui[:n_unlearn])
    else:
        unlearn_users = set(random.sample(all_users, n_unlearn))
    print(f"Unlearn {len(unlearn_users)} users (mode={args.unlearn_mode}, ratio={args.unlearn_ratio})")

    # AFTER: train trên RETAINED, eval trên RETAINED
    train_data_retained = {u: it for u, it in train_data.items() if u not in unlearn_users}
    test_data_retained = {u: it for u, it in test_data.items() if u not in unlearn_users}
    print(f"Retained: {len(train_data_retained)} users | Retained test: {len(test_data_retained)}")

    def train_shard_models(full_data):
        """K model. Mỗi model train trên 1/K users (shard bằng user-id, như RecEraser random)."""
        shard_users = [all_users[i::K] for i in range(K)]
        models = []
        for sid in range(K):
            sd = {u: it for u in shard_users[sid] if u in full_data}
            m = train_bprmf(sd, test_data_retained, n_users, n_items, args.emb_dim, device,
                            max_epochs=args.max_epochs, seed=sid, verbose=False)
            models.append(m)
            print(f"    shard {sid}: {len(sd)} users trained")
        return models

    # ------------------------------------------------------------------
    # C1: Full Single (1 model trên full retained data)
    # ------------------------------------------------------------------
    print("\n=== C1: Full Single (1 model, full data) ===")
    t0 = time.time()
    c1_models = [train_bprmf(train_data_retained, test_data_retained, n_users, n_items,
                              args.emb_dim, device, max_epochs=args.max_epochs,
                              seed=0, verbose=False)]
    c1_time = time.time() - t0
    c1 = evaluate_mean_ensemble(c1_models, train_data_retained, test_data_retained,
                                n_users, n_items, device)
    print(f"  C1 Recall@10 = {c1['recall'][0]:.4f}   time={c1_time:.1f}s")

    # ------------------------------------------------------------------
    # C2: Full Ensemble (K model, mỗi model TRÊN FULL data, mean-aggregate)
    # ------------------------------------------------------------------
    print(f"\n=== C2: Full Ensemble K={K} (mỗi model train trên FULL data) ===")
    t0 = time.time()
    c2_models = [train_bprmf(train_data_retained, test_data_retained, n_users, n_items,
                             args.emb_dim, device, max_epochs=args.max_epochs,
                             seed=sid, verbose=False) for sid in range(K)]
    c2_time = time.time() - t0
    c2 = evaluate_mean_ensemble(c2_models, train_data_retained, test_data_retained,
                                n_users, n_items, device)
    print(f"  C2 Recall@10 = {c2['recall'][0]:.4f}   time={c2_time:.1f}s")

    # ------------------------------------------------------------------
    # C3: Shard Ensemble (K model, mỗi model trên 1/K data, mean-aggregate)
    #     ~ equivalent với RecEraser random partition + mean
    # ------------------------------------------------------------------
    print(f"\n=== C3: Shard Ensemble K={K} (mỗi model train trên 1/K data) ===")
    t0 = time.time()
    c3_models = train_shard_models(train_data_retained)
    c3_time = time.time() - t0
    c3 = evaluate_mean_ensemble(c3_models, train_data_retained, test_data_retained,
                                n_users, n_items, device)
    print(f"  C3 Recall@10 = {c3['recall'][0]:.4f}   time={c3_time:.1f}s")

    # ------------------------------------------------------------------
    # TÓM TẮT + HIỆU ỨNG
    # ------------------------------------------------------------------
    r1, r2, r3 = c1['recall'][0], c2['recall'][0], c3['recall'][0]
    print("\n" + "=" * 60)
    print("TÓM TẮT (Recall@10 AFTER, RETAINED test)")
    print("=" * 60)
    print(f"  C1 Full Single        : {r1:.4f}   ({c1_time:.0f}s)")
    print(f"  C2 Full Ensemble K={K}: {r2:.4f}   ({c2_time:.0f}s)")
    print(f"  C3 Shard Ensemble K={K}: {r3:.4f}   ({c3_time:.0f}s)")
    print("-" * 60)
    print(f"  ENSEMBLE effect  (C2 - C1) = {r2 - r1:+.4f}")
    print(f"  SHARDING effect  (C3 - C2) = {r3 - r2:+.4f}")
    print("=" * 60)
    if r3 >= r2:
        print("  => Sharding GIÚP thêm (C3 >= C2)")
    else:
        print("  => Sharding KHÔNG giúp thêm (C3 < C2): lợi thế đến từ ENSEMBLE, không phải từ shard/unlearn")


if __name__ == '__main__':
    main()
