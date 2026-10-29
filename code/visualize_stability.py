"""
VISUALIZE DELETION-STABILITY ACROSS UNLEARN RATIOS
==================================================

Chạy measure_stability (không cần train) trên NẤY unlearn_ratio, rồi vẽ
2 figure để thể hiện rõ đóng góp:

  Figure 1: Naive vs Effective affected shards (bar, theo ratio)
            -> Rec "giấu" shards (naive << effective); Ours có effective=naive.

  Figure 2: Instability% theo ratio (line)
            -> Ours & SISA = 0 (stable); Rec (k-means) tăng vọt khi xóa nhiều.

Chạy:
    python code/visualize_stability.py --dataset ml-1m --n_shards 8 \
        --unlearn_mode most --ratios 0.001,0.005,0.01,0.02,0.05 \
        --out figures

Tạo:
    figures/stability_naive_vs_effective.png
    figures/stability_instability.png
    figures/stability_table.csv
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from measure_stability import (
    load_data, make_unlearn_users, kmeans_ubp, analyze,
)
from method_4_ours import UserSignatureBuilder, DeletionStableAssignment


def run_one_ratio(train_data, train_ret, N, n_items, unlearn, retained_ids,
                  K, signature_dim, user_emb, wmf_have, verbose=True):
    """Trả về dict {method: row} cho 1 ratio."""
    rows = {}
    all_users = list(train_data.keys())

    # Ours (signature)
    sb = UserSignatureBuilder(N, n_items, d=signature_dim, seed=42)
    sigs_full = sb.build_signature(train_data)
    asg = DeletionStableAssignment(n_shards=K, d=signature_dim, seed=42)
    A_f = np.asarray(asg.assign_users(sigs_full), dtype=np.int64)
    sigs_ret = sb.build_signature(train_ret)
    A_r = np.asarray(asg.assign_users(sigs_ret), dtype=np.int64)
    rows['Ours (sig)'] = analyze("Ours (signature)", A_f, A_r, unlearn,
                                  retained_ids, K, True, True, verbose=verbose)

    # SISA (modulo)
    A_f_s = np.array([u % K for u in range(N)], dtype=np.int64)
    A_r_s = np.array([u % K for u in range(N)], dtype=np.int64)
    rows['SISA (mod)'] = analyze("SISA (modulo)", A_f_s, A_r_s, unlearn,
                                  retained_ids, K, False, True, verbose=verbose)

    # Rec UBP (k-means)
    if wmf_have is not None:
        A_f_rec = kmeans_ubp(wmf_have, K, seed=42).astype(np.int64)
        A_r_pos = kmeans_ubp(wmf_have[retained_ids], K, seed=42).astype(np.int64)
        A_r_rec = np.full(N, -1, dtype=np.int64)
        A_r_rec[retained_ids] = A_r_pos
        rows['Rec UBP'] = analyze("Rec UBP (kmeans)", A_f_rec, A_r_rec, unlearn,
                                   retained_ids, K, True, False, verbose=verbose)
    else:
        print("  [skip Rec UBP - no WMF embeddings]")

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-1m')
    p.add_argument('--n_shards', type=int, default=8)
    p.add_argument('--signature_dim', type=int, default=64)
    p.add_argument('--unlearn_mode', default='most', choices=['random', 'fewest', 'most'])
    p.add_argument('--ratios', default='0.001,0.005,0.01,0.02,0.05')
    p.add_argument('--out', default='figures')
    args = p.parse_args()

    K = args.n_shards
    ratios = [float(x) for x in args.ratios.split(',')]
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.dataset} ...")
    data = load_data(args.dataset, batch_size=512)
    N, n_items = data.n_users, data.n_items
    train_data, _ = data.train_items, data.test_set
    all_users = list(train_data.keys())

    # WMF embeddings (nếu có) cho Rec UBP
    base = os.path.dirname(os.path.dirname(__file__))
    emb_path = os.path.join(base, 'data', args.dataset, 'wmf_embeddings.npz')
    wmf = np.load(emb_path)['user_embeddings'][:N] if os.path.exists(emb_path) else None
    print(f"WMF embeddings available: {wmf is not None}")

    ratios_x = []
    series = {m: {'naive': [], 'eff': [], 'inst': []} for m in
              ['Ours (sig)', 'SISA (mod)', 'Rec UBP']}
    table = []

    for ratio in ratios:
        unlearn = make_unlearn_users(train_data, all_users, ratio, args.unlearn_mode)
        train_ret = {u: it for u, it in train_data.items() if u not in unlearn}
        retained_ids = [u for u in range(N) if u not in unlearn]
        print(f"\n===== ratio={ratio}  (unlearn {len(unlearn)} users) =====")
        rows = run_one_ratio(train_data, train_ret, N, n_items, unlearn,
                              retained_ids, K, args.signature_dim, wmf,
                              verbose=False)
        ratios_x.append(ratio)
        for m in series:
            if m in rows:
                r = rows[m]
                series[m]['naive'].append(r['naive_affected'])
                series[m]['eff'].append(r['effective_affected'])
                series[m]['inst'].append(r['instability'])
                table.append({
                    'ratio': ratio, 'method': m,
                    'naive': r['naive_affected'],
                    'effective': r['effective_affected'],
                    'instability': round(r['instability'], 2),
                })
        if wmf is None:
            break  # không có Rec -> chỉ cần chạy 1 lần

    # ------------------------------------------------------------------
    # Lưu CSV
    # ------------------------------------------------------------------
    import csv
    csv_path = os.path.join(args.out, 'stability_table.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['ratio', 'method', 'naive', 'effective', 'instability'])
        w.writeheader()
        w.writerows(table)
    print(f"\nSaved table -> {csv_path}")

    if wmf is None:
        print("No WMF embeddings: skip figures (Rec UBP not available).")
        return

    # ------------------------------------------------------------------
    # FIGURE 1: Naive vs Effective affected shards
    # ------------------------------------------------------------------
    methods = ['Ours (sig)', 'SISA (mod)', 'Rec UBP']
    labels = ['Ours (signature)', 'SISA (modulo)', 'Rec UBP (k-means)']
    colors = ['#2a7de1', '#7a7a7a', '#e0533d']
    x = np.arange(len(ratios_x))
    width = 0.30

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, m in enumerate(methods):
        c = colors[i]
        off = (i - 1) * width
        ax.bar(x + off - width/2, series[m]['naive'], width*0.9,
               color=c, alpha=0.30, label=f'{labels[i]} (naive)')
        ax.bar(x + off, series[m]['eff'], width*0.9,
               color=c, alpha=1.0, label=f'{labels[i]} (effective)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r*100:g}%" for r in ratios_x])
    ax.set_xlabel("Unlearn ratio")
    ax.set_ylabel("Affected shards (of %d)" % K)
    ax.set_title("Naive vs Effective affected shards\n"
                 "(naive = shards chứa user bị xóa; effective = + shards bị re-partition)",
                 fontsize=12)
    ax.set_ylim(0, K + 1)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    p1 = os.path.join(args.out, 'stability_naive_vs_effective.png')
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved figure -> {p1}")

    # ------------------------------------------------------------------
    # FIGURE 2: Instability% theo ratio
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {
        'Ours (sig)': dict(color='#2a7de1', marker='o', ls='-'),
        'SISA (mod)': dict(color='#7a7a7a', marker='s', ls='--'),
        'Rec UBP':    dict(color='#e0533d', marker='^', ls='-'),
    }
    for m in methods:
        s = styles[m]
        ax.plot(ratios_x, series[m]['inst'], label=labels[methods.index(m)], **s)
        for xv, yv in zip(ratios_x, series[m]['inst']):
            ax.annotate(f"{yv:.1f}", (xv, yv), textcoords='offset points',
                        xytext=(0, 6), fontsize=8, ha='center', color=s['color'])
    ax.set_xlabel("Unlearn ratio")
    ax.set_ylabel("Instability (% retained users change shard)")
    ax.set_title("Deletion stability: share of retained users displaced after re-partition",
                 fontsize=12)
    ax.set_xscale('log')
    ax.set_xticks(ratios_x)
    ax.set_xticklabels([f"{r*100:g}%" for r in ratios_x])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    p2 = os.path.join(args.out, 'stability_instability.png')
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Saved figure -> {p2}")


if __name__ == '__main__':
    main()
