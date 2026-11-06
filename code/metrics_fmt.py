"""Định dạng metrics chung cho các method (Recall/NDCG @10/20/50).

Dùng chung để các log của Full / SISA / Ours / Rec in CÙNG định dạng:
    R@10=0.0756 | R@20=0.1120 | R@50=0.1800 | NDCG@10=0.1981 | ...
"""

KS = (10, 20, 50)


def fmt_metrics(results, ks=KS):
    """Format dict {'recall': [...], 'ndcg': [...]} thành 1 chuỗi.

    Tương thích ngược: nếu list chỉ có 1 phần tử (chỉ @10), các ô còn lại
    in '(na)' thay vì giá trị sai - để không bao giờ bịa số.
    """
    rec = results.get('recall', [])
    ndcg = results.get('ndcg', [])
    parts = []
    for i, k in enumerate(ks):
        parts.append(f"R@{k}={rec[i]:.4f}" if i < len(rec) else f"R@{k}=(na)")
    for i, k in enumerate(ks):
        parts.append(f"NDCG@{k}={ndcg[i]:.4f}" if i < len(ndcg) else f"NDCG@{k}=(na)")
    return " | ".join(parts)
