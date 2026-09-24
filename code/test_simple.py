"""
Test unlearning effectiveness - Simple version
Chạy: python test_simple.py
"""

import os, sys, random, numpy as np, torch, heapq

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

def load_data():
    """Load ml-1m data"""
    dd = os.path.join(os.path.dirname(PROJ), 'data', 'ml-1m')
    tr, te = {}, {}
    nu, ni = 0, 0
    for fn, d in [(os.path.join(dd,'train.txt'),tr),(os.path.join(dd,'test.txt'),te)]:
        with open(fn) as f:
            for l in f:
                p = l.strip().split()
                u = int(p[0])
                i = [int(x) for x in p[1:]]
                d[u] = i
                nu = max(nu, u+1)
                ni = max(ni, max(i)+1 if i else 0)
    return tr, te, nu, ni

def recall(model, tr, te, users, ni, dev):
    """Compute Recall@10 for a list of users"""
    rec = []
    model.eval()
    with torch.no_grad():
        for u in users:
            if u not in te: continue
            ut = torch.LongTensor([u]).to(dev)
            sc = []
            for i in range(0, ni, 256):
                bt = torch.LongTensor(list(range(i,min(i+256,ni)))).to(dev)
                # Compute scores: user_emb @ item_emb.T
                sc.extend((model.user_embedding(ut) @ model.item_embedding(bt).t()).cpu().numpy()[0])
            # Mask training items
            for it in tr.get(u,[]): sc[it] = -1e9
            # Top-10
            top = heapq.nlargest(10, range(len(sc)), key=sc.__getitem__)
            h = len(set(top) & set(te[u]))
            rec.append(h/len(te[u]) if te[u] else 0)
    model.train()
    return np.mean(rec) if rec else 0

def main():
    print("="*60)
    print("  UNLEARNING EFFECTIVENESS TEST")
    print("="*60)

    # Load data
    tr, te, nu, ni = load_data()
    print(f"\nData: {nu} users, {ni} items")

    # Select unlearn users (30%)
    random.seed(42)
    all_u = list(tr.keys())
    n_unl = int(len(all_u) * 0.3)
    unl = set(random.sample(all_u, n_unl))
    ret = set(all_u) - unl
    print(f"Unlearn: {len(unl)} users (30%)")
    print(f"Retained: {len(ret)} users")

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {dev}")

    # =========================================================================
    # TEST SISA
    # =========================================================================
    print("\n" + "="*60)
    print("METHOD: SISA")
    print("="*60)

    from method_2_sisa import SISAMethod, BPRMF

    # Train before unlearn
    print("\n[1] Training SISA (before unlearn)...")
    sisa = SISAMethod(BPRMF, nu, ni, 64, 8, batch_size=512, lr=0.05, max_epochs=30)
    sisa.train(tr, dev)
    m_before = sisa.models[0]

    # Evaluate BEFORE
    print("[2] Evaluating BEFORE unlearn...")
    r_bef_u = recall(m_before, tr, te, list(unl), ni, dev)
    r_bef_r = recall(m_before, tr, te, list(ret), ni, dev)
    print(f"    Unlearned users R@10: {r_bef_u:.4f}")
    print(f"    Retained users R@10:  {r_bef_r:.4f}")

    # Unlearn
    print("[3] Unlearning...")
    tc = {u:i.copy() for u,i in tr.items()}
    sisa.unlearn(unl, tc, dev, retrain_epochs=15)
    m_after = sisa.models[0]

    # Evaluate AFTER
    print("[4] Evaluating AFTER unlearn...")
    r_aft_u = recall(m_after, tr, te, list(unl), ni, dev)
    r_aft_r = recall(m_after, tr, te, list(ret), ni, dev)
    print(f"    Unlearned users R@10: {r_aft_u:.4f}")
    print(f"    Retained users R@10:  {r_aft_r:.4f}")

    # Results
    chg_u = ((r_aft_u - r_bef_u) / r_bef_u * 100) if r_bef_u > 0 else 0
    chg_r = ((r_aft_r - r_bef_r) / r_bef_r * 100) if r_bef_r > 0 else 0
    print(f"\n    Unlearned change: {chg_u:+.1f}%")
    print(f"    Retained change:  {chg_r:+.1f}%")
    if chg_u < -30:
        print("    => UNLEARNED: DA QUEN!")
    else:
        print("    => UNLEARNED: CON NHOR!")

    sisa_result = {'unlearn_change': chg_u, 'retained_change': chg_r}

    # =========================================================================
    # TEST OURS
    # =========================================================================
    print("\n" + "="*60)
    print("METHOD: OURS")
    print("="*60)

    from method_4_ours import OursMethod

    # Train before unlearn
    print("\n[1] Training Ours (before unlearn)...")
    ours = OursMethod(nu, ni, 64, 8, batch_size=512, lr=0.05, max_epochs=30)
    ours.train(tr, dev)
    m_before = ours.shard_models.get_model(0)

    # Evaluate BEFORE
    print("[2] Evaluating BEFORE unlearn...")
    r_bef_u = recall(m_before, tr, te, list(unl), ni, dev)
    r_bef_r = recall(m_before, tr, te, list(ret), ni, dev)
    print(f"    Unlearned users R@10: {r_bef_u:.4f}")
    print(f"    Retained users R@10:  {r_bef_r:.4f}")

    # Unlearn
    print("[3] Unlearning...")
    tc = {u:i.copy() for u,i in tr.items()}
    ours.unlearn(unl, tc, dev, retrain_epochs=15)
    m_after = ours.shard_models.get_model(0)

    # Evaluate AFTER
    print("[4] Evaluating AFTER unlearn...")
    r_aft_u = recall(m_after, tr, te, list(unl), ni, dev)
    r_aft_r = recall(m_after, tr, te, list(ret), ni, dev)
    print(f"    Unlearned users R@10: {r_aft_u:.4f}")
    print(f"    Retained users R@10:  {r_aft_r:.4f}")

    # Results
    chg_u = ((r_aft_u - r_bef_u) / r_bef_u * 100) if r_bef_u > 0 else 0
    chg_r = ((r_aft_r - r_bef_r) / r_bef_r * 100) if r_bef_r > 0 else 0
    print(f"\n    Unlearned change: {chg_u:+.1f}%")
    print(f"    Retained change:  {chg_r:+.1f}%")
    if chg_u < -30:
        print("    => UNLEARNED: DA QUEN!")
    else:
        print("    => UNLEARNED: CON NHOR!")

    ours_result = {'unlearn_change': chg_u, 'retained_change': chg_r}

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"\n{'Method':<10} {'Unlearn Change':>15} {'Retained Change':>15}")
    print("-" * 45)
    print(f"{'SISA':<10} {sisa_result['unlearn_change']:>+14.1f}% {sisa_result['retained_change']:>+14.1f}%")
    print(f"{'OURS':<10} {ours_result['unlearn_change']:>+14.1f}% {ours_result['retained_change']:>+14.1f}%")

    print("\n" + "="*60)
    print("DIEN GIAI:")
    print("="*60)
    print("- Unlearn Change: Giam nhieu -> DA QUEN, Tang -> CON NHOR")
    print("- Retained Change: ~0 -> ON DINH")
    print("="*60)

if __name__ == '__main__':
    main()
