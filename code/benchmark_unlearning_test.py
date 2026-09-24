"""
Benchmark 4 Methods - Compare unlearning effectiveness AND model quality

Run: python benchmark_unlearning_test.py
"""

import os, sys, random, numpy as np, torch, heapq, time

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
                sc.extend((model.user_embedding(ut) @ model.item_embedding(bt).t()).cpu().numpy()[0])
            for it in tr.get(u,[]): sc[it] = -1e9
            top = heapq.nlargest(10, range(len(sc)), key=sc.__getitem__)
            h = len(set(top) & set(te[u]))
            rec.append(h/len(te[u]) if te[u] else 0)
    model.train()
    return np.mean(rec) if rec else 0

def test_full_retrain(tr, te, nu, ni, unl, ret, dev, max_epochs=30, retrain_epochs=15):
    """Full Retrain - Oracle baseline"""
    from BPRMF import BPRMF
    from torch.optim import Adagrad

    print("\n" + "="*50)
    print("METHOD: FULL RETRAIN (Oracle Baseline)")
    print("="*50)

    # Train before
    print("\n[1] Training (before unlearn)...")
    model = BPRMF(nu, ni, 64).to(dev)
    opt = Adagrad(model.parameters(), lr=0.05)

    for ep in range(max_epochs):
        samples = []
        for u, it in tr.items():
            for pos in it:
                neg = random.randint(0, ni-1)
                while neg in it: neg = random.randint(0, ni-1)
                samples.append((u, pos, neg))
        random.shuffle(samples)

        for i in range(0, len(samples), 512):
            batch = samples[i:i+512]
            us = torch.LongTensor([s[0] for s in batch]).to(dev)
            ps = torch.LongTensor([s[1] for s in batch]).to(dev)
            ns = torch.LongTensor([s[2] for s in batch]).to(dev)
            opt.zero_grad()
            u_emb = model.user_embedding(us)
            p_emb = model.item_embedding(ps)
            n_emb = model.item_embedding(ns)
            loss = -torch.log(torch.sigmoid((u_emb * p_emb).sum(1) - (u_emb * n_emb).sum(1)) + 1e-10).mean()
            loss.backward()
            opt.step()

    m_before = model

    # Evaluate BEFORE
    print("[2] Evaluating BEFORE...")
    r_bef_u = recall(m_before, tr, te, list(unl), ni, dev)
    r_bef_r = recall(m_before, tr, te, list(ret), ni, dev)
    print(f"    Unlearned R@10: {r_bef_u:.4f}")
    print(f"    Retained R@10:  {r_bef_r:.4f}")

    # Full retrain after (với dữ liệu đã xóa)
    print("\n[3] Full Retrain (after unlearn)...")
    tr_after = {u: it for u, it in tr.items() if u not in unl}

    model = BPRMF(nu, ni, 64).to(dev)
    opt = Adagrad(model.parameters(), lr=0.05)

    for ep in range(max_epochs):
        samples = []
        for u, it in tr_after.items():
            for pos in it:
                neg = random.randint(0, ni-1)
                while neg in it: neg = random.randint(0, ni-1)
                samples.append((u, pos, neg))
        random.shuffle(samples)

        for i in range(0, len(samples), 512):
            batch = samples[i:i+512]
            us = torch.LongTensor([s[0] for s in batch]).to(dev)
            ps = torch.LongTensor([s[1] for s in batch]).to(dev)
            ns = torch.LongTensor([s[2] for s in batch]).to(dev)
            opt.zero_grad()
            u_emb = model.user_embedding(us)
            p_emb = model.item_embedding(ps)
            n_emb = model.item_embedding(ns)
            loss = -torch.log(torch.sigmoid((u_emb * p_emb).sum(1) - (u_emb * n_emb).sum(1)) + 1e-10).mean()
            loss.backward()
            opt.step()

    m_after = model

    # Evaluate AFTER
    print("[4] Evaluating AFTER...")
    r_aft_u = recall(m_after, tr, te, list(unl), ni, dev)
    r_aft_r = recall(m_after, tr, te, list(ret), ni, dev)
    print(f"    Unlearned R@10: {r_aft_u:.4f}")
    print(f"    Retained R@10:  {r_aft_r:.4f}")

    chg_u = ((r_aft_u - r_bef_u) / r_bef_u * 100) if r_bef_u > 0 else 0
    chg_r = ((r_aft_r - r_bef_r) / r_bef_r * 100) if r_bef_r > 0 else 0
    print(f"\n    Unlearned change: {chg_u:+.1f}%")
    print(f"    Retained change: {chg_r:+.1f}%")

    return {
        'name': 'Full Retrain',
        'r10_unlearned_before': r_bef_u,
        'r10_unlearned_after': r_aft_u,
        'r10_retained_before': r_bef_r,
        'r10_retained_after': r_aft_r,
        'unlearn_change': chg_u,
        'retained_change': chg_r
    }

def test_sisa(tr, te, nu, ni, unl, ret, dev, max_epochs=30, retrain_epochs=15):
    """SISA method"""
    from method_2_sisa import SISAMethod, BPRMF

    print("\n" + "="*50)
    print("METHOD: SISA")
    print("="*50)

    # Train before
    print("\n[1] Training (before unlearn)...")
    sisa = SISAMethod(BPRMF, nu, ni, 64, 8, batch_size=512, lr=0.05, max_epochs=max_epochs)
    sisa.train(tr, dev)
    m_before = sisa.models[0]

    # Evaluate BEFORE
    print("[2] Evaluating BEFORE...")
    r_bef_u = recall(m_before, tr, te, list(unl), ni, dev)
    r_bef_r = recall(m_before, tr, te, list(ret), ni, dev)
    print(f"    Unlearned R@10: {r_bef_u:.4f}")
    print(f"    Retained R@10:  {r_bef_r:.4f}")

    # Unlearn
    print("\n[3] Unlearning...")
    tc = {u:i.copy() for u,i in tr.items()}
    sisa.unlearn(unl, tc, dev, retrain_epochs=retrain_epochs)
    m_after = sisa.models[0]

    # Evaluate AFTER
    print("[4] Evaluating AFTER...")
    r_aft_u = recall(m_after, tr, te, list(unl), ni, dev)
    r_aft_r = recall(m_after, tr, te, list(ret), ni, dev)
    print(f"    Unlearned R@10: {r_aft_u:.4f}")
    print(f"    Retained R@10:  {r_aft_r:.4f}")

    chg_u = ((r_aft_u - r_bef_u) / r_bef_u * 100) if r_bef_u > 0 else 0
    chg_r = ((r_aft_r - r_bef_r) / r_bef_r * 100) if r_bef_r > 0 else 0
    print(f"\n    Unlearned change: {chg_u:+.1f}%")
    print(f"    Retained change: {chg_r:+.1f}%")

    return {
        'name': 'SISA',
        'r10_unlearned_before': r_bef_u,
        'r10_unlearned_after': r_aft_u,
        'r10_retained_before': r_bef_r,
        'r10_retained_after': r_aft_r,
        'unlearn_change': chg_u,
        'retained_change': chg_r
    }

def test_ours(tr, te, nu, ni, unl, ret, dev, max_epochs=30, retrain_epochs=15):
    """Ours method"""
    from method_4_ours import OursMethod

    print("\n" + "="*50)
    print("METHOD: OURS (Deletion-Stable)")
    print("="*50)

    # Train before
    print("\n[1] Training (before unlearn)...")
    ours = OursMethod(nu, ni, 64, 8, batch_size=512, lr=0.05, max_epochs=max_epochs)
    ours.train(tr, dev)
    m_before = ours.shard_models.get_model(0)

    # Evaluate BEFORE
    print("[2] Evaluating BEFORE...")
    r_bef_u = recall(m_before, tr, te, list(unl), ni, dev)
    r_bef_r = recall(m_before, tr, te, list(ret), ni, dev)
    print(f"    Unlearned R@10: {r_bef_u:.4f}")
    print(f"    Retained R@10:  {r_bef_r:.4f}")

    # Unlearn
    print("\n[3] Unlearning...")
    tc = {u:i.copy() for u,i in tr.items()}
    ours.unlearn(unl, tc, dev, retrain_epochs=retrain_epochs)
    m_after = ours.shard_models.get_model(0)

    # Evaluate AFTER
    print("[4] Evaluating AFTER...")
    r_aft_u = recall(m_after, tr, te, list(unl), ni, dev)
    r_aft_r = recall(m_after, tr, te, list(ret), ni, dev)
    print(f"    Unlearned R@10: {r_aft_u:.4f}")
    print(f"    Retained R@10:  {r_aft_r:.4f}")

    chg_u = ((r_aft_u - r_bef_u) / r_bef_u * 100) if r_bef_u > 0 else 0
    chg_r = ((r_aft_r - r_bef_r) / r_bef_r * 100) if r_bef_r > 0 else 0
    print(f"\n    Unlearned change: {chg_u:+.1f}%")
    print(f"    Retained change: {chg_r:+.1f}%")

    return {
        'name': 'Ours',
        'r10_unlearned_before': r_bef_u,
        'r10_unlearned_after': r_aft_u,
        'r10_retained_before': r_bef_r,
        'r10_retained_after': r_aft_r,
        'unlearn_change': chg_u,
        'retained_change': chg_r
    }

def main():
    print("="*70)
    print("  BENCHMARK: 4 METHODS UNLEARNING COMPARISON")
    print("="*70)

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

    results = []

    t0 = time.time()

    # Test all methods
    r = test_full_retrain(tr, te, nu, ni, unl, ret, dev)
    results.append(r)

    r = test_sisa(tr, te, nu, ni, unl, ret, dev)
    results.append(r)

    r = test_ours(tr, te, nu, ni, unl, ret, dev)
    results.append(r)

    total_time = time.time() - t0

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n\n" + "="*70)
    print("  SUMMARY TABLE")
    print("="*70)
    print(f"\nUnlearn Ratio: 30%")
    print(f"Total Time: {total_time:.1f}s")
    print()

    print("="*70)
    print(f"{'Method':<15} | {'Unlearn ↓':>12} | {'Retained':>12} | {'R@10 Unl Aft':>12} | {'R@10 Ret Aft':>12}")
    print("-"*70)

    for r in results:
        status = "✓" if r['unlearn_change'] < -50 else "⚠" if r['unlearn_change'] < 0 else "✗"
        print(f"{r['name']:<15} | {r['unlearn_change']:>+11.1f}% | {r['retained_change']:>+11.1f}% | {r['r10_unlearned_after']:>12.4f} | {r['r10_retained_after']:>12.4f}")

    print()
    print("="*70)
    print("DIEN GIAI:")
    print("="*70)
    print("- Unlearn Change: Giam nhieu (>50%) -> DA QUEN tot")
    print("- Retained Change: Gan 0% -> Khong bi anh huong")
    print("- R@10 Unl After: Can thap sau unlearn (model da quen)")
    print("- R@10 Ret After: Can cao (model con tot cho retained users)")
    print()
    print("="*70)
    print("DANH GIA:")
    print("="*70)

    for r in results:
        unlearn_ok = r['unlearn_change'] < -50
        retained_ok = abs(r['retained_change']) < 20

        if unlearn_ok and retained_ok:
            status = "TOT (Quen + On dinh)"
        elif unlearn_ok:
            status = "Quen nhung anh huong retained"
        elif retained_ok:
            status = "Khong anh huong nhung chua quen"
        else:
            status = "CHUA DAT YEU CAU"

        print(f"  {r['name']}: {status}")

    print("="*70)

if __name__ == '__main__':
    main()
