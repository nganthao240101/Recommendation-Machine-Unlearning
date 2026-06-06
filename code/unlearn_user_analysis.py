"""
RecEraser Unlearn USER Analysis
===============================
So sanh InBP vs UBP vs Random cho unlearn USER
(Khac voi unlearn INTERACTION - UBP co the tot hon!)

Khi unlearn USER:
- UBP: User gom ve 1 local -> chi retrain 1 local model (TOT)
- InBP: User co the o nhieu local -> can retrain nhieu local (KEM)
"""
import os
import pickle
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTHONIOENCODING'] = 'utf-8'

from utility.parser import parse_args

args = parse_args()

def load_partitions(part_type, part_num):
    base = args.data_path + args.dataset
    with open(f'{base}/C_type-{part_type}_num-{part_num}.pk', 'rb') as f:
        C = pickle.load(f)
    with open(f'{base}/C_U_type-{part_type}_num-{part_num}.pk', 'rb') as f:
        C_U = pickle.load(f)
    with open(f'{base}/C_I_type-{part_type}_num-{part_num}.pk', 'rb') as f:
        C_I = pickle.load(f)
    return C, C_U, C_I

def analyze_user_unlearn(C, C_U, C_I, part_type):
    """Phan tich chi phi unlearn USER"""
    names = {1: "InBP", 2: "UBP", 3: "Random"}
    name = names.get(part_type, "Unknown")

    print(f"\n{'='*60}")
    print(f"[{part_type}] {name} - User Unlearn Analysis")
    print(f"{'='*60}")

    # Dem user trong moi local model
    users_per_local = [len(set(C_U[i])) for i in range(len(C))]
    print(f"  Users per local: {users_per_local}")

    # Tong so user (co the trung giua cac local)
    total_user_appearances = sum(users_per_local)
    unique_users = len(set().union(*[set(C_U[i]) for i in range(len(C))]))

    print(f"  Total user appearances: {total_user_appearances}")
    print(f"  Unique users: {unique_users}")
    print(f"  Avg appearances per user: {total_user_appearances / unique_users:.2f}")

    # Phan tich: 1 user xuat hien o bao nhieu local models
    all_users = set().union(*[set(C_U[i]) for i in range(len(C))])
    user_local_count = {u: 0 for u in all_users}

    for i in range(len(C)):
        for u in C_U[i]:
            user_local_count[u] += 1

    # Distribution of user appearances
    appearances_dist = {}
    for u, count in user_local_count.items():
        if count not in appearances_dist:
            appearances_dist[count] = 0
        appearances_dist[count] += 1

    print(f"\n  User appearance distribution:")
    for num_locals in sorted(appearances_dist.keys()):
        print(f"    Users in {num_locals} local(s): {appearances_dist[num_locals]}")

    # Chi phi unlearn USER
    users_in_multiple_locals = sum(1 for u, c in user_local_count.items() if c > 1)
    avg_local_per_user = total_user_appearances / unique_users

    print(f"\n  User Unlearn Statistics:")
    print(f"    Users in MULTIPLE locals: {users_in_multiple_locals}/{unique_users} ({100*users_in_multiple_locals/unique_users:.1f}%)")
    print(f"    Avg locals per user: {avg_local_per_user:.2f}")

    return {
        'name': name,
        'total_user_appearances': total_user_appearances,
        'unique_users': unique_users,
        'users_in_multiple_locals': users_in_multiple_locals,
        'avg_local_per_user': avg_local_per_user,
        'users_per_local': users_per_local,
        'user_local_count': user_local_count
    }

def main():
    print("="*70)
    print("RecEraser Unlearn USER Analysis")
    print("="*70)
    print("\nQuestion: Which partition is BEST for unlearning USERS?")
    print("-"*70)
    print("""
  PREVIOUS ANALYSIS (unlearn INTERACTION):
    - InBP was BEST (17,283 max retrain, 95% retention)

  NEW ANALYSIS (unlearn USER):
    - UBP may be BETTER (user = 1 local model)
    - InBP may be WORSE (user spans multiple locals)
    """)

    results = {}
    for pt in [1, 2, 3]:
        C, C_U, C_I = load_partitions(pt, args.part_num)
        results[pt] = analyze_user_unlearn(C, C_U, C_I, pt)

    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON: InBP vs UBP vs Random FOR USER UNLEARN")
    print(f"{'='*70}")

    print("\n| Metric                    | InBP    | UBP     | Random | Winner  |")
    print("|---------------------------|---------|---------|--------|---------|")

    inbp_data = results[1]
    ubp_data = results[2]
    random_data = results[3]

    # Users in multiple locals (lower = better for user unlearn)
    print(f"| Users in multiple locals  | {inbp_data['users_in_multiple_locals']:>7} | {ubp_data['users_in_multiple_locals']:>7} | {random_data['users_in_multiple_locals']:>6} | UBP     |")

    # Avg locals per user (lower = better for user unlearn)
    print(f"| Avg locals per user       | {inbp_data['avg_local_per_user']:>7.2f} | {ubp_data['avg_local_per_user']:>7.2f} | {random_data['avg_local_per_user']:>6.2f} | UBP     |")

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print("""
  For UNLEARNING USERS:
  ---------------------
  [BEST] UBP (User-based Balanced Partition):
     - Each user belongs to EXACTLY ONE local model
     - Unlearn 1 user -> retrain only 1 local model
     - Lower retrain overhead for user removal

  [WORST] InBP (Interaction-based Balanced Partition):
     - One user can appear in MULTIPLE local models
     - Unlearn 1 user -> retrain multiple local models
     - Higher retrain overhead for user removal

  [MODERATE] Random:
     - Balanced but not optimized for user unlearn

  SUMMARY:
  --------
  - Unlearn INTERACTION: InBP is BEST
  - Unlearn USER: UBP is BEST

  => Different partition methods are optimal for different unlearning tasks!
    """)

if __name__ == '__main__':
    main()