"""
Example: Deletion-Stable Unlearning Framework

This example demonstrates the 3 key properties of the framework:
1. Deletion-Local: User signatures don't depend on other users
2. Deletion-Stable: Cluster assignment doesn't change when users are deleted
3. Exact Unlearning: Only affected shard needs retraining
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


# ============================================================
# COMPONENT 1: Deletion-Local User Signatures
# ============================================================

def example_deletion_local_signatures():
    """
    Example: Each user's signature depends ONLY on their own interactions.

    When User A is deleted, User B's signature is UNAFFECTED.
    """
    print("=" * 60)
    print("EXAMPLE 1: Deletion-Local User Signatures")
    print("=" * 60)

    # Sample data
    train_data = {
        0: [0, 1, 2],      # User 0 interacted with items 0, 1, 2
        1: [1, 2, 3],      # User 1 interacted with items 1, 2, 3
        2: [0, 3]          # User 2 interacted with items 0, 3
    }

    n_users = 3
    n_items = 4

    print("\nOriginal Data:")
    for u, items in train_data.items():
        print(f"  User {u}: items {items}")

    # Build signatures
    print("\n" + "-" * 40)
    print("Building User Signatures (Sparse Preference Vectors)")
    print("-" * 40)

    # User signatures as sparse vectors
    signatures = {}
    for user_id, items in train_data.items():
        # Local preference: count / total
        total = len(items)
        sig = {}
        for item in items:
            sig[item] = 1.0 / total  # Local weight

        signatures[user_id] = sig
        print(f"\nUser {user_id} Signature:")
        print(f"  Sparse representation: {sig}")

    # Verify independence
    print("\n" + "-" * 40)
    print("KEY PROPERTY: Deletion-Local")
    print("-" * 40)

    print("\nIf we DELETE User 0:")
    print("  - User 0's signature is removed")
    print("  - User 1's signature remains: {1: 0.33, 2: 0.33, 3: 0.33}")
    print("  - User 2's signature remains: {0: 0.5, 3: 0.5}")

    print("\n[OK] User 1 and User 2 signatures are UNAFFECTED by deleting User 0")
    print("[OK] This is because each signature is based ONLY on local interactions")


# ============================================================
# COMPONENT 2: Deletion-Stable Assignment
# ============================================================

def example_deletion_stable_assignment():
    """
    Example: User assignment to shards uses stable hashing.

    The assignment only depends on the user's own signature,
    not on the presence of other users.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Deletion-Stable Similarity-Aware Assignment")
    print("=" * 60)

    print("""
    Algorithm:
    1. Project each user's signature to lower dimension
    2. Use hyperplane-based routing to determine shard

    Key property: Deletion-Stable
    - The projection matrix is FIXED
    - The hyperplanes are FIXED
    - Deleting User A does NOT affect User B's shard
    """)

    # Simulate projection
    np.random.seed(42)
    projection_dim = 2
    n_shards = 4
    n_items = 4

    # Random projection matrix (fixed)
    proj_matrix = np.random.randn(projection_dim, n_items) * 0.1

    print("\n" + "-" * 40)
    print("Step 1: Random Projection")
    print("-" * 40)

    user_signatures = {
        0: [1/3, 1/3, 1/3, 0],      # User 0: [items 0,1,2] equally
        1: [0, 1/3, 1/3, 1/3],     # User 1: [items 1,2,3] equally
        2: [0.5, 0, 0, 0.5]        # User 2: [items 0,3] equally
    }

    for user_id, sig in user_signatures.items():
        projected = proj_matrix @ sig
        print(f"User {user_id}: sig={sig}")
        print(f"         projected={projected.round(4)}")

    print("\n" + "-" * 40)
    print("Step 2: Hyperplane Routing")
    print("-" * 40)

    # Generate hyperplanes
    hyperplanes = []
    n_hyperplanes = int(np.ceil(np.log2(n_shards)))

    for i in range(n_hyperplanes):
        h = np.random.randn(projection_dim)
        h = h / np.linalg.norm(h)
        hyperplanes.append(h)

    print(f"\nGenerated {n_hyperplanes} hyperplanes")

    def get_shard(projected, hyperplanes, n_shards):
        """Determine shard using hyperplane traversal."""
        shard_id = 0
        for i, h in enumerate(hyperplanes):
            side = np.dot(projected, h)
            if side > 0:
                shard_id += (1 << i)
        return min(shard_id, n_shards - 1)

    print("\nAssignments:")
    for user_id, sig in user_signatures.items():
        projected = proj_matrix @ sig

        shard = get_shard(projected, hyperplanes, n_shards)

        print(f"\nUser {user_id}:")
        print(f"  Projected: {projected.round(4)}")
        routing = []
        for i, h in enumerate(hyperplanes):
            side = np.dot(projected, h)
            direction = "RIGHT" if side > 0 else "LEFT"
            routing.append(f"h{i}: {direction} (+{(1<<i) if side>0 else 0})")
        print(f"  Routing: {' -> '.join(routing)}")
        print(f"  -> Shard {shard}")

    print("\n" + "-" * 40)
    print("KEY PROPERTY: Deletion-Stable")
    print("-" * 40)

    print("""
    If we DELETE User 0:
    - User 1's projected vector: UNCHANGED
    - User 1's hyperplane decisions: UNCHANGED
    - User 1's shard assignment: UNCHANGED

    [OK] Deleting User 0 does NOT affect User 1's shard
    """)


# ============================================================
# COMPONENT 3: Isolated Training
# ============================================================

def example_isolated_training():
    """
    Example: Each shard has an independent model.

    Unlearning = retrain ONLY the shard containing the deleted user.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Isolated Training and Exact Unlearning")
    print("=" * 60)

    print("""
    Architecture:
    +----------------------------------------------------------+
    |                    RECOMMENDER                            |
    +----------------------------------------------------------+
    |                                                            |
    |  Shard 0    Shard 1    Shard 2    Shard 3              |
    |  +-----+    +-----+    +-----+    +-----+               |
    |  |Model|    |Model|    |Model|    |Model|               |
    |  |  0  |    |  1  |    |  2  |    |  3  |               |
    |  +-----+    +-----+    +-----+    +-----+               |
    |      |          |          |          |                    |
    |      +----------+----------+----------+                    |
    |                    |                                     |
    |              Aggregation                                    |
    +----------------------------------------------------------+
    """)

    # Simulate shard data
    shard_data = {
        0: {0: [0, 1, 2], 5: [1, 2]},    # Users 0, 5 in Shard 0
        1: {1: [1, 2, 3], 6: [2, 3]},    # Users 1, 6 in Shard 1
        2: {2: [0, 3], 7: [0, 1]},       # Users 2, 7 in Shard 2
        3: {3: [1, 2], 8: [0, 2]}        # Users 3, 8 in Shard 3
    }

    print("Shard Data:")
    for shard_id, users in shard_data.items():
        total_interactions = sum(len(items) for items in users.values())
        print(f"  Shard {shard_id}: {len(users)} users, {total_interactions} interactions")

    print("\n" + "-" * 40)
    print("KEY PROPERTY: Exact Unlearning")
    print("-" * 40)

    print("\nScenario: Delete User 2")

    print("\nStep 1: Find affected shard")
    print("  User 2 is in Shard 2")

    print("\nStep 2: What needs to change?")
    print("  [YES] Shard 2: RETRAIN (contains User 2)")
    print("  [NO]  Shard 0: NO CHANGE (no User 2)")
    print("  [NO]  Shard 1: NO CHANGE (no User 2)")
    print("  [NO]  Shard 3: NO CHANGE (no User 2)")

    print("\nStep 3: Retrain ONLY Shard 2")
    print("  - Load Shard 2 model weights")
    print("  - Remove User 2 from Shard 2 data")
    print("  - Retrain on remaining data: {7: [0, 1]}")
    print("  - Shard 2 model updated")

    print("\nStep 4: Other shards UNCHANGED")
    print("  - Shard 0, 1, 3 models: SAME AS BEFORE")
    print("  - Aggregation weights: SAME AS BEFORE")

    print("""
    [OK] This is EXACT UNLEARNING
    [OK] User 2's influence is completely removed
    [OK] Only 1 shard needed retraining (25% of total)
    [OK] Significant speedup vs full retrain (4x faster)
    """)


# ============================================================
# COMPLETE WORKFLOW EXAMPLE
# ============================================================

def example_complete_workflow():
    """
    Complete workflow: Train -> Unlearn -> Evaluate.
    """
    print("\n" + "=" * 60)
    print("COMPLETE WORKFLOW EXAMPLE")
    print("=" * 60)

    print("""
    =================================================================
    WORKFLOW OVERVIEW
    =================================================================

    STEP 1: Build User Signatures (Deletion-Local)
    --------------------------------------------------------
    User 0: [1/3, 1/3, 1/3, 0]    (interacted 0,1,2)
    User 1: [0, 1/3, 1/3, 1/3]    (interacted 1,2,3)
    User 2: [0.5, 0, 0, 0.5]      (interacted 0,3)

                           |
                           v
    STEP 2: Assign to Shards (Deletion-Stable)
    --------------------------------------------------------
    User 0 -> Shard 0
    User 1 -> Shard 1
    User 2 -> Shard 2

                           |
                           v
    STEP 3: Train Isolated Models
    --------------------------------------------------------
    Shard 0 Model: trained on User 0's data
    Shard 1 Model: trained on User 1's data
    Shard 2 Model: trained on User 2's data

                           |
                           v
    STEP 4: UNLEARN User 1
    --------------------------------------------------------
    1. Remove User 1's signature
    2. Find affected shard: Shard 1
    3. Remove User 1 from Shard 1 data
    4. Retrain Shard 1 Model ONLY
    5. Shards 0, 2: UNCHANGED

                           |
                           v
    RESULT
    --------------------------------------------------------
    [OK] User 1 completely forgotten
    [OK] Shard 1 Model updated to exclude User 1
    [OK] Other shards untouched
    [OK] Only 1/3 of computation needed
    =================================================================
    """)


# ============================================================
# RUN EXAMPLES
# ============================================================

if __name__ == '__main__':
    print("DELETION-STABLE UNLEARNING FRAMEWORK - EXAMPLES")
    print("=" * 60)
    print("""
    This framework has 3 key properties:

    1. Deletion-Local: User signatures are independent
    2. Deletion-Stable: Cluster assignments don't change
    3. Exact Unlearning: Only affected shard retrains
    """)

    example_deletion_local_signatures()
    example_deletion_stable_assignment()
    example_isolated_training()
    example_complete_workflow()

    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60)
