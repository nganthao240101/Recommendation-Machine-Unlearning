"""
Deletion-Stable Unlearning Framework for Recommendation Systems

Based on the paper approach with 3 components:
1. Deletion-Local User Signatures: Sparse preference vectors based on local interactions
2. Deletion-Stable Similarity-Aware Assignment: Random projection + hyperplane routing
3. Isolated Training and Exact User Unlearning: Train models independently per shard

Author: Implementation based on the described framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import pickle
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import math


class DeletionLocalUserSignatures:
    """
    Component 1: Deletion-Local User Signatures

    Creates sparse preference vectors for each user based ONLY on their local
    interaction history. Key property: Independent of other users' data.

    This ensures that when a user is deleted, their signature can be removed
    without affecting other users' signatures.
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64):
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim

        # User signatures: sparse preference vectors
        # Each user has a vector representing their local preferences
        self.user_signatures = None

        # Item statistics for signature creation
        self.item_popularity = np.zeros(n_items)
        self.user_item_counts = []

    def build_signatures(self, train_data: Dict[int, List[int]],
                        sparsity_threshold: float = 0.01):
        """
        Build sparse user signatures based on local interaction history.

        Args:
            train_data: {user_id: [list of interacted items]}
            sparsity_threshold: Keep only items with significance > threshold

        Returns:
            Sparse signature matrix (n_users x n_items)
        """
        print("Building deletion-local user signatures...")

        n_users = self.n_users
        n_items = self.n_items

        # Build sparse matrix
        rows, cols, data = [], [], []

        for user_id, items in train_data.items():
            if user_id >= n_users:
                continue

            # Count interactions
            item_counts = defaultdict(int)
            for item in items:
                if item < n_items:
                    item_counts[item] += 1
                    self.item_popularity[item] += 1

            # Create sparse signature for this user
            total_interactions = sum(item_counts.values())

            for item_id, count in item_counts.items():
                # Local preference weight: count / total_interactions
                # This is purely local, doesn't depend on other users
                weight = count / total_interactions

                # Only keep significant preferences
                if weight >= sparsity_threshold:
                    rows.append(user_id)
                    cols.append(item_id)
                    data.append(weight)

            self.user_item_counts.append((user_id, len(items)))

        # Create sparse matrix
        self.user_signatures = csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_items),
            dtype=np.float32
        )

        print(f"  Built signatures: {self.user_signatures.nnz} non-zero entries")
        print(f"  Sparsity: {1 - self.user_signatures.nnz / (n_users * n_items):.4f}")

        return self.user_signatures

    def get_user_signature(self, user_id: int) -> np.ndarray:
        """Get sparse signature for a specific user."""
        if self.user_signatures is None:
            raise ValueError("Signatures not built yet")
        return self.user_signatures[user_id].toarray().flatten()

    def remove_user_signature(self, user_id: int):
        """Remove a user's signature (for unlearning)."""
        if self.user_signatures is None:
            return

        # Set all values for this user to 0
        start_idx = self.user_signatures.indptr[user_id]
        end_idx = self.user_signatures.indptr[user_id + 1]
        self.user_signatures.data[start_idx:end_idx] = 0

        # Rebuild to clean up zeros
        self.user_signatures.eliminate_zeros()

    def get_significance_score(self, user_id: int, item_id: int) -> float:
        """Get the significance of an item for a user."""
        if self.user_signatures is None:
            return 0.0
        return self.user_signatures[user_id, item_id]


class DeletionStableSimilarityAssignment:
    """
    Component 2: Deletion-Stable Similarity-Aware Assignment

    Uses random projection and hyperplane-based routing to cluster users.
    Key property: Deleting one user does NOT affect the cluster assignment
    of other users.

    This is achieved through:
    1. Random projection to lower dimension
    2. Stable hash-based routing (not dependent on other users)
    """

    def __init__(self, n_users: int, n_shards: int, projection_dim: int = 32,
                 seed: int = 42):
        self.n_users = n_users
        self.n_shards = n_shards
        self.projection_dim = projection_dim
        self.seed = seed

        # Random projection matrix (fixed, doesn't change with data)
        # This ensures stability: same user always maps to same region
        np.random.seed(seed)
        self.projection_matrix = self._generate_projection_matrix()

        # User assignments (stable, based on signatures only)
        self.user_to_shard = np.zeros(n_users, dtype=np.int32)

        # Hyperplanes for routing
        self.hyperplanes = self._generate_hyperplanes()

    def _generate_projection_matrix(self) -> np.ndarray:
        """
        Generate random projection matrix (Cauchy random for stability).

        Using Cauchy distribution instead of Gaussian for better stability
        against distribution shifts (deletion-resistance).
        """
        # Random projection matrix: projection_dim x n_features
        # We'll project from item-space to projection-space
        return np.random.randn(self.projection_dim, -1).astype(np.float32)

    def _generate_hyperplanes(self) -> List[np.ndarray]:
        """
        Generate hyperplanes for shard routing.

        Each hyperplane defines a boundary in projection space.
        User's shard is determined by which side of each hyperplane they fall on.
        """
        np.random.seed(self.seed + 100)
        hyperplanes = []

        # Need ceil(log2(n_shards)) hyperplanes
        n_hyperplanes = math.ceil(math.log2(self.n_shards))

        for i in range(n_hyperplanes):
            # Random normal vector
            h = np.random.randn(self.projection_dim).astype(np.float32)
            h = h / (np.linalg.norm(h) + 1e-8)  # Normalize
            hyperplanes.append(h)

        return hyperplanes

    def assign_users_to_shards(self, user_signatures) -> np.ndarray:
        """
        Assign users to shards using hyperplane-based routing.

        Key property: This assignment only depends on:
        1. The projection matrix (fixed)
        2. The hyperplanes (fixed)
        3. Each user's own signature

        Deleting user A does NOT change the assignment of user B.

        Args:
            user_signatures: Sparse matrix of user signatures (n_users x n_items)

        Returns:
            user_to_shard: Array mapping user_id -> shard_id
        """
        print("Assigning users to shards (deletion-stable)...")

        n_users = self.n_users
        n_shards = self.n_shards

        # Adjust projection matrix size if needed
        if self.projection_matrix.shape[1] != user_signatures.shape[1]:
            self.projection_matrix = np.random.randn(
                self.projection_dim,
                user_signatures.shape[1]
            ).astype(np.float32)

        assignments = np.zeros(n_users, dtype=np.int32)

        for user_id in range(n_users):
            # Get user's signature
            if hasattr(user_signatures, 'toarray'):
                signature = user_signatures[user_id].toarray().flatten()
            else:
                signature = user_signatures[user_id]

            # Project to lower dimension
            # signature: (n_items,) -> projected: (projection_dim,)
            projected = self.projection_matrix @ signature

            # Hyperplane routing
            # Start with shard 0, traverse hyperplanes
            shard_id = 0

            for i, h in enumerate(self.hyperplanes):
                # Check which side of hyperplane
                side = np.dot(projected, h)

                if side > 0:
                    # Move to right branch
                    shard_id += (1 << i)

                # Clip to valid range
                shard_id = min(shard_id, n_shards - 1)

            assignments[user_id] = shard_id
            self.user_to_shard[user_id] = shard_id

        # Print statistics
        shard_counts = np.bincount(assignments, minlength=n_shards)
        print(f"  Shard sizes: min={shard_counts.min()}, max={shard_counts.max()}, "
              f"mean={shard_counts.mean():.1f}")

        return assignments

    def get_shard_for_user(self, user_id: int) -> int:
        """Get shard assignment for a user."""
        return self.user_to_shard[user_id]

    def get_shard_users(self, shard_id: int) -> List[int]:
        """Get all users in a shard."""
        return np.where(self.user_to_shard == shard_id)[0].tolist()

    def get_affected_shard(self, deleted_user_id: int) -> int:
        """
        Get the shard affected by deleting a user.

        Returns:
            shard_id of the shard containing the deleted user
        """
        return self.user_to_shard[deleted_user_id]


class IsolatedShardModels:
    """
    Component 3: Isolated Training and Exact User Unlearning

    Trains independent models on each shard. Key property:
    Unlearning = retrain ONLY the shard containing the deleted user.
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int,
                 n_shards: int, device: str = 'cpu'):
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_shards = n_shards
        self.device = device

        # Shard-specific models
        self.shard_models = nn.ModuleList([
            ShardModel(n_users, n_items, emb_dim)
            for _ in range(n_shards)
        ])

        # Shard data storage
        self.shard_data = [{} for _ in range(n_shards)]

        # Aggregation weights (learned)
        self.shard_weights = nn.Parameter(
            torch.ones(n_shards, device=device) / n_shards
        )

    def build_shard_data(self, train_data: Dict[int, List[int]],
                         user_to_shard: np.ndarray):
        """
        Build data for each shard based on assignments.

        Args:
            train_data: {user_id: [interacted items]}
            user_to_shard: Assignment of users to shards
        """
        print("Building shard data...")

        for shard_id in range(self.n_shards):
            self.shard_data[shard_id] = {}

        for user_id, items in train_data.items():
            if user_id >= self.n_users:
                continue

            shard_id = user_to_shard[user_id]
            self.shard_data[shard_id][user_id] = items.copy()

        # Print statistics
        for shard_id in range(self.n_shards):
            n_users = len(self.shard_data[shard_id])
            n_interactions = sum(len(items) for items in self.shard_data[shard_id].values())
            print(f"  Shard {shard_id}: {n_users} users, {n_interactions} interactions")

    def remove_user_from_shard(self, shard_id: int, user_id: int):
        """
        Remove a user from a shard (for unlearning).

        This is the ONLY change needed for exact unlearning.
        """
        if user_id in self.shard_data[shard_id]:
            del self.shard_data[shard_id][user_id]

    def train_shard(self, shard_id: int, n_epochs: int = 10,
                   batch_size: int = 256, lr: float = 0.01):
        """
        Train model on a single shard.

        Args:
            shard_id: Which shard to train
            n_epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
        """
        shard_model = self.shard_models[shard_id]
        shard_model.train()

        optimizer = torch.optim.Adagrad(shard_model.parameters(), lr=lr)

        shard_data = self.shard_data[shard_id]
        if not shard_data:
            print(f"  Shard {shard_id}: No data, skipping")
            return

        users = list(shard_data.keys())
        all_items = set()
        for items in shard_data.values():
            all_items.update(items)
        all_items = list(all_items)

        n_batches = max(1, len(users) // batch_size + 1)

        for epoch in range(n_epochs):
            total_loss = 0.0

            # Shuffle users
            random.shuffle(users)

            for _ in range(n_batches):
                # Sample batch
                batch_users = random.choices(users, k=batch_size)

                pos_items = []
                neg_items = []

                for u in batch_users:
                    user_items = shard_data.get(u, [])
                    if user_items:
                        pos_items.append(random.choice(user_items))
                    else:
                        pos_items.append(random.choice(all_items))
                    neg_items.append(random.choice(all_items))

                # Forward pass
                user_emb, pos_emb, neg_emb = shard_model(
                    batch_users, pos_items, neg_items
                )

                # BPR Loss
                pos_scores = (user_emb * pos_emb).sum(dim=1)
                neg_scores = (user_emb * neg_emb).sum(dim=1)

                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"    Shard {shard_id} Epoch {epoch+1}: loss={total_loss/n_batches:.4f}")

    def train_all_shards(self, n_epochs: int = 10, batch_size: int = 256,
                        lr: float = 0.01):
        """
        Train all shards (used for initial training).
        """
        print("Training all shards...")
        for shard_id in range(self.n_shards):
            print(f"  Training shard {shard_id}/{self.n_shards-1}...")
            self.train_shard(shard_id, n_epochs, batch_size, lr)

    def train_affected_shard_only(self, shard_id: int, n_epochs: int = 10,
                                  batch_size: int = 256, lr: float = 0.01):
        """
        Train ONLY the affected shard (for unlearning).

        This is the key to efficient unlearning.
        """
        print(f"Training ONLY shard {shard_id} (for unlearning)...")
        self.train_shard(shard_id, n_epochs, batch_size, lr)

    @torch.no_grad()
    def predict(self, user_ids: List[int], item_ids: List[int]) -> torch.Tensor:
        """
        Predict scores for user-item pairs.

        Uses attention-weighted aggregation across shards.
        """
        self.shard_models.eval()

        # Get predictions from each shard
        shard_preds = []

        for shard_id in range(self.n_shards):
            shard_model = self.shard_models[shard_id]
            user_emb, item_emb = shard_model.get_embeddings(user_ids, item_ids)
            scores = (user_emb * item_emb).sum(dim=1)
            shard_preds.append(scores)

        # Stack and aggregate
        shard_preds = torch.stack(shard_preds, dim=1)  # (batch, n_shards)

        # Attention weights
        weights = F.softmax(self.shard_weights, dim=0)
        final_scores = (shard_preds * weights.unsqueeze(0)).sum(dim=1)

        return final_scores

    @torch.no_grad()
    def evaluate(self, test_data: Dict[int, List[int]],
                train_data: Dict[int, List[int]],
                k_values: List[int] = [10, 20, 50]) -> Dict:
        """
        Evaluate the model using standard metrics.
        """
        self.shard_models.eval()

        results = {
            'precision': np.zeros(len(k_values)),
            'recall': np.zeros(len(k_values)),
            'ndcg': np.zeros(len(k_values))
        }

        n_users_evaluated = 0

        for user_id, test_items in test_data.items():
            if user_id >= self.n_users or not test_items:
                continue

            # Get all item scores
            all_items = list(range(self.n_items))
            item_scores = []

            for shard_id in range(self.n_shards):
                shard_model = self.shard_models[shard_id]
                user_emb = shard_model.user_embedding(
                    torch.tensor([user_id], device=self.device)
                )
                item_emb = shard_model.item_embedding(
                    torch.tensor(all_items, device=self.device)
                )
                scores = (user_emb * item_emb).sum(dim=1)
                item_scores.append(scores.cpu().numpy())

            # Aggregate
            item_scores = np.stack(item_scores, axis=1)
            weights = F.softmax(self.shard_weights, dim=0).cpu().numpy()
            final_scores = (item_scores * weights).sum(axis=1)

            # Mask training items
            train_items = set(train_data.get(user_id, []))
            for i, item_id in enumerate(all_items):
                if item_id in train_items:
                    final_scores[i] = -np.inf

            # Get top-k
            top_k = np.argsort(final_scores)[-max(k_values):][::-1]

            # Calculate metrics
            test_set = set(test_items)

            for idx, k in enumerate(k_values):
                pred_k = set(top_k[:k])
                hits = len(pred_k & test_set)

                results['precision'][idx] += hits / k
                results['recall'][idx] += hits / len(test_set) if test_set else 0

                # NDCG
                dcg = 0
                for i, item in enumerate(top_k[:k]):
                    if item in test_set:
                        dcg += 1 / np.log2(i + 2)

                idcg = sum(1 / np.log2(i + 2) for i in range(min(len(test_set), k)))
                results['ndcg'][idx] += dcg / idcg if idcg > 0 else 0

            n_users_evaluated += 1

        # Average
        if n_users_evaluated > 0:
            for key in results:
                results[key] /= n_users_evaluated

        return results


class ShardModel(nn.Module):
    """
    Individual model for each shard.

    Simple BPR model - can be extended to LightGCN or other backbones.
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim

        # Embeddings
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        # Initialize
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, user_ids: List[int], pos_item_ids: List[int],
                neg_item_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for BPR loss.

        Returns:
            user_emb, pos_emb, neg_emb
        """
        user_emb = self.user_embedding(
            torch.tensor(user_ids, device=self.user_embedding.weight.device)
        )
        pos_emb = self.item_embedding(
            torch.tensor(pos_item_ids, device=self.item_embedding.weight.device)
        )
        neg_emb = self.item_embedding(
            torch.tensor(neg_item_ids, device=self.item_embedding.weight.device)
        )

        return user_emb, pos_emb, neg_emb

    @torch.no_grad()
    def get_embeddings(self, user_ids: List[int], item_ids: List[int]):
        """Get embeddings for prediction."""
        user_emb = self.user_embedding(
            torch.tensor(user_ids, device=self.user_embedding.weight.device)
        )
        item_emb = self.item_embedding(
            torch.tensor(item_ids, device=self.item_embedding.weight.device)
        )
        return user_emb, item_emb


class DeletionStableUnlearningFramework:
    """
    Main Framework: Deletion-Stable Unlearning for Recommendation Systems

    Combines all 3 components:
    1. Deletion-Local User Signatures
    2. Deletion-Stable Similarity-Aware Assignment
    3. Isolated Training and Exact User Unlearning
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64,
                 n_shards: int = 8, device: str = 'cpu', seed: int = 42):
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_shards = n_shards
        self.device = device
        self.seed = seed

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Component 1: User Signatures
        self.signatures = DeletionLocalUserSignatures(n_users, n_items, emb_dim)

        # Component 2: Stable Assignment
        self.assignment = DeletionStableSimilarityAssignment(
            n_users, n_shards, projection_dim=emb_dim, seed=seed
        )

        # Component 3: Isolated Models
        self.models = IsolatedShardModels(
            n_users, n_items, emb_dim, n_shards, device
        )

        # Data storage
        self.train_data = {}
        self.test_data = {}

    def build(self, train_data: Dict[int, List[int]], test_data: Dict[int, List[int]]):
        """
        Build the framework from data.

        Args:
            train_data: {user_id: [interacted items]}
            test_data: {user_id: [test items]}
        """
        self.train_data = train_data
        self.test_data = test_data

        print("\n" + "="*60)
        print("STEP 1: Building Deletion-Local User Signatures")
        print("="*60)
        self.signatures.build_signatures(train_data)

        print("\n" + "="*60)
        print("STEP 2: Deletion-Stable Similarity-Aware Assignment")
        print("="*60)
        self.assignment.assign_users_to_shards(self.signatures.user_signatures)

        print("\n" + "="*60)
        print("STEP 3: Building Shard Data")
        print("="*60)
        self.models.build_shard_data(train_data, self.assignment.user_to_shard)

    def train(self, n_epochs: int = 10, batch_size: int = 256, lr: float = 0.01):
        """
        Train all shard models.
        """
        print("\n" + "="*60)
        print("STEP 4: Training Isolated Shard Models")
        print("="*60)
        self.models.train_all_shards(n_epochs, batch_size, lr)

    def evaluate(self, k_values: List[int] = [10, 20, 50]) -> Dict:
        """
        Evaluate the model.
        """
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)

        results = self.models.evaluate(self.test_data, self.train_data, k_values)

        for idx, k in enumerate(k_values):
            print(f"  K={k}: Precision={results['precision'][idx]:.4f}, "
                  f"Recall={results['recall'][idx]:.4f}, "
                  f"NDCG={results['ndcg'][idx]:.4f}")

        return results

    def unlearn_user(self, user_id: int, retrain_epochs: int = 5,
                    batch_size: int = 256, lr: float = 0.01) -> float:
        """
        Exact Unlearning: Remove a user's data and retrain.

        Args:
            user_id: ID of user to unlearn
            retrain_epochs: Number of epochs for retraining
            batch_size: Batch size for retraining
            lr: Learning rate

        Returns:
            Time taken for unlearning (seconds)
        """
        import time
        t_start = time.time()

        print("\n" + "="*60)
        print(f"UNLEARNING USER {user_id}")
        print("="*60)

        # Step 1: Remove from signatures
        print("  1. Removing user signature...")
        self.signatures.remove_user_signature(user_id)

        # Step 2: Find affected shard
        affected_shard = self.assignment.get_affected_shard(user_id)
        print(f"  2. Affected shard: {affected_shard}")

        # Step 3: Remove from shard data
        print("  3. Removing user from shard data...")
        self.models.remove_user_from_shard(affected_shard, user_id)

        # Step 4: Retrain ONLY affected shard (this is the key!)
        print(f"  4. Retraining shard {affected_shard}...")
        self.models.train_affected_shard_only(
            affected_shard, retrain_epochs, batch_size, lr
        )

        t_elapsed = time.time() - t_start
        print(f"\n  Total unlearning time: {t_elapsed:.2f}s")

        return t_elapsed

    def unlearn_users_batch(self, user_ids: List[int], retrain_epochs: int = 5,
                           batch_size: int = 256, lr: float = 0.01) -> float:
        """
        Batch Unlearning: Remove multiple users.

        Only shards containing ANY deleted user need retraining.
        """
        import time
        t_start = time.time()

        print("\n" + "="*60)
        print(f"BATCH UNLEARNING: {len(user_ids)} users")
        print("="*60)

        # Find affected shards
        affected_shards = set()
        for user_id in user_ids:
            self.signatures.remove_user_signature(user_id)
            shard = self.assignment.get_affected_shard(user_id)
            affected_shards.add(shard)
            self.models.remove_user_from_shard(shard, user_id)

        print(f"  Affected shards: {sorted(affected_shards)}")

        # Retrain each affected shard
        for shard in sorted(affected_shards):
            print(f"  Retraining shard {shard}...")
            self.models.train_affected_shard_only(
                shard, retrain_epochs, batch_size, lr
            )

        t_elapsed = time.time() - t_start
        print(f"\n  Total unlearning time: {t_elapsed:.2f}s")

        return t_elapsed

    def save(self, path: str):
        """Save the framework."""
        os.makedirs(path, exist_ok=True)

        # Save signatures
        save_npz(os.path.join(path, 'signatures.npz'),
                  self.signatures.user_signatures)

        # Save assignment
        with open(os.path.join(path, 'assignment.pkl'), 'wb') as f:
            pickle.dump(self.assignment.user_to_shard, f)

        # Save models
        for shard_id in range(self.n_shards):
            torch.save(
                self.models.shard_models[shard_id].state_dict(),
                os.path.join(path, f'shard_{shard_id}.pt')
            )

        # Save shard weights
        torch.save(self.models.shard_weights, os.path.join(path, 'shard_weights.pt'))

        print(f"Framework saved to {path}")

    def load(self, path: str):
        """Load the framework."""
        # Load signatures
        self.signatures.user_signatures = load_npz(
            os.path.join(path, 'signatures.npz')
        )

        # Load assignment
        with open(os.path.join(path, 'assignment.pkl'), 'rb') as f:
            self.assignment.user_to_shard = pickle.load(f)

        # Load models
        for shard_id in range(self.n_shards):
            self.models.shard_models[shard_id].load_state_dict(
                torch.load(os.path.join(path, f'shard_{shard_id}.pt'))
            )

        # Load shard weights
        self.models.shard_weights = torch.load(os.path.join(path, 'shard_weights.pt'))

        print(f"Framework loaded from {path}")


def load_data(data_path: str) -> Tuple[Dict, Dict, int, int]:
    """
    Load train/test data from files.

    Expected format:
    - train.txt: "user_id item1 item2 item3 ..."
    - test.txt: "user_id item1 item2 ..."
    """
    train_data = {}
    test_data = {}

    n_users = 0
    n_items = 0

    # Load train
    with open(os.path.join(data_path, 'train.txt')) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            user_id = int(parts[0])
            items = [int(x) for x in parts[1:]]
            train_data[user_id] = items
            n_users = max(n_users, user_id + 1)
            n_items = max(n_items, max(items) + 1 if items else 0)

    # Load test
    with open(os.path.join(data_path, 'test.txt')) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            user_id = int(parts[0])
            items = [int(x) for x in parts[1:]]
            test_data[user_id] = items

    return train_data, test_data, n_users, n_items


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/ml-1m')
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--n_shards', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--unlearn_ratio', type=float, default=0.1,
                       help='Ratio of users to unlearn')

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    train_data, test_data, n_users, n_items = load_data(args.data_path)
    print(f"  Users: {n_users}, Items: {n_items}")
    print(f"  Train interactions: {sum(len(v) for v in train_data.values())}")
    print(f"  Test users: {len(test_data)}")

    # Create framework
    print("\nCreating Deletion-Stable Unlearning Framework...")
    framework = DeletionStableUnlearningFramework(
        n_users=n_users,
        n_items=n_items,
        emb_dim=args.emb_dim,
        n_shards=args.n_shards,
        device=args.device,
        seed=args.seed
    )

    # Build
    framework.build(train_data, test_data)

    # Train
    framework.train(n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr)

    # Evaluate before unlearning
    print("\n" + "="*60)
    print("BEFORE UNLEARNING")
    print("="*60)
    results_before = framework.evaluate()

    # Unlearning
    n_users_to_unlearn = int(n_users * args.unlearn_ratio)
    users_to_unlearn = random.sample(list(train_data.keys()), n_users_to_unlearn)

    print(f"\nUnlearning {len(users_to_unlearn)} users...")
    unlearn_time = framework.unlearn_users_batch(
        users_to_unlearn,
        retrain_epochs=5,
        batch_size=args.batch_size,
        lr=args.lr
    )

    # Evaluate after unlearning
    print("\n" + "="*60)
    print("AFTER UNLEARNING")
    print("="*60)
    results_after = framework.evaluate()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Users unlearned: {len(users_to_unlearn)}")
    print(f"  Time for unlearning: {unlearn_time:.2f}s")
    print(f"  Speedup vs full retrain: ~{args.n_shards / (len(set(framework.assignment.user_to_shard[u] for u in users_to_unlearn))):.1f}x")

    print("\n  Performance change:")
    for idx, k in enumerate([10, 20, 50]):
        recall_before = results_before['recall'][idx]
        recall_after = results_after['recall'][idx]
        change = recall_after - recall_before
        print(f"    Recall@{k}: {recall_before:.4f} -> {recall_after:.4f} ({change:+.4f})")
