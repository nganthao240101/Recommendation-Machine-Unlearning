"""
PyTorch implementation of RecEraser with LightGCN backbone.

LightGCN: Simplifying Graph Convolutional Networks
https://arxiv.org/abs/2002.02126

Changes from BPR version:
- Uses LightGCN instead of BPR for local training
- Graph convolution instead of matrix factorization
- Neighbor aggregation through graph structure
"""
import os
import sys
import math
import time
import random
from time import time as _timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adagrad
import torch.nn.init as init
from scipy.sparse import coo_matrix

# Import data utilities
from utility.helper import *
from utility.load_data import *
from utility.batch_test import *
from evaluator.python.evaluate_foldout import eval_score_matrix_foldout
import utility.batch_test

# Constants
DROPOUT_KEEP_PROB = 0.7
RANDOM_SEED = 2026


def _set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_adj_matrix(train_items, n_users, n_items):
    """Build normalized adjacency matrix for LightGCN.

    Returns:
        adj_matrix: Normalized adjacency matrix (sparse)
        norm_adj: Normalization factor
    """
    rows = []
    cols = []
    data = []

    # Add edges: user -> item (interaction)
    for user, items in train_items.items():
        for item in items:
            rows.append(user)
            cols.append(n_users + item)  # offset by n_users for items
            data.append(1.0)

    # Create sparse matrix
    n_nodes = n_users + n_items
    adj = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    # Make symmetric (add reverse edges)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Normalize: D^(-0.5) * A * D^(-0.5)
    rows = adj.row
    cols = adj.col
    data = adj.data

    # Degree for each node
    deg = np.zeros(n_nodes)
    for i in range(len(rows)):
        deg[rows[i]] += data[i]
        deg[cols[i]] += data[i]

    # Normalization
    deg_power = np.power(deg, -0.5)
    deg_power[deg_power == float('inf')] = 0.0

    norm_rows = rows * deg_power
    norm_cols = cols * deg_power
    norm_data = data * deg_power[rows] * deg_power[cols]

    adj_normalized = coo_matrix((norm_data, (norm_rows, norm_cols)), shape=(n_nodes, n_nodes))

    return adj_normalized.tocsr()


def build_graph_from_partition(C, n_users, n_items, shard_id):
    """Build adjacency matrix for a specific shard."""
    rows = []
    cols = []
    data = []

    shard_data = C[shard_id]

    for user, items in shard_data.items():
        for item in items:
            rows.append(user)
            cols.append(n_users + item)
            data.append(1.0)

    n_nodes = n_users + n_items
    adj = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    adj = adj + adj.T

    # Normalize
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_power = np.power(deg, -0.5)
    deg_power[deg_power == float('inf')] = 0.0

    rows = adj.row
    cols = adj.col
    data = adj.data * deg_power[rows] * deg_power[cols]

    return torch.sparse_coo_tensor(
        torch.LongTensor([rows, cols]),
        torch.FloatTensor(data),
        size=(n_nodes, n_nodes)
    )


# ---------------------------------------------------------------------------
# LightGCN Model
# ---------------------------------------------------------------------------
class LightGCN(nn.Module):
    """LightGCN for RecEraser.

    LightGCN only keeps the neighbor aggregation part:
    - No self-loop needed (handled by residual)
    - Simple weighted sum aggregation
    - Layer normalization at the end
    """

    def __init__(self, n_users, n_items, emb_dim, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        # Embeddings
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        # Initialize
        init.xavier_uniform_(self.user_embedding.weight)
        init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_matrix):
        """
        LightGCN forward pass with graph convolution.

        Args:
            adj_matrix: Normalized adjacency matrix (sparse)

        Returns:
            user_emb: User embeddings after aggregation
            item_emb: Item embeddings after aggregation
        """
        # Get initial embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        all_embeddings = torch.cat([user_emb, item_emb], dim=0)

        # Graph convolution layers
        embeddings_list = [all_embeddings]

        for layer in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

        # Average over layers
        final_embeddings = torch.stack(embeddings_list, dim=0).mean(dim=0)

        # Split back to user and item
        user_emb = final_embeddings[:self.n_users]
        item_emb = final_embeddings[self.n_users:]

        return user_emb, item_emb

    def get_embeddings(self):
        """Get current embeddings without graph convolution."""
        return self.user_embedding.weight, self.item_embedding.weight


class RecEraserLightGCN(nn.Module):
    """RecEraser with LightGCN backbone.

    Supports:
    - Per-shard LightGCN training
    - Attention aggregation
    - Mean aggregation
    """

    def __init__(self, n_users, n_items, emb_dim, num_local, n_layers=3,
                 regs=None, lr=0.01, agg_type='attention'):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.num_local = num_local
        self.n_layers = n_layers
        self.regs = regs if regs else [0.01]
        self.lr = lr
        self.agg_type = agg_type

        # Attention parameters (same as BPR version)
        self.attention_size = 32
        self.WA = nn.Parameter(torch.empty(emb_dim, self.attention_size))
        self.BA = nn.Parameter(torch.zeros(self.attention_size))
        self.HA = nn.Parameter(torch.ones(self.attention_size, 1) * 0.1)

        self.WB = nn.Parameter(torch.empty(emb_dim, self.attention_size))
        self.BB = nn.Parameter(torch.zeros(self.attention_size))
        self.HB = nn.Parameter(torch.ones(self.attention_size, 1) * 0.1)

        std_w = math.sqrt(2.0 / (emb_dim + self.attention_size))
        nn.init.trunc_normal_(self.WA, mean=0.0, std=std_w, a=-2*std_w, b=2*std_w)
        nn.init.trunc_normal_(self.WB, mean=0.0, std=std_w, a=-2*std_w, b=2*std_w)

        # Per-shard LightGCN models
        self.local_models = nn.ModuleList([
            LightGCN(n_users, n_items, emb_dim, n_layers)
            for _ in range(num_local)
        ])

        # Aggregation matrices
        self.trans_W = nn.Parameter(torch.empty(num_local, emb_dim, emb_dim))
        self.trans_B = nn.Parameter(torch.zeros(num_local, emb_dim))
        for k in range(num_local):
            self.trans_W.data[k] = torch.eye(emb_dim)

    def forward_shard(self, shard_id, adj_matrix):
        """Get embeddings from a specific shard."""
        return self.local_models[shard_id](adj_matrix)

    def get_all_embeddings(self):
        """Get embeddings from all shards."""
        return [model.get_embeddings() for model in self.local_models]

    def attention_aggregate(self, all_user_embs, all_item_embs):
        """Attention-based aggregation over shards."""
        user_embs = torch.stack(all_user_embs, dim=1)  # [n_users, num_local, emb_dim]
        item_embs = torch.stack(all_item_embs, dim=1)   # [n_items, num_local, emb_dim]

        # Transform
        u_trans = torch.einsum('bmd,dke->bmke', user_embs, self.trans_W) + self.trans_B
        i_trans = torch.einsum('bmd,dke->bmke', item_embs, self.trans_W) + self.trans_B

        # User attention
        hidden_u = torch.einsum('bmd,dc->bmc', u_trans, self.WA) + self.BA
        hidden_u = F.relu(hidden_u)
        score_u = torch.einsum('bmc,ca->bma', hidden_u, self.HA)
        attn_u = F.softmax(score_u, dim=1)
        final_user = (attn_u * user_embs).sum(dim=1)

        # Item attention
        hidden_i = torch.einsum('bmd,dc->bmc', i_trans, self.WB) + self.BB
        hidden_i = F.relu(hidden_i)
        score_i = torch.einsum('bmc,ca->bma', hidden_i, self.HB)
        attn_i = F.softmax(score_i, dim=1)
        final_item = (attn_i * item_embs).sum(dim=1)

        return final_user, final_item

    def mean_aggregate(self, all_user_embs, all_item_embs):
        """Mean aggregation over shards."""
        user_embs = torch.stack(all_user_embs, dim=1)  # [n_users, num_local, emb_dim]
        item_embs = torch.stack(all_item_embs, dim=1)   # [n_items, num_local, emb_dim]

        return user_embs.mean(dim=1), item_embs.mean(dim=1)

    def aggregate(self, all_user_embs, all_item_embs):
        """Aggregate embeddings from all shards."""
        if self.agg_type == 'attention':
            return self.attention_aggregate(all_user_embs, all_item_embs)
        else:
            return self.mean_aggregate(all_user_embs, all_item_embs)


def bpr_loss(users, pos_items, neg_items, user_emb, item_emb):
    """BPR loss for LightGCN."""
    pos_scores = (users * pos_items).sum(dim=1)
    neg_scores = (users * neg_items).sum(dim=1)

    diff = pos_scores - neg_scores
    loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

    # L2 regularization
    reg_loss = (users.pow(2).sum() + pos_items.pow(2).sum() +
                neg_items.pow(2).sum()) / users.size(0)

    return loss, reg_loss * 0.01


@torch.no_grad()
def test_lightgcn(model, users_to_test, adj_matrix, device='cpu'):
    """Test LightGCN model."""
    model.eval()

    n_users = model.n_users
    n_items = model.n_items
    Ks = [10, 20, 50]

    result = {
        'precision': np.zeros(len(Ks)),
        'recall': np.zeros(len(Ks)),
        'ndcg': np.zeros(len(Ks))
    }

    user_emb, item_emb = model.forward_shard(0, adj_matrix)
    all_user_emb = user_emb.cpu().numpy()
    all_item_emb = item_emb.cpu().numpy()

    for user_batch_start in range(0, len(users_to_test), args.batch_size):
        user_batch = users_to_test[user_batch_start:user_batch_start + args.batch_size]

        # Compute scores
        user_vectors = all_user_emb[user_batch]
        scores = np.matmul(user_vectors, all_item_emb.T)

        # Mask training items
        for idx, u in enumerate(user_batch):
            train_items = list(data_generator.train_items.get(u, []))
            scores[idx, train_items] = -np.inf

        # Get top-K
        for k_idx, k in enumerate(Ks):
            top_k = np.argsort(scores)[:, -k:]
            for idx, u in enumerate(user_batch):
                predicted = top_k[idx]
                actual = data_generator.test_set.get(u, [])

                # Precision
                hit = len(set(predicted) & set(actual))
                result['precision'][k_idx] += hit / k
                result['recall'][k_idx] += hit / len(actual) if actual else 0

                # NDCG
                dcg = 0
                for i, item in enumerate(predicted):
                    if item in actual:
                        dcg += 1 / np.log2(i + 2)
                idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actual), k)))
                result['ndcg'][k_idx] += dcg / idcg if idcg > 0 else 0

    n_users = len(users_to_test)
    result['precision'] /= n_users
    result['recall'] /= n_users
    result['ndcg'] /= n_users

    return result


def train_local_lightgcn(model, shard_id, adj_matrix, optimizer, device, epochs=100):
    """Train LightGCN on a specific shard."""
    model.train()

    n_batches = max(1, len(data_generator.n_C[shard_id]) // args.batch_size)

    for epoch in range(epochs):
        loss_sum = 0
        for _ in range(n_batches):
            # Sample batch
            users, pos_items, neg_items = data_generator.local_sample(shard_id)

            users_t = torch.LongTensor(users).to(device)
            pos_t = torch.LongTensor(pos_items).to(device)
            neg_t = torch.LongTensor(neg_items).to(device)

            # Forward pass
            user_emb, item_emb = model.forward_shard(shard_id, adj_matrix)

            # Get embeddings for batch
            batch_user_emb = user_emb[users_t]
            batch_pos_emb = item_emb[pos_t]
            batch_neg_emb = item_emb[neg_t]

            # BPR loss
            loss, reg_loss = bpr_loss(batch_user_emb, batch_pos_emb, batch_neg_emb,
                                       batch_user_emb, batch_pos_emb)

            total_loss = loss + reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_sum += total_loss.item()

        if epoch % 10 == 0:
            print(f'  [LightGCN Shard {shard_id}] Epoch {epoch}: loss={loss_sum:.4f}')

    return model


def train_aggregator_lightgcn(model, adj_matrix, optimizer, device, epochs=50):
    """Train aggregation layer."""
    model.train()

    n_batches = data_generator.n_train // args.batch_size + 1

    for epoch in range(epochs):
        loss_sum = 0
        attn_sum = 0

        for _ in range(n_batches):
            users, pos_items, neg_items = data_generator.sample()

            users_t = torch.LongTensor(users).to(device)
            pos_t = torch.LongTensor(pos_items).to(device)
            neg_t = torch.LongTensor(neg_items).to(device)

            # Get embeddings from all shards
            all_user_embs = []
            all_item_embs = []

            for shard_id in range(model.num_local):
                user_emb, item_emb = model.forward_shard(shard_id, adj_matrix)
                all_user_embs.append(user_emb)
                all_item_embs.append(item_emb)

            # Aggregate
            final_user, final_item = model.aggregate(all_user_embs, all_item_embs)

            # Get batch embeddings
            batch_user = final_user[users_t]
            batch_pos = final_item[pos_t]
            batch_neg = final_item[neg_t]

            # Loss
            pos_scores = (batch_user * batch_pos).sum(dim=1)
            neg_scores = (batch_user * batch_neg).sum(dim=1)
            diff = torch.clamp(pos_scores - neg_scores, -50, 50)
            loss = torch.mean(F.softplus(-diff))

            # Attention regularization
            attn_reg = 1e-6 * (model.HA.pow(2).sum() + model.HB.pow(2).sum())

            total_loss = loss + attn_reg

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            attn_sum += attn_reg.item()

        if epoch % 5 == 0:
            print(f'  [Aggregator] Epoch {epoch}: loss={loss_sum:.4f}, attn_reg={attn_sum:.6f}')

    return model


def main():
    _set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    n_users = data_generator.n_users
    n_items = data_generator.n_items
    emb_dim = args.embed_size
    num_local = args.part_num

    # Build adjacency matrix
    print("Building adjacency matrix...")
    adj_matrix = build_adj_matrix(data_generator.train_items, n_users, n_items)
    adj_tensor = torch.sparse_coo_tensor(
        torch.LongTensor([adj_matrix.row, adj_matrix.col]),
        torch.FloatTensor(adj_matrix.data),
        size=adj_matrix.shape
    ).to(device)

    # Create model
    model = RecEraserLightGCN(
        n_users=n_users,
        n_items=n_items,
        emb_dim=emb_dim,
        num_local=num_local,
        n_layers=3,
        regs=eval(args.regs),
        lr=args.lr,
        agg_type=args.agg_type
    ).to(device)

    print(f"\nModel created: {n_users} users, {n_items} items, {num_local} shards")

    # Phase 1: Local training
    print("\n" + "="*60)
    print("PHASE 1: LOCAL TRAINING (LightGCN)")
    print("="*60)

    local_optimizer = Adagrad(model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)

    for shard in range(num_local):
        print(f"\nTraining shard {shard}...")

        # Build shard-specific graph
        shard_adj = build_graph_from_partition(
            data_generator.interactions, n_users, n_items, shard
        ).to(device)

        model = train_local_lightgcn(
            model, shard, shard_adj, local_optimizer, device,
            epochs=args.epoch
        )

    # Phase 2: Aggregator training
    print("\n" + "="*60)
    print("PHASE 2: AGGREGATOR TRAINING")
    print("="*60)

    agg_optimizer = Adagrad(model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)
    model = train_aggregator_lightgcn(model, adj_tensor, agg_optimizer, device,
                                       epochs=args.epoch_agg)

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    users_to_test = list(data_generator.test_set.keys())
    ret = test_lightgcn(model, users_to_test, adj_tensor, device)

    print(f"\n{'='*60}")
    print(f"[LIGHTGCN FINAL] {args.agg_type.upper()}")
    print(f"  recall@10: {ret['recall'][0]:.4f}")
    print(f"  recall@20: {ret['recall'][1]:.4f}")
    print(f"  recall@50: {ret['recall'][2]:.4f}")
    print(f"  ndcg@10:   {ret['ndcg'][0]:.4f}")
    print(f"  ndcg@20:   {ret['ndcg'][1]:.4f}")
    print(f"  ndcg@50:   {ret['ndcg'][2]:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
