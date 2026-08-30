"""
Train LightGCN standalone to get pretrained embeddings.
This creates user_pretrain_lightgcn.pk and item_pretrain_lightgcn.pk
which can be used for RecEraser-LightGCN partitioning and initialization.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adagrad
import pickle
from scipy.sparse import coo_matrix

# Setup paths
PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

os.environ['RECUNLEARN_DATA_PATH'] = os.path.join(os.path.dirname(PROJ), 'data/')
os.environ['RECUNLEARN_DATASET'] = 'ml-1m'

from utility.parser import parse_args
from utility.load_data import Data

args = parse_args()
args.data_path = os.path.join(os.path.dirname(PROJ), 'data/ml-1m')

# Load data
data_gen = Data(
    path=args.data_path,
    batch_size=args.batch_size,
    part_type=1,
    part_num=1,
    part_T=5
)

n_users = data_gen.n_users
n_items = data_gen.n_items
emb_dim = args.embed_size


def build_adj_matrix(train_items, n_users, n_items):
    """Build normalized adjacency matrix."""
    rows = []
    cols = []
    data = []

    for user, items in train_items.items():
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


class LightGCNStandalone(nn.Module):
    """Standalone LightGCN for pretraining."""

    def __init__(self, n_users, n_items, emb_dim, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_matrix):
        """Get final embeddings after graph convolution."""
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        all_embeddings = torch.cat([user_emb, item_emb], dim=0)
        embeddings_list = [all_embeddings]

        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

        final_embeddings = torch.stack(embeddings_list, dim=0).mean(dim=0)

        return final_embeddings[:n_users], final_embeddings[n_users:]


def bpr_loss(users, pos_items, neg_items, user_emb, pos_emb, neg_emb):
    """BPR loss."""
    pos_scores = (users * pos_items).sum(dim=1)
    neg_scores = (users * neg_items).sum(dim=1)

    diff = pos_scores - neg_scores
    loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

    reg_loss = (users.pow(2).sum() + pos_items.pow(2).sum() +
                neg_items.pow(2).sum()) / users.size(0) * 0.01

    return loss + reg_loss


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Build adjacency matrix
    print("Building adjacency matrix...")
    adj_matrix = build_adj_matrix(data_gen.train_items, n_users, n_items).to(device)

    # Create model
    model = LightGCNStandalone(n_users, n_items, emb_dim, n_layers=3).to(device)
    optimizer = Adagrad(model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)

    n_train = data_gen.n_train
    n_batches = n_train // args.batch_size + 1

    print(f"\nTraining LightGCN: {n_users} users, {n_items} items")
    print(f"Training samples: {n_train}, Batches per epoch: {n_batches}")

    for epoch in range(args.epoch):
        loss_sum = 0
        for _ in range(n_batches):
            users, pos_items, neg_items = data_gen.sample()

            users_t = torch.LongTensor(users).to(device)
            pos_t = torch.LongTensor(pos_items).to(device)
            neg_t = torch.LongTensor(neg_items).to(device)

            # Get embeddings
            user_emb_all, item_emb_all = model(adj_matrix)

            batch_user_emb = user_emb_all[users_t]
            batch_pos_emb = item_emb_all[pos_t]
            batch_neg_emb = item_emb_all[neg_t]

            # BPR loss
            loss = bpr_loss(batch_user_emb, batch_pos_emb, batch_neg_emb,
                           batch_user_emb, batch_pos_emb, batch_neg_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={loss_sum:.4f}")

    # Get final embeddings
    print("\nExtracting embeddings...")
    user_emb_final, item_emb_final = model(adj_matrix)

    user_emb_np = user_emb_final.detach().cpu().numpy()
    item_emb_np = item_emb_final.detach().cpu().numpy()

    # Save embeddings
    save_path = args.data_path
    user_save_path = os.path.join(save_path, 'user_pretrain_lightgcn.pk')
    item_save_path = os.path.join(save_path, 'item_pretrain_lightgcn.pk')

    with open(user_save_path, 'wb') as f:
        pickle.dump(user_emb_np, f)

    with open(item_save_path, 'wb') as f:
        pickle.dump(item_emb_np, f)

    print(f"\nSaved LightGCN embeddings:")
    print(f"  User: {user_save_path} {user_emb_np.shape}")
    print(f"  Item: {item_save_path} {item_emb_np.shape}")


if __name__ == '__main__':
    train()
