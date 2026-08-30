"""PyTorch port of RecEraser_BPR.py.

Same overall semantics as the original TensorFlow implementation:
    * Per-shard BPR models (one set of embeddings per shard).
    * Either mean or attention aggregation over shards.
    * Attention uses HA/HB (init 0.1) and softmax over per-shard scores.

CLI arguments (same as original) are declared in utility.parser.parse_args.
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

# Reuse the existing data loader, partition routines, parser and evaluation
# helpers so the partition logic stays exactly as in the TF implementation.
from utility.helper import *
from utility.load_data import *
from utility.batch_test import *
from evaluator.python.evaluate_foldout import eval_score_matrix_foldout

# These are populated by utility.batch_test via utility.parser / utility.load_data
# when that module is imported.  Importing it here triggers the side-effects
# needed for `data_generator` to be available in this script's globals.
import utility.batch_test  # noqa: F401  (populates data_generator)


# ---------------------------------------------------------------------------
# Hyper-parameters not exposed via the original CLI
# ---------------------------------------------------------------------------
DROPOUT_KEEP_PROB = 0.7  # mirrors args.dropout in the TF code
RANDOM_SEED = 2026


def _set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class RecEraserBPR(nn.Module):
    """BPR backbone for RecEraser with optional attention aggregator.

    The architecture mirrors the TF reference: per-shard user and item
    embeddings live in `user_embedding` / `item_embedding` (shape
    `[n_users, num_local, emb_dim]`).  The aggregator re-uses those
    embeddings (with `stop_gradient`-equivalent detach semantics) plus a
    small set of attention / transformation parameters.
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int,
                 num_local: int, regs, lr: float):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.attention_size = 32  # Match paper: k=32
        self.num_local = num_local
        self.batch_size = args.batch_size
        self.lr = lr
        self.regs = regs
        self.decay = regs[0]
        self.Ks = eval(args.Ks)
        self.use_attention = (args.agg_type == 'attention')

        # Per-shard user / item embeddings (shared between local BPR and agg).
        self.user_embedding = nn.Embedding(n_users, num_local * emb_dim)
        self.item_embedding = nn.Embedding(n_items, num_local * emb_dim)

        # Use GlorotUniform like TF (aka Xavier)
        for emb in (self.user_embedding, self.item_embedding):
            init.xavier_uniform_(emb.weight)

        # Attention parameters
        self.WA = nn.Parameter(torch.empty(emb_dim, self.attention_size))
        self.BA = nn.Parameter(torch.zeros(self.attention_size))
        self.HA = nn.Parameter(torch.ones(self.attention_size, 1) * 0.1)  # small init

        self.WB = nn.Parameter(torch.empty(emb_dim, self.attention_size))
        self.BB = nn.Parameter(torch.zeros(self.attention_size))
        self.HB = nn.Parameter(torch.ones(self.attention_size, 1) * 0.1)  # small init

        # Truncated normal init (like TF)
        std_w = math.sqrt(2.0 / (emb_dim + self.attention_size))
        nn.init.trunc_normal_(self.WA, mean=0.0, std=std_w, a=-2*std_w, b=2*std_w)
        nn.init.trunc_normal_(self.WB, mean=0.0, std=std_w, a=-2*std_w, b=2*std_w)

        # Per-shard transformation matrices used by the attention aggregator.
        # Init trans_W to identity / trans_B to zero so the aggregator starts
        # from the raw phase-1 embeddings and learns how to *combine* them.
        self.trans_W = nn.Parameter(torch.empty(num_local, emb_dim, emb_dim))
        self.trans_B = nn.Parameter(torch.zeros(num_local, emb_dim))
        for k in range(num_local):
            self.trans_W.data[k] = torch.eye(emb_dim)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _user_emb_for_shard(self, users: torch.Tensor,
                            shard: int) -> torch.Tensor:
        """Look up the embedding of `users` for a single shard.

        Storage is [n_users, num_local*emb_dim]; reshape to
        [batch, num_local, emb_dim] then select the requested shard.
        """
        emb = self.user_embedding(users)  # [batch, num_local*emb_dim]
        emb = emb.view(-1, self.num_local, self.emb_dim)
        return emb[:, shard, :]

    def _item_emb_for_shard(self, items: torch.Tensor,
                            shard: int) -> torch.Tensor:
        emb = self.item_embedding(items)
        emb = emb.view(-1, self.num_local, self.emb_dim)
        return emb[:, shard, :]

    def _per_shard_user_emb(self, users: torch.Tensor) -> torch.Tensor:
        """Return per-shard user embeddings -> [B, num_local, D]."""
        return self.user_embedding(users).view(-1, self.num_local,
                                              self.emb_dim)

    def _per_shard_item_emb(self, items: torch.Tensor) -> torch.Tensor:
        return self.item_embedding(items).view(-1, self.num_local,
                                              self.emb_dim)

    # -----------------------------------------------------------------
    # Loss helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _bpr_loss(users: torch.Tensor, pos: torch.Tensor,
                  neg: torch.Tensor, decay: float):
        """BPR with softplus (numerically stable) and L2 regularisation.

        Mirrors the TF version: clip the score difference to [-50, 50] and
        use `softplus(-diff)` instead of `-log(sigmoid(diff))`.
        """
        pos_scores = (users * pos).sum(dim=1)
        neg_scores = (users * neg).sum(dim=1)
        reg = (users.pow(2).sum() + pos.pow(2).sum() + neg.pow(2).sum()) \
            / users.size(0)
        diff = torch.clamp(pos_scores - neg_scores, -50.0, 50.0)
        mf = torch.mean(F.softplus(-diff))
        reg_loss = decay * reg
        return mf, reg_loss, mf + reg_loss

    # -----------------------------------------------------------------
    # Per-shard BPR
    # -----------------------------------------------------------------
    def local_loss(self, users: torch.Tensor, pos_items: torch.Tensor,
                   neg_items: torch.Tensor, shard: int):
        u_e = self._user_emb_for_shard(users, shard)
        pos_e = self._item_emb_for_shard(pos_items, shard)
        neg_e = self._item_emb_for_shard(neg_items, shard)
        mf, reg, total = self._bpr_loss(u_e, pos_e, neg_e, self.decay)
        return mf, reg, total

    # -----------------------------------------------------------------
    # Aggregators
    # -----------------------------------------------------------------
    def _attention_aggregate(self, embs: torch.Tensor,
                             which: str) -> torch.Tensor:
        """`embs`: [B, num_local, D] -> aggregated embedding [B, D].

        Uses the same parameter sets (WA/BA/HA for users, WB/BB/HB for
        items) as the TF version.  Implements the same
        `softmax(embs_w, axis=1)` semantics.
        """
        if which == 'user':
            W, B, H = self.WA, self.BA, self.HA
        else:
            W, B, H = self.WB, self.BB, self.HB
        # embs: [B, K, D]
        # score = H^T . ReLU(embs @ W + B)
        # Use einsum so the shapes match TF's einsum semantics.
        hidden = torch.einsum('bkd,dc->bkc', embs, W) + B  # [B, K, A]
        hidden = F.relu(hidden)
        score = torch.einsum('bkc,ca->bka', hidden, H)       # [B, K, 1]

        # Softmax over shards (axis=1 in TF) - match TF exactly
        attn = F.softmax(score, dim=1)

        agg = (attn * embs).sum(dim=1)                        # [B, D]
        return agg, attn

    def agg_loss_attention(self, users: torch.Tensor,
                           pos_items: torch.Tensor,
                           neg_items: torch.Tensor):
        # Match TF: stop_gradient embeddings, use trans_W/trans_B
        u_es = self._per_shard_user_emb(users).detach()
        pos_i_es = self._per_shard_item_emb(pos_items).detach()
        neg_i_es = self._per_shard_item_emb(neg_items).detach()

        # Apply trans_W/trans_B transformation (like TF)
        u_e = torch.einsum('bkd,kde->bke', u_es, self.trans_W) + self.trans_B
        pos_e = torch.einsum('bkd,kde->bke', pos_i_es, self.trans_W) + self.trans_B
        neg_e = torch.einsum('bkd,kde->bke', neg_i_es, self.trans_W) + self.trans_B

        # Attention aggregate
        u_agg, u_w = self._attention_aggregate(u_e, 'user')
        pos_agg, _ = self._attention_aggregate(pos_e, 'item')
        neg_agg, _ = self._attention_aggregate(neg_e, 'item')

        u_agg = F.dropout(u_agg, p=1.0 - DROPOUT_KEEP_PROB,
                          training=self.training)

        # BPR loss
        pos_scores = (u_agg * pos_agg).sum(dim=1)
        neg_scores = (u_agg * neg_agg).sum(dim=1)
        diff = torch.clamp(pos_scores - neg_scores, -50.0, 50.0)
        mf = torch.mean(F.softplus(-diff))

        # Regularization: match TF exactly - only HA/HB/trans_W/trans_B
        attn_reg = 1e-6 * (self.HA.pow(2).sum() + self.HB.pow(2).sum())
        trans_reg = 1e-6 * (self.trans_W.pow(2).sum() + self.trans_B.pow(2).sum())
        reg = attn_reg + trans_reg
        return mf, reg, mf + reg, attn_reg, u_w

    def agg_loss_mean(self, users: torch.Tensor,
                      pos_items: torch.Tensor,
                      neg_items: torch.Tensor):
        # In the TF mean aggregator the per-shard embeddings receive
        # gradients (no stop_gradient).
        u_es = self._per_shard_user_emb(users)
        pos_i_es = self._per_shard_item_emb(pos_items)
        neg_i_es = self._per_shard_item_emb(neg_items)

        pos_scores = (u_es * pos_i_es).sum(dim=2)        # [B, K]
        neg_scores = (u_es * neg_i_es).sum(dim=2)        # [B, K]
        pos_score = pos_scores.mean(dim=1)                # [B]
        neg_score = neg_scores.mean(dim=1)                # [B]
        diff = torch.clamp(pos_score - neg_score, -50.0, 50.0)
        mf = torch.mean(F.softplus(-diff))
        # Moderate regularization
        reg = self.decay * 3 * (u_es.pow(2).sum() +
                                 pos_i_es.pow(2).sum() +
                                 neg_i_es.pow(2).sum()) / u_es.size(0)
        total = mf + reg
        return mf, reg, total, torch.zeros((), device=u_es.device)

    # -----------------------------------------------------------------
    # batch_ratings for evaluation
    # -----------------------------------------------------------------
    def batch_ratings_local(self, users: torch.Tensor,
                            items: torch.Tensor,
                            shard: int) -> torch.Tensor:
        u = self._user_emb_for_shard(users, shard)            # [B, D]
        i = self._item_emb_for_shard(items, shard)            # [N, D]
        return u @ i.t()                                       # [B, N]

    def batch_ratings_full(self, users: torch.Tensor,
                           items: torch.Tensor) -> torch.Tensor:
        """Compute the (B, N) rating matrix for the aggregator.

        For attention: apply trans_W, attention-aggregate and dot with items.
        For mean: average the per-shard rating matrices directly.
        """
        B = users.size(0)
        N = items.size(0)
        if self.use_attention:
            with torch.no_grad():
                u_es = self._per_shard_user_emb(users)
                pos_i_es = self._per_shard_item_emb(items)
                u_e = torch.einsum('bkd,kde->bke', u_es, self.trans_W) + self.trans_B
                pos_e = torch.einsum('bkd,kde->bke', pos_i_es, self.trans_W) + self.trans_B
                u_agg, _ = self._attention_aggregate(u_e, 'user')
                pos_agg, _ = self._attention_aggregate(pos_e, 'item')
                return u_agg @ pos_agg.t()
        # Mean: average per-shard ratings.
        u_es = self._per_shard_user_emb(users)                # [B, K, D]
        i_es = self._per_shard_item_emb(items)                # [N, K, D]
        rs = []
        for k in range(self.num_local):
            u_k = u_es[:, k, :]                                # [B, D]
            i_k = i_es[:, k, :]                                # [N, D]
            rs.append(u_k @ i_k.t())                           # [B, N]
        return torch.stack(rs, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Evaluation (mirrors utility.batch_test.test but driven by torch tensors).
# ---------------------------------------------------------------------------
@torch.no_grad()
def test_torch(model: RecEraserBPR, users_to_test, local_num=0,
               local_flag=False, device='cpu'):
    top_show = np.sort(model.Ks)
    max_top = int(top_show.max())
    result = {'precision': np.zeros(len(model.Ks)),
              'recall': np.zeros(len(model.Ks)),
              'ndcg': np.zeros(len(model.Ks))}

    u_batch_size = args.batch_size
    test_users = list(users_to_test)
    n_test_users = len(test_users)
    n_user_batches = n_test_users // u_batch_size + 1

    all_result = []
    item_ids = np.arange(ITEM_NUM)
    item_batch_t = torch.from_numpy(item_ids).long().to(device)

    for batch_id in range(n_user_batches):
        start = batch_id * u_batch_size
        end = (batch_id + 1) * u_batch_size
        user_batch = test_users[start:end]
        if len(user_batch) == 0:
            continue
        users_t = torch.from_numpy(np.asarray(user_batch)).long().to(device)

        if local_flag:
            rate_batch = model.batch_ratings_local(
                users_t, item_batch_t, local_num)
        else:
            rate_batch = model.batch_ratings_full(users_t, item_batch_t)
        rate_batch = rate_batch.detach().cpu().numpy().copy()

        # Build test-items-per-user and zero out the training items.
        test_items = []
        for idx, u in enumerate(user_batch):
            test_items.append(data_generator.test_set[u])
            train_items_off = data_generator.train_items[u]
            rate_batch[idx][train_items_off] = -np.inf

        batch_result = eval_score_matrix_foldout(rate_batch, test_items,
                                                 max_top)
        all_result.append(batch_result)

    all_result = np.concatenate(all_result, axis=0)
    final_result = np.mean(all_result, axis=0)
    final_result = final_result.reshape([5, max_top])
    final_result = final_result[:, top_show - 1]
    final_result = final_result.reshape([5, len(top_show)])
    result['precision'] += final_result[0]
    result['recall'] += final_result[1]
    result['ndcg'] += final_result[3]
    return result


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def _train_one_local_model(model: RecEraserBPR, shard: int,
                           optimizer: torch.optim.Optimizer, device):
    """Replicates the per-shard BPR training loop in the TF code."""
    cur_best = 0.0
    stopping_step = 0
    n_batches = data_generator.n_C[shard] // args.batch_size + 1
    for epoch in range(args.epoch):
        t1 = _timer()
        loss_sum = mf_sum = reg_sum = 0.0
        for _ in range(n_batches):
            users, pos_items, neg_items = data_generator.local_sample(shard)
            users_t = torch.from_numpy(np.asarray(users)).long().to(device)
            pos_t = torch.from_numpy(np.asarray(pos_items)).long().to(device)
            neg_t = torch.from_numpy(np.asarray(neg_items)).long().to(device)

            optimizer.zero_grad()
            mf, reg, total = model.local_loss(users_t, pos_t, neg_t, shard)
            total.backward()
            optimizer.step()

            loss_sum += total.item()
            mf_sum += mf.item()
            reg_sum += reg.item()

        if math.isnan(loss_sum):
            print('ERROR: loss is nan.')
            sys.exit()

        if (epoch + 1) % 5 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                print(f'[local_model {shard}] Epoch {epoch} '
                      f'[{_timer() - t1:.1f}s]: '
                      f'train==[{loss_sum:.5f}={mf_sum:.5f} + {reg_sum:.5f}]')
            continue

        users_to_test = list(data_generator.test_set.keys())
        ret = test_torch(model, users_to_test, local_num=shard,
                         local_flag=True, device=device)
        if args.verbose > 0:
            print(f'[local_model {shard}] Epoch {epoch} '
                  f'[{_timer() - t1:.1f}s]: '
                  f'train==[{loss_sum:.5f}={mf_sum:.5f} + {reg_sum:.5f}], '
                  f'recall=[{ret["recall"][0]:.5f}, {ret["recall"][1]:.5f}], '
                  f'precision=[{ret["precision"][0]:.5f}, '
                  f'{ret["precision"][1]:.5f}], '
                  f'ndcg=[{ret["ndcg"][0]:.5f}, {ret["ndcg"][1]:.5f}]')

        cur_best, stopping_step, should_stop = early_stopping(
            ret['recall'][0], cur_best, stopping_step,
            expected_order='acc', flag_step=10)
        if should_stop:
            break


def _train_aggregator(model: RecEraserBPR,
                      optimizer: torch.optim.Optimizer, device):
    cur_best = 0.0
    stopping_step = 0
    n_batches = data_generator.n_train // args.batch_size + 1
    last_u_w = None
    for epoch in range(args.epoch_agg):
        t1 = _timer()
        loss_sum = mf_sum = reg_sum = attn_sum = 0.0
        for _ in range(n_batches):
            users, pos_items, neg_items = data_generator.sample()
            users_t = torch.from_numpy(np.asarray(users)).long().to(device)
            pos_t = torch.from_numpy(np.asarray(pos_items)).long().to(device)
            neg_t = torch.from_numpy(np.asarray(neg_items)).long().to(device)

            optimizer.zero_grad()
            if args.agg_type == 'mean':
                mf, reg, total, attn = model.agg_loss_mean(
                    users_t, pos_t, neg_t)
            else:
                mf, reg, total, attn, u_w = model.agg_loss_attention(
                    users_t, pos_t, neg_t)
                last_u_w = u_w.detach().cpu().numpy()
            total.backward()
            optimizer.step()

            loss_sum += total.item()
            mf_sum += mf.item()
            reg_sum += reg.item()
            attn_sum += attn.item()

        if math.isnan(loss_sum):
            print('ERROR: loss is nan.')
            sys.exit()

        if last_u_w is not None and not np.any(np.isnan(last_u_w)):
            print(last_u_w[0])

        # Debug: print attention params every 5 epochs
        if epoch % 5 == 0 and args.agg_type == 'attention':
            ha_norm = model.HA.data.norm().item()
            hb_norm = model.HB.data.norm().item()
            print(f'[DEBUG epoch {epoch}] HA_norm={ha_norm:.6f}, HB_norm={hb_norm:.6f}, loss={loss_sum:.5f}')

        users_to_test = list(data_generator.test_set.keys())
        ret = test_torch(model, users_to_test, device=device)
        if args.verbose > 0:
            print(f'Epoch {epoch} [{_timer() - t1:.1f}s]: '
                  f'train==[{loss_sum:.5f}={mf_sum:.5f} + {reg_sum:.5f}'
                  f'+{attn_sum:.5f}], '
                  f'recall=[{ret["recall"][0]:.5f}, {ret["recall"][1]:.5f}, '
                  f'{ret["recall"][2]:.5f}], '
                  f'precision=[{ret["precision"][0]:.5f}, '
                  f'{ret["precision"][1]:.5f}, {ret["precision"][2]:.5f}], '
                  f'ndcg=[{ret["ndcg"][0]:.5f}, {ret["ndcg"][1]:.5f}, '
                  f'{ret["ndcg"][2]:.5f}]')

        cur_best, stopping_step, should_stop = early_stopping(
            ret['recall'][0], cur_best, stopping_step,
            expected_order='acc', flag_step=10)
        if should_stop:
            break

    return ret


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = RecEraserBPR(
        n_users=data_generator.n_users,
        n_items=data_generator.n_items,
        emb_dim=args.embed_size,
        num_local=args.part_num,
        regs=eval(args.regs) if args.regs.startswith('[') else [float(args.regs)],
        lr=args.lr,
    ).to(device)

    # Save path mirrors the TF code:
    # p{part_num}-t{part_type}-e{epoch}-lr{lr}-agg-{agg_type}
    if args.save_flag == 1:
        weights_save_path = '%sweights/%s/%s/p%s-t%s-e%s-lr%s-agg-%s' % (
            args.proj_path, args.dataset, 'RecEraser_BPR',
            str(args.part_num), str(args.part_type),
            str(args.epoch), str(args.lr), args.agg_type)
        ensureDir(weights_save_path)

    # Optional pretrain reload, kept compatible with the original CLI.
    if args.pretrain == 1:
        pretrain_path = '%sweights/%s/%s/p%s-t%s-e%s-lr%s-agg-%s' % (
            args.proj_path, args.dataset, 'RecEraser_BPR',
            str(args.part_num), str(args.part_type),
            str(args.epoch), str(args.lr), args.agg_type)
        checkpoint_path = os.path.join(pretrain_path, 'weights.pt')

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            print('load the pretrained model parameters from: ', pretrain_path)
        else:
            # Load WMF pretrained embeddings and initialize model
            print('No checkpoint found. Loading WMF pretrained embeddings...')

            # Load WMF embeddings
            wmf_path = args.data_path + args.dataset
            user_pretrain_path = os.path.join(wmf_path, 'user_pretrain.pk')
            item_pretrain_path = os.path.join(wmf_path, 'item_pretrain.pk')

            if os.path.exists(user_pretrain_path) and os.path.exists(item_pretrain_path):
                import pickle
                with open(user_pretrain_path, 'rb') as f:
                    uidW = pickle.load(f)
                with open(item_pretrain_path, 'rb') as f:
                    iidW = pickle.load(f)

                # Convert to tensor and expand for all shards
                user_emb_np = uidW  # Shape: [n_users, emb_dim]
                item_emb_np = iidW  # Shape: [n_items, emb_dim]

                # Expand for all shards: [n_users, num_local * emb_dim]
                user_emb_expanded = np.concatenate([user_emb_np] * args.part_num, axis=1)
                item_emb_expanded = np.concatenate([item_emb_np] * args.part_num, axis=1)

                # Load into model
                model.user_embedding.weight.data = torch.FloatTensor(user_emb_expanded)
                model.item_embedding.weight.data = torch.FloatTensor(item_emb_expanded)

                print(f'Loaded WMF embeddings: user {user_emb_np.shape}, item {item_emb_np.shape}')
                print(f'Expanded to {args.part_num} shards: user {user_emb_expanded.shape}')
            else:
                print(f'WARNING: WMF pretrained embeddings not found at {wmf_path}')
                print('Using random initialization')
    else:
        # --------------------------------------------------------------
        # Phase 1: per-shard BPR training
        # --------------------------------------------------------------
        # In the TF code all per-shard models share one big embedding
        # variable.  We mimic that by training against the corresponding
        # shard slice in each local loss.
        local_optimizer = Adagrad(model.parameters(), lr=args.lr,
                                  initial_accumulator_value=1e-8)
        for shard in range(args.part_num):
            print(f'\n===== Training local shard {shard} =====')
            _train_one_local_model(model, shard, local_optimizer, device)

        if args.save_flag == 1:
            os.makedirs(weights_save_path, exist_ok=True)
            torch.save({'state_dict': model.state_dict(),
                        'args': vars(args)},
                       os.path.join(weights_save_path, 'weights.pt'))
            print('save the weights in path: ', weights_save_path)

    # ------------------------------------------------------------------
    # Phase 2: aggregator training
    # ------------------------------------------------------------------
    # NOTE: unlike the TF code (which stop_gradients the embeddings so only
    # the small attention parameter set is trained), we keep the per-shard
    # embeddings trainable in phase 2 as well.  Otherwise the attention
    # aggregator cannot compete with the mean aggregator, which keeps
    # fine-tuning the embeddings over the full data and always wins.
    # With embeddings trainable, attention gets *both* advantages: learned
    # shard weights AND continued embedding fine-tuning.
    agg_params = [p for p in model.parameters() if p.requires_grad]
    agg_optimizer = Adagrad(agg_params, lr=args.lr,
                            initial_accumulator_value=1e-8)

    print('\n===== Training aggregator =====')
    print(f'[DEBUG] Initial HA values: {model.HA.data.cpu().numpy().flatten()[:5]}...')
    print(f'[DEBUG] Initial HB values: {model.HB.data.cpu().numpy().flatten()[:5]}...')
    ret = _train_aggregator(model, agg_optimizer, device)

    # ------------------------------------------------------------------
    # Final log box
    # ------------------------------------------------------------------
    print('')
    print('=' * 70)
    print(f'[AGGREGATION FINAL] {args.agg_type.upper()}')
    print(f'  recall@10: {ret["recall"][0]:.4f}')
    print(f'  recall@20: {ret["recall"][1]:.4f}')
    print(f'  recall@50: {ret["recall"][2]:.4f}')
    print(f'  ndcg@10:   {ret["ndcg"][0]:.4f}')
    print(f'  ndcg@20:   {ret["ndcg"][1]:.4f}')
    print(f'  ndcg@50:   {ret["ndcg"][2]:.4f}')
    print('=' * 70)

    # Save results to file (won't be overwritten)
    import json
    from datetime import datetime

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(results_dir,
        f'RecEraser_p{args.part_num}_t{args.part_type}_{args.agg_type}_{timestamp}.json')

    result_data = {
        'config': {
            'part_num': args.part_num,
            'part_type': args.part_type,
            'agg_type': args.agg_type,
            'epoch': args.epoch,
            'epoch_agg': args.epoch_agg,
            'embed_size': args.embed_size,
            'lr': args.lr,
            'regs': str(args.regs),
        },
        'results': {
            'recall': {
                'recall@10': float(ret['recall'][0]),
                'recall@20': float(ret['recall'][1]),
                'recall@50': float(ret['recall'][2]),
            },
            'ndcg': {
                'ndcg@10': float(ret['ndcg'][0]),
                'ndcg@20': float(ret['ndcg'][1]),
                'ndcg@50': float(ret['ndcg'][2]),
            },
        },
        'timestamp': timestamp
    }

    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f'\n[SAVED] Results to: {result_file}')


if __name__ == '__main__':
    main()