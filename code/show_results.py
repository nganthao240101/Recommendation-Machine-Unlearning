"""
Script to display evaluation metrics from trained weights.
Usage: python show_results.py
"""
import os
import sys
import glob
import heapq
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def find_weights(model_type, part_type, agg_type):
    """Find weights folder for given configuration."""
    base_path = 'weights/ml-1m/{}'.format(model_type)

    # Find all matching folders
    all_pattern = '{}/*type-{}_r*/'.format(base_path, part_type)
    all_matches = glob.glob(all_pattern)

    # Separate into attention (no _mean suffix) and mean (_mean suffix) folders
    attention_folders = []
    mean_folders = []

    for match in all_matches:
        if match.endswith('_mean'):
            mean_folders.append(match)
        else:
            attention_folders.append(match)

    print(f"  DEBUG: type-{part_type}, agg-{agg_type}")
    print(f"    All matches: {all_matches}")
    print(f"    Attention folders: {attention_folders}")
    print(f"    Mean folders: {mean_folders}")

    if agg_type == 'attention':
        # Return first attention folder that has checkpoint
        for folder in attention_folders:
            checkpoint_file = os.path.join(folder, 'checkpoint')
            if os.path.exists(checkpoint_file):
                return folder
    else:  # mean
        # Return first mean folder that has checkpoint
        for folder in mean_folders:
            checkpoint_file = os.path.join(folder, 'checkpoint')
            if os.path.exists(checkpoint_file):
                return folder

    return None

def load_train_test_data():
    """Load train and test data."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'ml-1m')

    train_file = os.path.join(data_path, 'train.txt')
    test_file = os.path.join(data_path, 'test.txt')

    train_data = {}
    test_data = {}
    all_users = set()
    n_items = 0

    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                user = int(parts[0])
                items = [int(x) for x in parts[1:]]
                train_data[user] = set(items)
                all_users.add(user)
                n_items = max(n_items, max(items) + 1)

    with open(test_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                user = int(parts[0])
                items = [int(x) for x in parts[1:]]
                test_data[user] = items
                all_users.add(user)

    return train_data, test_data, n_items

def recall_at_k(rank, ground_truth, k):
    """Calculate Recall@K."""
    hits = sum(1 for item in rank[:k] if item in ground_truth)
    return hits / len(ground_truth) if len(ground_truth) > 0 else 0

def ndcg_at_k(rank, ground_truth, k):
    """Calculate NDCG@K."""
    dcg = 0.0
    for i, item in enumerate(rank[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)

    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0

def evaluate_bpr_embeddings(weights_path, n_users, n_items, train_data, test_data, is_attention=False, Ks=[10, 20, 50]):
    """Evaluate BPR model by loading embeddings directly."""
    # Load checkpoint
    checkpoint = tf.train.load_checkpoint(weights_path)
    var_shape_map = checkpoint.get_variable_to_shape_map()

    print(f"  Found {len(var_shape_map)} variables in checkpoint")

    # Print all variable names for debugging
    print("  All variables in checkpoint:")
    for name in sorted(var_shape_map.keys()):
        print(f"    - {name}: {var_shape_map[name]}")

    # Find all model variables
    user_emb = None
    item_emb = None
    trans_W = None
    trans_B = None
    WA = None
    BA = None
    HA = None
    WB = None
    BB = None
    HB = None

    for name in var_shape_map.keys():
        # Skip optimizer states
        if '/Adagrad' in name or '/Adam' in name:
            continue

        # Main embeddings
        if name == 'user_embedding':
            user_emb = checkpoint.get_tensor(name)
            print(f"  User emb: {name} -> {user_emb.shape}")
        elif name == 'item_embedding':
            item_emb = checkpoint.get_tensor(name)
            print(f"  Item emb: {name} -> {item_emb.shape}")
        # trans weights
        elif name == 'user_embedding_1':
            trans_W = checkpoint.get_tensor(name)
            print(f"  Trans W: {name} -> {trans_W.shape}")
        elif name == 'user_embedding_2':
            trans_B = checkpoint.get_tensor(name)
            print(f"  Trans B: {name} -> {trans_B.shape}")
        # attention weights
        elif name == 'WA':
            WA = checkpoint.get_tensor(name)
            print(f"  WA: {name} -> {WA.shape}")
        elif name == 'WB':
            WB = checkpoint.get_tensor(name)
            print(f"  WB: {name} -> {WB.shape}")
        elif name == 'BA':
            BA = checkpoint.get_tensor(name)
            print(f"  BA: {name} -> {BA.shape}")
        elif name == 'BB':
            BB = checkpoint.get_tensor(name)
            print(f"  BB: {name} -> {BB.shape}")
        elif name == 'HA':
            HA = checkpoint.get_tensor(name)
            print(f"  HA: {name} -> {HA.shape}")
        elif name == 'HB':
            HB = checkpoint.get_tensor(name)
            print(f"  HB: {name} -> {HB.shape}")

    if user_emb is None or item_emb is None:
        print("  ERROR: Could not find embeddings")
        return None

    # For ATTENTION model: apply full attention mechanism
    if is_attention and trans_W is not None and trans_B is not None:
        print("  Applying attention mechanism...")

        # Transform: emb_transformed = emb @ trans_W + trans_B
        user_emb_t = np.einsum('ijk,jkl->ijl', user_emb, trans_W) + trans_B  # (n_users, num_local, emb_dim)
        item_emb_t = np.einsum('ijk,jkl->ijl', item_emb, trans_W) + trans_B  # (n_items, num_local, emb_dim)

        # Calculate attention weights for users: score = HA^T * ReLU(WA * emb + BA)
        if WA is not None and BA is not None and HA is not None:
            # user attention: (n_users, num_local, emb_dim) @ (emb_dim, attn_size) -> (n_users, num_local, attn_size)
            user_proj = np.einsum('ijk,kl->ijl', user_emb_t, WA) + BA
            user_proj = np.maximum(user_proj, 0)  # ReLU
            user_score = np.einsum('ijk,kl->ij', user_proj, HA)  # (n_users, num_local)
            # Softmax
            user_score_exp = np.exp(user_score - np.max(user_score, axis=1, keepdims=True))
            user_attn = user_score_exp / (np.sum(user_score_exp, axis=1, keepdims=True) + 1e-8)
        else:
            # Fallback to mean if attention weights not found
            print("  WARNING: Attention weights not found, using mean")
            user_attn = np.ones((user_emb_t.shape[0], user_emb_t.shape[1])) / user_emb_t.shape[1]

        # Calculate attention weights for items
        if WB is not None and BB is not None and HB is not None:
            item_proj = np.einsum('ijk,kl->ijl', item_emb_t, WB) + BB
            item_proj = np.maximum(item_proj, 0)  # ReLU
            item_score = np.einsum('ijk,kl->ij', item_proj, HB)  # (n_items, num_local)
            # Softmax
            item_score_exp = np.exp(item_score - np.max(item_score, axis=1, keepdims=True))
            item_attn = item_score_exp / (np.sum(item_score_exp, axis=1, keepdims=True) + 1e-8)
        else:
            print("  WARNING: Attention weights not found, using mean")
            item_attn = np.ones((item_emb_t.shape[0], item_emb_t.shape[1])) / item_emb_t.shape[1]

        # Weighted sum: output = sum(attn_weight * emb)
        user_emb = np.einsum('ij,ijk->ik', user_attn, user_emb_t)  # (n_users, emb_dim)
        item_emb = np.einsum('ij,ijk->ik', item_attn, item_emb_t)  # (n_items, emb_dim)

    else:
        # Simple mean aggregation for MEAN model
        print("  Using simple mean aggregation")
        user_emb = np.mean(user_emb, axis=1)
        item_emb = np.mean(item_emb, axis=1)

    print(f"  Final user emb shape: {user_emb.shape}, Item emb shape: {item_emb.shape}")

    # Check for NaN/Inf in embeddings
    if np.any(np.isnan(user_emb)) or np.any(np.isinf(user_emb)):
        print("  WARNING: user_emb contains NaN or Inf!")
        user_emb = np.nan_to_num(user_emb, nan=0.0, posinf=1.0, neginf=-1.0)
    if np.any(np.isnan(item_emb)) or np.any(np.isinf(item_emb)):
        print("  WARNING: item_emb contains NaN or Inf!")
        item_emb = np.nan_to_num(item_emb, nan=0.0, posinf=1.0, neginf=-1.0)

    # Evaluate
    results = {}
    for k in Ks:
        results[f'Recall@{k}'] = []
        results[f'NDCG@{k}'] = []

    test_users = list(test_data.keys())

    for user in test_users:
        test_items = test_data.get(user, [])
        if not test_items:
            continue

        # Compute scores
        scores = np.dot(user_emb[user], item_emb.T)

        # Mask training items
        train_items = train_data.get(user, set())
        for item in train_items:
            scores[item] = -np.inf

        # Get top-K
        top_k = heapq.nlargest(max(Ks), range(len(scores)), scores.take)

        # Calculate metrics
        for k in Ks:
            results[f'Recall@{k}'].append(recall_at_k(top_k, test_items, k))
            results[f'NDCG@{k}'].append(ndcg_at_k(top_k, test_items, k))

    # Average
    avg_results = {}
    for metric, values in results.items():
        avg_results[metric] = np.mean(values) if values else 0

    return avg_results

def get_epochs_from_checkpoint(weights_path):
    """Get number of epochs from checkpoint file."""
    checkpoint_file = os.path.join(weights_path, 'checkpoint')
    try:
        with open(checkpoint_file, 'r') as f:
            content = f.read()
            # Look for step number in checkpoint path
            import re
            match = re.search(r'model_checkpoint_path:\s*"[^"]*-(\d+)"', content)
            if match:
                return int(match.group(1))
    except:
        pass
    return None

def main():
    print("=" * 90)
    print("NUM-10 ORACLE RESULTS - EVALUATION FROM WEIGHTS")
    print("=" * 90)

    # Load data
    train_data, test_data, n_items = load_train_test_data()
    n_users = max(train_data.keys(), default=0) + 1

    print(f"\nLoaded: {n_users} users, {n_items} items")

    models = [
        ('RecEraser_BPR', 'BPR'),
    ]

    partitions = [
        (1, 'InP'),
        (2, 'UBP'),
        (3, 'Random'),
        (4, 'IBP')
    ]

    agg_types = [
        ('attention', 'Attention'),
        ('mean', 'MEAN')
    ]

    all_results = {}

    for model_name, model_display in models:
        print(f"\n{'=' * 90}")
        print(f"MODEL: {model_display}")
        print("=" * 90)

        model_results = {}

        for part_type, part_name in partitions:
            for agg_type, agg_display in agg_types:
                weights_path = find_weights(model_name, part_type, agg_type)

                if weights_path:
                    epochs = get_epochs_from_checkpoint(weights_path)
                    epochs_str = f"({epochs} epochs)" if epochs else ""
                    print(f"\nEvaluating {part_name} | {agg_display} {epochs_str}...")
                    print(f"  Path: {weights_path}")

                    is_attention = (agg_type == 'attention')

                    try:
                        results = evaluate_bpr_embeddings(
                            weights_path, n_users, n_items, train_data, test_data,
                            is_attention=is_attention
                        )
                        if results:
                            print(f"  Recall@10: {results['Recall@10']:.6f}")
                            print(f"  Recall@20: {results['Recall@20']:.6f}")
                            print(f"  Recall@50: {results['Recall@50']:.6f}")
                            print(f"  NDCG@10: {results['NDCG@10']:.6f}")
                            print(f"  NDCG@20: {results['NDCG@20']:.6f}")
                            print(f"  NDCG@50: {results['NDCG@50']:.6f}")
                            model_results[f'{part_name}_{agg_display}'] = results
                        else:
                            print(f"  ERROR: Could not load embeddings")
                            model_results[f'{part_name}_{agg_display}'] = None
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        model_results[f'{part_name}_{agg_display}'] = None
                else:
                    print(f"\n{part_name} | {agg_display}: NOT TRAINED")
                    model_results[f'{part_name}_{agg_display}'] = None

        all_results[model_display] = model_results

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)

    for model_display in ['BPR', 'LightGCN']:
        if model_display in all_results:
            print(f"\n{model_display}:")
            print("-" * 90)
            print(f"{'Partition':<12} {'Agg':<12} {'R@10':<10} {'R@20':<10} {'R@50':<10} {'N@10':<10} {'N@20':<10} {'N@50':<10}")
            print("-" * 90)

            for part_name in ['InP', 'UBP', 'Random', 'IBP']:
                for agg_display in ['Attention', 'MEAN']:
                    key = f'{part_name}_{agg_display}'
                    if all_results[model_display].get(key):
                        r = all_results[model_display][key]
                        r10 = r.get('Recall@10', 0)
                        r20 = r.get('Recall@20', 0)
                        r50 = r.get('Recall@50', 0)
                        n10 = r.get('NDCG@10', 0)
                        n20 = r.get('NDCG@20', 0)
                        n50 = r.get('NDCG@50', 0)
                        print(f"{part_name:<12} {agg_display:<12} {r10:<10.6f} {r20:<10.6f} {r50:<10.6f} {n10:<10.6f} {n20:<10.6f} {n50:<10.6f}")
                    else:
                        print(f"{part_name:<12} {agg_display:<12} {'N/A':<60}")

if __name__ == '__main__':
    main()
