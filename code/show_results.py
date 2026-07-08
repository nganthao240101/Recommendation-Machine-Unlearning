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

    # Try multiple patterns to find the weights folder
    patterns = []

    if agg_type == 'attention':
        # Attention: try both with and without _mean suffix
        patterns.append('{}/*type-{}_r*/'.format(base_path, part_type))
        patterns.append('{}/*type-{}_r*_mean*/'.format(base_path, part_type))
    else:  # mean
        # MEAN: try with _mean suffix first, then without
        patterns.append('{}/*type-{}_r*_mean*/'.format(base_path, part_type))
        patterns.append('{}/*type-{}_r*/'.format(base_path, part_type))

    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            for match in matches:
                checkpoint_file = os.path.join(match, 'checkpoint')
                if os.path.exists(checkpoint_file):
                    return match

    return None

def get_checkpoint_path(weights_path):
    """Get the full checkpoint path."""
    checkpoint_file = os.path.join(weights_path, 'checkpoint')
    with open(checkpoint_file, 'r') as f:
        for line in f:
            if 'model_checkpoint_path:' in line:
                return os.path.join(weights_path, line.split('model_checkpoint_path:')[1].strip().strip('"'))
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

def evaluate_model(weights_path, n_users, n_items, train_data, test_data, Ks=[10, 20, 50]):
    """Evaluate model using saved weights."""
    # List all variables in checkpoint
    checkpoint = tf.train.load_checkpoint(weights_path)
    var_shape_map = checkpoint.get_variable_to_shape_map()

    print(f"  Found {len(var_shape_map)} variables in checkpoint")

    # Filter only model variables (exclude optimizer states)
    model_vars = {}
    for name in var_shape_map.keys():
        # Only keep embedding and weight variables
        if 'user_embedding' in name or 'item_embedding' in name or 'trans_' in name:
            model_vars[name] = checkpoint.get_tensor(name)

    if not model_vars:
        print("  ERROR: No model variables found")
        return None

    # Get embeddings
    user_emb = None
    item_emb = None

    for name, tensor in model_vars.items():
        if 'user_embedding' in name and 'trans' not in name:
            user_emb = tensor
            print(f"  User emb: {name} -> {tensor.shape}")
        elif 'item_embedding' in name and 'trans' not in name:
            item_emb = tensor
            print(f"  Item emb: {name} -> {tensor.shape}")

    if user_emb is None or item_emb is None:
        print("  ERROR: Could not find embeddings")
        return None

    # Average across local models if multi-dimensional
    if len(user_emb.shape) == 3:
        user_emb = np.mean(user_emb, axis=1)
    if len(item_emb.shape) == 3:
        item_emb = np.mean(item_emb, axis=1)

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
        ('RecEraser_LightGCN', 'LightGCN')
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
                    print(f"\nEvaluating {part_name} | {agg_display}...")
                    print(f"  Path: {weights_path}")

                    try:
                        results = evaluate_model(weights_path, n_users, n_items, train_data, test_data)
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
