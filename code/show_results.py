"""
Script to evaluate and display metrics from trained weights.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def find_weights(model_type, part_type, agg_type):
    """Find weights folder for given configuration."""
    base_path = 'weights/ml-1m/{}'.format(model_type)

    agg_suffix = '' if agg_type == 'attention' else '_mean'
    pattern = '{}/*type-{}{}*/'.format(base_path, part_type, agg_suffix)

    matches = glob.glob(pattern)
    if matches:
        checkpoint_file = os.path.join(matches[0], 'checkpoint')
        if os.path.exists(checkpoint_file):
            return matches[0]
    return None

def get_checkpoint_path(weights_path):
    """Get the full checkpoint path."""
    checkpoint_file = os.path.join(weights_path, 'checkpoint')
    with open(checkpoint_file, 'r') as f:
        for line in f:
            if 'model_checkpoint_path:' in line:
                return os.path.join(weights_path, line.split('model_checkpoint_path:')[1].strip().strip('"'))
    return None

def precision_at_k(rank, ground_truth, k):
    """Calculate Precision@K."""
    hits = sum(1 for item in rank[:k] if item in ground_truth)
    return hits / k if k > 0 else 0

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

def evaluate_weights(users, items, train_data, test_data, Ks=[10, 20, 50]):
    """Evaluate using precomputed user and item embeddings."""
    results = {f'Recall@{k}': [] for k in Ks}
    results.update({f'NDCG@{k}': [] for k in Ks})

    for user in users:
        # Get test items for this user
        test_items = test_data.get(user, [])
        if not test_items:
            continue

        # Get user embedding
        user_vec = items[user]

        # Compute scores for all items
        scores = np.dot(items, user_vec)

        # Exclude training items
        train_items = train_data.get(user, [])
        for item in train_items:
            scores[item] = -np.inf

        # Get top-K items
        top_k = heapq.nlargest(max(Ks), range(len(scores)), scores.take)

        # Calculate metrics
        for k in Ks:
            results[f'Recall@{k}'].append(recall_at_k(top_k, test_items, k))
            results[f'NDCG@{k}'].append(ndcg_at_k(top_k, test_items, k))

    # Average results
    avg_results = {}
    for metric, values in results.items():
        avg_results[metric] = np.mean(values) if values else 0

    return avg_results

def load_data_for_eval():
    """Load data for evaluation."""
    train_file = 'data/ml-1m/train.txt'
    test_file = 'data/ml-1m/test.txt'

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
                train_data[user] = items
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

    users = sorted(list(all_users))
    return users, train_data, test_data, n_items

def evaluate_model(model_path, model_type):
    """Evaluate a trained model."""
    from RecEraser_BPR import RecEraser_BPR

    # Load data
    users, train_data, test_data, n_items = load_data_for_eval()

    config = {
        'n_users': max(users) + 1,
        'n_items': n_items,
        'norm_adj': None
    }

    # Create model
    model = RecEraser_BPR(data_config=config)

    # Load weights
    saver = tf.compat.v1.train.Saver()
    ckpt_path = get_checkpoint_path(model_path)

    sess = tf.compat.v1.Session()
    saver.restore(sess, ckpt_path)

    # Get embeddings
    user_emb = sess.run(model.weights['user_embedding'])
    item_emb = sess.run(model.weights['item_embedding'])

    # Average across local models
    user_emb = np.mean(user_emb, axis=1)
    item_emb = np.mean(item_emb, axis=1)

    # Evaluate
    results = evaluate_weights(users, item_emb, train_data, test_data, Ks=[10, 20, 50])

    sess.close()
    return results

def main():
    print("=" * 80)
    print("NUM-10 ORACLE RESULTS - EVALUATION FROM WEIGHTS")
    print("=" * 80)

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

    results = {}

    for model_name, model_display in models:
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model_display}")
        print("=" * 80)

        model_results = {}

        for part_type, part_name in partitions:
            for agg_type, agg_display in agg_types:
                weights_path = find_weights(model_name, part_type, agg_type)

                if weights_path:
                    print(f"\nEvaluating {part_name} | {agg_display}...")
                    print(f"  Path: {weights_path}")

                    if model_name == 'RecEraser_BPR':
                        try:
                            results = evaluate_model(weights_path, 'BPR')
                            print(f"  Recall@10: {results.get('Recall@10', 0):.6f}")
                            print(f"  Recall@20: {results.get('Recall@20', 0):.6f}")
                            print(f"  Recall@50: {results.get('Recall@50', 0):.6f}")
                            print(f"  NDCG@10: {results.get('NDCG@10', 0):.6f}")
                            print(f"  NDCG@20: {results.get('NDCG@20', 0):.6f}")
                            print(f"  NDCG@50: {results.get('NDCG@50', 0):.6f}")
                            model_results[f'{part_name}_{agg_display}'] = results
                        except Exception as e:
                            print(f"  ERROR: {e}")
                            model_results[f'{part_name}_{agg_display}'] = None
                    else:
                        print(f"  LightGCN evaluation not implemented yet")
                        model_results[f'{part_name}_{agg_display}'] = None
                else:
                    print(f"\n{part_name} | {agg_display}: NOT TRAINED")

        results[model_display] = model_results

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    for model_display in ['BPR', 'LightGCN']:
        if model_display in results:
            print(f"\n{model_display}:")
            print("-" * 70)
            print(f"{'Partition':<12} {'Agg':<12} {'R@10':<10} {'R@20':<10} {'R@50':<10}")
            print("-" * 70)

            for part_name in ['InP', 'UBP', 'Random', 'IBP']:
                for agg_display in ['Attention', 'MEAN']:
                    key = f'{part_name}_{agg_display}'
                    if results[model_display].get(key):
                        r = results[model_display][key]
                        r10 = r.get('Recall@10', 0)
                        r20 = r.get('Recall@20', 0)
                        r50 = r.get('Recall@50', 0)
                        print(f"{part_name:<12} {agg_display:<12} {r10:<10.6f} {r20:<10.6f} {r50:<10.6f}")
                    else:
                        print(f"{part_name:<12} {agg_display:<12} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

if __name__ == '__main__':
    main()
