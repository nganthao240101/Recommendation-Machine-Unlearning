"""
Script to evaluate and display metrics from trained weights.
Usage: python show_results.py
"""
import os
import sys
import glob
import json
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

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
    with open(os.path.join(weights_path, 'checkpoint'), 'r') as f:
        for line in f:
            if 'model_checkpoint_path:' in line:
                return line.split('model_checkpoint_path:')[1].strip().strip('"')
    return None

def load_data():
    """Load data for evaluation."""
    from utility.load_data import Data
    from utility.parser import parse_args

    args = parse_args()
    data = Data(path='data/ml-1m', batch_size=1024,
                part_type=args.part_type, part_num=args.part_num, part_T=args.part_T)
    return data, args

def evaluate_model(model_class, data_config, weights_path, sess):
    """Evaluate a single model."""
    model = model_class(data_config=data_config)
    saver = tf.compat.v1.train.Saver()

    ckpt_path = get_checkpoint_path(weights_path)
    if ckpt_path:
        saver.restore(sess, ckpt_path)
    else:
        return None

    users_to_test = list(data_config['test_users'])

    # Get all item embeddings for scoring
    items_to_test = list(range(data_config['n_items']))

    ratings = []
    batch_size = 512
    for start in range(0, len(users_to_test), batch_size):
        end = min(start + batch_size, len(users_to_test))
        batch_users = users_to_test[start:end]

        # Get user embeddings
        user_emb = sess.run(model.weights['user_embedding'])
        item_emb = sess.run(model.weights['item_embedding'])

        # Compute scores for this batch
        for u in batch_users:
            scores = np.dot(user_emb[u], item_emb.T)
            ratings.append(scores)

    return ratings

def main():
    print("=" * 80)
    print("NUM-10 ORACLE RESULTS (100 epochs)")
    print("=" * 80)

    results = {}

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

    # Load data first
    from utility.load_data import Data
    from utility.parser import parse_args
    args = parse_args()
    data_generator = Data(path='data/ml-1m', batch_size=1024,
                          part_type=1, part_num=10, part_T=10)

    config = {
        'n_users': data_generator.n_users,
        'n_items': data_generator.n_items,
        'test_users': data_generator.test_users
    }

    for model_name, model_display in models:
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model_display}")
        print("=" * 80)
        print(f"\n{'Partition':<12} {'Agg':<12} {'R@10':<10} {'R@20':<10} {'R@50':<10} {'N@10':<10} {'N@20':<10} {'N@50':<10}")
        print("-" * 80)

        for part_type, part_name in partitions:
            for agg_type, agg_display in agg_types:
                weights_path = find_weights(model_name, part_type, agg_type)

                if weights_path:
                    print(f"{part_name:<12} {agg_display:<12} {'TRAINED':^10}")
                    results['{}_{}'.format(part_name, agg_display)] = weights_path
                else:
                    print(f"{part_name:<12} {agg_display:<12} {'NOT TRAINED':^10}")

    print("\n" + "=" * 80)
    print("WEIGHTS FOUND")
    print("=" * 80)

    for key, path in sorted(results.items()):
        print(f"{key}: {path}")

    print("\n" + "=" * 80)
    print("Run evaluation with: python code/RecEraser_BPR.py --test_flag full")
    print("=" * 80)

if __name__ == '__main__':
    main()
