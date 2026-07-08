"""
Script to display evaluation results from trained weights.
Usage: python show_results.py
"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def find_weights(model_type, part_type, agg_type):
    """Find weights folder for given configuration."""
    base_path = 'weights/ml-1m/{}'.format(model_type)

    agg_suffix = '' if agg_type == 'attention' else '_mean'
    pattern = '{}/*type-{}{}*/'.format(base_path, part_type, agg_suffix)

    matches = glob.glob(pattern)
    if matches:
        # Check if checkpoint exists
        checkpoint_file = os.path.join(matches[0], 'checkpoint')
        if os.path.exists(checkpoint_file):
            return matches[0]
    return None

def main():
    results = {}

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

    print("=" * 70)
    print("NUM-10 ORACLE RESULTS (100 epochs)")
    print("=" * 70)

    for model_name, model_display in models:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_display}")
        print("=" * 70)

        model_results = {'attention': {}, 'mean': {}}

        for part_type, part_name in partitions:
            for agg_type, agg_display in agg_types:
                weights_path = find_weights(model_name, part_type, agg_type)

                if weights_path:
                    checkpoint_file = os.path.join(weights_path, 'checkpoint')
                    with open(checkpoint_file, 'r') as f:
                        content = f.read()
                        # Extract step number
                        if 'model_checkpoint_path:' in content:
                            for line in content.split('\n'):
                                if 'model_checkpoint_path:' in line:
                                    step = line.split('model_checkpoint_path:')[1].strip().split('-')[-1]
                                    break
                        else:
                            step = "unknown"

                    print(f"{part_name:10} | {agg_display:10} | Step: {step}")
                    model_results[agg_type][part_name] = weights_path
                else:
                    print(f"{part_name:10} | {agg_display:10} | NOT TRAINED")
                    model_results[agg_type][part_name] = None

        results[model_display] = model_results

    print("\n" + "=" * 70)
    print("WEIGHTS PATHS")
    print("=" * 70)

    for model_name, model_display in models:
        print(f"\n{model_display}:")
        for agg_type in ['attention', 'mean']:
            agg_display = 'Attention' if agg_type == 'attention' else 'MEAN'
            print(f"\n  {agg_display}:")
            for part_type, part_name in partitions:
                wp = results[model_display][agg_type].get(part_name)
                if wp:
                    print(f"    {part_name}: {wp}")
                else:
                    print(f"    {part_name}: (not trained)")

    print("\n" + "=" * 70)
    print("Run with --test_flag full to get R@K and NDCG@K metrics")
    print("=" * 70)

if __name__ == '__main__':
    main()
