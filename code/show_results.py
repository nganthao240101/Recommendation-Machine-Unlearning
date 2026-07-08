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

                if weights_path and os.path.exists(weights_path):
                    print(f"\n{part_name} | {agg_display}: {weights_path}")
                    print(f"  Weights found: YES ✓")
                    model_results[agg_type][part_name] = "Found"
                else:
                    print(f"\n{part_name} | {agg_display}: Not found")
                    print(f"  Weights found: NO ✗")
                    model_results[agg_type][part_name] = "N/A"

        results[model_display] = model_results

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for model_name, model_display in models:
        print(f"\n{model_display}:")
        print("-" * 50)
        for agg_type in ['attention', 'mean']:
            agg_display = 'Attention' if agg_type == 'attention' else 'MEAN'
            print(f"  {agg_display}:")
            for part_type, part_name in partitions:
                status = results[model_display][agg_type].get(part_name, 'N/A')
                print(f"    {part_name}: {status}")

    print("\n" + "=" * 70)
    print("To run evaluation, use --test_flag full with training scripts")
    print("=" * 70)

if __name__ == '__main__':
    main()
