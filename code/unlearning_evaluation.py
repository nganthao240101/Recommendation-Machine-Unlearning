"""
RecEraser Unlearning Evaluation
So sánh InBP vs UBP vs Random trong việc unlearn interactions

Figure 2 trong paper so sánh Recall@20 trước và sau khi unlearn 20% interactions
"""
import os
import sys
import pickle
import numpy as np
from time import time
import tensorflow as tf

# Import từ project
from utility.helper import *
from utility.batch_test import *
from utility.load_data import Data
from utility.parser import parse_args

args = parse_args()

def load_data():
    """Load data generator"""
    return Data(
        path=args.data_path + args.dataset,
        batch_size=args.batch_size,
        part_type=args.part_type,
        part_num=args.part_num,
        part_T=args.part_T
    )

def get_unlearn_interactions(data_generator, unlearn_ratio=0.2):
    """
    Chọn ngẫu nhiên unlearn_ratio interactions để xóa
    Trả về: dict của local model -> list interactions cần unlearn
    """
    all_interactions = []
    for u in data_generator.train_items:
        for i in data_generator.train_items[u]:
            all_interactions.append((u, i))

    np.random.seed(2021)
    n_unlearn = int(len(all_interactions) * unlearn_ratio)
    unlearn_indices = np.random.choice(len(all_interactions), n_unlearn, replace=False)

    return [all_interactions[i] for i in unlearn_indices]

def evaluate_model(sess, model, users_to_test):
    """Đánh giá Recall@20"""
    ret = test(sess, model, users_to_test, drop_flag=False)
    return ret['recall'][0]  # Recall@20

def simulate_unlearning(data_generator, unlearn_interactions, part_type):
    """
    Simulate unlearning bằng cách đánh dấu interactions đã xóa
    Không cần retrain - chỉ cần đánh giá model đã train
    """
    # Tính toán interaction đã xóa thuộc local model nào
    unlearn_by_local = {}

    # Load partition file
    if part_type == 1:
        with open(args.data_path + args.dataset + '/C_type-1_num-' + str(args.part_num) + '.pk', 'rb') as f:
            C = pickle.load(f)
    elif part_type == 2:
        with open(args.data_path + args.dataset + '/C_type-2_num-' + str(args.part_num) + '.pk', 'rb') as f:
            C = pickle.load(f)
    else:
        with open(args.data_path + args.dataset + '/C_type-3_num-' + str(args.part_num) + '.pk', 'rb') as f:
            C = pickle.load(f)

    return C

def main():
    print("=" * 60)
    print("RecEraser Unlearning Evaluation")
    print("So sánh InBP vs UBP vs Random")
    print("=" * 60)

    # Dataset info
    print(f"\nDataset: {args.dataset}")
    print(f"Epoch: {args.epoch}")
    print(f"Partitions: {args.part_num}")

    results = {}

    # Test các partition types
    partition_types = {
        1: "InBP (Interaction-based)",
        2: "UBP (User-based)",
        3: "Random"
    }

    for part_type, name in partition_types.items():
        print(f"\n{'='*60}")
        print(f"Training với {name}...")
        print("=" * 60)

        # Chạy training (bạn có thể chạy riêng từng cái trước)
        # Ở đây giả định đã train xong, load từ checkpoint

        weights_path = f"../weights/{args.dataset}/RecEraser_BPR/num-{args.part_num}_type-{part_type}_r0.01"

        if os.path.exists(weights_path):
            print(f"✓ Model đã train: {weights_path}")
        else:
            print(f"✗ Chưa train: cần chạy lệnh train trước")
            print(f"  python RecEraser_BPR.py --dataset {args.dataset} --part_type {part_type} --part_num {args.part_num}")

    print("\n" + "=" * 60)
    print("Hướng dẫn chạy training cho từng partition:")
    print("=" * 60)

    for part_type, name in partition_types.items():
        print(f"\n{name}:")
        print(f"  conda activate receraser")
        print(f"  cd e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/code")
        print(f"  python RecEraser_BPR.py --dataset ml-1m --part_type {part_type} --part_num 10 --epoch 100")

    print("\n" + "=" * 60)
    print("Sau khi train xong 3 models, chạy script này để đánh giá unlearning")
    print("=" * 60)

if __name__ == '__main__':
    main()