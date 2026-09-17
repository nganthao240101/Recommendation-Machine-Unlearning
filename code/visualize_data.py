"""
Visualize dữ liệu MovieLens 1M
Chạy: python visualize_data.py
"""

import os
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

PROJ = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(PROJ), 'data', 'ml-1m')


def load_data():
    """Load train và test data"""
    train_file = os.path.join(DATA_DIR, 'train.txt')
    test_file = os.path.join(DATA_DIR, 'test.txt')

    train_data = {}
    test_data = {}
    all_items = set()

    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            uid = int(parts[0])
            items = [int(i) for i in parts[1:]]
            train_data[uid] = items
            all_items.update(items)

    with open(test_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            uid = int(parts[0])
            items = [int(i) for i in parts[1:]]
            test_data[uid] = items

    return train_data, test_data, all_items


def visualize_basic_stats(train_data, test_data, all_items):
    """Thống kê cơ bản"""
    n_users = len(train_data)
    n_items = max(all_items) + 1
    n_train = sum(len(items) for items in train_data.values())
    n_test = sum(len(items) for items in test_data.values())

    print("=" * 60)
    print("THỐNG KÊ CƠ BẢN")
    print("=" * 60)
    print(f"Số users:          {n_users:,}")
    print(f"Số items:          {n_items:,}")
    print(f"Số train interactions: {n_train:,}")
    print(f"Số test interactions:  {n_test:,}")
    print(f"Avg interactions/user (train): {n_train/n_users:.2f}")
    print(f"Avg interactions/user (test): {n_test/n_users:.2f}")
    print(f"Sparsity:          {100 * n_train / (n_users * n_items):.4f}%")
    print("=" * 60)


def visualize_user_interactions(train_data, test_data):
    """Biểu đồ phân bố interactions của users"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Train interactions per user
    train_counts = [len(train_data[u]) for u in train_data]
    axes[0, 0].hist(train_counts, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    axes[0, 0].set_xlabel('Số interactions')
    axes[0, 0].set_ylabel('Số users')
    axes[0, 0].set_title('Train: Phân bố interactions/user')
    axes[0, 0].axvline(np.mean(train_counts), color='red', linestyle='--', label=f'Mean={np.mean(train_counts):.1f}')
    axes[0, 0].legend()

    # 2. Test interactions per user
    test_counts = [len(test_data.get(u, [])) for u in train_data]
    test_counts_nonzero = [c for c in test_counts if c > 0]
    axes[0, 1].hist(test_counts_nonzero, bins=50, color='coral', edgecolor='white', alpha=0.7)
    axes[0, 1].set_xlabel('Số interactions')
    axes[0, 1].set_ylabel('Số users')
    axes[0, 1].set_title('Test: Phân bố interactions/user')
    axes[0, 1].axvline(np.mean(test_counts_nonzero), color='red', linestyle='--', label=f'Mean={np.mean(test_counts_nonzero):.1f}')
    axes[0, 1].legend()

    # 3. So sánh train vs test
    axes[1, 0].boxplot([train_counts, test_counts_nonzero], labels=['Train', 'Test'])
    axes[1, 0].set_ylabel('Số interactions')
    axes[1, 0].set_title('So sánh Train vs Test interactions')

    # 4. Top users
    user_train_counts = [(u, len(items)) for u, items in train_data.items()]
    user_train_counts.sort(key=lambda x: x[1], reverse=True)
    top_users = user_train_counts[:20]
    axes[1, 1].barh([f'User {u}' for u, _ in top_users], [c for _, c in top_users], color='green', alpha=0.7)
    axes[1, 1].set_xlabel('Số interactions')
    axes[1, 1].set_title('Top 20 users có nhiều interactions nhất')
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('visualize_user_interactions.png', dpi=150, bbox_inches='tight')
    print("Đã lưu: visualize_user_interactions.png")
    plt.close()


def visualize_item_popularity(train_data, all_items):
    """Biểu đồ popularity của items"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Đếm số users đã tương tác với mỗi item
    item_counts = Counter()
    for items in train_data.values():
        item_counts.update(items)

    counts = sorted(item_counts.values(), reverse=True)

    # 1. Distribution
    axes[0].hist(counts, bins=100, color='purple', edgecolor='white', alpha=0.7)
    axes[0].set_xlabel('Số users đã tương tác')
    axes[0].set_ylabel('Số items')
    axes[0].set_title('Phân bố item popularity')
    axes[0].axvline(np.mean(counts), color='red', linestyle='--', label=f'Mean={np.mean(counts):.1f}')
    axes[0].legend()

    # 2. Top items
    top_items = item_counts.most_common(20)
    axes[1].barh([f'Item {i}' for i, _ in top_items], [c for _, c in top_items], color='orange', alpha=0.7)
    axes[1].set_xlabel('Số users')
    axes[1].set_title('Top 20 items phổ biến nhất')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('visualize_item_popularity.png', dpi=150, bbox_inches='tight')
    print("Đã lưu: visualize_item_popularity.png")
    plt.close()


def visualize_interaction_matrix(train_data, n_samples=100):
    """Visualize ma trận tương tác (sparse)"""
    # Lấy mẫu users
    users = list(train_data.keys())[:n_samples]
    n_users = len(users)
    n_items = max(max(items) for items in train_data.values()) + 1

    # Tạo ma trận
    matrix = np.zeros((n_users, n_items))
    for i, uid in enumerate(users):
        for item in train_data[uid]:
            if item < n_items:
                matrix[i, item] = 1

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(matrix, aspect='auto', cmap='Blues', interpolation='none')
    ax.set_xlabel('Item ID')
    ax.set_ylabel('User ID (mẫu)')
    ax.set_title(f'Ma trận tương tác (mẫu {n_users} users đầu tiên)')
    plt.tight_layout()
    plt.savefig('visualize_interaction_matrix.png', dpi=150, bbox_inches='tight')
    print("Đã lưu: visualize_interaction_matrix.png")
    plt.close()


def visualize_unlearn_impact(train_data):
    """Visualize impact của unlearning"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. User có ít interactions nhất
    user_counts = [(u, len(items)) for u, items in train_data.items()]
    user_counts.sort(key=lambda x: x[1])

    n_unlearn_list = [1, 10, 50, 100, 200, 500]
    retained_data_pct = []

    for n_unlearn in n_unlearn_list:
        # Unlearn n users ít nhất
        total_interactions = sum(len(items) for items in train_data.values())
        removed_interactions = sum(c for _, c in user_counts[:n_unlearn])
        retained_pct = 100 * (total_interactions - removed_interactions) / total_interactions
        retained_data_pct.append(retained_pct)

    axes[0].bar(range(len(n_unlearn_list)), retained_data_pct, color='green', alpha=0.7)
    axes[0].set_xticks(range(len(n_unlearn_list)))
    axes[0].set_xticklabels([str(n) for n in n_unlearn_list])
    axes[0].set_xlabel('Số users unlearn (ít nhất)')
    axes[0].set_ylabel('% Data còn lại')
    axes[0].set_title('Unlearn users ít interactions nhất')
    for i, v in enumerate(retained_data_pct):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center')

    # 2. User có nhiều interactions nhất
    user_counts.sort(key=lambda x: x[1], reverse=True)
    retained_data_pct2 = []

    for n_unlearn in n_unlearn_list:
        total_interactions = sum(len(items) for items in train_data.values())
        removed_interactions = sum(c for _, c in user_counts[:n_unlearn])
        retained_pct = 100 * (total_interactions - removed_interactions) / total_interactions
        retained_data_pct2.append(retained_pct)

    axes[1].bar(range(len(n_unlearn_list)), retained_data_pct2, color='red', alpha=0.7)
    axes[1].set_xticks(range(len(n_unlearn_list)))
    axes[1].set_xticklabels([str(n) for n in n_unlearn_list])
    axes[1].set_xlabel('Số users unlearn (nhiều nhất)')
    axes[1].set_ylabel('% Data còn lại')
    axes[1].set_title('Unlearn users nhiều interactions nhất')
    for i, v in enumerate(retained_data_pct2):
        axes[1].text(i, v + 1, f'{v:.1f}%', ha='center')

    # 3. Random unlearn
    random.seed(42)
    retained_data_pct3 = []

    for n_unlearn in n_unlearn_list:
        total_interactions = sum(len(items) for items in train_data.values())
        selected = random.sample(list(train_data.keys()), n_unlearn)
        removed_interactions = sum(len(train_data[u]) for u in selected)
        retained_pct = 100 * (total_interactions - removed_interactions) / total_interactions
        retained_data_pct3.append(retained_pct)

    axes[2].bar(range(len(n_unlearn_list)), retained_data_pct3, color='blue', alpha=0.7)
    axes[2].set_xticks(range(len(n_unlearn_list)))
    axes[2].set_xticklabels([str(n) for n in n_unlearn_list])
    axes[2].set_xlabel('Số users unlearn (ngẫu nhiên)')
    axes[2].set_ylabel('% Data còn lại')
    axes[2].set_title('Unlearn users ngẫu nhiên')
    for i, v in enumerate(retained_data_pct3):
        axes[2].text(i, v + 1, f'{v:.1f}%', ha='center')

    plt.tight_layout()
    plt.savefig('visualize_unlearn_impact.png', dpi=150, bbox_inches='tight')
    print("Đã lưu: visualize_unlearn_impact.png")
    plt.close()


def print_user_examples(train_data):
    """In ví dụ về users"""
    print("\n" + "=" * 60)
    print("VÍ DỤ VỀ USERS")
    print("=" * 60)

    # User có ít interactions nhất
    user_counts = [(u, len(items), items[:5]) for u, items in train_data.items()]
    user_counts.sort(key=lambda x: x[1])

    print("\n5 users có ÍT interactions nhất:")
    for u, count, items in user_counts[:5]:
        print(f"  User {u}: {count} interactions, ví dụ items: {items}")

    # User có nhiều interactions nhất
    user_counts.sort(key=lambda x: x[1], reverse=True)

    print("\n5 users có NHIỀU interactions nhất:")
    for u, count, items in user_counts[:5]:
        print(f"  User {u}: {count} interactions, ví dụ items: {items[:10]}...")

    # Random users
    random.seed(123)
    random_users = random.sample(list(train_data.keys()), 5)

    print("\n5 users NGẪU NHIÊN:")
    for u in random_users:
        items = train_data[u]
        print(f"  User {u}: {len(items)} interactions, ví dụ items: {items[:10]}...")


def main():
    print("Loading data...")
    train_data, test_data, all_items = load_data()

    visualize_basic_stats(train_data, test_data, all_items)
    print_user_examples(train_data)

    print("\nGenerating visualizations...")

    try:
        visualize_user_interactions(train_data, test_data)
        visualize_item_popularity(train_data, all_items)
        visualize_interaction_matrix(train_data, n_samples=200)
        visualize_unlearn_impact(train_data)
        print("\nTất cả visualizations đã được lưu!")
    except Exception as e:
        print(f"Lỗi khi tạo visualizations: {e}")
        print("Đảm bảo đã cài đặt: pip install matplotlib numpy")


if __name__ == '__main__':
    main()
