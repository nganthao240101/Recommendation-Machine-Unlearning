# RecEraser

This is our implementation of the paper:

*Chong Chen, Fei Sun, Min Zhang and Bolin Ding. 2022. [Recommendation Unlearning.](https://arxiv.org/pdf/2201.06820.pdf)
In TheWebConf'22.*

**Please cite our TheWebConf'22 paper if you use our codes. Thanks!**

## Note on Pretrain Method

This implementation uses **WMF (Weighted Matrix Factorization)** for pretrain embeddings, not BPR-WMF, for the following reasons:

1. **Stability**: WMF uses ALS (Alternating Least Squares) with closed-form updates, which is more stable than BPR's SGD
2. **No negative sampling bias**: WMF uses confidence weights, avoiding bias from negative sampling
3. **Better for clustering**: WMF embeddings are more uniformly distributed, leading to better K-means partitions
4. **Faster convergence**: WMF converges in ~50 epochs vs BPR's ~200 epochs
5. **Following original paper**: This matches the original RecEraser paper

The pretrain embeddings are **only used for data partitioning** (K-means clustering), not for the final model. Each local model in RecEraser trains from random initialization.

If you want to use BPR-WMF pretrain instead, you can train `BPRMF.py` and modify `utility/data_partition.py` to use those embeddings.

## Requirements

- Python 3.7
- conda (recommended for environment management)

## Installation

```bash
# Create conda environment
conda create -n receraser python=3.7 -y
conda activate receraser

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This implementation uses **MovieLens-1m** (`ml-1m`).
- **Users**: 6,040
- **Items**: 3,706
- **Interactions**: 920,193

## Quick Start

```bash
# Activate environment
conda activate receraser

# Navigate to code directory
cd code/

# Train WMF (creates pretrain embeddings)
python WMF.py --dataset ml-1m --epoch 50 --lr 0.01 --batch_size 256

# Train RecEraser with different partition methods
python RecEraser_BPR.py --dataset ml-1m --part_type 1 --part_num 5 --epoch 30   # InBP
python RecEraser_BPR.py --dataset ml-1m --part_type 2 --part_num 5 --epoch 30   # UBP
python RecEraser_BPR.py --dataset ml-1m --part_type 3 --part_num 5 --epoch 30   # Random
python RecEraser_BPR.py --dataset ml-1m --part_type 4 --part_num 5 --epoch 30   # IBP
```

## Partition Methods

- **InBP** (Interaction-based Balanced Partition): Uses both user+item embeddings
- **UBP** (User-based Partition): Partitions by user embeddings
- **IBP** (Item-based Partition): Partitions by item embeddings
- **Random**: No clustering baseline

## Results (MovieLens-1m, part_num=5, epoch=30)

| Method | Recall@20 | Precision@20 | NDCG@20 |
|--------|-----------|--------------|---------|
| UBP    | 0.2184    | 0.2433       | 0.3111  |
| InBP   | 0.2024    | 0.2295       | 0.2923  |
| Random | 0.1820    | 0.2146       | 0.2747  |

## Code Structure

```
code/
├── RecEraser_BPR.py       # Main RecEraser with BPR
├── RecEraser_LightGCN.py  # RecEraser with LightGCN
├── RecEraser_WMF.py       # RecEraser with WMF
├── LightGCN.py            # Baseline LightGCN
├── BPRMF.py               # Baseline BPR-MF
├── WMF.py                 # Baseline WMF
├── setup.py               # C++ evaluator compilation
├── utility/
│   ├── data_partition.py  # 4 partition methods (InBP, UBP, IBP, Random)
│   ├── load_data.py       # Data loading
│   ├── batch_test.py      # Batch evaluation
│   ├── helper.py          # Helper functions
│   └── parser.py          # Command line arguments
└── evaluator/
    ├── cpp/               # C++ evaluator (faster)
    └── python/            # Python evaluator (fallback)
```
