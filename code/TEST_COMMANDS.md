# Quick Test Commands - Run these in your terminal

## STEP 1: Activate Environment & Install matplotlib
```powershell
conda activate receraser
pip install matplotlib
```

## STEP 2: Quick Test (1 epoch per local model)
```powershell
cd e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/code
python quick_test.py --dataset ml-1m --part_type 1 --part_num 10
```

## STEP 3: Train Full Model (50 epochs) - Choose ONE

### Option A: InBP (Recommended)
```powershell
python RecEraser_BPR.py --dataset ml-1m --part_type 1 --part_num 10 --epoch 50 --verbose 5
```

### Option B: UBP
```powershell
python RecEraser_BPR.py --dataset ml-1m --part_type 2 --part_num 10 --epoch 50 --verbose 5
```

### Option C: Random
```powershell
python RecEraser_BPR.py --dataset ml-1m --part_type 3 --part_num 10 --epoch 50 --verbose 5
```

## STEP 4: Visualize Results
```powershell
python visualize_results.py --dataset ml-1m --part_num 10
```

## STEP 5: Run All (Train all 3 + Visualize)
```powershell
# Run all in sequence
python RecEraser_BPR.py --dataset ml-1m --part_type 1 --epoch 50 --verbose 5
python RecEraser_BPR.py --dataset ml-1m --part_type 2 --epoch 50 --verbose 5
python RecEraser_BPR.py --dataset ml-1m --part_type 3 --epoch 50 --verbose 5
python visualize_results.py --dataset ml-1m --part_num 10
```

---

## Expected Output Example:
```
[local_model 0] Epoch 0: train==[0.69315=0.69315 + 0.00000]
[local_model 0] Epoch 5: recall=[0.0025, 0.0052], precision=[...], ndcg=[...]
[local_model 0] Epoch 10: recall=[0.0050, 0.0100], precision=[...], ndcg=[...]
...
[AGG] Epoch 0: train==[...]
[AGG] Epoch 5: recall=[0.0100, 0.0200], precision=[...], ndcg=[...]
```

## Check Results:
- After training, weights saved to: `../weights/ml-1m/RecEraser_BPR/num-10_type-X_r0.01/`
- Visualization saved to: `../results/partition_comparison.png`