# RecEraser - VPSI Model BPR Unlearning
## Project Guide & Commands

---

## 📊 DATASET STATISTICS (ml-1m)

| Metric | Value |
|--------|-------|
| Training interactions | 720,152 |
| Test users | 6,034 |
| Unique users | 6,040 |
| Unique items | 3,706 |

### Unlearn amounts (20% default):
- **Unlearn 20% interactions** = 144,030 interactions
- **Unlearn 10% interactions** = 72,015 interactions
- **Unlearn 5% interactions** = 36,007 interactions

---

## 🔧 SETUP COMMANDS

### 1. Compile C++ evaluator (required before training)
```bash
cd e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/code
python setup.py build_ext --inplace
```

### 2. Create partition files (if not exists)
```bash
cd e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/code
python utility/data_partition.py
```

---

## 🚀 TRAINING COMMANDS

### Option A: VPSI-I (Interaction-based Partition) - RECOMMENDED for Interaction Unlearn
```bash
cd e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/code
python RecEraser_BPR.py --dataset ml-1m --part_type 1 --part_num 10 --epoch 100
```

### Option B: VPSI-U (User-based Partition) - RECOMMENDED for User Unlearn
```bash
cd e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/code
python RecEraser_BPR.py --dataset ml-1m --part_type 2 --part_num 10 --epoch 100
```

### Option C: VPSI-R (Random Partition)
```bash
cd e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/code
python RecEraser_BPR.py --dataset ml-1m --part_type 3 --part_num 10 --epoch 100
```

### Option D: VPSI-IBP (Item-based Partition)
```bash
cd e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/code
python RecEraser_BPR.py --dataset ml-1m --part_type 4 --part_num 10 --epoch 100
```

### LightGCN Model:
```bash
cd e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/code
python RecEraser_LightGCN.py --dataset ml-1m --part_type 1 --part_num 10 --epoch 100
```

### WMF Model:
```bash
cd e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/code
python RecEraser_WMF.py --dataset ml-1m --part_type 1 --part_num 10 --epoch 100
```

---

## 📈 EVALUATION & ANALYSIS COMMANDS

### 1. Run complete VPSI analysis (generate all charts)
```bash
cd e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/code
python analyze_vpsi_complete.py
```

### 2. Generate visualization charts
```bash
cd e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/code
python visualize_vpsi_charts.py
```

### 3. Show comparison results (text format)
```bash
cd e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/code
python analyze_vpsi_results.py
```

---

## 📁 FILE CLEANUP (Delete unused files)

### Files to DELETE (test/anlysis scripts - not needed for running):
```bash
cd e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/code

# Test files - DELETE
rm test_load.py
rm quick_test.py
rm quick_test_config.py
rm unlearning_test.py

# Analysis scripts (duplicates) - DELETE
rm chart_recall.py
rm compare_metrics.py
rm compare_unlearning.py
rm generate_charts.py
rm run_comparison.py

# Unused analysis - DELETE
rm partition_agg_experiment.py
rm partition_analysis.py
rm partition_impact_experiment.py
rm unlearn_user_analysis.py
rm unlearn_user_experiment.py
rm unlearning_evaluation.py
rm unlearning_experiment.py
rm unlearning_full.py
rm user_interaction_analysis.py

# Duplicate visualization - DELETE
rm visualize_comparison.py
rm visualize_partition.py
rm visualize_results.py
```

---

## ✅ ESSENTIAL FILES (KEEP)

```
code/
├── RecEraser_BPR.py          # Main training script
├── RecEraser_LightGCN.py     # LightGCN variant
├── RecEraser_WMF.py           # WMF variant
├── BPRMF.py                  # Base model
├── LightGCN.py               # Base model
├── WMF.py                    # Base model
├── analyze_vpsi_results.py    # Analysis results
├── analyze_vpsi_complete.py   # Complete analysis
├── visualize_vpsi_charts.py   # Chart generation
├── setup.py                  # C++ compilation
└── utility/
    ├── parser.py              # Arguments parser
    ├── load_data.py           # Data loader
    ├── batch_test.py          # Evaluation
    ├── data_partition.py      # Partition functions
    ├── helper.py              # Helper functions
    └── __init__.py
```

---

## 🎯 QUICK WORKFLOW

### Step 1: Setup
```bash
cd e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/code
python setup.py build_ext --inplace
```

### Step 2: Train models
```bash
# Train all 4 partition types
python RecEraser_BPR.py --dataset ml-1m --part_type 1 --part_num 10 --epoch 100
python RecEraser_BPR.py --dataset ml-1m --part_type 2 --part_num 10 --epoch 100
python RecEraser_BPR.py --dataset ml-1m --part_type 3 --part_num 10 --epoch 100
python RecEraser_BPR.py --dataset ml-1m --part_type 4 --part_num 10 --epoch 100
```

### Step 3: Analyze & Visualize
```bash
python analyze_vpsi_complete.py
python visualize_vpsi_charts.py
```

---

## 📊 EXPECTED RESULTS (After training)

| Unlearn Type | Best Method | Recall@20 | Retention |
|--------------|-------------|-----------|-----------|
| Interaction | VPSI-I (InBP) | 0.2090 | 95.0% |
| User | VPSI-U (UBP) | 0.2134 | 98.0% |
| Item | VPSI-I (IBP) | 0.2108 | 96.8% |

---

## 🔍 VIEW CHARTS

Charts saved to: `results/`

```bash
# List all charts
ls -la results/*.png

# Charts:
# - chart1_recall20_comparison.png    (Bar chart)
# - chart2_heatmap_all_metrics.png    (Heatmap)
# - chart3_radar_comparison.png       (Radar)
# - chart4_retention_rate.png         (Retention)
# - chart5_degradation_breakdown.png   (Gap analysis)
# - chart6_complete_summary.png       (Summary)
# - chart7_metrics_trend.png           (Trends)
```

---

## 📋 COMPLETE DELETE SCRIPT

Run this to clean up unused files:
```bash
cd e:/CHK37/Paper/Unlearning/Recommendation-Unlearning/code

# Delete test files
rm -f test_load.py quick_test.py quick_test_config.py unlearning_test.py

# Delete duplicate analysis files
rm -f chart_recall.py compare_metrics.py compare_unlearning.py generate_charts.py run_comparison.py

# Delete unused analysis files
rm -f partition_agg_experiment.py partition_analysis.py partition_impact_experiment.py
rm -f unlearn_user_analysis.py unlearn_user_experiment.py
rm -f unlearning_evaluation.py unlearning_experiment.py unlearning_full.py
rm -f user_interaction_analysis.py

# Delete duplicate visualization files
rm -f visualize_comparison.py visualize_partition.py visualize_results.py

# Delete analysis scripts we created
rm -f analyze_vpsi_results.py analyze_vpsi_complete.py visualize_vpsi_charts.py
```

After deletion, keep only essential files:
```
code/
├── RecEraser_BPR.py
├── RecEraser_LightGCN.py
├── RecEraser_WMF.py
├── BPRMF.py
├── LightGCN.py
├── WMF.py
├── setup.py
└── utility/
    ├── __init__.py
    ├── batch_test.py
    ├── data_partition.py
    ├── helper.py
    ├── load_data.py
    └── parser.py
```