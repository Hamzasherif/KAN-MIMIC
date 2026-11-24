# KAN-MIMIC

Research code for comparing Kolmogorov–Arnold Networks (KANs) with a CatBoost baseline on a preprocessed MIMIC-IV clinical prediction task.

## Project scope

This project evaluates models on structured clinical features derived from MIMIC-IV. It includes a standard KAN, a class-weighted KAN, oversampling experiments, an expanded SMOTE grid search, and a CatBoost baseline.

The repository contains code and documentation. It does not include raw MIMIC-IV records or prepared patient-level data.

## Evaluation approach

The evaluation uses five patient-grouped outer folds. Patients are kept separate across each fold’s training, validation, and test splits to reduce data leakage.

For the expanded SMOTE search, hyperparameters are selected using group-aware inner cross-validation on the training split. SMOTE is applied to training data only; validation and test data retain their original class balance. The reported test metrics use a classification threshold of 0.50.

## Current reference results

Metrics are mean ± population standard deviation across the five outer test folds. The fold partitions are patient-grouped. Threshold-dependent metrics use a threshold of 0.50.

### Ranking metrics

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| CatBoost baseline | 0.8156 ± 0.0364 | 0.2485 ± 0.0677 |
| Standard KAN | 0.7492 ± 0.0118 | 0.1838 ± 0.0592 |
| Class-weighted KAN | 0.7417 ± 0.0221 | 0.1714 ± 0.0433 |
| Oversampled KAN grid | 0.7601 ± 0.0135 | 0.1748 ± 0.0491 |
| SMOTE KAN expanded grid | 0.7214 ± 0.0342 | 0.1759 ± 0.0632 |

### Metrics at threshold 0.50

| Model | Accuracy | F1 | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|
| CatBoost baseline | 0.9444 ± 0.0084 | 0.2371 ± 0.0909 | 0.3665 ± 0.0503 | 0.1935 ± 0.1043 | 0.9825 ± 0.0112 |
| Standard KAN | 0.9511 ± 0.0036 | 0.0518 ± 0.0671 | 0.2286 ± 0.2942 | 0.0292 ± 0.0379 | 0.9980 ± 0.0014 |
| Class-weighted KAN | 0.7683 ± 0.1230 | 0.2006 ± 0.0614 | 0.1296 ± 0.0490 | 0.5370 ± 0.1608 | 0.7798 ± 0.1352 |
| Oversampled KAN grid | 0.7997 ± 0.0493 | 0.2103 ± 0.0440 | 0.1351 ± 0.0393 | 0.5300 ± 0.0934 | 0.8135 ± 0.0562 |
| SMOTE KAN expanded grid | 0.9219 ± 0.0130 | 0.2096 ± 0.0723 | 0.2051 ± 0.0546 | 0.2253 ± 0.1002 | 0.9570 ± 0.0155 |

The positive-class prevalence averaged 0.0485 ± 0.0046 across the folds. Because the outcome is imbalanced, interpret accuracy alongside PR-AUC, precision, recall, and F1.

CatBoost fold metrics are available in [`docs/catboost_baseline_metrics.csv`](docs/catboost_baseline_metrics.csv). Per-fold KAN outputs are generated under `experiments/` and are excluded from Git; this section records their five-fold summaries.

## Repository layout

- `configs/` — experiment configuration
- `docs/` — methodology, experiment notes, and baseline metrics
- `src/data/` — data loading and validation
- `src/preprocessing/` — feature preparation
- `src/models/` — KAN and CatBoost model definitions
- `src/training/` — model training and grid-search scripts
- `src/evaluation/` — evaluation utilities
- `src/original_catboost/` — original CatBoost implementation
- `data/` — location for locally prepared data; data files are not committed
- `experiments/` — location for generated results; outputs are not committed

See [`docs/methodology.md`](docs/methodology.md) and [`docs/experiment_log.md`](docs/experiment_log.md) for additional project details.

## Setup

Clone the repository, activate a Python environment compatible with the packages in `requirements.txt`, and install the dependencies:

```bash
git clone https://github.com/Hamzasherif/KAN-MIMIC.git
cd KAN-MIMIC
python -m pip install -r requirements.txt
```

GPU training requires a compatible PyTorch and CUDA setup. The expanded SMOTE script can also run on CPU, though the grid search may take substantially longer.

## Data requirements

Obtain MIMIC-IV access through the appropriate PhysioNet process and prepare the project’s fold files separately. The expanded SMOTE script currently expects them under:

```text
data/catboost_saved_data/training/mimic_cohort_NF_30_days/concatenate/minority/feature_selection_True/feat_type_VMD/agg_interval_24h/
```

For each fold, it expects pickle files named:

```text
X_train__fold_0.pkl
X_val__fold_0.pkl
X_test__fold_0.pkl
Y_train__fold_0.pkl
Y_val__fold_0.pkl
Y_test__fold_0.pkl
Sub_train__fold_0.pkl
```

The same file pattern is required for folds `1` through `4`. `Sub_train` contains the training patient IDs used for grouped inner cross-validation. If your prepared files are stored elsewhere, update `DATA_PATH` in `src/training/train_kan_smote_grid_expanded.py`.

Never commit raw clinical data, patient-level files, or derived patient-level feature matrices.

## Run the expanded SMOTE KAN experiment

Run one outer fold at a time:

```bash
python -u src/training/train_kan_smote_grid_expanded.py --fold 0 --device cuda
python -u src/training/train_kan_smote_grid_expanded.py --fold 1 --device cuda
python -u src/training/train_kan_smote_grid_expanded.py --fold 2 --device cuda
python -u src/training/train_kan_smote_grid_expanded.py --fold 3 --device cuda
python -u src/training/train_kan_smote_grid_expanded.py --fold 4 --device cuda
```

Use `--device cpu` on a machine without a CUDA GPU. After all five folds finish, create the summary:

```bash
python -u src/training/train_kan_smote_grid_expanded.py --summarize
```

The search evaluates 240 sampled KAN configurations across SMOTE ratios `0.25`, `0.50`, `0.75`, and `1.00`, and neighbor counts `3`, `5`, and `7`. It also varies the KAN grid size, spline order, hidden width and depth, learning rate, weight decay, and batch size.

On a Slurm cluster, `train_kan_smote_grid_expanded_fold.sh` is provided as a starting point. Check its environment activation, working directory, and resource requests for your cluster before submitting jobs.

## Other training scripts

- `src/training/train_catboost.py` — CatBoost baseline
- `src/training/train_kan.py` — standard KAN
- `src/training/train_kan_weighted.py` — class-weighted KAN
- `src/training/train_kan_oversampled_grid.py` — oversampling grid search
- `src/training/train_kan_smote_grid_expanded.py` — expanded SMOTE grid search

Training scripts write generated results under `experiments/`. Those outputs are intentionally excluded from Git.
