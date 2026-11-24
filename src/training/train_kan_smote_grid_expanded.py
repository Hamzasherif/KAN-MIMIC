#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import product
import json
import random
import statistics
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from kan import KAN
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    ROOT
    / "data/catboost_saved_data/training/mimic_cohort_NF_30_days/concatenate/minority/"
    / "feature_selection_True/feat_type_VMD/agg_interval_24h"
)
OUTPUT_PATH = ROOT / "experiments/kan_smote_grid_expanded"

# Search one oversampling method (SMOTE) at several ratios and neighbor counts.
SMOTE_RATIOS = (0.25, 0.5, 0.75, 1.0)
SMOTE_NEIGHBORS = (3, 5, 7)
CANDIDATES_PER_SMOTE_SETTING = 20

# Broad KAN search space. A balanced 240-candidate sample is drawn from this
# space so every SMOTE setting is paired with twenty KAN configurations.
KAN_GRID_VALUES = (3, 5, 7, 10)
SPLINE_ORDERS = (2, 3, 4)
HIDDEN_WIDTHS = (8, 16, 32)
HIDDEN_LAYERS = (1, 2)
LEARNING_RATES = (3e-5, 1e-4, 3e-4, 1e-3)
WEIGHT_DECAYS = (0.0, 1e-5, 1e-4, 1e-3)
BATCH_SIZES = (64, 128, 256)
METRIC_NAMES = (
    "ROC_AUC",
    "PR_AUC",
    "Accuracy",
    "F1",
    "Precision",
    "Recall",
    "Specificity",
    "Prevalence",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_frame(value, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.reset_index(drop=True).copy()
    if isinstance(value, pd.Series):
        return value.reset_index(drop=True).to_frame(name=name)
    return pd.DataFrame(value)


def as_vector(value, name: str) -> np.ndarray:
    if isinstance(value, pd.DataFrame):
        if value.shape[1] > 1:
            matching = [c for c in value.columns if str(c).lower() in {name.lower(), "label", "subject_id"}]
            if len(matching) == 1:
                value = value[matching[0]]
            else:
                raise ValueError(f"{name} should contain one column; found {list(value.columns)}")
        else:
            value = value.iloc[:, 0]
    if isinstance(value, pd.Series):
        value = value.to_numpy()
    return np.asarray(value).reshape(-1)


def load_fold(fold: int):
    def load(name: str):
        path = DATA_PATH / f"{name}__fold_{fold}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Required prepared fold file was not found: {path}")
        return pd.read_pickle(path)

    x_train = as_frame(load("X_train"), "feature")
    x_val = as_frame(load("X_val"), "feature")
    x_test = as_frame(load("X_test"), "feature")
    for split, x in (("validation", x_val), ("test", x_test)):
        if list(x.columns) != list(x_train.columns):
            raise ValueError(f"{split} feature columns do not match training columns in fold {fold}")

    y_train = as_vector(load("Y_train"), "label").astype(np.int64)
    y_val = as_vector(load("Y_val"), "label").astype(np.int64)
    y_test = as_vector(load("Y_test"), "label").astype(np.int64)
    groups_train = as_vector(load("Sub_train"), "subject_id")

    if not (len(x_train) == len(y_train) == len(groups_train)):
        raise ValueError(
            f"Fold {fold}: train rows/labels/patient IDs differ: "
            f"{len(x_train)}/{len(y_train)}/{len(groups_train)}"
        )
    if len(x_val) != len(y_val) or len(x_test) != len(y_test):
        raise ValueError(f"Fold {fold}: validation or test row/label counts do not match")
    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError(f"Fold {fold}: training labels must include both classes")
    if pd.isna(groups_train).any():
        raise ValueError(f"Fold {fold}: missing patient IDs in Sub_train")

    arrays = (x_train.to_numpy(), x_val.to_numpy(), x_test.to_numpy())
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError(f"Fold {fold}: features contain missing or non-finite values")
    return x_train, y_train, groups_train, x_val, y_val, x_test, y_test


def build_param_grid(seed: int) -> list[dict]:
    """Build a reproducible 240-candidate, balanced sample of the broad search space."""
    model_configs = [
        {
            "grid": grid,
            "spline_order": spline_order,
            "hidden_width": hidden_width,
            "hidden_layers": hidden_layers,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        }
        for (
            grid,
            spline_order,
            hidden_width,
            hidden_layers,
            learning_rate,
            weight_decay,
            batch_size,
        ) in product(
            KAN_GRID_VALUES,
            SPLINE_ORDERS,
            HIDDEN_WIDTHS,
            HIDDEN_LAYERS,
            LEARNING_RATES,
            WEIGHT_DECAYS,
            BATCH_SIZES,
        )
    ]
    rng = np.random.default_rng(seed)
    available = np.arange(len(model_configs))
    rng.shuffle(available)
    cursor = 0
    candidates = []
    candidate_number = 0
    for ratio, neighbors in product(SMOTE_RATIOS, SMOTE_NEIGHBORS):
        for _ in range(CANDIDATES_PER_SMOTE_SETTING):
            config_index = int(available[cursor])
            cursor += 1
            candidate_number += 1
            candidates.append(
                {
                    "candidate": candidate_number,
                    "sampling_strategy": float(ratio),
                    "smote_k_neighbors": int(neighbors),
                    **model_configs[config_index],
                }
            )
    return candidates


def smote_train(
    x: np.ndarray,
    y: np.ndarray,
    sampling_strategy: float,
    k_neighbors: int,
    seed: int,
):
    """Generate numeric SMOTE rows; call only on the current training partition."""
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if len(x) != len(y):
        raise ValueError("SMOTE features and labels have misaligned rows")
    if not 0.0 < sampling_strategy <= 1.0:
        raise ValueError("SMOTE sampling_strategy must be in (0, 1] for binary labels")
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) != 2:
        raise ValueError("SMOTE requires both binary classes")
    minority_label = labels[int(np.argmin(counts))]
    majority_count = int(np.max(counts))
    minority_x = np.asarray(x[y == minority_label], dtype=np.float32)
    target_minority_count = int(majority_count * sampling_strategy)
    rows_to_add = target_minority_count - len(minority_x)
    if rows_to_add <= 0:
        return x.astype(np.float32, copy=False), y.astype(np.float32)
    if len(minority_x) <= k_neighbors:
        raise ValueError(
            f"SMOTE k_neighbors={k_neighbors} needs at least {k_neighbors + 1} "
            f"minority rows in this training partition; found {len(minority_x)}"
        )

    neighbor_model = NearestNeighbors(n_neighbors=k_neighbors + 1)
    neighbor_model.fit(minority_x)
    neighbor_indices = neighbor_model.kneighbors(
        minority_x, return_distance=False
    )
    neighbor_choices = []
    for row_index, indices in enumerate(neighbor_indices):
        choices = indices[indices != row_index]
        if len(choices) < k_neighbors:
            raise RuntimeError("Could not find enough distinct SMOTE neighbors")
        neighbor_choices.append(choices[:k_neighbors])

    rng = np.random.default_rng(seed)
    base_indices = rng.integers(0, len(minority_x), size=rows_to_add)
    synthetic_neighbors = np.fromiter(
        (
            rng.choice(neighbor_choices[base_index])
            for base_index in base_indices
        ),
        dtype=np.int64,
        count=rows_to_add,
    )
    interpolation = rng.random((rows_to_add, 1), dtype=np.float32)
    synthetic_x = minority_x[base_indices] + interpolation * (
        minority_x[synthetic_neighbors] - minority_x[base_indices]
    )
    synthetic_y = np.full(rows_to_add, minority_label, dtype=np.float32)
    return (
        np.concatenate([x.astype(np.float32, copy=False), synthetic_x], axis=0),
        np.concatenate([y.astype(np.float32), synthetic_y], axis=0),
    )


def make_model(input_dim: int, params: dict, seed: int) -> KAN:
    # Keep the established CPU spline-grid adaptation path while tuning width/depth.
    width = [input_dim] + [int(params["hidden_width"])] * int(params["hidden_layers"]) + [1]
    return KAN(
        width=width,
        grid=int(params["grid"]),
        k=int(params["spline_order"]),
        seed=seed,
        device="cpu",
        save_act=False,
        auto_save=False,
        symbolic_enabled=False,
    )


def predict_probabilities(model: KAN, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            batch = torch.as_tensor(x[start : start + 512], dtype=torch.float32, device=device)
            logits = model(batch).reshape(-1)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(probabilities, dtype=np.float64)


def train_with_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: str,
    verbose: bool = False,
):
    set_seed(seed)
    train_x = torch.as_tensor(x_train, dtype=torch.float32)
    train_y = torch.as_tensor(y_train, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=int(params.get("batch_size", batch_size)),
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )
    val_x_tensor = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    val_y_tensor = torch.as_tensor(y_val, dtype=torch.float32, device=device)

    model = make_model(x_train.shape[1], params, seed)
    sample_size = min(256, len(x_train))
    sample_indices = np.random.choice(len(x_train), size=sample_size, replace=False)
    grid_samples = torch.as_tensor(x_train[sample_indices], dtype=torch.float32, device="cpu")
    model.update_grid(grid_samples)
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x).reshape(-1)
            loss = criterion(logits, batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_y)
            seen += len(batch_y)

        model.eval()
        with torch.no_grad():
            val_logits = []
            for start in range(0, len(val_x_tensor), 512):
                val_logits.append(model(val_x_tensor[start : start + 512]).reshape(-1))
            logits = torch.cat(val_logits)
            val_loss = float(criterion(logits, val_y_tensor).item())
            val_prob = torch.sigmoid(logits).cpu().numpy()
        val_auc = float(roc_auc_score(y_val, val_prob))
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(seen, 1),
                "val_loss": val_loss,
                "val_roc_auc": val_auc,
            }
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(
                f"Epoch {epoch:03d}/{epochs} | train_loss={running_loss / max(seen, 1):.5f} "
                f"val_loss={val_loss:.5f} val_ROC_AUC={val_auc:.5f}"
            )
        if stale_epochs >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return model, float(best_auc), int(best_epoch), history, best_state


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "ROC_AUC": float(roc_auc_score(y_true, probabilities)),
        "PR_AUC": float(average_precision_score(y_true, probabilities)),
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "Prevalence": float(np.mean(y_true)),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Threshold": 0.5,
    }


def run_fold(args) -> None:
    fold = args.fold
    fold_output = OUTPUT_PATH / f"fold_{fold}"
    if fold_output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment output: {fold_output}"
        )

    x_train_df, y_train, groups_train, x_val_df, y_val, x_test_df, y_test = load_fold(fold)
    x_train = x_train_df.to_numpy(dtype=np.float32)
    x_val = x_val_df.to_numpy(dtype=np.float32)
    x_test = x_test_df.to_numpy(dtype=np.float32)
    unique_groups = len(pd.unique(groups_train))
    print(
        f"Fold {fold}: train rows={len(y_train)}, patients={unique_groups}, "
        f"positive={int(y_train.sum())}, negative={int((y_train == 0).sum())}, "
        f"validation rows={len(y_val)}, test rows={len(y_test)}"
    )

    splitter = StratifiedGroupKFold(
        n_splits=args.inner_splits,
        shuffle=True,
        random_state=args.seed,
    )
    param_grid = build_param_grid(args.seed)
    print(
        f"Fold {fold}: running {len(param_grid)} candidates "
        f"({len(SMOTE_RATIOS) * len(SMOTE_NEIGHBORS)} SMOTE settings, "
        f"{CANDIDATES_PER_SMOTE_SETTING} KAN configurations per setting)"
    )
    candidate_rows = []
    for candidate_index, params in enumerate(param_grid):
        fold_scores = []
        fold_epochs = []
        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
            splitter.split(x_train, y_train, groups=groups_train)
        ):
            x_inner_train_base = x_train[inner_train_idx]
            y_inner_train = y_train[inner_train_idx]
            x_inner_val = x_train[inner_val_idx]
            y_inner_val = y_train[inner_val_idx]
            if len(np.unique(y_inner_val)) != 2:
                raise ValueError(
                    f"Fold {fold}, inner fold {inner_fold}: validation split lacks a class"
                )

            # Fit scaling on original inner-train patients only, then apply SMOTE
            # to that training partition. Inner validation remains untouched.
            inner_scaler = StandardScaler().fit(x_inner_train_base)
            x_inner_train_scaled = inner_scaler.transform(x_inner_train_base).astype(np.float32)
            x_inner_val_scaled = inner_scaler.transform(x_inner_val).astype(np.float32)
            x_inner_smote, y_inner_smote = smote_train(
                x_inner_train_scaled,
                y_inner_train,
                sampling_strategy=params["sampling_strategy"],
                k_neighbors=params["smote_k_neighbors"],
                seed=args.seed + fold * 1000 + candidate_index * 10 + inner_fold,
            )
            _, best_auc, best_epoch, _, _ = train_with_validation(
                x_inner_smote,
                y_inner_smote,
                x_inner_val_scaled,
                y_inner_val.astype(np.float32),
                params=params,
                epochs=args.cv_epochs,
                patience=args.cv_patience,
                batch_size=int(params["batch_size"]),
                seed=args.seed + fold * 1000 + candidate_index * 10 + inner_fold,
                device=args.device,
            )
            fold_scores.append(best_auc)
            fold_epochs.append(best_epoch)
            print(
                f"Fold {fold} | candidate {candidate_index + 1}/{len(param_grid)} "
                f"inner {inner_fold + 1}/{args.inner_splits}: "
                f"SMOTE ratio={params['sampling_strategy']:.2f} "
                f"k={params['smote_k_neighbors']} grid={params['grid']} "
                f"spline={params['spline_order']} width={params['hidden_width']} "
                f"layers={params['hidden_layers']} lr={params['learning_rate']:.5g} "
                f"best_val_ROC_AUC={best_auc:.5f} epoch={best_epoch}"
            )

        row = {
            **params,
            "mean_inner_ROC_AUC": float(np.mean(fold_scores)),
            "std_inner_ROC_AUC": float(np.std(fold_scores, ddof=0)),
            "mean_best_epoch": float(np.mean(fold_epochs)),
        }
        row.update({f"inner_fold_{i}_ROC_AUC": score for i, score in enumerate(fold_scores)})
        candidate_rows.append(row)

    best_row = max(candidate_rows, key=lambda row: row["mean_inner_ROC_AUC"])
    best_params = {
        "grid": int(best_row["grid"]),
        "spline_order": int(best_row["spline_order"]),
        "hidden_width": int(best_row["hidden_width"]),
        "hidden_layers": int(best_row["hidden_layers"]),
        "learning_rate": float(best_row["learning_rate"]),
        "weight_decay": float(best_row["weight_decay"]),
        "batch_size": int(best_row["batch_size"]),
        "sampling_strategy": float(best_row["sampling_strategy"]),
        "smote_k_neighbors": int(best_row["smote_k_neighbors"]),
        "selection_metric": (
            f"mean {args.inner_splits}-fold patient-grouped validation ROC-AUC"
        ),
        "oversampling": "SMOTE applied to inner training rows only",
        "search_candidates": int(len(param_grid)),
    }

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    fold_output.mkdir(parents=False, exist_ok=False)
    pd.DataFrame(candidate_rows).to_csv(fold_output / "grid_search_results.csv", index=False)
    (fold_output / "best_params.json").write_text(json.dumps(best_params, indent=2))

    # Refit the chosen configuration on this outer fold's train split, then select
    # the checkpoint using the untouched outer validation split. Test is evaluated once.
    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train).astype(np.float32)
    x_val_scaled = scaler.transform(x_val).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)
    x_train_smote, y_train_smote = smote_train(
        x_train_scaled,
        y_train,
        sampling_strategy=best_params["sampling_strategy"],
        k_neighbors=best_params["smote_k_neighbors"],
        seed=args.seed + fold,
    )
    print(
        f"Fold {fold}: after SMOTE, train rows={len(y_train_smote)}, "
        f"positives={int(y_train_smote.sum())}, "
        f"negatives={int((y_train_smote == 0).sum())}; validation/test unchanged"
    )

    model, best_val_auc, best_epoch, history, best_state = train_with_validation(
        x_train_smote,
        y_train_smote,
        x_val_scaled,
        y_val.astype(np.float32),
        params=best_params,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=best_params["batch_size"],
        seed=args.seed + fold,
        device=args.device,
        verbose=True,
    )
    torch.save(best_state, fold_output / "best_model.pt")
    joblib.dump(scaler, fold_output / "scaler.joblib")
    pd.DataFrame(history).to_csv(fold_output / "training_history.csv", index=False)

    val_prob = predict_probabilities(model, x_val_scaled, args.device)
    test_prob = predict_probabilities(model, x_test_scaled, args.device)
    val_metrics = calculate_metrics(y_val, val_prob)
    test_metrics = calculate_metrics(y_test, test_prob)
    val_metrics["Best_Validation_ROC_AUC"] = best_val_auc
    val_metrics["Best_Epoch"] = best_epoch
    test_metrics["Best_Epoch"] = best_epoch
    test_metrics["Best_Validation_ROC_AUC"] = best_val_auc

    (fold_output / "validation_metrics.json").write_text(json.dumps(val_metrics, indent=2))
    (fold_output / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    pd.DataFrame({"Labels": y_val, "Prob": val_prob}).to_csv(
        fold_output / "validation_predictions.csv", index=False
    )
    pd.DataFrame({"Labels": y_test, "Prob": test_prob}).to_csv(
        fold_output / "test_predictions.csv", index=False
    )
    metadata = {
        "fold": fold,
        "train_rows_before_smote": int(len(y_train)),
        "train_positives_before_smote": int(y_train.sum()),
        "train_negatives_before_smote": int((y_train == 0).sum()),
        "train_rows_after_smote": int(len(y_train_smote)),
        "validation_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "train_unique_patients": int(unique_groups),
        "inner_cv_splits": int(args.inner_splits),
        "search_candidates": int(len(param_grid)),
        "smote_sampling_strategy": best_params["sampling_strategy"],
        "smote_k_neighbors": best_params["smote_k_neighbors"],
        "selected_kan_params": best_params,
        "device": args.device,
        "threshold": 0.5,
    }
    (fold_output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Fold {fold} test metrics at threshold 0.50: {json.dumps(test_metrics)}")
    print(f"Saved outputs to: {fold_output}")


def summarize_folds() -> None:
    rows = []
    for fold in range(5):
        path = OUTPUT_PATH / f"fold_{fold}" / "test_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Fold {fold} results are not available yet: {path}")
        rows.append(json.loads(path.read_text()))

    summary = {
        metric: {
            "mean": float(statistics.mean(row[metric] for row in rows)),
            "population_sd": float(statistics.pstdev(row[metric] for row in rows)),
        }
        for metric in METRIC_NAMES
    }
    (OUTPUT_PATH / "five_fold_test_summary.json").write_text(json.dumps(summary, indent=2))
    print("Five-fold test metrics, mean ± population SD (threshold 0.50):")
    for metric, values in summary.items():
        print(f"{metric}: {values['mean']:.6f} ± {values['population_sd']:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Expanded patient-grouped SMOTE grid search for the NF KAN experiment."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fold", type=int, choices=range(5))
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--cv-epochs", type=int, default=30)
    parser.add_argument("--cv-patience", type=int, default=7)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
    )
    args = parser.parse_args()
    if args.inner_splits < 2:
        parser.error("--inner-splits must be at least 2")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    torch.set_num_threads(3)
    return args


def main():
    args = parse_args()
    if args.summarize:
        summarize_folds()
    else:
        run_fold(args)


if __name__ == "__main__":
    main()
