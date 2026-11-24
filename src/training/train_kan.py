import os
import json
import random
import argparse
import copy

import numpy as np
import pandas as pd
import torch

from kan import KAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from torch.utils.data import TensorDataset, DataLoader
import joblib


DATA_PATH = (
    "data/catboost_saved_data/training/"
    "mimic_cohort_NF_30_days/concatenate/minority/"
    "feature_selection_True/feat_type_VMD/agg_interval_24h"
)

OUTPUT_PATH = "experiments/kan"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "ROC_AUC": float(roc_auc_score(y_true, probabilities)),
        "PR_AUC": float(average_precision_score(y_true, probabilities)),
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "Precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "Prevalence": float(np.mean(y_true)),
    }


def evaluate(model, X, y, criterion):
    model.eval()

    with torch.no_grad():
        logits = model(X).reshape(-1)
        loss = criterion(logits, y.reshape(-1))
        probabilities = torch.sigmoid(logits).cpu().numpy()

    return float(loss.item()), probabilities


def train_fold(fold, epochs, patience, batch_size, learning_rate, seed, skip_test=False):

    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fold_output = os.path.join(OUTPUT_PATH, f"fold_{fold}")
    os.makedirs(fold_output, exist_ok=True)

    print(f"\nLoading Fold {fold}...")

    def load(name):
        file_path = os.path.join(
            DATA_PATH, f"{name}__fold_{fold}.pkl"
        )
        return pd.read_pickle(file_path)

    X_train = load("X_train")
    X_val = load("X_val")

    y_train = load("Y_train").to_numpy(dtype=np.float32)
    y_val = load("Y_val").to_numpy(dtype=np.float32)

    if not skip_test:
        X_test = load("X_test")
        y_test = load("Y_test").to_numpy(dtype=np.float32)

    assert list(X_train.columns) == list(X_val.columns)
    if not skip_test:
        assert list(X_train.columns) == list(X_test.columns)

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)

    if not skip_test:
      print("Test:", X_test.shape)
    else:
      print("Test: NOT LOADED (development mode)")

    print("Training positives:", int(y_train.sum()))

    # Fit preprocessing on training data only.
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    if not skip_test:
        X_test = scaler.transform(X_test).astype(np.float32)

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)

    if not skip_test:
        X_test = torch.tensor(X_test, dtype=torch.float32, device=device)

    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=device)

    if not skip_test:
        y_test = torch.tensor(y_test, dtype=torch.float32, device=device)



    train_dataset = TensorDataset(X_train, y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    print("\nCreating KAN model on CPU for grid adaptation...")

    model = KAN(
        width=[X_train.shape[1], 8, 1],
        grid=3,
        k=3,
        seed=seed,
        device="cpu",
        save_act=False,
        auto_save=False,
        symbolic_enabled=False,
    )

    # Adapt the spline grid using training samples only.
    # Original PyKAN least-squares solver runs on CPU.
    grid_sample_size = min(256, len(X_train))
    grid_generator = torch.Generator().manual_seed(seed)
    grid_indices = torch.randperm(
        len(X_train), generator=grid_generator
    )[:grid_sample_size]

    print(f"Adapting grid on CPU using {grid_sample_size} training samples...")
    model.update_grid(X_train[grid_indices].cpu())

    # Move the initialized model to GPU for training.
    if device == "cuda":
        print("Moving KAN model to GPU...")
        model.to(device)

    print("Model device after grid adaptation:", model.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    best_auc = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None
    history = []

    print("\nStarting training...")

    for epoch in range(1, epochs + 1):

        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()

            logits = model(batch_X).reshape(-1)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_X)
            total_samples += len(batch_X)

        train_loss = running_loss / total_samples

        val_loss, val_probabilities = evaluate(
            model, X_val, y_val, criterion
        )

        val_auc = roc_auc_score(
            y_val.cpu().numpy(), val_probabilities
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_roc_auc": float(val_auc),
        })

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val ROC-AUC: {val_auc:.5f}"
        )

        if val_auc > best_auc:
            best_auc = float(val_auc)
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())

            torch.save(
                best_state,
                os.path.join(fold_output, "best_model.pt"),
            )

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("No valid model checkpoint was saved.")

    # Restore the best validation checkpoint.
    model.load_state_dict(best_state)

        # Save common training artifacts.
    joblib.dump(
        scaler,
        os.path.join(fold_output, "scaler.joblib"),
    )

    pd.DataFrame(history).to_csv(
        os.path.join(fold_output, "training_history.csv"),
        index=False,
    )

    # Development mode: save validation results without touching test data.
    if skip_test:
        val_loss, val_probabilities = evaluate(
            model, X_val, y_val, criterion
        )

        val_metrics = calculate_metrics(
            y_val.cpu().numpy(), val_probabilities
        )

        val_metrics["Validation_Loss"] = val_loss
        val_metrics["Best_Epoch"] = best_epoch
        val_metrics["Best_Validation_ROC_AUC"] = best_auc

        pd.DataFrame({
            "Labels": y_val.cpu().numpy(),
            "Prob": val_probabilities,
        }).to_csv(
            os.path.join(fold_output, "validation_predictions.csv"),
            index=False,
        )

        with open(
            os.path.join(fold_output, "validation_metrics.json"), "w"
        ) as f:
            json.dump(val_metrics, f, indent=4)

        print("\nValidation results:")
        for key, value in val_metrics.items():
            print(f"{key}: {value:.5f}" if isinstance(value, float) else f"{key}: {value}")

        print("\nDevelopment run complete. Test data was not evaluated.")
        return

    # Final evaluation mode: evaluate test only after model selection.
    test_loss, test_probabilities = evaluate(
        model, X_test, y_test, criterion
    )

    test_metrics = calculate_metrics(
        y_test.cpu().numpy(), test_probabilities
    )

    test_metrics["Test_Loss"] = test_loss
    test_metrics["Best_Epoch"] = best_epoch
    test_metrics["Best_Validation_ROC_AUC"] = best_auc

    joblib.dump(
        scaler,
        os.path.join(fold_output, "scaler.joblib"),
    )

    pd.DataFrame(history).to_csv(
        os.path.join(fold_output, "training_history.csv"),
        index=False,
    )

    pd.DataFrame({
        "Labels": y_test.cpu().numpy(),
        "Prob": test_probabilities,
        "Logits": np.log(
            np.clip(test_probabilities, 1e-7, 1 - 1e-7)
            / np.clip(1 - test_probabilities, 1e-7, 1)
        ),
    }).to_csv(
        os.path.join(fold_output, "test_predictions.csv"),
        index=False,
    )

    with open(
        os.path.join(fold_output, "test_metrics.json"), "w"
    ) as f:
        json.dump(test_metrics, f, indent=4)

    print("\nFold results:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.5f}" if isinstance(value, float) else f"{key}: {value}")

    print(f"\nSaved outputs to: {fold_output}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-test", action="store_true")

    args = parser.parse_args()

    if args.fold not in range(5):
        raise ValueError("Fold must be between 0 and 4.")

    train_fold(
        fold=args.fold,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        skip_test=args.skip_test,
    )
