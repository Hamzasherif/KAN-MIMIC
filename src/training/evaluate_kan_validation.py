import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
import joblib

from kan import KAN
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


DATA_PATH = (
    "data/catboost_saved_data/training/"
    "mimic_cohort_NF_30_days/concatenate/minority/"
    "feature_selection_True/feat_type_VMD/agg_interval_24h"
)

OUTPUT_PATH = "experiments/kan"


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


def main(fold):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    fold_output = os.path.join(OUTPUT_PATH, f"fold_{fold}")

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    # Load validation data.
    X_val = pd.read_pickle(
        os.path.join(DATA_PATH, f"X_val__fold_{fold}.pkl")
    )
    y_val = pd.read_pickle(
        os.path.join(DATA_PATH, f"Y_val__fold_{fold}.pkl")
    ).to_numpy(dtype=np.float32)

    # Load the scaler fitted on the training data.
    scaler = joblib.load(
        os.path.join(fold_output, "scaler.joblib")
    )

    assert list(X_val.columns) == list(scaler.feature_names_in_)
    assert X_val.shape[1] == scaler.n_features_in_

    X_val = scaler.transform(X_val).astype(np.float32)

    X_val = torch.tensor(
        X_val, dtype=torch.float32, device=device
    )
    y_val = torch.tensor(
        y_val, dtype=torch.float32, device=device
    )

    print("Validation shape:", tuple(X_val.shape))
    print("Positive cases:", int(y_val.sum().item()))

    # Reconstruct the saved KAN architecture.
    model = KAN(
        width=[X_val.shape[1], 8, 1],
        grid=3,
        k=3,
        seed=42,
        device=str(device),
        save_act=False,
        auto_save=False,
        symbolic_enabled=False,
    )

    checkpoint_path = os.path.join(
        fold_output, "best_model.pt"
    )

    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state)
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        logits = model(X_val).reshape(-1)
        val_loss = criterion(logits, y_val).item()
        probabilities = torch.sigmoid(logits).cpu().numpy()

    y_true = y_val.cpu().numpy()

    metrics = calculate_metrics(y_true, probabilities)
    metrics["Validation_Loss"] = float(val_loss)

    # Save new validation files only.
    pd.DataFrame({
        "Labels": y_true,
        "Prob": probabilities,
        "Logits": logits.cpu().numpy(),
    }).to_csv(
        os.path.join(fold_output, "validation_predictions.csv"),
        index=False,
    )

    with open(
        os.path.join(fold_output, "validation_metrics.json"), "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    print("\nFold", fold, "validation results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    print("\nSaved validation evaluation results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    if args.fold not in [1, 2, 3, 4]:
        raise ValueError("This script is intended for Folds 1–4 only.")

    main(args.fold)
