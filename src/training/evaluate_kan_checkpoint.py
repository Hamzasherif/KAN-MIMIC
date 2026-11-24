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

    # Load test data.
    X_test = pd.read_pickle(
        os.path.join(DATA_PATH, f"X_test__fold_{fold}.pkl")
    )
    y_test = pd.read_pickle(
        os.path.join(DATA_PATH, f"Y_test__fold_{fold}.pkl")
    ).to_numpy(dtype=np.float32)

    # Load the original training scaler.
    scaler = joblib.load(
        os.path.join(fold_output, "scaler.joblib")
    )

    assert list(X_test.columns) == list(scaler.feature_names_in_)
    assert X_test.shape[1] == scaler.n_features_in_

    X_test = scaler.transform(X_test).astype(np.float32)

    X_test = torch.tensor(
        X_test, dtype=torch.float32, device=device
    )
    y_test = torch.tensor(
        y_test, dtype=torch.float32, device=device
    )

    print("Test shape:", tuple(X_test.shape))
    print("Positive cases:", int(y_test.sum().item()))

    # Reconstruct the exact saved KAN architecture.
    model = KAN(
        width=[X_test.shape[1], 8, 1],
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
        logits = model(X_test).reshape(-1)
        test_loss = criterion(logits, y_test).item()
        probabilities = torch.sigmoid(logits).cpu().numpy()

    metrics = calculate_metrics(
        y_test.cpu().numpy(), probabilities
    )

    metrics["Test_Loss"] = float(test_loss)

    # Preserve original files by using new filenames.
    pd.DataFrame({
        "Labels": y_test.cpu().numpy(),
        "Prob": probabilities,
        "Logits": logits.cpu().numpy(),
    }).to_csv(
        os.path.join(fold_output, "test_predictions_hpc.csv"),
        index=False,
    )

    with open(
        os.path.join(fold_output, "test_metrics_hpc.json"), "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    print("\nFold", fold, "test results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    print("\nSaved HPC evaluation results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    if args.fold not in range(5):
        raise ValueError("Fold must be between 0 and 4.")

    main(args.fold)
