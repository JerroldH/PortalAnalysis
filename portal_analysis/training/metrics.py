"""Evaluation metrics and reporting."""

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)


def print_results(y_test, y_pred, test_ids=None) -> dict:
    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    accuracy = float(np.mean(y_pred == y_test))
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))

    print("\nOverall Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  MAE:      {mae:.4f}")
    print(f"  MSE:      {mse:.4f}")

    diff = np.abs(y_pred - y_test)
    large_errors = np.where(diff >= 2)[0]
    if len(large_errors) > 0:
        print(f"\nLarge Mismatches (≥2 levels): {len(large_errors)}")
        if test_ids is not None and len(test_ids) > 0:
            print("Sample IDs with large errors:")
            for idx in large_errors[:10]:
                if idx < len(test_ids):
                    print(
                        f"  ID: {test_ids[idx]}, "
                        f"Predicted: {y_pred[idx]}, True: {y_test[idx]}"
                    )

    print("=" * 70 + "\n")
    return {"accuracy": accuracy, "mae": mae, "mse": mse}
