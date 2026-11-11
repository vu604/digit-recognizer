"""Train a NumPy digit classifier on scikit-learn's handwritten digits dataset."""
import csv
import gzip
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from preprocessing import preprocess


BASE = Path(__file__).resolve().parent


def load_handwritten_digits():
    """Load the bundled 8x8 handwritten digits without importing scikit-learn."""
    sklearn_spec = importlib.util.find_spec("sklearn")
    if sklearn_spec is None or sklearn_spec.origin is None:
        raise RuntimeError("scikit-learn is required to provide the handwritten digits dataset")

    dataset_path = Path(sklearn_spec.origin).parent / "datasets" / "data" / "digits.csv.gz"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Handwritten digits dataset not found: {dataset_path}")

    with gzip.open(dataset_path, "rt", newline="") as dataset_file:
        rows = np.asarray([[int(value) for value in row] for row in csv.reader(dataset_file)])
    if rows.ndim != 2 or rows.shape[1] != 65:
        raise ValueError(f"Expected 65 columns in handwritten digits dataset, got {rows.shape}")

    # The dataset stores the 64 pixels first and the digit label in the final column.
    labels = rows[:, -1]
    if not np.all((0 <= labels) & (labels <= 9)):
        raise ValueError("Handwritten digits dataset contains an invalid label")
    images = rows[:, :-1].reshape(-1, 8, 8)
    features = np.vstack([
        preprocess((image.astype(float) * (255.0 / 16.0)).tolist())
        for image in images
    ])
    return features, labels


def split_by_digit(labels, test_fraction=0.2, seed=42):
    """Create a deterministic holdout split with examples from every digit."""
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []
    for digit in range(10):
        indices = np.flatnonzero(labels == digit)
        if indices.size < 2:
            raise ValueError(f"Not enough samples for digit {digit}")
        rng.shuffle(indices)
        split_at = int(indices.size * (1 - test_fraction))
        train_indices.extend(indices[:split_at])
        test_indices.extend(indices[split_at:])
    return np.asarray(train_indices), np.asarray(test_indices)


def main():
    print("Loading handwritten digits...")
    X, y = load_handwritten_digits()
    train_indices, test_indices = split_by_digit(y)

    X_train = X[train_indices].reshape(-1, 8, 8)
    y_train = y[train_indices]
    translations = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]
    augmented_images = [X_train]
    for dy, dx in translations:
        shifted = np.zeros_like(X_train)
        y_start, y_end = max(0, dy), X_train.shape[1] + min(0, dy)
        x_start, x_end = max(0, dx), X_train.shape[2] + min(0, dx)
        shifted[:, y_start:y_end, x_start:x_end] = X_train[
            :, y_start - dy:y_end - dy, x_start - dx:x_end - dx
        ]
        augmented_images.append(shifted)

    X_augmented = np.concatenate(augmented_images).reshape(-1, 64)
    y_augmented = np.tile(y_train, len(augmented_images))
    model = CalibratedClassifierCV(
        SVC(C=10, gamma="scale"),
        method="sigmoid",
        cv=3,
        ensemble=False,
    )
    print("Training shift-augmented RBF support vector classifier...")
    model.fit(X_augmented, y_augmented)

    train_accuracy = float(np.mean(model.predict(X[train_indices]) == y[train_indices]))
    test_accuracy = float(np.mean(model.predict(X[test_indices]) == y[test_indices]))
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Handwritten-digit holdout accuracy: {test_accuracy * 100:.2f}%")
    if test_accuracy < 0.95:
        raise RuntimeError(f"Holdout accuracy is unexpectedly low: {test_accuracy:.4f}")

    joblib.dump(model, BASE / "model.joblib")
    with open(BASE / "metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump({
            "accuracy": round(test_accuracy, 4),
            "model": "Handwritten digits RBF SVM",
        }, metrics_file)
    print("Saved model.joblib and metrics.json.")


if __name__ == "__main__":
    main()
