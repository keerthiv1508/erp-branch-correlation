#!/usr/bin/env python3
import argparse
import json
import os
import random

import numpy as np

DATA_PATH = os.path.expanduser("~/training_data.json")
SEED = 42


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def train_logistic_regression(X, y, lr=0.1, epochs=1000, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = rng.normal(0, 0.01, size=d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        preds = sigmoid(z)
        error = preds - y
        grad_w = X.T @ error / n
        grad_b = error.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def make_function_folds(functions, k, seed):
    rng = random.Random(seed)
    shuffled = list(functions)
    rng.shuffle(shuffled)
    folds = [[] for _ in range(k)]
    for i, fn in enumerate(shuffled):
        folds[i % k].append(fn)
    return folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    with open(DATA_PATH) as f:
        examples = json.load(f)
    functions = sorted(set(e["function"] for e in examples))
    folds = make_function_folds(functions, args.k, SEED)

    total_tp = total_tn = total_fp = total_fn = 0

    for i, test_functions in enumerate(folds):
        test_functions_set = set(test_functions)
        train_examples = [e for e in examples if e["function"] not in test_functions_set]
        test_examples = [e for e in examples if e["function"] in test_functions_set]
        if not test_examples or not train_examples:
            continue

        X_train = np.array([e["x"] for e in train_examples])
        y_train = np.array([e["y_label"] for e in train_examples], dtype=float)
        X_test = np.array([e["x"] for e in test_examples])
        y_test = np.array([e["y_label"] for e in test_examples], dtype=float)

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std

        w, b = train_logistic_regression(X_train_norm, y_train, lr=0.1, epochs=1000)
        probs = sigmoid(X_test_norm @ w + b)
        preds = (probs >= args.threshold).astype(int)

        tp = int(((preds == 1) & (y_test == 1)).sum())
        tn = int(((preds == 0) & (y_test == 0)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        fn = int(((preds == 0) & (y_test == 1)).sum())

        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        print(f"Fold {i+1}: TP={tp} TN={tn} FP={fp} FN={fn}")

    print("\n" + "=" * 50)
    print(f"AGGREGATE CONFUSION MATRIX (threshold={args.threshold})")
    print("=" * 50)
    print(f"                 Predicted correlated   Predicted uncorrelated")
    print(f"Actual correlated:      TP={total_tp:5d}              FN={total_fn:5d}")
    print(f"Actual uncorrelated:    FP={total_fp:5d}              TN={total_tn:5d}")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)

    print(f"\nPrecision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"Accuracy:  {accuracy:.3f}")


if __name__ == "__main__":
    main()
