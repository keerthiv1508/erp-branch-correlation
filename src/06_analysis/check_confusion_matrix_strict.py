#!/usr/bin/env python3
import argparse
import json
import os
import random

import numpy as np

DATA_PATH = os.path.expanduser("~/training_data.json")
PAIRS_PATH = os.path.expanduser("~/pairs_multirun.json")
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
        w -= lr * (X.T @ error / n)
        b -= lr * error.mean()
    return w, b


def make_program_folds(programs, k, seed):
    rng = random.Random(seed)
    shuffled = list(programs)
    rng.shuffle(shuffled)
    folds = [[] for _ in range(k)]
    for i, p in enumerate(shuffled):
        folds[i % k].append(p)
    return folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--strict_threshold", type=float, default=0.9)
    args = parser.parse_args()

    with open(DATA_PATH) as f:
        examples = json.load(f)
    with open(PAIRS_PATH) as f:
        pairs = json.load(f)

    # Model still TRAINS on the original 0.5-threshold labels (unchanged methodology) --
    # only EVALUATION ground truth changes, to test the stricter real-world question.
    pair_lookup = {}
    for p in pairs:
        key = (p["sample_index"], p["function"], round(p["pearson_correlation"], 6))
        pair_lookup[key] = p

    programs = sorted(set(e["sample_index"] for e in examples))
    folds = make_program_folds(programs, args.k, SEED)

    total_tp = total_tn = total_fp = total_fn = 0
    total_strict_positive = 0

    for i, test_programs in enumerate(folds):
        test_set = set(test_programs)
        train_examples = [e for e in examples if e["sample_index"] not in test_set]
        test_examples = [e for e in examples if e["sample_index"] in test_set]
        if not train_examples or not test_examples:
            continue

        X_train = np.array([e["x"] for e in train_examples])
        y_train = np.array([e["y_label"] for e in train_examples], dtype=float)  # 0.5-threshold labels
        X_test = np.array([e["x"] for e in test_examples])

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std

        w, b = train_logistic_regression(X_train_norm, y_train, lr=0.1, epochs=1000)
        probs = sigmoid(X_test_norm @ w + b)
        preds = (probs >= 0.5).astype(int)  # model's own decision threshold, unchanged

        # STRICT ground truth: is the true |correlation| >= 0.9? (not the 0.5 label used for training)
        strict_labels = []
        for e in test_examples:
            key = (e["sample_index"], e["function"], round(e["y_correlation"], 6))
            p = pair_lookup.get(key)
            true_r = p["pearson_correlation"] if p else e["y_correlation"]
            strict_labels.append(1 if abs(true_r) >= args.strict_threshold else 0)
        strict_labels = np.array(strict_labels)

        tp = int(((preds == 1) & (strict_labels == 1)).sum())
        tn = int(((preds == 0) & (strict_labels == 0)).sum())
        fp = int(((preds == 1) & (strict_labels == 0)).sum())
        fn = int(((preds == 0) & (strict_labels == 1)).sum())

        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_strict_positive += int(strict_labels.sum())

        print(f"Fold {i+1}: n_test={len(test_examples)}, strict-positive (|r|>={args.strict_threshold})={strict_labels.sum()}, "
              f"TP={tp} TN={tn} FP={fp} FN={fn}")

    print("\n" + "=" * 60)
    print(f"CONFUSION MATRIX -- model's 0.5-decision vs STRICT ground truth |r| >= {args.strict_threshold}")
    print("=" * 60)
    print(f"Total examples where true |r| >= {args.strict_threshold}: {total_strict_positive} "
          f"({100*total_strict_positive/len(examples):.1f}% of dataset)")
    print(f"                 Predicted correlated   Predicted uncorrelated")
    print(f"Actual strict+:        TP={total_tp:5d}              FN={total_fn:5d}")
    print(f"Actual strict-:        FP={total_fp:5d}              TN={total_tn:5d}")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"\nPrecision (of predicted-correlated, how many are truly |r|>={args.strict_threshold}): {precision:.3f}")
    print(f"Recall (of truly |r|>={args.strict_threshold} pairs, how many did we catch): {recall:.3f}")
    print(f"F1: {f1:.3f}")


if __name__ == "__main__":
    main()
