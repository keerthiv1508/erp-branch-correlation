#!/usr/bin/env python3
"""
Same training and fold logic as lr_pooled_confusion.py (already confirmed
today, pooled accuracy 0.5800 matched Table 1's 0.581). Only addition:
per-fold accuracy tracked and printed, to directly verify the LR array
hardcoded in significance_test.py: [0.515, 0.586, 0.624, 0.544, 0.636]
"""
import json, os, random
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
    with open(DATA_PATH) as f:
        examples = json.load(f)

    programs = sorted(set(e["sample_index"] for e in examples))
    folds = make_program_folds(programs, 5, SEED)

    per_fold_acc = []
    all_true, all_prob = [], []

    for i, test_programs in enumerate(folds):
        test_set = set(test_programs)
        train_examples = [e for e in examples if e["sample_index"] not in test_set]
        test_examples = [e for e in examples if e["sample_index"] in test_set]
        if not train_examples or not test_examples:
            continue

        X_train = np.array([e["x"] for e in train_examples])
        y_train = np.array([e["y_label"] for e in train_examples], dtype=float)
        X_test = np.array([e["x"] for e in test_examples])
        y_test = np.array([e["y_label"] for e in test_examples], dtype=float)

        mean, std = X_train.mean(axis=0), X_train.std(axis=0)
        std[std == 0] = 1.0
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std

        w, b = train_logistic_regression(X_train_norm, y_train, lr=0.1, epochs=1000)
        probs = sigmoid(X_test_norm @ w + b)
        preds = (probs >= 0.5).astype(int)

        fold_acc = (preds == y_test).mean()
        per_fold_acc.append(float(fold_acc))

        all_true.extend(y_test.tolist())
        all_prob.extend(probs.tolist())

        print(f"Fold {i+1}: n_test={len(test_examples)}, accuracy={fold_acc:.4f}")

    y_true = np.array(all_true)
    y_pred = (np.array(all_prob) >= 0.5).astype(int)
    pooled_acc = (y_pred == y_true).mean()

    print("\n" + "=" * 60)
    print(f"Per-fold accuracy: {[round(a, 4) for a in per_fold_acc]}")
    print(f"Mean of per-fold: {np.mean(per_fold_acc):.4f}")
    print(f"Pooled accuracy:  {pooled_acc:.4f}")
    print()
    print(f"Hardcoded in significance_test.py: [0.515, 0.586, 0.624, 0.544, 0.636]")
    print(f"Just computed:                     {[round(a, 4) for a in per_fold_acc]}")

    with open("lr_per_fold_results.json", "w") as f:
        json.dump({"per_fold_accuracy": per_fold_acc, "pooled_accuracy": float(pooled_acc)}, f, indent=2)


if __name__ == "__main__":
    main()
