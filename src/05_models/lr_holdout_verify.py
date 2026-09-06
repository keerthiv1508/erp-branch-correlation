#!/usr/bin/env python3
"""
LR holdout evaluation, split by sample_index (program-level), unlike
train_final_model.py and check_holdout_variance*.py which split by
function name, the leakage risk Section 3.6 already identified and
fixed for the main results. LR training logic copied verbatim from
check_full_metric_suite.py / lr_per_fold_verify.py, both confirmed
correct multiple times today.

IMPORTANT: N_HOLDOUT_PROGRAMS and SEED must match gnn_holdout_and_strict.py
for this to describe the same held-out programs as GNN's holdout result.
"""
import json, os, random
import numpy as np

DATA_PATH = os.path.expanduser("~/training_data.json")
SEED = 42
N_HOLDOUT_PROGRAMS = 20


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


def main():
    with open(DATA_PATH) as f:
        examples = json.load(f)

    all_programs = sorted(set(e["sample_index"] for e in examples))
    rng = random.Random(SEED)
    shuffled = list(all_programs)
    rng.shuffle(shuffled)
    holdout_programs = set(shuffled[:N_HOLDOUT_PROGRAMS])
    train_programs = set(shuffled[N_HOLDOUT_PROGRAMS:])

    train_examples = [e for e in examples if e["sample_index"] not in holdout_programs]
    holdout_examples = [e for e in examples if e["sample_index"] in holdout_programs]

    print(f"Total: {len(examples)} examples, {len(all_programs)} distinct programs")
    print(f"Held out {len(holdout_programs)} programs ({len(holdout_examples)} examples)")
    print(f"Training on {len(train_programs)} programs ({len(train_examples)} examples)")

    X_train = np.array([e["x"] for e in train_examples])
    y_train = np.array([e["y_label"] for e in train_examples], dtype=float)
    X_hold = np.array([e["x"] for e in holdout_examples])
    y_hold = np.array([e["y_label"] for e in holdout_examples], dtype=float)

    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train_norm = (X_train - mean) / std
    X_hold_norm = (X_hold - mean) / std

    w, b = train_logistic_regression(X_train_norm, y_train, lr=0.1, epochs=1000)

    hold_probs = sigmoid(X_hold_norm @ w + b)
    hold_preds = (hold_probs >= 0.5).astype(int)
    holdout_acc = (hold_preds == y_hold).mean()

    tp = int(((hold_preds == 1) & (y_hold == 1)).sum())
    tn = int(((hold_preds == 0) & (y_hold == 0)).sum())
    fp = int(((hold_preds == 1) & (y_hold == 0)).sum())
    fn = int(((hold_preds == 0) & (y_hold == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0
    majority_label = 1 if y_train.mean() >= 0.5 else 0
    baseline_acc = (np.full_like(y_hold, majority_label) == y_hold).mean()

    print("\n" + "=" * 60)
    print(f"LR GENUINE HOLDOUT RESULT ({len(holdout_programs)} unseen programs)")
    print("=" * 60)
    print(f"Holdout accuracy: {holdout_acc:.4f}")
    print(f"Holdout baseline (majority class): {baseline_acc:.4f}")
    print(f"Holdout recall: {recall:.4f}")
    print(f"Confusion matrix: TP={tp} FP={fp} TN={tn} FN={fn}")

    with open("lr_holdout_results.json", "w") as f:
        json.dump({
            "holdout_accuracy": float(holdout_acc),
            "baseline_accuracy": float(baseline_acc),
            "recall": float(recall),
            "n_holdout_programs": len(holdout_programs),
            "n_holdout_examples": len(holdout_examples),
            "holdout_program_ids": sorted(holdout_programs),
        }, f, indent=2)
    print(f"\nSaved lr_holdout_results.json")
    print(f"holdout_program_ids saved -- compare against gnn_holdout_and_strict.py's")
    print(f"holdout set to confirm both are evaluating the same 20 programs")


if __name__ == "__main__":
    main()
