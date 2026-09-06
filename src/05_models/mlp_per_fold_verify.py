#!/usr/bin/env python3
"""
Same training and fold logic as mlp_pooled_confusion.py (already
confirmed today, pooled accuracy 0.5910 matched Table 1's 0.592). Only
addition: per-fold accuracy tracked and printed, to directly verify the
MLP array hardcoded in significance_test.py:
[0.543, 0.577, 0.649, 0.561, 0.627]
"""
import json, os, random
import numpy as np

DATA_PATH = os.path.expanduser("~/training_data.json")
SEED = 42


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def relu(z):
    return np.maximum(0, z)


def relu_deriv(z):
    return (z > 0).astype(float)


class SmallMLP:
    def __init__(self, input_dim, hidden_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim,))
        self.b2 = 0.0

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        return sigmoid(z2), (X, z1, a1, z2)

    def train(self, X, y, lr=0.05, epochs=2000, l2=0.001):
        n = X.shape[0]
        for _ in range(epochs):
            preds, (X_c, z1, a1, z2) = self.forward(X)
            d_z2 = (preds - y) / n
            d_W2 = a1.T @ d_z2 + l2 * self.W2
            d_b2 = d_z2.sum()
            d_a1 = np.outer(d_z2, self.W2)
            d_z1 = d_a1 * relu_deriv(z1)
            d_W1 = X_c.T @ d_z1 + l2 * self.W1
            d_b1 = d_z1.sum(axis=0)
            self.W2 -= lr * d_W2; self.b2 -= lr * d_b2
            self.W1 -= lr * d_W1; self.b1 -= lr * d_b1

    def predict_proba(self, X):
        preds, _ = self.forward(X)
        return preds


def make_function_folds(functions, k, seed):
    rng = random.Random(seed)
    shuffled = list(functions)
    rng.shuffle(shuffled)
    folds = [[] for _ in range(k)]
    for i, fn in enumerate(shuffled):
        folds[i % k].append(fn)
    return folds


def main():
    with open(DATA_PATH) as f:
        examples = json.load(f)

    programs = sorted(set(e["sample_index"] for e in examples))
    folds = make_function_folds(programs, 5, SEED)

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

        model = SmallMLP(input_dim=X_train.shape[1], hidden_dim=32, seed=i)
        model.train(X_train_norm, y_train, lr=0.05, epochs=2000)

        probs = model.predict_proba(X_test_norm)
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
    print(f"Hardcoded in significance_test.py: [0.543, 0.577, 0.649, 0.561, 0.627]")
    print(f"Just computed:                     {[round(a, 4) for a in per_fold_acc]}")

    with open("mlp_per_fold_results.json", "w") as f:
        json.dump({"per_fold_accuracy": per_fold_acc, "pooled_accuracy": float(pooled_acc)}, f, indent=2)


if __name__ == "__main__":
    main()
